"""Synthetic MOD-19 Stage 4A tests; no original participant observations."""

import json
import sys
import tempfile
import unittest
from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))

import MOD_19_fuzzy_rules as BASE
import MOD_19_tree_aligned_memberships as ALIGNED
import MOD_19_mamdani_inference as MAMDANI
import MOD_19_output_term_sensitivity as STAGE4


class OutputVocabularySensitivityTests(unittest.TestCase):
    @staticmethod
    def fixture():
        part = ALIGNED.make_partition("mental", 0.0, 8.0, (1.5, 3.5))
        rules = [
            {"source_leaf_ids": [1], "sample_count": 4,
             "tree_output": 0.8, "consequent": "Medium",
             "antecedent": (("mental", ("Low",)),)},
            {"source_leaf_ids": [2], "sample_count": 4,
             "tree_output": 1.4, "consequent": "Medium",
             "antecedent": (("mental", ("Medium",)),)},
            {"source_leaf_ids": [3], "sample_count": 4,
             "tree_output": 2.7, "consequent": "High",
             "antecedent": (("mental", ("High",)),)},
        ]
        frame = pd.DataFrame([
            {"participant_id": "synthetic-A", "difficulty_level": 0,
             "difficulty_stage": 0, "mental": 0.0,
             "frozen_mlr_prediction": 0.5},
            {"participant_id": "synthetic-A", "difficulty_level": 2,
             "difficulty_stage": 1, "mental": 2.5,
             "frozen_mlr_prediction": 1.3},
            {"participant_id": "synthetic-B", "difficulty_level": 6,
             "difficulty_stage": 2, "mental": 4.0,
             "frozen_mlr_prediction": 1.8},
            {"participant_id": "synthetic-B", "difficulty_level": 10,
             "difficulty_stage": 3, "mental": 8.0,
             "frozen_mlr_prediction": 2.7},
        ])
        converted = {
            "rules": rules, "partitions": {"mental": part},
            "output_centres": np.array([0.0, 1.5, 3.0]),
        }
        crisp = np.array([0.8, 1.4, 2.7, 2.7])
        return frame, converted, crisp

    def test_three_and_five_labels_derived_from_same_tree_leaves(self):
        frame, converted, _ = self.fixture()
        rules = converted["rules"]
        three, c3, l3 = STAGE4.output_rule_labels(rules, 0, 3, 3)
        five, c5, l5 = STAGE4.output_rule_labels(rules, 0, 3, 5)
        self.assertEqual(three, ("Low", "Medium", "High"))
        self.assertEqual(
            five, ("Very low", "Low", "Medium", "High", "Very high")
        )
        self.assertEqual(l3, ["Medium", "Medium", "High"])
        self.assertEqual(l5, ["Low", "Medium", "Very high"])
        self.assertEqual(c3[0], c5[0])
        self.assertEqual(c3[-1], c5[-1])

    def test_reject_invalid_terms_and_bad_output_range(self):
        _, converted, _ = self.fixture()
        with self.assertRaises(BASE.ConversionError):
            STAGE4.output_rule_labels(converted["rules"], 0, 3, 4)
        with self.assertRaises(BASE.ConversionError):
            STAGE4.output_rule_labels(converted["rules"], 3, 3, 3)

    def test_compression_reports_duplicate_consequents_without_deleting_rules(self):
        result = STAGE4.compression(
            ["Medium", "Medium", "High"], ("Low", "Medium", "High")
        )
        self.assertEqual(result["distinct_consequents_used"], 2)
        self.assertEqual(result["pairs_sharing_consequent"], 1)
        self.assertEqual(result["largest_consequent_group"], 2)
        self.assertEqual(result["rules_per_consequent"]["Low"], 0)

    def test_three_term_reproduces_stage3_mamdani_exactly(self):
        frame, converted, crisp = self.fixture()
        generated, metrics, mapping, _ = STAGE4.compute_comparison(
            frame, converted, crisp
        )
        x, membership = MAMDANI.output_grid(converted["output_centres"])
        legacy_predicted = []
        for _, row in frame.iterrows():
            info, _, _ = MAMDANI.infer_observation(
                row, converted["rules"], converted["partitions"],
                x, membership,
            )
            legacy_predicted.append(info["mamdani_prediction"])
        np.testing.assert_allclose(
            generated["output_3_terms_prediction"].to_numpy(),
            legacy_predicted, atol=1e-10, rtol=0
        )
        self.assertEqual(len(metrics), 2)
        self.assertEqual(len(mapping), 6)
        self.assertEqual(int(metrics.iloc[0]["n_unavailable"]), 0)

    def test_five_terms_change_only_the_output_vocab_not_input_rules(self):
        frame, converted, crisp = self.fixture()
        original_rules = deepcopy(converted["rules"])
        _, _, maps, _ = STAGE4.compute_comparison(frame, converted, crisp)
        self.assertEqual(converted["rules"], original_rules)
        by_count = {
            int(name.split("_")[1]): group.sort_values(
                "source_leaf_ids"
            ).reset_index(drop=True)
            for name, group in maps.groupby("configuration", sort=True)
        }
        self.assertEqual(set(by_count), {3, 5})
        preserved_columns = [
            "source_leaf_ids", "source_tree_prediction",
            "development_leaf_rows", "antecedent_or_groups_json",
        ]
        pd.testing.assert_frame_equal(
            by_count[3][preserved_columns],
            by_count[5][preserved_columns],
        )
        self.assertNotEqual(
            by_count[3]["output_consequent"].tolist(),
            by_count[5]["output_consequent"].tolist(),
        )
        self.assertTrue((maps.groupby("configuration").size() == 3).all())

    def test_original_stage3_predictions_must_match(self):
        frame, converted, crisp = self.fixture()
        generated, _, _, _ = STAGE4.compute_comparison(
            frame, converted, crisp
        )
        saved = pd.DataFrame({
            "participant_id": generated["participant_id"],
            "difficulty_level": generated["difficulty_level"],
            "difficulty_stage": generated["difficulty_stage"],
            "frozen_mlr_prediction": generated["frozen_mlr_prediction"],
            "provisional_tree_prediction": generated["crisp_tree_prediction"],
            "mamdani_prediction": generated["output_3_terms_prediction"],
            "output_area": generated["output_3_terms_output_area"],
            "status": generated["output_3_terms_status"],
        })
        STAGE4.verify_stage3_baseline(generated, saved)
        saved.loc[0, "mamdani_prediction"] += 0.02
        with self.assertRaisesRegex(
            BASE.ConversionError, "does not reproduce Stage 3"
        ):
            STAGE4.verify_stage3_baseline(generated, saved)

    def test_original_stage3_row_reordering_is_rejected(self):
        frame, converted, crisp = self.fixture()
        generated, _, _, _ = STAGE4.compute_comparison(frame, converted, crisp)
        saved = pd.DataFrame({
            "participant_id": generated["participant_id"],
            "difficulty_level": generated["difficulty_level"],
            "difficulty_stage": generated["difficulty_stage"],
            "frozen_mlr_prediction": generated["frozen_mlr_prediction"],
            "provisional_tree_prediction": generated["crisp_tree_prediction"],
            "mamdani_prediction": generated["output_3_terms_prediction"],
            "output_area": generated["output_3_terms_output_area"],
            "status": generated["output_3_terms_status"],
        })
        saved.iloc[[0, 2]] = saved.iloc[[2, 0]].to_numpy()
        with self.assertRaisesRegex(BASE.ConversionError, "row order"):
            STAGE4.verify_stage3_baseline(generated, saved)

    def test_missing_antecedent_coverage_rejected(self):
        frame, converted, crisp = self.fixture()
        for rule in converted["rules"]:
            rule["antecedent"] = (("mental", ("Low",)),)
        with self.assertRaisesRegex(BASE.ConversionError, "coverage"):
            STAGE4.compute_comparison(frame, converted, crisp)

    def test_baseline_manifest_hashes_checked(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            file = directory / "fake.csv"
            file.write_text("original\n")
            import MOD_19_tree_mamdani as MOD19
            signature = MAMDANI.file_sha256(file)
            manifest = {
                "module": "MOD-19",
                "stage": "3: fixed Mamdani development baseline",
                "model_id": MOD19.MODEL_ID,
                "development_participants": 26,
                "development_rows": 104,
                "test_participants_used": 0,
                "source_sha256": {"fixture": signature},
            }
            STAGE4.verify_stage3_provenance(
                manifest, {"fixture": file}, development_rows=104
            )
            file.write_text("changed\n")
            with self.assertRaisesRegex(BASE.ConversionError, "provenance changed"):
                STAGE4.verify_stage3_provenance(
                    manifest, {"fixture": file}, development_rows=104
                )

    def test_empty_activation_remains_unavailable(self):
        x = np.linspace(0, 2, 4001)
        centres = np.linspace(0, 2, 3)
        membership = STAGE4.calculate_memberships(x, centres)
        pred, status, area = STAGE4.infer_with_labels(
            np.array([0.]), ["Medium"],
            ("Low", "Medium", "High"), x, membership
        )
        self.assertEqual(status, "Unavailable")
        self.assertTrue(np.isnan(pred))
        self.assertEqual(area, 0)

    def test_output_guard_before_creation(self):
        frame, converted, crisp = self.fixture()
        result, metrics, mapping, vocab = STAGE4.compute_comparison(
            frame, converted, crisp
        )
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            target = root / "occupied"
            target.mkdir()
            (target / "keep.txt").write_text("existing")
            with self.assertRaisesRegex(BASE.ConversionError, "already exists"):
                STAGE4.save_comparison(
                    target, result, metrics, mapping, vocab,
                    provenance={}, modeling_data=root / "missing.csv",
                    holdout_split=root / "missing_split.csv", stage3=root,
                )
            self.assertTrue((target / "keep.txt").exists())


if __name__ == "__main__":
    unittest.main()
