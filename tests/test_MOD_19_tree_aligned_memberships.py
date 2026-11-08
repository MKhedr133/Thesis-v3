"""Synthetic MOD-19 Stage 2B checks; no thesis participant data are included."""

import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))

import MOD_14_direct_linear_regression_evaluation as MOD14
import MOD_19_tree_mamdani as MOD19
import MOD_19_fuzzy_rules as BASE
import MOD_19_tree_aligned_memberships as ALIGNED


class TreeAlignedMembershipTests(unittest.TestCase):
    def test_mental_demand_two_splits_give_three_terms(self):
        partition = ALIGNED.make_partition("mental", 0, 8, (1.5, 3.5))
        self.assertEqual(partition.labels, ("Low", "Medium", "High"))
        np.testing.assert_allclose(partition.degrees(1.5), (0.5, 0.5, 0))
        np.testing.assert_allclose(partition.degrees(3.5), (0, 0.5, 0.5))
        self.assertAlmostEqual(sum(partition.degrees(4)), 1.0)

    def test_head_turning_three_splits_give_four_terms(self):
        boundaries = (88.177, 94.38, 109.167)
        partition = ALIGNED.make_partition("head", 49.03, 180.85, boundaries)
        self.assertEqual(partition.labels, ("Very low", "Low", "Medium", "High"))
        for i, boundary in enumerate(boundaries):
            observed = partition.degrees(boundary)
            self.assertAlmostEqual(observed[i], 0.5)
            self.assertAlmostEqual(observed[i + 1], 0.5)
            self.assertAlmostEqual(sum(observed), 1.0)

    def test_every_input_value_has_coverage(self):
        part = ALIGNED.make_partition("feature", 0, 8, (1.5, 3.5))
        for value in np.linspace(-2, 10, 500):
            m = part.degrees(value)
            self.assertTrue(np.isfinite(m).all())
            self.assertAlmostEqual(float(m.sum()), 1, places=9)
            self.assertGreaterEqual(float(m.max()), 0.5 - 1e-10)

    def test_one_sided_tree_conditions_keep_all_regions(self):
        part = ALIGNED.make_partition("head", 49.03, 180.85,
                                      (88.177, 94.38, 109.167))
        self.assertEqual(
            ALIGNED.route_terms(88.177, None, part),
            ("Low", "Medium", "High"),
        )
        self.assertEqual(
            ALIGNED.route_terms(None, 94.38, part),
            ("Very low", "Low"),
        )
        self.assertEqual(
            ALIGNED.route_terms(94.38, 109.167, part),
            ("Medium",),
        )

    def test_correction_keeps_high_mental_rule_active_at_four(self):
        mental = ALIGNED.make_partition("mental", 0, 8, (1.5, 3.5))
        antecedent = (("mental", ALIGNED.route_terms(3.5, None, mental)),)
        observed = ALIGNED.antecedent_degree(
            antecedent, {"mental": mental}, pd.Series({"mental": 4.0})
        )
        self.assertGreater(observed, 0)  # old uniform 'High' was zero
        self.assertGreaterEqual(observed, 0.5)

    def test_reject_invalid_split_and_unsupported_condition(self):
        with self.assertRaises(BASE.ConversionError):
            ALIGNED.make_partition("x", 0, 8, (9,))
        with self.assertRaises(BASE.ConversionError):
            ALIGNED.make_partition("x", 0, 8, (2, 2, 8))
        part = ALIGNED.make_partition("mental", 0, 8, (1.5, 3.5))
        with self.assertRaises(BASE.ConversionError):
            ALIGNED.route_terms(2.3, None, part)

    @staticmethod
    def fixture():
        names = MOD14.AUDITORY_PREDICTORS
        rows = []
        for p in range(26):
            for stage, label in enumerate((0, 2, 6, 10)):
                i = p * 4 + stage
                rows.append({
                    "participant_id": f"synthetic-{p:02d}",
                    "difficulty_stage": stage,
                    "difficulty_level": label,
                    names[0]: (p + 2 * stage) % 9,
                    names[1]: 3.3 + ((p * 3 + stage) % 14) * 0.4,
                    names[2]: 0.7 + ((p + 3 * stage) % 11) * 0.16,
                    names[3]: 49 + ((p * 7 + stage * 4) % 27) * 5,
                })
        frame = pd.DataFrame(rows)
        frame["frozen_mlr_prediction"] = (
            0.4 + frame[names[0]] * 0.13
            + frame[names[1]] * 0.07
            + frame[names[2]] * 0.2
            + frame[names[3]] * 0.004
        )
        tree = MOD19.fit_development_tree(
            frame, max_depth=3, min_samples_leaf=4
        )
        preds = frame.loc[:, [
            "participant_id", "difficulty_level",
            "difficulty_stage", "frozen_mlr_prediction",
        ]].copy()
        preds["provisional_tree_prediction"] = tree.predict(
            frame.loc[:, names].to_numpy(dtype=float)
        )
        return frame, MOD19.extract_raw_rules(tree), preds

    def test_frozen_teacher_and_one_original_route_coverage(self):
        frame, raw, preds = self.fixture()
        result = ALIGNED.aligned_conversion(frame, raw, preds)
        self.assertEqual(result["development_participants"], 26)
        self.assertEqual(result["development_rows"], 104)
        self.assertEqual(len(result["rules"]), len(raw))
        self.assertEqual(len(result["diagnostics"]), 104)
        self.assertTrue(all(
            row["source_rule_activation"] >= 0.5 - 1e-10
            for row in result["diagnostics"]
        ))
        self.assertEqual(len(result["partitions"]), 4)
        self.assertTrue(all(
            len(terms) >= 1
            for rule in result["rules"]
            for feature, terms in rule["antecedent"]
        ))

    def test_modified_teacher_prediction_rejected(self):
        frame, raw, preds = self.fixture()
        preds.loc[0, "frozen_mlr_prediction"] += 0.1
        with self.assertRaisesRegex(BASE.ConversionError, "MLR targets differ"):
            ALIGNED.aligned_conversion(frame, raw, preds)

    def test_outputs_and_overwrite_guard(self):
        frame, raw, preds = self.fixture()
        result = ALIGNED.aligned_conversion(frame, raw, preds)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)
            source = path / "stage1"
            source.mkdir()
            for filename in (
                "manifest.json", "numerical_rules.csv", "development_predictions.csv"
            ):
                (source / filename).write_text("synthetic only\n")
            target = path / "aligned"
            ALIGNED.write_outputs(result, target, source_dir=source)
            self.assertTrue(
                (target / "tree_aligned_membership_functions.png").is_file()
            )
            self.assertTrue(
                (target / "tree_aligned_fuzzy_rules.csv").is_file()
            )
            manifest = json.loads(
                (target / "tree_aligned_manifest.json").read_text()
            )
            self.assertEqual(manifest["development_participants"], 26)
            self.assertEqual(manifest["test_participants_used"], 0)
            self.assertEqual(manifest["source_rule_count"], len(raw))
            self.assertIn("NOT Mamdani", manifest["research_status"])
            with self.assertRaises(BASE.ConversionError):
                ALIGNED.write_outputs(result, target, source_dir=source)


if __name__ == "__main__":
    unittest.main()
