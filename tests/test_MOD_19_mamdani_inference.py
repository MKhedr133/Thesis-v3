"""Synthetic tests for MOD-19 Stage 3; no thesis participants in fixtures."""

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
import MOD_19_mamdani_inference as MAMDANI


class FixedMamdaniTests(unittest.TestCase):
    def test_triangular_output_grid_partition(self):
        x, membership = MAMDANI.output_grid(np.array([0.0, 1.0, 2.0]))
        self.assertEqual(len(x), 4001)
        np.testing.assert_allclose(membership.sum(axis=1), 1.0)
        np.testing.assert_allclose(membership[2000], [0, 1, 0])
        self.assertAlmostEqual(x[0], 0)
        self.assertAlmostEqual(x[-1], 2)

    def test_invalid_output_grid_rejected(self):
        with self.assertRaises(BASE.ConversionError):
            MAMDANI.output_grid(np.array([0.0, 1.0, 1.0]))
        with self.assertRaises(BASE.ConversionError):
            MAMDANI.output_grid(np.array([0.0, 1.0, 2.0]), grid_size=10)

    def test_min_implication_max_aggregation(self):
        _, membership = MAMDANI.output_grid(np.array([0.0, 1.0, 2.0]))
        rules = [
            {"consequent": "Low"},
            {"consequent": "High"},
            {"consequent": "Low"},
        ]
        strengths = np.array([0.2, 0.6, 0.7])
        aggregate = MAMDANI.aggregate_consequents(
            strengths, rules, membership
        )
        expected = np.maximum(
            np.minimum(0.7, membership[:, 0]),
            np.minimum(0.6, membership[:, 2]),
        )
        np.testing.assert_allclose(aggregate, expected)

    def test_centroid_symmetry_and_zero_area(self):
        x = np.linspace(0, 2, 4001)
        symmetric = np.maximum(0, 1 - abs(x - 1))
        centre, area = MAMDANI.centroid(x, symmetric)
        self.assertAlmostEqual(centre, 1.0, places=9)
        self.assertGreater(area, 0)
        unavailable, empty_area = MAMDANI.centroid(x, np.zeros_like(x))
        self.assertTrue(np.isnan(unavailable))
        self.assertEqual(empty_area, 0)

    def test_centroid_invalid_membership_rejected(self):
        x = np.array([0, 1, 2], dtype=float)
        with self.assertRaises(BASE.ConversionError):
            MAMDANI.centroid(x, np.array([0, -0.5, 0]))

    def test_or_inside_feature_and_min_across_features(self):
        first = ALIGNED.make_partition("demand", 0, 8, (1.5, 3.5))
        second = ALIGNED.make_partition("time", 0, 6, (3,))
        row = pd.Series({"demand": 4.0, "time": 1.0})
        rule = {
            "antecedent": (
                ("demand", ("Medium", "High")),
                ("time", ("Low",)),
            ),
            "consequent": "Medium",
        }
        strength, trace = MAMDANI.rule_strength(
            row, rule, {"demand": first, "time": second}
        )
        expected = min(max(first.degrees(4.0)[1:]), second.degrees(1.0)[0])
        self.assertAlmostEqual(strength, expected)
        self.assertEqual(len(trace[0]["alternatives"]), 2)
        self.assertAlmostEqual(trace[0]["or_activation"],
                               max(first.degrees(4.0)[1:]))

    def test_unavailable_without_rule_activation(self):
        mental = ALIGNED.make_partition("mental", 0, 8, (1.5, 3.5))
        x, m = MAMDANI.output_grid(np.array([0., 1., 2.]))
        rule = {
            "source_leaf_ids": [1],
            "antecedent": (("mental", ("High",)),),
            "consequent": "Medium",
        }
        result, activations, aggregate = MAMDANI.infer_observation(
            pd.Series({"mental": 0}), [rule], {"mental": mental}, x, m
        )
        self.assertEqual(result["status"], "Unavailable")
        self.assertTrue(np.isnan(result["mamdani_prediction"]))
        self.assertEqual(result["output_area"], 0)
        self.assertEqual(activations[0]["activation"], 0)
        np.testing.assert_allclose(aggregate, 0)

    def test_participant_balance_not_row_pooled(self):
        data = pd.DataFrame([
            {"participant_id": "A", "ref": 1., "pred": 0.},
            {"participant_id": "B", "ref": 1., "pred": 1.},
            {"participant_id": "B", "ref": 1., "pred": 1.},
            {"participant_id": "B", "ref": 1., "pred": 1.},
        ])
        self.assertAlmostEqual(MAMDANI.participant_balanced_mae(
            data, reference="ref", prediction="pred"), 0.5)
        self.assertNotAlmostEqual(
            (data["ref"] - data["pred"]).abs().mean(), 0.5
        )

    def test_r2_reference_and_nonconstant_baseline(self):
        data = pd.DataFrame({
            "ref": [0., 1., 2.],
            "identical": [0., 1., 2.],
            "zero": [0., 0., 0.],
        })
        self.assertAlmostEqual(MAMDANI.pooled_r2(
            data, reference="ref", prediction="identical"), 1.)
        self.assertAlmostEqual(MAMDANI.pooled_r2(
            data, reference="ref", prediction="zero"), -1.5)

    @staticmethod
    def fixture():
        names = MOD14.AUDITORY_PREDICTORS
        rows = []
        for p in range(26):
            for stage, label in enumerate((0, 2, 6, 10)):
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
        original = frame[[
            "participant_id", "difficulty_level", "difficulty_stage",
            "frozen_mlr_prediction",
        ]].copy()
        original["provisional_tree_prediction"] = tree.predict(
            frame.loc[:, names].to_numpy(dtype=float)
        )
        raw = MOD19.extract_raw_rules(tree)
        return frame, raw, original

    @classmethod
    def sources(cls, directory):
        frame, raw, original = cls.fixture()
        stage1 = directory / "stage1"
        stage1.mkdir()
        (stage1 / "manifest.json").write_text(json.dumps({
            "stage": "1: crisp tree extraction only", "model_id": "auditory",
            "development_participants": 26, "development_rows": 104,
            "test_participants_used": 0, "raw_rule_count": len(raw),
        }))
        pd.DataFrame([{
            "leaf_id": rule.leaf_id, "development_row_count": rule.sample_count,
            "tree_output": rule.prediction,
            "conditions_json": json.dumps([
                {"feature": condition.feature,
                 "operator": condition.operator,
                 "threshold": condition.threshold}
                for condition in rule.conditions
            ]),
        } for rule in raw]).to_csv(stage1 / "numerical_rules.csv", index=False)
        original.to_csv(stage1 / "development_predictions.csv", index=False)
        converted = ALIGNED.aligned_conversion(frame, raw, original)
        stage2b = directory / "stage2b"
        ALIGNED.write_outputs(converted, stage2b, source_dir=stage1)
        manifest = json.loads(
            (stage2b / "tree_aligned_manifest.json").read_text()
        )
        return frame, raw, original, stage1, stage2b, manifest

    def test_stage2b_sources_and_full_inference(self):
        with tempfile.TemporaryDirectory() as temp:
            frame, raw, original, stage1, stage2b, manifest = self.sources(
                Path(temp)
            )
            first = json.loads((stage1 / "manifest.json").read_text())
            converted = MAMDANI.verify_sources(
                stage1_dir=stage1, stage2b_dir=stage2b,
                stage1_manifest=first, stage2b_manifest=manifest,
                raw_rules=raw, stage1_predictions=original, frame=frame,
            )
            predicted, activated, metrics, plot = (
                MAMDANI.evaluate_fixed_rules(frame, converted)
            )
            self.assertEqual(len(predicted), 104)
            self.assertEqual(len(activated), 104 * len(raw))
            self.assertEqual(predicted["status"].value_counts()["OK"], 104)
            self.assertTrue(
                predicted["mamdani_prediction"].between(
                    float(converted["output_centres"][0]),
                    float(converted["output_centres"][-1])
                ).all()
            )
            self.assertEqual(metrics.iloc[0]["comparison"], "MLR vs Mamdani")
            self.assertEqual(metrics.iloc[0]["n_unavailable"], 0)
            self.assertTrue(np.isfinite(plot["mamdani"]))

    def test_source_tampering_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            frame, raw, original, stage1, stage2b, manifest = self.sources(
                Path(temp)
            )
            first = json.loads((stage1 / "manifest.json").read_text())
            (stage1 / "numerical_rules.csv").write_text("tampered")
            with self.assertRaisesRegex(BASE.ConversionError, "SHA256"):
                MAMDANI.verify_sources(
                    stage1_dir=stage1, stage2b_dir=stage2b,
                    stage1_manifest=first, stage2b_manifest=manifest,
                    raw_rules=raw, stage1_predictions=original, frame=frame,
                )

    def test_output_files_and_overwrite_guard(self):
        with tempfile.TemporaryDirectory() as temp:
            frame, raw, original, stage1, stage2b, manifest = self.sources(
                Path(temp)
            )
            converted = MAMDANI.verify_sources(
                stage1_dir=stage1, stage2b_dir=stage2b,
                stage1_manifest=json.loads(
                    (stage1 / "manifest.json").read_text()),
                stage2b_manifest=manifest,
                raw_rules=raw, stage1_predictions=original, frame=frame,
            )
            predicted, activated, metrics, plot = (
                MAMDANI.evaluate_fixed_rules(frame, converted)
            )
            # Existing MOD-14 paths are used only for provenance hashes here.
            mod14 = Path(temp) / "mod14"
            mod14.mkdir()
            for name in ("regression_coefficients.csv", "training_predictions.csv"):
                (mod14 / name).write_text("synthetic")
            out = Path(temp) / "stage3"
            MAMDANI.write_outputs(
                out, predicted, activated, metrics, plot,
                frame=frame, rule_count=len(raw),
                stage1_dir=stage1, stage2b_dir=stage2b, mod14_dir=mod14,
            )
            self.assertTrue((out / "example_mamdani_aggregation.png").is_file())
            self.assertEqual(len(pd.read_csv(out / "rule_activations.csv")),
                             104 * len(raw))
            saved_manifest = json.loads(
                (out / "mamdani_manifest.json").read_text()
            )
            self.assertEqual(saved_manifest["test_participants_used"], 0)
            self.assertEqual(saved_manifest["rule_count"], len(raw))
            with self.assertRaises(BASE.ConversionError):
                MAMDANI.write_outputs(
                    out, predicted, activated, metrics, plot,
                    frame=frame, rule_count=len(raw),
                    stage1_dir=stage1, stage2b_dir=stage2b, mod14_dir=mod14,
                )


if __name__ == "__main__":
    unittest.main()
