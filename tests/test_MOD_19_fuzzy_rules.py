"""Synthetic checks for MOD-19 Stage 2: numerical-to-fuzzy conversion."""

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
import MOD_19_fuzzy_rules as RULES


class Mod19FuzzyConversionTests(unittest.TestCase):
    def test_normalisation_uses_tightest_bounds(self):
        feature = MOD14.AUDITORY_PREDICTORS[0]
        raw = MOD19.RawTreeRule(
            leaf_id=7, sample_count=4, prediction=1.2,
            conditions=(
                MOD19.TreeCondition(feature, "<=", 5),
                MOD19.TreeCondition(feature, ">", 1),
                MOD19.TreeCondition(feature, ">", 2),
                MOD19.TreeCondition(feature, "<=", 4),
            ),
        )
        normal = RULES.normalise(raw)
        self.assertEqual(normal.bounds, ((feature, 2, 4),))

    def test_contradictory_bounds_fail(self):
        name = MOD14.AUDITORY_PREDICTORS[0]
        raw = MOD19.RawTreeRule(
            leaf_id=7, sample_count=4, prediction=1.2,
            conditions=(
                MOD19.TreeCondition(name, ">", 3),
                MOD19.TreeCondition(name, "<=", 3),
            ),
        )
        with self.assertRaises(RULES.ConversionError):
            RULES.normalise(raw)

    def test_duplicate_numeric_rules_average_without_silent_drop(self):
        name = MOD14.AUDITORY_PREDICTORS[0]
        rules = (
            RULES.NumericRule((2,), 5, 0.5, ((name, None, 2),)),
            RULES.NumericRule((4,), 7, 1.5, ((name, None, 2),)),
        )
        reduced = RULES.reduce_identical_numeric(rules)
        self.assertEqual(len(reduced), 1)
        self.assertEqual(reduced[0].leaf_ids, (2, 4))
        self.assertEqual(reduced[0].sample_count, 12)
        self.assertAlmostEqual(reduced[0].output, 1.0)

    def test_triangular_partition_and_shoulder_membership(self):
        centers = np.linspace(0, 4, 5)
        np.testing.assert_allclose(RULES.membership(1.25, centers),
                                   [0, 0.75, 0.25, 0, 0])
        np.testing.assert_allclose(RULES.membership(-1, centers),
                                   [1, 0, 0, 0, 0])
        np.testing.assert_allclose(RULES.membership(5, centers),
                                   [0, 0, 0, 0, 1])
        self.assertAlmostEqual(RULES.membership(1.5, centers).sum(), 1)

    def test_duplicate_fuzzy_rules_are_audited(self):
        feature = MOD14.AUDITORY_PREDICTORS[0]
        rules = [
            {"source_leaf_ids": [1], "antecedent": [(feature, "Medium")],
             "consequent": "Low", "membership_sum": 0.9},
            {"source_leaf_ids": [2], "antecedent": [(feature, "Medium")],
             "consequent": "High", "membership_sum": 0.4},
        ]
        kept, collisions = RULES.reduce_identical_fuzzy(rules)
        self.assertEqual(len(kept), 1)
        self.assertEqual(len(collisions), 1)
        self.assertEqual(kept[0]["source_leaf_ids"], [2])
        self.assertEqual(collisions[0]["candidate_consequents"], ["Low", "High"])

    @staticmethod
    def synthetic_frames():
        predictors = MOD14.AUDITORY_PREDICTORS
        rows = []
        for i in range(32):
            stage = i % 4
            rows.append({
                "participant_id": f"F{i // 4:02d}",
                "difficulty_level": (0, 2, 6, 10)[stage],
                "difficulty_stage": stage,
                predictors[0]: i % 7,
                predictors[1]: 2.5 + (i % 10) * 0.4,
                predictors[2]: 1.1 + (i % 8) * 0.18,
                predictors[3]: 60 + (i % 12) * 5,
            })
        frame = pd.DataFrame(rows)
        frame["frozen_mlr_prediction"] = (
            0.2 + frame[predictors[0]] * 0.15
            + frame[predictors[1]] * 0.08
            + frame[predictors[2]] * 0.12
            + frame[predictors[3]] * 0.003
        )
        tree = MOD19.fit_development_tree(
            frame, max_depth=3, min_samples_leaf=2
        )
        tree_prediction = tree.predict(frame.loc[:, predictors])
        predictions = frame.loc[:, [
            "participant_id", "difficulty_level",
            "difficulty_stage", "frozen_mlr_prediction"
        ]].copy()
        predictions["provisional_tree_prediction"] = tree_prediction
        return frame, MOD19.extract_raw_rules(tree), predictions

    def test_end_to_end_conversion_outputs_and_provenance(self):
        frame, raw, preds = self.synthetic_frames()
        result = RULES.build_conversion(
            frame, raw, preds, number_terms=5
        )
        self.assertEqual(len(result["normalised"]), len(raw))
        self.assertEqual(len(result["fuzzy_raw"]), len(result["numeric"]))
        self.assertLessEqual(len(result["fuzzy_final"]), len(result["fuzzy_raw"]))
        self.assertEqual(len(result["centres"]), 5)  # 4 predictors + target
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "rules"
            RULES.write_conversion(result, out, frame=frame)
            self.assertTrue((out / "membership_functions.png").is_file())
            self.assertTrue((out / "fuzzy_rules.csv").is_file())
            self.assertTrue((out / "fuzzy_collisions.csv").is_file())
            manifest = json.loads((out / "conversion_manifest.json").read_text())
            self.assertEqual(manifest["test_participants_used"], 0)
            self.assertIn("NOT implemented", manifest["research_status"])
            with self.assertRaises(RULES.ConversionError):
                RULES.write_conversion(result, out, frame=frame)

    def test_mismatched_development_prediction_fails(self):
        frame, raw, preds = self.synthetic_frames()
        preds.loc[0, "frozen_mlr_prediction"] += 0.2
        with self.assertRaisesRegex(RULES.ConversionError, "MLR targets differ"):
            RULES.build_conversion(frame, raw, preds, number_terms=5)

    def test_wrong_row_order_fails(self):
        frame, raw, preds = self.synthetic_frames()
        preds.iloc[[0, 1]] = preds.iloc[[1, 0]].to_numpy()
        with self.assertRaisesRegex(RULES.ConversionError, "row order"):
            RULES.build_conversion(frame, raw, preds, number_terms=3)

    def test_invalid_number_of_terms_fails(self):
        frame, raw, preds = self.synthetic_frames()
        with self.assertRaises(RULES.ConversionError):
            RULES.build_conversion(frame, raw, preds, number_terms=4)


if __name__ == "__main__":
    unittest.main()
