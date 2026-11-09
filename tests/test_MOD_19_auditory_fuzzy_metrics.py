"""Synthetic tests for development-only Auditory fuzzy comparison.

No private participants or reserved-test outcomes are included.
"""

import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))
import MOD_19_auditory_fuzzy_metrics as E


def synthetic_frame():
    rows = []
    for participant in range(26):
        for stage, level in enumerate((0, 2, 6, 10)):
            row = {
                "participant_id": f"synthetic-{participant:02d}",
                "difficulty_level": level, "difficulty_stage": stage,
                E.FEATURES[0]: 1.0 + (participant + stage) % 7,
                E.FEATURES[1]: 2.0 + (participant * 3 + stage) % 12,
                E.FEATURES[2]: 0.1 + (participant + 3 * stage) % 9,
                E.FEATURES[3]: 30.0 + (participant * 7 + stage) % 28,
            }
            row["frozen_mlr_prediction"] = (
                .3 + row[E.FEATURES[0]] * .1 + row[E.FEATURES[1]] * .04
                + row[E.FEATURES[2]] * .2 + row[E.FEATURES[3]] * .005
            )
            rows.append(row)
    return pd.DataFrame(rows)


def other_method(frame):
    table = frame.loc[:, [*E.KEYS, "frozen_mlr_prediction"]].copy()
    table["output_3_terms_prediction"] = table["frozen_mlr_prediction"] + 0.2
    table["output_3_terms_status"] = "OK"
    table["output_5_terms_prediction"] = table["frozen_mlr_prediction"] - 0.1
    table["output_5_terms_status"] = "OK"
    return table


class AuditoryFuzzyMetricsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.frame = synthetic_frame()
        cls.scaled_fit = E.grouped_sugeno(cls.frame)
        cls.sugeno_grouped, cls.folds = cls.scaled_fit

    def test_grouped_validation_has_exactly_26_folds_and_104_rows(self):
        self.assertEqual(len(self.sugeno_grouped), 104)
        self.assertEqual(len(self.folds), 26)
        self.assertEqual(self.sugeno_grouped["participant_id"].nunique(), 26)
        self.assertTrue(self.folds["training_participants"].eq(25).all())
        self.assertTrue(self.folds["validation_participants"].eq(1).all())
        self.assertTrue(self.folds["training_observations"].eq(100).all())
        self.assertTrue(self.folds["validation_observations"].eq(4).all())
        for _, records in self.sugeno_grouped.groupby("fold"):
            self.assertEqual(len(records), 4)
            self.assertEqual(records["participant_id"].nunique(), 1)
            self.assertEqual(set(records["difficulty_stage"]), {0, 1, 2, 3})

    def test_scaling_and_clustering_fit_only_100_rows_per_fold(self):
        originals = []
        existing_scale = E.FCM.scale_by_training_max
        existing_cluster = E.FCM.fit_fcm
        with patch.object(E.FCM, "scale_by_training_max",
                          side_effect=lambda x: (originals.append(("scale", len(x)))
                                                  or existing_scale(x))):
            with patch.object(E.FCM, "fit_fcm",
                              side_effect=lambda x, **kwargs: (
                                  originals.append(("cluster", len(x)))
                                  or existing_cluster(x, **kwargs))):
                E.grouped_sugeno(self.frame)
        self.assertEqual(len(originals), 52)
        self.assertEqual(originals.count(("scale", 100)), 26)
        self.assertEqual(originals.count(("cluster", 100)), 26)

    def test_training_targets_never_contain_excluded_participant(self):
        recorded = []
        original = E.SUGENO.fit_zero_order_constants
        with patch.object(
            E.SUGENO, "fit_zero_order_constants",
            side_effect=lambda y, memberships: (
                recorded.append((len(y), memberships.shape))
                or original(y, memberships)),
        ):
            E.grouped_sugeno(self.frame)
        self.assertEqual(recorded, [(100, (100, 3))] * 26)

    def test_cv_predictions_are_finite_and_in_cluster_constant_hull(self):
        self.assertTrue(np.isfinite(
            self.sugeno_grouped["zero_order_sugeno_prediction"]).all())
        self.assertEqual(len(self.sugeno_grouped[list(E.KEYS)].drop_duplicates()), 104)
        self.assertTrue(self.folds["fcm_iterations"].between(1, 500).all())

    def test_strict_accuracy_boundary_is_not_inclusive(self):
        table = pd.DataFrame({
            "participant_id": ["a", "b"],
            "difficulty_stage": [0, 1],
            "frozen_mlr_prediction": [0, 1],
            "example": [0.25, 1.5],
        })
        values = E.record_metrics(table, method="example", subset="training",
                                  prediction="example")
        actual = next(x for x in values if x["reference"] == "observed_difficulty_stage")
        self.assertEqual(actual["accuracy_within_0p25_percent"], 0)
        self.assertEqual(actual["accuracy_within_0p50_percent"], 0)
        self.assertEqual(actual["accuracy_within_0p75_percent"], 100)

    def test_accuracy_is_absolute_units_not_fraction_of_stage_range(self):
        table = pd.DataFrame({
            "participant_id": ["a", "b"], "difficulty_stage": [0, 3],
            "frozen_mlr_prediction": [0, 3], "prediction": [.6, 2.4],
        })
        result = E.record_metrics(table, method="m", subset="training",
                                  prediction="prediction")[0]
        self.assertEqual(result["accuracy_within_0p25_percent"], 0)
        self.assertEqual(result["accuracy_within_0p50_percent"], 0)
        self.assertEqual(result["accuracy_within_0p75_percent"], 100)

    def test_mamdani_unavailable_remains_excluded_with_explicit_coverage(self):
        table = pd.DataFrame({
            "participant_id": ["a", "b"], "difficulty_stage": [0, 1],
            "frozen_mlr_prediction": [0.2, 1.2],
            "pred": [0.1, np.nan], "state": ["OK", "Unavailable"],
        })
        result = E.record_metrics(
            table, method="Mamdani", subset="validation",
            prediction="pred", status="state")[0]
        self.assertEqual(result["available_observations"], 1)
        self.assertEqual(result["unavailable_observations"], 1)
        self.assertEqual(result["coverage_percent"], 50.0)
        self.assertEqual(result["accuracy_within_0p25_percent"], 100.0)

    def test_reject_unavailable_silently_marked_ok(self):
        table = pd.DataFrame({
            "participant_id": ["a"], "difficulty_stage": [0],
            "frozen_mlr_prediction": [0.1],
            "pred": [np.nan], "state": ["OK"],
        })
        with self.assertRaisesRegex(E.ComparisonError, "must be finite"):
            E.record_metrics(
                table, method="Mamdani", subset="validation",
                prediction="pred", status="state",
            )

    def test_reject_wrong_availability_status(self):
        table = pd.DataFrame({
            "participant_id": ["a"], "difficulty_stage": [0],
            "frozen_mlr_prediction": [0.1],
            "pred": [np.nan], "state": ["Review"],
        })
        with self.assertRaisesRegex(E.ComparisonError, "availability status"):
            E.record_metrics(
                table, method="Mamdani", subset="validation",
                prediction="pred", status="state",
            )

    def test_participant_balanced_mae_differs_from_row_average(self):
        table = pd.DataFrame({
            "participant_id": ["a", "a", "b"], "difficulty_stage": [0, 0, 0],
            "frozen_mlr_prediction": [0, 0, 0], "pred": [0, 0, 1],
        })
        result = E.record_metrics(table, method="m", subset="training",
                                  prediction="pred")[0]
        self.assertAlmostEqual(result["participant_balanced_mae"], 0.5)

    def test_mismatched_prediction_identity_rejected(self):
        data = self.sugeno_grouped.copy()
        data.loc[data.index[0], "participant_id"] = "wrong-person"
        with self.assertRaisesRegex(E.ComparisonError, "identities"):
            E.check_keys_and_teacher(data, self.frame)

    def test_changed_teacher_prediction_rejected(self):
        data = self.sugeno_grouped.copy()
        data.loc[data.index[0], "frozen_mlr_prediction"] += .01
        with self.assertRaisesRegex(E.ComparisonError, "regression"):
            E.check_keys_and_teacher(data, self.frame)

    def test_incomplete_participant_rows_rejected(self):
        with self.assertRaisesRegex(E.ComparisonError, "104"):
            E.check_keys_and_teacher(self.sugeno_grouped.iloc[:-1], self.frame)

    def test_comparison_keeps_reference_and_method_separate(self):
        full = self.frame.loc[:, [*E.KEYS, "frozen_mlr_prediction"]].copy()
        full["zero_order_sugeno_prediction"] = full["frozen_mlr_prediction"]
        mamdani = other_method(self.frame)
        result = E.compare(
            self.frame, full, self.sugeno_grouped, mamdani, mamdani,
        )
        self.assertEqual(len(result), 13)
        self.assertEqual(set(result["reference"]), {
            "observed_difficulty_stage", "original_regression_prediction"})
        self.assertTrue(result["total_observations"].eq(104).all())
        self.assertEqual(set(result["evaluation"]), {
            "training", "participant-excluded", "training reference"})

    def test_grouped_validation_rejects_missing_stage(self):
        bad = self.frame.drop(index=[0])
        with self.assertRaises(E.FCM.FcmStage1Error):
            E.grouped_sugeno(bad)


if __name__ == "__main__":
    unittest.main()
