"""Synthetic-data checks for the read-only MOD-14 accuracy analysis."""

from __future__ import annotations

import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import unittest

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))

import MOD_14_accuracy_sensitivity as ACC
import MOD_14_direct_linear_regression_evaluation as MOD14


def synthetic_input_tables() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Create realistic 26/6 participant predictions for each frozen model."""
    registry = MOD14.build_model_registry()
    tables = {}
    for split, ids in (
        ("training", [f"D{i:02d}" for i in range(1, 27)]),
        ("test", [f"T{i:02d}" for i in range(1, 7)]),
    ):
        records = []
        for model_index, (model_id, spec) in enumerate(registry.items()):
            for index, participant in enumerate(ids):
                for stage in spec.allowed_stages:
                    error_offset = (0.11, -0.29, 0.49, -0.85)[
                        (index + stage + model_index) % 4
                    ]
                    predicted = stage + error_offset
                    records.append({
                        "participant_id": participant,
                        "model": model_id,
                        "condition": spec.condition,
                        "observed_difficulty_stage": float(stage),
                        "predicted_difficulty_stage": predicted,
                        "error": float(stage - predicted),
                        "absolute_error": abs(stage - predicted),
                    })
        tables[split] = pd.DataFrame(records)

    performance = []
    for model_id, spec in registry.items():
        row = {"model": model_id, "condition": spec.condition}
        for split, prefix in (("training", "development"), ("test", "test")):
            pairs = tables[split].loc[tables[split]["model"].eq(model_id)]
            row[f"{prefix}_participant_count"] = pairs["participant_id"].nunique()
            row[f"{prefix}_row_count"] = len(pairs)
            row[f"{split}_mae"] = MOD14.participant_balanced_mae(
                participant_ids=pairs["participant_id"],
                observed=pairs["observed_difficulty_stage"],
                predicted=pairs["predicted_difficulty_stage"],
            )
            row[f"{split}_r2"] = MOD14.pooled_r2(
                observed=pairs["observed_difficulty_stage"],
                predicted=pairs["predicted_difficulty_stage"],
            )
            row[f"{split}_out_of_range_prediction_count"] = (
                MOD14._count_out_of_range_predictions(
                    pairs["predicted_difficulty_stage"]
                )
            )
        performance.append(row)
    return tables["training"], tables["test"], pd.DataFrame(performance)


def save_inputs(directory: Path, training: pd.DataFrame, testing: pd.DataFrame,
                performance: pd.DataFrame) -> tuple[Path, Path, Path]:
    directory.mkdir(parents=True, exist_ok=True)
    paths = tuple(directory / filename for filename in (
        "training_predictions.csv",
        "test_predictions.csv",
        "train_test_performance.csv",
    ))
    for path, frame in zip(paths, (training, testing, performance)):
        frame.to_csv(path, index=False)
    return paths


class ThresholdScoringTests(unittest.TestCase):
    def test_correct_and_incorrect_predictions(self):
        observed = [0, 1, 2, 3]
        predicted = [0.1, 1.3, 1.6, 2.2]
        expected = ((0.25, (1, 4, 25.0)),
                    (0.50, (3, 4, 75.0)),
                    (0.75, (3, 4, 75.0)))
        for threshold, result in expected:
            with self.subTest(threshold=threshold):
                self.assertEqual(ACC.exact_tolerance_accuracy(
                    observed, predicted, threshold=threshold), result)

    def test_strict_boundary_for_all_three_thresholds(self):
        for threshold in ACC.THRESHOLDS:
            with self.subTest(threshold=threshold):
                self.assertEqual(
                    ACC.exact_tolerance_accuracy(
                        [1, 1], [1 + threshold, 1 + threshold - 0.01],
                        threshold=threshold),
                    (1, 2, 50.0),
                )

    def test_out_of_range_predictions_are_not_clipped(self):
        self.assertEqual(
            ACC.exact_tolerance_accuracy([0, 3], [-0.2, 3.8], threshold=0.5),
            (1, 2, 50.0),
        )

    def test_invalid_thresholds(self):
        for threshold in (0, -1, float("nan"), float("inf")):
            with self.subTest(threshold=threshold):
                with self.assertRaises(ACC.AccuracyReportError):
                    ACC.exact_tolerance_accuracy([1], [1.1], threshold=threshold)

    def test_invalid_observed_or_predicted_values(self):
        cases = (
            ([], []), ([1], [1, 2]), ([1, np.nan], [1, 1]),
            ([1], [np.inf]), ([[1]], [[1]]),
        )
        for observed, predicted in cases:
            with self.subTest(observed=observed, predicted=predicted):
                with self.assertRaises(ACC.AccuracyReportError):
                    ACC.exact_tolerance_accuracy(
                        observed, predicted, threshold=0.5)


class InputContractTests(unittest.TestCase):
    def setUp(self):
        self.training, self.testing, self.performance = synthetic_input_tables()
        self.temp = TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def evaluate(self, training=None, testing=None, performance=None):
        paths = save_inputs(
            self.root / "mod14",
            self.training if training is None else training,
            self.testing if testing is None else testing,
            self.performance if performance is None else performance,
        )
        return ACC.load_and_calculate(
            training_path=paths[0], test_path=paths[1], performance_path=paths[2]
        )

    def test_four_frozen_models_and_split_counts(self):
        summary, details, rows = self.evaluate()
        self.assertEqual(tuple(summary["model"]), tuple(MOD14.build_model_registry()))
        self.assertEqual(len(details), 4 * 2 * 3)
        self.assertEqual(len(rows), len(self.training) + len(self.testing))
        self.assertEqual(set(summary["development_participant_count"]), {26})
        self.assertEqual(set(summary["test_participant_count"]), {6})
        self.assertEqual(set(rows["split"]), {"training", "test"})
        self.assertEqual(len(rows.loc[rows["model"].eq("cognitive_later")]), 32 * 3)

    def test_original_mae_and_r2_are_preserved(self):
        summary, _, _ = self.evaluate()
        merged = summary.merge(self.performance, on="model", suffixes=("_new", "_old"))
        for metric in ("training_mae", "test_mae", "training_r2", "test_r2"):
            np.testing.assert_allclose(
                merged[metric + "_new"], merged[metric + "_old"])

    def test_denominator_matches_all_eligible_trials(self):
        _, details, _ = self.evaluate()
        for _, row in details.iterrows():
            source = self.training if row["split"] == "training" else self.testing
            pairs = source.loc[source["model"].eq(row["model"])]
            self.assertEqual(row["evaluation_count"], len(pairs))
            self.assertEqual(
                row["correct_count"],
                int((pairs["absolute_error"] < row["threshold_stages"]).sum()),
            )

    def test_accuracy_is_monotonic_in_threshold(self):
        _, details, _ = self.evaluate()
        for _, group in details.groupby(["model", "split"]):
            self.assertTrue(group.sort_values(
                "threshold_stages")["accuracy_pct"].is_monotonic_increasing)

    def test_missing_prediction_rejected_not_removed(self):
        changed = self.testing.copy()
        changed.loc[0, "predicted_difficulty_stage"] = np.nan
        with self.assertRaisesRegex(ACC.AccuracyReportError, "invalid/non-finite"):
            self.evaluate(testing=changed)

    def test_invalid_stage_rejected(self):
        changed = self.training.copy()
        later = changed.index[changed["model"].eq("cognitive_later")][0]
        changed.loc[later, "observed_difficulty_stage"] = 0
        with self.assertRaisesRegex(ACC.AccuracyReportError, "observed stage outside"):
            self.evaluate(training=changed)

    def test_duplicate_participant_model_stage_rejected(self):
        changed = pd.concat([self.training, self.training.iloc[[0]]], ignore_index=True)
        with self.assertRaisesRegex(ACC.AccuracyReportError, "duplicate"):
            self.evaluate(training=changed)

    def test_invalid_signed_error_rejected(self):
        changed = self.testing.copy()
        changed.loc[0, "error"] += 0.15
        with self.assertRaisesRegex(ACC.AccuracyReportError, "signed errors"):
            self.evaluate(testing=changed)

    def test_invalid_absolute_error_rejected(self):
        changed = self.testing.copy()
        changed.loc[0, "absolute_error"] += 0.15
        with self.assertRaisesRegex(ACC.AccuracyReportError, "absolute errors"):
            self.evaluate(testing=changed)

    def test_development_and_test_participants_do_not_overlap(self):
        changed = self.testing.copy()
        changed.loc[changed["participant_id"].eq("T01"), "participant_id"] = "D01"
        with self.assertRaisesRegex(ACC.AccuracyReportError, "overlap"):
            self.evaluate(testing=changed)

    def test_each_model_has_same_participant_membership(self):
        changed = self.training.copy()
        omit = changed["model"].eq("visual") & changed["participant_id"].eq("D01")
        changed = changed.loc[~omit]
        with self.assertRaisesRegex(ACC.AccuracyReportError, "membership differs"):
            self.evaluate(training=changed)

    def test_modified_mae_rejected(self):
        changed = self.performance.copy()
        changed.loc[0, "training_mae"] += 0.03
        with self.assertRaisesRegex(ACC.AccuracyReportError, "MAE"):
            self.evaluate(performance=changed)

    def test_modified_r2_rejected(self):
        changed = self.performance.copy()
        changed.loc[1, "test_r2"] += 0.03
        with self.assertRaisesRegex(ACC.AccuracyReportError, "R²"):
            self.evaluate(performance=changed)

    def test_mismatched_trial_count_rejected(self):
        changed = self.performance.copy()
        changed.loc[0, "test_row_count"] += 1
        with self.assertRaisesRegex(ACC.AccuracyReportError, "row count mismatch"):
            self.evaluate(performance=changed)

    def test_invalid_model_or_condition_rejected(self):
        changed = self.training.copy()
        changed.loc[0, "model"] = "unknown_model"
        with self.assertRaisesRegex(ACC.AccuracyReportError, "expected exactly models"):
            self.evaluate(training=changed)
        changed = self.testing.copy()
        changed.loc[0, "condition"] = "Cognitive"
        with self.assertRaisesRegex(ACC.AccuracyReportError, "incorrect condition"):
            self.evaluate(testing=changed)

    def test_six_charts_written_and_original_inputs_unmodified(self):
        paths = save_inputs(self.root / "mod14", self.training,
                            self.testing, self.performance)
        original_contents = {path: path.read_bytes() for path in paths}
        summary, details, audit = self.evaluate()
        output_dir = self.root / "new_results"
        generated = ACC.create_outputs(
            wide=summary, long=details, audit=audit,
            output_dir=output_dir, input_paths=paths,
        )
        self.assertEqual(len(generated), 16)
        for path in generated:
            self.assertTrue(path.is_file(), str(path))
            self.assertGreater(path.stat().st_size, 0)
        for path, data in original_contents.items():
            self.assertEqual(path.read_bytes(), data)
        manifest = json.loads((output_dir / "analysis_manifest.json").read_text())
        self.assertEqual(manifest["thresholds_stages"], [0.25, 0.5, 0.75])
        self.assertFalse(manifest["round_or_clip_predictions"])
        with self.assertRaisesRegex(ACC.AccuracyReportError, "already exist"):
            ACC.create_outputs(
                wide=summary, long=details, audit=audit,
                output_dir=output_dir, input_paths=paths,
            )
        with self.assertRaisesRegex(ACC.AccuracyReportError, "separate directory"):
            ACC.create_outputs(
                wide=summary, long=details, audit=audit,
                output_dir=paths[0].parent, input_paths=paths, overwrite=True,
            )


if __name__ == "__main__":
    unittest.main()
