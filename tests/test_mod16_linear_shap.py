"""Synthetic and integration-contract tests for MOD-16 independent linear SHAP."""
from __future__ import annotations

from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))

import MOD_13_difficulty_prediction as MOD13
import MOD_14_direct_linear_regression_evaluation as MOD14
import MOD_16_linear_shap as MOD16
from test_mod14_direct_linear_regression_evaluation import (
    regression_ready_split_data,
    synthetic_holdout_split,
)


class ExactLinearShapTests(unittest.TestCase):
    def setUp(self):
        self.predictors = (
            "mental_demand_score_0_to_10",
            "median_time_to_target_seconds",
        )
        self.model = MOD13.LinearModelFit(
            model_id="visual",
            predictors=self.predictors,
            coefficients=pd.Series({
                "Intercept": 0.40,
                self.predictors[0]: 0.20,
                self.predictors[1]: 0.01,
            }),
        )
        self.background = pd.DataFrame({
            self.predictors[0]: [4.0, 6.0],
            self.predictors[1]: [30.0, 20.0],
        })
        self.observations = pd.DataFrame({
            self.predictors[0]: [7.0, 4.0],
            self.predictors[1]: [20.0, 30.0],
            "participant_id": ["P01", "P02"],
            "condition_name": ["Visual", "Visual"],
            "difficulty_level": [6, 2],
            "difficulty_stage": [2, 1],
        })

    def test_exact_linear_shap_matches_coefficient_centering(self):
        result, base, means = MOD16.explain_linear_frame(
            model_id="visual", model=self.model,
            background=self.background, observations=self.observations,
            split_name="test",
        )
        self.assertAlmostEqual(float(means[self.predictors[0]]), 5.0)
        self.assertAlmostEqual(float(means[self.predictors[1]]), 25.0)
        self.assertAlmostEqual(base, 0.4 + 0.2 * 5 + 0.01 * 25)
        self.assertAlmostEqual(float(result.loc[0, f"shap__{self.predictors[0]}"]), 0.4)
        self.assertAlmostEqual(float(result.loc[0, f"shap__{self.predictors[1]}"]), -0.05)
        self.assertAlmostEqual(float(result.loc[0, "predicted_difficulty_stage"]), 2.0)
        self.assertLess(float(result["reconstruction_error"].abs().max()), 1e-8)

    def test_test_observations_do_not_modify_reference(self):
        _, base1, means1 = MOD16.explain_linear_frame(
            model_id="visual", model=self.model, background=self.background,
            observations=self.observations, split_name="test",
        )
        changed = self.observations.copy()
        changed.loc[:, self.predictors[0]] += 100.0
        _, base2, means2 = MOD16.explain_linear_frame(
            model_id="visual", model=self.model, background=self.background,
            observations=changed, split_name="test",
        )
        self.assertAlmostEqual(base1, base2)
        pd.testing.assert_series_equal(means1, means2)

    def test_prediction_integrity_is_mandatory(self):
        result, _, _ = MOD16.explain_linear_frame(
            model_id="visual", model=self.model, background=self.background,
            observations=self.observations, split_name="test",
        )
        existing = result[["participant_id", "predicted_difficulty_stage"]].copy()
        existing.insert(0, "model", "visual")
        existing.loc[0, "predicted_difficulty_stage"] += 0.1
        with self.assertRaises(MOD16.Mod16Error):
            MOD16.verify_saved_predictions(
                result, existing, model_id="visual", split_name="test"
            )

    def test_bad_background_is_rejected(self):
        background = self.background.copy()
        background.loc[0, self.predictors[0]] = np.nan
        with self.assertRaises(MOD16.Mod16Error):
            MOD16.explain_linear_frame(
                model_id="visual", model=self.model, background=background,
                observations=self.observations, split_name="development",
            )


class FourModelIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        (cls.development, cls.test, cls.dev_ids, cls.test_ids) = (
            regression_ready_split_data()
        )
        cls.modeling_data = pd.concat(
            [cls.development, cls.test], ignore_index=True
        )
        cls.holdout = synthetic_holdout_split()
        cls.synthetic_fits = MOD14.evaluate_all_models(
            development_data=cls.development, test_data=cls.test,
        )
        cls.coefficients = MOD14.build_coefficients_table(cls.synthetic_fits)
        cls.training_predictions = MOD14.combine_prediction_tables(
            cls.synthetic_fits, split="training"
        )
        cls.test_predictions = MOD14.combine_prediction_tables(
            cls.synthetic_fits, split="test"
        )

    def _run(self, directory: Path, *, training_predictions=None):
        return MOD16.run_analysis(
            modeling_data=self.modeling_data,
            holdout_split=self.holdout,
            coefficient_table=self.coefficients,
            development_predictions=(
                self.training_predictions
                if training_predictions is None else training_predictions
            ),
            test_predictions=self.test_predictions,
            output_dir=directory,
        )

    def test_four_models_two_splits_outputs_and_cognitive_later_excludes_d0(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary) / "mod16"
            checks = self._run(directory)
            self.assertEqual(len(checks), 8)
            self.assertEqual(
                set(checks["model"]), set(MOD14.build_model_registry())
            )
            self.assertEqual(set(checks["split"]), {"development", "test"})
            self.assertTrue(checks["passes"].all())
            self.assertLess(
                float(checks["max_absolute_reconstruction_error"].max()), 1e-8
            )
            self.assertLess(
                float(checks["max_absolute_mod14_prediction_mismatch"].max()), 1e-8
            )
            references = pd.read_csv(directory / "development_references.csv")
            later = references.loc[references["model"].eq("cognitive_later")]
            self.assertTrue(later["eligible_stages"].eq("1,2,3").all())
            for model_id, spec in MOD14.build_model_registry().items():
                directory_model = directory / model_id
                dev = pd.read_csv(directory_model / "attributions_development.csv")
                test = pd.read_csv(directory_model / "attributions_test.csv")
                self.assertTrue(set(dev["participant_id"]).issubset(set(self.dev_ids)))
                self.assertTrue(set(test["participant_id"]).issubset(set(self.test_ids)))
                self.assertTrue(set(dev["difficulty_stage"]).issubset(set(spec.allowed_stages)))
                self.assertEqual(int(dev["participant_id"].nunique()), 26)
                self.assertEqual(int(test["participant_id"].nunique()), 6)
                self.assertAlmostEqual(
                    float(dev["reference_prediction"].iloc[0]),
                    float(test["reference_prediction"].iloc[0]),
                )
                for name in (
                    "global_importance.png",
                    "development_distribution.png",
                    "local_examples.png",
                ):
                    self.assertGreater((directory_model / name).stat().st_size, 500)
            later_frame = pd.read_csv(
                directory / "cognitive_later" / "attributions_development.csv"
            )
            self.assertNotIn(0, later_frame["difficulty_stage"].tolist())
            self.assertTrue((directory / "manifest.json").is_file())

    def test_wrong_saved_predictions_fail_before_writing(self):
        erroneous = self.training_predictions.copy()
        first = erroneous["model"].eq("visual").idxmax()
        erroneous.loc[first, "predicted_difficulty_stage"] += 1.0
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary) / "mod16"
            with self.assertRaises(MOD16.Mod16Error):
                self._run(directory, training_predictions=erroneous)
            self.assertFalse(directory.exists())

    def test_existing_outputs_are_not_overwritten(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary) / "mod16"
            directory.mkdir()
            sentinel = directory / "important.txt"
            sentinel.write_text("do not overwrite", encoding="utf-8")
            with self.assertRaises(MOD16.Mod16Error):
                self._run(directory)
            self.assertEqual(sentinel.read_text(encoding="utf-8"), "do not overwrite")


if __name__ == "__main__":
    unittest.main()
