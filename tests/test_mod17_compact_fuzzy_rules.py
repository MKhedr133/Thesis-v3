"""MOD-17: fuzzy-rule interpolation, fixed holdout, fidelity and figure tests."""
from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))

import MOD_13_difficulty_prediction as MOD13
import MOD_14_direct_linear_regression_evaluation as MOD14
import MOD_17_compact_fuzzy_rules as MOD17
from test_mod14_direct_linear_regression_evaluation import (
    regression_ready_modeling_data, synthetic_holdout_split,
)


class MembershipTests(unittest.TestCase):
    def test_partition_and_linear_reconstruction(self):
        xdev = pd.Series([0., 3., 8., 4., 6.])
        var = MOD17.calibrate("mental_demand_score_0_to_10", xdev, 0.225687)
        self.assertEqual(var.anchors(), (0., 4., 8.))
        for value in np.linspace(0, 8, 27):
            w, residual, out = MOD17.memberships(float(value), var)
            self.assertAlmostEqual(float(w.sum()), 1.)
            self.assertTrue((w >= 0).all())
            self.assertTrue((w <= 1).all())
            self.assertFalse(out)
            self.assertAlmostEqual(residual, 0.)
            expected = var.coefficient * (value - var.development_mean)
            self.assertAlmostEqual(MOD17.evaluate_feature(value, var)["fuzzy_adjustment"], expected)

    def test_untouched_development_anchors_for_test_outlier(self):
        var = MOD17.calibrate("test", pd.Series([0., 3., 8.]), 0.25)
        w, correction, outside = MOD17.memberships(10., var)
        self.assertTrue(outside)
        self.assertAlmostEqual(correction, 0.5)
        self.assertEqual(var.anchors(), (0., 3., 8.))
        self.assertAlmostEqual(float(w.sum()), 1.)
        self.assertAlmostEqual(MOD17.evaluate_feature(10., var)["fuzzy_adjustment"],
                               .25 * (10. - var.development_mean))

    def test_tied_and_constant_anchors(self):
        for values in ([2., 2., 2.], [2., 2., 5.], [2., 5., 5.]):
            var = MOD17.calibrate("discrete", pd.Series(values), -0.1)
            for x in [1., 2., 3., 5., 7.]:
                w, _, _ = MOD17.memberships(x, var)
                self.assertAlmostEqual(w.sum(), 1.)
                self.assertTrue(np.isfinite(w).all())
                self.assertAlmostEqual(MOD17.evaluate_feature(x, var)["fuzzy_adjustment"],
                                       var.coefficient * (x - var.development_mean))

    def test_bad_values_are_rejected(self):
        with self.assertRaises(MOD17.Mod17Error):
            MOD17.calibrate("x", pd.Series([1., np.nan]), 2.)
        var = MOD17.calibrate("x", pd.Series([1., 2.]), 2.)
        with self.assertRaises(MOD17.Mod17Error):
            MOD17.memberships(np.inf, var)


class ExplanationPlotRegressionTests(unittest.TestCase):
    def test_plot_reads_feature_name_from_index(self):
        """Regression: set_index('feature') must not cause KeyError('feature')."""
        feature = "mental_demand_score_0_to_10"
        trials = pd.DataFrame([{
            "participant_id": "P23",
            "model_row_index": 0,
            "difficulty_level": 10,
            "difficulty_stage": 3,
            "reference_prediction": 1.500,
            "fuzzy_reconstruction": 2.391,
            "predicted_difficulty_stage": 2.391,
            "explanation": "Illustrative regression test explanation.",
            "out_of_anchor_range_count": 0,
        }])
        activations = pd.DataFrame([{
            "participant_id": "P23",
            "model_row_index": 0,
            "feature": feature,
            "value": 6.0,
            "low_membership": 0.0,
            "medium_membership": 0.4,
            "high_membership": 0.6,
            "fuzzy_adjustment": 0.729,
            "out_of_anchor_range": False,
        }])
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "participant_P23.png"
            MOD17.plot_participant_explanations(
                "auditory", trials, activations, (feature,), output
            )
            self.assertTrue(output.is_file())
            self.assertGreater(output.stat().st_size, 0)


class IntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.raw = regression_ready_modeling_data()
        cls.split = synthetic_holdout_split()
        dev, test, _, _ = MOD14.prepare_split_data(
            modeling_data=cls.raw, holdout_split=cls.split)
        cls.registry = MOD14.build_model_registry()
        cls.coefficients = []
        cls.train = []
        cls.test = []
        for model_id, spec in cls.registry.items():
            dv = MOD14.prepare_model_frame(dev, model_id=model_id)
            ts = MOD14.prepare_model_frame(test, model_id=model_id)
            # Synthetic, fixed coefficients with a non-zero intercept.
            coefficients = pd.Series(
                {"Intercept": 0.8, **{
                    feature: (0.12 if k % 2 == 0 else -0.03)
                    for k, feature in enumerate(spec.predictors)
                }})
            model = MOD13.LinearModelFit(model_id=model_id,
                                         predictors=spec.predictors,
                                         coefficients=coefficients)
            for term, value in coefficients.items():
                cls.coefficients.append({"model": model_id, "condition": spec.condition,
                                         "predictor": term, "coefficient": value})
            for frame, destination in ((dv, cls.train), (ts, cls.test)):
                predictions = model.predict(frame).to_numpy()
                for pos, (_, row) in enumerate(frame.iterrows()):
                    destination.append({
                        "model": model_id, "participant_id": row["participant_id"],
                        "observed_difficulty_stage": int(row["difficulty_stage"]),
                        "predicted_difficulty_stage": float(predictions[pos]),
                    })
        cls.coefficients = pd.DataFrame(cls.coefficients)
        cls.train = pd.DataFrame(cls.train)
        cls.test = pd.DataFrame(cls.test)

    def test_all_models_outputs_and_90_rules(self):
        with tempfile.TemporaryDirectory() as temporary:
            result = MOD17.run_analysis(
                modeling_data=self.raw, holdout_split=self.split,
                coefficient_table=self.coefficients,
                development_predictions=self.train,
                test_predictions=self.test,
                output_dir=Path(temporary) / "mod17",
            )
            self.assertEqual(len(result), 8)
            self.assertTrue(result["passes"].all())
            self.assertEqual(set(result["participant_count"]), {6, 26})
            output = Path(temporary) / "mod17"
            rules = pd.read_csv(output / "fuzzy_rules.csv")
            self.assertEqual(len(rules), 90)
            for model, num in [("visual", 7), ("auditory", 4),
                               ("cognitive_primary", 9), ("cognitive_later", 10)]:
                directory = output / model
                self.assertEqual(len(rules[rules["model"] == model]), 3 * num)
                self.assertTrue((directory / "membership_functions.png").is_file())
                self.assertTrue((directory / "rule_usage_development_vs_test.png").is_file())
                for folder in ("rule_activations_test", "fuzzy_explanations_test"):
                    self.assertEqual(len(list((directory / folder).glob("participant_*.png"))), 6)
                summary = pd.read_csv(directory / "explanations_test.csv")
                self.assertTrue(np.allclose(summary["predicted_difficulty_stage"],
                                            summary["fuzzy_reconstruction"], atol=1e-8, rtol=0))
                stages = {1, 2, 3} if model == "cognitive_later" else {0, 1, 2, 3}
                self.assertEqual(set(summary["difficulty_stage"]), stages)

    def test_original_predictions_cannot_be_changed(self):
        wrong = self.test.copy()
        wrong.loc[0, "predicted_difficulty_stage"] += 0.01
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaises(MOD17.Mod17Error):
                MOD17.run_analysis(
                    modeling_data=self.raw, holdout_split=self.split,
                    coefficient_table=self.coefficients,
                    development_predictions=self.train,
                    test_predictions=wrong,
                    output_dir=Path(temporary) / "mod17",
                )
            self.assertFalse((Path(temporary) / "mod17").exists())

    def test_test_participant_changes_do_not_change_calibration(self):
        dev, tst, _, _ = MOD14.prepare_split_data(
            modeling_data=self.raw, holdout_split=self.split)
        spec = self.registry["auditory"]
        dv = MOD14.prepare_model_frame(dev, model_id="auditory")
        before = MOD17.calibrate(spec.predictors[0], dv[spec.predictors[0]], .25)
        tst.loc[tst["condition_name"] == "Auditory", spec.predictors[0]] += 999.
        after = MOD17.calibrate(spec.predictors[0], dv[spec.predictors[0]], .25)
        self.assertEqual(before, after)


if __name__ == "__main__":
    unittest.main()
