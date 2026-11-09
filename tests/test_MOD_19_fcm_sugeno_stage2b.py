"""Synthetic behavioural and scientific-boundary tests for MOD-19 Stage 2B."""

import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))
import MOD_19_fcm_sugeno_stage2b as Z


def synthetic_development():
    """Synthetic complete Auditory development frame; NO participant data."""
    rows = []
    for participant in range(26):
        for stage, level in enumerate((0, 2, 6, 10)):
            rows.append({
                "participant_id": f"synthetic-{participant:02d}",
                "difficulty_level": level,
                "difficulty_stage": stage,
                Z.FEATURES[0]: 1.0 + (participant + stage) % 7,
                Z.FEATURES[1]: 2.0 + (participant * 3 + stage) % 12,
                Z.FEATURES[2]: .1 + (participant + 3 * stage) % 9,
                Z.FEATURES[3]: 30.0 + (participant * 7 + stage) % 28,
            })
    frame = pd.DataFrame(rows)
    teacher_coefficients = np.array([.3, .1, .04, .2, .005])
    features = frame[list(Z.FEATURES)].to_numpy(dtype=float)
    frame["frozen_mlr_prediction"] = (
        np.column_stack([np.ones(len(frame)), features]) @ teacher_coefficients
    )
    scaled, maxima = Z.STAGE1.scale_by_training_max(features)
    fcm = Z.STAGE1.fit_fcm(scaled, clusters=3)
    assert fcm.converged
    return frame, maxima, fcm


class Stage2BTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.frame, cls.maxima, cls.fcm = synthetic_development()
        cls.target = cls.frame["frozen_mlr_prediction"].to_numpy(dtype=float)

    def make_model(self):
        constants, effective = Z.fit_zero_order_constants(
            self.target, self.fcm.memberships
        )
        model = Z.ZeroOrderAuditory(self.fcm.centres, self.maxima, constants)
        return model, effective

    def test_constants_are_weighted_teacher_means(self):
        u = np.array([[1., 0.], [.5, .5], [0., 1.]])
        y = np.array([1., 3., 9.])
        c, n_eff = Z.fit_zero_order_constants(y, u)
        self.assertAlmostEqual(c[0], (1. + .25 * 3.) / 1.25)
        self.assertAlmostEqual(c[1], (.25 * 3. + 9.) / 1.25)
        self.assertTrue((n_eff > 0).all())
        self.assertGreater(c[1], c[0])

    def test_three_realistically_distinct_synthetic_constants(self):
        model, effective = self.make_model()
        self.assertEqual(model.constants.shape, (3,))
        self.assertGreater(np.ptp(model.constants), 0.05)
        self.assertGreaterEqual(model.constants.min(), self.target.min())
        self.assertLessEqual(model.constants.max(), self.target.max())
        self.assertTrue((effective > 0).all())

    def test_fcm_inference_matches_saved_training_memberships(self):
        model, _ = self.make_model()
        prediction, u, contributions = model.infer(
            self.frame[list(Z.FEATURES)].to_numpy(dtype=float)
        )
        np.testing.assert_allclose(u, self.fcm.memberships, atol=1e-7)
        np.testing.assert_allclose(prediction, u @ model.constants, atol=1e-10)
        np.testing.assert_allclose(contributions.sum(axis=1), prediction, atol=1e-10)
        self.assertTrue(np.isfinite(prediction).all())
        self.assertTrue((prediction >= min(model.constants) - 1e-12).all())
        self.assertTrue((prediction <= max(model.constants) + 1e-12).all())

    def test_distinct_inputs_change_rule_weights_and_predictions(self):
        model, _ = self.make_model()
        centres_original = model.centres * model.training_maxima
        predictions, memberships, _ = model.infer(centres_original)
        np.testing.assert_allclose(np.diag(memberships), np.ones(3), atol=1e-10)
        np.testing.assert_allclose(predictions, model.constants, atol=1e-10)
        self.assertGreater(np.ptp(predictions), 0.05)

    def test_outside_training_range_is_not_hidden_imputation(self):
        model, _ = self.make_model()
        values = np.array([[16., 20., 8., 250.]])
        pred, memberships, _ = model.infer(values)
        self.assertTrue(np.isfinite(pred).all())
        np.testing.assert_allclose(memberships.sum(axis=1), [1.])
        # Mathematical extrapolation exists; clinical use is NOT validated.

    def test_invalid_memberships_are_rejected(self):
        u = self.fcm.memberships
        for damaged in (u[:-1], u * 2, np.full_like(u, np.nan), -u,
                        np.ones((len(u), 1))):
            with self.subTest(shape=damaged.shape):
                with self.assertRaises(Z.Stage2BError):
                    Z.fit_zero_order_constants(self.target, damaged)

    def test_unsupported_zero_firing_is_rejected(self):
        u = np.column_stack([np.ones(104), np.zeros(104)])
        with self.assertRaisesRegex(Z.Stage2BError, "no consequent training support"):
            Z.fit_zero_order_constants(self.target, u)

    def test_inference_refuses_missing_or_negative_features(self):
        model, _ = self.make_model()
        for bad in (np.array([[1., 2.]]), np.full((1, 4), np.nan),
                    np.array([[-1., 2., 1., 40.]])):
            with self.assertRaises(Z.Stage2BError):
                model.infer(bad)

    def test_rule_count_and_invalid_constant_rejection(self):
        with self.assertRaises(Z.Stage2BError):
            Z.combine_rule_outputs(self.fcm.memberships, np.array([1., 2.]))
        with self.assertRaises(Z.Stage2BError):
            Z.combine_rule_outputs(
                self.fcm.memberships, np.array([1., np.nan, 3.])
            )

    def test_sugeno_fit_uses_prediction_not_stage_label(self):
        shuffled = self.frame.copy()
        shuffled["difficulty_stage"] = list(reversed(shuffled["difficulty_stage"]))
        # Fit routine takes the frozen teacher and memberships, not stage labels.
        first, _ = Z.fit_zero_order_constants(
            self.target, self.fcm.memberships)
        second, _ = Z.fit_zero_order_constants(
            shuffled["frozen_mlr_prediction"].to_numpy(), self.fcm.memberships)
        np.testing.assert_allclose(first, second)

    def make_stage1(self, root):
        sources = {}
        for name in ("modeling_data.csv", "participant_holdout_split.csv",
                     "mod14_regression_coefficients.csv", "mod14_training_predictions.csv"):
            path = root / name
            path.write_text(f"synthetic-only {name}\n", encoding="utf8")
            sources[name] = path
        stage1 = root / "stage1"
        Z.STAGE1.write_outputs(
            stage1, self.frame, self.fcm, self.maxima,
            clusters=3, sources=sources)
        return stage1, sources

    def test_stage1_integrity_and_output_provenance(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            stage1, sources = self.make_stage1(root)
            saved_u, manifest = Z.STAGE2A.read_verified_stage1(
                self.frame, stage1, sources)
            model = Z.model_from_saved_stage1(
                stage1, manifest,
                Z.fit_zero_order_constants(self.target, saved_u)[0])
            predictions, _, _ = model.infer(
                self.frame[list(Z.FEATURES)].to_numpy(dtype=float))
            effective = Z.fit_zero_order_constants(self.target, saved_u)[1]
            output = root / "stage2b"
            mae, r2 = Z.write_outputs(
                output, frame=self.frame, model=model, memberships=saved_u,
                predictions=predictions, effective_n=effective,
                stage1_dir=stage1, source_paths=sources)
            summary = json.loads((output / "zero_order_manifest.json").read_text())
            self.assertEqual(summary["reserved_test_participants_used"], 0)
            self.assertIn("in-sample", summary["status"])
            self.assertIn("NOT Gaussian", summary["antecedent"])
            self.assertNotIn("synthetic-00", json.dumps(summary))
            self.assertAlmostEqual(
                summary["development_in_sample_participant_balanced_MAE"], mae)
            self.assertAlmostEqual(
                summary["development_in_sample_pooled_R2"], r2)
            self.assertEqual(
                len(pd.read_csv(output / "development_predictions.csv")), 104)
            self.assertEqual(len(pd.read_csv(output / "zero_order_rules.csv")), 3)
            self.assertNotIn(
                "participant_id", pd.read_csv(output / "zero_order_rules.csv").columns)
            with self.assertRaisesRegex(Z.Stage2BError, "already exists"):
                Z.write_outputs(
                    output, frame=self.frame, model=model, memberships=saved_u,
                    predictions=predictions, effective_n=effective,
                    stage1_dir=stage1, source_paths=sources)

    def test_misaligned_training_predictions_are_rejected(self):
        model, effective = self.make_model()
        prediction, u, _ = model.infer(
            self.frame[list(Z.FEATURES)].to_numpy(dtype=float)
        )
        with tempfile.TemporaryDirectory() as td:
            stage1, sources = self.make_stage1(Path(td))
            with self.assertRaisesRegex(Z.Stage2BError, "Prediction differs"):
                Z.write_outputs(
                    Path(td) / "new", frame=self.frame, model=model,
                    memberships=u, predictions=prediction[::-1],
                    effective_n=effective, stage1_dir=stage1,
                    source_paths=sources)


if __name__ == "__main__":
    unittest.main()
