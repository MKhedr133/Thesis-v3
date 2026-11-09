"""Synthetic tests for MOD-19's independent Auditory FCM clustering stage."""

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))
import MOD_19_fcm_sugeno_stage1 as FCM


def fixture():
    names = FCM.MOD14.AUDITORY_PREDICTORS
    records = []
    for person in range(26):
        for stage, level in enumerate((0, 2, 6, 10)):
            records.append({
                "participant_id": f"synthetic-{person:02d}",
                "difficulty_level": level, "difficulty_stage": stage,
                names[0]: 1.0 + (person + stage) % 7,
                names[1]: 2.0 + (person * 3 + stage) % 12,
                names[2]: 0.1 + (person + 3 * stage) % 9,
                names[3]: 30.0 + (person * 7 + stage) % 28,
            })
    frame = pd.DataFrame(records)
    frame["frozen_mlr_prediction"] = (
        0.3 + frame[names[0]] * 0.1 + frame[names[1]] * 0.04
        + frame[names[2]] * 0.2 + frame[names[3]] * 0.005
    )
    return frame


class FcmStage1Tests(unittest.TestCase):
    def test_normalized_memberships_include_exact_centres(self):
        x = np.array([[0., 0.], [1., 1.], [0.5, 0.5]])
        centres = np.array([[0., 0.], [1., 1.]])
        membership = FCM.membership_from_centres(x, centres)
        np.testing.assert_allclose(membership, [[1., 0.], [0., 1.], [0.5, 0.5]])

    def test_identical_centres_split_tie_without_nan(self):
        x = np.array([[0., 0.], [0.5, 0.5]])
        centres = np.array([[0., 0.], [0., 0.], [1., 1.]])
        membership = FCM.membership_from_centres(x, centres)
        np.testing.assert_allclose(membership[0], [0.5, 0.5, 0.])
        self.assertTrue(np.isfinite(membership).all())
        np.testing.assert_allclose(membership.sum(axis=1), 1.)

    def test_training_max_scaling_preserves_zero(self):
        x = np.array([[0., 2.], [10., 4.], [5., 1.]])
        scaled, maxima = FCM.scale_by_training_max(x)
        np.testing.assert_allclose(maxima, [10., 4.])
        np.testing.assert_allclose(scaled[0], [0., 0.5])
        np.testing.assert_allclose(scaled.max(axis=0), [1., 1.])

    def test_scaling_rejects_negative_missing_and_constant_zero(self):
        for value in [np.array([[-1., 2.], [3., 4.]]),
                      np.array([[np.nan, 2.], [3., 4.]]),
                      np.array([[0., 2.], [0., 4.]])]:
            with self.subTest(value=value):
                with self.assertRaises(FCM.FcmStage1Error):
                    FCM.scale_by_training_max(value)

    def test_fcm_determinism_and_finite_objective(self):
        x = np.array([[0., 0.], [0.1, 0.], [0., .1],
                      [0.9, 1.], [1., 0.9], [1., 1.]])
        first = FCM.fit_fcm(x, clusters=2)
        second = FCM.fit_fcm(x, clusters=2)
        self.assertTrue(first.converged)
        np.testing.assert_allclose(first.centres, second.centres)
        np.testing.assert_allclose(first.memberships, second.memberships)
        self.assertTrue(np.isfinite(first.objective))
        np.testing.assert_allclose(first.memberships.sum(axis=1), 1.)
        self.assertEqual(first.centres.shape, (2, 2))
        self.assertLess(first.objective, 0.06)

    def test_input_target_never_passed_to_fcm(self):
        frame = fixture()
        names = FCM.MOD14.AUDITORY_PREDICTORS
        original = frame.loc[:, names].to_numpy(dtype=float)
        scaled, _ = FCM.scale_by_training_max(original)
        a = FCM.fit_fcm(scaled, clusters=3)
        altered = frame.copy()
        altered["frozen_mlr_prediction"] += 10000
        altered_scaled, _ = FCM.scale_by_training_max(
            altered.loc[:, names].to_numpy(dtype=float))
        b = FCM.fit_fcm(altered_scaled, clusters=3)
        np.testing.assert_array_equal(a.centres, b.centres)
        np.testing.assert_array_equal(a.memberships, b.memberships)

    def test_invalid_clusters_and_nan_rejected(self):
        x = np.array([[0., 0.], [.2, .4], [1., 1.]])
        for k in (0, 1, 4, True):
            with self.subTest(k=k):
                with self.assertRaises(FCM.FcmStage1Error):
                    FCM.fit_fcm(x, clusters=k)
        with self.assertRaises(FCM.FcmStage1Error):
            FCM.fit_fcm(x, clusters=2, fuzzifier=1.0)
        x[1, 0] = np.nan
        with self.assertRaises(FCM.FcmStage1Error):
            FCM.fit_fcm(x, clusters=2)

    def test_enforces_26_person_four_stage_contract(self):
        frame = fixture()
        FCM.validate_development_frame(frame)
        with self.assertRaises(FCM.FcmStage1Error):
            FCM.validate_development_frame(frame.iloc[:-1])
        bad = frame.copy()
        bad.loc[1, "difficulty_level"] = 0
        with self.assertRaisesRegex(FCM.FcmStage1Error, "Duplicate"):
            FCM.validate_development_frame(bad)
        bad = frame.copy()
        bad.loc[1, "difficulty_stage"] = 3
        with self.assertRaisesRegex(FCM.FcmStage1Error, "stage coding"):
            FCM.validate_development_frame(bad)

    def test_reconstruct_frozen_teacher_and_reject_tampered_outputs(self):
        frame = fixture()
        names = FCM.MOD14.AUDITORY_PREDICTORS
        coef = pd.DataFrame([
            {"model": "auditory", "predictor": "Intercept", "coefficient": 0.3},
            *({"model": "auditory", "predictor": n, "coefficient": c}
              for n, c in zip(names, (.1, .04, .2, .005))),
        ])
        stored = pd.DataFrame({"model": "auditory",
                               "participant_id": frame["participant_id"],
                               "predicted_difficulty_stage": frame["frozen_mlr_prediction"]})
        ids = tuple(sorted(frame["participant_id"].unique()))
        with (patch.object(FCM.MOD14, "prepare_split_data", return_value=(frame, None, ids, ())),
              patch.object(FCM.MOD14, "prepare_model_frame", return_value=frame)):
            result = FCM.prepare_frozen_development(frame, pd.DataFrame(), coef, stored)
            np.testing.assert_allclose(result["frozen_mlr_prediction"],
                                       frame["frozen_mlr_prediction"], atol=1e-8)
            modified = stored.copy()
            modified.loc[0, "predicted_difficulty_stage"] += .02
            with self.assertRaisesRegex(FCM.FcmStage1Error, "reconstruction"):
                FCM.prepare_frozen_development(frame, pd.DataFrame(), coef, modified)
            modified = stored.copy()
            modified.loc[0, "participant_id"] = "other"
            with self.assertRaisesRegex(FCM.FcmStage1Error, "identity/order"):
                FCM.prepare_frozen_development(frame, pd.DataFrame(), coef, modified)

    def test_manifest_guard_and_development_only_outputs(self):
        frame = fixture()
        scaled, maxima = FCM.scale_by_training_max(
            frame.loc[:, FCM.MOD14.AUDITORY_PREDICTORS].to_numpy(dtype=float))
        fit = FCM.fit_fcm(scaled, clusters=2)
        self.assertTrue(fit.converged)
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            source = root / "source.csv"
            source.write_text("synthetic-only", encoding="utf8")
            dest = root / "output"
            FCM.write_outputs(dest, frame, fit, maxima, clusters=2,
                              sources={"synthetic.csv": source})
            manifest = json.loads((dest / "cluster_manifest.json").read_text())
            self.assertEqual(manifest["development_rows"], 104)
            self.assertEqual(manifest["reserved_test_participants_used"], 0)
            self.assertEqual(manifest["source_sha256"]["synthetic.csv"], FCM.sha256(source))
            self.assertNotIn("synthetic-00", json.dumps(manifest))
            self.assertEqual(len(pd.read_csv(dest / "development_memberships.csv")), 104)
            with self.assertRaisesRegex(FCM.FcmStage1Error, "already exists"):
                FCM.write_outputs(dest, frame, fit, maxima, clusters=2,
                                  sources={"synthetic.csv": source})

    def test_nonconverged_fcm_cannot_be_saved(self):
        frame = fixture()
        scaled, maxima = FCM.scale_by_training_max(
            frame.loc[:, FCM.MOD14.AUDITORY_PREDICTORS].to_numpy(dtype=float))
        fit = FCM.fit_fcm(scaled, clusters=4, max_iter=1)
        self.assertFalse(fit.converged)
        with tempfile.TemporaryDirectory() as td:
            source = Path(td) / "fixture.txt"
            source.write_text("test")
            with self.assertRaisesRegex(FCM.FcmStage1Error, "unconverged"):
                FCM.write_outputs(Path(td) / "new", frame, fit, maxima,
                                  clusters=4, sources={"fixture": source})


if __name__ == "__main__":
    unittest.main()
