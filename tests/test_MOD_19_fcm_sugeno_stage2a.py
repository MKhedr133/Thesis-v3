"""Synthetic checks for Auditory FCM-Sugeno Stage 2A degeneracy diagnostics."""

import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))
import MOD_19_fcm_sugeno_stage2a as S2


def synthetic():
    rows = []
    for participant in range(26):
        for stage, level in enumerate((0, 2, 6, 10)):
            rows.append({
                "participant_id": f"synthetic-{participant:02d}",
                "difficulty_level": level, "difficulty_stage": stage,
                S2.FEATURES[0]: 1.0 + (participant + stage) % 7,
                S2.FEATURES[1]: 2.0 + (participant * 3 + stage) % 12,
                S2.FEATURES[2]: .1 + (participant + 3 * stage) % 9,
                S2.FEATURES[3]: 30.0 + (participant * 7 + stage) % 28,
            })
    frame = pd.DataFrame(rows)
    teacher = np.array([.3, .1, .04, .2, .005])
    x = np.column_stack([np.ones(len(frame)), frame[list(S2.FEATURES)].to_numpy(dtype=float)])
    frame["frozen_mlr_prediction"] = x @ teacher
    scaled, maxima = S2.STAGE1.scale_by_training_max(
        frame[list(S2.FEATURES)].to_numpy(dtype=float)
    )
    fit = S2.STAGE1.fit_fcm(scaled, clusters=3)
    assert fit.converged
    return frame, teacher, maxima, fit


class Stage2ATests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.frame, cls.teacher, cls.maxima, cls.fit = synthetic()

    def test_full_rank_linear_teacher_yields_identical_consequents(self):
        result = S2.diagnose_local_consequents(self.frame, self.fit.memberships, self.teacher)
        self.assertTrue(result.degenerate_identical_consequents)
        self.assertEqual(len(result.rows), 3)
        self.assertEqual(result.coefficients.shape, (3, 5))
        np.testing.assert_allclose(result.coefficients, np.tile(self.teacher, (3, 1)), atol=1e-9)
        self.assertLess(result.maximum_teacher_error, 1e-9)
        self.assertLess(result.maximum_consequent_difference, 1e-9)
        self.assertTrue(all(row["weighted_design_rank"] == 5 for row in result.rows))

    def test_different_membership_weights_still_recover_same_teacher(self):
        weights = self.fit.memberships[:, [2, 0, 1]]
        result = S2.diagnose_local_consequents(self.frame, weights, self.teacher)
        self.assertTrue(result.degenerate_identical_consequents)
        np.testing.assert_allclose(result.coefficients, np.tile(self.teacher, (3, 1)), atol=1e-9)

    def test_teacher_tampering_fails(self):
        damaged = self.frame.copy()
        damaged.loc[0, "frozen_mlr_prediction"] += .2
        with self.assertRaisesRegex(S2.Stage2AError, "reconstruct"):
            S2.diagnose_local_consequents(damaged, self.fit.memberships, self.teacher)

    def test_invalid_memberships_fail(self):
        for damaged in (
            self.fit.memberships[:-1, :],
            np.full((104, 3), np.nan),
            self.fit.memberships * 2,
        ):
            with self.subTest(shape=damaged.shape):
                with self.assertRaises(S2.Stage2AError):
                    S2.diagnose_local_consequents(self.frame, damaged, self.teacher)

    def test_rank_deficient_local_design_fails(self):
        damaged = self.frame.copy()
        damaged.loc[:, list(S2.FEATURES)] = 1.0
        damaged["frozen_mlr_prediction"] = np.sum(self.teacher)
        with self.assertRaisesRegex(S2.Stage2AError, "rank-deficient"):
            S2.diagnose_local_consequents(damaged, self.fit.memberships, self.teacher)

    def test_teacher_coefficient_schema_and_order(self):
        table = pd.DataFrame([
            {"model": "auditory", "predictor": term, "coefficient": v}
            for term, v in zip(S2.COEFFICIENT_NAMES, self.teacher)
        ]).iloc[::-1]
        np.testing.assert_allclose(S2.frozen_teacher_coefficients(table), self.teacher)
        invalid = table.iloc[:-1]
        with self.assertRaisesRegex(S2.Stage2AError, "coefficient names"):
            S2.frozen_teacher_coefficients(invalid)

    def make_sources_and_stage1(self, root: Path):
        sources = {}
        for filename in (
            "modeling_data.csv", "participant_holdout_split.csv",
            "mod14_regression_coefficients.csv", "mod14_training_predictions.csv",
        ):
            path = root / filename
            path.write_text(f"synthetic only: {filename}\n", encoding="utf-8")
            sources[filename] = path
        out = root / "fcm"
        S2.STAGE1.write_outputs(
            out, self.frame, self.fit, self.maxima,
            clusters=3, sources=sources,
        )
        return sources, out

    def test_reconcile_saved_memberships_with_centres_and_provenance(self):
        with tempfile.TemporaryDirectory() as td:
            sources, out = self.make_sources_and_stage1(Path(td))
            actual, manifest = S2.read_verified_stage1(self.frame, out, sources)
            np.testing.assert_allclose(actual, self.fit.memberships, atol=1e-9)
            self.assertEqual(manifest["reserved_test_participants_used"], 0)

    def test_source_hash_change_is_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            sources, out = self.make_sources_and_stage1(Path(td))
            sources["modeling_data.csv"].write_text("tampered", encoding="utf8")
            with self.assertRaisesRegex(S2.Stage2AError, "source changed"):
                S2.read_verified_stage1(self.frame, out, sources)

    def test_membership_alignment_and_content_checks(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            sources, out = self.make_sources_and_stage1(root)
            csv = out / "development_memberships.csv"
            original = pd.read_csv(csv)
            reversed_rows = original.iloc[::-1].reset_index(drop=True)
            reversed_rows.to_csv(csv, index=False)
            with self.assertRaisesRegex(S2.Stage2AError, "alignment"):
                S2.read_verified_stage1(self.frame, out, sources)
            original.loc[0, "cluster_1_membership"] = 0.99
            original.to_csv(csv, index=False)
            with self.assertRaisesRegex(S2.Stage2AError, "invalid"):
                S2.read_verified_stage1(self.frame, out, sources)

    def test_centre_corruption_fails(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            sources, out = self.make_sources_and_stage1(root)
            csv = out / "cluster_centres.csv"
            centres = pd.read_csv(csv)
            centres.loc[0, S2.FEATURES[0]] += .05
            centres.to_csv(csv, index=False)
            with self.assertRaisesRegex(S2.Stage2AError, "memberships do not match"):
                S2.read_verified_stage1(self.frame, out, sources)

    def test_output_manifest_is_aggregate_and_non_overwriting(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            sources, out = self.make_sources_and_stage1(root)
            diag = S2.diagnose_local_consequents(self.frame, self.fit.memberships, self.teacher)
            destination = root / "stage2a"
            S2.write_diagnostic(destination, diag, sources, out)
            manifest = json.loads((destination / "degeneracy_manifest.json").read_text())
            self.assertTrue(manifest["identical_MLR_consequent_degeneracy_detected"])
            self.assertEqual(manifest["reserved_test_participants_used"], 0)
            self.assertNotIn("synthetic-00", json.dumps(manifest))
            table = pd.read_csv(destination / "consequent_diagnostics.csv")
            self.assertEqual(len(table), 3)
            self.assertNotIn("participant_id", table.columns)
            with self.assertRaisesRegex(S2.Stage2AError, "already exists"):
                S2.write_diagnostic(destination, diag, sources, out)


if __name__ == "__main__":
    unittest.main()
