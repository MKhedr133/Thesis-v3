"""Synthetic tests for MOD-07 training-only RI/RI+RS bootstrap evidence."""
from __future__ import annotations

from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PYDIR = ROOT / "python"
sys.path.insert(0, str(PYDIR))
SPEC = spec_from_file_location(
    "mod07_test",
    PYDIR / "MOD_07_training_random_effects_bootstrap.py",
)
if SPEC is None or SPEC.loader is None:
    raise ImportError(PYDIR / "MOD_07_training_random_effects_bootstrap.py")
MOD07 = module_from_spec(SPEC)
sys.modules[SPEC.name] = MOD07
SPEC.loader.exec_module(MOD07)


def synthetic_data():
    rows = []
    for group, prefix in (("Young", "Y"), ("Old", "O")):
        for n in range(1, 17):
            pid = f"{prefix}{n:02d}"
            for cond_i, condition in enumerate(("Visual", "Auditory", "Cognitive")):
                for stage, level in enumerate((0, 2, 6, 10)):
                    rows.append(
                        {
                            "participant_id": pid,
                            "participant_group": group,
                            "condition_name": condition,
                            "difficulty_stage": float(stage),
                            "difficulty_level": level,
                            "trial_order": stage + 1,
                            "performance_change_from_d0_percentage_points": (
                                0.0 if stage == 0 else float(stage + n % 3)
                            ),
                            "mental_demand_score_0_to_10": float(
                                stage + cond_i + n / 100
                            ),
                        }
                    )
    return pd.DataFrame(rows)


def split_and_manifest(data):
    participants = (
        data[["participant_id", "participant_group"]]
        .drop_duplicates()
        .sort_values("participant_id")
        .reset_index(drop=True)
    )
    records = []
    for group in ("Young", "Old"):
        ids = participants.loc[
            participants.participant_group.eq(group), "participant_id"
        ].tolist()
        for pos, pid in enumerate(ids, start=1):
            records.append(
                {
                    "participant_id": pid,
                    "participant_group": group,
                    "partition": "test" if pos <= 3 else "train",
                    "draw_position_within_group": pos,
                }
            )
    assignments = (
        pd.DataFrame(records)
        .sort_values("participant_id")
        .reset_index(drop=True)
    )
    import MOD_06_participant_split as mod06

    manifest = {
        "assignment_sha256": mod06.assignment_signature(assignments),
        "participant_source_sha256": mod06.participant_source_signature(
            mod06.canonical_participants(data)
        ),
    }
    return assignments, manifest


class FakeResult:
    def __init__(self, structure):
        self.converged = True
        if structure == "RI":
            self.llf = -45.0
            self.aic = 100.0
            self.bic = 105.0
            self.cov_re = np.array([[1.0]])
        else:
            self.llf = -41.0
            self.aic = 95.0
            self.bic = 103.0
            self.cov_re = np.array([[1.2, 0.1], [0.1, 0.2]])


def fake_fit(
    formula,
    data,
    structure,
    difficulty_column,
    participant_column="participant_id",
    **kwargs,
):
    return FakeResult(structure), "lbfgs", (), ()


class Mod07Tests(unittest.TestCase):
    def setUp(self):
        self.data = synthetic_data()
        self.assignments, self.manifest = split_and_manifest(self.data)

    def test_m0_formulas_exclude_age_and_tmt(self):
        specs = MOD07.build_m0_target_specs(
            (
                "mental_demand_score_0_to_10",
                "performance_change_from_d0_percentage_points",
            )
        )
        self.assertEqual(
            specs["mental_demand_score_0_to_10"].formula,
            "mental_demand_score_0_to_10 ~ difficulty_stage",
        )
        self.assertEqual(
            specs["performance_change_from_d0_percentage_points"].formula,
            (
                "performance_change_from_d0_percentage_points "
                "~ performance_difficulty_stage"
            ),
        )
        self.assertNotIn(
            "participant_group",
            specs["mental_demand_score_0_to_10"].formula,
        )
        self.assertNotIn(
            "tmt_b_seconds",
            specs["mental_demand_score_0_to_10"].formula,
        )

    def test_frozen_split_is_verified_and_test_participants_are_excluded(self):
        training, train_ids, test_ids = MOD07.validate_and_filter_training_data(
            self.data,
            self.assignments,
            self.manifest,
        )
        self.assertEqual(len(train_ids), 26)
        self.assertEqual(len(test_ids), 6)
        self.assertEqual(training.participant_id.nunique(), 26)
        self.assertTrue(set(training.participant_id).isdisjoint(test_ids))

    def test_changed_assignment_is_rejected_by_hash(self):
        bad = self.assignments.copy()
        test_i = bad.index[bad["partition"].eq("test")][0]
        train_i = bad.index[bad["partition"].eq("train")][0]
        bad.loc[test_i, "partition"] = "train"
        bad.loc[train_i, "partition"] = "test"

        with self.assertRaisesRegex(MOD07.Mod07Error, "SHA-256"):
            MOD07.validate_and_filter_training_data(
                self.data,
                bad,
                self.manifest,
            )

    def test_bootstrap_draws_are_reproducible_and_training_only(self):
        _, train_ids, test_ids = MOD07.validate_and_filter_training_data(
            self.data,
            self.assignments,
            self.manifest,
        )
        first = MOD07.generate_bootstrap_draws(
            train_ids,
            bootstrap_replicates=3,
            bootstrap_seed=77,
        )
        second = MOD07.generate_bootstrap_draws(
            train_ids,
            bootstrap_replicates=3,
            bootstrap_seed=77,
        )

        pd.testing.assert_frame_equal(first, second)
        self.assertEqual(len(first), 78)
        self.assertTrue(set(first.participant_id).issubset(set(train_ids)))
        self.assertTrue(set(first.participant_id).isdisjoint(test_ids))

    def test_duplicate_draws_receive_distinct_bootstrap_cluster_ids(self):
        training, _, _ = MOD07.validate_and_filter_training_data(
            self.data,
            self.assignments,
            self.manifest,
        )
        spec = MOD07.build_m0_target_specs(
            ("mental_demand_score_0_to_10",)
        )["mental_demand_score_0_to_10"]
        frame, _ = MOD07._prepare_target_frame(training, "Visual", spec)
        participant_id = frame.participant_id.iloc[0]

        sample = MOD07._materialize_bootstrap_sample(
            frame,
            [participant_id, participant_id],
        )

        self.assertEqual(
            sample[MOD07.BOOTSTRAP_CLUSTER_COLUMN].nunique(),
            2,
        )
        original_rows = len(
            frame.loc[frame.participant_id.eq(participant_id)]
        )
        self.assertEqual(len(sample), 2 * original_rows)

    def test_fit_pair_reports_signed_ri_rs_minus_ri_deltas(self):
        training, _, _ = MOD07.validate_and_filter_training_data(
            self.data,
            self.assignments,
            self.manifest,
        )
        spec = MOD07.build_m0_target_specs(
            ("mental_demand_score_0_to_10",)
        )["mental_demand_score_0_to_10"]
        frame, _ = MOD07._prepare_target_frame(training, "Visual", spec)

        with patch.object(
            MOD07,
            "fit_mixedlm_with_fallback",
            side_effect=fake_fit,
        ):
            pair = MOD07._fit_pair(frame, spec)

        self.assertTrue(pair["paired_converged"])
        self.assertEqual(pair["delta_aic_ri_rs_minus_ri"], -5.0)
        self.assertEqual(pair["delta_bic_ri_rs_minus_ri"], -2.0)
        self.assertAlmostEqual(
            pair["ri_rs_random_slope_variance"],
            0.2,
        )

    def test_small_synthetic_run_uses_only_training_and_summarizes_bootstrap(self):
        with patch.object(
            MOD07,
            "fit_mixedlm_with_fallback",
            side_effect=fake_fit,
        ):
            (
                comparison,
                audit,
                draws,
                replicates,
                summary,
                train_ids,
                test_ids,
            ) = MOD07.run_mod07(
                self.data,
                self.assignments,
                self.manifest,
                bootstrap_seed=11,
                bootstrap_replicates=3,
                targets=("mental_demand_score_0_to_10",),
                conditions=("Visual",),
            )

        self.assertEqual(len(comparison), 1)
        self.assertEqual(len(audit), 1)
        self.assertEqual(len(replicates), 3)
        self.assertEqual(len(draws), 78)
        self.assertEqual(comparison.loc[0, "participant_count"], 26)
        self.assertTrue(replicates["bootstrap_cluster_count"].eq(26).all())
        self.assertTrue(
            replicates["delta_aic_ri_rs_minus_ri"].eq(-5.0).all()
        )
        self.assertEqual(
            summary.loc[0, "ri_rs_lower_aic_percentage_of_valid"],
            100.0,
        )
        self.assertEqual(
            summary.loc[0, "confidence_interval_status"],
            "not_frozen_not_computed",
        )
        self.assertEqual(len(train_ids), 26)
        self.assertEqual(len(test_ids), 6)

    def test_manifest_records_no_test_use_and_no_automatic_selection(self):
        _, train_ids, test_ids = MOD07.validate_and_filter_training_data(
            self.data,
            self.assignments,
            self.manifest,
        )
        manifest = MOD07.build_manifest(
            modeling_data_path=Path("model.csv"),
            split_assignments_path=Path("split.csv"),
            split_manifest_path=Path("manifest.json"),
            split_manifest=self.manifest,
            training_ids=train_ids,
            test_ids=test_ids,
            bootstrap_seed=123,
            bootstrap_replicates=2000,
        )

        self.assertEqual(manifest["bootstrap_replicates"], 2000)
        self.assertFalse(manifest["test_participants_used_for_fitting"])
        self.assertFalse(manifest["test_participants_used_for_bootstrap"])
        self.assertFalse(manifest["automatic_random_structure_selection"])
        self.assertIsNone(manifest["confidence_interval_method"])

    def test_output_writer_refuses_overwrite(self):
        blank = pd.DataFrame({"x": [1]})
        result = MOD07.Mod07Result(
            blank,
            blank,
            blank,
            blank,
            blank,
            {"ok": True},
        )
        with tempfile.TemporaryDirectory() as directory:
            MOD07.write_outputs(result, Path(directory))
            with self.assertRaisesRegex(
                FileExistsError,
                "Refusing to overwrite",
            ):
                MOD07.write_outputs(result, Path(directory))


if __name__ == "__main__":
    unittest.main()
