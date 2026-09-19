"""Synthetic tests for staged and parallel MOD-07 random-effects evidence."""
from __future__ import annotations

from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import json
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


def synthetic_data() -> pd.DataFrame:
    rows = []
    for group, prefix in (("Young", "Y"), ("Old", "O")):
        for number in range(1, 17):
            participant_id = f"{prefix}{number:02d}"
            for condition_index, condition in enumerate(
                ("Visual", "Auditory", "Cognitive")
            ):
                for stage, level in enumerate((0, 2, 6, 10)):
                    rows.append(
                        {
                            "participant_id": participant_id,
                            "participant_group": group,
                            "condition_name": condition,
                            "difficulty_stage": float(stage),
                            "difficulty_level": level,
                            "trial_order": stage + 1,
                            "performance_change_from_d0_percentage_points": (
                                0.0
                                if stage == 0
                                else float(stage + number % 3)
                            ),
                            "mental_demand_score_0_to_10": float(
                                stage + condition_index + number / 100
                            ),
                            "median_time_between_qualifying_grabs_seconds": float(
                                stage + number / 20
                            ),
                            "list_recheck_count": float(
                                (stage + number) % 4
                            ),
                            "total_list_recheck_duration_seconds": float(
                                stage * 2 + number / 10
                            ),
                            "median_time_to_target_seconds": float(
                                stage + number / 15
                            ),
                            "median_irrelevant_focus_duration_seconds": float(
                                stage / 2 + number / 50
                            ),
                            "median_head_turning_degrees": float(
                                stage * 10 + number
                            ),
                            "median_reach_duration_seconds": float(
                                stage + number / 10
                            ),
                            "median_reach_path_ratio": float(
                                1.0 + stage / 10 + number / 1000
                            ),
                        }
                    )
    return pd.DataFrame(rows)


def split_and_manifest(data: pd.DataFrame):
    participants = (
        data[["participant_id", "participant_group"]]
        .drop_duplicates()
        .sort_values("participant_id")
        .reset_index(drop=True)
    )
    records = []
    for group in ("Young", "Old"):
        ids = participants.loc[
            participants["participant_group"].eq(group), "participant_id"
        ].tolist()
        for position, participant_id in enumerate(ids, start=1):
            records.append(
                {
                    "participant_id": participant_id,
                    "participant_group": group,
                    "partition": "test" if position <= 3 else "train",
                    "draw_position_within_group": position,
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
    def __init__(self, structure: str):
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


class FakeProcessPoolExecutor:
    """Exercise the parallel orchestration deterministically in-process."""

    def __init__(self, *, max_workers, initializer, initargs):
        self.max_workers = max_workers
        self.initializer = initializer
        self.initargs = initargs

    def __enter__(self):
        self.initializer(*self.initargs)
        return self

    def __exit__(self, exc_type, exc, traceback):
        return False

    def map(self, function, tasks, chunksize=1):
        return map(function, tasks)


class Mod07Tests(unittest.TestCase):
    def setUp(self):
        self.data = synthetic_data()
        self.assignments, self.manifest = split_and_manifest(self.data)

    def training(self):
        return MOD07.validate_and_filter_training_data(
            self.data,
            self.assignments,
            self.manifest,
        )

    def test_default_scope_is_all_30_gaussian_pairs(self):
        pairs = MOD07.comparison_pairs()

        self.assertEqual(len(pairs), 30)
        self.assertEqual(
            {target for _, target in pairs},
            set(MOD07.GAUSSIAN_TARGETS),
        )
        self.assertEqual(
            {condition for condition, _ in pairs},
            {"Visual", "Auditory", "Cognitive"},
        )
        self.assertIn(
            "median_reach_duration_seconds",
            {target for _, target in pairs},
        )
        self.assertNotIn(
            "total_error_count",
            {target for _, target in pairs},
        )

    def test_m0_formulas_exclude_age_and_tmt(self):
        specs = MOD07.build_m0_target_specs()

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
        for spec in specs.values():
            self.assertNotIn("participant_group", spec.formula)
            self.assertNotIn("tmt_b_seconds", spec.formula)

    def test_frozen_split_is_verified_and_test_participants_are_excluded(self):
        training, training_ids, test_ids = self.training()

        self.assertEqual(len(training_ids), 26)
        self.assertEqual(len(test_ids), 6)
        self.assertEqual(training["participant_id"].nunique(), 26)
        self.assertTrue(set(training["participant_id"]).isdisjoint(test_ids))

    def test_changed_assignment_is_rejected_by_hash(self):
        bad = self.assignments.copy()
        test_index = bad.index[bad["partition"].eq("test")][0]
        train_index = bad.index[bad["partition"].eq("train")][0]
        bad.loc[test_index, "partition"] = "train"
        bad.loc[train_index, "partition"] = "test"

        with self.assertRaisesRegex(MOD07.Mod07Error, "SHA-256"):
            MOD07.validate_and_filter_training_data(
                self.data,
                bad,
                self.manifest,
            )

    def test_compare_only_uses_30_pairs_and_no_bootstrap(self):
        with patch.object(
            MOD07,
            "fit_mixedlm_with_fallback",
            side_effect=fake_fit,
        ), patch.object(
            MOD07,
            "generate_bootstrap_draws",
            side_effect=AssertionError("compare-only must not bootstrap"),
        ):
            comparison, audit, training_ids, test_ids = MOD07.run_compare_only(
                modeling_data=self.data,
                assignments=self.assignments,
                split_manifest=self.manifest,
            )

        self.assertEqual(len(comparison), 30)
        self.assertEqual(len(audit), 30)
        self.assertEqual(len(training_ids), 26)
        self.assertEqual(len(test_ids), 6)
        self.assertTrue(comparison["paired_converged"].all())

    def test_bootstrap_draws_are_reproducible_and_training_only(self):
        _, training_ids, test_ids = self.training()
        first = MOD07.generate_bootstrap_draws(
            training_ids,
            bootstrap_replicates=3,
            bootstrap_seed=MOD07.BOOTSTRAP_SEED,
        )
        second = MOD07.generate_bootstrap_draws(
            training_ids,
            bootstrap_replicates=3,
            bootstrap_seed=MOD07.BOOTSTRAP_SEED,
        )

        pd.testing.assert_frame_equal(first, second)
        self.assertEqual(len(first), 78)
        self.assertTrue(
            set(first["participant_id"]).issubset(set(training_ids))
        )
        self.assertTrue(
            set(first["participant_id"]).isdisjoint(set(test_ids))
        )

    def test_duplicate_draws_receive_distinct_bootstrap_cluster_ids(self):
        training, _, _ = self.training()
        spec = MOD07.build_m0_target_specs(
            ("mental_demand_score_0_to_10",)
        )["mental_demand_score_0_to_10"]
        frame, _ = MOD07._prepare_target_frame(training, "Visual", spec)
        participant_id = frame["participant_id"].iloc[0]

        sample = MOD07._materialize_bootstrap_sample(
            frame,
            [participant_id, participant_id],
        )

        self.assertEqual(
            sample[MOD07.BOOTSTRAP_CLUSTER_COLUMN].nunique(),
            2,
        )
        original_rows = len(
            frame.loc[frame["participant_id"].eq(participant_id)]
        )
        self.assertEqual(len(sample), 2 * original_rows)

    def test_fit_pair_reports_signed_ri_rs_minus_ri_deltas(self):
        training, _, _ = self.training()
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

    def test_selected_pairs_csv_accepts_any_gaussian_pair_only(self):
        with tempfile.TemporaryDirectory() as directory:
            valid_path = Path(directory) / "valid.csv"
            pd.DataFrame(
                [
                    {
                        "condition_name": "Visual",
                        "target_name": "mental_demand_score_0_to_10",
                    },
                    {
                        "condition_name": "Cognitive",
                        "target_name": "median_reach_duration_seconds",
                    },
                ]
            ).to_csv(valid_path, index=False)

            selected = MOD07.load_selected_pairs(valid_path)
            self.assertEqual(len(selected), 2)

            invalid_path = Path(directory) / "invalid.csv"
            pd.DataFrame(
                [
                    {
                        "condition_name": "Visual",
                        "target_name": "not_a_gaussian_target",
                    }
                ]
            ).to_csv(invalid_path, index=False)

            with self.assertRaisesRegex(
                MOD07.Mod07Error,
                "Gaussian MixedLM scope",
            ):
                MOD07.load_selected_pairs(invalid_path)

    def test_only_explicit_selected_pairs_are_bootstrapped(self):
        training, training_ids, _ = self.training()
        pairs = (("Visual", "mental_demand_score_0_to_10"),)
        frames, specs = MOD07._prepare_frames_for_pairs(training, pairs)
        draws = MOD07.generate_bootstrap_draws(
            training_ids,
            bootstrap_replicates=3,
        )

        with patch.object(
            MOD07,
            "fit_mixedlm_with_fallback",
            side_effect=fake_fit,
        ):
            results = MOD07.run_bootstrap_tasks(
                frames=frames,
                specs=specs,
                draws=draws,
                pairs=pairs,
                workers=1,
            )

        self.assertEqual(len(results), 3)
        self.assertEqual(set(results["condition_name"]), {"Visual"})
        self.assertEqual(
            set(results["target_name"]),
            {"mental_demand_score_0_to_10"},
        )

    def test_serial_and_parallel_orchestration_are_equivalent(self):
        training, training_ids, _ = self.training()
        pairs = (("Visual", "mental_demand_score_0_to_10"),)
        frames, specs = MOD07._prepare_frames_for_pairs(training, pairs)
        draws = MOD07.generate_bootstrap_draws(
            training_ids,
            bootstrap_replicates=3,
        )

        with patch.object(
            MOD07,
            "fit_mixedlm_with_fallback",
            side_effect=fake_fit,
        ):
            serial = MOD07.run_bootstrap_tasks(
                frames=frames,
                specs=specs,
                draws=draws,
                pairs=pairs,
                workers=1,
            )

        with patch.object(
            MOD07,
            "fit_mixedlm_with_fallback",
            side_effect=fake_fit,
        ), patch.object(
            MOD07,
            "ProcessPoolExecutor",
            FakeProcessPoolExecutor,
        ):
            parallel = MOD07.run_bootstrap_tasks(
                frames=frames,
                specs=specs,
                draws=draws,
                pairs=pairs,
                workers=4,
            )

        pd.testing.assert_frame_equal(serial, parallel)

    def test_checkpoint_resume_skips_completed_tasks_without_duplicates(self):
        training, training_ids, _ = self.training()
        pairs = (("Visual", "mental_demand_score_0_to_10"),)
        frames, specs = MOD07._prepare_frames_for_pairs(training, pairs)
        draws = MOD07.generate_bootstrap_draws(
            training_ids,
            bootstrap_replicates=3,
        )
        configuration = MOD07.build_bootstrap_configuration(
            pairs=pairs,
            frames=frames,
            training_ids=training_ids,
            split_manifest=self.manifest,
            draws=draws,
            bootstrap_replicates=3,
            bootstrap_seed=MOD07.BOOTSTRAP_SEED,
        )

        with tempfile.TemporaryDirectory() as directory:
            output_dir = Path(directory)
            MOD07._initialise_bootstrap_outputs(
                output_dir=output_dir,
                pairs=pairs,
                draws=draws,
                configuration=configuration,
                resume=False,
            )

            first_draw = draws.loc[
                draws["bootstrap_replicate"].eq(1)
            ].copy()
            with patch.object(
                MOD07,
                "fit_mixedlm_with_fallback",
                side_effect=fake_fit,
            ):
                partial = MOD07.run_bootstrap_tasks(
                    frames=frames,
                    specs=specs,
                    draws=first_draw,
                    pairs=pairs,
                    workers=1,
                    checkpoint_path=(
                        output_dir / MOD07.CHECKPOINT_FILENAME
                    ),
                    checkpoint_every=1,
                )
            self.assertEqual(len(partial), 1)

            existing = MOD07._initialise_bootstrap_outputs(
                output_dir=output_dir,
                pairs=pairs,
                draws=draws,
                configuration=configuration,
                resume=True,
            )
            self.assertEqual(len(existing), 1)

            with patch.object(
                MOD07,
                "fit_mixedlm_with_fallback",
                side_effect=fake_fit,
            ):
                resumed = MOD07.run_bootstrap_tasks(
                    frames=frames,
                    specs=specs,
                    draws=draws,
                    pairs=pairs,
                    workers=1,
                    existing_results=existing,
                    checkpoint_path=(
                        output_dir / MOD07.CHECKPOINT_FILENAME
                    ),
                    checkpoint_every=1,
                )

            self.assertEqual(len(resumed), 3)
            self.assertFalse(
                resumed.duplicated(
                    [
                        "bootstrap_replicate",
                        "condition_name",
                        "target_name",
                    ]
                ).any()
            )

    def test_incompatible_resume_configuration_is_rejected(self):
        training, training_ids, _ = self.training()
        pairs = (("Visual", "mental_demand_score_0_to_10"),)
        frames, _ = MOD07._prepare_frames_for_pairs(training, pairs)
        draws = MOD07.generate_bootstrap_draws(
            training_ids,
            bootstrap_replicates=2,
        )
        configuration = MOD07.build_bootstrap_configuration(
            pairs=pairs,
            frames=frames,
            training_ids=training_ids,
            split_manifest=self.manifest,
            draws=draws,
            bootstrap_replicates=2,
            bootstrap_seed=MOD07.BOOTSTRAP_SEED,
        )

        with tempfile.TemporaryDirectory() as directory:
            output_dir = Path(directory)
            MOD07._initialise_bootstrap_outputs(
                output_dir=output_dir,
                pairs=pairs,
                draws=draws,
                configuration=configuration,
                resume=False,
            )
            incompatible = dict(configuration)
            incompatible["configuration_sha256"] = "0" * 64

            with self.assertRaisesRegex(
                MOD07.Mod07Error,
                "does not match",
            ):
                MOD07._initialise_bootstrap_outputs(
                    output_dir=output_dir,
                    pairs=pairs,
                    draws=draws,
                    configuration=incompatible,
                    resume=True,
                )

    def test_frozen_bootstrap_seed_cannot_be_changed(self):
        pairs = (("Visual", "mental_demand_score_0_to_10"),)
        with self.assertRaisesRegex(
            MOD07.Mod07Error,
            "frozen",
        ):
            MOD07.run_bootstrap_selected(
                modeling_data=self.data,
                assignments=self.assignments,
                split_manifest=self.manifest,
                pairs=pairs,
                output_dir=Path("unused"),
                workers=1,
                resume=False,
                bootstrap_replicates=2,
                bootstrap_seed=123,
            )

    def test_manifest_records_no_test_use_and_no_automatic_selection(self):
        _, training_ids, test_ids = self.training()
        manifest = MOD07.build_manifest(
            mode="bootstrap-selected",
            modeling_data_path=Path("model.csv"),
            split_assignments_path=Path("split.csv"),
            split_manifest_path=Path("manifest.json"),
            split_manifest=self.manifest,
            training_ids=training_ids,
            test_ids=test_ids,
            pairs=(("Visual", "mental_demand_score_0_to_10"),),
            workers=4,
            bootstrap_replicates=2000,
            bootstrap_seed=MOD07.BOOTSTRAP_SEED,
        )

        self.assertEqual(manifest["bootstrap_replicates"], 2000)
        self.assertEqual(
            manifest["bootstrap_seed"],
            MOD07.BOOTSTRAP_SEED,
        )
        self.assertFalse(manifest["test_participants_used_for_fitting"])
        self.assertFalse(manifest["test_participants_used_for_bootstrap"])
        self.assertFalse(manifest["automatic_random_structure_selection"])
        self.assertIsNone(manifest["confidence_interval_method"])

    def test_compare_output_writer_refuses_overwrite(self):
        blank = pd.DataFrame({"x": [1]})
        with tempfile.TemporaryDirectory() as directory:
            output_dir = Path(directory)
            MOD07.write_compare_outputs(
                comparison=blank,
                audit=blank,
                manifest={"ok": True},
                output_dir=output_dir,
            )
            with self.assertRaisesRegex(
                FileExistsError,
                "Refusing to overwrite",
            ):
                MOD07.write_compare_outputs(
                    comparison=blank,
                    audit=blank,
                    manifest={"ok": True},
                    output_dir=output_dir,
                )


if __name__ == "__main__":
    unittest.main()
