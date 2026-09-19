"""Synthetic tests for MOD-07 all-participant LOPO random-effects comparison."""
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
    "mod07_lopo_test",
    PYDIR / "MOD_07_lopo_random_effects_comparison.py",
)
if SPEC is None or SPEC.loader is None:
    raise ImportError(PYDIR / "MOD_07_lopo_random_effects_comparison.py")
MOD07 = module_from_spec(SPEC)
sys.modules[SPEC.name] = MOD07
SPEC.loader.exec_module(MOD07)


def synthetic_data() -> pd.DataFrame:
    rows = []
    for group, prefix in (("Young", "Y"), ("Old", "O")):
        for number in range(1, 17):
            participant_id = f"{prefix}{number:02d}"
            tmt = float(35 + number + (10 if group == "Old" else 0))
            for condition_index, condition in enumerate(
                ("Visual", "Auditory", "Cognitive")
            ):
                for stage, level in enumerate((0, 2, 6, 10)):
                    rows.append(
                        {
                            "participant_id": participant_id,
                            "participant_group": group,
                            "tmt_b_seconds": tmt,
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
                            "list_recheck_count": float((stage + number) % 4),
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


def fake_lopo_result() -> MOD07.MOD05.LopoResult:
    summary_rows = []
    check_rows = []
    for condition, target in MOD07.comparison_pairs():
        prediction_count = (
            96
            if target == "performance_change_from_d0_percentage_points"
            else 128
        )
        for structure in MOD07.RANDOM_STRUCTURES:
            summary_rows.append(
                {
                    "condition_name": condition,
                    "target_name": target,
                    "random_structure": structure,
                    "model_id": "M0",
                    "participant_count": 32,
                    "prediction_count": prediction_count,
                    "participant_balanced_mae": (
                        2.0 if structure == "RI" else 1.5
                    ),
                    "participant_balanced_rmse": (
                        2.5 if structure == "RI" else 2.0
                    ),
                    "lopo_r2": 0.20 if structure == "RI" else 0.30,
                }
            )
            check_rows.append(
                {
                    "condition_name": condition,
                    "target_name": target,
                    "random_structure": structure,
                    "status": "pass",
                    "message": "LOPO completed with 0 failed model-fold fit(s).",
                }
            )

    return MOD07.MOD05.LopoResult(
        True,
        (),
        (),
        pd.DataFrame(
            columns=[
                "heldout_participant",
                "condition_name",
                "target_name",
                "model_id",
                "random_structure",
                "observed",
                "predicted",
            ]
        ),
        pd.DataFrame(
            columns=[
                "participant_id",
                "condition_name",
                "target_name",
                "model_id",
                "random_structure",
                "participant_mae",
            ]
        ),
        pd.DataFrame(summary_rows),
        pd.DataFrame(check_rows),
    )


class Mod07LopoTests(unittest.TestCase):
    def setUp(self):
        self.data = synthetic_data()

    def test_scope_is_all_30_gaussian_pairs(self):
        pairs = MOD07.comparison_pairs()
        self.assertEqual(len(pairs), 30)
        self.assertEqual(
            {condition for condition, _ in pairs},
            {"Visual", "Auditory", "Cognitive"},
        )
        self.assertEqual(
            {target for _, target in pairs},
            set(MOD07.MOD02.GAUSSIAN_TARGETS),
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

    def test_requires_all_32_participants(self):
        ids = MOD07.validate_all_participants(self.data)
        self.assertEqual(len(ids), 32)

        reduced = self.data.loc[
            ~self.data["participant_id"].eq(ids[0])
        ].copy()
        with self.assertRaisesRegex(MOD07.Mod07Error, "all 32"):
            MOD07.validate_all_participants(reduced)

    def test_pooled_lopo_r2_helper(self):
        self.assertAlmostEqual(
            MOD07.MOD05._pooled_r2([1, 2, 3], [1, 2, 3]),
            1.0,
        )
        self.assertAlmostEqual(
            MOD07.MOD05._pooled_r2([1, 2, 3], [2, 2, 2]),
            0.0,
        )
        self.assertTrue(
            np.isnan(MOD07.MOD05._pooled_r2([1, 1, 1], [1, 1, 1]))
        )
        self.assertLess(
            MOD07.MOD05._pooled_r2([1, 2, 3], [3, 3, 3]),
            0.0,
        )

    def test_full_data_fit_reports_signed_information_criterion_deltas(self):
        with patch.object(
            MOD07.MOD02,
            "fit_mixedlm_with_fallback",
            side_effect=fake_fit,
        ):
            full, audit = MOD07.fit_full_data_m0(
                self.data,
                pairs=(("Visual", "mental_demand_score_0_to_10"),),
            )

        self.assertEqual(len(full), 1)
        row = full.iloc[0]
        self.assertEqual(row["participant_count"], 32)
        self.assertEqual(row["delta_aic_ri_rs_minus_ri"], -5.0)
        self.assertEqual(row["delta_bic_ri_rs_minus_ri"], -2.0)
        self.assertAlmostEqual(row["ri_rs_random_slope_variance"], 0.2)
        self.assertEqual(audit.iloc[0]["status"], "pass")

    def test_run_mod07_combines_lopo_and_full_fit_deltas(self):
        fake_lopo = fake_lopo_result()
        with patch.object(
            MOD07.MOD05,
            "run_lopo_prediction",
            return_value=fake_lopo,
        ), patch.object(
            MOD07.MOD02,
            "fit_mixedlm_with_fallback",
            side_effect=fake_fit,
        ):
            comparison, audit, result, participant_ids = MOD07.run_mod07(
                self.data,
                jobs=1,
            )

        self.assertEqual(len(comparison), 30)
        self.assertEqual(len(audit), 30)
        self.assertEqual(len(participant_ids), 32)
        self.assertTrue(audit["status"].eq("pass").all())
        self.assertTrue(
            np.allclose(
                comparison["delta_lopo_mae_ri_rs_minus_ri"],
                -0.5,
            )
        )
        self.assertTrue(
            np.allclose(
                comparison["delta_lopo_r2_ri_rs_minus_ri"],
                0.10,
            )
        )
        self.assertTrue(
            np.allclose(
                comparison["delta_aic_ri_rs_minus_ri"],
                -5.0,
            )
        )
        self.assertTrue(
            np.allclose(
                comparison["delta_bic_ri_rs_minus_ri"],
                -2.0,
            )
        )
        self.assertTrue(
            comparison["selection_status"].eq(
                "evidence_only_no_automatic_selection"
            ).all()
        )
        self.assertIs(result, fake_lopo)

    def test_manifest_records_all32_lopo_and_no_bootstrap(self):
        ids = MOD07.validate_all_participants(self.data)
        manifest = MOD07.build_manifest(
            modeling_data_path=Path("model.csv"),
            participant_ids=ids,
            jobs=4,
            checkpoint_dir=Path("checkpoints"),
        )
        self.assertTrue(manifest["uses_all_32_participants"])
        self.assertEqual(manifest["lopo_fold_count"], 32)
        self.assertEqual(manifest["participants_fit_per_fold"], 31)
        self.assertFalse(manifest["bootstrap_used"])
        self.assertFalse(manifest["train_test_split_used"])
        self.assertFalse(manifest["automatic_random_structure_selection"])
        self.assertEqual(
            manifest["heldout_prediction_scope"],
            "fixed_effect_population_only",
        )

    def test_cli_has_no_bootstrap_or_split_arguments(self):
        parser = MOD07.build_cli_parser()
        destinations = {action.dest for action in parser._actions}
        self.assertNotIn("bootstrap_replicates", destinations)
        self.assertNotIn("bootstrap_seed", destinations)
        self.assertNotIn("split_assignments", destinations)
        self.assertNotIn("split_manifest", destinations)

    def test_output_writer_refuses_overwrite(self):
        fake_lopo = fake_lopo_result()
        comparison = pd.DataFrame({"x": [1]})
        audit = pd.DataFrame({"status": ["pass"]})
        manifest = {"method_id": MOD07.METHOD_ID}

        with tempfile.TemporaryDirectory() as directory:
            output_dir = Path(directory)
            paths = MOD07.write_outputs(
                comparison=comparison,
                audit=audit,
                lopo=fake_lopo,
                manifest=manifest,
                output_dir=output_dir,
            )
            self.assertTrue(all(path.exists() for path in paths.values()))

            with self.assertRaisesRegex(
                FileExistsError,
                "Refusing to overwrite",
            ):
                MOD07.write_outputs(
                    comparison=comparison,
                    audit=audit,
                    lopo=fake_lopo,
                    manifest=manifest,
                    output_dir=output_dir,
                )


if __name__ == "__main__":
    unittest.main()
