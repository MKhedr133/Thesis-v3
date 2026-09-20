"""Synthetic tests for MOD-08 fixed-effects model comparison."""
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
    "mod08_test",
    PYDIR / "MOD_08_fixed_effects_model_comparison.py",
)
if SPEC is None or SPEC.loader is None:
    raise ImportError(PYDIR / "MOD_08_fixed_effects_model_comparison.py")
MOD08 = module_from_spec(SPEC)
sys.modules[SPEC.name] = MOD08
SPEC.loader.exec_module(MOD08)


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
                                0.0 if stage == 0 else float(stage + number % 3)
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


def model_aic_from_formula(formula: str) -> float:
    age = "C(participant_group" in formula
    tmt = "tmt_b_seconds" in formula
    age_interaction = ":C(participant_group" in formula
    tmt_interaction = ":tmt_b_seconds" in formula
    if age_interaction and tmt_interaction:
        return 92.0
    if age_interaction:
        return 94.0
    if tmt_interaction:
        return 95.0
    if age and tmt:
        return 96.0
    if age:
        return 98.0
    if tmt:
        return 97.0
    return 100.0


class FakeResult:
    def __init__(self, formula: str):
        aic = model_aic_from_formula(formula)
        self.converged = True
        self.llf = -aic / 2
        self.aic = aic
        self.bic = aic + 5.0
        self.fe_params = pd.Series(
            [1.0, 0.5],
            index=["Intercept", "difficulty"],
            dtype=float,
        )
        self.params = pd.Series([1.0, 0.5, 1.0], dtype=float)
        self.df_modelwc = 2


def fake_fit(
    formula,
    data,
    structure,
    difficulty_column,
    participant_column="participant_id",
    **kwargs,
):
    return FakeResult(formula), "lbfgs", (), ()


def fake_lopo_summary() -> pd.DataFrame:
    rows = []
    mae_by_model = {
        "M0": 2.00,
        "MA": 1.90,
        "MT": 1.80,
        "MAT": 1.75,
        "MDA": 1.70,
        "MDT": 1.65,
        "MDAT": 1.60,
    }
    r2_by_model = {
        "M0": 0.10,
        "MA": 0.12,
        "MT": 0.14,
        "MAT": 0.16,
        "MDA": 0.18,
        "MDT": 0.20,
        "MDAT": 0.22,
    }
    for condition in MOD08.MOD02.CONDITIONS:
        structure = MOD08.CONDITION_RANDOM_STRUCTURES[condition]
        for target in MOD08.TARGETS:
            prediction_count = (
                96
                if target == "performance_change_from_d0_percentage_points"
                else 128
            )
            for model_id in MOD08.MODEL_ORDER:
                mae = mae_by_model[model_id]
                rows.append(
                    {
                        "condition_name": condition,
                        "target_name": target,
                        "random_structure": structure,
                        "model_id": model_id,
                        "participant_count": 32,
                        "prediction_count": prediction_count,
                        "participant_balanced_mae": mae,
                        "participant_balanced_rmse": mae + 0.2,
                        "lopo_r2": r2_by_model[model_id],
                        "delta_mae_vs_m0": 2.0 - mae,
                    }
                )
    return pd.DataFrame(rows)


class Mod08Tests(unittest.TestCase):
    def setUp(self):
        self.data = synthetic_data()

    def test_frozen_condition_random_structures(self):
        self.assertEqual(
            MOD08.CONDITION_RANDOM_STRUCTURES,
            {
                "Visual": "RI",
                "Auditory": "RI",
                "Cognitive": "RI_RS",
            },
        )
        MOD08.validate_method_contract()

    def test_candidate_registry_is_exact_seven_models(self):
        self.assertEqual(
            tuple(MOD08.MODEL_RHS),
            ("M0", "MA", "MT", "MAT", "MDA", "MDT", "MDAT"),
        )
        for rhs in MOD08.MODEL_RHS.values():
            self.assertNotIn("tmt_z", rhs)
        self.assertIn("tmt_b_seconds", MOD08.MODEL_RHS["MT"])

    def test_requires_all_32_participants(self):
        ids = MOD08.validate_all_participants(self.data)
        self.assertEqual(len(ids), 32)

        reduced = self.data.loc[
            ~self.data["participant_id"].eq(ids[0])
        ].copy()
        with self.assertRaisesRegex(MOD08.Mod08Error, "all 32"):
            MOD08.validate_all_participants(reduced)

    def test_full_data_fit_uses_210_rows_and_frozen_structures(self):
        with patch.object(
            MOD08.MOD02,
            "fit_mixedlm_with_fallback",
            side_effect=fake_fit,
        ):
            fits, audit = MOD08.fit_full_data_candidates(self.data)

        self.assertEqual(len(fits), 3 * 10 * 7)
        self.assertEqual(len(audit), 30)
        self.assertTrue(audit["status"].eq("pass").all())
        self.assertTrue(
            fits.loc[
                fits["condition_name"].eq("Visual"),
                "random_structure",
            ].eq("RI").all()
        )
        self.assertTrue(
            fits.loc[
                fits["condition_name"].eq("Auditory"),
                "random_structure",
            ].eq("RI").all()
        )
        self.assertTrue(
            fits.loc[
                fits["condition_name"].eq("Cognitive"),
                "random_structure",
            ].eq("RI_RS").all()
        )

    def test_all_candidates_share_identical_rows_within_family(self):
        with patch.object(
            MOD08.MOD02,
            "fit_mixedlm_with_fallback",
            side_effect=fake_fit,
        ):
            fits, _ = MOD08.fit_full_data_candidates(self.data)

        counts = fits.groupby(
            ["condition_name", "target_name"]
        )["included_row_signature"].nunique()
        self.assertTrue(counts.eq(1).all())

    def test_full_data_deltas_are_candidate_minus_m0(self):
        with patch.object(
            MOD08.MOD02,
            "fit_mixedlm_with_fallback",
            side_effect=fake_fit,
        ):
            fits, _ = MOD08.fit_full_data_candidates(self.data)

        family = fits.loc[
            fits["condition_name"].eq("Visual")
            & fits["target_name"].eq("mental_demand_score_0_to_10")
        ]
        m0 = family.loc[family["model_id"].eq("M0")].iloc[0]
        mdat = family.loc[family["model_id"].eq("MDAT")].iloc[0]
        self.assertEqual(m0["delta_aic_vs_m0"], 0.0)
        self.assertEqual(mdat["delta_aic_vs_m0"], -8.0)
        self.assertEqual(mdat["delta_bic_vs_m0"], -8.0)

    def test_combined_table_has_predictive_deltas_and_percent_improvement(self):
        with patch.object(
            MOD08.MOD02,
            "fit_mixedlm_with_fallback",
            side_effect=fake_fit,
        ):
            full_fit, _ = MOD08.fit_full_data_candidates(self.data)
        combined = MOD08.combine_full_fit_and_lopo(
            full_fit,
            fake_lopo_summary(),
        )

        self.assertEqual(len(combined), 210)
        row = combined.loc[
            combined["condition_name"].eq("Visual")
            & combined["target_name"].eq("mental_demand_score_0_to_10")
            & combined["model_id"].eq("MDAT")
        ].iloc[0]
        self.assertAlmostEqual(row["delta_mae_vs_m0"], 0.4)
        self.assertAlmostEqual(
            row["mae_improvement_percent_vs_m0"],
            20.0,
        )
        self.assertAlmostEqual(row["delta_lopo_r2_vs_m0"], 0.12)
        self.assertEqual(
            row["selection_status"],
            "evidence_only_no_automatic_selection",
        )

    def test_nested_comparisons_use_prespecified_reference_models(self):
        with patch.object(
            MOD08.MOD02,
            "fit_mixedlm_with_fallback",
            side_effect=fake_fit,
        ):
            full_fit, _ = MOD08.fit_full_data_candidates(self.data)
        comparison = MOD08.combine_full_fit_and_lopo(
            full_fit,
            fake_lopo_summary(),
        )
        nested = MOD08.build_nested_comparisons(comparison)

        self.assertEqual(len(nested), 30 * len(MOD08.NESTED_COMPARISONS))
        mda = nested.loc[
            nested["condition_name"].eq("Visual")
            & nested["target_name"].eq("mental_demand_score_0_to_10")
            & nested["candidate_model"].eq("MDA")
            & nested["reference_model"].eq("MA")
        ].iloc[0]
        self.assertLess(
            mda["delta_mae_candidate_minus_reference"],
            0,
        )
        self.assertGreater(
            mda["delta_r2_candidate_minus_reference"],
            0,
        )
        self.assertLess(
            mda["delta_aic_candidate_minus_reference"],
            0,
        )

    def test_manifest_records_no_automatic_selection(self):
        ids = MOD08.validate_all_participants(self.data)
        manifest = MOD08.build_manifest(
            modeling_data_path=Path("model.csv"),
            participant_ids=ids,
            jobs=4,
            checkpoint_dir=Path("checkpoints"),
        )
        self.assertEqual(manifest["participant_count"], 32)
        self.assertEqual(manifest["lopo_fold_count"], 32)
        self.assertEqual(manifest["candidate_model_count"], 7)
        self.assertEqual(
            manifest["condition_random_structures"]["Cognitive"],
            "RI_RS",
        )
        self.assertFalse(manifest["p_values_used_for_model_selection"])
        self.assertFalse(manifest["automatic_fixed_effects_selection"])

    def test_output_writer_refuses_overwrite(self):
        blank = pd.DataFrame({"x": [1]})
        result = MOD08.Mod08Result(
            comparison=blank,
            nested_comparisons=blank,
            audit=pd.DataFrame({"status": ["pass"]}),
            predictions=blank,
            participant_errors=blank,
            lopo_checks=blank,
            errors=(),
            warnings=(),
            participant_ids=tuple(f"P{i:02d}" for i in range(1, 33)),
        )
        with tempfile.TemporaryDirectory() as directory:
            output_dir = Path(directory)
            paths = MOD08.write_outputs(
                result=result,
                manifest={"method_id": MOD08.METHOD_ID},
                output_dir=output_dir,
            )
            self.assertTrue(all(path.exists() for path in paths.values()))
            with self.assertRaisesRegex(
                FileExistsError,
                "Refusing to overwrite",
            ):
                MOD08.write_outputs(
                    result=result,
                    manifest={"method_id": MOD08.METHOD_ID},
                    output_dir=output_dir,
                )


if __name__ == "__main__":
    unittest.main()
