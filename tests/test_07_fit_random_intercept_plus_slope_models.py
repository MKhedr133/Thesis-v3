"""Synthetic MOD-02C random-intercept-plus-slope fitting tests."""

from __future__ import annotations

from dataclasses import FrozenInstanceError
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import sys
import unittest
from unittest.mock import patch
import warnings

import pandas as pd


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
REGISTRY_PATH = REPOSITORY_ROOT / "python" / "05_compare_random_effects.py"
MODULE_PATH = (
    REPOSITORY_ROOT
    / "python"
    / "07_fit_random_intercept_plus_slope_models.py"
)


def load_module(path: Path, name: str):
    if not path.exists():
        return None
    specification = spec_from_file_location(name, path)
    if specification is None or specification.loader is None:
        return None
    module = module_from_spec(specification)
    sys.modules[name] = module
    specification.loader.exec_module(module)
    return module


REGISTRY_MODULE = load_module(
    REGISTRY_PATH,
    "mod02a_registry_for_mod02c_tests",
)
MODULE = load_module(
    MODULE_PATH,
    "mod02c_fit_random_intercept_plus_slope_models",
)

PERFORMANCE_TARGET = "performance_change_from_d0_percentage_points"
MENTAL_DEMAND_TARGET = "mental_demand_score_0_to_10"
CONDITIONS = ("Visual", "Auditory", "Cognitive")

FORMULAS = {
    PERFORMANCE_TARGET: (
        "performance_change_from_d0_percentage_points ~ performance_D"
    ),
    MENTAL_DEMAND_TARGET: (
        "mental_demand_score_0_to_10 ~ difficulty_stage"
    ),
}

DIFFICULTY_COLUMNS = {
    PERFORMANCE_TARGET: "performance_D",
    MENTAL_DEMAND_TARGET: "difficulty_stage",
}


def make_registry():
    return REGISTRY_MODULE.build_random_effects_registry(
        "D * Age + D * TMT_z"
    )


def make_single_registry(target_name: str, condition_name: str = "Visual"):
    registry = make_registry()
    comparison = next(
        item
        for item in registry.comparisons
        if item.target_name == target_name
        and item.condition_name == condition_name
    )
    return registry.__class__(
        conditions=(condition_name,),
        targets=(target_name,),
        difficulty_stage_mapping=registry.difficulty_stage_mapping,
        comparison_fields=registry.comparison_fields,
        comparisons=(comparison,),
    )


def make_modeling_data() -> pd.DataFrame:
    rows = []
    difficulties = ((0, 0), (2, 1), (6, 2), (10, 3))
    for participant_index in range(1, 5):
        participant_id = f"P{participant_index:02d}"
        for condition_index, condition_name in enumerate(CONDITIONS):
            for difficulty_level, difficulty_stage in difficulties:
                rows.append(
                    {
                        "participant_id": participant_id,
                        "condition_name": condition_name,
                        "difficulty_level": difficulty_level,
                        "difficulty_stage": difficulty_stage,
                        "performance_D": float(difficulty_stage),
                        PERFORMANCE_TARGET: float(
                            difficulty_stage + participant_index / 10
                        ),
                        MENTAL_DEMAND_TARGET: float(
                            difficulty_stage
                            + condition_index / 10
                            + participant_index / 100
                        ),
                    }
                )
    return pd.DataFrame(rows)


class FakeFitResult:
    def __init__(
        self,
        *,
        converged: bool | None = True,
        log_likelihood: float = -10.0,
        warning_text: str | None = None,
        boundary_flag: bool | None = None,
        singularity_flag: bool | None = None,
    ):
        self.converged = converged
        self.llf = log_likelihood
        self.aic = 25.0
        self.bic = 30.0
        self.fe_params = pd.Series(
            {"Intercept": 1.0, "difficulty": 0.5}
        )
        self.params = pd.Series(
            {
                "Intercept": 1.0,
                "difficulty": 0.5,
                "Group Var": 0.7,
                "Group Cov": 0.2,
                "difficulty Var": 0.5,
            }
        )
        self.cov_re = pd.DataFrame(
            [[0.7, 0.2], [0.2, 0.5]],
            index=["Group", "difficulty"],
            columns=["Group", "difficulty"],
        )
        self.warning_text = warning_text
        if boundary_flag is not None:
            self.boundary_flag = boundary_flag
        if singularity_flag is not None:
            self.singularity_flag = singularity_flag


class FakeModel:
    def __init__(self, attempts):
        self.attempts = list(attempts)
        self.fit_calls = []

    def fit(self, *, method, reml, maxiter, disp):
        self.fit_calls.append(
            {
                "method": method,
                "reml": reml,
                "maxiter": maxiter,
                "disp": disp,
            }
        )
        action = self.attempts.pop(0)
        if isinstance(action, BaseException):
            raise action
        if action.warning_text:
            warnings.warn(action.warning_text, UserWarning)
        return action


class RandomInterceptPlusSlopeFittingTests(unittest.TestCase):
    def require_module(self):
        self.assertIsNotNone(
            REGISTRY_MODULE,
            "MOD-02A registry module must be importable",
        )
        self.assertIsNotNone(
            MODULE,
            "python/07_fit_random_intercept_plus_slope_models.py must exist",
        )
        return MODULE

    def test_six_primary_rs_records_use_target_specific_random_slopes(self):
        module = self.require_module()
        data = make_modeling_data()
        calls = []

        def mixedlm_side_effect(**kwargs):
            calls.append(kwargs)
            return FakeModel([FakeFitResult()])

        with patch.object(
            module.smf,
            "mixedlm",
            side_effect=mixedlm_side_effect,
        ):
            result = module.fit_random_intercept_plus_slope_models(
                data,
                make_registry(),
                FORMULAS,
                DIFFICULTY_COLUMNS,
            )

        self.assertTrue(result.valid)
        self.assertEqual(len(result.fit_records), 6)
        self.assertEqual(
            {
                (record.condition_name, record.target_name)
                for record in result.fit_records
            },
            {
                (condition, target)
                for condition in CONDITIONS
                for target in FORMULAS
            },
        )
        self.assertTrue(
            all(record.model_id == "RI_RS" for record in result.fit_records)
        )
        self.assertTrue(
            all(record.random_structure == "RI_RS" for record in result.fit_records)
        )
        self.assertTrue(
            all(
                record.random_formula_label == "(1 + D | participant)"
                for record in result.fit_records
            )
        )
        self.assertEqual(len(calls), 6)
        self.assertEqual(
            {call["re_formula"] for call in calls},
            {"1 + difficulty_stage", "1 + performance_D"},
        )
        self.assertTrue(
            all(
                call["groups"].reset_index(drop=True).equals(
                    call["data"]["participant_id"].reset_index(drop=True)
                )
                for call in calls
            )
        )

    def test_missing_performance_mapping_fails_without_stage_substitution(self):
        module = self.require_module()
        data = make_modeling_data()

        with patch.object(
            module.smf,
            "mixedlm",
            side_effect=lambda **_kwargs: FakeModel([FakeFitResult()]),
        ) as mixedlm:
            result = module.fit_random_intercept_plus_slope_models(
                data,
                make_registry(),
                {MENTAL_DEMAND_TARGET: FORMULAS[MENTAL_DEMAND_TARGET]},
                {MENTAL_DEMAND_TARGET: "difficulty_stage"},
            )

        performance_records = [
            record
            for record in result.fit_records
            if record.target_name == PERFORMANCE_TARGET
        ]
        self.assertFalse(result.valid)
        self.assertEqual(len(performance_records), 3)
        self.assertTrue(
            all(record.fit_status == "failed" for record in performance_records)
        )
        self.assertTrue(any("difficulty" in error.lower() for error in result.errors))
        self.assertEqual(mixedlm.call_count, 3)

    def test_mental_demand_rejects_non_stage_random_slope_mapping(self):
        module = self.require_module()
        data = make_modeling_data()
        difficulty_columns = {
            PERFORMANCE_TARGET: "performance_D",
            MENTAL_DEMAND_TARGET: "performance_D",
        }

        with patch.object(
            module.smf,
            "mixedlm",
            side_effect=lambda **_kwargs: FakeModel([FakeFitResult()]),
        ):
            result = module.fit_random_intercept_plus_slope_models(
                data,
                make_registry(),
                FORMULAS,
                difficulty_columns,
            )

        mental_records = [
            record
            for record in result.fit_records
            if record.target_name == MENTAL_DEMAND_TARGET
        ]
        self.assertTrue(
            all(record.fit_status == "failed" for record in mental_records)
        )
        self.assertTrue(
            any("difficulty_stage" in error for error in result.errors)
        )

    def test_performance_d0_is_excluded_and_signatures_are_exact(self):
        module = self.require_module()
        data = make_modeling_data()

        with patch.object(
            module.smf,
            "mixedlm",
            side_effect=lambda **_kwargs: FakeModel([FakeFitResult()]),
        ):
            result = module.fit_random_intercept_plus_slope_models(
                data,
                make_registry(),
                FORMULAS,
                DIFFICULTY_COLUMNS,
            )

        visual_records = {
            record.target_name: record
            for record in result.fit_records
            if record.condition_name == "Visual"
        }
        self.assertEqual(
            visual_records[PERFORMANCE_TARGET].observation_count,
            12,
        )
        self.assertEqual(
            visual_records[PERFORMANCE_TARGET].included_row_signature,
            "1,2,3,13,14,15,25,26,27,37,38,39",
        )
        self.assertEqual(
            visual_records[MENTAL_DEMAND_TARGET].observation_count,
            16,
        )
        self.assertEqual(
            visual_records[MENTAL_DEMAND_TARGET].included_row_signature,
            "0,1,2,3,12,13,14,15,24,25,26,27,36,37,38,39",
        )

    def test_metrics_convergence_and_common_schema_match_mod02b(self):
        module = self.require_module()
        data = make_modeling_data()
        fake_model = FakeModel([FakeFitResult()])

        with patch.object(module.smf, "mixedlm", return_value=fake_model):
            result = module.fit_random_intercept_plus_slope_models(
                data,
                make_single_registry(MENTAL_DEMAND_TARGET),
                {MENTAL_DEMAND_TARGET: FORMULAS[MENTAL_DEMAND_TARGET]},
                {MENTAL_DEMAND_TARGET: "difficulty_stage"},
            )

        record = result.fit_records[0]
        self.assertIs(False, record.reml)
        self.assertEqual(record.estimation_method, "maximum_likelihood")
        self.assertEqual(record.fixed_effect_count, 2)
        self.assertEqual(record.total_parameter_count, 5)
        self.assertEqual(record.random_intercept_variance, 0.7)
        self.assertEqual(record.random_slope_variance, 0.5)
        self.assertEqual(record.intercept_slope_covariance, 0.2)
        self.assertIsNone(record.delta_aic)
        self.assertIsNone(record.delta_bic)
        self.assertIsNone(record.likelihood_ratio_statistic)
        self.assertIsNone(record.likelihood_ratio_p_value)
        self.assertNotIn("executable_random_formula", result.fit_table.columns)
        self.assertIn("included_row_signature", result.fit_table.columns)

    def test_lbfgs_error_falls_back_and_preserves_warnings(self):
        module = self.require_module()
        data = make_modeling_data()
        fake_model = FakeModel(
            [
                RuntimeError("lbfgs failure"),
                FakeFitResult(warning_text="powell warning"),
            ]
        )

        with patch.object(module.smf, "mixedlm", return_value=fake_model):
            result = module.fit_random_intercept_plus_slope_models(
                data,
                make_single_registry(MENTAL_DEMAND_TARGET),
                {MENTAL_DEMAND_TARGET: FORMULAS[MENTAL_DEMAND_TARGET]},
                {MENTAL_DEMAND_TARGET: "difficulty_stage"},
            )

        self.assertTrue(result.valid)
        self.assertEqual(
            [call["method"] for call in fake_model.fit_calls],
            ["lbfgs", "powell"],
        )
        self.assertIn("lbfgs failure", result.fit_records[0].fit_error)
        self.assertIn("powell warning", result.fit_records[0].warnings)

    def test_nonconvergence_fallback_and_failure_status_are_preserved(self):
        module = self.require_module()
        data = make_modeling_data()
        fake_model = FakeModel(
            [
                FakeFitResult(converged=False, warning_text="lbfgs warning"),
                FakeFitResult(converged=False, warning_text="powell warning"),
            ]
        )

        with patch.object(module.smf, "mixedlm", return_value=fake_model):
            result = module.fit_random_intercept_plus_slope_models(
                data,
                make_single_registry(MENTAL_DEMAND_TARGET),
                {MENTAL_DEMAND_TARGET: FORMULAS[MENTAL_DEMAND_TARGET]},
                {MENTAL_DEMAND_TARGET: "difficulty_stage"},
            )

        self.assertFalse(result.valid)
        self.assertEqual(result.fit_records[0].fit_status, "non_converged")
        self.assertFalse(result.fit_records[0].convergence_status)
        self.assertIn("lbfgs warning", result.fit_records[0].warnings)
        self.assertIn("powell warning", result.fit_records[0].warnings)

    def test_unknown_convergence_preserves_mod02b_semantics(self):
        module = self.require_module()
        data = make_modeling_data()
        fake_model = FakeModel(
            [
                FakeFitResult(converged=False),
                FakeFitResult(converged=None),
            ]
        )

        with patch.object(module.smf, "mixedlm", return_value=fake_model):
            result = module.fit_random_intercept_plus_slope_models(
                data,
                make_single_registry(MENTAL_DEMAND_TARGET),
                {MENTAL_DEMAND_TARGET: FORMULAS[MENTAL_DEMAND_TARGET]},
                {MENTAL_DEMAND_TARGET: "difficulty_stage"},
            )

        self.assertTrue(result.valid)
        self.assertIsNone(result.fit_records[0].convergence_status)
        self.assertEqual(result.fit_records[0].fit_status, "fitted")

    def test_backend_flags_are_only_retained_when_explicit(self):
        module = self.require_module()
        data = make_modeling_data()
        fake_model = FakeModel(
            [FakeFitResult(boundary_flag=True, singularity_flag=False)]
        )

        with patch.object(module.smf, "mixedlm", return_value=fake_model):
            result = module.fit_random_intercept_plus_slope_models(
                data,
                make_single_registry(MENTAL_DEMAND_TARGET),
                {MENTAL_DEMAND_TARGET: FORMULAS[MENTAL_DEMAND_TARGET]},
                {MENTAL_DEMAND_TARGET: "difficulty_stage"},
            )

        self.assertTrue(result.fit_records[0].boundary_flag)
        self.assertFalse(result.fit_records[0].singularity_flag)

    def test_records_are_frozen_and_returned_table_is_independent(self):
        module = self.require_module()
        data = make_modeling_data()
        original = data.copy(deep=True)
        fake_model = FakeModel([FakeFitResult()])

        with patch.object(module.smf, "mixedlm", return_value=fake_model):
            result = module.fit_random_intercept_plus_slope_models(
                data,
                make_single_registry(MENTAL_DEMAND_TARGET),
                {MENTAL_DEMAND_TARGET: FORMULAS[MENTAL_DEMAND_TARGET]},
                {MENTAL_DEMAND_TARGET: "difficulty_stage"},
            )

        self.assertTrue(
            module.RandomEffectsFitRecord.__dataclass_params__.frozen
        )
        self.assertTrue(
            module.RandomEffectsFitResult.__dataclass_params__.frozen
        )
        with self.assertRaises(FrozenInstanceError):
            result.valid = False

        result.fit_table.loc[0, "difficulty_stage"] = "changed"
        self.assertTrue(data.equals(original))


if __name__ == "__main__":
    unittest.main()
