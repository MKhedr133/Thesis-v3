"""Synthetic MOD-02B random-intercept fitting contract tests."""

from __future__ import annotations

from dataclasses import FrozenInstanceError, replace
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from types import SimpleNamespace
import sys
import unittest
from unittest.mock import patch
import warnings

import pandas as pd


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
REGISTRY_PATH = REPOSITORY_ROOT / "python" / "05_compare_random_effects.py"
MODULE_PATH = REPOSITORY_ROOT / "python" / "06_fit_random_intercept_models.py"


def load_module(path: Path, name: str):
    if not path.exists():
        return None
    spec = spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        return None
    module = module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


REGISTRY_MODULE = load_module(REGISTRY_PATH, "mod02a_registry_for_mod02b_tests")
MODULE = load_module(MODULE_PATH, "mod02b_fit_random_intercepts")

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
    return replace(
        registry,
        conditions=(condition_name,),
        targets=(target_name,),
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
        converged: bool = True,
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
            {"Intercept": 1.0, "difficulty": 0.5, "Group Var": 0.7}
        )
        self.cov_re = pd.DataFrame([[0.7]], index=["Group"], columns=["Group"])
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


class RandomInterceptFittingTests(unittest.TestCase):
    def require_module(self):
        self.assertIsNotNone(
            REGISTRY_MODULE,
            "MOD-02A registry module must be importable",
        )
        self.assertIsNotNone(
            MODULE,
            "python/06_fit_random_intercept_models.py must be implemented",
        )
        return MODULE

    def test_six_primary_condition_target_ri_records_are_returned(self):
        module = self.require_module()
        data = make_modeling_data()

        with patch.object(
            module.smf,
            "mixedlm",
            side_effect=lambda **_kwargs: FakeModel([FakeFitResult()]),
        ):
            result = module.fit_random_intercept_models(
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
            {(condition, target) for condition in CONDITIONS for target in FORMULAS},
        )
        self.assertTrue(
            all(record.model_id == "RI" for record in result.fit_records)
        )
        self.assertTrue(
            all(record.random_structure == "RI" for record in result.fit_records)
        )

    def test_mental_demand_uses_stage_and_performance_uses_explicit_column(self):
        module = self.require_module()
        data = make_modeling_data()

        with patch.object(
            module.smf,
            "mixedlm",
            side_effect=lambda **_kwargs: FakeModel([FakeFitResult()]),
        ):
            result = module.fit_random_intercept_models(
                data,
                make_registry(),
                FORMULAS,
                DIFFICULTY_COLUMNS,
            )

        mental = next(
            record
            for record in result.fit_records
            if record.target_name == MENTAL_DEMAND_TARGET
        )
        performance = next(
            record
            for record in result.fit_records
            if record.target_name == PERFORMANCE_TARGET
        )
        self.assertEqual(mental.difficulty_source_column, "difficulty_stage")
        self.assertEqual(performance.difficulty_source_column, "performance_D")
        self.assertEqual(performance.difficulty_coding_status, "unapproved")

    def test_missing_performance_mapping_fails_without_stage_substitution(self):
        module = self.require_module()
        data = make_modeling_data()
        formulas = {MENTAL_DEMAND_TARGET: FORMULAS[MENTAL_DEMAND_TARGET]}
        difficulty_columns = {
            MENTAL_DEMAND_TARGET: "difficulty_stage",
        }
        with patch.object(
            module.smf,
            "mixedlm",
            side_effect=lambda **_kwargs: FakeModel([FakeFitResult()]),
        ):
            result = module.fit_random_intercept_models(
                data,
                make_registry(),
                formulas,
                difficulty_columns,
            )

        performance_records = [
            record
            for record in result.fit_records
            if record.target_name == PERFORMANCE_TARGET
        ]
        self.assertEqual(len(performance_records), 3)
        self.assertTrue(all(record.fit_status == "failed" for record in performance_records))
        self.assertTrue(
            any("difficulty" in error.lower() for error in result.errors)
        )

    def test_performance_excludes_d0_and_mental_demand_retains_d0(self):
        module = self.require_module()
        data = make_modeling_data()

        with patch.object(
            module.smf,
            "mixedlm",
            side_effect=lambda **_kwargs: FakeModel([FakeFitResult()]),
        ):
            result = module.fit_random_intercept_models(
                data,
                make_registry(),
                FORMULAS,
                DIFFICULTY_COLUMNS,
            )

        expected_signatures = {
            "Visual": {
                PERFORMANCE_TARGET: "1,2,3,13,14,15,25,26,27,37,38,39",
                MENTAL_DEMAND_TARGET: "0,1,2,3,12,13,14,15,24,25,26,27,36,37,38,39",
            },
            "Auditory": {
                PERFORMANCE_TARGET: "5,6,7,17,18,19,29,30,31,41,42,43",
                MENTAL_DEMAND_TARGET: "4,5,6,7,16,17,18,19,28,29,30,31,40,41,42,43",
            },
            "Cognitive": {
                PERFORMANCE_TARGET: "9,10,11,21,22,23,33,34,35,45,46,47",
                MENTAL_DEMAND_TARGET: "8,9,10,11,20,21,22,23,32,33,34,35,44,45,46,47",
            },
        }

        for record in result.fit_records:
            if record.target_name == PERFORMANCE_TARGET:
                self.assertEqual(record.observation_count, 12)
                self.assertEqual(record.excluded_observation_count, 4)
            else:
                self.assertEqual(record.observation_count, 16)
                self.assertEqual(record.excluded_observation_count, 0)
            self.assertEqual(
                record.included_row_signature,
                expected_signatures[record.condition_name][record.target_name],
            )

    def test_missing_values_are_reported_and_inputs_are_unchanged(self):
        module = self.require_module()
        data = make_modeling_data()
        data.loc[0, "difficulty_stage"] = pd.NA
        data.loc[1, MENTAL_DEMAND_TARGET] = pd.NA
        original = data.copy(deep=True)

        with patch.object(
            module.smf,
            "mixedlm",
            side_effect=lambda **_kwargs: FakeModel([FakeFitResult()]),
        ):
            result = module.fit_random_intercept_models(
                data,
                make_registry(),
                FORMULAS,
                DIFFICULTY_COLUMNS,
            )

        self.assertTrue(data.equals(original))
        self.assertTrue(
            any(record.excluded_observation_count > 0 for record in result.fit_records)
        )
        self.assertTrue(result.warnings)

    def test_formula_and_ml_metadata_are_preserved(self):
        module = self.require_module()
        data = make_modeling_data()

        with patch.object(
            module.smf,
            "mixedlm",
            side_effect=lambda **_kwargs: FakeModel([FakeFitResult()]),
        ):
            result = module.fit_random_intercept_models(
                data,
                make_registry(),
                FORMULAS,
                DIFFICULTY_COLUMNS,
            )

        self.assertTrue(all(record.reml is False for record in result.fit_records))
        self.assertTrue(
            all(
                record.executable_formula == FORMULAS[record.target_name]
                for record in result.fit_records
            )
        )
        self.assertTrue(
            all(
                record.random_formula_label == "(1 | participant)"
                for record in result.fit_records
            )
        )
        self.assertTrue(
            all(record.delta_aic is None for record in result.fit_records)
        )
        self.assertTrue(
            all(record.delta_bic is None for record in result.fit_records)
        )
        self.assertIn("included_row_signature", result.fit_table.columns)
        self.assertEqual(
            tuple(result.fit_table.columns[: len(make_registry().comparison_fields)]),
            make_registry().comparison_fields,
        )

    def test_lbfgs_success_does_not_attempt_powell(self):
        module = self.require_module()
        data = make_modeling_data()
        fake_model = FakeModel([FakeFitResult()])

        with patch.object(module.smf, "mixedlm", return_value=fake_model):
            result = module.fit_random_intercept_models(
                data,
                make_single_registry(MENTAL_DEMAND_TARGET),
                {MENTAL_DEMAND_TARGET: FORMULAS[MENTAL_DEMAND_TARGET]},
                {MENTAL_DEMAND_TARGET: "difficulty_stage"},
            )

        self.assertTrue(result.valid)
        self.assertEqual([call["method"] for call in fake_model.fit_calls], ["lbfgs"])

    def test_lbfgs_error_attempts_powell_and_preserves_warnings(self):
        module = self.require_module()
        data = make_modeling_data()
        fake_model = FakeModel(
            [
                RuntimeError("lbfgs failure"),
                FakeFitResult(warning_text="powell warning"),
            ]
        )

        with patch.object(module.smf, "mixedlm", return_value=fake_model):
            result = module.fit_random_intercept_models(
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
        self.assertIn("powell warning", result.fit_records[0].warnings)
        self.assertIn("lbfgs failure", result.fit_records[0].fit_error)

    def test_lbfgs_nonconvergence_attempts_powell(self):
        module = self.require_module()
        data = make_modeling_data()
        fake_model = FakeModel(
            [
                FakeFitResult(converged=False, warning_text="lbfgs warning"),
                FakeFitResult(converged=True, warning_text="powell warning"),
            ]
        )

        with patch.object(module.smf, "mixedlm", return_value=fake_model):
            result = module.fit_random_intercept_models(
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
        self.assertIn("lbfgs warning", result.fit_records[0].warnings)
        self.assertIn("powell warning", result.fit_records[0].warnings)

    def test_powell_nonconvergence_is_not_reported_as_success(self):
        module = self.require_module()
        data = make_modeling_data()
        fake_model = FakeModel(
            [
                FakeFitResult(converged=False),
                FakeFitResult(converged=False),
            ]
        )

        with patch.object(module.smf, "mixedlm", return_value=fake_model):
            result = module.fit_random_intercept_models(
                data,
                make_single_registry(MENTAL_DEMAND_TARGET),
                {MENTAL_DEMAND_TARGET: FORMULAS[MENTAL_DEMAND_TARGET]},
                {MENTAL_DEMAND_TARGET: "difficulty_stage"},
            )

        self.assertFalse(result.valid)
        self.assertEqual(result.fit_records[0].fit_status, "non_converged")
        self.assertFalse(result.fit_records[0].convergence_status)

    def test_warnings_from_all_attempts_are_preserved(self):
        module = self.require_module()
        data = make_modeling_data()
        fake_model = FakeModel(
            [
                FakeFitResult(converged=False, warning_text="first warning"),
                FakeFitResult(converged=True, warning_text="second warning"),
            ]
        )

        with patch.object(module.smf, "mixedlm", return_value=fake_model):
            result = module.fit_random_intercept_models(
                data,
                make_single_registry(MENTAL_DEMAND_TARGET),
                {MENTAL_DEMAND_TARGET: FORMULAS[MENTAL_DEMAND_TARGET]},
                {MENTAL_DEMAND_TARGET: "difficulty_stage"},
            )

        warnings_text = result.fit_records[0].warnings
        self.assertIn("first warning", warnings_text)
        self.assertIn("second warning", warnings_text)

    def test_fit_failure_is_recorded_and_other_fits_continue(self):
        module = self.require_module()
        data = make_modeling_data()
        calls = {"count": 0}

        def mixedlm_side_effect(**_kwargs):
            calls["count"] += 1
            if calls["count"] == 1:
                return FakeModel(
                    [
                        RuntimeError("lbfgs failure"),
                        RuntimeError("powell failure"),
                    ]
                )
            return FakeModel([FakeFitResult()])

        with patch.object(module.smf, "mixedlm", side_effect=mixedlm_side_effect):
            result = module.fit_random_intercept_models(
                data,
                make_registry(),
                FORMULAS,
                DIFFICULTY_COLUMNS,
            )

        self.assertFalse(result.valid)
        self.assertEqual(len(result.fit_records), 6)
        self.assertEqual(
            sum(record.fit_status == "failed" for record in result.fit_records),
            1,
        )
        self.assertEqual(
            sum(record.fit_status == "fitted" for record in result.fit_records),
            5,
        )

    def test_boundary_and_singularity_flags_remain_unset_without_diagnostics(self):
        module = self.require_module()
        data = make_modeling_data()
        fake_model = FakeModel([FakeFitResult(warning_text="boundary warning")])

        with patch.object(module.smf, "mixedlm", return_value=fake_model):
            result = module.fit_random_intercept_models(
                data,
                make_single_registry(MENTAL_DEMAND_TARGET),
                {MENTAL_DEMAND_TARGET: FORMULAS[MENTAL_DEMAND_TARGET]},
                {MENTAL_DEMAND_TARGET: "difficulty_stage"},
            )

        record = result.fit_records[0]
        self.assertIsNone(record.boundary_flag)
        self.assertIsNone(record.singularity_flag)

    def test_explicit_backend_diagnostics_are_retained(self):
        module = self.require_module()
        data = make_modeling_data()
        fake_model = FakeModel(
            [
                FakeFitResult(
                    boundary_flag=True,
                    singularity_flag=False,
                )
            ]
        )

        with patch.object(module.smf, "mixedlm", return_value=fake_model):
            result = module.fit_random_intercept_models(
                data,
                make_single_registry(MENTAL_DEMAND_TARGET),
                {MENTAL_DEMAND_TARGET: FORMULAS[MENTAL_DEMAND_TARGET]},
                {MENTAL_DEMAND_TARGET: "difficulty_stage"},
            )

        self.assertTrue(result.fit_records[0].boundary_flag)
        self.assertFalse(result.fit_records[0].singularity_flag)

    def test_result_and_records_are_frozen_and_table_is_independent(self):
        module = self.require_module()
        data = make_modeling_data()
        fake_model = FakeModel([FakeFitResult()])

        with patch.object(module.smf, "mixedlm", return_value=fake_model):
            result = module.fit_random_intercept_models(
                data,
                make_single_registry(MENTAL_DEMAND_TARGET),
                {MENTAL_DEMAND_TARGET: FORMULAS[MENTAL_DEMAND_TARGET]},
                {MENTAL_DEMAND_TARGET: "difficulty_stage"},
            )

        with self.assertRaises(FrozenInstanceError):
            result.valid = False
        with self.assertRaises(FrozenInstanceError):
            result.fit_records[0].model_id = "RI_RS"

        original_value = data.loc[0, "difficulty_stage"]
        result.fit_table.loc[0, "difficulty_stage"] = "changed"
        self.assertEqual(data.loc[0, "difficulty_stage"], original_value)


if __name__ == "__main__":
    unittest.main()
