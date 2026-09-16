"""Synthetic MOD-01A modelling-input contract tests."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import unittest

import pandas as pd


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = REPOSITORY_ROOT / "python" / "03_validate_modeling_inputs.py"


def load_validator_module():
    """Load the numeric-prefix validator without requiring a package."""

    if not MODULE_PATH.exists():
        return None
    specification = importlib.util.spec_from_file_location(
        "validate_modeling_inputs",
        MODULE_PATH,
    )
    if specification is None or specification.loader is None:
        return None
    module = importlib.util.module_from_spec(specification)
    sys.modules[specification.name] = module
    specification.loader.exec_module(module)
    return module


VALIDATOR = load_validator_module()


BEHAVIOURAL_COLUMNS = (
    "median_time_between_qualifying_grabs_seconds",
    "list_recheck_count",
    "total_list_recheck_duration_seconds",
    "median_time_to_target_seconds",
    "median_irrelevant_focus_duration_seconds",
    "median_head_turning_degrees",
    "median_reach_duration_seconds",
    "median_reach_path_ratio",
)

ERROR_COMPONENT_COLUMNS = (
    "errors_missing",
    "errors_wrong_order",
    "errors_duplicate",
    "errors_not_in_list",
)


def make_trial_features() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    conditions = ("Visual", "Auditory", "Cognitive")
    difficulties = (0, 2, 6, 10)
    for participant_number, group in ((1, "Young"), (2, "Old")):
        participant_id = f"P{participant_number:02d}"
        for condition_index, condition in enumerate(conditions):
            for trial_index, difficulty in enumerate(difficulties, start=1):
                errors = (1, 2, 3, 4)
                row: dict[str, object] = {
                    "participant_id": participant_id,
                    "session_id": "S001",
                    "source_tracker_csv_filename": (
                        f"data_collector_vr_sample_{participant_id}_{condition}_{difficulty}.csv"
                    ),
                    "participant_group": group,
                    "condition_name": condition,
                    "difficulty_level": difficulty,
                    "trial_order": f"T{trial_index}",
                    "language": "EN",
                    "performance": 10.0,
                    "correct_products_collected_count": 10.0,
                    "performance_percent": 50.0,
                    "performance_change_from_d0_percentage_points": 0.0,
                    "mental_demand_score_0_to_10": float(
                        (condition_index + trial_index) % 11
                    ),
                    "error_change_from_d0": 0.0,
                    "total_error_count": float(sum(errors)),
                    "median_time_between_qualifying_grabs_seconds": 1.0,
                    "list_recheck_count": 0,
                    "total_list_recheck_duration_seconds": 0.0,
                    "median_time_to_target_seconds": 2.0,
                    "median_irrelevant_focus_duration_seconds": 0.5,
                    "median_head_turning_degrees": 15.0,
                    "median_reach_duration_seconds": 1.5,
                    "median_reach_path_ratio": 1.2,
                }
                row.update(dict(zip(ERROR_COMPONENT_COLUMNS, errors)))
                rows.append(row)
    return pd.DataFrame(rows)


def make_covariates() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "participant_id": ["P01", "P02"],
            "age_years": [25.0, 68.0],
            "tmt_b_seconds": [42.0, 91.0],
        }
    )


class ModelingInputValidationTests(unittest.TestCase):
    def require_validator(self):
        self.assertIsNotNone(
            VALIDATOR,
            "python/03_validate_modeling_inputs.py is not implemented",
        )
        return VALIDATOR

    def validate(self, trial_features=None, participant_covariates=None):
        validator = self.require_validator()
        return validator.validate_modeling_inputs(
            make_trial_features() if trial_features is None else trial_features,
            make_covariates()
            if participant_covariates is None
            else participant_covariates,
        )

    def test_valid_multi_condition_input_maps_ordered_stages(self):
        result = self.validate()

        self.assertTrue(result.valid)
        self.assertEqual(
            set(result.trial_features["condition_name"]),
            {"Visual", "Auditory", "Cognitive"},
        )
        mapping = dict(
            zip(
                result.trial_features["difficulty_level"],
                result.trial_features["difficulty_stage"],
            )
        )
        self.assertEqual(mapping, {0: 0, 2: 1, 6: 2, 10: 3})

    def test_result_containers_are_frozen_dataclasses(self):
        validator = self.require_validator()

        self.assertTrue(validator.ValidationCheck.__dataclass_params__.frozen)
        self.assertTrue(
            validator.ModelingInputValidationResult.__dataclass_params__.frozen
        )

    def test_missing_required_trial_column_fails(self):
        trial_features = make_trial_features().drop(columns=["performance"])

        result = self.validate(trial_features=trial_features)

        self.assertFalse(result.valid)
        self.assertTrue(any("missing" in error.lower() for error in result.errors))

    def test_missing_required_covariate_column_fails(self):
        covariates = make_covariates().drop(columns=["tmt_b_seconds"])

        result = self.validate(participant_covariates=covariates)

        self.assertFalse(result.valid)
        self.assertTrue(any("tmt_b_seconds" in error for error in result.errors))

    def test_invalid_condition_fails(self):
        trial_features = make_trial_features()
        trial_features.loc[0, "condition_name"] = "Motor"

        result = self.validate(trial_features=trial_features)

        self.assertFalse(result.valid)
        self.assertTrue(any("condition" in error.lower() for error in result.errors))

    def test_invalid_difficulty_fails(self):
        trial_features = make_trial_features()
        trial_features.loc[0, "difficulty_level"] = 4

        result = self.validate(trial_features=trial_features)

        self.assertFalse(result.valid)
        self.assertTrue(any("difficulty" in error.lower() for error in result.errors))

    def test_duplicate_trial_key_fails(self):
        trial_features = make_trial_features()
        trial_features = pd.concat(
            [trial_features, trial_features.iloc[[0]]],
            ignore_index=True,
        )

        result = self.validate(trial_features=trial_features)

        self.assertFalse(result.valid)
        self.assertTrue(any("duplicate" in error.lower() for error in result.errors))

    def test_d0_performance_change_may_be_missing_or_zero(self):
        trial_features = make_trial_features()
        trial_features.loc[0, "performance_change_from_d0_percentage_points"] = pd.NA

        result = self.validate(trial_features=trial_features)

        self.assertTrue(result.valid)

    def test_nonzero_d0_performance_change_fails(self):
        trial_features = make_trial_features()
        trial_features.loc[0, "performance_change_from_d0_percentage_points"] = 1.0

        result = self.validate(trial_features=trial_features)

        self.assertFalse(result.valid)
        self.assertTrue(any("D0" in error for error in result.errors))

    def test_mental_demand_boundaries_are_valid(self):
        trial_features = make_trial_features()
        trial_features.loc[0, "mental_demand_score_0_to_10"] = 0.0
        trial_features.loc[1, "mental_demand_score_0_to_10"] = 10.0

        result = self.validate(trial_features=trial_features)

        self.assertTrue(result.valid)

    def test_missing_mental_demand_is_warning(self):
        trial_features = make_trial_features()
        trial_features.loc[0, "mental_demand_score_0_to_10"] = pd.NA

        result = self.validate(trial_features=trial_features)

        self.assertTrue(result.valid)
        self.assertTrue(any("mental" in warning.lower() for warning in result.warnings))

    def test_invalid_mental_demand_values_fail(self):
        for invalid_value in (-0.1, 10.1, float("inf"), float("-inf")):
            with self.subTest(invalid_value=invalid_value):
                trial_features = make_trial_features()
                trial_features.loc[0, "mental_demand_score_0_to_10"] = invalid_value

                result = self.validate(trial_features=trial_features)

                self.assertFalse(result.valid)
                self.assertTrue(
                    any("mental" in error.lower() for error in result.errors)
                )

    def test_missing_behavioural_values_are_warnings(self):
        trial_features = make_trial_features()
        trial_features.loc[0, BEHAVIOURAL_COLUMNS[0]] = pd.NA

        result = self.validate(trial_features=trial_features)

        self.assertTrue(result.valid)
        self.assertTrue(any("behaviour" in warning.lower() for warning in result.warnings))

    def test_matching_total_error_sum_passes(self):
        result = self.validate()

        self.assertTrue(result.valid)
        self.assertTrue(any("total" in check.name for check in result.checks))

    def test_mismatching_total_error_sum_fails_without_repair(self):
        trial_features = make_trial_features()
        trial_features.loc[0, "total_error_count"] = 999.0
        original_total = trial_features.loc[0, "total_error_count"]

        result = self.validate(trial_features=trial_features)

        self.assertFalse(result.valid)
        self.assertEqual(result.trial_features.loc[0, "total_error_count"], original_total)
        self.assertEqual(trial_features.loc[0, "total_error_count"], original_total)
        self.assertTrue(any("mismatch" in error.lower() for error in result.errors))

    def test_incomplete_total_error_audit_is_modelling_readiness_failure(self):
        trial_features = make_trial_features()
        trial_features.loc[0, "errors_duplicate"] = pd.NA

        result = self.validate(trial_features=trial_features)

        self.assertFalse(result.valid)
        self.assertTrue(any("incomplete" in error.lower() for error in result.errors))
        self.assertTrue(pd.isna(result.trial_features.loc[0, "errors_duplicate"]))
        self.assertTrue(pd.isna(trial_features.loc[0, "errors_duplicate"]))

    def test_duplicate_participant_covariates_fail(self):
        covariates = pd.concat(
            [make_covariates(), make_covariates().iloc[[0]]],
            ignore_index=True,
        )

        result = self.validate(participant_covariates=covariates)

        self.assertFalse(result.valid)
        self.assertTrue(any("duplicate" in error.lower() for error in result.errors))

    def test_missing_participant_covariate_coverage_fails(self):
        covariates = make_covariates().iloc[[0]].copy()

        result = self.validate(participant_covariates=covariates)

        self.assertFalse(result.valid)
        self.assertTrue(any("coverage" in error.lower() for error in result.errors))

    def test_inconsistent_participant_group_fails(self):
        trial_features = make_trial_features()
        trial_features.loc[1, "participant_group"] = "Old"

        result = self.validate(trial_features=trial_features)

        self.assertFalse(result.valid)
        self.assertTrue(any("group" in error.lower() for error in result.errors))

    def test_inconsistent_age_and_tmt_values_fail(self):
        for column in ("age_years", "tmt_b_seconds"):
            with self.subTest(column=column):
                covariates = pd.DataFrame(
                    {
                        "participant_id": ["P01", "P01", "P02"],
                        "age_years": [25.0, 25.0, 68.0],
                        "tmt_b_seconds": [42.0, 42.0, 91.0],
                    }
                )
                covariates.loc[1, column] = (
                    26.0 if column == "age_years" else 43.0
                )

                result = self.validate(participant_covariates=covariates)

                self.assertFalse(result.valid)
                self.assertTrue(
                    any("inconsistent" in error.lower() for error in result.errors)
                )

    def test_input_frames_are_unchanged_and_returned_frames_are_deep_copies(self):
        trial_features = make_trial_features()
        participant_covariates = make_covariates()
        original_trial_features = trial_features.copy(deep=True)
        original_participant_covariates = participant_covariates.copy(deep=True)

        result = self.validate(
            trial_features=trial_features,
            participant_covariates=participant_covariates,
        )

        pd.testing.assert_frame_equal(trial_features, original_trial_features)
        pd.testing.assert_frame_equal(
            participant_covariates,
            original_participant_covariates,
        )
        self.assertNotIn("difficulty_stage", trial_features.columns)

        result.trial_features.loc[0, "performance"] = -123.0
        result.participant_covariates.loc[0, "age_years"] = -456.0

        self.assertEqual(trial_features.loc[0, "performance"], original_trial_features.loc[0, "performance"])
        self.assertEqual(
            participant_covariates.loc[0, "age_years"],
            original_participant_covariates.loc[0, "age_years"],
        )

    def test_missing_required_columns_do_not_prevent_result_copy(self):
        trial_features = pd.DataFrame({"participant_id": ["P01"]})
        covariates = pd.DataFrame({"participant_id": ["P01"]})

        result = self.validate(
            trial_features=trial_features,
            participant_covariates=covariates,
        )

        self.assertFalse(result.valid)
        self.assertIn("difficulty_stage", result.trial_features.columns)
        self.assertTrue(result.trial_features["difficulty_stage"].isna().all())


if __name__ == "__main__":
    unittest.main()
