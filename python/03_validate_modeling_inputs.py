"""Validate the normalized FE-01 input contract for modelling.

This module intentionally stops at structural and modelling-readiness checks.
It does not construct a final modelling table, fit a model, or write files.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from numbers import Real
from typing import Any

import pandas as pd


SUPPORTED_CONDITIONS = ("Visual", "Auditory", "Cognitive")
DIFFICULTY_TO_STAGE = {0: 0, 2: 1, 6: 2, 10: 3}

TRIAL_IDENTIFIER_COLUMNS = (
    "participant_id",
    "session_id",
    "source_tracker_csv_filename",
    "participant_group",
    "condition_name",
    "difficulty_level",
    "trial_order",
    "language",
)

TRIAL_OUTCOME_COLUMNS = (
    "performance",
    "correct_products_collected_count",
    "performance_percent",
    "performance_change_from_d0_percentage_points",
    "mental_demand_score_0_to_10",
    # Retained for historical/supplementary provenance, not as a primary target.
    "error_change_from_d0",
)

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

TOTAL_ERROR_COLUMN = "total_error_count"
PARTICIPANT_COVARIATE_COLUMNS = (
    "participant_id",
    "age_years",
    "tmt_b_seconds",
)

TRIAL_KEY_COLUMNS = (
    "participant_id",
    "session_id",
    "source_tracker_csv_filename",
    "condition_name",
    "difficulty_level",
    "trial_order",
)

REQUIRED_TRIAL_COLUMNS = (
    *TRIAL_IDENTIFIER_COLUMNS,
    *TRIAL_OUTCOME_COLUMNS,
    *BEHAVIOURAL_COLUMNS,
    *ERROR_COMPONENT_COLUMNS,
    TOTAL_ERROR_COLUMN,
)


@dataclass(frozen=True)
class ValidationCheck:
    """One explicit modelling-input validation result."""

    name: str
    status: str
    count: int
    message: str


@dataclass(frozen=True)
class ModelingInputValidationResult:
    """Immutable result container holding validation evidence and deep copies."""

    valid: bool
    errors: tuple[str, ...]
    warnings: tuple[str, ...]
    checks: tuple[ValidationCheck, ...]
    trial_features: pd.DataFrame
    participant_covariates: pd.DataFrame


def _is_missing(value: Any) -> bool:
    """Return whether a scalar value is missing without ambiguous NA booleans."""

    result = pd.isna(value)
    try:
        return bool(result)
    except (TypeError, ValueError):
        return False


def _finite_number(value: Any) -> float | None:
    """Convert a scalar numeric value to finite float, otherwise return None."""

    if _is_missing(value) or isinstance(value, bool) or not isinstance(value, Real):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def _difficulty_value(value: Any) -> int | None:
    """Return the accepted numeric FE-01 difficulty value, if valid."""

    number = _finite_number(value)
    if number is None or not number.is_integer():
        return None
    integer = int(number)
    return integer if integer in DIFFICULTY_TO_STAGE else None


def validate_modeling_inputs(
    trial_features: pd.DataFrame,
    participant_covariates: pd.DataFrame,
) -> ModelingInputValidationResult:
    """Validate normalized FE-01 trial and participant modelling inputs.

    The returned data frames are deep copies. The trial copy receives the
    derived ``difficulty_stage`` column; caller-owned frames are never changed.
    """

    if not isinstance(trial_features, pd.DataFrame):
        raise TypeError("trial_features must be a pandas DataFrame")
    if not isinstance(participant_covariates, pd.DataFrame):
        raise TypeError("participant_covariates must be a pandas DataFrame")

    trial_copy = trial_features.copy(deep=True)
    covariate_copy = participant_covariates.copy(deep=True)
    checks: list[ValidationCheck] = []
    errors: list[str] = []
    warnings: list[str] = []

    def add_check(name: str, status: str, count: int, message: str) -> None:
        check = ValidationCheck(name=name, status=status, count=int(count), message=message)
        checks.append(check)
        if status == "fail":
            errors.append(f"{name}: {message}")
        elif status == "warning":
            warnings.append(f"{name}: {message}")

    missing_trial_columns = [
        column for column in REQUIRED_TRIAL_COLUMNS if column not in trial_copy.columns
    ]
    if missing_trial_columns:
        add_check(
            "required_trial_columns",
            "fail",
            len(missing_trial_columns),
            "Missing trial-feature columns: " + ", ".join(missing_trial_columns),
        )
    else:
        add_check(
            "required_trial_columns",
            "pass",
            len(REQUIRED_TRIAL_COLUMNS),
            "All required normalized FE-01 trial columns are present.",
        )

    missing_covariate_columns = [
        column
        for column in PARTICIPANT_COVARIATE_COLUMNS
        if column not in covariate_copy.columns
    ]
    if missing_covariate_columns:
        add_check(
            "required_participant_covariate_columns",
            "fail",
            len(missing_covariate_columns),
            "Missing participant-covariate columns: "
            + ", ".join(missing_covariate_columns),
        )
    else:
        add_check(
            "required_participant_covariate_columns",
            "pass",
            len(PARTICIPANT_COVARIATE_COLUMNS),
            "All required normalized participant-covariate columns are present.",
        )

    if "difficulty_level" in trial_copy.columns:
        difficulty_values = trial_copy["difficulty_level"].map(_difficulty_value)
        invalid_difficulty = difficulty_values.isna()
        invalid_count = int(invalid_difficulty.sum())
        add_check(
            "supported_difficulty_values",
            "fail" if invalid_count else "pass",
            invalid_count,
            (
                "Unsupported or missing difficulty values were found. Expected "
                "numeric FE-01 values 0, 2, 6, or 10."
                if invalid_count
                else "All difficulty values are supported."
            ),
        )
        trial_copy["difficulty_stage"] = pd.Series(
            difficulty_values.map(
                lambda value: DIFFICULTY_TO_STAGE.get(int(value))
                if not _is_missing(value)
                else pd.NA
            ),
            index=trial_copy.index,
            dtype="Int64",
        )
    else:
        add_check(
            "supported_difficulty_values",
            "fail",
            0,
            "Cannot validate difficulty values because difficulty_level is missing.",
        )
        trial_copy["difficulty_stage"] = pd.Series(
            pd.array([pd.NA] * len(trial_copy), dtype="Int64"),
            index=trial_copy.index,
        )

    if "condition_name" in trial_copy.columns:
        invalid_conditions = ~trial_copy["condition_name"].isin(SUPPORTED_CONDITIONS)
        invalid_count = int(invalid_conditions.sum())
        add_check(
            "supported_condition_values",
            "fail" if invalid_count else "pass",
            invalid_count,
            (
                "Unsupported or missing condition values were found. Expected "
                "Visual, Auditory, or Cognitive."
                if invalid_count
                else "All condition values are supported and remain separate."
            ),
        )

    available_key_columns = [
        column for column in TRIAL_KEY_COLUMNS if column in trial_copy.columns
    ]
    if len(available_key_columns) == len(TRIAL_KEY_COLUMNS):
        duplicate_mask = trial_copy.duplicated(
            subset=TRIAL_KEY_COLUMNS,
            keep=False,
        )
        duplicate_count = int(duplicate_mask.sum())
        add_check(
            "duplicate_trial_keys",
            "fail" if duplicate_count else "pass",
            duplicate_count,
            (
                "Duplicate normalized trial keys were found."
                if duplicate_count
                else "No duplicate normalized trial keys were found."
            ),
        )

    available_identifier_columns = [
        column for column in TRIAL_IDENTIFIER_COLUMNS if column in trial_copy.columns
    ]
    if len(available_identifier_columns) == len(TRIAL_IDENTIFIER_COLUMNS):
        missing_identifier_count = int(
            trial_copy[list(TRIAL_IDENTIFIER_COLUMNS)].isna().any(axis=1).sum()
        )
        add_check(
            "trial_identifier_values",
            "fail" if missing_identifier_count else "pass",
            missing_identifier_count,
            (
                "Rows contain missing required trial identifiers."
                if missing_identifier_count
                else "Required trial identifier values are present."
            ),
        )

        group_counts = trial_copy.groupby(
            "participant_id",
            dropna=False,
        )["participant_group"].nunique(dropna=False)
        inconsistent_group_ids = group_counts[group_counts > 1]
        inconsistent_group_count = int(inconsistent_group_ids.size)
        add_check(
            "participant_group_consistency",
            "fail" if inconsistent_group_count else "pass",
            inconsistent_group_count,
            (
                "Participants have inconsistent participant_group values."
                if inconsistent_group_count
                else "Participant groups are internally consistent."
            ),
        )

    if "mental_demand_score_0_to_10" in trial_copy.columns:
        mental_demand = trial_copy["mental_demand_score_0_to_10"]
        missing_count = int(mental_demand.isna().sum())
        add_check(
            "mental_demand_missingness",
            "warning" if missing_count else "pass",
            missing_count,
            (
                "Mental-demand values are missing; rows are preserved."
                if missing_count
                else "No mental-demand values are missing."
            ),
        )
        invalid_mental_demand = []
        for value in mental_demand:
            if _is_missing(value):
                continue
            numeric_value = _finite_number(value)
            if numeric_value is None or not 0.0 <= numeric_value <= 10.0:
                invalid_mental_demand.append(value)
        invalid_count = len(invalid_mental_demand)
        add_check(
            "mental_demand_range",
            "fail" if invalid_count else "pass",
            invalid_count,
            (
                "Nonmissing mental-demand values must be finite and within 0–10."
                if invalid_count
                else "All nonmissing mental-demand values are finite and within 0–10."
            ),
        )

    if "performance_change_from_d0_percentage_points" in trial_copy.columns and (
        "difficulty_level" in trial_copy.columns
    ):
        d0_mask = trial_copy["difficulty_level"].map(_difficulty_value).eq(0)
        invalid_d0_change = 0
        for value in trial_copy.loc[
            d0_mask,
            "performance_change_from_d0_percentage_points",
        ]:
            if _is_missing(value):
                continue
            numeric_value = _finite_number(value)
            if numeric_value is None or numeric_value != 0.0:
                invalid_d0_change += 1
        add_check(
            "d0_performance_change",
            "fail" if invalid_d0_change else "pass",
            invalid_d0_change,
            (
                "D0 performance change must be missing or zero."
                if invalid_d0_change
                else "D0 performance change is missing or zero in every D0 row."
            ),
        )

    present_behavioural_columns = [
        column for column in BEHAVIOURAL_COLUMNS if column in trial_copy.columns
    ]
    if present_behavioural_columns:
        missing_behavioural_count = int(
            trial_copy[present_behavioural_columns].isna().sum().sum()
        )
        add_check(
            "behavioural_feature_missingness",
            "warning" if missing_behavioural_count else "pass",
            missing_behavioural_count,
            (
                "Behavioural values are missing; rows are preserved."
                if missing_behavioural_count
                else "No behavioural feature values are missing."
            ),
        )

    if all(column in trial_copy.columns for column in (*ERROR_COMPONENT_COLUMNS, TOTAL_ERROR_COLUMN)):
        incomplete_count = 0
        mismatch_count = 0
        for _, row in trial_copy.iterrows():
            component_values = [
                _finite_number(row[column]) for column in ERROR_COMPONENT_COLUMNS
            ]
            total_value = _finite_number(row[TOTAL_ERROR_COLUMN])
            if total_value is None or any(value is None for value in component_values):
                incomplete_count += 1
                continue
            if not math.isclose(
                sum(component_values),
                total_value,
                rel_tol=0.0,
                abs_tol=1e-12,
            ):
                mismatch_count += 1

        add_check(
            "total_error_incomplete_audit",
            "fail" if incomplete_count else "pass",
            incomplete_count,
            (
                "Some rows cannot be audited because a total or component value "
                "is missing or nonnumeric; rows were preserved."
                if incomplete_count
                else "Every row has complete total-error audit inputs."
            ),
        )
        add_check(
            "total_error_component_mismatch",
            "fail" if mismatch_count else "pass",
            mismatch_count,
            (
                "Total error counts do not match the sum of their four components; "
                "values were not repaired."
                if mismatch_count
                else "Total error counts match the four component sums."
            ),
        )

    if all(column in covariate_copy.columns for column in PARTICIPANT_COVARIATE_COLUMNS):
        duplicate_covariate_mask = covariate_copy.duplicated(
            subset=["participant_id"],
            keep=False,
        )
        duplicate_covariate_count = int(duplicate_covariate_mask.sum())
        add_check(
            "duplicate_participant_covariates",
            "fail" if duplicate_covariate_count else "pass",
            duplicate_covariate_count,
            (
                "Duplicate participant-covariate rows were found."
                if duplicate_covariate_count
                else "No duplicate participant-covariate rows were found."
            ),
        )

        missing_covariate_value_count = int(
            covariate_copy[list(PARTICIPANT_COVARIATE_COLUMNS)].isna().any(axis=1).sum()
        )
        add_check(
            "participant_covariate_values",
            "fail" if missing_covariate_value_count else "pass",
            missing_covariate_value_count,
            (
                "Participant covariates contain missing values."
                if missing_covariate_value_count
                else "Participant covariate values are present."
            ),
        )

        invalid_covariate_value_count = 0
        for column in ("age_years", "tmt_b_seconds"):
            invalid_covariate_value_count += sum(
                _finite_number(value) is None
                for value in covariate_copy[column]
                if not _is_missing(value)
            )
        add_check(
            "participant_covariate_finiteness",
            "fail" if invalid_covariate_value_count else "pass",
            invalid_covariate_value_count,
            (
                "Age/TMT-B covariates must be finite numeric values."
                if invalid_covariate_value_count
                else "Age/TMT-B covariates are finite numeric values."
            ),
        )

        inconsistent_covariate_participants: set[Any] = set()
        for column in ("age_years", "tmt_b_seconds"):
            counts = covariate_copy.groupby(
                "participant_id",
                dropna=False,
            )[column].nunique(dropna=False)
            inconsistent_covariate_participants.update(counts[counts > 1].index)
        inconsistent_covariate_count = len(inconsistent_covariate_participants)
        add_check(
            "participant_covariate_consistency",
            "fail" if inconsistent_covariate_count else "pass",
            inconsistent_covariate_count,
            (
                "Participants have inconsistent Age/TMT-B covariate values."
                if inconsistent_covariate_count
                else "Age/TMT-B values are internally consistent per participant."
            ),
        )

        if "participant_id" in trial_copy.columns:
            trial_participants = set(trial_copy["participant_id"].dropna())
            covariate_participants = set(
                covariate_copy["participant_id"].dropna()
            )
            missing_coverage = trial_participants - covariate_participants
            add_check(
                "participant_covariate_coverage",
                "fail" if missing_coverage else "pass",
                len(missing_coverage),
                (
                    "Trial participants are missing covariates: "
                    + ", ".join(sorted(map(str, missing_coverage)))
                    if missing_coverage
                    else "Every trial participant has covariates."
                ),
            )

    valid = not any(check.status == "fail" for check in checks)
    return ModelingInputValidationResult(
        valid=valid,
        errors=tuple(errors),
        warnings=tuple(warnings),
        checks=tuple(checks),
        trial_features=trial_copy,
        participant_covariates=covariate_copy,
    )
