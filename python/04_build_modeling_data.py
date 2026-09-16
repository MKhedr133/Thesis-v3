"""Build the auditable FE-01 modelling table without fitting models."""

from __future__ import annotations

from dataclasses import dataclass
import importlib.util
from pathlib import Path
import sys
from typing import Any

import pandas as pd


_VALIDATOR_PATH = Path(__file__).with_name("03_validate_modeling_inputs.py")
_VALIDATOR_SPEC = importlib.util.spec_from_file_location(
    "mod01a_validate_modeling_inputs",
    _VALIDATOR_PATH,
)
if _VALIDATOR_SPEC is None or _VALIDATOR_SPEC.loader is None:
    raise ImportError(f"Cannot load MOD-01A validator from {_VALIDATOR_PATH}")
_VALIDATOR = importlib.util.module_from_spec(_VALIDATOR_SPEC)
sys.modules[_VALIDATOR_SPEC.name] = _VALIDATOR
_VALIDATOR_SPEC.loader.exec_module(_VALIDATOR)


ValidationCheck = _VALIDATOR.ValidationCheck
validate_modeling_inputs = _VALIDATOR.validate_modeling_inputs

PARTICIPANT_COVARIATE_COLUMNS = ("age_years", "tmt_b_seconds")
CHECK_COLUMNS = ("check_id", "check_name", "status", "count", "message")


@dataclass(frozen=True)
class ModelingDataBuildResult:
    """Immutable result container for the MOD-01B assembly step."""

    valid: bool
    errors: tuple[str, ...]
    warnings: tuple[str, ...]
    checks: tuple[ValidationCheck, ...]
    modeling_data: pd.DataFrame
    modeling_data_checks: pd.DataFrame


def _append_check(
    checks: list[ValidationCheck],
    errors: list[str],
    warnings: list[str],
    *,
    name: str,
    status: str,
    count: int,
    message: str,
) -> None:
    check = ValidationCheck(
        name=name,
        status=status,
        count=int(count),
        message=message,
    )
    checks.append(check)
    if status == "fail":
        errors.append(f"{name}: {message}")
    elif status == "warning":
        warnings.append(f"{name}: {message}")


def _ordered_trial_columns(
    original_columns: list[str],
    validated_columns: list[str],
) -> list[str]:
    """Keep input columns in order and place the derived stage beside difficulty."""

    columns = list(original_columns)
    if "difficulty_stage" not in columns:
        insert_at = (
            columns.index("difficulty_level") + 1
            if "difficulty_level" in columns
            else len(columns)
        )
        columns.insert(insert_at, "difficulty_stage")

    return [column for column in columns if column in validated_columns]


def _checks_frame(checks: tuple[ValidationCheck, ...]) -> pd.DataFrame:
    rows = [
        {
            "check_id": f"MOD-01B:{check.name}",
            "check_name": check.name,
            "status": check.status,
            "count": check.count,
            "message": check.message,
        }
        for check in checks
    ]
    return pd.DataFrame(rows, columns=CHECK_COLUMNS)


def _duplicate_covariate_ids(covariates: pd.DataFrame) -> set[Any]:
    if "participant_id" not in covariates.columns:
        return set()
    duplicate_mask = covariates.duplicated(
        subset=["participant_id"],
        keep=False,
    )
    return set(
        covariates.loc[
            duplicate_mask & covariates["participant_id"].notna(),
            "participant_id",
        ].tolist()
    )


def build_modeling_data(
    trial_features: pd.DataFrame,
    participant_covariates: pd.DataFrame,
) -> ModelingDataBuildResult:
    """Assemble one auditable row per trial from validated normalized inputs.

    Participant covariates are joined with a pandas many-to-one merge. Duplicate
    participant keys are excluded from the lookup so they can never multiply
    trial rows; affected Age/TMT-B values remain missing and the validator's
    failure is preserved.
    """

    validation = validate_modeling_inputs(trial_features, participant_covariates)
    trial_copy = validation.trial_features.copy(deep=True)
    covariate_copy = validation.participant_covariates.copy(deep=True)
    checks = list(validation.checks)
    errors = list(validation.errors)
    warnings = list(validation.warnings)

    ordered_columns = _ordered_trial_columns(
        list(trial_features.columns),
        list(trial_copy.columns),
    )
    for column in ordered_columns:
        if column not in trial_copy.columns:
            trial_copy[column] = pd.NA
    trial_copy = trial_copy.loc[:, ordered_columns].copy(deep=True)

    collisions = [
        column
        for column in PARTICIPANT_COVARIATE_COLUMNS
        if column in trial_copy.columns
    ]
    assembled = trial_copy

    if collisions:
        _append_check(
            checks,
            errors,
            warnings,
            name="participant_covariate_column_collision",
            status="fail",
            count=len(collisions),
            message=(
                "Participant-covariate columns already exist in trial_features: "
                + ", ".join(collisions)
                + "; existing trial columns were preserved."
            ),
        )
    elif all(
        column in covariate_copy.columns
        for column in ("participant_id", *PARTICIPANT_COVARIATE_COLUMNS)
    ) and "participant_id" in assembled.columns:
        duplicate_ids = _duplicate_covariate_ids(covariate_copy)
        lookup = covariate_copy.loc[
            ~covariate_copy["participant_id"].isin(duplicate_ids)
            & covariate_copy["participant_id"].notna(),
            ["participant_id", *PARTICIPANT_COVARIATE_COLUMNS],
        ].copy(deep=True)

        if lookup["participant_id"].duplicated().any():
            _append_check(
                checks,
                errors,
                warnings,
                name="many_to_one_covariate_merge",
                status="fail",
                count=int(lookup["participant_id"].duplicated().sum()),
                message=(
                    "Participant covariate lookup is not unique; no covariate "
                    "values were merged."
                ),
            )
            for column in PARTICIPANT_COVARIATE_COLUMNS:
                assembled[column] = pd.NA
        else:
            assembled = assembled.copy(deep=True)
            assembled["__mod01b_row_order"] = range(len(assembled))
            merged = assembled.merge(
                lookup,
                how="left",
                on="participant_id",
                sort=False,
                validate="many_to_one",
            )
            merged = merged.sort_values(
                "__mod01b_row_order",
                kind="stable",
            ).drop(columns=["__mod01b_row_order"])
            assembled = merged.reset_index(drop=True)
    else:
        for column in PARTICIPANT_COVARIATE_COLUMNS:
            assembled[column] = pd.NA

    if len(assembled) != len(trial_copy):
        _append_check(
            checks,
            errors,
            warnings,
            name="modeling_row_preservation",
            status="fail",
            count=abs(len(assembled) - len(trial_copy)),
            message=(
                "The assembled table changed the trial-row count; the result "
                "was replaced with a one-row-per-input-trial table."
            ),
        )
        assembled = trial_copy.copy(deep=True)
        for column in PARTICIPANT_COVARIATE_COLUMNS:
            assembled[column] = pd.NA
    else:
        _append_check(
            checks,
            errors,
            warnings,
            name="modeling_row_preservation",
            status="pass",
            count=len(trial_copy),
            message="Every input trial row appears exactly once in modeling_data.",
        )

    output_columns = [
        *ordered_columns,
        *[
            column
            for column in PARTICIPANT_COVARIATE_COLUMNS
            if column not in ordered_columns
        ],
    ]
    for column in output_columns:
        if column not in assembled.columns:
            assembled[column] = pd.NA
    assembled = assembled.loc[:, output_columns].copy(deep=True)

    final_checks = tuple(checks)
    checks_frame = _checks_frame(final_checks)
    valid = not any(check.status == "fail" for check in final_checks)
    return ModelingDataBuildResult(
        valid=valid,
        errors=tuple(errors),
        warnings=tuple(warnings),
        checks=final_checks,
        modeling_data=assembled,
        modeling_data_checks=checks_frame,
    )


def write_modeling_outputs(
    result: ModelingDataBuildResult,
    modeling_data_path: Path,
    checks_path: Path,
) -> None:
    """Write only the two explicit MOD-01B CSV outputs."""

    modeling_destination = Path(modeling_data_path)
    checks_destination = Path(checks_path)
    modeling_destination.parent.mkdir(parents=True, exist_ok=True)
    checks_destination.parent.mkdir(parents=True, exist_ok=True)
    result.modeling_data.to_csv(
        modeling_destination,
        index=False,
        encoding="utf-8",
        na_rep="",
    )
    result.modeling_data_checks.to_csv(
        checks_destination,
        index=False,
        encoding="utf-8",
        na_rep="",
    )
