"""Fit synthetic-contract random-intercept alternatives for MOD-02B.

The module consumes a caller-supplied modelling table and the MOD-02A
pre-fit registry. It fits one random-intercept MixedLM per condition-target
specification, records the comparison contract in memory, and does not write
modelling outputs or select a random-effects structure.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import importlib.util
from pathlib import Path
import re
import sys
from typing import Any
import warnings

import pandas as pd
import statsmodels.formula.api as smf


_REGISTRY_PATH = Path(__file__).with_name("05_compare_random_effects.py")
_REGISTRY_SPEC = importlib.util.spec_from_file_location(
    "mod02a_random_effects_registry",
    _REGISTRY_PATH,
)
if _REGISTRY_SPEC is None or _REGISTRY_SPEC.loader is None:
    raise ImportError(f"Cannot load MOD-02A registry from {_REGISTRY_PATH}")
_REGISTRY = importlib.util.module_from_spec(_REGISTRY_SPEC)
sys.modules[_REGISTRY_SPEC.name] = _REGISTRY
_REGISTRY_SPEC.loader.exec_module(_REGISTRY)

RandomEffectsComparisonRegistry = _REGISTRY.RandomEffectsComparisonRegistry
RandomEffectsComparisonSpec = _REGISTRY.RandomEffectsComparisonSpec
LIKELIHOOD_RATIO_ROLE = _REGISTRY.LIKELIHOOD_RATIO_ROLE
SELECTION_STATUS = _REGISTRY.SELECTION_STATUS


PERFORMANCE_TARGET = "performance_change_from_d0_percentage_points"
MENTAL_DEMAND_TARGET = "mental_demand_score_0_to_10"
PERFORMANCE_STAGE_COLUMN = "difficulty_stage"
DEFAULT_OPTIMIZERS = ("lbfgs", "powell")
COMPARISON_EXTRA_FIELDS = (
    "executable_formula",
    "fit_status",
    "excluded_observation_count",
    "excluded_participant_count",
    "fit_error",
    "included_row_signature",
)
_FORMULA_TOKEN = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
_FORMULA_RESERVED = {
    "C",
    "I",
    "Q",
    "True",
    "False",
    "and",
    "or",
    "not",
    "np",
    "log",
    "sqrt",
}
_SOURCE_ROW_POSITION = "__mod02_source_row_position"


@dataclass(frozen=True)
class RandomInterceptFitRecord:
    """One in-memory MOD-02B RI fit record."""

    condition_name: str
    target_name: str
    model_id: str
    target_role: str
    response_coding: str
    difficulty_term: str
    difficulty_source_column: str | None
    difficulty_coding_status: str
    difficulty_mapping: tuple[tuple[int, int], ...] | None
    fixed_effects_formula: str
    executable_formula: str
    fixed_effects_status: str
    random_structure: str
    random_formula_label: str
    participant_count: int
    observation_count: int
    excluded_observation_count: int
    excluded_participant_count: int
    fixed_effect_count: int | None
    total_parameter_count: int | None
    log_likelihood: float | None
    aic: float | None
    delta_aic: float | None
    bic: float | None
    delta_bic: float | None
    estimation_method: str
    reml: bool
    convergence_status: bool | None
    optimizer: str
    warnings: tuple[str, ...]
    fit_status: str
    fit_error: str | None
    random_intercept_variance: float | None
    random_slope_variance: float | None
    intercept_slope_covariance: float | None
    boundary_flag: bool | None
    singularity_flag: bool | None
    likelihood_ratio_statistic: float | None
    likelihood_ratio_p_value: float | None
    likelihood_ratio_role: str
    selection_status: str
    included_row_signature: str


@dataclass(frozen=True)
class RandomInterceptFitResult:
    """Immutable result container; contained pandas frames remain mutable."""

    valid: bool
    errors: tuple[str, ...]
    warnings: tuple[str, ...]
    fit_records: tuple[RandomInterceptFitRecord, ...]
    fit_table: pd.DataFrame


@dataclass(frozen=True)
class _FitAttempt:
    result: Any | None
    optimizer: str
    warnings: tuple[str, ...]
    error: str | None


def _unique(values: list[str]) -> tuple[str, ...]:
    return tuple(dict.fromkeys(value for value in values if value))


def _safe_float(value: Any) -> float | None:
    try:
        if value is None or pd.isna(value):
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _safe_len(value: Any) -> int | None:
    try:
        return int(len(value))
    except (TypeError, ValueError):
        return None


def _backend_flag(result: Any, names: tuple[str, ...]) -> bool | None:
    """Read only explicit backend boolean diagnostics; never infer thresholds."""
    for name in names:
        value = getattr(result, name, None)
        if value is True or value is False:
            return bool(value)
    return None


def _random_intercept_spec(
    comparison: RandomEffectsComparisonSpec,
) -> Any | None:
    for model in comparison.models:
        if model.model_id == "RI":
            return model
    return None


def _formula_response(formula: str) -> str:
    return formula.split("~", 1)[0].strip() if "~" in formula else ""


def _formula_tokens(formula: str) -> tuple[str, ...]:
    return tuple(
        token
        for token in _FORMULA_TOKEN.findall(formula)
        if token not in _FORMULA_RESERVED
    )


def _missing_formula_columns(
    formula: str,
    available_columns: set[str],
) -> tuple[str, ...]:
    return tuple(
        token
        for token in _formula_tokens(formula)
        if token not in available_columns
    )


def _formula_columns(
    formula: str,
    available_columns: set[str],
) -> tuple[str, ...]:
    return tuple(
        column
        for column in available_columns
        if re.search(rf"(?<![A-Za-z0-9_]){re.escape(column)}(?![A-Za-z0-9_])", formula)
    )


def _included_row_signature(data: pd.DataFrame) -> str:
    positions = sorted(
        int(value)
        for value in data[_SOURCE_ROW_POSITION].tolist()
    )
    return ",".join(str(position) for position in positions)


def _attempt_fit(
    model: Any,
    optimizer: str,
    maxiter: int,
) -> _FitAttempt:
    caught: list[warnings.WarningMessage] = []
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            fitted = model.fit(
                method=optimizer,
                reml=False,
                maxiter=maxiter,
                disp=False,
            )
        warning_text = tuple(dict.fromkeys(str(item.message) for item in caught))
        return _FitAttempt(
            result=fitted,
            optimizer=optimizer,
            warnings=warning_text,
            error=None,
        )
    except Exception as exc:
        warning_text = tuple(dict.fromkeys(str(item.message) for item in caught))
        return _FitAttempt(
            result=None,
            optimizer=optimizer,
            warnings=warning_text,
            error=f"{optimizer} failed: {exc}",
        )


def _fit_with_fallback(
    data: pd.DataFrame,
    formula: str,
    participant_column: str,
    optimizer_methods: tuple[str, ...],
    maxiter: int,
) -> tuple[_FitAttempt | None, tuple[str, ...], tuple[str, ...]]:
    construction_warnings: tuple[str, ...] = ()
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            model = smf.mixedlm(
                formula=formula,
                data=data,
                groups=data[participant_column],
                re_formula="1",
            )
        construction_warnings = tuple(
            dict.fromkeys(str(item.message) for item in caught)
        )
    except Exception as exc:
        return (
            None,
            construction_warnings,
            (f"mixedlm construction failed: {exc}",),
        )

    attempts: list[_FitAttempt] = []
    for optimizer in optimizer_methods:
        attempt = _attempt_fit(
            model,
            optimizer,
            maxiter,
        )
        attempts.append(attempt)
        if attempt.result is not None and getattr(attempt.result, "converged", True):
            break

    selected = next(
        (attempt for attempt in reversed(attempts) if attempt.result is not None),
        None,
    )
    warning_text = _unique(
        [
            *construction_warnings,
            *[warning for attempt in attempts for warning in attempt.warnings],
        ]
    )
    errors = _unique(
        [attempt.error for attempt in attempts if attempt.error is not None]
    )
    return selected, warning_text, errors


def _random_intercept_variance(result: Any) -> float | None:
    covariance = getattr(result, "cov_re", None)
    try:
        if hasattr(covariance, "iloc"):
            return _safe_float(covariance.iloc[0, 0])
        return _safe_float(covariance[0][0])
    except (IndexError, KeyError, TypeError):
        return None


def _record(
    comparison: RandomEffectsComparisonSpec,
    *,
    executable_formula: str,
    random_model: Any | None,
    difficulty_source_column: str | None = None,
    participant_count: int = 0,
    observation_count: int = 0,
    excluded_observation_count: int = 0,
    excluded_participant_count: int = 0,
    fixed_effect_count: int | None = None,
    total_parameter_count: int | None = None,
    log_likelihood: float | None = None,
    aic: float | None = None,
    bic: float | None = None,
    convergence_status: bool | None = None,
    optimizer: str = "not_attempted",
    warnings_text: tuple[str, ...] = (),
    fit_status: str = "failed",
    fit_error: str | None = None,
    random_intercept_variance: float | None = None,
    boundary_flag: bool | None = None,
    singularity_flag: bool | None = None,
    included_row_signature: str = "",
) -> RandomInterceptFitRecord:
    difficulty = comparison.difficulty_coding
    return RandomInterceptFitRecord(
        condition_name=comparison.condition_name,
        target_name=comparison.target_name,
        model_id="RI",
        target_role=comparison.target_role,
        response_coding=comparison.response_coding,
        difficulty_term=difficulty.term_name,
        difficulty_source_column=(
            difficulty_source_column
            if difficulty_source_column is not None
            else difficulty.source_column
        ),
        difficulty_coding_status=difficulty.coding_status,
        difficulty_mapping=difficulty.mapping,
        fixed_effects_formula=comparison.fixed_effects_formula,
        executable_formula=executable_formula,
        fixed_effects_status=comparison.fixed_effects_status,
        random_structure=random_model.random_structure if random_model else "RI",
        random_formula_label=(
            random_model.random_formula_label
            if random_model
            else "(1 | participant)"
        ),
        participant_count=participant_count,
        observation_count=observation_count,
        excluded_observation_count=excluded_observation_count,
        excluded_participant_count=excluded_participant_count,
        fixed_effect_count=fixed_effect_count,
        total_parameter_count=total_parameter_count,
        log_likelihood=log_likelihood,
        aic=aic,
        delta_aic=None,
        bic=bic,
        delta_bic=None,
        estimation_method="maximum_likelihood",
        reml=False,
        convergence_status=convergence_status,
        optimizer=optimizer,
        warnings=warnings_text,
        fit_status=fit_status,
        fit_error=fit_error,
        random_intercept_variance=random_intercept_variance,
        random_slope_variance=None,
        intercept_slope_covariance=None,
        boundary_flag=boundary_flag,
        singularity_flag=singularity_flag,
        likelihood_ratio_statistic=None,
        likelihood_ratio_p_value=None,
        likelihood_ratio_role=LIKELIHOOD_RATIO_ROLE,
        selection_status=SELECTION_STATUS,
        included_row_signature=included_row_signature,
    )


def _table_from_records(
    records: tuple[RandomInterceptFitRecord, ...],
    registry: RandomEffectsComparisonRegistry,
) -> pd.DataFrame:
    fields = tuple(registry.comparison_fields) + COMPARISON_EXTRA_FIELDS
    rows: list[dict[str, Any]] = []
    for record in records:
        row: dict[str, Any] = {}
        for field in registry.comparison_fields:
            value = getattr(record, field)
            if field == "warnings":
                value = "; ".join(record.warnings)
            row[field] = value
        for field in COMPARISON_EXTRA_FIELDS:
            row[field] = getattr(record, field)
        rows.append(row)
    return pd.DataFrame(rows, columns=fields)


def fit_random_intercept_models(
    modeling_data: pd.DataFrame,
    registry: RandomEffectsComparisonRegistry,
    executable_formulas: Mapping[str, str],
    difficulty_columns: Mapping[str, str],
    *,
    target_columns: Mapping[str, str] | None = None,
    participant_column: str = "participant_id",
    condition_column: str = "condition_name",
    maxiter: int = 1000,
    optimizer_methods: tuple[str, ...] = DEFAULT_OPTIMIZERS,
) -> RandomInterceptFitResult:
    """Fit one RI alternative for each supplied registry condition-target pair."""
    if not isinstance(modeling_data, pd.DataFrame):
        raise TypeError("modeling_data must be a pandas DataFrame")
    if not isinstance(executable_formulas, Mapping):
        raise TypeError("executable_formulas must be a mapping")
    if not isinstance(difficulty_columns, Mapping):
        raise TypeError("difficulty_columns must be a mapping")
    if target_columns is not None and not isinstance(target_columns, Mapping):
        raise TypeError("target_columns must be a mapping")
    if not optimizer_methods:
        raise ValueError("optimizer_methods must not be empty")
    if maxiter <= 0:
        raise ValueError("maxiter must be positive")

    data = modeling_data.copy(deep=True)
    data[_SOURCE_ROW_POSITION] = list(range(len(data)))
    target_columns = target_columns or {}
    available_columns = set(data.columns)
    records: list[RandomInterceptFitRecord] = []
    errors: list[str] = []
    result_warnings: list[str] = []

    for comparison in registry.comparisons:
        random_model = _random_intercept_spec(comparison)
        target_name = comparison.target_name
        formula = executable_formulas.get(target_name, "")
        response_column = target_columns.get(target_name, target_name)
        difficulty_column = difficulty_columns.get(target_name)
        local_warnings: list[str] = []

        if random_model is None:
            message = (
                f"{comparison.condition_name}/{target_name}: registry has no RI model"
            )
            errors.append(message)
            records.append(
                _record(
                    comparison,
                    executable_formula=formula,
                    random_model=None,
                    difficulty_source_column=difficulty_column,
                    fit_error=message,
                )
            )
            continue

        contract_errors: list[str] = []
        if not isinstance(formula, str) or not formula.strip():
            contract_errors.append("missing executable formula")
        elif _formula_response(formula) != response_column:
            contract_errors.append(
                f"formula response must be {response_column!r}"
            )
        if response_column not in available_columns:
            contract_errors.append(f"missing response column {response_column!r}")
        if not difficulty_column:
            contract_errors.append("missing explicit difficulty-column mapping")
        elif difficulty_column not in available_columns:
            contract_errors.append(
                f"missing difficulty column {difficulty_column!r}"
            )
        if (
            target_name == MENTAL_DEMAND_TARGET
            and difficulty_column != "difficulty_stage"
        ):
            contract_errors.append(
                "mental demand must use difficulty_stage"
            )
        if (
            target_name == PERFORMANCE_TARGET
            and not difficulty_column
        ):
            contract_errors.append(
                "performance requires an explicit difficulty-column mapping"
            )
        if (
            difficulty_column
            and difficulty_column in available_columns
            and formula
            and not re.search(
                rf"(?<![A-Za-z0-9_]){re.escape(difficulty_column)}(?![A-Za-z0-9_])",
                formula,
            )
        ):
            contract_errors.append(
                f"executable formula must use mapped difficulty column "
                f"{difficulty_column!r}"
            )
        missing_formula_columns = _missing_formula_columns(
            formula,
            available_columns,
        )
        if missing_formula_columns:
            contract_errors.append(
                "missing formula columns: "
                + ", ".join(missing_formula_columns)
            )
        required_base_columns = {
            participant_column,
            condition_column,
            response_column,
            difficulty_column,
        }
        missing_base_columns = sorted(
            column
            for column in required_base_columns
            if column and column not in available_columns
        )
        if missing_base_columns:
            contract_errors.append(
                "missing required columns: " + ", ".join(missing_base_columns)
            )

        if contract_errors:
            message = f"{comparison.condition_name}/{target_name}: " + "; ".join(
                _unique(contract_errors)
            )
            errors.append(message)
            records.append(
                _record(
                    comparison,
                    executable_formula=formula,
                    random_model=random_model,
                    difficulty_source_column=difficulty_column,
                    fit_error=message,
                )
            )
            continue

        condition_frame = data.loc[
            data[condition_column].eq(comparison.condition_name)
        ].copy(deep=True)
        before_ids = set(condition_frame[participant_column].dropna().tolist())

        eligible_frame = condition_frame
        if target_name == PERFORMANCE_TARGET:
            stage = pd.to_numeric(
                condition_frame[PERFORMANCE_STAGE_COLUMN],
                errors="coerce",
            )
            eligible_frame = condition_frame.loc[
                stage.notna() & stage.ne(0)
            ].copy(deep=True)

        formula_columns = _formula_columns(formula, available_columns)
        required_fit_columns = list(
            dict.fromkeys(
                [
                    participant_column,
                    response_column,
                    difficulty_column,
                    *formula_columns,
                ]
            )
        )
        fit_frame = eligible_frame.dropna(
            subset=required_fit_columns
        ).copy(deep=True)
        included_row_signature = _included_row_signature(fit_frame)
        fit_data = fit_frame.drop(
            columns=[_SOURCE_ROW_POSITION],
            errors="ignore",
        ).copy(deep=True)
        excluded_observation_count = len(condition_frame) - len(fit_frame)
        after_ids = set(fit_data[participant_column].dropna().tolist())
        excluded_participant_count = len(before_ids - after_ids)
        if excluded_observation_count:
            local_warnings.append(
                f"excluded {excluded_observation_count} rows from fit eligibility"
            )
        if not len(fit_data):
            message = (
                f"{comparison.condition_name}/{target_name}: no eligible "
                "observations remain"
            )
            errors.append(message)
            record = _record(
                comparison,
                executable_formula=formula,
                random_model=random_model,
                difficulty_source_column=difficulty_column,
                excluded_observation_count=excluded_observation_count,
                excluded_participant_count=excluded_participant_count,
                warnings_text=tuple(local_warnings),
                fit_error=message,
                included_row_signature=included_row_signature,
            )
            records.append(record)
            result_warnings.extend(local_warnings)
            continue
        if fit_data[participant_column].nunique(dropna=True) < 2:
            message = (
                f"{comparison.condition_name}/{target_name}: at least two "
                "participants are required for a random-intercept fit"
            )
            errors.append(message)
            record = _record(
                comparison,
                executable_formula=formula,
                random_model=random_model,
                difficulty_source_column=difficulty_column,
                participant_count=int(
                    fit_data[participant_column].nunique(dropna=True)
                ),
                observation_count=len(fit_data),
                excluded_observation_count=excluded_observation_count,
                excluded_participant_count=excluded_participant_count,
                warnings_text=tuple(local_warnings),
                fit_error=message,
                included_row_signature=included_row_signature,
            )
            records.append(record)
            result_warnings.extend(local_warnings)
            continue

        selected, fit_warnings, fit_errors = _fit_with_fallback(
            fit_data,
            formula,
            participant_column,
            tuple(optimizer_methods),
            maxiter,
        )
        all_warnings = _unique(local_warnings + list(fit_warnings))
        result_warnings.extend(all_warnings)
        fit_error = "; ".join(fit_errors) if fit_errors else None

        if selected is None:
            message = (
                f"{comparison.condition_name}/{target_name}: "
                f"{fit_error or 'all optimizer attempts failed'}"
            )
            errors.append(message)
            records.append(
                _record(
                    comparison,
                    executable_formula=formula,
                    random_model=random_model,
                    difficulty_source_column=difficulty_column,
                    participant_count=int(
                        fit_data[participant_column].nunique(dropna=True)
                    ),
                    observation_count=len(fit_data),
                    excluded_observation_count=excluded_observation_count,
                    excluded_participant_count=excluded_participant_count,
                    warnings_text=all_warnings,
                    fit_error=fit_error or message,
                    included_row_signature=included_row_signature,
                )
            )
            continue

        fitted = selected.result
        convergence_status = getattr(fitted, "converged", None)
        if convergence_status is not True and convergence_status is not False:
            convergence_status = None
        fit_status = "fitted" if convergence_status is not False else "non_converged"
        if fit_status == "non_converged":
            errors.append(
                f"{comparison.condition_name}/{target_name}: final optimizer "
                "returned a non-converged fit"
            )
        record = _record(
            comparison,
            executable_formula=formula,
            random_model=random_model,
            difficulty_source_column=difficulty_column,
            participant_count=int(
                fit_data[participant_column].nunique(dropna=True)
            ),
            observation_count=len(fit_data),
            excluded_observation_count=excluded_observation_count,
            excluded_participant_count=excluded_participant_count,
            fixed_effect_count=_safe_len(getattr(fitted, "fe_params", None)),
            total_parameter_count=_safe_len(getattr(fitted, "params", None)),
            log_likelihood=_safe_float(getattr(fitted, "llf", None)),
            aic=_safe_float(getattr(fitted, "aic", None)),
            bic=_safe_float(getattr(fitted, "bic", None)),
            convergence_status=convergence_status,
            optimizer=selected.optimizer,
            warnings_text=all_warnings,
            fit_status=fit_status,
            fit_error=fit_error,
            random_intercept_variance=_random_intercept_variance(fitted),
            boundary_flag=_backend_flag(
                fitted,
                ("boundary_flag", "is_boundary"),
            ),
            singularity_flag=_backend_flag(
                fitted,
                ("singularity_flag", "is_singular"),
            ),
            included_row_signature=included_row_signature,
        )
        records.append(record)

    frozen_records = tuple(records)
    fit_table = _table_from_records(frozen_records, registry)
    valid = not errors and all(
        record.fit_status == "fitted" for record in frozen_records
    )
    return RandomInterceptFitResult(
        valid=valid,
        errors=_unique(errors),
        warnings=_unique(result_warnings),
        fit_records=frozen_records,
        fit_table=fit_table,
    )
