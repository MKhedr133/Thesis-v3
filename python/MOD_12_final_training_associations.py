#!/usr/bin/env python3
"""MOD-12 — Final training-only difficulty association analysis."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

import MOD_10_training_random_effects_comparison as MOD10


MOD02 = MOD10.MOD02

CONDITIONS = tuple(MOD10.CONDITIONS)
TARGETS = tuple(MOD10.TARGETS)

PERFORMANCE_TARGET = (
    "performance_change_from_d0_percentage_points"
)

RANDOM_STRUCTURE = "RI"
ALPHA = 0.05


class Mod12Error(ValueError):
    """Raised when the MOD-12 analysis contract is violated."""


@dataclass(frozen=True)
class AssociationSpec:
    target: str
    difficulty_column: str
    formula: str


def build_target_specs() -> dict[str, AssociationSpec]:
    """Return the frozen M0 specification for all 10 measurements."""
    specs: dict[str, AssociationSpec] = {}

    for target in TARGETS:
        if target == PERFORMANCE_TARGET:
            difficulty_column = (
                MOD02.PERFORMANCE_DIFFICULTY_COLUMN
            )
        else:
            difficulty_column = "difficulty_stage"

        specs[target] = AssociationSpec(
            target=target,
            difficulty_column=difficulty_column,
            formula=f"{target} ~ {difficulty_column}",
        )

    return specs


def prepare_training_data(
    *,
    modeling_data: pd.DataFrame,
    holdout_split: pd.DataFrame,
) -> tuple[
    pd.DataFrame,
    tuple[str, ...],
    tuple[str, ...],
]:
    """Return only the locked 26 MOD-09 training participants."""
    return MOD10.prepare_training_data(
        modeling_data=modeling_data,
        holdout_split=holdout_split,
    )


def prepare_target_frame(
    training_data: pd.DataFrame,
    *,
    condition: str,
    target: str,
) -> pd.DataFrame:
    """Prepare valid observations for one condition and measurement."""
    if condition not in CONDITIONS:
        raise Mod12Error(
            f"Unknown condition: {condition}"
        )

    specs = build_target_specs()

    if target not in specs:
        raise Mod12Error(
            f"Unknown measurement: {target}"
        )

    spec = specs[target]

    data = MOD02.prepare_internal_modeling_data(
        training_data
    )

    frame = data.loc[
        data["condition_name"].eq(condition)
    ].copy()

    if target == PERFORMANCE_TARGET:
        frame = frame.loc[
            ~frame["difficulty_level"].eq(0)
        ].copy()

    required = [
        "participant_id",
        target,
        spec.difficulty_column,
    ]

    missing_columns = [
        column
        for column in required
        if column not in frame.columns
    ]

    if missing_columns:
        raise Mod12Error(
            f"Missing required columns: {missing_columns}"
        )

    frame[target] = pd.to_numeric(
        frame[target],
        errors="coerce",
    )

    frame[spec.difficulty_column] = pd.to_numeric(
        frame[spec.difficulty_column],
        errors="coerce",
    )

    valid = (
        frame["participant_id"].notna()
        & np.isfinite(frame[target])
        & np.isfinite(frame[spec.difficulty_column])
    )

    frame = frame.loc[valid].copy()

    if frame.empty:
        raise Mod12Error(
            f"{condition}/{target}: no valid observations"
        )

    return frame

def _empty_evidence(
    *,
    condition: str,
    target: str,
    frame: pd.DataFrame,
    convergence_status: str,
    optimizer: str,
    warnings_errors: str,
) -> dict[str, object]:
    """Return an unevaluable association record."""
    return {
        "condition": condition,
        "measurement": target,
        "difficulty_coefficient": np.nan,
        "standard_error": np.nan,
        "ci_95_lower": np.nan,
        "ci_95_upper": np.nan,
        "p_value": np.nan,
        "participant_count": int(
            frame["participant_id"].nunique()
        ),
        "observation_count": int(len(frame)),
        "convergence_status": convergence_status,
        "optimizer": optimizer,
        "warnings_errors": warnings_errors,
        "retention_status": "not_evaluable",
    }


def fit_association(
    training_data: pd.DataFrame,
    *,
    condition: str,
    target: str,
) -> dict[str, object]:
    """Fit one final REML association and extract Difficulty inference."""
    specs = build_target_specs()

    if target not in specs:
        raise Mod12Error(
            f"Unknown measurement: {target}"
        )

    spec = specs[target]

    frame = prepare_target_frame(
        training_data,
        condition=condition,
        target=target,
    )

    result, optimizer, fit_warnings, fit_errors = (
        MOD02.fit_mixedlm_with_fallback(
            spec.formula,
            frame,
            RANDOM_STRUCTURE,
            spec.difficulty_column,
            reml=True,
        )
    )

    messages = tuple(fit_warnings) + tuple(fit_errors)

    warnings_errors = " | ".join(
        str(message)
        for message in messages
        if str(message)
    )

    if result is None:
        return _empty_evidence(
            condition=condition,
            target=target,
            frame=frame,
            convergence_status="failed",
            optimizer=optimizer,
            warnings_errors=warnings_errors,
        )

    if not bool(
        getattr(result, "converged", False)
    ):
        return _empty_evidence(
            condition=condition,
            target=target,
            frame=frame,
            convergence_status="non_converged",
            optimizer=optimizer,
            warnings_errors=warnings_errors,
        )

    difficulty_column = spec.difficulty_column

    try:
        coefficient = float(
            result.fe_params[difficulty_column]
        )

        standard_error = float(
            result.bse[difficulty_column]
        )

        p_value = float(
            result.pvalues[difficulty_column]
        )

        confidence_interval = (
            result.conf_int().loc[difficulty_column]
        )

        ci_lower = float(
            confidence_interval.iloc[0]
        )

        ci_upper = float(
            confidence_interval.iloc[1]
        )

    except (KeyError, IndexError, TypeError, ValueError) as exc:
        raise Mod12Error(
            f"{condition}/{target}: "
            "could not extract Difficulty inference"
        ) from exc

    values = (
        coefficient,
        standard_error,
        p_value,
        ci_lower,
        ci_upper,
    )

    if not all(
        np.isfinite(value)
        for value in values
    ):
        raise Mod12Error(
            f"{condition}/{target}: "
            "Difficulty inference contains non-finite values"
        )

    retention_status = (
        "retained"
        if p_value < ALPHA
        else "not_retained"
    )

    return {
        "condition": condition,
        "measurement": target,
        "difficulty_coefficient": coefficient,
        "standard_error": standard_error,
        "ci_95_lower": ci_lower,
        "ci_95_upper": ci_upper,
        "p_value": p_value,
        "participant_count": int(
            frame["participant_id"].nunique()
        ),
        "observation_count": int(len(frame)),
        "convergence_status": "converged",
        "optimizer": optimizer,
        "warnings_errors": warnings_errors,
        "retention_status": retention_status,
    }