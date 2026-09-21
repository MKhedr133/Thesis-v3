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