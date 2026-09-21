#!/usr/bin/env python3
"""MOD-13 — Difficulty-stage prediction data contract and model registry.

Batch 1 only.

This module defines:
- the frozen MOD-09 training/test boundary;
- the MOD-13 difficulty-stage target coding;
- the four frozen prediction model specifications;
- model-specific row eligibility;
- complete-row missing-value handling;
- final-test participant leakage protection.

Regression fitting and participant bootstrap validation are implemented in
later MOD-13 batches.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd

import MOD_10_training_random_effects_comparison as MOD10


PARTICIPANT_COLUMN = "participant_id"
CONDITION_COLUMN = "condition_name"
DIFFICULTY_LEVEL_COLUMN = "difficulty_level"
TARGET_COLUMN = "difficulty_stage"

DIFFICULTY_STAGE_BY_LEVEL = {
    0: 0,
    2: 1,
    6: 2,
    10: 3,
}

PERFORMANCE_CHANGE_PREDICTOR = (
    "performance_change_from_d0_percentage_points"
)


VISUAL_PREDICTORS = (
    "mental_demand_score_0_to_10",
    "median_time_between_qualifying_grabs_seconds",
    "total_list_recheck_duration_seconds",
    "median_time_to_target_seconds",
    "median_irrelevant_focus_duration_seconds",
    "median_head_turning_degrees",
    "median_reach_duration_seconds",
)

AUDITORY_PREDICTORS = (
    "mental_demand_score_0_to_10",
    "median_time_between_qualifying_grabs_seconds",
    "median_time_to_target_seconds",
    "median_head_turning_degrees",
)

COGNITIVE_PRIMARY_PREDICTORS = (
    "mental_demand_score_0_to_10",
    "median_time_between_qualifying_grabs_seconds",
    "list_recheck_count",
    "total_list_recheck_duration_seconds",
    "median_time_to_target_seconds",
    "median_irrelevant_focus_duration_seconds",
    "median_head_turning_degrees",
    "median_reach_duration_seconds",
    "median_reach_path_ratio",
)

COGNITIVE_LATER_PREDICTORS = (
    *COGNITIVE_PRIMARY_PREDICTORS,
    PERFORMANCE_CHANGE_PREDICTOR,
)


class Mod13Error(ValueError):
    """Raised when the frozen MOD-13 contract is violated."""


@dataclass(frozen=True)
class ModelSpec:
    """Frozen specification of one MOD-13 regression model."""

    model_id: str
    condition: str
    predictors: tuple[str, ...]
    allowed_stages: tuple[int, ...]


def build_model_registry() -> dict[str, ModelSpec]:
    """Return the four frozen MOD-13 model specifications."""
    return {
        "visual": ModelSpec(
            model_id="visual",
            condition="Visual",
            predictors=VISUAL_PREDICTORS,
            allowed_stages=(0, 1, 2, 3),
        ),
        "auditory": ModelSpec(
            model_id="auditory",
            condition="Auditory",
            predictors=AUDITORY_PREDICTORS,
            allowed_stages=(0, 1, 2, 3),
        ),
        "cognitive_primary": ModelSpec(
            model_id="cognitive_primary",
            condition="Cognitive",
            predictors=COGNITIVE_PRIMARY_PREDICTORS,
            allowed_stages=(0, 1, 2, 3),
        ),
        "cognitive_later": ModelSpec(
            model_id="cognitive_later",
            condition="Cognitive",
            predictors=COGNITIVE_LATER_PREDICTORS,
            allowed_stages=(1, 2, 3),
        ),
    }


def validate_holdout_split(
    holdout_split: pd.DataFrame,
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Validate and return the frozen 26/6 MOD-09 participant split."""
    try:
        return MOD10.validate_holdout_split(holdout_split)
    except ValueError as exc:
        raise Mod13Error(str(exc)) from exc


def prepare_training_data(
    *,
    modeling_data: pd.DataFrame,
    holdout_split: pd.DataFrame,
) -> tuple[
    pd.DataFrame,
    tuple[str, ...],
    tuple[str, ...],
]:
    """Return only the 26 MOD-09 training participants."""
    try:
        training_data, training_ids, test_ids = (
            MOD10.prepare_training_data(
                modeling_data=modeling_data,
                holdout_split=holdout_split,
            )
        )
    except ValueError as exc:
        raise Mod13Error(str(exc)) from exc

    observed_training_ids = set(
        training_data[PARTICIPANT_COLUMN]
    )

    if not observed_training_ids.isdisjoint(test_ids):
        raise Mod13Error(
            "Final-test participants entered MOD-13 development data"
        )

    return training_data, training_ids, test_ids


def _normalise_forbidden_ids(
    forbidden_participant_ids: Iterable[str] | None,
) -> set[str]:
    """Return clean participant IDs for explicit leakage checks."""
    if forbidden_participant_ids is None:
        return set()

    return {
        str(participant_id).strip()
        for participant_id in forbidden_participant_ids
        if str(participant_id).strip()
    }


def _validate_target_coding(
    condition_data: pd.DataFrame,
) -> pd.DataFrame:
    """Verify the frozen D0/D2/D6/D10 -> 0/1/2/3 coding."""
    required = (
        DIFFICULTY_LEVEL_COLUMN,
        TARGET_COLUMN,
    )

    missing_columns = [
        column
        for column in required
        if column not in condition_data.columns
    ]

    if missing_columns:
        raise Mod13Error(
            f"Missing required target columns: {missing_columns}"
        )

    validated = condition_data.copy()

    difficulty_level = pd.to_numeric(
        validated[DIFFICULTY_LEVEL_COLUMN],
        errors="coerce",
    )

    difficulty_stage = pd.to_numeric(
        validated[TARGET_COLUMN],
        errors="coerce",
    )

    if (
        difficulty_level.isna().any()
        or difficulty_stage.isna().any()
        or not np.isfinite(difficulty_level).all()
        or not np.isfinite(difficulty_stage).all()
    ):
        raise Mod13Error(
            "Difficulty level and difficulty stage must contain "
            "finite numeric values"
        )

    if not difficulty_level.isin(
        DIFFICULTY_STAGE_BY_LEVEL
    ).all():
        unexpected = sorted(
            set(difficulty_level)
            - set(DIFFICULTY_STAGE_BY_LEVEL)
        )

        raise Mod13Error(
            f"Unexpected difficulty levels: {unexpected}"
        )

    expected_stage = difficulty_level.map(
        DIFFICULTY_STAGE_BY_LEVEL
    )

    if not difficulty_stage.eq(expected_stage).all():
        raise Mod13Error(
            "difficulty_stage does not match frozen coding: "
            "D0=0, D2=1, D6=2, D10=3"
        )

    validated[DIFFICULTY_LEVEL_COLUMN] = (
        difficulty_level.astype(int)
    )

    validated[TARGET_COLUMN] = (
        difficulty_stage.astype(int)
    )

    return validated


def prepare_model_frame(
    data: pd.DataFrame,
    *,
    model_id: str,
    forbidden_participant_ids: Iterable[str] | None = None,
) -> pd.DataFrame:
    """Return complete eligible rows for one MOD-13 model.

    The returned frame contains only:
    - participant identity;
    - condition;
    - difficulty level;
    - prediction target;
    - predictors registered for that model.

    Missing values in required predictors remove only that trial row.
    No imputation or standardisation is performed.
    """
    registry = build_model_registry()

    if model_id not in registry:
        raise Mod13Error(
            f"Unknown MOD-13 model: {model_id}"
        )

    spec = registry[model_id]

    required_identity = (
        PARTICIPANT_COLUMN,
        CONDITION_COLUMN,
        DIFFICULTY_LEVEL_COLUMN,
        TARGET_COLUMN,
    )

    required_columns = (
        *required_identity,
        *spec.predictors,
    )

    missing_columns = [
        column
        for column in required_columns
        if column not in data.columns
    ]

    if missing_columns:
        raise Mod13Error(
            f"{model_id}: missing required columns: "
            f"{missing_columns}"
        )

    frame = data.copy(deep=True)

    if frame[PARTICIPANT_COLUMN].isna().any():
        raise Mod13Error(
            "participant_id contains missing values"
        )

    frame[PARTICIPANT_COLUMN] = (
        frame[PARTICIPANT_COLUMN]
        .astype(str)
        .str.strip()
    )

    if frame[PARTICIPANT_COLUMN].eq("").any():
        raise Mod13Error(
            "participant_id contains blank values"
        )

    # Explicit second protection against using the six final-test
    # participants during development.
    forbidden_ids = _normalise_forbidden_ids(
        forbidden_participant_ids
    )

    observed_ids = set(
        frame[PARTICIPANT_COLUMN]
    )

    leaked_ids = sorted(
        observed_ids & forbidden_ids
    )

    if leaked_ids:
        raise Mod13Error(
            "Final-test participants are forbidden in MOD-13 "
            "development frames: "
            + ", ".join(leaked_ids)
        )

    # Keep only the condition belonging to this model.
    frame = frame.loc[
        frame[CONDITION_COLUMN].eq(
            spec.condition
        )
    ].copy()

    if frame.empty:
        raise Mod13Error(
            f"{model_id}: no rows found for condition "
            f"{spec.condition}"
        )

    # Validate the target before filtering any difficulty stages.
    frame = _validate_target_coding(frame)

    # Primary models: 0,1,2,3.
    # Cognitive later: 1,2,3 only.
    frame = frame.loc[
        frame[TARGET_COLUMN].isin(
            spec.allowed_stages
        )
    ].copy()

    numeric_columns = (
        TARGET_COLUMN,
        *spec.predictors,
    )

    # Invalid numeric values become NaN and are handled by the
    # complete-row rule below.
    for column in numeric_columns:
        frame[column] = pd.to_numeric(
            frame[column],
            errors="coerce",
        )

    complete_mask = np.ones(
        len(frame),
        dtype=bool,
    )

    for column in numeric_columns:
        values = frame[column]

        complete_mask &= (
            values.notna().to_numpy()
        )

        complete_mask &= np.isfinite(
            values.to_numpy(dtype=float)
        )

    frame = frame.loc[
        complete_mask
    ].copy()

    if frame.empty:
        raise Mod13Error(
            f"{model_id}: no complete eligible rows remain"
        )

    output_columns = (
        PARTICIPANT_COLUMN,
        CONDITION_COLUMN,
        DIFFICULTY_LEVEL_COLUMN,
        TARGET_COLUMN,
        *spec.predictors,
    )

    return (
        frame.loc[:, output_columns]
        .reset_index(drop=True)
    )