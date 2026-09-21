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

import argparse
from pathlib import Path

import MOD_10_training_random_effects_comparison as MOD10

import statsmodels.api as sm


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

BOOTSTRAP_REPETITIONS = 500
BOOTSTRAP_SEED = 20260921

INTERCEPT_NAME = "Intercept"

COEFFICIENTS_FILENAME = "regression_coefficients.csv"
BOOTSTRAP_PERFORMANCE_FILENAME = "bootstrap_performance.csv"

FINAL_TEST_PERFORMANCE_FILENAME = "final_test_performance.csv"
FINAL_TEST_PREDICTIONS_FILENAME = "final_test_predictions.csv"

class Mod13Error(ValueError):
    """Raised when the frozen MOD-13 contract is violated."""


@dataclass(frozen=True)
class ModelSpec:
    """Frozen specification of one MOD-13 regression model."""

    model_id: str
    condition: str
    predictors: tuple[str, ...]
    allowed_stages: tuple[int, ...]

@dataclass(frozen=True)
class LinearModelFit:
    """One fitted MOD-13 multiple linear regression model."""

    model_id: str
    predictors: tuple[str, ...]
    coefficients: pd.Series

    def predict(
        self,
        frame: pd.DataFrame,
    ) -> pd.Series:
        """Generate raw continuous difficulty-stage predictions."""
        design = _build_design_matrix(
            frame,
            predictors=self.predictors,
        )

        coefficient_vector = self.coefficients.loc[
            design.columns
        ].to_numpy(dtype=float)

        predictions = (
            design.to_numpy(dtype=float)
            @ coefficient_vector
        )

        return pd.Series(
            predictions,
            index=frame.index,
            name="predicted_difficulty_stage",
        )


@dataclass(frozen=True)
class BootstrapResult:
    """Participant-bootstrap evidence for one MOD-13 model."""

    model_id: str
    attempted_repetitions: int
    metrics: pd.DataFrame
    coefficients: pd.DataFrame


@dataclass(frozen=True)
class ModelDevelopmentResult:
    """Internal development evidence for one MOD-13 model."""

    model_id: str
    model_frame: pd.DataFrame
    bootstrap: BootstrapResult
    final_fit: LinearModelFit
    coefficient_summary: pd.DataFrame
    performance_summary: pd.DataFrame

@dataclass(frozen=True)
class DevelopmentOutputs:
    """Concise output tables from training-only MOD-13 development."""

    coefficients: pd.DataFrame
    bootstrap_performance: pd.DataFrame


@dataclass(frozen=True)
class FinalTestResult:
    """Frozen-model evaluation on the six final-test participants."""

    performance: pd.DataFrame
    predictions: pd.DataFrame

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


def _build_design_matrix(
    frame: pd.DataFrame,
    *,
    predictors: tuple[str, ...],
) -> pd.DataFrame:
    """Build an unscaled regression design matrix with an intercept."""
    missing = [
        predictor
        for predictor in predictors
        if predictor not in frame.columns
    ]

    if missing:
        raise Mod13Error(
            f"Missing regression predictors: {missing}"
        )

    design = frame.loc[
        :,
        list(predictors),
    ].copy()

    for predictor in predictors:
        design[predictor] = pd.to_numeric(
            design[predictor],
            errors="coerce",
        )

    values = design.to_numpy(dtype=float)

    if not np.isfinite(values).all():
        raise Mod13Error(
            "Regression design matrix contains missing "
            "or non-finite values"
        )

    design.insert(
        0,
        INTERCEPT_NAME,
        1.0,
    )

    return design


def fit_linear_model(
    frame: pd.DataFrame,
    *,
    model_id: str,
) -> LinearModelFit:
    """Fit one multiple linear regression using ordinary least squares."""
    registry = build_model_registry()

    if model_id not in registry:
        raise Mod13Error(
            f"Unknown MOD-13 model: {model_id}"
        )

    spec = registry[model_id]

    if TARGET_COLUMN not in frame.columns:
        raise Mod13Error(
            f"{model_id}: missing target column "
            f"{TARGET_COLUMN}"
        )

    design = _build_design_matrix(
        frame,
        predictors=spec.predictors,
    )

    target = pd.to_numeric(
        frame[TARGET_COLUMN],
        errors="coerce",
    ).to_numpy(dtype=float)

    if not np.isfinite(target).all():
        raise Mod13Error(
            f"{model_id}: target contains missing "
            "or non-finite values"
        )

    observation_count = len(frame)
    parameter_count = design.shape[1]

    if observation_count <= parameter_count:
        raise Mod13Error(
            f"{model_id}: insufficient observations "
            f"({observation_count}) for "
            f"{parameter_count} regression parameters"
        )

    design_values = design.to_numpy(
        dtype=float
    )

    rank = np.linalg.matrix_rank(
        design_values
    )

    if rank < parameter_count:
        raise Mod13Error(
            f"{model_id}: regression design matrix "
            f"is rank deficient "
            f"(rank={rank}, parameters={parameter_count})"
        )

    try:
        fitted = sm.OLS(
            target,
            design,
        ).fit()
    except Exception as exc:
        raise Mod13Error(
            f"{model_id}: OLS fitting failed"
        ) from exc

    coefficients = fitted.params.astype(
        float
    )

    if not np.isfinite(
        coefficients.to_numpy()
    ).all():
        raise Mod13Error(
            f"{model_id}: fitted coefficients "
            "contain non-finite values"
        )

    return LinearModelFit(
        model_id=model_id,
        predictors=spec.predictors,
        coefficients=coefficients,
    )


def participant_balanced_mae(
    *,
    participant_ids: pd.Series,
    observed: pd.Series,
    predicted: pd.Series,
) -> float:
    """Calculate MAE giving each participant equal weight."""
    if not (
        len(participant_ids)
        == len(observed)
        == len(predicted)
    ):
        raise Mod13Error(
            "Participant IDs, observations, and predictions "
            "must have equal length"
        )

    evaluation = pd.DataFrame(
        {
            PARTICIPANT_COLUMN: (
                participant_ids.astype(str).to_numpy()
            ),
            "observed": np.asarray(
                observed,
                dtype=float,
            ),
            "predicted": np.asarray(
                predicted,
                dtype=float,
            ),
        }
    )

    if not np.isfinite(
        evaluation[
            ["observed", "predicted"]
        ].to_numpy()
    ).all():
        raise Mod13Error(
            "MAE inputs contain non-finite values"
        )

    evaluation["absolute_error"] = (
        evaluation["observed"]
        - evaluation["predicted"]
    ).abs()

    participant_mae = (
        evaluation
        .groupby(
            PARTICIPANT_COLUMN,
            sort=False,
        )["absolute_error"]
        .mean()
    )

    if participant_mae.empty:
        raise Mod13Error(
            "Cannot calculate MAE without participants"
        )

    return float(
        participant_mae.mean()
    )


def pooled_r2(
    *,
    observed: pd.Series,
    predicted: pd.Series,
) -> float:
    """Calculate ordinary pooled R² for held-out observations."""
    y_true = np.asarray(
        observed,
        dtype=float,
    )

    y_pred = np.asarray(
        predicted,
        dtype=float,
    )

    if len(y_true) != len(y_pred):
        raise Mod13Error(
            "Observed and predicted values must "
            "have equal length"
        )

    if len(y_true) == 0:
        raise Mod13Error(
            "Cannot calculate R² with no observations"
        )

    if (
        not np.isfinite(y_true).all()
        or not np.isfinite(y_pred).all()
    ):
        raise Mod13Error(
            "R² inputs contain non-finite values"
        )

    residual_sum_squares = float(
        np.sum(
            (y_true - y_pred) ** 2
        )
    )

    total_sum_squares = float(
        np.sum(
            (
                y_true
                - np.mean(y_true)
            ) ** 2
        )
    )

    if total_sum_squares == 0.0:
        return np.nan

    return float(
        1.0
        - residual_sum_squares
        / total_sum_squares
    )


def draw_participant_bootstrap(
    frame: pd.DataFrame,
    *,
    rng: np.random.Generator,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    tuple[str, ...],
]:
    """Draw one participant-level bootstrap sample.

    Each selected participant contributes all of their eligible rows.
    Participants selected multiple times contribute repeated copies of
    all their rows.

    Participants never selected form the out-of-bag set.
    """
    participant_ids = tuple(
        sorted(
            frame[
                PARTICIPANT_COLUMN
            ]
            .astype(str)
            .unique()
            .tolist()
        )
    )

    if len(participant_ids) < 2:
        raise Mod13Error(
            "Participant bootstrap requires at least "
            "two participants"
        )

    sampled_array = rng.choice(
        np.asarray(
            participant_ids,
            dtype=object,
        ),
        size=len(participant_ids),
        replace=True,
    )

    sampled_ids = tuple(
        str(value)
        for value in sampled_array.tolist()
    )

    inbag_parts: list[pd.DataFrame] = []

    for participant_id in sampled_ids:
        participant_rows = frame.loc[
            frame[
                PARTICIPANT_COLUMN
            ].astype(str).eq(
                participant_id
            )
        ].copy()

        inbag_parts.append(
            participant_rows
        )

    inbag = pd.concat(
        inbag_parts,
        ignore_index=True,
    )

    sampled_unique = set(
        sampled_ids
    )

    oob_ids = tuple(
        participant_id
        for participant_id in participant_ids
        if participant_id
        not in sampled_unique
    )

    oob = frame.loc[
        frame[
            PARTICIPANT_COLUMN
        ].astype(str).isin(
            oob_ids
        )
    ].copy()

    return (
        inbag,
        oob.reset_index(drop=True),
        sampled_ids,
    )


def run_participant_bootstrap(
    frame: pd.DataFrame,
    *,
    model_id: str,
    repetitions: int = BOOTSTRAP_REPETITIONS,
    seed: int = BOOTSTRAP_SEED,
) -> BootstrapResult:
    """Run participant-level bootstrap with OOB evaluation."""
    if repetitions <= 0:
        raise Mod13Error(
            "Bootstrap repetitions must be positive"
        )

    rng = np.random.default_rng(
        seed
    )

    metric_rows: list[
        dict[str, object]
    ] = []

    coefficient_rows: list[
        dict[str, object]
    ] = []

    for bootstrap_rep in range(
        1,
        repetitions + 1,
    ):
        inbag, oob, sampled_ids = (
            draw_participant_bootstrap(
                frame,
                rng=rng,
            )
        )

        inbag_unique_count = len(
            set(sampled_ids)
        )

        oob_participant_count = int(
            oob[
                PARTICIPANT_COLUMN
            ].nunique()
        )

        oob_row_count = int(
            len(oob)
        )

        if oob.empty:
            metric_rows.append(
                {
                    "model": model_id,
                    "bootstrap_rep": (
                        bootstrap_rep
                    ),
                    "status": "no_oob",
                    (
                        "inbag_unique_"
                        "participant_count"
                    ): inbag_unique_count,
                    (
                        "oob_participant_count"
                    ): 0,
                    "oob_row_count": 0,
                    "mae": np.nan,
                    "r2": np.nan,
                    "error": (
                        "No out-of-bag "
                        "participants"
                    ),
                }
            )

            continue

        try:
            fitted = fit_linear_model(
                inbag,
                model_id=model_id,
            )

            predictions = fitted.predict(
                oob
            )

            mae = participant_balanced_mae(
                participant_ids=(
                    oob[
                        PARTICIPANT_COLUMN
                    ]
                ),
                observed=(
                    oob[TARGET_COLUMN]
                ),
                predicted=predictions,
            )

            r2 = pooled_r2(
                observed=(
                    oob[TARGET_COLUMN]
                ),
                predicted=predictions,
            )

            if not np.isfinite(mae):
                raise Mod13Error(
                    "Bootstrap MAE is non-finite"
                )

            if not np.isfinite(r2):
                raise Mod13Error(
                    "Bootstrap R² is non-finite"
                )

        except Mod13Error as exc:
            metric_rows.append(
                {
                    "model": model_id,
                    "bootstrap_rep": (
                        bootstrap_rep
                    ),
                    "status": "failed",
                    (
                        "inbag_unique_"
                        "participant_count"
                    ): inbag_unique_count,
                    (
                        "oob_participant_count"
                    ): (
                        oob_participant_count
                    ),
                    "oob_row_count": (
                        oob_row_count
                    ),
                    "mae": np.nan,
                    "r2": np.nan,
                    "error": str(exc),
                }
            )

            continue

        metric_rows.append(
            {
                "model": model_id,
                "bootstrap_rep": (
                    bootstrap_rep
                ),
                "status": "success",
                (
                    "inbag_unique_"
                    "participant_count"
                ): inbag_unique_count,
                (
                    "oob_participant_count"
                ): (
                    oob_participant_count
                ),
                "oob_row_count": (
                    oob_row_count
                ),
                "mae": mae,
                "r2": r2,
                "error": "",
            }
        )

        for (
            predictor,
            coefficient,
        ) in fitted.coefficients.items():

            coefficient_rows.append(
                {
                    "model": model_id,
                    "bootstrap_rep": (
                        bootstrap_rep
                    ),
                    "predictor": predictor,
                    "coefficient": float(
                        coefficient
                    ),
                }
            )

    metrics = pd.DataFrame(
        metric_rows
    )

    coefficients = pd.DataFrame(
        coefficient_rows,
        columns=(
            "model",
            "bootstrap_rep",
            "predictor",
            "coefficient",
        ),
    )

    return BootstrapResult(
        model_id=model_id,
        attempted_repetitions=(
            repetitions
        ),
        metrics=metrics,
        coefficients=coefficients,
    )


def summarize_bootstrap_performance(
    bootstrap: BootstrapResult,
) -> pd.DataFrame:
    """Summarise successful OOB bootstrap performance."""
    successful = bootstrap.metrics.loc[
        bootstrap.metrics[
            "status"
        ].eq("success")
    ].copy()

    if successful.empty:
        raise Mod13Error(
            f"{bootstrap.model_id}: "
            "no successful bootstrap repetitions"
        )

    mae = successful[
        "mae"
    ].to_numpy(dtype=float)

    r2 = successful[
        "r2"
    ].to_numpy(dtype=float)

    return pd.DataFrame(
        [
            {
                "model": (
                    bootstrap.model_id
                ),
                (
                    "bootstrap_repetitions"
                ): (
                    bootstrap
                    .attempted_repetitions
                ),
                (
                    "usable_bootstrap_"
                    "repetitions"
                ): int(
                    len(successful)
                ),
                "oob_mae": float(
                    np.mean(mae)
                ),
                (
                    "mae_ci_95_lower"
                ): float(
                    np.percentile(
                        mae,
                        2.5,
                    )
                ),
                (
                    "mae_ci_95_upper"
                ): float(
                    np.percentile(
                        mae,
                        97.5,
                    )
                ),
                "oob_r2": float(
                    np.mean(r2)
                ),
                (
                    "r2_ci_95_lower"
                ): float(
                    np.percentile(
                        r2,
                        2.5,
                    )
                ),
                (
                    "r2_ci_95_upper"
                ): float(
                    np.percentile(
                        r2,
                        97.5,
                    )
                ),
            }
        ]
    )


def build_coefficient_summary(
    *,
    final_fit: LinearModelFit,
    bootstrap: BootstrapResult,
) -> pd.DataFrame:
    """Combine final coefficients with bootstrap percentile CIs."""
    if (
        final_fit.model_id
        != bootstrap.model_id
    ):
        raise Mod13Error(
            "Final fit and bootstrap model IDs "
            "do not match"
        )

    successful_repetitions = (
        bootstrap.metrics.loc[
            bootstrap.metrics[
                "status"
            ].eq("success"),
            "bootstrap_rep",
        ]
        .nunique()
    )

    if successful_repetitions == 0:
        raise Mod13Error(
            f"{bootstrap.model_id}: "
            "no successful bootstrap coefficients"
        )

    rows: list[
        dict[str, object]
    ] = []

    for (
        predictor,
        final_coefficient,
    ) in final_fit.coefficients.items():

        draws = bootstrap.coefficients.loc[
            bootstrap.coefficients[
                "predictor"
            ].eq(predictor),
            "coefficient",
        ].to_numpy(
            dtype=float
        )

        if len(draws) == 0:
            raise Mod13Error(
                f"{bootstrap.model_id}/"
                f"{predictor}: "
                "missing bootstrap coefficients"
            )

        rows.append(
            {
                "model": (
                    final_fit.model_id
                ),
                "predictor": predictor,
                "coefficient": float(
                    final_coefficient
                ),
                (
                    "bootstrap_ci_95_lower"
                ): float(
                    np.percentile(
                        draws,
                        2.5,
                    )
                ),
                (
                    "bootstrap_ci_95_upper"
                ): float(
                    np.percentile(
                        draws,
                        97.5,
                    )
                ),
                (
                    "usable_bootstrap_"
                    "repetitions"
                ): int(
                    len(draws)
                ),
            }
        )

    return pd.DataFrame(
        rows
    )


def develop_model(
    training_data: pd.DataFrame,
    *,
    model_id: str,
    forbidden_participant_ids: Iterable[str],
    bootstrap_repetitions: int = (
        BOOTSTRAP_REPETITIONS
    ),
    bootstrap_seed: int = (
        BOOTSTRAP_SEED
    ),
) -> ModelDevelopmentResult:
    """Run training-only MOD-13 development for one model.

    Order:
    1. prepare eligible training rows;
    2. perform participant bootstrap and OOB validation;
    3. fit one final regression using all eligible training rows;
    4. summarise coefficient and prediction evidence.

    No final-test observations are used.
    """
    model_frame = prepare_model_frame(
        training_data,
        model_id=model_id,
        forbidden_participant_ids=(
            forbidden_participant_ids
        ),
    )

    bootstrap = (
        run_participant_bootstrap(
            model_frame,
            model_id=model_id,
            repetitions=(
                bootstrap_repetitions
            ),
            seed=bootstrap_seed,
        )
    )

    final_fit = fit_linear_model(
        model_frame,
        model_id=model_id,
    )

    coefficient_summary = (
        build_coefficient_summary(
            final_fit=final_fit,
            bootstrap=bootstrap,
        )
    )

    performance_summary = (
        summarize_bootstrap_performance(
            bootstrap
        )
    )

    return ModelDevelopmentResult(
        model_id=model_id,
        model_frame=model_frame,
        bootstrap=bootstrap,
        final_fit=final_fit,
        coefficient_summary=(
            coefficient_summary
        ),
        performance_summary=(
            performance_summary
        ),
    )

def develop_all_models(
    training_data: pd.DataFrame,
    *,
    forbidden_participant_ids: Iterable[str],
    bootstrap_repetitions: int = BOOTSTRAP_REPETITIONS,
    bootstrap_seed: int = BOOTSTRAP_SEED,
) -> dict[str, ModelDevelopmentResult]:
    """Develop all four frozen MOD-13 models using training data only."""
    results: dict[
        str,
        ModelDevelopmentResult,
    ] = {}

    for model_id in build_model_registry():
        results[model_id] = develop_model(
            training_data,
            model_id=model_id,
            forbidden_participant_ids=(
                forbidden_participant_ids
            ),
            bootstrap_repetitions=(
                bootstrap_repetitions
            ),
            bootstrap_seed=(
                bootstrap_seed
            ),
        )

    return results


def build_development_outputs(
    results: dict[
        str,
        ModelDevelopmentResult,
    ],
) -> DevelopmentOutputs:
    """Combine all training-only MOD-13 development evidence."""
    expected_models = tuple(
        build_model_registry()
    )

    if tuple(results) != expected_models:
        raise Mod13Error(
            "Development results do not contain the "
            "four frozen MOD-13 models in the expected order"
        )

    coefficient_parts: list[
        pd.DataFrame
    ] = []

    performance_parts: list[
        pd.DataFrame
    ] = []

    for model_id in expected_models:
        result = results[model_id]

        coefficients = (
            result
            .coefficient_summary
            .copy()
        )

        coefficient_parts.append(
            coefficients
        )

        performance = (
            result
            .performance_summary
            .copy()
        )

        participant_count = int(
            result
            .model_frame[
                PARTICIPANT_COLUMN
            ]
            .nunique()
        )

        eligible_row_count = int(
            len(result.model_frame)
        )

        performance.insert(
            1,
            "participant_count",
            participant_count,
        )

        performance.insert(
            2,
            "eligible_row_count",
            eligible_row_count,
        )

        performance_parts.append(
            performance
        )

    coefficient_table = pd.concat(
        coefficient_parts,
        ignore_index=True,
    )

    performance_table = pd.concat(
        performance_parts,
        ignore_index=True,
    )

    coefficient_columns = (
        "model",
        "predictor",
        "coefficient",
        "bootstrap_ci_95_lower",
        "bootstrap_ci_95_upper",
        "usable_bootstrap_repetitions",
    )

    performance_columns = (
        "model",
        "participant_count",
        "eligible_row_count",
        "bootstrap_repetitions",
        "usable_bootstrap_repetitions",
        "oob_mae",
        "mae_ci_95_lower",
        "mae_ci_95_upper",
        "oob_r2",
        "r2_ci_95_lower",
        "r2_ci_95_upper",
    )

    return DevelopmentOutputs(
        coefficients=(
            coefficient_table.loc[
                :,
                coefficient_columns,
            ]
        ),
        bootstrap_performance=(
            performance_table.loc[
                :,
                performance_columns,
            ]
        ),
    )


def write_development_outputs(
    outputs: DevelopmentOutputs,
    *,
    output_dir: Path,
) -> tuple[Path, Path]:
    """Write the two frozen training-development output files."""
    output_dir = Path(
        output_dir
    )

    coefficient_path = (
        output_dir
        / COEFFICIENTS_FILENAME
    )

    performance_path = (
        output_dir
        / BOOTSTRAP_PERFORMANCE_FILENAME
    )

    existing = [
        path
        for path in (
            coefficient_path,
            performance_path,
        )
        if path.exists()
    ]

    if existing:
        raise Mod13Error(
            "Refusing to overwrite existing MOD-13 "
            "development outputs: "
            + ", ".join(
                str(path)
                for path in existing
            )
        )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    outputs.coefficients.to_csv(
        coefficient_path,
        index=False,
        encoding="utf-8",
    )

    outputs.bootstrap_performance.to_csv(
        performance_path,
        index=False,
        encoding="utf-8",
    )

    return (
        coefficient_path,
        performance_path,
    )

def prepare_final_test_data(
    *,
    modeling_data: pd.DataFrame,
    holdout_split: pd.DataFrame,
) -> tuple[
    pd.DataFrame,
    tuple[str, ...],
]:
    """Return only the six frozen MOD-09 final-test participants."""
    training_ids, test_ids = (
        validate_holdout_split(
            holdout_split
        )
    )

    if PARTICIPANT_COLUMN not in modeling_data.columns:
        raise Mod13Error(
            f"Missing required column: "
            f"{PARTICIPANT_COLUMN}"
        )

    data = modeling_data.copy(
        deep=True
    )

    if data[
        PARTICIPANT_COLUMN
    ].isna().any():
        raise Mod13Error(
            "participant_id contains missing values"
        )

    data[PARTICIPANT_COLUMN] = (
        data[
            PARTICIPANT_COLUMN
        ]
        .astype(str)
        .str.strip()
    )

    if data[
        PARTICIPANT_COLUMN
    ].eq("").any():
        raise Mod13Error(
            "participant_id contains blank values"
        )

    modeling_ids = set(
        data[PARTICIPANT_COLUMN]
    )

    expected_ids = (
        set(training_ids)
        | set(test_ids)
    )

    missing_ids = sorted(
        expected_ids
        - modeling_ids
    )

    unexpected_ids = sorted(
        modeling_ids
        - expected_ids
    )

    if missing_ids:
        raise Mod13Error(
            "Participants from the frozen split are "
            "missing from modeling data: "
            + ", ".join(missing_ids)
        )

    if unexpected_ids:
        raise Mod13Error(
            "Modeling data contain participants not "
            "present in the frozen split: "
            + ", ".join(unexpected_ids)
        )

    test_data = data.loc[
        data[
            PARTICIPANT_COLUMN
        ].isin(test_ids)
    ].copy()

    observed_test_ids = set(
        test_data[
            PARTICIPANT_COLUMN
        ]
    )

    if observed_test_ids != set(
        test_ids
    ):
        raise Mod13Error(
            "Final-test data do not contain exactly "
            "the six frozen test participants"
        )

    if not observed_test_ids.isdisjoint(
        training_ids
    ):
        raise Mod13Error(
            "Training participants entered the "
            "final-test dataset"
        )

    if test_data[
        PARTICIPANT_COLUMN
    ].nunique() != 6:
        raise Mod13Error(
            "Final-test dataset must contain exactly "
            "six participants"
        )

    return (
        test_data.reset_index(
            drop=True
        ),
        test_ids,
    )

def load_frozen_models(
    coefficient_table: pd.DataFrame,
) -> dict[
    str,
    LinearModelFit,
]:
    """Reconstruct the frozen final models from development coefficients."""
    required_columns = {
        "model",
        "predictor",
        "coefficient",
    }

    missing_columns = (
        required_columns
        - set(
            coefficient_table.columns
        )
    )

    if missing_columns:
        raise Mod13Error(
            "Coefficient table is missing required columns: "
            + ", ".join(
                sorted(
                    missing_columns
                )
            )
        )

    registry = (
        build_model_registry()
    )

    observed_models = set(
        coefficient_table[
            "model"
        ].astype(str)
    )

    expected_models = set(
        registry
    )

    if observed_models != expected_models:
        raise Mod13Error(
            "Frozen coefficient table must contain "
            "exactly the four MOD-13 models"
        )

    frozen_models: dict[
        str,
        LinearModelFit,
    ] = {}

    for model_id, spec in registry.items():
        model_rows = (
            coefficient_table.loc[
                coefficient_table[
                    "model"
                ].astype(str).eq(
                    model_id
                )
            ]
            .copy()
        )

        if model_rows[
            "predictor"
        ].duplicated().any():
            raise Mod13Error(
                f"{model_id}: duplicate coefficient terms"
            )

        expected_terms = (
            INTERCEPT_NAME,
            *spec.predictors,
        )

        observed_terms = set(
            model_rows[
                "predictor"
            ].astype(str)
        )

        if observed_terms != set(
            expected_terms
        ):
            missing_terms = sorted(
                set(expected_terms)
                - observed_terms
            )

            unexpected_terms = sorted(
                observed_terms
                - set(expected_terms)
            )

            raise Mod13Error(
                f"{model_id}: coefficient terms do not "
                "match the frozen model. "
                f"Missing={missing_terms}; "
                f"unexpected={unexpected_terms}"
            )

        indexed = (
            model_rows
            .set_index(
                "predictor"
            )
        )

        coefficients = pd.Series(
            {
                term: float(
                    indexed.loc[
                        term,
                        "coefficient",
                    ]
                )
                for term in expected_terms
            },
            dtype=float,
        )

        if not np.isfinite(
            coefficients.to_numpy()
        ).all():
            raise Mod13Error(
                f"{model_id}: frozen coefficients "
                "contain non-finite values"
            )

        frozen_models[
            model_id
        ] = LinearModelFit(
            model_id=model_id,
            predictors=spec.predictors,
            coefficients=coefficients,
        )

    return frozen_models


def evaluate_final_test_model(
    test_data: pd.DataFrame,
    *,
    model_id: str,
    frozen_model: LinearModelFit,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
]:
    """Evaluate one already-fitted model on final-test participants."""
    if frozen_model.model_id != model_id:
        raise Mod13Error(
            "Frozen model ID does not match "
            "requested final-test model"
        )

    frame = prepare_model_frame(
        test_data,
        model_id=model_id,
    )

    predictions = frozen_model.predict(
        frame
    )

    observed = (
        frame[TARGET_COLUMN]
        .astype(float)
    )

    mae = participant_balanced_mae(
        participant_ids=(
            frame[PARTICIPANT_COLUMN]
        ),
        observed=observed,
        predicted=predictions,
    )

    r2 = pooled_r2(
        observed=observed,
        predicted=predictions,
    )

    error = (
        observed.to_numpy(dtype=float)
        - predictions.to_numpy(dtype=float)
    )

    absolute_error = np.abs(
        error
    )

    prediction_values = (
        predictions.to_numpy(
            dtype=float
        )
    )

    out_of_range = (
        (prediction_values < 0.0)
        | (prediction_values > 3.0)
    )

    performance = pd.DataFrame(
        [
            {
                "model": model_id,
                "participant_count": int(
                    frame[
                        PARTICIPANT_COLUMN
                    ].nunique()
                ),
                "eligible_row_count": int(
                    len(frame)
                ),
                "mae": float(mae),
                "r2": float(r2),
                (
                    "out_of_range_"
                    "prediction_count"
                ): int(
                    np.sum(
                        out_of_range
                    )
                ),
            }
        ]
    )

    predictions_table = pd.DataFrame(
        {
            "participant_id": (
                frame[
                    PARTICIPANT_COLUMN
                ].to_numpy()
            ),
            "model": model_id,
            "condition": (
                frame[
                    CONDITION_COLUMN
                ].to_numpy()
            ),
            "observed_difficulty_stage": (
                observed.to_numpy(
                    dtype=float
                )
            ),
            "predicted_difficulty_stage": (
                prediction_values
            ),
            "error": error,
            "absolute_error": (
                absolute_error
            ),
        }
    )

    return (
        performance,
        predictions_table,
    )


def run_final_test(
    test_data: pd.DataFrame,
    *,
    coefficient_table: pd.DataFrame,
) -> FinalTestResult:
    """Evaluate frozen MOD-13 equations without any refitting."""
    frozen_models = (
        load_frozen_models(
            coefficient_table
        )
    )

    performance_parts: list[
        pd.DataFrame
    ] = []

    prediction_parts: list[
        pd.DataFrame
    ] = []

    for model_id in build_model_registry():
        performance, predictions = (
            evaluate_final_test_model(
                test_data,
                model_id=model_id,
                frozen_model=(
                    frozen_models[
                        model_id
                    ]
                ),
            )
        )

        performance_parts.append(
            performance
        )

        prediction_parts.append(
            predictions
        )

    return FinalTestResult(
        performance=pd.concat(
            performance_parts,
            ignore_index=True,
        ),
        predictions=pd.concat(
            prediction_parts,
            ignore_index=True,
        ),
    )


def write_final_test_outputs(
    result: FinalTestResult,
    *,
    output_dir: Path,
) -> tuple[Path, Path]:
    """Write final-test results while refusing accidental overwrite."""
    output_dir = Path(
        output_dir
    )

    performance_path = (
        output_dir
        / FINAL_TEST_PERFORMANCE_FILENAME
    )

    predictions_path = (
        output_dir
        / FINAL_TEST_PREDICTIONS_FILENAME
    )

    existing = [
        path
        for path in (
            performance_path,
            predictions_path,
        )
        if path.exists()
    ]

    if existing:
        raise Mod13Error(
            "Refusing to overwrite existing MOD-13 "
            "final-test outputs: "
            + ", ".join(
                str(path)
                for path in existing
            )
        )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    result.performance.to_csv(
        performance_path,
        index=False,
        encoding="utf-8",
    )

    result.predictions.to_csv(
        predictions_path,
        index=False,
        encoding="utf-8",
    )

    return (
        performance_path,
        predictions_path,
    )

def run_development_cli(
    *,
    modeling_data_path: Path,
    holdout_split_path: Path,
    output_dir: Path,
) -> None:
    """Execute the frozen training-only MOD-13 development pipeline."""
    modeling_data = pd.read_csv(
        modeling_data_path
    )

    holdout_split = pd.read_csv(
        holdout_split_path
    )

    (
        training_data,
        _,
        test_ids,
    ) = prepare_training_data(
        modeling_data=modeling_data,
        holdout_split=holdout_split,
    )

    results = develop_all_models(
        training_data,
        forbidden_participant_ids=test_ids,
    )

    outputs = build_development_outputs(
        results
    )

    write_development_outputs(
        outputs,
        output_dir=output_dir,
    )

    print("MOD-13 development complete")

    for _, row in (
        outputs
        .bootstrap_performance
        .iterrows()
    ):
        print(
            f"{row['model']}: "
            f"N={int(row['participant_count'])}, "
            f"rows={int(row['eligible_row_count'])}, "
            f"OOB MAE={row['oob_mae']:.4f}, "
            f"OOB R2={row['oob_r2']:.4f}"
        )

    print(
        "Final-test participants were not evaluated."
    )


def require_final_test_confirmation(
    confirmed: bool,
) -> None:
    """Prevent accidental execution of the one-time final test."""
    if not confirmed:
        raise Mod13Error(
            "Final-test execution requires the explicit "
            "--confirm-final-test flag"
        )

def build_cli_parser() -> argparse.ArgumentParser:
    """Build the MOD-13 command-line interface."""
    parser = argparse.ArgumentParser(
        description=__doc__,
    )

    subparsers = parser.add_subparsers(
        dest="command",
        required=True,
    )

    develop_parser = (
        subparsers.add_parser(
            "develop",
            help=(
                "Run training-only MOD-13 "
                "development"
            ),
        )
    )

    develop_parser.add_argument(
        "--modeling-data",
        type=Path,
        required=True,
    )

    develop_parser.add_argument(
        "--holdout-split",
        type=Path,
        required=True,
    )

    develop_parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
    )

    final_parser = (
        subparsers.add_parser(
            "final-test",
            help=(
                "Run the one-time frozen "
                "final-test evaluation"
            ),
        )
    )

    final_parser.add_argument(
        "--modeling-data",
        type=Path,
        required=True,
    )

    final_parser.add_argument(
        "--holdout-split",
        type=Path,
        required=True,
    )

    final_parser.add_argument(
        "--coefficients",
        type=Path,
        required=True,
        help=(
            "Frozen regression_coefficients.csv "
            "created by the develop command"
        ),
    )

    final_parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
    )

    final_parser.add_argument(
        "--confirm-final-test",
        action="store_true",
        help=(
            "Explicitly confirm execution of the "
            "one-time final-test evaluation"
        ),
    )

    return parser

def run_final_test_cli(
    *,
    modeling_data_path: Path,
    holdout_split_path: Path,
    coefficients_path: Path,
    output_dir: Path,
    confirmed: bool,
) -> None:
    """Execute the explicit frozen one-time final test."""
    require_final_test_confirmation(
        confirmed
    )

    modeling_data = pd.read_csv(
        modeling_data_path
    )

    holdout_split = pd.read_csv(
        holdout_split_path
    )

    coefficient_table = pd.read_csv(
        coefficients_path
    )

    test_data, _ = (
        prepare_final_test_data(
            modeling_data=modeling_data,
            holdout_split=holdout_split,
        )
    )

    result = run_final_test(
        test_data,
        coefficient_table=(
            coefficient_table
        ),
    )

    write_final_test_outputs(
        result,
        output_dir=output_dir,
    )

    print(
        "MOD-13 final test complete"
    )

    for _, row in (
        result
        .performance
        .iterrows()
    ):
        print(
            f"{row['model']}: "
            f"N={int(row['participant_count'])}, "
            f"rows={int(row['eligible_row_count'])}, "
            f"MAE={row['mae']:.4f}, "
            f"R2={row['r2']:.4f}, "
            f"outside[0,3]="
            f"{int(row['out_of_range_prediction_count'])}"
        )

def main() -> int:
    """MOD-13 CLI entry point."""
    parser = build_cli_parser()

    args = parser.parse_args()

    try:
        if args.command == "develop":
            run_development_cli(
                modeling_data_path=(
                    args.modeling_data
                ),
                holdout_split_path=(
                    args.holdout_split
                ),
                output_dir=(
                    args.output_dir
                ),
            )

        elif args.command == "final-test":
            run_final_test_cli(
                modeling_data_path=(
                    args.modeling_data
                ),
                holdout_split_path=(
                    args.holdout_split
                ),
                coefficients_path=(
                    args.coefficients
                ),
                output_dir=(
                    args.output_dir
                ),
                confirmed=(
                    args.confirm_final_test
                ),
            )

        else:
            raise Mod13Error(
                f"Unknown command: "
                f"{args.command}"
            )

    except Mod13Error as exc:
        parser.error(
            str(exc)
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(
        main()
    )