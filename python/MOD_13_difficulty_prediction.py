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