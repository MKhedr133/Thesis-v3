#!/usr/bin/env python3
"""MOD-14 — Direct linear regression training/test evaluation.

Batch 1 establishes the frozen data contract for direct regression
evaluation.

MOD-14 differs from MOD-13 internal validation in one important way:
participant bootstrap resampling is not used. Each final regression will
later be fitted once using the original eligible rows from the 26
development participants and evaluated on both development and test data.

This batch defines:
- the frozen 26-development / 6-test participant boundary;
- the difficulty-stage target coding;
- the four frozen regression model specifications;
- model-specific complete-row eligibility;
- separation of development and test observations.

Regression fitting, metrics, figures, and output writing are added in later
batches.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

import MOD_10_training_random_effects_comparison as MOD10

import statsmodels.api as sm


PARTICIPANT_COLUMN = "participant_id"
CONDITION_COLUMN = "condition_name"
DIFFICULTY_LEVEL_COLUMN = "difficulty_level"
TARGET_COLUMN = "difficulty_stage"

EXPECTED_DEVELOPMENT_PARTICIPANTS = 26
EXPECTED_TEST_PARTICIPANTS = 6

# MOD-14 deliberately performs no participant bootstrap fitting.
BOOTSTRAP_REPETITIONS = 0

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


class Mod14Error(ValueError):
    """Raised when the frozen MOD-14 data contract is violated."""


@dataclass(frozen=True)
class ModelSpec:
    """Frozen specification for one MOD-14 regression model."""

    model_id: str
    condition: str
    predictors: tuple[str, ...]
    allowed_stages: tuple[int, ...]


@dataclass(frozen=True)
class LinearModelFit:
    """One directly fitted MOD-14 multiple linear regression."""

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

        coefficient_vector = (
            self.coefficients
            .loc[
                design.columns
            ]
            .to_numpy(
                dtype=float
            )
        )

        predictions = (
            design
            .to_numpy(
                dtype=float
            )
            @ coefficient_vector
        )

        return pd.Series(
            predictions,
            index=frame.index,
            name=(
                "predicted_"
                "difficulty_stage"
            ),
        )


@dataclass(frozen=True)
class DirectEvaluationResult:
    """Direct training/test evaluation for one frozen model."""

    model_id: str
    condition: str
    development_frame: pd.DataFrame
    test_frame: pd.DataFrame
    fitted_model: LinearModelFit
    training_mae: float
    training_r2: float
    test_mae: float
    test_r2: float
    training_out_of_range_count: int
    test_out_of_range_count: int
    training_predictions: pd.DataFrame
    test_predictions: pd.DataFrame



def build_model_registry() -> dict[str, ModelSpec]:
    """Return the four frozen regression model specifications."""
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
    """Validate and return the frozen 26/6 participant split.

    MOD-14 reuses the MOD-09/MOD-10 participant boundary rather than
    defining a new split.
    """
    try:
        development_ids, test_ids = (
            MOD10.validate_holdout_split(
                holdout_split
            )
        )
    except ValueError as exc:
        raise Mod14Error(
            str(exc)
        ) from exc

    if (
        len(development_ids)
        != EXPECTED_DEVELOPMENT_PARTICIPANTS
    ):
        raise Mod14Error(
            "MOD-14 requires exactly "
            f"{EXPECTED_DEVELOPMENT_PARTICIPANTS} "
            "development participants"
        )

    if (
        len(test_ids)
        != EXPECTED_TEST_PARTICIPANTS
    ):
        raise Mod14Error(
            "MOD-14 requires exactly "
            f"{EXPECTED_TEST_PARTICIPANTS} "
            "test participants"
        )

    if not set(
        development_ids
    ).isdisjoint(
        test_ids
    ):
        raise Mod14Error(
            "Development and test participant IDs overlap"
        )

    return (
        development_ids,
        test_ids,
    )


def _normalise_participant_ids(
    data: pd.DataFrame,
) -> pd.DataFrame:
    """Return a copy with validated, stripped participant IDs."""
    if PARTICIPANT_COLUMN not in data.columns:
        raise Mod14Error(
            f"Missing required column: "
            f"{PARTICIPANT_COLUMN}"
        )

    frame = data.copy(
        deep=True
    )

    if frame[
        PARTICIPANT_COLUMN
    ].isna().any():
        raise Mod14Error(
            "participant_id contains missing values"
        )

    frame[
        PARTICIPANT_COLUMN
    ] = (
        frame[
            PARTICIPANT_COLUMN
        ]
        .astype(str)
        .str.strip()
    )

    if frame[
        PARTICIPANT_COLUMN
    ].eq("").any():
        raise Mod14Error(
            "participant_id contains blank values"
        )

    return frame


def prepare_split_data(
    *,
    modeling_data: pd.DataFrame,
    holdout_split: pd.DataFrame,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    tuple[str, ...],
    tuple[str, ...],
]:
    """Separate modelling data into frozen development and test sets.

    All rows from one participant remain in one split. The modelling table
    must contain exactly the participants defined by the frozen holdout
    split.
    """
    development_ids, test_ids = (
        validate_holdout_split(
            holdout_split
        )
    )

    data = _normalise_participant_ids(
        modeling_data
    )

    observed_ids = set(
        data[
            PARTICIPANT_COLUMN
        ]
    )

    expected_ids = (
        set(development_ids)
        | set(test_ids)
    )

    missing_ids = sorted(
        expected_ids
        - observed_ids
    )

    unexpected_ids = sorted(
        observed_ids
        - expected_ids
    )

    if missing_ids:
        raise Mod14Error(
            "Participants from the frozen split are "
            "missing from modeling data: "
            + ", ".join(
                missing_ids
            )
        )

    if unexpected_ids:
        raise Mod14Error(
            "Modeling data contain participants not "
            "present in the frozen split: "
            + ", ".join(
                unexpected_ids
            )
        )

    development = data.loc[
        data[
            PARTICIPANT_COLUMN
        ].isin(
            development_ids
        )
    ].copy()

    test = data.loc[
        data[
            PARTICIPANT_COLUMN
        ].isin(
            test_ids
        )
    ].copy()

    observed_development_ids = set(
        development[
            PARTICIPANT_COLUMN
        ]
    )

    observed_test_ids = set(
        test[
            PARTICIPANT_COLUMN
        ]
    )

    if (
        observed_development_ids
        != set(
            development_ids
        )
    ):
        raise Mod14Error(
            "Development data do not contain exactly "
            "the frozen development participants"
        )

    if (
        observed_test_ids
        != set(
            test_ids
        )
    ):
        raise Mod14Error(
            "Test data do not contain exactly "
            "the frozen test participants"
        )

    if not observed_development_ids.isdisjoint(
        observed_test_ids
    ):
        raise Mod14Error(
            "Participant overlap detected between "
            "development and test data"
        )

    if (
        development[
            PARTICIPANT_COLUMN
        ].nunique()
        != EXPECTED_DEVELOPMENT_PARTICIPANTS
    ):
        raise Mod14Error(
            "Development data must contain exactly "
            f"{EXPECTED_DEVELOPMENT_PARTICIPANTS} participants"
        )

    if (
        test[
            PARTICIPANT_COLUMN
        ].nunique()
        != EXPECTED_TEST_PARTICIPANTS
    ):
        raise Mod14Error(
            "Test data must contain exactly "
            f"{EXPECTED_TEST_PARTICIPANTS} participants"
        )

    return (
        development.reset_index(
            drop=True
        ),
        test.reset_index(
            drop=True
        ),
        development_ids,
        test_ids,
    )


def _validate_target_coding(
    condition_data: pd.DataFrame,
) -> pd.DataFrame:
    """Verify the frozen D0/D2/D6/D10 to 0/1/2/3 coding."""
    required_columns = (
        DIFFICULTY_LEVEL_COLUMN,
        TARGET_COLUMN,
    )

    missing_columns = [
        column
        for column in required_columns
        if column not in condition_data.columns
    ]

    if missing_columns:
        raise Mod14Error(
            "Missing required target columns: "
            f"{missing_columns}"
        )

    validated = condition_data.copy()

    difficulty_level = pd.to_numeric(
        validated[
            DIFFICULTY_LEVEL_COLUMN
        ],
        errors="coerce",
    )

    difficulty_stage = pd.to_numeric(
        validated[
            TARGET_COLUMN
        ],
        errors="coerce",
    )

    if (
        difficulty_level.isna().any()
        or difficulty_stage.isna().any()
    ):
        raise Mod14Error(
            "Difficulty level and difficulty stage "
            "must contain finite numeric values"
        )

    if (
        not np.isfinite(
            difficulty_level.to_numpy(
                dtype=float
            )
        ).all()
        or not np.isfinite(
            difficulty_stage.to_numpy(
                dtype=float
            )
        ).all()
    ):
        raise Mod14Error(
            "Difficulty level and difficulty stage "
            "must contain finite numeric values"
        )

    if not difficulty_level.isin(
        DIFFICULTY_STAGE_BY_LEVEL
    ).all():
        unexpected = sorted(
            set(
                difficulty_level
            )
            - set(
                DIFFICULTY_STAGE_BY_LEVEL
            )
        )

        raise Mod14Error(
            "Unexpected difficulty levels: "
            f"{unexpected}"
        )

    expected_stage = (
        difficulty_level.map(
            DIFFICULTY_STAGE_BY_LEVEL
        )
    )

    if not difficulty_stage.eq(
        expected_stage
    ).all():
        raise Mod14Error(
            "difficulty_stage does not match frozen coding: "
            "D0=0, D2=1, D6=2, D10=3"
        )

    validated[
        DIFFICULTY_LEVEL_COLUMN
    ] = (
        difficulty_level.astype(int)
    )

    validated[
        TARGET_COLUMN
    ] = (
        difficulty_stage.astype(int)
    )

    return validated


def prepare_model_frame(
    data: pd.DataFrame,
    *,
    model_id: str,
) -> pd.DataFrame:
    """Return complete eligible rows for one frozen MOD-14 model.

    Eligibility is model-specific:
    - keep only the condition belonging to the model;
    - verify the frozen difficulty-stage coding;
    - keep only the stages allowed by that model;
    - require complete finite target and predictor values;
    - preserve genuine numerical zeros;
    - perform no imputation or scaling.
    """
    registry = build_model_registry()

    if model_id not in registry:
        raise Mod14Error(
            f"Unknown MOD-14 model: "
            f"{model_id}"
        )

    spec = registry[
        model_id
    ]

    required_identity_columns = (
        PARTICIPANT_COLUMN,
        CONDITION_COLUMN,
        DIFFICULTY_LEVEL_COLUMN,
        TARGET_COLUMN,
    )

    required_columns = (
        *required_identity_columns,
        *spec.predictors,
    )

    missing_columns = [
        column
        for column in required_columns
        if column not in data.columns
    ]

    if missing_columns:
        raise Mod14Error(
            f"{model_id}: missing required columns: "
            f"{missing_columns}"
        )

    frame = _normalise_participant_ids(
        data
    )

    frame = frame.loc[
        frame[
            CONDITION_COLUMN
        ].eq(
            spec.condition
        )
    ].copy()

    if frame.empty:
        raise Mod14Error(
            f"{model_id}: no rows found for condition "
            f"{spec.condition}"
        )

    # Validate the complete condition before stage filtering.
    frame = _validate_target_coding(
        frame
    )

    frame = frame.loc[
        frame[
            TARGET_COLUMN
        ].isin(
            spec.allowed_stages
        )
    ].copy()

    if frame.empty:
        raise Mod14Error(
            f"{model_id}: no rows remain after "
            "difficulty-stage filtering"
        )

    numeric_columns = (
        TARGET_COLUMN,
        *spec.predictors,
    )

    # Coerce invalid numerical entries to NaN so the same complete-row
    # rule handles missing and non-numeric values.
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
        values = frame[
            column
        ]

        complete_mask &= (
            values
            .notna()
            .to_numpy()
        )

        complete_mask &= np.isfinite(
            values.to_numpy(
                dtype=float
            )
        )

    frame = frame.loc[
        complete_mask
    ].copy()

    if frame.empty:
        raise Mod14Error(
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
        frame.loc[
            :,
            output_columns,
        ]
        .reset_index(
            drop=True
        )
    )


def _build_design_matrix(
    frame: pd.DataFrame,
    *,
    predictors: tuple[str, ...],
) -> pd.DataFrame:
    """Build the unscaled OLS design matrix with an intercept."""
    missing_predictors = [
        predictor
        for predictor in predictors
        if predictor not in frame.columns
    ]

    if missing_predictors:
        raise Mod14Error(
            "Missing regression predictors: "
            f"{missing_predictors}"
        )

    design = frame.loc[
        :,
        list(
            predictors
        ),
    ].copy()

    for predictor in predictors:
        design[
            predictor
        ] = pd.to_numeric(
            design[
                predictor
            ],
            errors="coerce",
        )

    values = design.to_numpy(
        dtype=float
    )

    if not np.isfinite(
        values
    ).all():
        raise Mod14Error(
            "Regression design matrix contains "
            "missing or non-finite values"
        )

    design.insert(
        0,
        "Intercept",
        1.0,
    )

    return design


def fit_linear_model(
    frame: pd.DataFrame,
    *,
    model_id: str,
) -> LinearModelFit:
    """Fit exactly one OLS regression to the supplied development rows."""
    registry = (
        build_model_registry()
    )

    if model_id not in registry:
        raise Mod14Error(
            f"Unknown MOD-14 model: "
            f"{model_id}"
        )

    spec = registry[
        model_id
    ]

    if TARGET_COLUMN not in frame.columns:
        raise Mod14Error(
            f"{model_id}: missing target column "
            f"{TARGET_COLUMN}"
        )

    design = (
        _build_design_matrix(
            frame,
            predictors=(
                spec.predictors
            ),
        )
    )

    target = pd.to_numeric(
        frame[
            TARGET_COLUMN
        ],
        errors="coerce",
    ).to_numpy(
        dtype=float
    )

    if not np.isfinite(
        target
    ).all():
        raise Mod14Error(
            f"{model_id}: target contains "
            "missing or non-finite values"
        )

    observation_count = len(
        frame
    )

    parameter_count = (
        design.shape[1]
    )

    if (
        observation_count
        <= parameter_count
    ):
        raise Mod14Error(
            f"{model_id}: insufficient observations "
            f"({observation_count}) for "
            f"{parameter_count} regression parameters"
        )

    design_values = (
        design.to_numpy(
            dtype=float
        )
    )

    rank = int(
        np.linalg.matrix_rank(
            design_values
        )
    )

    if rank < parameter_count:
        raise Mod14Error(
            f"{model_id}: regression design matrix "
            "is rank deficient "
            f"(rank={rank}, "
            f"parameters={parameter_count})"
        )

    try:
        fitted = sm.OLS(
            target,
            design,
        ).fit()

    except Exception as exc:
        raise Mod14Error(
            f"{model_id}: OLS fitting failed"
        ) from exc

    coefficients = (
        fitted
        .params
        .astype(float)
    )

    if not np.isfinite(
        coefficients.to_numpy()
    ).all():
        raise Mod14Error(
            f"{model_id}: fitted coefficients "
            "contain non-finite values"
        )

    return LinearModelFit(
        model_id=model_id,
        predictors=(
            spec.predictors
        ),
        coefficients=(
            coefficients
        ),
    )


def participant_balanced_mae(
    *,
    participant_ids: pd.Series,
    observed: pd.Series,
    predicted: pd.Series,
) -> float:
    """Calculate MAE with equal weight for every participant."""
    if not (
        len(
            participant_ids
        )
        == len(
            observed
        )
        == len(
            predicted
        )
    ):
        raise Mod14Error(
            "Participant IDs, observations, "
            "and predictions must have equal length"
        )

    if len(
        observed
    ) == 0:
        raise Mod14Error(
            "Cannot calculate MAE "
            "without observations"
        )

    evaluation = pd.DataFrame(
        {
            PARTICIPANT_COLUMN: (
                participant_ids
                .astype(str)
                .to_numpy()
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
            [
                "observed",
                "predicted",
            ]
        ].to_numpy()
    ).all():
        raise Mod14Error(
            "MAE inputs contain "
            "non-finite values"
        )

    evaluation[
        "absolute_error"
    ] = (
        evaluation[
            "observed"
        ]
        - evaluation[
            "predicted"
        ]
    ).abs()

    participant_mae = (
        evaluation
        .groupby(
            PARTICIPANT_COLUMN,
            sort=False,
        )[
            "absolute_error"
        ]
        .mean()
    )

    if participant_mae.empty:
        raise Mod14Error(
            "Cannot calculate MAE "
            "without participants"
        )

    return float(
        participant_mae.mean()
    )


def pooled_r2(
    *,
    observed: pd.Series,
    predicted: pd.Series,
) -> float:
    """Calculate ordinary pooled predictive R²."""
    y_true = np.asarray(
        observed,
        dtype=float,
    )

    y_pred = np.asarray(
        predicted,
        dtype=float,
    )

    if (
        len(
            y_true
        )
        != len(
            y_pred
        )
    ):
        raise Mod14Error(
            "Observed and predicted values "
            "must have equal length"
        )

    if len(
        y_true
    ) == 0:
        raise Mod14Error(
            "Cannot calculate R² "
            "without observations"
        )

    if (
        not np.isfinite(
            y_true
        ).all()
        or not np.isfinite(
            y_pred
        ).all()
    ):
        raise Mod14Error(
            "R² inputs contain "
            "non-finite values"
        )

    residual_sum_squares = float(
        np.sum(
            (
                y_true
                - y_pred
            )
            ** 2
        )
    )

    mean_observed = float(
        np.mean(
            y_true
        )
    )

    total_sum_squares = float(
        np.sum(
            (
                y_true
                - mean_observed
            )
            ** 2
        )
    )

    if total_sum_squares == 0.0:
        return np.nan

    return float(
        1.0
        - (
            residual_sum_squares
            / total_sum_squares
        )
    )


def evaluate_model(
    *,
    development_data: pd.DataFrame,
    test_data: pd.DataFrame,
    model_id: str,
) -> DirectEvaluationResult:
    """Fit once on development data and evaluate training and test data.

    The regression is fitted only to the eligible rows from the original
    development participants. The already fitted model is then applied to
    the eligible test rows without refitting.
    """
    registry = (
        build_model_registry()
    )

    if model_id not in registry:
        raise Mod14Error(
            f"Unknown MOD-14 model: "
            f"{model_id}"
        )

    spec = registry[
        model_id
    ]

    development_frame = (
        prepare_model_frame(
            development_data,
            model_id=model_id,
        )
    )

    test_frame = (
        prepare_model_frame(
            test_data,
            model_id=model_id,
        )
    )

    development_ids = set(
        development_frame[
            PARTICIPANT_COLUMN
        ]
    )

    test_ids = set(
        test_frame[
            PARTICIPANT_COLUMN
        ]
    )

    if not development_ids.isdisjoint(
        test_ids
    ):
        raise Mod14Error(
            f"{model_id}: participant overlap "
            "between development and test frames"
        )

    # The only fitting call in this evaluation.
    fitted_model = (
        fit_linear_model(
            development_frame,
            model_id=model_id,
        )
    )

    training_prediction_values = (
        fitted_model.predict(
            development_frame
        )
    )

    test_prediction_values = (
        fitted_model.predict(
            test_frame
        )
    )

    training_observed = (
        development_frame[
            TARGET_COLUMN
        ].astype(float)
    )

    test_observed = (
        test_frame[
            TARGET_COLUMN
        ].astype(float)
    )

    training_mae = (
        participant_balanced_mae(
            participant_ids=(
                development_frame[
                    PARTICIPANT_COLUMN
                ]
            ),
            observed=(
                training_observed
            ),
            predicted=(
                training_prediction_values
            ),
        )
    )

    training_r2 = (
        pooled_r2(
            observed=(
                training_observed
            ),
            predicted=(
                training_prediction_values
            ),
        )
    )

    test_mae = (
        participant_balanced_mae(
            participant_ids=(
                test_frame[
                    PARTICIPANT_COLUMN
                ]
            ),
            observed=(
                test_observed
            ),
            predicted=(
                test_prediction_values
            ),
        )
    )

    test_r2 = (
        pooled_r2(
            observed=(
                test_observed
            ),
            predicted=(
                test_prediction_values
            ),
        )
    )

    training_predictions = (
        _build_prediction_table(
            development_frame,
            model_id=model_id,
            predictions=(
                training_prediction_values
            ),
        )
    )

    test_predictions = (
        _build_prediction_table(
            test_frame,
            model_id=model_id,
            predictions=(
                test_prediction_values
            ),
        )
    )

    training_out_of_range_count = (
        _count_out_of_range_predictions(
            training_prediction_values
        )
    )

    test_out_of_range_count = (
        _count_out_of_range_predictions(
            test_prediction_values
        )
    )

    return DirectEvaluationResult(
        model_id=model_id,
        condition=(
            spec.condition
        ),
        development_frame=(
            development_frame
        ),
        test_frame=(
            test_frame
        ),
        fitted_model=(
            fitted_model
        ),
        training_mae=(
            training_mae
        ),
        training_r2=(
            training_r2
        ),
        test_mae=(
            test_mae
        ),
        test_r2=(
            test_r2
        ),
        training_out_of_range_count=(
            training_out_of_range_count
        ),
        test_out_of_range_count=(
            test_out_of_range_count
        ),
        training_predictions=(
            training_predictions
        ),
        test_predictions=(
            test_predictions
        ),
    )


def evaluate_all_models(
    *,
    development_data: pd.DataFrame,
    test_data: pd.DataFrame,
) -> dict[
    str,
    DirectEvaluationResult,
]:
    """Directly evaluate all four frozen MOD-14 models."""
    results: dict[
        str,
        DirectEvaluationResult,
    ] = {}

    for model_id in (
        build_model_registry()
    ):
        results[
            model_id
        ] = evaluate_model(
            development_data=(
                development_data
            ),
            test_data=(
                test_data
            ),
            model_id=model_id,
        )

    return results


def build_performance_table(
    results: dict[
        str,
        DirectEvaluationResult,
    ],
) -> pd.DataFrame:
    """Build the direct training-versus-test performance table."""
    expected_models = tuple(
        build_model_registry()
    )

    if tuple(
        results
    ) != expected_models:
        raise Mod14Error(
            "Evaluation results must contain "
            "the four frozen MOD-14 models "
            "in the expected order"
        )

    rows: list[
        dict[str, object]
    ] = []

    for model_id in expected_models:
        result = results[
            model_id
        ]

        development_participant_count = int(
            result
            .development_frame[
                PARTICIPANT_COLUMN
            ]
            .nunique()
        )

        test_participant_count = int(
            result
            .test_frame[
                PARTICIPANT_COLUMN
            ]
            .nunique()
        )

        rows.append(
            {
                "model": model_id,
                "condition": (
                    result.condition
                ),
                (
                    "development_"
                    "participant_count"
                ): (
                    development_participant_count
                ),
                (
                    "development_"
                    "row_count"
                ): int(
                    len(
                        result
                        .development_frame
                    )
                ),
                (
                    "test_"
                    "participant_count"
                ): (
                    test_participant_count
                ),
                "test_row_count": int(
                    len(
                        result
                        .test_frame
                    )
                ),
                "training_mae": float(
                    result.training_mae
                ),
                "test_mae": float(
                    result.test_mae
                ),
                (
                    "test_minus_"
                    "training_mae"
                ): float(
                    result.test_mae
                    - result.training_mae
                ),
                "training_r2": float(
                    result.training_r2
                ),
                "test_r2": float(
                    result.test_r2
                ),
                (
                    "training_minus_"
                    "test_r2"
                ): float(
                    result.training_r2
                    - result.test_r2
                ),
                (
                    "training_out_of_range_"
                    "prediction_count"
                ): int(
                    result
                    .training_out_of_range_count
                ),
                (
                    "test_out_of_range_"
                    "prediction_count"
                ): int(
                    result
                    .test_out_of_range_count
                ),
            }
        )

    columns = (
        "model",
        "condition",
        (
            "development_"
            "participant_count"
        ),
        (
            "development_"
            "row_count"
        ),
        (
            "test_"
            "participant_count"
        ),
        "test_row_count",
        "training_mae",
        "test_mae",
        (
            "test_minus_"
            "training_mae"
        ),
        "training_r2",
        "test_r2",
        (
            "training_minus_"
            "test_r2"
        ),
        (
            "training_out_of_range_"
            "prediction_count"
        ),
        (
            "test_out_of_range_"
            "prediction_count"
        ),
    )

    return pd.DataFrame(
        rows,
        columns=columns,
    )

def _build_prediction_table(
    frame: pd.DataFrame,
    *,
    model_id: str,
    predictions: pd.Series,
) -> pd.DataFrame:
    """Build row-level prediction diagnostics."""
    if len(frame) != len(predictions):
        raise Mod14Error(
            f"{model_id}: prediction count "
            "does not match row count"
        )

    observed = (
        frame[
            TARGET_COLUMN
        ]
        .astype(float)
        .to_numpy()
    )

    predicted = (
        predictions
        .to_numpy(
            dtype=float
        )
    )

    error = (
        observed
        - predicted
    )

    return pd.DataFrame(
        {
            "participant_id": (
                frame[
                    PARTICIPANT_COLUMN
                ]
                .to_numpy()
            ),
            "model": model_id,
            "condition": (
                frame[
                    CONDITION_COLUMN
                ]
                .to_numpy()
            ),
            (
                "observed_"
                "difficulty_stage"
            ): observed,
            (
                "predicted_"
                "difficulty_stage"
            ): predicted,
            "error": error,
            "absolute_error": np.abs(
                error
            ),
        }
    )


def _count_out_of_range_predictions(
    predictions: pd.Series,
) -> int:
    """Count predictions outside the coded range [0, 3]."""
    values = (
        predictions
        .to_numpy(
            dtype=float
        )
    )

    return int(
        np.sum(
            (values < 0.0)
            | (values > 3.0)
        )
    )