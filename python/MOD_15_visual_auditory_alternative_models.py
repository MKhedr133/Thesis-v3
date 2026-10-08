
"""
MOD-15: Alternative regression models for Visual and Auditory.

Batch 1:
    - Frozen participant split validation
    - Predictor contracts
    - Eligible row preparation
    - Regression algorithm registry

No official participant data is loaded automatically.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm

from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GroupKFold, ParameterGrid
from sklearn.metrics import r2_score

import argparse
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

TARGET_COLUMN = "difficulty_stage"

VALID_DIFFICULTY_STAGES = {0, 1, 2, 3}

DEVELOPMENT_PARTICIPANT_COUNT = 26
TEST_PARTICIPANT_COUNT = 6

PREDICTORS_BY_CONDITION = {
    "Visual": [
        "mental_demand_score_0_to_10",
        "median_time_between_qualifying_grabs_seconds",
        "total_list_recheck_duration_seconds",
        "median_time_to_target_seconds",
        "median_irrelevant_focus_duration_seconds",
        "median_head_turning_degrees",
        "median_reach_duration_seconds",
    ],
    "Auditory": [
        "mental_demand_score_0_to_10",
        "median_time_between_qualifying_grabs_seconds",
        "median_time_to_target_seconds",
        "median_head_turning_degrees",
    ],
}


def load_frozen_split(split_path):
    """
    Load and validate the frozen participant split.

    Expected CSV columns:
        participant_id
        split

    Expected split labels:
        development
        test

    Returns:
        development_ids: set[str]
        test_ids: set[str]
    """

    split_path = Path(split_path)

    if not split_path.is_file():
        raise FileNotFoundError(
            f"Frozen split not found: {split_path}"
        )

    split_frame = pd.read_csv(
        split_path,
        dtype=str,
    )

    required_columns = {
        "participant_id",
        "split",
    }

    missing_columns = (
        required_columns - set(split_frame.columns)
    )

    if missing_columns:
        raise ValueError(
            "Frozen split is missing columns: "
            f"{sorted(missing_columns)}"
        )

    split_frame = split_frame.copy()

    split_frame["participant_id"] = (
        split_frame["participant_id"].str.strip()
    )

    split_frame["split"] = (
        split_frame["split"]
        .str.strip()
        .str.lower()
        .replace({"train": "development"})
    )

    if (
        split_frame["participant_id"].isna().any()
        or split_frame["participant_id"].eq("").any()
    ):
        raise ValueError(
            "Frozen split contains missing participant IDs."
        )

    if split_frame["split"].isna().any():
        raise ValueError(
            "Frozen split contains missing split labels."
        )

    valid_labels = {"development", "test"}

    actual_labels = set(split_frame["split"])

    if not actual_labels.issubset(valid_labels):
        raise ValueError(
            "Unexpected split labels: "
            f"{sorted(actual_labels - valid_labels)}"
        )

    if split_frame["participant_id"].duplicated().any():
        raise ValueError(
            "Frozen split contains duplicate participant IDs."
        )

    development_ids = set(
        split_frame.loc[
            split_frame["split"] == "development",
            "participant_id",
        ]
    )

    test_ids = set(
        split_frame.loc[
            split_frame["split"] == "test",
            "participant_id",
        ]
    )

    if len(development_ids) != DEVELOPMENT_PARTICIPANT_COUNT:
        raise ValueError(
            "Expected 26 development participants; "
            f"found {len(development_ids)}."
        )

    if len(test_ids) != TEST_PARTICIPANT_COUNT:
        raise ValueError(
            "Expected 6 test participants; "
            f"found {len(test_ids)}."
        )

    if not development_ids.isdisjoint(test_ids):
        raise ValueError(
            "Development and test participants overlap."
        )

    return development_ids, test_ids


def prepare_condition_rows(
    data,
    condition,
    participant_ids,
):
    """
    Select eligible rows for one condition and participant set.

    Rules:
        - Only requested participants are retained.
        - Only the exact condition predictors are retained.
        - Rows missing required values are excluded.
        - No imputation.
        - No predictor scaling.
        - Genuine numerical zeros are preserved.
        - Difficulty stages must use 0, 1, 2, 3.

    Returns:
        A new DataFrame containing:
            participant_id
            condition
            difficulty_stage
            required predictors
    """

    if condition not in PREDICTORS_BY_CONDITION:
        raise ValueError(
            f"Unknown condition: {condition}"
        )

    predictors = PREDICTORS_BY_CONDITION[condition]

    required_columns = [
        "participant_id",
        "condition",
        TARGET_COLUMN,
        *predictors,
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in data.columns
    ]

    if missing_columns:
        raise KeyError(
            f"Missing required columns: {missing_columns}"
        )

    participant_ids = {
        str(participant_id).strip()
        for participant_id in participant_ids
    }

    participant_series = (
        data["participant_id"]
        .astype("string")
        .str.strip()
    )

    condition_mask = (
        data["condition"] == condition
    )

    participant_mask = (
        participant_series.isin(participant_ids)
    )

    selected = data.loc[
        condition_mask & participant_mask,
        required_columns,
    ].copy()

    selected["participant_id"] = (
        selected["participant_id"]
        .astype("string")
        .str.strip()
    )

    numerical_columns = [
        TARGET_COLUMN,
        *predictors,
    ]

    for column in numerical_columns:
        selected[column] = pd.to_numeric(
            selected[column],
            errors="raise",
        )

    selected = selected.dropna(
        subset=numerical_columns
    ).copy()

    if not selected.empty:
        numerical_values = selected[
            numerical_columns
        ].to_numpy(dtype=float)

        if not np.isfinite(numerical_values).all():
            raise ValueError(
                "Eligible rows contain infinite numerical values."
            )

        actual_stages = set(
            selected[TARGET_COLUMN].unique()
        )

        if not actual_stages.issubset(
            VALID_DIFFICULTY_STAGES
        ):
            raise ValueError(
                "Difficulty stage must be coded "
                "as 0, 1, 2, or 3."
            )

    return selected.reset_index(drop=True)


class DirectOLSRegressor:
    """
    Direct ordinary least squares regression.

    Includes an intercept and validates full column rank.

    Provides fit() and predict() methods for compatibility
    with the other regression models.
    """

    def __init__(self):
        self.results_ = None
        self.n_features_in_ = None
        self.intercept_ = None
        self.coef_ = None

    @staticmethod
    def _prepare_design_matrix(X):
        """
        Convert predictors to a numerical matrix and add
        an intercept column.
        """

        X = np.asarray(X, dtype=float)

        if X.ndim != 2:
            raise ValueError(
                "X must be a two-dimensional matrix."
            )

        if not np.isfinite(X).all():
            raise ValueError(
                "Predictors must contain finite values."
            )

        intercept = np.ones(
            (X.shape[0], 1),
            dtype=float,
        )

        return np.column_stack(
            [intercept, X]
        )

    def fit(self, X, y):
        """
        Fit OLS directly.

        Raises ValueError when the design matrix is not
        full rank.
        """

        design_matrix = self._prepare_design_matrix(X)

        y = np.asarray(y, dtype=float).reshape(-1)

        if design_matrix.shape[0] != len(y):
            raise ValueError(
                "X and y have different numbers of rows."
            )

        if len(y) == 0:
            raise ValueError(
                "Cannot fit OLS on an empty dataset."
            )

        if not np.isfinite(y).all():
            raise ValueError(
                "Target values must be finite."
            )

        matrix_rank = np.linalg.matrix_rank(
            design_matrix
        )

        required_rank = design_matrix.shape[1]

        if matrix_rank != required_rank:
            raise ValueError(
                "OLS design matrix is not full rank. "
                f"Rank: {matrix_rank}; "
                f"required: {required_rank}."
            )

        model = sm.OLS(
            y,
            design_matrix,
        )

        self.results_ = model.fit()

        self.n_features_in_ = (
            design_matrix.shape[1] - 1
        )

        self.intercept_ = float(
            self.results_.params[0]
        )

        self.coef_ = np.asarray(
            self.results_.params[1:],
            dtype=float,
        )

        return self

    def predict(self, X):
        """
        Generate continuous predictions.

        Predictions are neither rounded nor clipped.
        """

        if self.results_ is None:
            raise RuntimeError(
                "OLS model must be fitted before prediction."
            )

        design_matrix = self._prepare_design_matrix(X)

        supplied_features = (
            design_matrix.shape[1] - 1
        )

        if supplied_features != self.n_features_in_:
            raise ValueError(
                "Prediction feature count does not match "
                "the fitted model."
            )

        predictions = self.results_.predict(
            design_matrix
        )

        return np.asarray(
            predictions,
            dtype=float,
        )


def build_model_registry(random_state=42):
    """
    Create the three regression algorithms.

    Hyperparameter selection is implemented in Batch 2.

    The values defined here are initial configurations,
    not selected final hyperparameters.
    """

    return {
        "multiple_linear_regression": DirectOLSRegressor(),

        "decision_tree_regression": DecisionTreeRegressor(
            random_state=random_state,
        ),

        "random_forest_regression": RandomForestRegressor(
            n_estimators=500,
            random_state=random_state,
        ),
    }



# ============================================================
# MOD-15 — Batch 2
# Development-only nonlinear model selection
# ============================================================

CV_N_SPLITS = 5
CV_MAE_TIE_TOLERANCE = 0.01

DECISION_TREE_SEARCH_SPACE = {
    "max_depth": [2, 3, 4, None],
    "min_samples_leaf": [1, 2, 4],
}

RANDOM_FOREST_SEARCH_SPACE = {
    "n_estimators": [500],
    "max_depth": [2, 3, 4, None],
    "min_samples_leaf": [1, 2, 4],
    "max_features": ["sqrt", 1.0],
}


def participant_balanced_mae(
    observed,
    predicted,
    participant_ids,
):
    """
    Calculate MAE with equal weight per participant.

    Step 1: Absolute error for each observation.
    Step 2: Mean absolute error per participant.
    Step 3: Average the participant MAEs.
    """

    observed = np.asarray(
        observed, dtype=float
    ).reshape(-1)

    predicted = np.asarray(
        predicted, dtype=float
    ).reshape(-1)

    participant_ids = np.asarray(
        participant_ids
    ).reshape(-1)

    if not (
        len(observed)
        == len(predicted)
        == len(participant_ids)
    ):
        raise ValueError(
            "Observed, predicted, and participant IDs "
            "must have equal lengths."
        )

    if len(observed) == 0:
        raise ValueError(
            "Cannot calculate MAE with zero observations."
        )

    if not (
        np.isfinite(observed).all()
        and np.isfinite(predicted).all()
    ):
        raise ValueError(
            "Observed and predicted values must be finite."
        )

    errors = pd.DataFrame({
        "participant_id": participant_ids,
        "absolute_error": np.abs(
            observed - predicted
        ),
    })

    if errors["participant_id"].isna().any():
        raise ValueError(
            "Participant IDs cannot be missing."
        )

    participant_mae = (
        errors.groupby("participant_id")[
            "absolute_error"
        ].mean()
    )

    return float(participant_mae.mean())


def make_participant_cv_splits(
    rows,
    n_splits=CV_N_SPLITS,
):
    """
    Create participant-grouped cross-validation folds.

    All rows from one participant remain together.
    Each participant appears in exactly one
    validation fold.
    """

    if "participant_id" not in rows.columns:
        raise ValueError(
            "Missing participant_id column."
        )

    groups = (
        rows["participant_id"]
        .astype("string")
        .str.strip()
    )

    if groups.isna().any() or groups.eq("").any():
        raise ValueError(
            "Missing participant IDs in CV data."
        )

    n_participants = groups.nunique()

    if not isinstance(n_splits, int):
        raise ValueError(
            "n_splits must be an integer."
        )

    if n_splits < 2:
        raise ValueError(
            "Cross-validation requires at least two folds."
        )

    if n_splits > n_participants:
        raise ValueError(
            "Number of CV folds exceeds number "
            "of participants."
        )

    cv = GroupKFold(
        n_splits=n_splits
    )

    dummy_features = np.zeros(
        (len(rows), 1)
    )

    folds = list(
        cv.split(
            dummy_features,
            groups=groups.to_numpy(),
        )
    )

    return folds


def _create_nonlinear_regressor(
    model_name,
    params,
    random_state,
):
    """
    Construct a new unfitted nonlinear regressor.

    A fresh model is created for every CV fold.
    """

    params = dict(params)

    if "random_state" in params:
        raise ValueError(
            "Set random_state through the function "
            "argument, not the parameter search."
        )

    if model_name == "decision_tree_regression":
        return DecisionTreeRegressor(
            **params,
            random_state=random_state,
        )

    if model_name == "random_forest_regression":
        return RandomForestRegressor(
            **params,
            random_state=random_state,
        )

    raise ValueError(
        f"Unsupported nonlinear model: {model_name}"
    )


def evaluate_nonlinear_cv_candidate(
    rows,
    condition,
    model_name,
    params,
    n_splits=CV_N_SPLITS,
    random_state=42,
):
    """
    Evaluate one nonlinear hyperparameter configuration.

    Input must already contain eligible rows.

    Returns one out-of-fold prediction per row.

    No final model is fitted here.
    """

    if condition not in PREDICTORS_BY_CONDITION:
        raise ValueError(
            f"Unknown condition: {condition}"
        )

    if model_name not in {
        "decision_tree_regression",
        "random_forest_regression",
    }:
        raise ValueError(
            f"Unsupported nonlinear model: {model_name}"
        )

    predictors = PREDICTORS_BY_CONDITION[
        condition
    ]

    required_columns = [
        "participant_id",
        "condition",
        TARGET_COLUMN,
        *predictors,
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in rows.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing columns: {missing_columns}"
        )

    if len(rows) == 0:
        raise ValueError(
            "No eligible rows available for CV."
        )

    if not rows["condition"].eq(condition).all():
        raise ValueError(
            "CV rows contain another condition."
        )

    numerical_columns = [
        TARGET_COLUMN,
        *predictors,
    ]

    if rows[numerical_columns].isna().any().any():
        raise ValueError(
            "CV input contains missing numerical values."
        )

    numerical_values = rows[
        numerical_columns
    ].to_numpy(dtype=float)

    if not np.isfinite(numerical_values).all():
        raise ValueError(
            "CV input contains non-finite numerical values."
        )

    if not set(rows[TARGET_COLUMN].unique()).issubset(
        VALID_DIFFICULTY_STAGES
    ):
        raise ValueError(
            "Invalid difficulty-stage coding."
        )

    folds = make_participant_cv_splits(
        rows,
        n_splits=n_splits,
    )

    observed = rows[
        TARGET_COLUMN
    ].to_numpy(dtype=float)

    predictions = np.full(
        len(rows),
        np.nan,
        dtype=float,
    )

    fold_assignments = np.full(
        len(rows),
        -1,
        dtype=int,
    )

    for fold_number, (
        training_idx,
        validation_idx,
    ) in enumerate(folds, start=1):

        training_rows = rows.iloc[
            training_idx
        ]

        validation_rows = rows.iloc[
            validation_idx
        ]

        training_ids = set(
            training_rows["participant_id"]
        )

        validation_ids = set(
            validation_rows["participant_id"]
        )

        if not training_ids.isdisjoint(
            validation_ids
        ):
            raise RuntimeError(
                "Participant leakage within CV fold."
            )

        X_train = training_rows[
            predictors
        ].to_numpy(dtype=float)

        y_train = training_rows[
            TARGET_COLUMN
        ].to_numpy(dtype=float)

        X_validation = validation_rows[
            predictors
        ].to_numpy(dtype=float)

        model = _create_nonlinear_regressor(
            model_name=model_name,
            params=params,
            random_state=random_state,
        )

        model.fit(
            X_train,
            y_train,
        )

        fold_predictions = np.asarray(
            model.predict(X_validation),
            dtype=float,
        ).reshape(-1)

        if len(fold_predictions) != len(
            validation_idx
        ):
            raise RuntimeError(
                "Unexpected number of CV predictions."
            )

        if not np.isfinite(
            fold_predictions
        ).all():
            raise RuntimeError(
                "CV generated non-finite predictions."
            )

        predictions[
            validation_idx
        ] = fold_predictions

        fold_assignments[
            validation_idx
        ] = fold_number

    if np.isnan(predictions).any():
        raise RuntimeError(
            "Some development rows have no "
            "cross-validation prediction."
        )

    if (fold_assignments < 1).any():
        raise RuntimeError(
            "Some rows have no validation fold."
        )

    return pd.DataFrame({
        "participant_id": rows[
            "participant_id"
        ].to_numpy(),
        "observed_difficulty_stage": observed,
        "predicted_difficulty_stage": predictions,
        "fold": fold_assignments,
    })


def _parameter_complexity_key(
    params,
    model_name,
):
    """
    Rank configurations from simpler to more complex.

    Simpler means:
        1. Smaller maximum depth.
        2. Larger minimum leaf size.
        3. For forests, fewer candidate features
           per split, then fewer trees.

    Used only for effectively tied CV scores.
    """

    max_depth = params.get(
        "max_depth",
        None,
    )

    depth_rank = (
        float("inf")
        if max_depth is None
        else int(max_depth)
    )

    leaf_rank = -int(
        params.get("min_samples_leaf", 1)
    )

    if model_name == "decision_tree_regression":
        return (
            depth_rank,
            leaf_rank,
        )

    max_features = params.get(
        "max_features",
        1.0,
    )

    feature_rank = (
        0 if max_features == "sqrt" else 1
    )

    n_estimators = int(
        params.get("n_estimators", 500)
    )

    return (
        depth_rank,
        leaf_rank,
        feature_rank,
        n_estimators,
    )


def select_nonlinear_hyperparameters(
    data,
    condition,
    model_name,
    development_ids,
    candidate_params=None,
    n_splits=CV_N_SPLITS,
    random_state=42,
):
    """
    Select nonlinear hyperparameters using development
    participants only.

    The six unseen test participants must not influence:
        - Cross-validation fitting
        - Cross-validation predictions
        - MAE calculation
        - Hyperparameter selection

    Selection:
        1. Filter to development participants.
        2. Prepare eligible rows.
        3. Evaluate every candidate with GroupKFold.
        4. Calculate participant-balanced CV MAE.
        5. Identify minimum CV MAE.
        6. Prefer simpler candidates within the
           predefined 0.01 MAE tie tolerance.

    Returns:
        {
            "best_params": dict,
            "best_cv_mae": float,
            "cv_results": DataFrame,
        }

    Does not refit the final model.
    Final fitting belongs to Batch 3.
    """

    if condition not in PREDICTORS_BY_CONDITION:
        raise ValueError(
            f"Unknown condition: {condition}"
        )

    if model_name == "decision_tree_regression":
        search_space = DECISION_TREE_SEARCH_SPACE

    elif model_name == "random_forest_regression":
        search_space = RANDOM_FOREST_SEARCH_SPACE

    else:
        raise ValueError(
            f"Unsupported nonlinear model: {model_name}"
        )

    development_ids = {
        str(participant_id).strip()
        for participant_id in development_ids
    }

    # Crucial: filter to development participants
    # before cross-validation or scoring.
    development_rows = prepare_condition_rows(
        data,
        condition=condition,
        participant_ids=development_ids,
    )

    if development_rows.empty:
        raise ValueError(
            "No eligible development observations."
        )

    if not set(
        development_rows["participant_id"]
    ).issubset(development_ids):
        raise RuntimeError(
            "Non-development participant entered selection."
        )

    # None means the complete official search space.
    # Explicit candidates are useful for synthetic tests.
    if candidate_params is None:
        candidates = list(
            ParameterGrid(search_space)
        )
    else:
        candidates = [
            dict(params)
            for params in candidate_params
        ]

    if not candidates:
        raise ValueError(
            "No hyperparameter candidates supplied."
        )

    results = []

    for candidate_number, params in enumerate(
        candidates,
        start=1,
    ):

        oof = evaluate_nonlinear_cv_candidate(
            development_rows,
            condition=condition,
            model_name=model_name,
            params=params,
            n_splits=n_splits,
            random_state=random_state,
        )

        cv_mae = participant_balanced_mae(
            observed=oof[
                "observed_difficulty_stage"
            ],
            predicted=oof[
                "predicted_difficulty_stage"
            ],
            participant_ids=oof[
                "participant_id"
            ],
        )

        result_row = {
            "condition": condition,
            "model": model_name,
            "candidate_id": candidate_number,
            "max_depth": params.get(
                "max_depth"
            ),
            "min_samples_leaf": params.get(
                "min_samples_leaf"
            ),
            "n_estimators": params.get(
                "n_estimators"
            ),
            "max_features": params.get(
                "max_features"
            ),
            "cv_mae": float(cv_mae),
            "n_splits": n_splits,
            "development_participant_count":
                development_rows[
                    "participant_id"
                ].nunique(),
            "development_row_count": len(
                development_rows
            ),
        }

        results.append(result_row)

    cv_results = pd.DataFrame(results)

    minimum_cv_mae = float(
        cv_results["cv_mae"].min()
    )

    # Treat configurations within 0.01 stage units
    # of the minimum MAE as effectively tied.
    tied_candidate_indices = [
        index
        for index, result in enumerate(results)
        if result["cv_mae"]
        <= minimum_cv_mae + CV_MAE_TIE_TOLERANCE
    ]

    # Prefer the simplest configuration among ties.
    winning_index = min(
        tied_candidate_indices,
        key=lambda index: (
            _parameter_complexity_key(
                candidates[index],
                model_name,
            ),
            index,
        ),
    )

    best_params = dict(
        candidates[winning_index]
    )

    best_cv_mae = float(
        results[winning_index]["cv_mae"]
    )

    cv_results["selected"] = (
        cv_results["candidate_id"]
        == winning_index + 1
    )

    return {
        "best_params": best_params,
        "best_cv_mae": best_cv_mae,
        "cv_results": cv_results,
    }



# ============================================================
# MOD-15 — Batch 3
# Final model fitting and evaluation
# ============================================================

FINAL_MODEL_NAMES = (
    "multiple_linear_regression",
    "decision_tree_regression",
    "random_forest_regression",
)

PERFORMANCE_COLUMNS = [
    "condition",
    "model",
    "development_participant_count",
    "development_row_count",
    "test_participant_count",
    "test_row_count",
    "training_mae",
    "test_mae",
    "test_minus_training_mae",
    "training_r2",
    "test_r2",
    "training_minus_test_r2",
]

PREDICTION_COLUMNS = [
    "participant_id",
    "condition",
    "model",
    "observed_difficulty_stage",
    "predicted_difficulty_stage",
    "error",
    "absolute_error",
]


def _validate_final_participant_ids(
    development_ids,
    test_ids=None,
):
    """
    Validate participant identifiers for final fitting
    and evaluation.

    The frozen split itself is validated separately by
    load_frozen_split() from Batch 1.
    """

    development_ids = {
        str(pid).strip()
        for pid in development_ids
    }

    if len(development_ids) != 26:
        raise ValueError(
            "Final fitting requires exactly "
            "26 development participant IDs."
        )

    if not development_ids or "" in development_ids:
        raise ValueError(
            "Invalid development participant IDs."
        )

    if test_ids is None:
        return development_ids, None

    test_ids = {
        str(pid).strip()
        for pid in test_ids
    }

    if len(test_ids) != 6:
        raise ValueError(
            "Final evaluation requires exactly "
            "6 test participant IDs."
        )

    if "" in test_ids:
        raise ValueError(
            "Invalid test participant IDs."
        )

    if not development_ids.isdisjoint(test_ids):
        raise ValueError(
            "Development and test participant IDs overlap."
        )

    return development_ids, test_ids


def fit_final_condition_models(
    data,
    condition,
    development_ids,
    selected_hyperparameters,
    random_state=42,
):
    """
    Fit all three final regression models.

    Only eligible observations belonging to the
    26 development participants are used.

    Multiple Linear Regression:
        Direct OLS with intercept and rank validation.

    Decision Tree:
        Uses hyperparameters selected in Batch 2.

    Random Forest:
        Uses hyperparameters selected in Batch 2.

    No test data is used for fitting.

    Returns:
        Dictionary containing the three fitted models.
    """

    if condition not in PREDICTORS_BY_CONDITION:
        raise ValueError(
            f"Unknown condition: {condition}"
        )

    development_ids, _ = (
        _validate_final_participant_ids(
            development_ids
        )
    )

    required_nonlinear_models = {
        "decision_tree_regression",
        "random_forest_regression",
    }

    if not isinstance(
        selected_hyperparameters, dict
    ):
        raise ValueError(
            "Selected hyperparameters must be a dictionary."
        )

    missing_models = (
        required_nonlinear_models
        - set(selected_hyperparameters)
    )

    if missing_models:
        raise ValueError(
            "Missing selected hyperparameters for: "
            f"{sorted(missing_models)}"
        )

    # Select development participants before fitting.
    development_rows = prepare_condition_rows(
        data,
        condition=condition,
        participant_ids=development_ids,
    )

    if development_rows.empty:
        raise ValueError(
            "No eligible development observations."
        )

    observed_participants = set(
        development_rows["participant_id"]
    )

    if not observed_participants.issubset(
        development_ids
    ):
        raise RuntimeError(
            "Non-development participant entered fitting."
        )

    predictors = PREDICTORS_BY_CONDITION[
        condition
    ]

    X_development = development_rows[
        predictors
    ].to_numpy(dtype=float)

    y_development = development_rows[
        TARGET_COLUMN
    ].to_numpy(dtype=float)

    fitted_models = {}

    # ----------------------------------------------------
    # 1. Multiple Linear Regression
    # ----------------------------------------------------

    linear_model = DirectOLSRegressor()

    linear_model.fit(
        X_development,
        y_development,
    )

    fitted_models[
        "multiple_linear_regression"
    ] = linear_model

    # ----------------------------------------------------
    # 2. Decision Tree Regression
    # 3. Random Forest Regression
    # ----------------------------------------------------

    for model_name in (
        "decision_tree_regression",
        "random_forest_regression",
    ):
        params = dict(
            selected_hyperparameters[model_name]
        )

        model = _create_nonlinear_regressor(
            model_name=model_name,
            params=params,
            random_state=random_state,
        )

        # Exactly one final fit using every eligible
        # development observation.
        model.fit(
            X_development,
            y_development,
        )

        fitted_models[model_name] = model

    return fitted_models


def _build_final_prediction_table(
    model,
    rows,
    condition,
    model_name,
):
    """
    Generate raw continuous predictions using
    an already fitted model.

    Never calls fit().

    Predictions are not scaled, rounded, or clipped.
    """

    predictors = PREDICTORS_BY_CONDITION[
        condition
    ]

    X = rows[
        predictors
    ].to_numpy(dtype=float)

    observed = rows[
        TARGET_COLUMN
    ].to_numpy(dtype=float)

    predicted = np.asarray(
        model.predict(X),
        dtype=float,
    ).reshape(-1)

    if len(predicted) != len(observed):
        raise RuntimeError(
            "Prediction count does not match "
            "the number of eligible observations."
        )

    if not np.isfinite(predicted).all():
        raise ValueError(
            "Model generated non-finite predictions."
        )

    # Signed prediction error:
    # Positive: model predicted a stage too low.
    # Negative: model predicted a stage too high.
    error = observed - predicted

    absolute_error = np.abs(error)

    predictions = pd.DataFrame({
        "participant_id": rows[
            "participant_id"
        ].to_numpy(),
        "condition": condition,
        "model": model_name,
        "observed_difficulty_stage": observed,
        "predicted_difficulty_stage": predicted,
        "error": error,
        "absolute_error": absolute_error,
    })

    return predictions[PREDICTION_COLUMNS]


def _calculate_final_performance(
    predictions,
):
    """
    Calculate two metrics:

    MAE:
        Participant-balanced mean absolute error.

    R2:
        Ordinary pooled coefficient of determination
        across all eligible observations.

    Negative R2 values are preserved.
    """

    if len(predictions) < 2:
        raise ValueError(
            "At least two eligible observations "
            "are required to calculate R2."
        )

    observed = predictions[
        "observed_difficulty_stage"
    ].to_numpy(dtype=float)

    predicted = predictions[
        "predicted_difficulty_stage"
    ].to_numpy(dtype=float)

    participant_ids = predictions[
        "participant_id"
    ].to_numpy()

    mae = participant_balanced_mae(
        observed=observed,
        predicted=predicted,
        participant_ids=participant_ids,
    )

    r2 = float(
        r2_score(
            observed,
            predicted,
        )
    )

    if not np.isfinite(r2):
        raise ValueError(
            "R2 is undefined for this evaluation split."
        )

    return {
        "mae": float(mae),
        "r2": r2,
    }


def evaluate_final_condition_models(
    models,
    data,
    condition,
    development_ids,
    test_ids,
):
    """
    Evaluate fitted models on development and test data.

    Training:
        The same development participants used for
        final fitting.

    Test:
        The six previously unseen participants.

    No model is refitted.
    No hyperparameters are modified.
    No predictions are rounded or clipped.

    Returns:
        {
            "performance": DataFrame,
            "training_predictions": DataFrame,
            "test_predictions": DataFrame,
        }
    """

    if condition not in PREDICTORS_BY_CONDITION:
        raise ValueError(
            f"Unknown condition: {condition}"
        )

    development_ids, test_ids = (
        _validate_final_participant_ids(
            development_ids,
            test_ids,
        )
    )

    missing_models = (
        set(FINAL_MODEL_NAMES)
        - set(models)
    )

    if missing_models:
        raise ValueError(
            "Missing fitted models: "
            f"{sorted(missing_models)}"
        )

    # ----------------------------------------------------
    # Prepare the two datasets independently.
    # ----------------------------------------------------

    development_rows = prepare_condition_rows(
        data,
        condition=condition,
        participant_ids=development_ids,
    )

    test_rows = prepare_condition_rows(
        data,
        condition=condition,
        participant_ids=test_ids,
    )

    if development_rows.empty:
        raise ValueError(
            "No eligible development observations."
        )

    if test_rows.empty:
        raise ValueError(
            "No eligible test observations."
        )

    development_participants = set(
        development_rows["participant_id"]
    )

    test_participants = set(
        test_rows["participant_id"]
    )

    if not development_participants.isdisjoint(
        test_participants
    ):
        raise RuntimeError(
            "Participant overlap in evaluation data."
        )

    # ----------------------------------------------------
    # Evaluate each previously fitted model.
    # ----------------------------------------------------

    performance_records = []
    training_prediction_tables = []
    test_prediction_tables = []

    for model_name in FINAL_MODEL_NAMES:

        model = models[model_name]

        # Training predictions
        training_predictions = (
            _build_final_prediction_table(
                model=model,
                rows=development_rows,
                condition=condition,
                model_name=model_name,
            )
        )

        # Test predictions from the same fitted model
        test_predictions = (
            _build_final_prediction_table(
                model=model,
                rows=test_rows,
                condition=condition,
                model_name=model_name,
            )
        )

        training_metrics = (
            _calculate_final_performance(
                training_predictions
            )
        )

        test_metrics = (
            _calculate_final_performance(
                test_predictions
            )
        )

        training_mae = training_metrics["mae"]
        test_mae = test_metrics["mae"]

        training_r2 = training_metrics["r2"]
        test_r2 = test_metrics["r2"]

        performance_records.append({
            "condition": condition,
            "model": model_name,

            "development_participant_count":
                len(development_participants),

            "development_row_count":
                len(development_rows),

            "test_participant_count":
                len(test_participants),

            "test_row_count":
                len(test_rows),

            "training_mae": training_mae,
            "test_mae": test_mae,

            "test_minus_training_mae":
                test_mae - training_mae,

            "training_r2": training_r2,
            "test_r2": test_r2,

            "training_minus_test_r2":
                training_r2 - test_r2,
        })

        training_prediction_tables.append(
            training_predictions
        )

        test_prediction_tables.append(
            test_predictions
        )

    # ----------------------------------------------------
    # Combine model results.
    # ----------------------------------------------------

    performance = pd.DataFrame(
        performance_records,
        columns=PERFORMANCE_COLUMNS,
    )

    all_training_predictions = pd.concat(
        training_prediction_tables,
        ignore_index=True,
    )

    all_test_predictions = pd.concat(
        test_prediction_tables,
        ignore_index=True,
    )

    return {
        "performance": performance,

        "training_predictions":
            all_training_predictions[
                PREDICTION_COLUMNS
            ],

        "test_predictions":
            all_test_predictions[
                PREDICTION_COLUMNS
            ],
    }



# ============================================================
# MOD-15 — Batch 4
# CSV outputs, figures, and standalone CLI
# ============================================================

DEFAULT_OUTPUT_DIRECTORY = (
    "outputs/MOD_15_visual_auditory_alternative_models"
)

MOD15_CSV_FILES = {
    "performance": "model_comparison_performance.csv",
    "selected_hyperparameters": "selected_hyperparameters.csv",
    "development_cv_results": "development_cv_results.csv",
    "training_predictions": "training_predictions.csv",
    "test_predictions": "test_predictions.csv",
}

MOD15_CONDITIONS = ("Visual", "Auditory")

MOD15_MODEL_LABELS = {
    "multiple_linear_regression": (
        "Multiple Linear Regression"
    ),
    "decision_tree_regression": (
        "Decision Tree Regression"
    ),
    "random_forest_regression": (
        "Random Forest Regression"
    ),
}

MOD15_MODEL_FILE_LABELS = {
    "multiple_linear_regression": "linear_regression",
    "decision_tree_regression": "decision_tree",
    "random_forest_regression": "random_forest",
}

MOD15_METRICS = {
    "mae": {
        "training": "training_mae",
        "test": "test_mae",
        "ylabel": "Participant-balanced MAE",
    },
    "r2": {
        "training": "training_r2",
        "test": "test_r2",
        "ylabel": "Pooled R²",
    },
}


def _mod15_figure_filenames():
    """Return the exact 16 required figure filenames."""

    filenames = []

    for condition in MOD15_CONDITIONS:
        prefix = condition.lower()

        for model_name in FINAL_MODEL_NAMES:
            model_prefix = MOD15_MODEL_FILE_LABELS[
                model_name
            ]

            for metric in MOD15_METRICS:
                filenames.append(
                    f"{prefix}_{model_prefix}_{metric}.png"
                )

    for condition in MOD15_CONDITIONS:
        prefix = condition.lower()

        for metric in MOD15_METRICS:
            filenames.append(
                f"{prefix}_model_comparison_{metric}.png"
            )

    return filenames


def _check_output_conflicts(
    output_directory,
    filenames,
    overwrite=False,
):
    """
    Check every planned file before writing anything.

    Existing files are never overwritten unless
    overwrite=True is explicitly supplied.
    """

    output_directory = Path(output_directory)

    if (
        output_directory.exists()
        and not output_directory.is_dir()
    ):
        raise NotADirectoryError(
            f"Output location is not a directory: "
            f"{output_directory}"
        )

    if not overwrite:
        existing = [
            name
            for name in filenames
            if (output_directory / name).exists()
        ]

        if existing:
            raise FileExistsError(
                "Existing MOD-15 outputs would be "
                "overwritten: "
                + ", ".join(existing)
            )

    return output_directory


def write_mod15_outputs(
    performance,
    selected_hyperparameters,
    development_cv_results,
    training_predictions,
    test_predictions,
    output_directory,
    overwrite=False,
):
    """
    Write the five required CSV files.

    All output paths are checked before any file
    is created.
    """

    tables = {
        "performance": performance,
        "selected_hyperparameters":
            selected_hyperparameters,
        "development_cv_results":
            development_cv_results,
        "training_predictions":
            training_predictions,
        "test_predictions":
            test_predictions,
    }

    output_directory = _check_output_conflicts(
        output_directory=output_directory,
        filenames=list(MOD15_CSV_FILES.values()),
        overwrite=overwrite,
    )

    for key, table in tables.items():
        if not isinstance(table, pd.DataFrame):
            raise TypeError(
                f"{key} must be a pandas DataFrame."
            )

    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    written_paths = {}

    for key, filename in MOD15_CSV_FILES.items():
        path = output_directory / filename

        tables[key].to_csv(
            path,
            index=False,
        )

        written_paths[key] = path

    return written_paths


def _validate_figure_performance(performance):
    """
    Require one performance row for each condition
    and regression model.
    """

    required_columns = {
        "condition",
        "model",
        "training_mae",
        "test_mae",
        "training_r2",
        "test_r2",
    }

    missing = (
        required_columns - set(performance.columns)
    )

    if missing:
        raise ValueError(
            f"Missing figure data columns: "
            f"{sorted(missing)}"
        )

    expected_pairs = {
        (condition, model_name)
        for condition in MOD15_CONDITIONS
        for model_name in FINAL_MODEL_NAMES
    }

    actual_pairs = list(
        zip(
            performance["condition"],
            performance["model"],
        )
    )

    if (
        len(actual_pairs) != len(expected_pairs)
        or set(actual_pairs) != expected_pairs
    ):
        raise ValueError(
            "Performance table must contain exactly "
            "one row for every condition and model."
        )

    metric_columns = [
        "training_mae",
        "test_mae",
        "training_r2",
        "test_r2",
    ]

    values = performance[
        metric_columns
    ].to_numpy(dtype=float)

    if not np.isfinite(values).all():
        raise ValueError(
            "Performance metrics must be finite."
        )

    if (
        performance[
            ["training_mae", "test_mae"]
        ].to_numpy(dtype=float) < 0
    ).any():
        raise ValueError(
            "MAE cannot be negative."
        )


def _format_mod15_bar_labels(axis, bars):
    """Show three-decimal numerical labels."""

    axis.bar_label(
        bars,
        fmt="%.3f",
        padding=3,
        fontsize=9,
    )


def _create_individual_model_chart(
    condition,
    model_name,
    metric,
    training_value,
    test_value,
    output_path,
):
    """
    Create one figure comparing training and test
    performance for a single regression model.
    """

    metric_info = MOD15_METRICS[metric]

    figure, axis = plt.subplots(
        figsize=(6.5, 4.5)
    )

    try:
        training_bars = axis.bar(
            [0],
            [training_value],
            width=0.55,
            label="Training",
        )

        test_bars = axis.bar(
            [1],
            [test_value],
            width=0.55,
            label="Test",
        )

        _format_mod15_bar_labels(
            axis, training_bars
        )

        _format_mod15_bar_labels(
            axis, test_bars
        )

        axis.set_xticks(
            [0, 1],
            ["Training", "Test"],
        )

        axis.set_ylabel(
            metric_info["ylabel"]
        )

        axis.set_title(
            f"{condition} — "
            f"{MOD15_MODEL_LABELS[model_name]} — "
            f"{metric.upper()}"
        )

        if metric == "r2":
            axis.axhline(
                y=0,
                linestyle="--",
                linewidth=1,
            )

        axis.margins(y=0.20)

        figure.tight_layout()

        figure.savefig(
            output_path,
            dpi=300,
            bbox_inches="tight",
        )

    finally:
        plt.close(figure)


def _create_model_comparison_chart(
    condition,
    metric,
    condition_performance,
    output_path,
):
    """
    Create a grouped bar chart comparing all three
    algorithms using training and test results.
    """

    metric_info = MOD15_METRICS[metric]

    ordered = (
        condition_performance
        .set_index("model")
        .loc[list(FINAL_MODEL_NAMES)]
    )

    training_values = ordered[
        metric_info["training"]
    ].to_numpy(dtype=float)

    test_values = ordered[
        metric_info["test"]
    ].to_numpy(dtype=float)

    model_labels = [
        MOD15_MODEL_LABELS[name]
        for name in FINAL_MODEL_NAMES
    ]

    x_positions = np.arange(
        len(FINAL_MODEL_NAMES)
    )

    width = 0.36

    figure, axis = plt.subplots(
        figsize=(10, 5.5)
    )

    try:
        training_bars = axis.bar(
            x_positions - width / 2,
            training_values,
            width=width,
            label="Training",
        )

        test_bars = axis.bar(
            x_positions + width / 2,
            test_values,
            width=width,
            label="Test",
        )

        _format_mod15_bar_labels(
            axis, training_bars
        )

        _format_mod15_bar_labels(
            axis, test_bars
        )

        axis.set_xticks(
            x_positions,
            model_labels,
            rotation=12,
            ha="right",
        )

        axis.set_ylabel(
            metric_info["ylabel"]
        )

        axis.set_title(
            f"{condition} — Regression Model "
            f"Comparison — {metric.upper()}"
        )

        axis.legend()

        if metric == "r2":
            axis.axhline(
                y=0,
                linestyle="--",
                linewidth=1,
            )

        axis.margins(y=0.20)

        figure.tight_layout()

        figure.savefig(
            output_path,
            dpi=300,
            bbox_inches="tight",
        )

    finally:
        plt.close(figure)


def create_mod15_figures(
    performance,
    output_directory,
    overwrite=False,
):
    """
    Generate all 16 performance bar charts.

    Individual:
        2 conditions × 3 models × 2 metrics = 12.

    Comparison:
        2 conditions × 2 metrics = 4.

    No manual bar colors are assigned.
    """

    _validate_figure_performance(
        performance
    )

    filenames = _mod15_figure_filenames()

    output_directory = _check_output_conflicts(
        output_directory=output_directory,
        filenames=filenames,
        overwrite=overwrite,
    )

    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    written_paths = []

    for condition in MOD15_CONDITIONS:

        condition_performance = (
            performance.loc[
                performance["condition"] == condition
            ].copy()
        )

        for model_name in FINAL_MODEL_NAMES:

            row = condition_performance.loc[
                condition_performance["model"]
                == model_name
            ].iloc[0]

            model_file_label = (
                MOD15_MODEL_FILE_LABELS[model_name]
            )

            for metric, metric_info in (
                MOD15_METRICS.items()
            ):

                filename = (
                    f"{condition.lower()}_"
                    f"{model_file_label}_"
                    f"{metric}.png"
                )

                path = output_directory / filename

                _create_individual_model_chart(
                    condition=condition,
                    model_name=model_name,
                    metric=metric,
                    training_value=float(
                        row[metric_info["training"]]
                    ),
                    test_value=float(
                        row[metric_info["test"]]
                    ),
                    output_path=path,
                )

                written_paths.append(path)

        for metric in MOD15_METRICS:

            filename = (
                f"{condition.lower()}_"
                f"model_comparison_{metric}.png"
            )

            path = output_directory / filename

            _create_model_comparison_chart(
                condition=condition,
                metric=metric,
                condition_performance=(
                    condition_performance
                ),
                output_path=path,
            )

            written_paths.append(path)

    return written_paths


def run_mod15_analysis(
    data_path,
    split_path,
    output_directory=DEFAULT_OUTPUT_DIRECTORY,
    overwrite=False,
    random_state=42,
):
    """
    Execute the complete MOD-15 analysis.

    1. Load the existing frozen participant split.
    2. Load trial data.
    3. Select nonlinear hyperparameters using
       development participants only.
    4. Fit final models using development data.
    5. Evaluate training and test performance.
    6. Save all CSV files.
    7. Create all 16 figures.

    The six test participants never enter
    hyperparameter selection or model fitting.
    """

    data_path = Path(data_path)
    split_path = Path(split_path)
    output_directory = Path(
        output_directory
    )

    if not data_path.is_file():
        raise FileNotFoundError(
            f"Trial data not found: {data_path}"
        )

    if not split_path.is_file():
        raise FileNotFoundError(
            f"Frozen split not found: {split_path}"
        )

    # Protect every planned artifact before training.
    all_output_files = (
        list(MOD15_CSV_FILES.values())
        + _mod15_figure_filenames()
    )

    _check_output_conflicts(
        output_directory,
        all_output_files,
        overwrite=overwrite,
    )

    development_ids, test_ids = (
        load_frozen_split(split_path)
    )

    data = pd.read_csv(
        data_path,
        dtype={"participant_id": str},
    )

    # Support the existing modeling_data.csv schema.
    if "condition" not in data.columns:
        if "condition_name" not in data.columns:
            raise ValueError(
                "Trial data must contain either "
                "'condition' or 'condition_name'."
            )

        data = data.rename(
            columns={"condition_name": "condition"}
        )

    # Standardize Visual and Auditory labels.
    data["condition"] = (
        data["condition"].astype("string").str.strip()
    )

    for condition_label in ("Visual", "Auditory"):
        mask = (
            data["condition"].str.casefold()
            == condition_label.casefold()
        )

        data.loc[mask, "condition"] = condition_label

    performance_tables = []
    selected_rows = []
    cv_result_tables = []
    training_tables = []
    test_tables = []

    for condition in MOD15_CONDITIONS:

        selected_hyperparameters = {}

        # --------------------------------------------
        # Development-only hyperparameter selection
        # --------------------------------------------

        for model_name in (
            "decision_tree_regression",
            "random_forest_regression",
        ):

            selection = (
                select_nonlinear_hyperparameters(
                    data=data,
                    condition=condition,
                    model_name=model_name,
                    development_ids=development_ids,
                    random_state=random_state,
                )
            )

            params = dict(
                selection["best_params"]
            )

            selected_hyperparameters[
                model_name
            ] = params

            selected_rows.append({
                "condition": condition,
                "model": model_name,
                "max_depth": params.get(
                    "max_depth"
                ),
                "min_samples_leaf": params.get(
                    "min_samples_leaf"
                ),
                "n_estimators": params.get(
                    "n_estimators"
                ),
                "max_features": params.get(
                    "max_features"
                ),
                "cv_mae": selection[
                    "best_cv_mae"
                ],
            })

            cv_result_tables.append(
                selection["cv_results"]
            )

        # --------------------------------------------
        # Final fitting: 26 development participants
        # --------------------------------------------

        fitted_models = (
            fit_final_condition_models(
                data=data,
                condition=condition,
                development_ids=development_ids,
                selected_hyperparameters=(
                    selected_hyperparameters
                ),
                random_state=random_state,
            )
        )

        # --------------------------------------------
        # Evaluation: development and unseen test
        # --------------------------------------------

        evaluation = (
            evaluate_final_condition_models(
                models=fitted_models,
                data=data,
                condition=condition,
                development_ids=development_ids,
                test_ids=test_ids,
            )
        )

        performance_tables.append(
            evaluation["performance"]
        )

        training_tables.append(
            evaluation["training_predictions"]
        )

        test_tables.append(
            evaluation["test_predictions"]
        )

    # ------------------------------------------------
    # Combine both conditions
    # ------------------------------------------------

    performance = pd.concat(
        performance_tables,
        ignore_index=True,
    )

    selected_hyperparameters_table = pd.DataFrame(
        selected_rows
    )

    development_cv_results = pd.concat(
        cv_result_tables,
        ignore_index=True,
    )

    training_predictions = pd.concat(
        training_tables,
        ignore_index=True,
    )

    test_predictions = pd.concat(
        test_tables,
        ignore_index=True,
    )

    # ------------------------------------------------
    # Save results
    # ------------------------------------------------

    csv_paths = write_mod15_outputs(
        performance=performance,
        selected_hyperparameters=(
            selected_hyperparameters_table
        ),
        development_cv_results=(
            development_cv_results
        ),
        training_predictions=(
            training_predictions
        ),
        test_predictions=test_predictions,
        output_directory=output_directory,
        overwrite=overwrite,
    )

    figure_paths = create_mod15_figures(
        performance=performance,
        output_directory=output_directory,
        overwrite=overwrite,
    )

    return {
        "performance": performance,
        "selected_hyperparameters":
            selected_hyperparameters_table,
        "development_cv_results":
            development_cv_results,
        "training_predictions":
            training_predictions,
        "test_predictions":
            test_predictions,
        "csv_paths": csv_paths,
        "figure_paths": figure_paths,
    }


def main(argv=None):
    """
    Command-line entry point for MOD-15.

    Official data must be supplied explicitly.
    """

    parser = argparse.ArgumentParser(
        description=(
            "MOD-15: Compare linear and nonlinear "
            "regression models for Visual and Auditory."
        )
    )

    parser.add_argument(
        "--data",
        required=True,
        help="CSV containing trial data.",
    )

    parser.add_argument(
        "--split",
        required=True,
        help="Frozen participant_holdout_split.csv.",
    )

    parser.add_argument(
        "--output-dir",
        default=DEFAULT_OUTPUT_DIRECTORY,
        help="Directory for CSV and PNG outputs.",
    )

    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random state for reproducibility.",
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Explicitly allow replacing MOD-15 outputs.",
    )

    args = parser.parse_args(argv)

    return run_mod15_analysis(
        data_path=args.data,
        split_path=args.split,
        output_directory=args.output_dir,
        overwrite=args.overwrite,
        random_state=args.random_state,
    )


if __name__ == "__main__":
    main()
