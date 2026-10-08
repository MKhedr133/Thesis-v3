
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
