
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
