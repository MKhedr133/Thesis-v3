
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = (
    ROOT
    / "python"
    / "MOD_15_visual_auditory_alternative_models.py"
)

VISUAL_PREDICTORS = [
    "mental_demand_score_0_to_10",
    "median_time_between_qualifying_grabs_seconds",
    "total_list_recheck_duration_seconds",
    "median_time_to_target_seconds",
    "median_irrelevant_focus_duration_seconds",
    "median_head_turning_degrees",
    "median_reach_duration_seconds",
]

AUDITORY_PREDICTORS = [
    "mental_demand_score_0_to_10",
    "median_time_between_qualifying_grabs_seconds",
    "median_time_to_target_seconds",
    "median_head_turning_degrees",
]

MODEL_NAMES = {
    "multiple_linear_regression",
    "decision_tree_regression",
    "random_forest_regression",
}


@pytest.fixture(scope="module")
def mod15():
    assert MODULE_PATH.is_file(), (
        f"Expected implementation file: {MODULE_PATH}"
    )

    spec = importlib.util.spec_from_file_location(
        "mod15_under_test", MODULE_PATH
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def split_frame():
    ids = [f"P{i:02d}" for i in range(1, 33)]
    labels = ["development"] * 26 + ["test"] * 6

    return pd.DataFrame({
        "participant_id": ids,
        "split": labels,
    })


@pytest.fixture
def synthetic_trials():
    rows = []

    for participant_number in range(1, 33):
        participant_id = f"P{participant_number:02d}"

        for condition in ["Visual", "Auditory"]:
            for stage in range(4):
                row = {
                    "participant_id": participant_id,
                    "condition": condition,
                    "difficulty_stage": stage,
                    "mental_demand_score_0_to_10": float(stage),
                    "median_time_between_qualifying_grabs_seconds":
                        2.0 + stage,
                    "total_list_recheck_duration_seconds":
                        5.0 + stage,
                    "median_time_to_target_seconds":
                        3.0 + stage,
                    "median_irrelevant_focus_duration_seconds":
                        1.0 + stage,
                    "median_head_turning_degrees":
                        15.0 + stage,
                    "median_reach_duration_seconds":
                        0.5 + stage,
                    "age_group": "Young",
                    "TMT_B": 80.0,
                }
                rows.append(row)

    return pd.DataFrame(rows)


def write_split(frame, tmp_path):
    path = tmp_path / "participant_holdout_split.csv"
    frame.to_csv(path, index=False)
    return path


def test_frozen_split_has_exact_26_6_participants(
    mod15, split_frame, tmp_path
):
    path = write_split(split_frame, tmp_path)

    development_ids, test_ids = mod15.load_frozen_split(path)

    assert len(development_ids) == 26
    assert len(test_ids) == 6

    assert development_ids == {
        f"P{i:02d}" for i in range(1, 27)
    }
    assert test_ids == {
        f"P{i:02d}" for i in range(27, 33)
    }

    assert development_ids.isdisjoint(test_ids)


def test_frozen_split_rejects_wrong_counts(
    mod15, split_frame, tmp_path
):
    invalid = split_frame.copy()
    invalid.loc[25, "split"] = "test"

    path = write_split(invalid, tmp_path)

    with pytest.raises(ValueError):
        mod15.load_frozen_split(path)


def test_frozen_split_rejects_duplicate_participants(
    mod15, split_frame, tmp_path
):
    invalid = split_frame.copy()
    invalid.loc[26, "participant_id"] = "P01"

    path = write_split(invalid, tmp_path)

    with pytest.raises(ValueError):
        mod15.load_frozen_split(path)


def test_frozen_split_rejects_unknown_labels(
    mod15, split_frame, tmp_path
):
    invalid = split_frame.copy()
    invalid.loc[0, "split"] = "unknown"

    path = write_split(invalid, tmp_path)

    with pytest.raises(ValueError):
        mod15.load_frozen_split(path)


def test_exact_predictor_sets(mod15):
    assert mod15.PREDICTORS_BY_CONDITION == {
        "Visual": VISUAL_PREDICTORS,
        "Auditory": AUDITORY_PREDICTORS,
    }

    assert mod15.TARGET_COLUMN == "difficulty_stage"

    for predictors in mod15.PREDICTORS_BY_CONDITION.values():
        assert "age_group" not in predictors
        assert "TMT_B" not in predictors


@pytest.mark.parametrize(
    "condition,expected_predictors",
    [
        ("Visual", VISUAL_PREDICTORS),
        ("Auditory", AUDITORY_PREDICTORS),
    ],
)
def test_condition_rows_use_exact_predictors(
    mod15,
    synthetic_trials,
    condition,
    expected_predictors,
):
    development_ids = {
        f"P{i:02d}" for i in range(1, 27)
    }

    result = mod15.prepare_condition_rows(
        synthetic_trials,
        condition=condition,
        participant_ids=development_ids,
    )

    expected_columns = [
        "participant_id",
        "condition",
        "difficulty_stage",
        *expected_predictors,
    ]

    assert list(result.columns) == expected_columns

    assert len(result) == 26 * 4
    assert result["participant_id"].nunique() == 26
    assert set(result["condition"]) == {condition}
    assert set(result["difficulty_stage"]) == {0, 1, 2, 3}

    assert set(result["participant_id"]).issubset(
        development_ids
    )


def test_test_participants_never_enter_development_rows(
    mod15, synthetic_trials
):
    development_ids = {
        f"P{i:02d}" for i in range(1, 27)
    }
    test_ids = {
        f"P{i:02d}" for i in range(27, 33)
    }

    for condition in ["Visual", "Auditory"]:
        result = mod15.prepare_condition_rows(
            synthetic_trials,
            condition=condition,
            participant_ids=development_ids,
        )

        assert not (
            set(result["participant_id"]) & test_ids
        )


def test_missing_required_predictors_are_not_imputed(
    mod15, synthetic_trials
):
    data = synthetic_trials.copy()

    mask = (
        (data["participant_id"] == "P01")
        & (data["condition"] == "Visual")
        & (data["difficulty_stage"] == 2)
    )

    data.loc[
        mask, "median_reach_duration_seconds"
    ] = np.nan

    result = mod15.prepare_condition_rows(
        data,
        condition="Visual",
        participant_ids={"P01", "P02"},
    )

    assert len(result) == 7

    assert not (
        (result["participant_id"] == "P01")
        & (result["difficulty_stage"] == 2)
    ).any()

    assert result[VISUAL_PREDICTORS].notna().all().all()


def test_missing_unrelated_predictors_do_not_remove_rows(
    mod15, synthetic_trials
):
    data = synthetic_trials.copy()

    auditory_mask = data["condition"] == "Auditory"

    data.loc[
        auditory_mask,
        "total_list_recheck_duration_seconds",
    ] = np.nan

    result = mod15.prepare_condition_rows(
        data,
        condition="Auditory",
        participant_ids={"P01", "P02"},
    )

    assert len(result) == 8


def test_genuine_zeros_and_raw_values_are_preserved(
    mod15, synthetic_trials
):
    result = mod15.prepare_condition_rows(
        synthetic_trials,
        condition="Visual",
        participant_ids={"P01"},
    )

    result = result.sort_values("difficulty_stage")

    assert result[
        "mental_demand_score_0_to_10"
    ].tolist() == [0.0, 1.0, 2.0, 3.0]

    assert result[
        "median_head_turning_degrees"
    ].tolist() == [15.0, 16.0, 17.0, 18.0]

    assert result[
        "median_reach_duration_seconds"
    ].tolist() == [0.5, 1.5, 2.5, 3.5]


def test_input_dataframe_is_not_modified(
    mod15, synthetic_trials
):
    original = synthetic_trials.copy(deep=True)

    mod15.prepare_condition_rows(
        synthetic_trials,
        condition="Visual",
        participant_ids={"P01", "P02"},
    )

    pd.testing.assert_frame_equal(
        synthetic_trials,
        original,
    )


def test_missing_required_column_raises_error(
    mod15, synthetic_trials
):
    invalid = synthetic_trials.drop(
        columns=["median_time_to_target_seconds"]
    )

    with pytest.raises((KeyError, ValueError)):
        mod15.prepare_condition_rows(
            invalid,
            condition="Visual",
            participant_ids={"P01"},
        )


def test_unknown_condition_raises_error(
    mod15, synthetic_trials
):
    with pytest.raises(ValueError):
        mod15.prepare_condition_rows(
            synthetic_trials,
            condition="Unknown",
            participant_ids={"P01"},
        )


def test_algorithm_registry_contains_exact_models(mod15):
    registry = mod15.build_model_registry(
        random_state=42
    )

    assert set(registry) == MODEL_NAMES

    linear = registry["multiple_linear_regression"]
    tree = registry["decision_tree_regression"]
    forest = registry["random_forest_regression"]

    assert callable(getattr(linear, "fit", None))
    assert callable(getattr(linear, "predict", None))

    assert isinstance(tree, DecisionTreeRegressor)
    assert isinstance(forest, RandomForestRegressor)

    assert tree.random_state == 42
    assert forest.random_state == 42

    assert forest.n_estimators == 500


def test_algorithm_random_states_are_configurable(mod15):
    registry = mod15.build_model_registry(
        random_state=123
    )

    assert (
        registry["decision_tree_regression"].random_state
        == 123
    )

    assert (
        registry["random_forest_regression"].random_state
        == 123
    )



# ============================================================
# MOD-15 — Batch 2
# Development-only nonlinear model selection
# ============================================================

def _development_ids():
    return {f"P{i:02d}" for i in range(1, 27)}


def _test_ids():
    return {f"P{i:02d}" for i in range(27, 33)}


def _tree_candidate(depth=2, leaf=1):
    return {
        "max_depth": depth,
        "min_samples_leaf": leaf,
    }


def _forest_candidate(trees=7):
    # Small forest for synthetic tests only.
    # Official selection must use 500 trees.
    return {
        "n_estimators": trees,
        "max_depth": 2,
        "min_samples_leaf": 1,
        "max_features": "sqrt",
    }


def test_batch2_exact_search_spaces(mod15):
    assert mod15.CV_N_SPLITS == 5
    assert mod15.CV_MAE_TIE_TOLERANCE == 0.01

    assert mod15.DECISION_TREE_SEARCH_SPACE == {
        "max_depth": [2, 3, 4, None],
        "min_samples_leaf": [1, 2, 4],
    }

    assert mod15.RANDOM_FOREST_SEARCH_SPACE == {
        "n_estimators": [500],
        "max_depth": [2, 3, 4, None],
        "min_samples_leaf": [1, 2, 4],
        "max_features": ["sqrt", 1.0],
    }


def test_grouped_cv_keeps_participants_together(
    mod15, synthetic_trials
):
    rows = mod15.prepare_condition_rows(
        synthetic_trials,
        condition="Visual",
        participant_ids=_development_ids(),
    )

    folds = mod15.make_participant_cv_splits(
        rows,
        n_splits=5,
    )

    assert len(folds) == 5

    validation_ids = []
    validation_row_positions = []

    for training_idx, validation_idx in folds:
        training = rows.iloc[training_idx]
        validation = rows.iloc[validation_idx]

        training_participants = set(
            training["participant_id"]
        )
        validation_participants = set(
            validation["participant_id"]
        )

        assert training_participants.isdisjoint(
            validation_participants
        )

        assert not (
            training_participants & _test_ids()
        )

        assert not (
            validation_participants & _test_ids()
        )

        validation_ids.extend(validation_participants)
        validation_row_positions.extend(
            validation_idx.tolist()
        )

    # Every development participant belongs to exactly
    # one validation fold.
    assert len(validation_ids) == 26
    assert set(validation_ids) == _development_ids()

    # Every development row is validated exactly once.
    assert sorted(validation_row_positions) == list(
        range(len(rows))
    )


def test_grouped_cv_rejects_too_many_folds(
    mod15, synthetic_trials
):
    rows = mod15.prepare_condition_rows(
        synthetic_trials,
        condition="Visual",
        participant_ids={"P01", "P02"},
    )

    with pytest.raises(ValueError):
        mod15.make_participant_cv_splits(
            rows,
            n_splits=3,
        )


def test_participant_balanced_mae(mod15):
    # Participant A: four rows, all correct.
    # Participant B: one row, absolute error = 2.
    #
    # Participant-balanced MAE = (0 + 2) / 2 = 1.
    # Ordinary pooled MAE = 2 / 5 = 0.4.

    observed = np.array([0, 0, 0, 0, 0])
    predicted = np.array([0, 0, 0, 0, 2])

    participants = np.array([
        "A", "A", "A", "A", "B"
    ])

    score = mod15.participant_balanced_mae(
        observed,
        predicted,
        participants,
    )

    assert score == pytest.approx(1.0)


def test_tree_cv_generates_one_prediction_per_row(
    mod15, synthetic_trials
):
    rows = mod15.prepare_condition_rows(
        synthetic_trials,
        condition="Visual",
        participant_ids=_development_ids(),
    )

    oof = mod15.evaluate_nonlinear_cv_candidate(
        rows,
        condition="Visual",
        model_name="decision_tree_regression",
        params=_tree_candidate(),
        n_splits=5,
        random_state=42,
    )

    required_columns = {
        "participant_id",
        "observed_difficulty_stage",
        "predicted_difficulty_stage",
        "fold",
    }

    assert required_columns.issubset(oof.columns)

    assert len(oof) == len(rows)
    assert oof["participant_id"].nunique() == 26

    # All four trials from a participant belong to
    # one validation fold.
    assert (
        oof.groupby("participant_id")["fold"]
        .nunique()
        .eq(1)
        .all()
    )

    assert oof[
        "predicted_difficulty_stage"
    ].notna().all()

    assert np.isfinite(
        oof["predicted_difficulty_stage"]
    ).all()


def test_both_algorithms_use_identical_eligible_rows(
    mod15, synthetic_trials
):
    rows = mod15.prepare_condition_rows(
        synthetic_trials,
        condition="Auditory",
        participant_ids=_development_ids(),
    )

    tree_oof = mod15.evaluate_nonlinear_cv_candidate(
        rows,
        condition="Auditory",
        model_name="decision_tree_regression",
        params=_tree_candidate(),
        n_splits=5,
        random_state=42,
    )

    forest_oof = mod15.evaluate_nonlinear_cv_candidate(
        rows,
        condition="Auditory",
        model_name="random_forest_regression",
        params=_forest_candidate(),
        n_splits=5,
        random_state=42,
    )

    columns = [
        "participant_id",
        "observed_difficulty_stage",
    ]

    tree_rows = tree_oof[columns].sort_values(
        columns
    ).reset_index(drop=True)

    forest_rows = forest_oof[columns].sort_values(
        columns
    ).reset_index(drop=True)

    pd.testing.assert_frame_equal(
        tree_rows,
        forest_rows,
    )


def test_selection_uses_development_participants_only(
    mod15, synthetic_trials, monkeypatch
):
    data = synthetic_trials.copy()

    # Give the six test participants extreme predictor
    # values so leakage becomes detectable.
    test_mask = data["participant_id"].isin(
        _test_ids()
    )

    data.loc[
        test_mask,
        "median_time_to_target_seconds",
    ] = 10000.0

    original_fit = DecisionTreeRegressor.fit
    original_predict = DecisionTreeRegressor.predict

    fitted_matrices = []
    predicted_matrices = []

    def audited_fit(self, X, y, *args, **kwargs):
        matrix = np.asarray(X, dtype=float)

        assert matrix.max() < 1000.0

        fitted_matrices.append(matrix.copy())

        return original_fit(
            self, X, y, *args, **kwargs
        )

    def audited_predict(self, X, *args, **kwargs):
        matrix = np.asarray(X, dtype=float)

        assert matrix.max() < 1000.0

        predicted_matrices.append(matrix.copy())

        return original_predict(
            self, X, *args, **kwargs
        )

    monkeypatch.setattr(
        DecisionTreeRegressor,
        "fit",
        audited_fit,
    )

    monkeypatch.setattr(
        DecisionTreeRegressor,
        "predict",
        audited_predict,
    )

    result = mod15.select_nonlinear_hyperparameters(
        data,
        condition="Visual",
        model_name="decision_tree_regression",
        development_ids=_development_ids(),
        candidate_params=[
            _tree_candidate(depth=2),
        ],
        n_splits=5,
        random_state=42,
    )

    # Five CV fits, without a final full-data refit.
    assert len(fitted_matrices) == 5
    assert len(predicted_matrices) == 5

    assert result["best_params"] == _tree_candidate(
        depth=2
    )


def test_selection_ignores_test_targets(
    mod15, synthetic_trials
):
    data_a = synthetic_trials.copy()
    data_b = synthetic_trials.copy()

    # Change every target for the six unseen participants.
    data_b.loc[
        data_b["participant_id"].isin(_test_ids()),
        "difficulty_stage",
    ] = 3

    kwargs = {
        "condition": "Visual",
        "model_name": "decision_tree_regression",
        "development_ids": _development_ids(),
        "candidate_params": [
            _tree_candidate(depth=1),
            _tree_candidate(depth=2),
        ],
        "n_splits": 5,
        "random_state": 42,
    }

    result_a = mod15.select_nonlinear_hyperparameters(
        data_a, **kwargs
    )

    result_b = mod15.select_nonlinear_hyperparameters(
        data_b, **kwargs
    )

    assert (
        result_a["best_params"]
        == result_b["best_params"]
    )

    assert result_a["best_cv_mae"] == pytest.approx(
        result_b["best_cv_mae"]
    )

    pd.testing.assert_frame_equal(
        result_a["cv_results"],
        result_b["cv_results"],
    )


def test_tree_selects_lower_cv_mae(
    mod15, synthetic_trials
):
    result = mod15.select_nonlinear_hyperparameters(
        synthetic_trials,
        condition="Visual",
        model_name="decision_tree_regression",
        development_ids=_development_ids(),
        candidate_params=[
            _tree_candidate(depth=1),
            _tree_candidate(depth=2),
        ],
        n_splits=5,
        random_state=42,
    )

    # The synthetic relationship has four discrete
    # target stages. Depth 2 can separate all four.
    assert result["best_params"] == _tree_candidate(
        depth=2
    )

    cv_results = result["cv_results"]

    assert len(cv_results) == 2
    assert "cv_mae" in cv_results.columns

    assert np.isfinite(
        cv_results["cv_mae"]
    ).all()

    assert result["best_cv_mae"] == pytest.approx(
        cv_results["cv_mae"].min()
    )


def test_tied_models_prefer_simpler_configuration(
    mod15, synthetic_trials
):
    data = synthetic_trials.copy()

    # Constant target gives identical predictions
    # across the tested tree configurations.
    data.loc[
        data["condition"] == "Visual",
        "difficulty_stage",
    ] = 2

    result = mod15.select_nonlinear_hyperparameters(
        data,
        condition="Visual",
        model_name="decision_tree_regression",
        development_ids=_development_ids(),
        candidate_params=[
            _tree_candidate(depth=None, leaf=1),
            _tree_candidate(depth=2, leaf=4),
            _tree_candidate(depth=2, leaf=1),
        ],
        n_splits=5,
        random_state=42,
    )

    assert result["best_params"] == {
        "max_depth": 2,
        "min_samples_leaf": 4,
    }


def test_random_forest_selection_runs(
    mod15, synthetic_trials
):
    result = mod15.select_nonlinear_hyperparameters(
        synthetic_trials,
        condition="Auditory",
        model_name="random_forest_regression",
        development_ids=_development_ids(),
        candidate_params=[
            _forest_candidate(trees=7),
        ],
        n_splits=5,
        random_state=42,
    )

    assert result["best_params"] == (
        _forest_candidate(trees=7)
    )

    assert np.isfinite(
        result["best_cv_mae"]
    )

    assert len(result["cv_results"]) == 1


def test_random_forest_selection_is_reproducible(
    mod15, synthetic_trials
):
    kwargs = {
        "condition": "Auditory",
        "model_name": "random_forest_regression",
        "development_ids": _development_ids(),
        "candidate_params": [
            _forest_candidate(trees=7),
        ],
        "n_splits": 5,
        "random_state": 42,
    }

    first = mod15.select_nonlinear_hyperparameters(
        synthetic_trials,
        **kwargs,
    )

    second = mod15.select_nonlinear_hyperparameters(
        synthetic_trials,
        **kwargs,
    )

    assert first["best_params"] == second["best_params"]

    assert first["best_cv_mae"] == pytest.approx(
        second["best_cv_mae"]
    )

    pd.testing.assert_frame_equal(
        first["cv_results"],
        second["cv_results"],
    )


def test_unknown_nonlinear_model_is_rejected(
    mod15, synthetic_trials
):
    with pytest.raises(ValueError):
        mod15.select_nonlinear_hyperparameters(
            synthetic_trials,
            condition="Visual",
            model_name="multiple_linear_regression",
            development_ids=_development_ids(),
            candidate_params=[],
            n_splits=5,
            random_state=42,
        )




# ============================================================
# MOD-15 — Batch 3 RED tests
# Final model fitting and evaluation
# ============================================================

from sklearn.metrics import r2_score


@pytest.fixture
def batch3_trials(synthetic_trials):
    """
    Construct reproducible synthetic trials with
    independent predictor variation.

    Unlike the original Batch 1 fixture, this produces
    a full-rank OLS design matrix.
    """
    data = synthetic_trials.copy(deep=True)

    rng = np.random.default_rng(1503)
    stages = data["difficulty_stage"].to_numpy()

    predictors = sorted(
        set(VISUAL_PREDICTORS)
        | set(AUDITORY_PREDICTORS)
    )

    for index, predictor in enumerate(predictors):
        data[predictor] = (
            stages * (0.3 + index * 0.07)
            + rng.normal(
                loc=0.0,
                scale=1.2,
                size=len(data),
            )
        )

    return data


def _batch3_selected_params():
    """
    Small forest for synthetic testing only.

    Official MOD-15 selection retains 500 trees.
    """
    return {
        "decision_tree_regression": {
            "max_depth": 3,
            "min_samples_leaf": 2,
        },
        "random_forest_regression": {
            "n_estimators": 9,
            "max_depth": 3,
            "min_samples_leaf": 2,
            "max_features": "sqrt",
        },
    }


class _Batch3Predictor:
    """
    Deterministic test double for evaluation.

    Deliberately provides predict() but no fit(),
    so evaluation cannot refit the model.
    """

    def __init__(self, mode="constant"):
        self.mode = mode
        self.calls = []

    def predict(self, X):
        X = np.asarray(X, dtype=float)
        self.calls.append(X.copy())

        if self.mode == "sequence":
            return np.linspace(
                -1.25,
                4.25,
                num=len(X),
            )

        if self.mode == "large_error":
            return np.full(len(X), 10.0)

        return np.full(len(X), 1.5)


def _batch3_fake_models(mode="constant"):
    return {
        name: _Batch3Predictor(mode)
        for name in MODEL_NAMES
    }


def test_batch3_final_models_fit_development_only(
    mod15, batch3_trials, monkeypatch
):
    data = batch3_trials.copy()

    # Extreme unseen-participant values must never
    # appear in any model's fitting matrix.
    data.loc[
        data["participant_id"].isin(_test_ids()),
        "median_time_to_target_seconds",
    ] = 10000.0

    fitted_data = {}

    model_classes = {
        "multiple_linear_regression":
            mod15.DirectOLSRegressor,
        "decision_tree_regression":
            DecisionTreeRegressor,
        "random_forest_regression":
            RandomForestRegressor,
    }

    for name, model_class in model_classes.items():
        original_fit = model_class.fit

        def make_spy(model_name, original):
            def audited_fit(self, X, y, *args, **kwargs):
                fitted_data.setdefault(
                    model_name, []
                ).append((
                    np.asarray(X, dtype=float).copy(),
                    np.asarray(y, dtype=float).copy(),
                ))

                return original(
                    self, X, y, *args, **kwargs
                )
            return audited_fit

        monkeypatch.setattr(
            model_class,
            "fit",
            make_spy(name, original_fit),
        )

    models = mod15.fit_final_condition_models(
        data,
        condition="Visual",
        development_ids=_development_ids(),
        selected_hyperparameters=_batch3_selected_params(),
        random_state=42,
    )

    assert set(models) == MODEL_NAMES

    expected_rows = mod15.prepare_condition_rows(
        data,
        condition="Visual",
        participant_ids=_development_ids(),
    )

    expected_X = expected_rows[
        VISUAL_PREDICTORS
    ].to_numpy(dtype=float)

    expected_y = expected_rows[
        "difficulty_stage"
    ].to_numpy(dtype=float)

    for name in MODEL_NAMES:
        # Exactly one final fit per algorithm.
        assert len(fitted_data[name]) == 1

        actual_X, actual_y = fitted_data[name][0]

        assert actual_X.shape == (104, 7)
        assert actual_y.shape == (104,)

        np.testing.assert_allclose(
            actual_X, expected_X
        )
        np.testing.assert_allclose(
            actual_y, expected_y
        )

        assert np.max(actual_X) < 1000.0


def test_batch3_uses_selected_hyperparameters(
    mod15, batch3_trials
):
    models = mod15.fit_final_condition_models(
        batch3_trials,
        condition="Auditory",
        development_ids=_development_ids(),
        selected_hyperparameters=_batch3_selected_params(),
        random_state=42,
    )

    tree = models["decision_tree_regression"]
    forest = models["random_forest_regression"]

    assert isinstance(tree, DecisionTreeRegressor)
    assert isinstance(forest, RandomForestRegressor)

    assert tree.max_depth == 3
    assert tree.min_samples_leaf == 2

    assert forest.n_estimators == 9
    assert forest.max_depth == 3
    assert forest.min_samples_leaf == 2
    assert forest.max_features == "sqrt"

    assert tree.random_state == 42
    assert forest.random_state == 42

    assert hasattr(tree, "tree_")
    assert hasattr(forest, "estimators_")
    assert len(forest.estimators_) == 9


def test_batch3_performance_output_contract(
    mod15, batch3_trials
):
    models = mod15.fit_final_condition_models(
        batch3_trials,
        condition="Visual",
        development_ids=_development_ids(),
        selected_hyperparameters=_batch3_selected_params(),
        random_state=42,
    )

    result = mod15.evaluate_final_condition_models(
        models=models,
        data=batch3_trials,
        condition="Visual",
        development_ids=_development_ids(),
        test_ids=_test_ids(),
    )

    assert set(result) == {
        "performance",
        "training_predictions",
        "test_predictions",
    }

    performance = result["performance"]
    train = result["training_predictions"]
    test = result["test_predictions"]

    required_performance_columns = [
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

    assert list(performance.columns) == (
        required_performance_columns
    )

    assert len(performance) == 3
    assert set(performance["model"]) == MODEL_NAMES
    assert set(performance["condition"]) == {"Visual"}

    assert (
        performance["development_participant_count"]
        == 26
    ).all()

    assert (
        performance["test_participant_count"]
        == 6
    ).all()

    assert (
        performance["development_row_count"]
        == 104
    ).all()

    assert (
        performance["test_row_count"]
        == 24
    ).all()

    required_prediction_columns = [
        "participant_id",
        "condition",
        "model",
        "observed_difficulty_stage",
        "predicted_difficulty_stage",
        "error",
        "absolute_error",
    ]

    assert list(train.columns) == (
        required_prediction_columns
    )

    assert list(test.columns) == (
        required_prediction_columns
    )

    assert len(train) == 3 * 104
    assert len(test) == 3 * 24

    assert set(train["participant_id"]) == (
        _development_ids()
    )

    assert set(test["participant_id"]) == (
        _test_ids()
    )

    assert set(train["model"]) == MODEL_NAMES
    assert set(test["model"]) == MODEL_NAMES


def test_batch3_evaluation_uses_raw_predictors(
    mod15, batch3_trials
):
    models = _batch3_fake_models()

    mod15.evaluate_final_condition_models(
        models=models,
        data=batch3_trials,
        condition="Visual",
        development_ids=_development_ids(),
        test_ids=_test_ids(),
    )

    expected_train = mod15.prepare_condition_rows(
        batch3_trials,
        condition="Visual",
        participant_ids=_development_ids(),
    )[VISUAL_PREDICTORS].to_numpy(dtype=float)

    expected_test = mod15.prepare_condition_rows(
        batch3_trials,
        condition="Visual",
        participant_ids=_test_ids(),
    )[VISUAL_PREDICTORS].to_numpy(dtype=float)

    for model in models.values():
        # Once for training, once for test.
        assert len(model.calls) == 2

        np.testing.assert_allclose(
            model.calls[0],
            expected_train,
        )

        np.testing.assert_allclose(
            model.calls[1],
            expected_test,
        )


def test_batch3_predictions_are_not_rounded_or_clipped(
    mod15, batch3_trials
):
    models = _batch3_fake_models(
        mode="sequence"
    )

    result = mod15.evaluate_final_condition_models(
        models=models,
        data=batch3_trials,
        condition="Auditory",
        development_ids=_development_ids(),
        test_ids=_test_ids(),
    )

    predictions = result["test_predictions"]

    one_model = predictions.loc[
        predictions["model"]
        == "multiple_linear_regression"
    ]

    actual = one_model[
        "predicted_difficulty_stage"
    ].to_numpy(dtype=float)

    expected = np.linspace(
        -1.25,
        4.25,
        num=24,
    )

    np.testing.assert_allclose(
        actual, expected
    )

    assert actual.min() < 0
    assert actual.max() > 3

    assert np.any(
        np.abs(actual - np.round(actual)) > 1e-8
    )

    observed = one_model[
        "observed_difficulty_stage"
    ].to_numpy(dtype=float)

    np.testing.assert_allclose(
        one_model["error"].to_numpy(dtype=float),
        observed - actual,
    )

    np.testing.assert_allclose(
        one_model["absolute_error"].to_numpy(
            dtype=float
        ),
        np.abs(observed - actual),
    )


def test_batch3_mae_r2_and_differences(
    mod15, batch3_trials
):
    data = batch3_trials.copy()

    # Remove three eligible test observations from
    # P27 while keeping that participant represented.
    #
    # This checks participant-balanced MAE when
    # participants contribute unequal row counts.
    missing_mask = (
        (data["participant_id"] == "P27")
        & (data["condition"] == "Auditory")
        & (data["difficulty_stage"].isin([1, 2, 3]))
    )

    data.loc[
        missing_mask,
        "median_time_to_target_seconds",
    ] = np.nan

    models = _batch3_fake_models(
        mode="large_error"
    )

    result = mod15.evaluate_final_condition_models(
        models=models,
        data=data,
        condition="Auditory",
        development_ids=_development_ids(),
        test_ids=_test_ids(),
    )

    performance = result["performance"]

    for _, summary in performance.iterrows():
        name = summary["model"]

        train = result["training_predictions"].loc[
            lambda frame: frame["model"] == name
        ]

        test = result["test_predictions"].loc[
            lambda frame: frame["model"] == name
        ]

        expected_train_mae = (
            train.groupby("participant_id")[
                "absolute_error"
            ].mean().mean()
        )

        expected_test_mae = (
            test.groupby("participant_id")[
                "absolute_error"
            ].mean().mean()
        )

        expected_train_r2 = r2_score(
            train["observed_difficulty_stage"],
            train["predicted_difficulty_stage"],
        )

        expected_test_r2 = r2_score(
            test["observed_difficulty_stage"],
            test["predicted_difficulty_stage"],
        )

        assert summary["development_row_count"] == 104
        assert summary["test_row_count"] == 21
        assert summary["test_participant_count"] == 6

        assert summary["training_mae"] == pytest.approx(
            expected_train_mae
        )

        assert summary["test_mae"] == pytest.approx(
            expected_test_mae
        )

        assert summary["training_r2"] == pytest.approx(
            expected_train_r2
        )

        assert summary["test_r2"] == pytest.approx(
            expected_test_r2
        )

        assert summary[
            "test_minus_training_mae"
        ] == pytest.approx(
            expected_test_mae - expected_train_mae
        )

        assert summary[
            "training_minus_test_r2"
        ] == pytest.approx(
            expected_train_r2 - expected_test_r2
        )

        # Negative R² must not be replaced with zero.
        assert summary["test_r2"] < 0
        assert summary["training_r2"] < 0


def test_batch3_evaluation_never_refits(
    mod15, batch3_trials, monkeypatch
):
    models = mod15.fit_final_condition_models(
        batch3_trials,
        condition="Auditory",
        development_ids=_development_ids(),
        selected_hyperparameters=_batch3_selected_params(),
        random_state=42,
    )

    def forbidden_fit(*args, **kwargs):
        raise AssertionError(
            "Evaluation must not refit any model."
        )

    monkeypatch.setattr(
        mod15.DirectOLSRegressor,
        "fit",
        forbidden_fit,
    )

    monkeypatch.setattr(
        DecisionTreeRegressor,
        "fit",
        forbidden_fit,
    )

    monkeypatch.setattr(
        RandomForestRegressor,
        "fit",
        forbidden_fit,
    )

    result = mod15.evaluate_final_condition_models(
        models=models,
        data=batch3_trials,
        condition="Auditory",
        development_ids=_development_ids(),
        test_ids=_test_ids(),
    )

    assert len(result["performance"]) == 3


def test_batch3_same_eligible_rows_across_models(
    mod15, batch3_trials
):
    data = batch3_trials.copy()

    data.loc[
        (
            (data["participant_id"] == "P01")
            & (data["condition"] == "Visual")
            & (data["difficulty_stage"] == 2)
        ),
        "median_reach_duration_seconds",
    ] = np.nan

    data.loc[
        (
            (data["participant_id"] == "P27")
            & (data["condition"] == "Visual")
            & (data["difficulty_stage"] == 1)
        ),
        "median_reach_duration_seconds",
    ] = np.nan

    models = mod15.fit_final_condition_models(
        data,
        condition="Visual",
        development_ids=_development_ids(),
        selected_hyperparameters=_batch3_selected_params(),
        random_state=42,
    )

    result = mod15.evaluate_final_condition_models(
        models=models,
        data=data,
        condition="Visual",
        development_ids=_development_ids(),
        test_ids=_test_ids(),
    )

    performance = result["performance"]

    assert (
        performance["development_row_count"] == 103
    ).all()

    assert (
        performance["test_row_count"] == 23
    ).all()

    for name in MODEL_NAMES:
        train = result["training_predictions"]
        test = result["test_predictions"]

        assert len(
            train.loc[train["model"] == name]
        ) == 103

        assert len(
            test.loc[test["model"] == name]
        ) == 23


def test_batch3_rejects_overlapping_splits(
    mod15, batch3_trials
):
    models = _batch3_fake_models()

    invalid_test_ids = set(_test_ids())
    invalid_test_ids.remove("P27")
    invalid_test_ids.add("P26")

    with pytest.raises(ValueError):
        mod15.evaluate_final_condition_models(
            models=models,
            data=batch3_trials,
            condition="Visual",
            development_ids=_development_ids(),
            test_ids=invalid_test_ids,
        )


def test_batch3_preserves_input_dataframe(
    mod15, batch3_trials
):
    data = batch3_trials.copy(deep=True)
    original = data.copy(deep=True)

    models = mod15.fit_final_condition_models(
        data,
        condition="Visual",
        development_ids=_development_ids(),
        selected_hyperparameters=_batch3_selected_params(),
        random_state=42,
    )

    mod15.evaluate_final_condition_models(
        models=models,
        data=data,
        condition="Visual",
        development_ids=_development_ids(),
        test_ids=_test_ids(),
    )

    pd.testing.assert_frame_equal(
        data,
        original,
    )


def test_batch3_direct_ols_rejects_rank_deficiency(
    mod15
):
    # Second predictor duplicates the first.
    X = np.array([
        [1.0, 2.0],
        [2.0, 4.0],
        [3.0, 6.0],
        [4.0, 8.0],
    ])

    y = np.array([
        0.0, 1.0, 2.0, 3.0
    ])

    model = mod15.DirectOLSRegressor()

    with pytest.raises(ValueError):
        model.fit(X, y)



# ============================================================
# MOD-15 — Batch 4 RED tests
# CSV outputs, figures, and standalone CLI
# ============================================================

import matplotlib
matplotlib.use("Agg")

from matplotlib.figure import Figure


BATCH4_CSV_FILES = [
    "model_comparison_performance.csv",
    "selected_hyperparameters.csv",
    "development_cv_results.csv",
    "training_predictions.csv",
    "test_predictions.csv",
]

BATCH4_FIGURE_FILES = [
    "visual_linear_regression_mae.png",
    "visual_linear_regression_r2.png",
    "visual_decision_tree_mae.png",
    "visual_decision_tree_r2.png",
    "visual_random_forest_mae.png",
    "visual_random_forest_r2.png",
    "auditory_linear_regression_mae.png",
    "auditory_linear_regression_r2.png",
    "auditory_decision_tree_mae.png",
    "auditory_decision_tree_r2.png",
    "auditory_random_forest_mae.png",
    "auditory_random_forest_r2.png",
    "visual_model_comparison_mae.png",
    "visual_model_comparison_r2.png",
    "auditory_model_comparison_mae.png",
    "auditory_model_comparison_r2.png",
]


@pytest.fixture
def batch4_tables():
    """Small synthetic tables for output and chart tests."""

    performance_rows = []

    model_names = [
        "multiple_linear_regression",
        "decision_tree_regression",
        "random_forest_regression",
    ]

    for condition in ["Visual", "Auditory"]:
        for index, model_name in enumerate(model_names):
            training_mae = 0.3 + 0.1 * index
            test_mae = 0.6 + 0.1 * index

            training_r2 = 0.7 - 0.1 * index
            test_r2 = -0.30 + 0.05 * index

            performance_rows.append({
                "condition": condition,
                "model": model_name,
                "development_participant_count": 26,
                "development_row_count": 104,
                "test_participant_count": 6,
                "test_row_count": 24,
                "training_mae": training_mae,
                "test_mae": test_mae,
                "test_minus_training_mae":
                    test_mae - training_mae,
                "training_r2": training_r2,
                "test_r2": test_r2,
                "training_minus_test_r2":
                    training_r2 - test_r2,
            })

    performance = pd.DataFrame(
        performance_rows
    )

    selected_rows = []
    cv_rows = []

    for condition in ["Visual", "Auditory"]:
        for model_name in [
            "decision_tree_regression",
            "random_forest_regression",
        ]:
            is_forest = (
                model_name == "random_forest_regression"
            )

            row = {
                "condition": condition,
                "model": model_name,
                "max_depth": 3,
                "min_samples_leaf": 2,
                "n_estimators":
                    500 if is_forest else np.nan,
                "max_features":
                    "sqrt" if is_forest else np.nan,
                "cv_mae": 0.55,
            }

            selected_rows.append(row)

            cv_rows.append({
                **row,
                "candidate_id": 1,
                "n_splits": 5,
                "development_participant_count": 26,
                "development_row_count": 104,
                "selected": True,
            })

    selected_hyperparameters = pd.DataFrame(
        selected_rows
    )

    development_cv_results = pd.DataFrame(
        cv_rows
    )

    prediction_rows = []

    for condition in ["Visual", "Auditory"]:
        for model_name in model_names:
            for participant_id in ["P01", "P27"]:
                observed = 2.0
                predicted = 1.7

                prediction_rows.append({
                    "participant_id": participant_id,
                    "condition": condition,
                    "model": model_name,
                    "observed_difficulty_stage": observed,
                    "predicted_difficulty_stage": predicted,
                    "error": observed - predicted,
                    "absolute_error": abs(
                        observed - predicted
                    ),
                })

    all_predictions = pd.DataFrame(
        prediction_rows
    )

    training_predictions = (
        all_predictions.loc[
            all_predictions["participant_id"] == "P01"
        ].reset_index(drop=True)
    )

    test_predictions = (
        all_predictions.loc[
            all_predictions["participant_id"] == "P27"
        ].reset_index(drop=True)
    )

    return {
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


def _write_batch4_tables(
    mod15,
    tables,
    output_dir,
    overwrite=False,
):
    return mod15.write_mod15_outputs(
        performance=tables["performance"],
        selected_hyperparameters=(
            tables["selected_hyperparameters"]
        ),
        development_cv_results=(
            tables["development_cv_results"]
        ),
        training_predictions=(
            tables["training_predictions"]
        ),
        test_predictions=(
            tables["test_predictions"]
        ),
        output_directory=output_dir,
        overwrite=overwrite,
    )


def test_batch4_writes_exact_five_csv_files(
    mod15, batch4_tables, tmp_path
):
    output_dir = tmp_path / "mod15"

    _write_batch4_tables(
        mod15,
        batch4_tables,
        output_dir,
    )

    actual_files = {
        path.name
        for path in output_dir.iterdir()
        if path.is_file()
    }

    assert actual_files == set(
        BATCH4_CSV_FILES
    )

    for name in BATCH4_CSV_FILES:
        path = output_dir / name

        assert path.is_file()
        assert path.stat().st_size > 0


def test_batch4_csv_contents_are_preserved(
    mod15, batch4_tables, tmp_path
):
    output_dir = tmp_path / "mod15"

    _write_batch4_tables(
        mod15,
        batch4_tables,
        output_dir,
    )

    filenames = {
        "performance":
            "model_comparison_performance.csv",
        "selected_hyperparameters":
            "selected_hyperparameters.csv",
        "development_cv_results":
            "development_cv_results.csv",
        "training_predictions":
            "training_predictions.csv",
        "test_predictions":
            "test_predictions.csv",
    }

    for key, filename in filenames.items():
        actual = pd.read_csv(
            output_dir / filename
        )

        expected = batch4_tables[key]

        assert list(actual.columns) == list(
            expected.columns
        )

        assert len(actual) == len(expected)

        pd.testing.assert_frame_equal(
            actual.reset_index(drop=True),
            expected.reset_index(drop=True),
            check_dtype=False,
            check_exact=False,
            rtol=1e-10,
            atol=1e-10,
        )


def test_batch4_csv_overwrite_is_blocked(
    mod15, batch4_tables, tmp_path
):
    output_dir = tmp_path / "mod15"
    output_dir.mkdir()

    # Place a pre-existing output near the end of
    # the file list to test overwrite preflight.
    protected = (
        output_dir / "test_predictions.csv"
    )

    protected.write_text(
        "EXISTING OUTPUT",
        encoding="utf-8",
    )

    with pytest.raises(FileExistsError):
        _write_batch4_tables(
            mod15,
            batch4_tables,
            output_dir,
            overwrite=False,
        )

    assert protected.read_text(
        encoding="utf-8"
    ) == "EXISTING OUTPUT"

    # No earlier file should have been written.
    assert {
        path.name
        for path in output_dir.iterdir()
    } == {"test_predictions.csv"}


def test_batch4_explicit_csv_overwrite(
    mod15, batch4_tables, tmp_path
):
    output_dir = tmp_path / "mod15"
    output_dir.mkdir()

    protected = (
        output_dir / "test_predictions.csv"
    )

    protected.write_text(
        "EXISTING OUTPUT",
        encoding="utf-8",
    )

    _write_batch4_tables(
        mod15,
        batch4_tables,
        output_dir,
        overwrite=True,
    )

    assert set(
        path.name
        for path in output_dir.iterdir()
    ) == set(BATCH4_CSV_FILES)

    assert "EXISTING OUTPUT" not in (
        protected.read_text(encoding="utf-8")
    )


def test_batch4_generates_all_sixteen_figures(
    mod15, batch4_tables, tmp_path
):
    output_dir = tmp_path / "figures"

    mod15.create_mod15_figures(
        performance=batch4_tables["performance"],
        output_directory=output_dir,
    )

    actual_files = {
        path.name
        for path in output_dir.iterdir()
        if path.is_file()
    }

    assert actual_files == set(
        BATCH4_FIGURE_FILES
    )

    for name in BATCH4_FIGURE_FILES:
        path = output_dir / name

        assert path.stat().st_size > 1000

        # Verify actual PNG format.
        with path.open("rb") as file:
            assert file.read(8) == (
                b"\x89PNG\r\n\x1a\n"
            )


def test_batch4_charts_have_correct_bars_and_labels(
    mod15, batch4_tables, tmp_path, monkeypatch
):
    captured = {}

    def inspect_savefig(
        figure,
        filename,
        *args,
        **kwargs,
    ):
        ax = figure.axes[0]
        name = Path(filename).name

        bar_heights = [
            patch.get_height()
            for patch in ax.patches
        ]

        zero_reference = any(
            len(line.get_ydata()) >= 2
            and np.allclose(
                np.asarray(
                    line.get_ydata(),
                    dtype=float,
                ),
                0.0,
            )
            for line in ax.lines
        )

        captured[name] = {
            "title": ax.get_title(),
            "bar_heights": bar_heights,
            "number_of_labels": len(ax.texts),
            "zero_reference": zero_reference,
            "dpi": kwargs.get("dpi", 0),
            "legend": (
                [
                    item.get_text()
                    for item in ax.get_legend().get_texts()
                ]
                if ax.get_legend() is not None
                else []
            ),
        }

    monkeypatch.setattr(
        Figure,
        "savefig",
        inspect_savefig,
    )

    mod15.create_mod15_figures(
        performance=batch4_tables["performance"],
        output_directory=tmp_path / "figures",
    )

    assert set(captured) == set(
        BATCH4_FIGURE_FILES
    )

    for filename, chart in captured.items():

        expected_bars = (
            6 if "model_comparison" in filename
            else 2
        )

        assert len(chart["bar_heights"]) == (
            expected_bars
        )

        # Every bar requires a numerical label.
        assert chart["number_of_labels"] >= (
            expected_bars
        )

        assert chart["dpi"] >= 200

        if filename.endswith("_r2.png"):
            assert chart["zero_reference"]

            # Synthetic test performance includes
            # negative R² values.
            assert any(
                height < 0
                for height in chart["bar_heights"]
            )

        if "model_comparison" in filename:
            assert "Training" in chart["legend"]
            assert "Test" in chart["legend"]


def test_batch4_figures_are_closed_after_saving(
    mod15, batch4_tables, tmp_path
):
    import matplotlib.pyplot as plt

    before = set(plt.get_fignums())

    mod15.create_mod15_figures(
        performance=batch4_tables["performance"],
        output_directory=tmp_path / "figures",
    )

    after = set(plt.get_fignums())

    assert after == before


def test_batch4_figure_overwrite_is_blocked(
    mod15, batch4_tables, tmp_path
):
    output_dir = tmp_path / "figures"
    output_dir.mkdir()

    protected = (
        output_dir
        / "auditory_model_comparison_r2.png"
    )

    protected.write_bytes(
        b"EXISTING FIGURE"
    )

    with pytest.raises(FileExistsError):
        mod15.create_mod15_figures(
            performance=batch4_tables["performance"],
            output_directory=output_dir,
            overwrite=False,
        )

    assert protected.read_bytes() == (
        b"EXISTING FIGURE"
    )

    # Preflight must prevent partial figure output.
    assert {
        path.name
        for path in output_dir.iterdir()
    } == {protected.name}


def test_batch4_full_workflow_with_synthetic_data(
    mod15,
    batch3_trials,
    split_frame,
    tmp_path,
    monkeypatch,
):
    """
    Integration test using 32 synthetic participants.

    Restricts CV to one candidate per nonlinear model
    to keep the test fast. Official analysis still uses
    the complete hyperparameter search.
    """

    data_path = tmp_path / "synthetic_trials.csv"
    split_path = tmp_path / "participant_holdout_split.csv"
    output_dir = tmp_path / "results"

    batch3_trials.to_csv(
        data_path,
        index=False,
    )

    split_frame.to_csv(
        split_path,
        index=False,
    )

    original_selection = (
        mod15.select_nonlinear_hyperparameters
    )

    def fast_synthetic_selection(
        data,
        condition,
        model_name,
        development_ids,
        **kwargs,
    ):
        if model_name == "decision_tree_regression":
            candidate = {
                "max_depth": 3,
                "min_samples_leaf": 2,
            }
        else:
            candidate = {
                "n_estimators": 9,
                "max_depth": 3,
                "min_samples_leaf": 2,
                "max_features": "sqrt",
            }

        return original_selection(
            data=data,
            condition=condition,
            model_name=model_name,
            development_ids=development_ids,
            candidate_params=[candidate],
            n_splits=5,
            random_state=kwargs.get(
                "random_state", 42
            ),
        )

    monkeypatch.setattr(
        mod15,
        "select_nonlinear_hyperparameters",
        fast_synthetic_selection,
    )

    mod15.run_mod15_analysis(
        data_path=data_path,
        split_path=split_path,
        output_directory=output_dir,
        random_state=42,
    )

    actual_files = {
        path.name
        for path in output_dir.iterdir()
        if path.is_file()
    }

    assert actual_files == (
        set(BATCH4_CSV_FILES)
        | set(BATCH4_FIGURE_FILES)
    )

    performance = pd.read_csv(
        output_dir
        / "model_comparison_performance.csv"
    )

    assert len(performance) == 6

    assert set(performance["condition"]) == {
        "Visual", "Auditory"
    }

    assert set(performance["model"]) == (
        MODEL_NAMES
    )

    assert (
        performance["development_participant_count"]
        == 26
    ).all()

    assert (
        performance["test_participant_count"]
        == 6
    ).all()

    selected = pd.read_csv(
        output_dir / "selected_hyperparameters.csv"
    )

    assert len(selected) == 4

    assert set(selected["model"]) == {
        "decision_tree_regression",
        "random_forest_regression",
    }

    cv_results = pd.read_csv(
        output_dir / "development_cv_results.csv"
    )

    assert len(cv_results) == 4

    training = pd.read_csv(
        output_dir / "training_predictions.csv"
    )

    testing = pd.read_csv(
        output_dir / "test_predictions.csv"
    )

    assert set(training["participant_id"]) == (
        _development_ids()
    )

    assert set(testing["participant_id"]) == (
        _test_ids()
    )

    assert len(training) == 2 * 3 * 104
    assert len(testing) == 2 * 3 * 24


def test_batch4_cli_passes_arguments(
    mod15, monkeypatch, tmp_path
):
    captured = []

    def fake_run(
        data_path,
        split_path,
        output_directory,
        overwrite=False,
        random_state=42,
    ):
        captured.append({
            "data_path": Path(data_path),
            "split_path": Path(split_path),
            "output_directory": Path(
                output_directory
            ),
            "overwrite": overwrite,
            "random_state": random_state,
        })

    monkeypatch.setattr(
        mod15,
        "run_mod15_analysis",
        fake_run,
    )

    data_path = tmp_path / "data.csv"
    split_path = tmp_path / "split.csv"
    output_dir = tmp_path / "outputs"

    mod15.main([
        "--data", str(data_path),
        "--split", str(split_path),
        "--output-dir", str(output_dir),
        "--random-state", "123",
        "--overwrite",
    ])

    assert len(captured) == 1

    assert captured[0] == {
        "data_path": data_path,
        "split_path": split_path,
        "output_directory": output_dir,
        "overwrite": True,
        "random_state": 123,
    }


def test_batch4_default_cli_overwrite_is_false(
    mod15, monkeypatch, tmp_path
):
    captured = []

    def fake_run(
        data_path,
        split_path,
        output_directory,
        overwrite=False,
        random_state=42,
    ):
        captured.append(overwrite)

    monkeypatch.setattr(
        mod15,
        "run_mod15_analysis",
        fake_run,
    )

    mod15.main([
        "--data", str(tmp_path / "data.csv"),
        "--split", str(tmp_path / "split.csv"),
        "--output-dir", str(tmp_path / "results"),
    ])

    assert captured == [False]



def test_batch4_runner_rejects_invalid_split(
    mod15, batch3_trials, split_frame, tmp_path
):
    data_path = tmp_path / "synthetic_trials.csv"
    split_path = tmp_path / "invalid_split.csv"
    output_dir = tmp_path / "results"

    batch3_trials.to_csv(
        data_path,
        index=False,
    )

    # Deliberately remove one test participant.
    split_frame.iloc[:-1].to_csv(
        split_path,
        index=False,
    )

    with pytest.raises(ValueError):
        mod15.run_mod15_analysis(
            data_path=data_path,
            split_path=split_path,
            output_directory=output_dir,
        )

    assert not output_dir.exists()
