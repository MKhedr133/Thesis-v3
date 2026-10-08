
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
