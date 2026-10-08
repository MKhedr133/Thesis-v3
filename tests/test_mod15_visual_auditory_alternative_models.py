
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
