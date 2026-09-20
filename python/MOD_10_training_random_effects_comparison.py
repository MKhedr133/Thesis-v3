#!/usr/bin/env python3
"""MOD-10 — Training-only RI versus RI+RS comparison.

MOD-10 uses the participant split created by MOD-09. Only the 26 participants
labelled as training participants may enter model fitting or training-only
cross validation. The 6 final test participants remain completely excluded
from model development.

This first implementation establishes and validates that data boundary.
"""

from __future__ import annotations

import pandas as pd
import MOD_02_random_effects_structure as MOD02

PARTICIPANT_COLUMN = "participant_id"
GROUP_COLUMN = "participant_group"
SPLIT_COLUMN = "split"

EXPECTED_PARTICIPANTS = 32
EXPECTED_TRAINING_PARTICIPANTS = 26
EXPECTED_TEST_PARTICIPANTS = 6

EXPECTED_TRAIN_GROUP_COUNTS = {
    "Young": 13,
    "Old": 13,
}

EXPECTED_TEST_GROUP_COUNTS = {
    "Young": 3,
    "Old": 3,
}


class Mod10Error(ValueError):
    """Raised when the MOD-10 training-data contract is violated."""


def build_target_specs() -> dict[str, MOD02.TargetSpec]:
    """Return the 10 Gaussian MOD-10 targets with the MDAT fixed specification.

    MOD-10 deliberately reuses the MOD-02 target registry so that the
    difficulty coding and fixed-effects formulas remain identical to the
    previously defined modelling specification.

    The common fixed-effects structure is MDAT:

        Difficulty
        + Age
        + TMT-B
        + Difficulty × Age
        + Difficulty × TMT-B

    Relative performance uses performance_difficulty_stage. The remaining
    Gaussian targets use difficulty_stage.
    """
    registry = MOD02.build_internal_registry()

    return {
        target: registry[target]
        for target in MOD02.GAUSSIAN_TARGETS
    }


def _normalise_string_column(
    frame: pd.DataFrame,
    column: str,
) -> pd.Series:
    """Return a stripped string column after validating missing/blank values."""
    if column not in frame.columns:
        raise Mod10Error(f"Missing required column: {column}")

    if frame[column].isna().any():
        raise Mod10Error(f"{column} contains missing values")

    values = frame[column].astype(str).str.strip()

    if values.eq("").any():
        raise Mod10Error(f"{column} contains blank values")

    return values


def validate_holdout_split(
    holdout_split: pd.DataFrame,
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Validate the MOD-09 split and return training and test IDs."""
    split = holdout_split.copy(deep=True)

    split[PARTICIPANT_COLUMN] = _normalise_string_column(
        split,
        PARTICIPANT_COLUMN,
    )
    split[GROUP_COLUMN] = _normalise_string_column(
        split,
        GROUP_COLUMN,
    )
    split[SPLIT_COLUMN] = _normalise_string_column(
        split,
        SPLIT_COLUMN,
    )

    if len(split) != EXPECTED_PARTICIPANTS:
        raise Mod10Error(
            "MOD-10 requires exactly 32 rows in the participant holdout split; "
            f"found {len(split)}"
        )

    if split[PARTICIPANT_COLUMN].nunique() != EXPECTED_PARTICIPANTS:
        raise Mod10Error(
            "MOD-10 requires exactly 32 unique participant IDs "
            "in the participant holdout split"
        )

    allowed_groups = {"Young", "Old"}
    observed_groups = set(split[GROUP_COLUMN])

    if observed_groups != allowed_groups:
        raise Mod10Error(
            "participant_group must contain only Young and Old"
        )

    allowed_split_labels = {"train", "test"}
    observed_split_labels = set(split[SPLIT_COLUMN])

    if observed_split_labels != allowed_split_labels:
        raise Mod10Error(
            "split must contain only train and test labels"
        )

    training = split.loc[split[SPLIT_COLUMN].eq("train")].copy()
    test = split.loc[split[SPLIT_COLUMN].eq("test")].copy()

    if len(training) != EXPECTED_TRAINING_PARTICIPANTS:
        raise Mod10Error(
            "MOD-10 requires exactly 26 training participants; "
            f"found {len(training)}"
        )

    if len(test) != EXPECTED_TEST_PARTICIPANTS:
        raise Mod10Error(
            "MOD-10 requires exactly 6 final test participants; "
            f"found {len(test)}"
        )

    training_group_counts = (
        training[GROUP_COLUMN]
        .value_counts()
        .to_dict()
    )
    test_group_counts = (
        test[GROUP_COLUMN]
        .value_counts()
        .to_dict()
    )

    if training_group_counts != EXPECTED_TRAIN_GROUP_COUNTS:
        raise Mod10Error(
            "Training split must contain exactly "
            "13 Young and 13 Old participants"
        )

    if test_group_counts != EXPECTED_TEST_GROUP_COUNTS:
        raise Mod10Error(
            "Test split must contain exactly "
            "3 Young and 3 Old participants"
        )

    training_ids = tuple(
        sorted(training[PARTICIPANT_COLUMN].tolist())
    )
    test_ids = tuple(
        sorted(test[PARTICIPANT_COLUMN].tolist())
    )

    if set(training_ids) & set(test_ids):
        raise Mod10Error(
            "Training and test participant IDs must be disjoint"
        )

    return training_ids, test_ids


def prepare_training_data(
    *,
    modeling_data: pd.DataFrame,
    holdout_split: pd.DataFrame,
) -> tuple[pd.DataFrame, tuple[str, ...], tuple[str, ...]]:
    """Return modelling rows belonging only to the 26 training participants."""
    training_ids, test_ids = validate_holdout_split(holdout_split)

    data = modeling_data.copy(deep=True)

    data[PARTICIPANT_COLUMN] = _normalise_string_column(
        data,
        PARTICIPANT_COLUMN,
    )

    modeling_ids = set(data[PARTICIPANT_COLUMN])
    split_ids = set(training_ids) | set(test_ids)

    missing_from_modeling = sorted(split_ids - modeling_ids)
    unexpected_in_modeling = sorted(modeling_ids - split_ids)

    if missing_from_modeling:
        raise Mod10Error(
            "Participants from the MOD-09 split are missing from modeling_data: "
            + ", ".join(missing_from_modeling)
        )

    if unexpected_in_modeling:
        raise Mod10Error(
            "modeling_data contains participants not present in the MOD-09 split: "
            + ", ".join(unexpected_in_modeling)
        )

    training_data = data.loc[
        data[PARTICIPANT_COLUMN].isin(training_ids)
    ].copy()

    observed_training_ids = set(training_data[PARTICIPANT_COLUMN])

    if observed_training_ids != set(training_ids):
        raise Mod10Error(
            "Filtered MOD-10 data do not contain exactly the "
            "26 expected training participants"
        )

    if not observed_training_ids.isdisjoint(test_ids):
        raise Mod10Error(
            "Final test participants entered MOD-10 training data"
        )

    if training_data[PARTICIPANT_COLUMN].nunique() != (
        EXPECTED_TRAINING_PARTICIPANTS
    ):
        raise Mod10Error(
            "MOD-10 training data must contain exactly 26 participants"
        )

    return training_data, training_ids, test_ids