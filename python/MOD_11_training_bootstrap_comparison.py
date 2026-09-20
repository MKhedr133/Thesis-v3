#!/usr/bin/env python3
"""MOD-11 — Training-only participant bootstrap model comparison.

MOD-11 compares seven frozen fixed-effects candidate models using only the
26 participants assigned to the MOD-09 training set.

The six final test participants remain completely excluded from fitting,
bootstrap validation, diagnostics, tuning, and model-selection decisions.

This module currently implements the MOD-11 bootstrap foundation. Mixed-model
fitting, OOB prediction, metric aggregation, and output generation are added in
subsequent implementation stages.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd

import MOD_10_training_random_effects_comparison as MOD10


SCRIPT_VERSION = "1.0.0"

EXPECTED_TRAINING_PARTICIPANTS = 26
EXPECTED_TEST_PARTICIPANTS = 6

DEFAULT_BOOTSTRAP_REPLICATES = 500

RANDOM_STRUCTURES = ("RI",)

MODEL_IDS = (
    "M0",
    "MA",
    "MT",
    "MAT",
    "MDA",
    "MDT",
    "MDAT",
)

PARTICIPANT_COLUMN = "participant_id"
BOOTSTRAP_CLUSTER_COLUMN = "bootstrap_cluster_id"
BOOTSTRAP_DRAW_COLUMN = "bootstrap_draw_index"
BOOTSTRAP_REPLICATE_COLUMN = "bootstrap_replicate"

AGE_TERM = (
    "C(participant_group, Treatment(reference='Young'))"
)


class Mod11Error(ValueError):
    """Raised when a frozen MOD-11 data contract is violated."""


def build_candidate_model_rhs_registry() -> dict[str, str]:
    """Return the seven frozen MOD-11 fixed-effects candidates.

    ``{D}`` is replaced later by the appropriate difficulty variable for the
    target being modelled.

    TMT-B always enters in its original unit through ``tmt_b_seconds``.
    """
    return {
        "M0": "{D}",
        "MA": f"{{D}} + {AGE_TERM}",
        "MT": "{D} + tmt_b_seconds",
        "MAT": f"{{D}} + {AGE_TERM} + tmt_b_seconds",
        "MDA": (
            f"{{D}} + {AGE_TERM} + "
            f"{{D}}:{AGE_TERM}"
        ),
        "MDT": (
            "{D} + tmt_b_seconds + "
            "{D}:tmt_b_seconds"
        ),
        "MDAT": (
            f"{{D}} + {AGE_TERM} + tmt_b_seconds + "
            f"{{D}}:{AGE_TERM} + "
            "{D}:tmt_b_seconds"
        ),
    }


def prepare_training_data(
    *,
    modeling_data: pd.DataFrame,
    holdout_split: pd.DataFrame,
) -> tuple[
    pd.DataFrame,
    tuple[str, ...],
    tuple[str, ...],
]:
    """Return only the locked 26 MOD-09 training participants.

    MOD-10 already implements and tests the frozen MOD-09 split contract.
    MOD-11 reuses that boundary rather than maintaining a second independent
    implementation.
    """
    training_data, training_ids, test_ids = (
        MOD10.prepare_training_data(
            modeling_data=modeling_data,
            holdout_split=holdout_split,
        )
    )

    if len(training_ids) != EXPECTED_TRAINING_PARTICIPANTS:
        raise Mod11Error(
            "MOD-11 requires exactly 26 training participants"
        )

    if len(test_ids) != EXPECTED_TEST_PARTICIPANTS:
        raise Mod11Error(
            "MOD-11 requires exactly 6 final test participants"
        )

    observed_training = set(
        training_data[PARTICIPANT_COLUMN].astype(str)
    )

    if not observed_training.isdisjoint(test_ids):
        raise Mod11Error(
            "Final MOD-09 test participants entered MOD-11"
        )

    return training_data, training_ids, test_ids


def _validate_training_participant_ids(
    participant_ids: Sequence[str],
) -> tuple[str, ...]:
    """Validate and normalise the frozen training participant IDs."""
    ids = tuple(str(value).strip() for value in participant_ids)

    if len(ids) != EXPECTED_TRAINING_PARTICIPANTS:
        raise Mod11Error(
            "MOD-11 bootstrap requires exactly "
            "26 training participant IDs"
        )

    if any(not value for value in ids):
        raise Mod11Error(
            "Training participant IDs must not be blank"
        )

    if len(set(ids)) != EXPECTED_TRAINING_PARTICIPANTS:
        raise Mod11Error(
            "Training participant IDs must be unique"
        )

    return ids


def draw_participant_ids(
    training_participant_ids: Sequence[str],
    *,
    rng: np.random.Generator,
) -> tuple[str, ...]:
    """Draw 26 training participants with replacement.

    The output contains one entry per bootstrap draw. Therefore the same
    original participant may appear multiple times.
    """
    ids = _validate_training_participant_ids(
        training_participant_ids
    )

    sampled = rng.choice(
        np.asarray(ids, dtype=object),
        size=EXPECTED_TRAINING_PARTICIPANTS,
        replace=True,
    )

    return tuple(str(value) for value in sampled.tolist())


def generate_bootstrap_plan(
    training_participant_ids: Sequence[str],
    *,
    n_replicates: int,
    seed: int,
) -> tuple[tuple[str, ...], ...]:
    """Generate one reproducible participant bootstrap plan.

    The resulting plan should be generated once for an analysis and reused
    across all seven candidate models so model comparisons are based on the
    same participant resamples.

    The seed is supplied explicitly rather than silently fixed here because the
    official MOD-11 bootstrap seed has not yet been frozen as a thesis decision.
    """
    ids = _validate_training_participant_ids(
        training_participant_ids
    )

    if isinstance(n_replicates, bool):
        raise Mod11Error(
            "n_replicates must be a positive integer"
        )

    try:
        replicate_count = int(n_replicates)
    except (TypeError, ValueError) as exc:
        raise Mod11Error(
            "n_replicates must be a positive integer"
        ) from exc

    if replicate_count != n_replicates or replicate_count < 1:
        raise Mod11Error(
            "n_replicates must be a positive integer"
        )

    rng = np.random.default_rng(seed)

    return tuple(
        draw_participant_ids(
            ids,
            rng=rng,
        )
        for _ in range(replicate_count)
    )


def identify_oob_participants(
    training_participant_ids: Sequence[str],
    sampled_participant_ids: Sequence[str],
) -> tuple[str, ...]:
    """Return original training participants absent from one bootstrap sample."""
    training_ids = _validate_training_participant_ids(
        training_participant_ids
    )

    sampled = tuple(
        str(value).strip()
        for value in sampled_participant_ids
    )

    if len(sampled) != EXPECTED_TRAINING_PARTICIPANTS:
        raise Mod11Error(
            "A MOD-11 bootstrap replicate must contain "
            "exactly 26 participant draws"
        )

    unexpected = sorted(
        set(sampled) - set(training_ids)
    )

    if unexpected:
        raise Mod11Error(
            "Bootstrap sample contains IDs outside the "
            "locked training set: "
            + ", ".join(unexpected)
        )

    sampled_unique = set(sampled)

    return tuple(
        participant_id
        for participant_id in training_ids
        if participant_id not in sampled_unique
    )


def build_bootstrap_sample(
    training_data: pd.DataFrame,
    sampled_participant_ids: Sequence[str],
    *,
    replicate_index: int,
) -> pd.DataFrame:
    """Materialise one participant-cluster bootstrap training sample.

    All repeated rows belonging to a sampled participant are copied together.

    If an original participant is drawn multiple times, each draw receives a
    different ``bootstrap_cluster_id``. The original participant ID remains in
    ``participant_id`` for traceability, while the bootstrap cluster ID is the
    grouping variable that will later be supplied to the mixed model.
    """
    if PARTICIPANT_COLUMN not in training_data.columns:
        raise Mod11Error(
            f"Missing required column: {PARTICIPANT_COLUMN}"
        )

    sampled = tuple(
        str(value).strip()
        for value in sampled_participant_ids
    )

    if len(sampled) != EXPECTED_TRAINING_PARTICIPANTS:
        raise Mod11Error(
            "A MOD-11 bootstrap replicate must contain "
            "exactly 26 participant draws"
        )

    available_ids = set(
        training_data[PARTICIPANT_COLUMN]
        .astype(str)
        .str.strip()
    )

    unexpected = sorted(
        set(sampled) - available_ids
    )

    if unexpected:
        raise Mod11Error(
            "Bootstrap sample contains participants not present "
            "in training_data: "
            + ", ".join(unexpected)
        )

    copies: list[pd.DataFrame] = []

    participant_values = (
        training_data[PARTICIPANT_COLUMN]
        .astype(str)
        .str.strip()
    )

    for draw_index, participant_id in enumerate(
        sampled,
        start=1,
    ):
        participant_rows = training_data.loc[
            participant_values.eq(participant_id)
        ].copy()

        if participant_rows.empty:
            raise Mod11Error(
                "No repeated rows found for sampled participant "
                f"{participant_id}"
            )

        cluster_id = (
            f"b{int(replicate_index):04d}"
            f"_d{draw_index:02d}"
            f"_{participant_id}"
        )

        participant_rows[
            BOOTSTRAP_CLUSTER_COLUMN
        ] = cluster_id

        participant_rows[
            BOOTSTRAP_DRAW_COLUMN
        ] = draw_index

        participant_rows[
            BOOTSTRAP_REPLICATE_COLUMN
        ] = int(replicate_index)

        copies.append(participant_rows)

    bootstrap = pd.concat(
        copies,
        axis=0,
        ignore_index=True,
    )

    if (
        bootstrap[BOOTSTRAP_CLUSTER_COLUMN].nunique()
        != EXPECTED_TRAINING_PARTICIPANTS
    ):
        raise Mod11Error(
            "Every bootstrap draw must receive a distinct "
            "bootstrap cluster ID"
        )

    return bootstrap