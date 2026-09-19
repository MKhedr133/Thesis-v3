#!/usr/bin/env python3
"""MOD-06 — Deterministic participant-level training/test split.

This module implements the frozen 26/6 participant holdout procedure used before
final model selection. The split decision uses only participant identity and the
recorded Young/Old group. TMT-B, outcomes, behavioural measurements, condition,
difficulty, and model results are deliberately ignored.

The official CLI has no seed override. Changing the split specification therefore
requires an explicit code change and Git review rather than an ad-hoc rerun.
"""
from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


SCRIPT_VERSION = "1.0.0"
METHOD_ID = "age_stratified_fixed_holdout_v1"
RANDOM_SEED = 20260919
BIT_GENERATOR = "PCG64"
GROUP_ORDER = ("Young", "Old")
EXPECTED_GROUP_SIZE = 16
TEST_PER_GROUP = 3
EXPECTED_PARTICIPANTS = 32
EXPECTED_TRAIN = 26
EXPECTED_TEST = 6

PARTICIPANT_COLUMN = "participant_id"
GROUP_COLUMN = "participant_group"
REQUIRED_COLUMNS = (PARTICIPANT_COLUMN, GROUP_COLUMN)

ASSIGNMENT_FILENAME = "participant_split_assignments.csv"
MANIFEST_FILENAME = "participant_split_manifest.json"


class ParticipantSplitError(ValueError):
    """Raised when the participant table cannot support the frozen split."""


def canonical_participants(data: pd.DataFrame) -> pd.DataFrame:
    """Return one validated, sorted Young/Old row per participant.

    Repeated trial rows are allowed. They are collapsed only after confirming
    that every participant has one consistent recorded age group.
    """
    missing = [column for column in REQUIRED_COLUMNS if column not in data.columns]
    if missing:
        raise ParticipantSplitError(f"Missing required columns: {missing}")

    frame = data.loc[:, REQUIRED_COLUMNS].copy()
    if frame.isna().any(axis=None):
        raise ParticipantSplitError(
            "participant_id and participant_group must be complete for all rows"
        )

    frame[PARTICIPANT_COLUMN] = (
        frame[PARTICIPANT_COLUMN].astype(str).str.strip()
    )
    frame[GROUP_COLUMN] = frame[GROUP_COLUMN].astype(str).str.strip()

    if frame[PARTICIPANT_COLUMN].eq("").any():
        raise ParticipantSplitError("participant_id contains an empty value")

    unexpected_groups = sorted(set(frame[GROUP_COLUMN]) - set(GROUP_ORDER))
    if unexpected_groups:
        raise ParticipantSplitError(
            f"Unexpected participant_group values: {unexpected_groups}"
        )

    group_counts_per_participant = frame.groupby(PARTICIPANT_COLUMN)[
        GROUP_COLUMN
    ].nunique(dropna=False)
    conflicts = group_counts_per_participant[group_counts_per_participant != 1]
    if not conflicts.empty:
        raise ParticipantSplitError(
            "Each participant must have one consistent participant_group; "
            f"conflicts: {sorted(conflicts.index.astype(str))}"
        )

    participants = (
        frame.drop_duplicates(PARTICIPANT_COLUMN)
        .sort_values(PARTICIPANT_COLUMN, kind="mergesort")
        .reset_index(drop=True)
    )

    if len(participants) != EXPECTED_PARTICIPANTS:
        raise ParticipantSplitError(
            f"Expected {EXPECTED_PARTICIPANTS} unique participants, "
            f"found {len(participants)}"
        )

    counts = participants[GROUP_COLUMN].value_counts().to_dict()
    expected_counts = {group: EXPECTED_GROUP_SIZE for group in GROUP_ORDER}
    if counts != expected_counts:
        raise ParticipantSplitError(
            f"Expected participant-group counts {expected_counts}, found {counts}"
        )

    return participants


def participant_source_signature(participants: pd.DataFrame) -> str:
    """Hash the canonical participant IDs and groups used by the split."""
    canonical = participants.sort_values(
        PARTICIPANT_COLUMN, kind="mergesort"
    )
    payload = "\n".join(
        f"{row.participant_id}|{row.participant_group}"
        for row in canonical.itertuples(index=False)
    )
    return sha256(payload.encode("utf-8")).hexdigest()


def assignment_signature(assignments: pd.DataFrame) -> str:
    """Hash the complete participant assignment in canonical participant order."""
    canonical = assignments.sort_values(
        PARTICIPANT_COLUMN, kind="mergesort"
    )
    payload = "\n".join(
        (
            f"{row.participant_id}|{row.participant_group}|"
            f"{row.partition}|{row.draw_position_within_group}"
        )
        for row in canonical.itertuples(index=False)
    )
    return sha256(payload.encode("utf-8")).hexdigest()


def build_participant_split(data: pd.DataFrame) -> pd.DataFrame:
    """Build the frozen Young/Old-stratified participant assignment.

    Procedure:
    1. Canonicalize and sort participant IDs.
    2. Initialize NumPy PCG64 once with RANDOM_SEED.
    3. Shuffle the sorted Young IDs and mark the first three as test.
    4. Continue with the same generator state, shuffle the sorted Old IDs,
       and mark the first three as test.
    5. All remaining participants are training participants.
    """
    participants = canonical_participants(data)
    rng = np.random.Generator(np.random.PCG64(RANDOM_SEED))

    records: list[dict[str, Any]] = []
    for group in GROUP_ORDER:
        sorted_ids = (
            participants.loc[
                participants[GROUP_COLUMN].eq(group), PARTICIPANT_COLUMN
            ]
            .sort_values(kind="mergesort")
            .to_numpy(dtype=str)
        )
        shuffled_ids = rng.permutation(sorted_ids)
        for position, participant_id in enumerate(shuffled_ids, start=1):
            records.append(
                {
                    PARTICIPANT_COLUMN: str(participant_id),
                    GROUP_COLUMN: group,
                    "partition": (
                        "test" if position <= TEST_PER_GROUP else "train"
                    ),
                    "draw_position_within_group": int(position),
                }
            )

    assignments = (
        pd.DataFrame.from_records(records)
        .sort_values(PARTICIPANT_COLUMN, kind="mergesort")
        .reset_index(drop=True)
    )
    _validate_assignment(assignments)
    return assignments


def _validate_assignment(assignments: pd.DataFrame) -> None:
    """Check invariants that must hold for every official split."""
    if len(assignments) != EXPECTED_PARTICIPANTS:
        raise ParticipantSplitError("Assignment does not contain 32 participants")
    if assignments[PARTICIPANT_COLUMN].duplicated().any():
        raise ParticipantSplitError("A participant appears more than once")

    split_counts = assignments["partition"].value_counts().to_dict()
    if split_counts != {"train": EXPECTED_TRAIN, "test": EXPECTED_TEST}:
        raise ParticipantSplitError(
            f"Expected train/test counts 26/6, found {split_counts}"
        )

    group_split_counts = (
        assignments.groupby([GROUP_COLUMN, "partition"])
        .size()
        .to_dict()
    )
    expected = {
        ("Young", "train"): 13,
        ("Young", "test"): 3,
        ("Old", "train"): 13,
        ("Old", "test"): 3,
    }
    if group_split_counts != expected:
        raise ParticipantSplitError(
            f"Unexpected group-by-partition counts: {group_split_counts}"
        )


def build_manifest(
    data: pd.DataFrame,
    assignments: pd.DataFrame,
) -> dict[str, Any]:
    """Return deterministic audit metadata for the split."""
    participants = canonical_participants(data)
    return {
        "schema_version": 1,
        "script_version": SCRIPT_VERSION,
        "method_id": METHOD_ID,
        "random_seed": RANDOM_SEED,
        "numpy_bit_generator": BIT_GENERATOR,
        "numpy_version": np.__version__,
        "group_order": list(GROUP_ORDER),
        "expected_unique_participants": EXPECTED_PARTICIPANTS,
        "expected_group_size": EXPECTED_GROUP_SIZE,
        "test_participants_per_group": TEST_PER_GROUP,
        "training_participant_count": EXPECTED_TRAIN,
        "test_participant_count": EXPECTED_TEST,
        "input_row_count": int(len(data)),
        "input_columns_used_for_split": list(REQUIRED_COLUMNS),
        "participant_source_sha256": participant_source_signature(participants),
        "assignment_sha256": assignment_signature(assignments),
        "overwrite_policy": "refuse_if_output_exists",
        "tmt_b_used_for_split": False,
        "outcomes_used_for_split": False,
        "behavioural_features_used_for_split": False,
        "condition_used_for_split": False,
    }


def write_split_outputs(
    data: pd.DataFrame,
    output_dir: Path,
) -> tuple[Path, Path]:
    """Write the assignment and manifest without overwriting an existing split."""
    output_dir = Path(output_dir)
    assignment_path = output_dir / ASSIGNMENT_FILENAME
    manifest_path = output_dir / MANIFEST_FILENAME

    existing = [path for path in (assignment_path, manifest_path) if path.exists()]
    if existing:
        raise FileExistsError(
            "Refusing to overwrite an existing participant split: "
            + ", ".join(str(path) for path in existing)
        )

    assignments = build_participant_split(data)
    manifest = build_manifest(data, assignments)

    output_dir.mkdir(parents=True, exist_ok=True)
    assignment_tmp = assignment_path.with_suffix(
        assignment_path.suffix + ".tmp"
    )
    manifest_tmp = manifest_path.with_suffix(manifest_path.suffix + ".tmp")

    try:
        assignments.to_csv(assignment_tmp, index=False, lineterminator="\n")
        with manifest_tmp.open("w", encoding="utf-8", newline="\n") as handle:
            json.dump(manifest, handle, indent=2, sort_keys=True)
            handle.write("\n")
        assignment_tmp.replace(assignment_path)
        manifest_tmp.replace(manifest_path)
    finally:
        assignment_tmp.unlink(missing_ok=True)
        manifest_tmp.unlink(missing_ok=True)

    return assignment_path, manifest_path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create the frozen participant-level 26/6 train/test assignment."
        )
    )
    parser.add_argument(
        "input_csv",
        type=Path,
        help=(
            "CSV containing participant_id and participant_group. Repeated "
            "trial rows are allowed when group labels are consistent."
        ),
    )
    parser.add_argument(
        "output_dir",
        type=Path,
        help="Directory for the assignment CSV and audit manifest.",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    data = pd.read_csv(args.input_csv)
    assignment_path, manifest_path = write_split_outputs(
        data, args.output_dir
    )

    assignments = pd.read_csv(assignment_path)
    test_ids = assignments.loc[
        assignments["partition"].eq("test"), PARTICIPANT_COLUMN
    ].astype(str)

    print(f"Wrote: {assignment_path}")
    print(f"Wrote: {manifest_path}")
    print(f"Training participants: {EXPECTED_TRAIN}")
    print(f"Test participants: {EXPECTED_TEST}")
    print("Held-out participant IDs: " + ", ".join(test_ids.tolist()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
