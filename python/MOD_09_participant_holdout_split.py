#!/usr/bin/env python3
"""MOD-09 — Reproducible participant-level 26/6 holdout split.

This module creates the single participant split used by the later modelling
pipeline. The split is stratified only by participant age group:

- 26 training participants: 13 Young and 13 Old;
- 6 final test participants: 3 Young and 3 Old.

The final test participants are reserved for the final difficulty-prediction
evaluation. Outcomes, TMT-B, condition, difficulty, and behavioural features
must not influence split selection.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd


SCRIPT_VERSION = "1.0.0"
METHOD_ID = "age_stratified_26_train_6_test_holdout_v1"

PARTICIPANT_COLUMN = "participant_id"
GROUP_COLUMN = "participant_group"
EXPECTED_PARTICIPANTS = 32
EXPECTED_GROUP_COUNTS = {"Young": 16, "Old": 16}
TEST_PER_GROUP = 3
TRAIN_PER_GROUP = 13
DEFAULT_SEED = 20260920

SPLIT_FILENAME = "participant_holdout_split.csv"
MANIFEST_FILENAME = "mod09_manifest.json"


class Mod09Error(ValueError):
    """Raised when the MOD-09 holdout-split contract is violated."""


def _normalise_required_strings(
    data: pd.DataFrame,
) -> pd.DataFrame:
    required = [PARTICIPANT_COLUMN, GROUP_COLUMN]
    missing = [column for column in required if column not in data.columns]
    if missing:
        raise Mod09Error(f"Missing required column(s): {missing}")

    out = data[required].copy()
    if out[required].isna().any().any():
        raise Mod09Error(
            "participant_id and participant_group must not contain missing values"
        )

    for column in required:
        out[column] = out[column].astype(str).str.strip()
        if out[column].eq("").any():
            raise Mod09Error(
                f"{column} contains blank values"
            )

    return out


def build_participant_table(
    modeling_data: pd.DataFrame,
) -> pd.DataFrame:
    """Return one validated row per participant using only ID and age group."""
    values = _normalise_required_strings(modeling_data)

    group_nunique = values.groupby(
        PARTICIPANT_COLUMN,
        sort=False,
    )[GROUP_COLUMN].nunique()
    inconsistent = group_nunique.loc[group_nunique.ne(1)].index.tolist()
    if inconsistent:
        raise Mod09Error(
            "Found inconsistent participant_group values within participant: "
            + ", ".join(sorted(map(str, inconsistent)))
        )

    participants = (
        values.drop_duplicates(subset=[PARTICIPANT_COLUMN, GROUP_COLUMN])
        .sort_values(PARTICIPANT_COLUMN, kind="mergesort")
        .reset_index(drop=True)
    )

    if len(participants) != EXPECTED_PARTICIPANTS:
        raise Mod09Error(
            "MOD-09 requires exactly 32 participants; "
            f"found {len(participants)}"
        )

    observed_groups = set(participants[GROUP_COLUMN])
    expected_groups = set(EXPECTED_GROUP_COUNTS)
    if observed_groups != expected_groups:
        raise Mod09Error(
            "MOD-09 requires participant_group values Young and Old only; "
            f"found {sorted(observed_groups)}"
        )

    counts = participants[GROUP_COLUMN].value_counts().to_dict()
    if counts != EXPECTED_GROUP_COUNTS:
        raise Mod09Error(
            "MOD-09 requires exactly 16 Young and 16 Old participants; "
            f"found Young={counts.get('Young', 0)}, Old={counts.get('Old', 0)}"
        )

    return participants


def create_holdout_split(
    modeling_data: pd.DataFrame,
    *,
    seed: int = DEFAULT_SEED,
) -> pd.DataFrame:
    """Create the deterministic age-stratified 26/6 participant split."""
    if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)):
        raise Mod09Error("seed must be a non-negative integer")
    seed = int(seed)
    if seed < 0:
        raise Mod09Error("seed must be a non-negative integer")

    participants = build_participant_table(modeling_data)
    rng = np.random.default_rng(seed)

    test_ids: set[str] = set()
    for group in ("Young", "Old"):
        ids = (
            participants.loc[
                participants[GROUP_COLUMN].eq(group),
                PARTICIPANT_COLUMN,
            ]
            .sort_values(kind="mergesort")
            .to_numpy(dtype=str)
        )
        chosen = rng.choice(
            ids,
            size=TEST_PER_GROUP,
            replace=False,
        )
        test_ids.update(str(value) for value in chosen)

    split = participants.copy()
    split["split"] = np.where(
        split[PARTICIPANT_COLUMN].isin(test_ids),
        "test",
        "train",
    )
    split = split.sort_values(
        PARTICIPANT_COLUMN,
        kind="mergesort",
    ).reset_index(drop=True)

    _validate_created_split(split)
    return split


def _validate_created_split(split: pd.DataFrame) -> None:
    if len(split) != EXPECTED_PARTICIPANTS:
        raise Mod09Error("Created split does not contain exactly 32 participants")
    if split[PARTICIPANT_COLUMN].nunique() != EXPECTED_PARTICIPANTS:
        raise Mod09Error("Created split contains duplicate participant IDs")

    split_counts = split["split"].value_counts().to_dict()
    if split_counts.get("train", 0) != 26 or split_counts.get("test", 0) != 6:
        raise Mod09Error(
            "Created split must contain exactly 26 train and 6 test participants"
        )

    table = (
        split.groupby(["split", GROUP_COLUMN])
        .size()
        .to_dict()
    )
    expected = {
        ("train", "Young"): TRAIN_PER_GROUP,
        ("train", "Old"): TRAIN_PER_GROUP,
        ("test", "Young"): TEST_PER_GROUP,
        ("test", "Old"): TEST_PER_GROUP,
    }
    if table != expected:
        raise Mod09Error(
            "Created split does not preserve the required age-group balance"
        )


def _group_counts(
    split: pd.DataFrame,
    label: str,
) -> dict[str, int]:
    counts = (
        split.loc[split["split"].eq(label), GROUP_COLUMN]
        .value_counts()
        .to_dict()
    )
    return {
        "Old": int(counts.get("Old", 0)),
        "Young": int(counts.get("Young", 0)),
    }


def build_manifest(
    *,
    split: pd.DataFrame,
    seed: int,
    modeling_data_path: Path,
) -> dict[str, Any]:
    """Build an auditable description of the split method and data boundary."""
    _validate_created_split(split)

    return {
        "script_version": SCRIPT_VERSION,
        "method_id": METHOD_ID,
        "analysis_role": "participant_holdout_definition",
        "participant_count": int(len(split)),
        "training_participant_count": int(split["split"].eq("train").sum()),
        "test_participant_count": int(split["split"].eq("test").sum()),
        "training_group_counts": _group_counts(split, "train"),
        "test_group_counts": _group_counts(split, "test"),
        "stratification_variable": GROUP_COLUMN,
        "selection_variables": [
            PARTICIPANT_COLUMN,
            GROUP_COLUMN,
            "seed",
        ],
        "outcomes_used_for_split_selection": False,
        "tmt_b_used_for_split_selection": False,
        "condition_used_for_split_selection": False,
        "difficulty_used_for_split_selection": False,
        "test_set_locked_for_final_evaluation": True,
        "test_set_allowed_during_model_development": False,
        "random_seed": int(seed),
        "sampling_method": (
            "within each age group, sample 3 participant IDs without replacement"
        ),
        "modeling_data_path": str(modeling_data_path),
        "warning": (
            "Do not change the seed or regenerate the split after inspecting "
            "model-development or test-set results."
        ),
    }


def _refuse_existing_outputs(output_dir: Path) -> None:
    existing = [
        output_dir / filename
        for filename in (SPLIT_FILENAME, MANIFEST_FILENAME)
        if (output_dir / filename).exists()
    ]
    if existing:
        raise FileExistsError(
            "Refusing to overwrite existing MOD-09 output(s): "
            + ", ".join(path.name for path in existing)
        )


def write_outputs(
    *,
    split: pd.DataFrame,
    manifest: Mapping[str, Any],
    output_dir: Path,
) -> dict[str, Path]:
    """Write the local split and manifest without overwriting prior results."""
    _validate_created_split(split)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    _refuse_existing_outputs(output_dir)

    paths = {
        "split": output_dir / SPLIT_FILENAME,
        "manifest": output_dir / MANIFEST_FILENAME,
    }
    split.to_csv(paths["split"], index=False, encoding="utf-8")
    paths["manifest"].write_text(
        json.dumps(dict(manifest), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return paths


def build_cli_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--modeling-data", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help=(
            "Fixed non-negative random seed. Default: %(default)s. "
            "Do not seed-shop after inspecting results."
        ),
    )
    return parser


def main() -> int:
    args = build_cli_parser().parse_args()

    modeling_data = pd.read_csv(args.modeling_data)
    split = create_holdout_split(
        modeling_data,
        seed=args.seed,
    )
    manifest = build_manifest(
        split=split,
        seed=args.seed,
        modeling_data_path=args.modeling_data,
    )
    paths = write_outputs(
        split=split,
        manifest=manifest,
        output_dir=args.output_dir,
    )

    print(
        "[MOD-09] participant split created: "
        "train=26 (13 Young, 13 Old); "
        "test=6 (3 Young, 3 Old)"
    )
    print(f"[MOD-09] seed={args.seed}")
    print(f"[MOD-09] split: {paths['split']}")
    print(f"[MOD-09] manifest: {paths['manifest']}")
    print(
        "[MOD-09] final test participants are locked and must not be used "
        "during model development."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
