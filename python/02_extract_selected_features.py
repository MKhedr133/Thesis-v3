#!/usr/bin/env python3
"""Prepare VR-supermarket trial inputs for selected-feature extraction.

FE-01.1 establishes the input boundary for the rebuilt Python extractor.  It
normalizes the master trial table, reads the difficulty-specific product lists,
resolves tracker and participant-detail paths, and loads recorded tracker
timestamps.  It deliberately calculates no behavioural features and writes no
feature files.

All paths are supplied explicitly at the command line.  Original participant
data remain outside this repository.  Missing supporting tracker fields are
represented as missing values and reported as warnings; recorded timestamps
are never replaced with frame-derived values.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from pathlib import Path
import re
from typing import NamedTuple

import numpy as np
import pandas as pd


MASTER_COLUMN_ALIASES = {
    "participant": "participant_id",
    "participant_id": "participant_id",
    "session": "session_id",
    "session_id": "session_id",
    "trial": "source_tracker_csv_filename",
    "source_tracker_csv_filename": "source_tracker_csv_filename",
    "group": "participant_group",
    "participant_group": "participant_group",
    "condition": "condition_name",
    "condition_name": "condition_name",
    "level3": "difficulty_level",
    "level": "difficulty_level",
    "difficulty": "difficulty_level",
    "difficulty_level": "difficulty_level",
    "trial2": "trial_order",
    "trial_order": "trial_order",
    "language": "language",
}

REQUIRED_MASTER_COLUMNS = (
    "participant_id",
    "session_id",
    "source_tracker_csv_filename",
    "participant_group",
    "condition_name",
    "difficulty_level",
    "trial_order",
)

# These raw signals are retained for later reviewed microsteps.  Their absence
# is recorded, not repaired or interpreted during FE-01.1.
SUPPORTING_TRACKER_COLUMNS = (
    "enableBlackout",
    "focus_object_name",
    "focus_object_tag",
    "num_items_in_cart",
    "is_grabbing_right",
    "grabbed_object_right",
    "is_grabbing_left",
    "grabbed_object_left",
    "right_pos_x",
    "right_pos_y",
    "right_pos_z",
    "left_pos_x",
    "left_pos_y",
    "left_pos_z",
    "hmd_rot_x",
    "hmd_rot_y",
    "hmd_rot_z",
    "hmd_rot_w",
)


class ProductCatalog:
    """Ordered shopping-list product names keyed by difficulty and language."""

    def __init__(self, ordered_lists: Mapping[tuple[int, str], tuple[str, ...]]):
        self.ordered_lists = dict(ordered_lists)

    def products_for(self, difficulty: int, language: str) -> tuple[str, ...]:
        """Return the ordered list for one difficulty and language."""
        return self.ordered_lists.get((int(difficulty), str(language).upper()), ())


class TrialPaths(NamedTuple):
    """Exact files belonging to one participant and session."""

    tracker: Path
    participant_details: Path


class TrackerLoadResult(NamedTuple):
    """Loaded tracker rows plus non-fatal input warnings."""

    table: pd.DataFrame
    warnings: list[str]


def read_table(path: Path) -> pd.DataFrame:
    """Read a comma, semicolon, or tab-delimited table from ``path``.

    The header selects semicolon, tab, or comma parsing.  Checking the header
    explicitly keeps a valid one-column CSV from being split by an unreliable
    delimiter guess.  The source file is read only and is never rewritten.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Input table not found: {path}")
    with path.open("r", encoding="utf-8-sig", errors="replace") as source:
        header = source.readline()
    if ";" in header:
        separator = ";"
    elif "\t" in header:
        separator = "\t"
    else:
        separator = ","
    return pd.read_csv(path, sep=separator)


def normalize_participant(value: object) -> str:
    """Normalize identifiers such as ``P1`` and ``1`` to ``P01``."""
    match = re.search(r"(\d+)", str(value))
    if match is None:
        raise ValueError(f"Participant identifier is not understood: {value!r}")
    return f"P{int(match.group(1)):02d}"


def normalize_condition(value: object) -> str:
    """Normalize the three recorded experimental condition names."""
    text = str(value).strip().casefold()
    conditions = {
        "visual": "Visual",
        "auditory": "Auditory",
        "cognitive": "Cognitive",
    }
    if text not in conditions:
        raise ValueError(f"Condition is not understood: {value!r}")
    return conditions[text]


def normalize_difficulty(value: object) -> int | float:
    """Return a numeric difficulty, using an integer for whole levels."""
    numeric = float(value)
    if not np.isfinite(numeric):
        raise ValueError(f"Difficulty is not finite: {value!r}")
    return int(numeric) if numeric.is_integer() else numeric


def normalize_trial_order(value: object) -> str:
    """Normalize chronological labels such as ``t3`` to ``T3``."""
    match = re.search(r"(\d+)", str(value))
    if match is None:
        raise ValueError(f"Trial order is not understood: {value!r}")
    return f"T{int(match.group(1))}"


def tracker_filename(value: object) -> str:
    """Return the recorded tracker filename for a master-table trial token."""
    text = str(value).strip()
    if text.casefold().endswith(".csv"):
        return Path(text).name
    match = re.search(r"T?(\d+)", text, flags=re.IGNORECASE)
    if match is None:
        raise ValueError(f"Tracker trial token is not understood: {value!r}")
    return f"data_collector_vr_sample_T{int(match.group(1)):03d}.csv"


def standardize_trial_table(data: pd.DataFrame) -> pd.DataFrame:
    """Normalize master-trial columns while retaining unrelated source fields.

    Required metadata are participant, session, tracker trial, participant
    group, condition, difficulty, and chronological trial order.  Language is
    retained when present and defaults to English only when the source omits
    the column.
    """
    rename = {
        column: MASTER_COLUMN_ALIASES[str(column).strip().casefold()]
        for column in data.columns
        if str(column).strip().casefold() in MASTER_COLUMN_ALIASES
    }
    output = data.rename(columns=rename).copy()
    missing = [column for column in REQUIRED_MASTER_COLUMNS if column not in output]
    if missing:
        raise ValueError(
            "Master trial table is missing required columns: " + ", ".join(missing)
        )

    output["participant_id"] = output["participant_id"].map(normalize_participant)
    output["session_id"] = (
        output["session_id"].astype(str).str.strip().str.upper()
    )
    output["source_tracker_csv_filename"] = output[
        "source_tracker_csv_filename"
    ].map(tracker_filename)
    output["participant_group"] = output["participant_group"].astype(str).str.strip()
    output["condition_name"] = output["condition_name"].map(normalize_condition)
    output["difficulty_level"] = output["difficulty_level"].map(
        normalize_difficulty
    )
    output["trial_order"] = output["trial_order"].map(normalize_trial_order)
    if "language" not in output:
        output["language"] = "EN"
    output["language"] = output["language"].astype(str).str.strip().str.upper()
    return output


def parse_correct_lists(path: Path) -> ProductCatalog:
    """Parse ordered English and Dutch lists from ``CorrectLists.txt``.

    Expected definitions use the historical MATLAB form
    ``correctLists.level2.EN = {'Product A', 'Product B'};``.  Product spelling
    and order are preserved for later event matching and auditing.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Correct-product list not found: {path}")
    text = path.read_text(encoding="utf-8", errors="replace")
    pattern = re.compile(
        r"correctLists\.level(\d+)\.(EN|NL)\s*=\s*\{(.*?)\};",
        flags=re.IGNORECASE | re.DOTALL,
    )
    ordered_lists: dict[tuple[int, str], tuple[str, ...]] = {}
    for match in pattern.finditer(text):
        difficulty = int(match.group(1))
        language = match.group(2).upper()
        products = tuple(re.findall(r"'([^']+)'", match.group(3)))
        ordered_lists[(difficulty, language)] = products

    required = {(difficulty, language) for difficulty in (0, 2, 6, 10) for language in ("EN", "NL")}
    missing = sorted(required - set(ordered_lists))
    if missing:
        formatted = ", ".join(f"D{difficulty}/{language}" for difficulty, language in missing)
        raise ValueError(f"Correct-product definitions are missing: {formatted}")
    return ProductCatalog(ordered_lists)


def build_trial_paths(row: Mapping[str, object], raw_root: Path) -> TrialPaths:
    """Build exact tracker and participant-detail paths for one trial.

    No recursive fallback is used because tracker filenames repeat across
    participants.  The historical outer folder uses ``P1`` while inner
    identifiers use ``P01``.
    """
    participant = normalize_participant(row["participant_id"])
    participant_number = int(participant[1:])
    session = str(row["session_id"]).strip().upper()
    filename = tracker_filename(row["source_tracker_csv_filename"])
    trial_root = (
        Path(raw_root)
        / f"P{participant_number}"
        / f"{participant}_HMD_Data"
        / participant
        / session
    )
    return TrialPaths(
        tracker=trial_root / "trackers" / filename,
        participant_details=trial_root / "session_info" / "participant_details.csv",
    )


def normalize_hand(value: object) -> str | None:
    """Return ``left`` or ``right`` for common recorded hand labels."""
    text = str(value).strip().casefold()
    if text in {"r", "right", "right-handed", "right handed"}:
        return "right"
    if text in {"l", "left", "left-handed", "left handed"}:
        return "left"
    return None


def read_participant_hand(path: Path) -> tuple[str | None, list[str]]:
    """Read optional reported hand metadata without inventing a default."""
    path = Path(path)
    if not path.is_file():
        return None, ["participant_details_file_missing"]
    try:
        details = read_table(path)
    except (OSError, pd.errors.ParserError) as error:
        return None, [f"participant_details_read_error:{type(error).__name__}"]

    by_lower = {str(column).strip().casefold(): column for column in details.columns}
    column = next(
        (
            by_lower[name]
            for name in ("side", "dominant_hand", "dominant hand", "handedness", "active_hand")
            if name in by_lower
        ),
        None,
    )
    if column is None or details[column].dropna().empty:
        return None, ["participant_hand_not_found"]
    hand = normalize_hand(details[column].dropna().iloc[0])
    if hand is None:
        return None, ["participant_hand_not_understood"]
    return hand, []


def load_tracker_table(path: Path) -> TrackerLoadResult:
    """Load tracker signals and preserve the source's actual timestamps.

    A missing or entirely unusable ``time`` column is fatal because later
    durations cannot be calculated honestly.  Other required signals are added
    as missing columns and named in warnings so later microsteps can retain the
    trial and explain which measurements are unavailable.
    """
    table = read_table(Path(path))
    if "time" not in table.columns:
        raise ValueError("Tracker table is missing required column: time")
    table = table.copy()
    table["time"] = pd.to_numeric(table["time"], errors="coerce")
    if table["time"].notna().sum() == 0:
        raise ValueError("Tracker table has no usable recorded time values")

    warnings: list[str] = []
    for column in SUPPORTING_TRACKER_COLUMNS:
        if column not in table:
            table[column] = np.nan
            warnings.append(f"missing_tracker_column:{column}")
    return TrackerLoadResult(table=table, warnings=warnings)


def build_argument_parser() -> argparse.ArgumentParser:
    """Create the read-only FE-01.1 command-line interface."""
    parser = argparse.ArgumentParser(
        description="Validate and normalize inputs for selected-feature extraction."
    )
    parser.add_argument("--performance", type=Path, required=True)
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--correct-lists", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=None)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Load requested trials and print a summary without writing outputs."""
    arguments = build_argument_parser().parse_args(argv)
    if arguments.limit is not None and arguments.limit < 1:
        raise ValueError("--limit must be at least 1")

    trials = standardize_trial_table(read_table(arguments.performance))
    if arguments.limit is not None:
        trials = trials.head(arguments.limit).copy()
    catalog = parse_correct_lists(arguments.correct_lists)

    tracker_count = 0
    hand_metadata_count = 0
    warning_count = 0
    for row in trials.to_dict(orient="records"):
        paths = build_trial_paths(row, arguments.raw_root)
        tracker_result = load_tracker_table(paths.tracker)
        tracker_count += 1
        warning_count += len(tracker_result.warnings)
        hand, hand_warnings = read_participant_hand(paths.participant_details)
        if hand is not None:
            hand_metadata_count += 1
        warning_count += len(hand_warnings)

    print(f"Loaded trial rows: {len(trials)}")
    print(f"Tracker files read: {tracker_count}")
    print(f"Correct-list definitions: {len(catalog.ordered_lists)}")
    print(f"Participant hand metadata found: {hand_metadata_count}")
    print(f"Input warnings: {warning_count}")
    print("FE-01.1 completed without writing feature outputs.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
