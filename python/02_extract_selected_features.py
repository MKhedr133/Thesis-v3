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
    "is_left_eye_blinking",
    "is_right_eye_blinking",
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

PLACEHOLDER_OBJECT_NAMES = {
    "",
    "nan",
    "none",
    "noobjectgrabbed",
    "notassigned",
    "untagged",
    "null",
}

NON_PRODUCT_OBJECT_NAMES = {
    "tablet",
    "samsung_tab",
    "cart",
    "shopping_basket",
    "basket",
    "mainshelf",
    "main shelf",
    "npc",
    "bodum",
    "side l",
    "side r",
    "cube",
    "first shelf",
    "second shelf",
    "third shelf",
    "fourth shelf",
}


class ProductCatalog:
    """Ordered shopping-list product names keyed by difficulty and language."""

    def __init__(self, ordered_lists: Mapping[tuple[int, str], tuple[str, ...]]):
        self.ordered_lists = dict(ordered_lists)
        self.alias_to_canonical: dict[str, str] = {}
        canonical_products: set[str] = set()

        difficulties = sorted({difficulty for difficulty, _ in self.ordered_lists})
        for difficulty in difficulties:
            english = self.ordered_lists.get((difficulty, "EN"), ())
            dutch = self.ordered_lists.get((difficulty, "NL"), ())
            for index, english_name in enumerate(english):
                canonical = clean_product_name(english_name)
                if not canonical:
                    continue
                canonical_products.add(canonical)
                self.alias_to_canonical[canonical] = canonical
                if index < len(dutch):
                    self.alias_to_canonical[clean_product_name(dutch[index])] = canonical

        self.all_canonical_products = frozenset(canonical_products)

    def products_for(self, difficulty: int, language: str) -> tuple[str, ...]:
        """Return the ordered list for one difficulty and language."""
        return self.ordered_lists.get((int(difficulty), str(language).upper()), ())

    def canonicalize(self, value: object) -> str:
        """Map English or Dutch product labels to one cleaned English name."""
        cleaned = clean_product_name(value)
        return self.alias_to_canonical.get(cleaned, cleaned)

    def correct_canonical_products(
        self, difficulty: int, language: str
    ) -> frozenset[str]:
        """Return canonical products assigned to one trial list."""
        return frozenset(
            self.canonicalize(product)
            for product in self.products_for(difficulty, language)
        )


class TrialPaths(NamedTuple):
    """Exact files belonging to one participant and session."""

    tracker: Path
    participant_details: Path


class TrackerLoadResult(NamedTuple):
    """Loaded tracker rows plus non-fatal input warnings."""

    table: pd.DataFrame
    warnings: list[str]


class GrabDetectionConfig(NamedTuple):
    """Provisional historical product-grab settings used before validation."""

    minimum_duration_seconds: float = 0.20
    merge_gap_seconds: float = 0.10


class ProductGrabEvent(NamedTuple):
    """One complete real-product grab retained for later auditing."""

    canonical_product_name: str
    raw_product_label: str
    hand: str
    grab_start_seconds: float
    grab_release_seconds: float
    grab_duration_seconds: float
    is_on_list: bool
    is_first_time_on_list: bool
    overlaps_other_hand_grab: bool


def actual_grab_hand(event: ProductGrabEvent) -> str:
    """Return the recorded hand that completed a product grab."""
    if event.hand not in {"left", "right"}:
        raise ValueError(f"Unsupported grab hand: {event.hand!r}")
    return event.hand


class GrabDetectionResult(NamedTuple):
    """Chronological product-grab events and unique processing warnings."""

    events: tuple[ProductGrabEvent, ...]
    warnings: tuple[str, ...]


class FocusEpisode(NamedTuple):
    """One complete continuous engine-labelled focus episode."""

    canonical_focus_name: str
    raw_focus_name: str
    cleaned_focus_tag: str
    raw_focus_tag: str
    focus_start_seconds: float
    focus_end_seconds: float
    focus_duration_seconds: float


class FocusDetectionResult(NamedTuple):
    """Chronological focus episodes and unique processing warnings."""

    episodes: tuple[FocusEpisode, ...]
    warnings: tuple[str, ...]


class ListVisit(NamedTuple):
    """One complete engine-labelled shopping-list viewing episode."""

    list_visit_start_seconds: float
    list_visit_end_seconds: float
    list_visit_duration_seconds: float
    is_initial_view: bool


class SearchInterval(NamedTuple):
    """Search timing for one qualifying first-time-on-list product grab."""

    canonical_product_name: str
    qualifying_grab_start_seconds: float
    search_start_seconds: float | None
    first_target_focus_seconds: float | None
    is_valid: bool


class SearchIntervalResult(NamedTuple):
    """Auditable qualifying-grab search records and unique warnings."""

    intervals: tuple[SearchInterval, ...]
    warnings: tuple[str, ...]


class PaceAndListRecheckAggregation(NamedTuple):
    """Trial-level pace and list-recheck measurements from established events."""

    median_time_between_qualifying_grabs_seconds: float | None
    list_recheck_count: int
    total_list_recheck_duration_seconds: float


class LocatingFeatureAggregation(NamedTuple):
    """Trial-level locating, irrelevant-focus and head-turning measures."""

    median_time_to_target_seconds: float | None
    median_irrelevant_focus_duration_seconds: float | None
    median_head_turning_degrees: float | None
    warnings: tuple[str, ...]


class ReachFeatureAggregation(NamedTuple):
    """Trial-level reach duration and 3-D path-ratio measurements."""

    median_reach_duration_seconds: float | None
    median_reach_path_ratio: float | None
    warnings: tuple[str, ...]


class ErrorOutcomeRecord(NamedTuple):
    """One normalized performance row with raw errors and D0 change."""

    participant_id: str
    session_id: str
    condition_name: str
    difficulty_level: int | float
    trial_order: str
    errors_missing: float | None
    errors_wrong_order: float | None
    errors_duplicate: float | None
    errors_not_in_list: float | None
    total_error_count: float | None
    error_change_from_d0: float | None


class ErrorOutcomeResult(NamedTuple):
    """Normalized error-outcome records and unique processing warnings."""

    records: tuple[ErrorOutcomeRecord, ...]
    warnings: tuple[str, ...]


class ProductGrabOutputResult(NamedTuple):
    """One trial's auditable product-grab output table and warnings."""

    table: pd.DataFrame
    warnings: tuple[str, ...]


class TrialFeatureOutputResult(NamedTuple):
    """One trial's aggregated feature table and unique processing warnings."""

    table: pd.DataFrame
    warnings: tuple[str, ...]


PRODUCT_GRAB_OUTPUT_COLUMNS = (
    "participant_id",
    "session_id",
    "source_tracker_csv_filename",
    "participant_group",
    "condition_name",
    "difficulty_level",
    "trial_order",
    "language",
    "canonical_product_name",
    "raw_product_label",
    "hand",
    "grab_start_seconds",
    "grab_release_seconds",
    "grab_duration_seconds",
    "is_on_list",
    "is_first_time_on_list",
    "is_repeated_on_list",
    "is_off_list",
    "overlaps_other_hand_grab",
    "search_start_seconds",
    "first_target_focus_seconds",
    "search_interval_valid",
    "reach_start_seconds",
    "reach_duration_seconds",
    "reach_path_ratio",
    "reach_interval_valid",
    "processing_warnings",
)


TRIAL_FEATURE_COLUMNS = (
    "participant_id",
    "session_id",
    "source_tracker_csv_filename",
    "participant_group",
    "condition_name",
    "difficulty_level",
    "trial_order",
    "language",
    "performance",
    "mental_demand_score_0_to_10",
    "errors_missing",
    "errors_wrong_order",
    "errors_duplicate",
    "errors_not_in_list",
    "total_error_count",
    "error_change_from_d0",
    "median_time_between_qualifying_grabs_seconds",
    "list_recheck_count",
    "total_list_recheck_duration_seconds",
    "median_time_to_target_seconds",
    "median_irrelevant_focus_duration_seconds",
    "median_head_turning_degrees",
    "median_reach_duration_seconds",
    "median_reach_path_ratio",
    "real_product_grab_count",
    "first_time_on_list_grab_count",
    "list_visit_count",
    "valid_search_interval_count",
    "valid_reach_duration_count",
    "valid_reach_path_ratio_count",
    "processing_warnings",
)


class HandPositionSample(NamedTuple):
    """One recorded hand position and its preceding motion step."""

    timestamp_seconds: float | None
    x_meters: float | None
    y_meters: float | None
    z_meters: float | None
    step_distance_meters: float | None
    step_speed_meters_per_second: float | None
    is_valid: bool


class HandMotionSeries(NamedTuple):
    """Recorded hand positions with timestamp-based motion steps."""

    hand: str
    samples: tuple[HandPositionSample, ...]
    warnings: tuple[str, ...]


class ReachDetectionConfig(NamedTuple):
    """Provisional historical settings for reach-onset detection."""

    hand_speed_threshold_meters_per_second: float = 0.05
    minimum_rest_duration_seconds: float = 0.20


class ReachInterval(NamedTuple):
    """Auditable reach interval ending at one qualifying grab."""

    canonical_product_name: str
    hand: str
    reach_start_seconds: float | None
    grab_start_seconds: float
    is_valid: bool


class ReachDetectionResult(NamedTuple):
    """Reach intervals and unique detector warnings."""

    intervals: tuple[ReachInterval, ...]
    warnings: tuple[str, ...]


class _RawGrabSegment(NamedTuple):
    """Internal complete segment before duration and list classification."""

    canonical_product_name: str
    raw_product_label: str
    hand: str
    start: float
    release: float


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


def clean_product_name(value: object) -> str:
    """Clean known Unity instance suffixes without changing product meaning."""
    if pd.isna(value):
        return ""
    text = str(value).strip().casefold()
    text = re.sub(r"\s*\(clone\)\s*$", "", text, flags=re.IGNORECASE)
    previous = None
    while text != previous:
        previous = text
        text = re.sub(r"\s*\(\s*\d+\s*\)(?:\s+\d+)*\s*$", "", text)
        text = re.sub(r"\s+\d+\s*$", "", text)
    return re.sub(r"\s+", " ", text).strip()


def is_probable_product_object(value: object, catalog: ProductCatalog) -> bool:
    """Distinguish plausible products from known placeholders and task AOIs."""
    cleaned = clean_product_name(value)
    if cleaned in PLACEHOLDER_OBJECT_NAMES:
        return False
    canonical = catalog.canonicalize(cleaned)
    if canonical in catalog.all_canonical_products:
        return True
    if cleaned in NON_PRODUCT_OBJECT_NAMES:
        return False
    if re.search(r"(?:^|\s)(?:shelf|cube|npc)(?:\s|$)", cleaned):
        return False
    if "business man" in cleaned or cleaned.startswith("swpd"):
        return False
    return True


def parse_boolean_series(values: pd.Series) -> pd.Series:
    """Read boolean tracker values without treating missing values as true."""
    if pd.api.types.is_bool_dtype(values):
        return values.fillna(False).astype(bool)
    return values.astype(str).str.strip().str.casefold().isin(
        {"true", "1", "yes", "y"}
    )


def add_unique_warning(warnings: list[str], warning: str) -> None:
    """Append a warning once while retaining discovery order."""
    if warning not in warnings:
        warnings.append(warning)


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


def derive_error_outcomes(performance: pd.DataFrame) -> ErrorOutcomeResult:
    """Normalize raw performance errors and derive same-condition D0 changes."""
    error_aliases = {
        "errors_total": "errors_total",
        "total_error_count": "errors_total",
        "errors_missing": "errors_missing",
        "errors_wrongorder": "errors_wrong_order",
        "errors_wrong_order": "errors_wrong_order",
        "errors_collectedmorethanonce": "errors_duplicate",
        "errors_duplicate": "errors_duplicate",
        "errors_notinlist": "errors_not_in_list",
        "errors_not_in_list": "errors_not_in_list",
    }
    rename = {
        column: error_aliases[str(column).strip().casefold()]
        for column in performance.columns
        if str(column).strip().casefold() in error_aliases
    }
    normalized = standardize_trial_table(performance.rename(columns=rename))
    warnings: list[str] = []
    error_columns = (
        "errors_missing",
        "errors_wrong_order",
        "errors_duplicate",
        "errors_not_in_list",
    )
    for column in (*error_columns, "errors_total"):
        if column not in normalized.columns:
            normalized[column] = np.nan
        normalized[column] = pd.to_numeric(normalized[column], errors="coerce")

    prepared: list[dict[str, object]] = []
    for row_number, row in normalized.iterrows():
        components = [row[column] for column in error_columns]
        component_sum = (
            float(sum(float(value) for value in components))
            if all(pd.notna(value) and np.isfinite(float(value)) for value in components)
            else None
        )
        recorded_total = row["errors_total"]
        total = float(recorded_total) if pd.notna(recorded_total) and np.isfinite(float(recorded_total)) else None
        if total is not None and component_sum is not None and not np.isclose(total, component_sum):
            add_unique_warning(warnings, "error_total_component_mismatch")
        if total is None and component_sum is not None:
            total = component_sum
        if total is None:
            add_unique_warning(warnings, "error_total_unavailable")
        prepared.append(
            {
                "participant_id": str(row["participant_id"]),
                "session_id": str(row["session_id"]),
                "condition_name": str(row["condition_name"]),
                "difficulty_level": row["difficulty_level"],
                "trial_order": str(row["trial_order"]),
                "errors_missing": (
                    float(row["errors_missing"]) if pd.notna(row["errors_missing"]) else None
                ),
                "errors_wrong_order": (
                    float(row["errors_wrong_order"]) if pd.notna(row["errors_wrong_order"]) else None
                ),
                "errors_duplicate": (
                    float(row["errors_duplicate"]) if pd.notna(row["errors_duplicate"]) else None
                ),
                "errors_not_in_list": (
                    float(row["errors_not_in_list"]) if pd.notna(row["errors_not_in_list"]) else None
                ),
                "total_error_count": total,
                "error_change_from_d0": None,
            }
        )

    baseline_values: dict[tuple[str, str], list[float]] = {}
    for record in prepared:
        if record["difficulty_level"] == 0 and record["total_error_count"] is not None:
            key = (str(record["participant_id"]), str(record["condition_name"]))
            baseline_values.setdefault(key, []).append(float(record["total_error_count"]))
    baselines: dict[tuple[str, str], float] = {}
    for key, values in baseline_values.items():
        if len(values) > 1:
            add_unique_warning(warnings, f"duplicate_d0_baseline:{key[0]}:{key[1]}")
        baselines[key] = float(np.median(values))

    missing_baselines: set[tuple[str, str]] = set()
    for record in prepared:
        difficulty = record["difficulty_level"]
        if difficulty == 0:
            continue
        key = (str(record["participant_id"]), str(record["condition_name"]))
        if difficulty not in (2, 6, 10):
            add_unique_warning(warnings, f"unsupported_difficulty:{difficulty}")
            continue
        baseline = baselines.get(key)
        total = record["total_error_count"]
        if baseline is None or total is None:
            if baseline is None and key not in missing_baselines:
                add_unique_warning(warnings, f"d0_baseline_unavailable:{key[0]}:{key[1]}")
                missing_baselines.add(key)
            continue
        record["error_change_from_d0"] = float(total) - baseline

    records = tuple(ErrorOutcomeRecord(**record) for record in prepared)
    return ErrorOutcomeResult(records=records, warnings=tuple(warnings))


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


def prepare_hand_motion(tracker: pd.DataFrame, hand: str) -> HandMotionSeries:
    """Prepare recorded 3-D hand positions, distances and speeds."""
    if hand not in {"left", "right"}:
        raise ValueError(f"Unsupported hand: {hand!r}")
    columns = ("time", f"{hand}_pos_x", f"{hand}_pos_y", f"{hand}_pos_z")
    warnings: list[str] = []
    if any(column not in tracker for column in columns):
        add_unique_warning(warnings, f"hand_position_columns_unavailable:{hand}")
        return HandMotionSeries(hand=hand, samples=(), warnings=tuple(warnings))

    numeric = tracker.loc[:, columns].copy()
    for column in columns:
        numeric[column] = pd.to_numeric(numeric[column], errors="coerce")

    samples: list[HandPositionSample] = []
    previous_position: np.ndarray | None = None
    previous_time: float | None = None
    for row in numeric.itertuples(index=False):
        timestamp = float(row[0]) if np.isfinite(row[0]) else None
        position_values = np.asarray(row[1:4], dtype=float)
        position_valid = bool(np.isfinite(position_values).all())
        is_valid = timestamp is not None and position_valid
        distance: float | None = None
        speed: float | None = None
        if is_valid:
            current_position = position_values
            if previous_position is not None and previous_time is not None:
                distance = float(np.linalg.norm(current_position - previous_position))
                delta_time = timestamp - previous_time
                if np.isfinite(delta_time) and delta_time > 0:
                    speed = distance / float(delta_time)
                else:
                    add_unique_warning(warnings, f"nonpositive_time_delta:{hand}")
            previous_position = current_position
            previous_time = timestamp
        else:
            previous_position = None
            previous_time = None
        samples.append(
            HandPositionSample(
                timestamp_seconds=timestamp,
                x_meters=float(position_values[0]) if np.isfinite(position_values[0]) else None,
                y_meters=float(position_values[1]) if np.isfinite(position_values[1]) else None,
                z_meters=float(position_values[2]) if np.isfinite(position_values[2]) else None,
                step_distance_meters=distance,
                step_speed_meters_per_second=speed,
                is_valid=is_valid,
            )
        )
    return HandMotionSeries(hand=hand, samples=tuple(samples), warnings=tuple(warnings))


def detect_reach_intervals(
    tracker: pd.DataFrame,
    grab_events: Sequence[ProductGrabEvent],
    config: ReachDetectionConfig | None = None,
) -> ReachDetectionResult:
    """Detect the final sustained low-speed rest before each qualifying grab."""
    config = config or ReachDetectionConfig()
    if config.hand_speed_threshold_meters_per_second < 0:
        raise ValueError("hand_speed_threshold_meters_per_second cannot be negative")
    if config.minimum_rest_duration_seconds < 0:
        raise ValueError("minimum_rest_duration_seconds cannot be negative")

    warnings: list[str] = []
    intervals: list[ReachInterval] = []
    qualifying_grabs = sorted(
        (
            event
            for event in grab_events
            if event.is_on_list and event.is_first_time_on_list
        ),
        key=lambda event: event.grab_start_seconds,
    )
    motion_by_hand: dict[str, HandMotionSeries] = {}
    for event in qualifying_grabs:
        hand = actual_grab_hand(event)
        if hand not in motion_by_hand:
            motion_by_hand[hand] = prepare_hand_motion(tracker, hand)
            for warning in motion_by_hand[hand].warnings:
                add_unique_warning(warnings, warning)
        motion = motion_by_hand[hand]
        pre_grab = [
            sample
            for sample in motion.samples
            if sample.timestamp_seconds is not None
            and sample.timestamp_seconds <= event.grab_start_seconds
        ]
        rest_blocks: list[tuple[list[HandPositionSample], float]] = []
        active_block: list[HandPositionSample] = []
        active_start: float | None = None
        previous_valid_timestamp: float | None = None
        for sample in pre_grab:
            if not sample.is_valid or sample.timestamp_seconds is None:
                if active_block and active_start is not None:
                    rest_blocks.append((active_block, active_start))
                active_block = []
                active_start = None
                previous_valid_timestamp = None
                continue
            is_rest = (
                sample.step_speed_meters_per_second is not None
                and sample.step_speed_meters_per_second
                <= config.hand_speed_threshold_meters_per_second
            )
            if (
                sample.step_speed_meters_per_second is None
                and not active_block
            ):
                is_rest = True
            if is_rest:
                if not active_block:
                    active_start = (
                        previous_valid_timestamp
                        if previous_valid_timestamp is not None
                        else sample.timestamp_seconds
                    )
                active_block.append(sample)
            elif active_block:
                rest_blocks.append((active_block, active_start))
                active_block = []
                active_start = None
            previous_valid_timestamp = sample.timestamp_seconds
        if active_block and active_start is not None:
            rest_blocks.append((active_block, active_start))
        qualifying_blocks = []
        for block, rest_start in rest_blocks:
            rest_duration = block[-1].timestamp_seconds - rest_start
            if rest_duration >= config.minimum_rest_duration_seconds:
                qualifying_blocks.append(block)
        if not qualifying_blocks:
            add_unique_warning(warnings, f"reach_onset_not_found:{hand}")
            intervals.append(
                ReachInterval(
                    canonical_product_name=event.canonical_product_name,
                    hand=hand,
                    reach_start_seconds=None,
                    grab_start_seconds=event.grab_start_seconds,
                    is_valid=False,
                )
            )
            continue
        reach_start = float(qualifying_blocks[-1][-1].timestamp_seconds)
        intervals.append(
            ReachInterval(
                canonical_product_name=event.canonical_product_name,
                hand=hand,
                reach_start_seconds=reach_start,
                grab_start_seconds=event.grab_start_seconds,
                is_valid=True,
            )
        )
    return ReachDetectionResult(intervals=tuple(intervals), warnings=tuple(warnings))


def _complete_grab_segments_for_hand(
    tracker: pd.DataFrame,
    hand: str,
    catalog: ProductCatalog,
    warnings: list[str],
) -> list[_RawGrabSegment]:
    """Return complete same-hand, same-product segments before gap merging."""
    grab_column = f"is_grabbing_{hand}"
    object_column = f"grabbed_object_{hand}"
    if grab_column not in tracker or tracker[grab_column].isna().all():
        add_unique_warning(warnings, f"grab_signal_unavailable:{hand}")
        return []
    if object_column not in tracker:
        add_unique_warning(warnings, f"grab_object_unavailable:{hand}")
        return []

    grabbing = parse_boolean_series(tracker[grab_column]).to_numpy(bool)
    raw_labels = tracker[object_column]
    times = pd.to_numeric(tracker["time"], errors="coerce").to_numpy(float)

    segments: list[_RawGrabSegment] = []
    active_product: str | None = None
    active_raw_label = ""
    active_start = np.nan

    def discard_active_segment() -> None:
        nonlocal active_product, active_raw_label, active_start
        active_product = None
        active_raw_label = ""
        active_start = np.nan

    def close_active_segment(release: float) -> None:
        nonlocal active_product, active_raw_label, active_start
        if active_product is None:
            return
        if release < active_start:
            add_unique_warning(warnings, f"non_monotonic_grab_time:{hand}")
        else:
            segments.append(
                _RawGrabSegment(
                    canonical_product_name=active_product,
                    raw_product_label=active_raw_label,
                    hand=hand,
                    start=float(active_start),
                    release=float(release),
                )
            )
        discard_active_segment()

    for index in range(len(tracker)):
        timestamp = times[index]
        if not np.isfinite(timestamp):
            if active_product is not None:
                add_unique_warning(warnings, f"missing_time_during_grab:{hand}")
                discard_active_segment()
            continue

        raw_value = raw_labels.iloc[index]
        probable_product = grabbing[index] and is_probable_product_object(
            raw_value, catalog
        )
        product = catalog.canonicalize(raw_value) if probable_product else None

        if active_product is None:
            if product is not None:
                active_product = product
                active_raw_label = str(raw_value).strip()
                active_start = float(timestamp)
            continue

        if product == active_product:
            continue

        close_active_segment(float(timestamp))
        if product is not None:
            active_product = product
            active_raw_label = str(raw_value).strip()
            active_start = float(timestamp)

    if active_product is not None:
        add_unique_warning(warnings, f"open_ended_grab_discarded:{hand}")
    return segments


def _merge_grab_segments(
    segments: Sequence[_RawGrabSegment],
    config: GrabDetectionConfig,
    warnings: list[str],
) -> list[_RawGrabSegment]:
    """Merge brief same-hand interruptions of the same canonical product."""
    if not segments:
        return []
    merged = [segments[0]]
    for segment in segments[1:]:
        previous = merged[-1]
        gap = segment.start - previous.release
        if gap < 0:
            add_unique_warning(warnings, f"non_monotonic_grab_time:{segment.hand}")
        if (
            segment.hand == previous.hand
            and segment.canonical_product_name == previous.canonical_product_name
            and -1e-12 <= gap <= config.merge_gap_seconds + 1e-12
        ):
            merged[-1] = _RawGrabSegment(
                canonical_product_name=previous.canonical_product_name,
                raw_product_label=previous.raw_product_label,
                hand=previous.hand,
                start=previous.start,
                release=segment.release,
            )
        else:
            merged.append(segment)
    return merged


def detect_product_grab_events(
    tracker: pd.DataFrame,
    catalog: ProductCatalog,
    difficulty: int,
    language: str,
    config: GrabDetectionConfig | None = None,
) -> GrabDetectionResult:
    """Detect complete real-product grabs from both recorded hands.

    The default 0.20-second minimum duration and 0.10-second merge gap are
    provisional historical settings.  This function does not validate or tune
    them.  Open-ended segments are discarded because no release was observed.
    """
    if "time" not in tracker:
        raise ValueError("Tracker table is missing required column: time")
    config = config or GrabDetectionConfig()
    if config.minimum_duration_seconds <= 0:
        raise ValueError("minimum_duration_seconds must be greater than zero")
    if config.merge_gap_seconds < 0:
        raise ValueError("merge_gap_seconds cannot be negative")

    warnings: list[str] = []
    retained_segments: list[_RawGrabSegment] = []
    for hand in ("left", "right"):
        raw_segments = _complete_grab_segments_for_hand(
            tracker, hand, catalog, warnings
        )
        merged_segments = _merge_grab_segments(raw_segments, config, warnings)
        retained_segments.extend(
            segment
            for segment in merged_segments
            if segment.release - segment.start
            >= config.minimum_duration_seconds - 1e-12
        )

    retained_segments.sort(key=lambda segment: (segment.start, segment.hand))
    correct_products = catalog.correct_canonical_products(difficulty, language)
    seen_on_list: set[str] = set()
    events: list[ProductGrabEvent] = []

    for index, segment in enumerate(retained_segments):
        overlaps_other_hand = any(
            other_index != index
            and other.hand != segment.hand
            and max(segment.start, other.start) < min(segment.release, other.release)
            for other_index, other in enumerate(retained_segments)
        )
        is_on_list = segment.canonical_product_name in correct_products
        is_first_time_on_list = (
            is_on_list and segment.canonical_product_name not in seen_on_list
        )
        if is_on_list:
            seen_on_list.add(segment.canonical_product_name)
        events.append(
            ProductGrabEvent(
                canonical_product_name=segment.canonical_product_name,
                raw_product_label=segment.raw_product_label,
                hand=segment.hand,
                grab_start_seconds=segment.start,
                grab_release_seconds=segment.release,
                grab_duration_seconds=segment.release - segment.start,
                is_on_list=is_on_list,
                is_first_time_on_list=is_first_time_on_list,
                overlaps_other_hand_grab=overlaps_other_hand,
            )
        )

    return GrabDetectionResult(events=tuple(events), warnings=tuple(warnings))


def detect_focus_episodes(
    tracker: pd.DataFrame,
    catalog: ProductCatalog,
) -> FocusDetectionResult:
    """Detect complete continuous engine-labelled object-focus episodes.

    These records describe the object's label reported by the VR engine.  They
    are not physiological eye fixations.  Durations use observed transition
    timestamps, with no minimum-duration filter or inferred final-row time.
    """
    if "time" not in tracker:
        raise ValueError("Tracker table is missing required column: time")

    warnings: list[str] = []

    focus_values: dict[str, pd.Series] = {}
    for column in ("focus_object_name", "focus_object_tag"):
        if column not in tracker:
            add_unique_warning(warnings, f"focus_signal_unavailable:{column}")
            focus_values[column] = pd.Series("", index=tracker.index, dtype=object)
        else:
            focus_values[column] = tracker[column]

    blink_values: dict[str, pd.Series | None] = {}
    for eye in ("left", "right"):
        column = f"is_{eye}_eye_blinking"
        if column not in tracker or tracker[column].isna().all():
            add_unique_warning(warnings, f"blink_signal_unavailable:{eye}")
            blink_values[eye] = None
        else:
            blink_values[eye] = parse_boolean_series(tracker[column])

    times = pd.to_numeric(tracker["time"], errors="coerce").to_numpy(float)
    episodes: list[FocusEpisode] = []
    active_key: tuple[str, str] | None = None
    active_raw_name = ""
    active_raw_tag = ""
    active_start = np.nan

    def discard_active_episode() -> None:
        nonlocal active_key, active_raw_name, active_raw_tag, active_start
        active_key = None
        active_raw_name = ""
        active_raw_tag = ""
        active_start = np.nan

    def close_active_episode(end: float) -> None:
        if active_key is None:
            return
        if end < active_start:
            add_unique_warning(warnings, "non_monotonic_focus_time")
        else:
            episodes.append(
                FocusEpisode(
                    canonical_focus_name=active_key[0],
                    raw_focus_name=active_raw_name,
                    cleaned_focus_tag=active_key[1],
                    raw_focus_tag=active_raw_tag,
                    focus_start_seconds=float(active_start),
                    focus_end_seconds=float(end),
                    focus_duration_seconds=float(end - active_start),
                )
            )
        discard_active_episode()

    nonempty_placeholders = PLACEHOLDER_OBJECT_NAMES - {""}
    for index in range(len(tracker)):
        timestamp = times[index]
        if not np.isfinite(timestamp):
            if active_key is not None:
                add_unique_warning(warnings, "missing_time_during_focus")
                discard_active_episode()
            continue

        raw_name_value = focus_values["focus_object_name"].iloc[index]
        raw_tag_value = focus_values["focus_object_tag"].iloc[index]
        raw_name = "" if pd.isna(raw_name_value) else str(raw_name_value).strip()
        raw_tag = "" if pd.isna(raw_tag_value) else str(raw_tag_value).strip()
        cleaned_name = clean_product_name(raw_name_value)
        cleaned_tag = clean_product_name(raw_tag_value)

        placeholder_present = (
            cleaned_name in nonempty_placeholders
            or cleaned_tag in nonempty_placeholders
        )
        has_assigned_label = bool(cleaned_name or cleaned_tag) and not placeholder_present
        is_blinking = any(
            values is not None and bool(values.iloc[index])
            for values in blink_values.values()
        )
        key = (
            (catalog.canonicalize(cleaned_name), cleaned_tag)
            if has_assigned_label and not is_blinking
            else None
        )

        if active_key is None:
            if key is not None:
                active_key = key
                active_raw_name = raw_name
                active_raw_tag = raw_tag
                active_start = float(timestamp)
            continue

        if key == active_key:
            continue

        close_active_episode(float(timestamp))
        if key is not None:
            active_key = key
            active_raw_name = raw_name
            active_raw_tag = raw_tag
            active_start = float(timestamp)

    if active_key is not None:
        add_unique_warning(warnings, "open_ended_focus_discarded")

    episodes.sort(key=lambda episode: episode.focus_start_seconds)
    return FocusDetectionResult(episodes=tuple(episodes), warnings=tuple(warnings))


def detect_list_visits(
    focus_episodes: Sequence[FocusEpisode],
) -> tuple[ListVisit, ...]:
    """Return chronological initial-list and later recheck visits.

    A list visit is exactly one already-complete focus episode whose cleaned
    tag is ``tablet`` or whose canonical name is ``tablet`` or ``samsung_tab``.
    Separate focus episodes remain separate visits.
    """
    tablet_names = {"tablet", "samsung_tab"}
    tablet_episodes = sorted(
        (
            episode
            for episode in focus_episodes
            if episode.cleaned_focus_tag == "tablet"
            or episode.canonical_focus_name in tablet_names
        ),
        key=lambda episode: (
            episode.focus_start_seconds,
            episode.focus_end_seconds,
        ),
    )
    return tuple(
        ListVisit(
            list_visit_start_seconds=episode.focus_start_seconds,
            list_visit_end_seconds=episode.focus_end_seconds,
            list_visit_duration_seconds=episode.focus_duration_seconds,
            is_initial_view=index == 0,
        )
        for index, episode in enumerate(tablet_episodes)
    )


def aggregate_trial_pace_and_list_rechecks(
    grab_events: Sequence[ProductGrabEvent],
    list_visits: Sequence[ListVisit],
) -> PaceAndListRecheckAggregation:
    """Aggregate qualifying-grab pace and later list visits for one trial."""
    qualifying_grab_starts = sorted(
        event.grab_start_seconds
        for event in grab_events
        if event.is_on_list and event.is_first_time_on_list
    )
    if len(qualifying_grab_starts) < 2:
        median_pace: float | None = None
    else:
        intervals = np.diff(qualifying_grab_starts)
        median_pace = float(np.median(intervals))

    rechecks = [visit for visit in list_visits if not visit.is_initial_view]
    return PaceAndListRecheckAggregation(
        median_time_between_qualifying_grabs_seconds=median_pace,
        list_recheck_count=len(rechecks),
        total_list_recheck_duration_seconds=float(
            sum(visit.list_visit_duration_seconds for visit in rechecks)
        ),
    )


def _focus_name_key(value: object) -> str:
    return clean_product_name(value).casefold()


def _is_excluded_focus(episode: FocusEpisode, target_names: set[str]) -> bool:
    name = _focus_name_key(episode.canonical_focus_name)
    tag = _focus_name_key(episode.cleaned_focus_tag)
    if not name and not tag:
        return True
    if name in target_names:
        return True
    excluded = {
        _focus_name_key(value) for value in PLACEHOLDER_OBJECT_NAMES | NON_PRODUCT_OBJECT_NAMES
    }
    return name in excluded or tag in excluded


def _is_explicitly_irrelevant_focus(episode: FocusEpisode) -> bool:
    irrelevant = {"npc", "bodum", "side l", "side r"}
    return (
        _focus_name_key(episode.canonical_focus_name) in irrelevant
        or _focus_name_key(episode.cleaned_focus_tag) in irrelevant
    )


def _head_rotation_path_degrees(
    tracker: pd.DataFrame,
    start: float,
    end: float,
    warnings: list[str],
) -> float | None:
    required = ("time", "hmd_rot_x", "hmd_rot_y", "hmd_rot_z", "hmd_rot_w")
    if any(column not in tracker for column in required[1:]):
        add_unique_warning(warnings, "head_rotation_unavailable")
        return None
    values = tracker.loc[:, required].copy()
    for column in required:
        values[column] = pd.to_numeric(values[column], errors="coerce")
    values = values[
        np.isfinite(values["time"])
        & (values["time"] >= start)
        & (values["time"] <= end)
    ]
    rotations: list[np.ndarray] = []
    for row in values.itertuples(index=False):
        quaternion = np.asarray(row[1:5], dtype=float)
        norm = float(np.linalg.norm(quaternion))
        if np.isfinite(norm) and norm > 0:
            rotations.append(quaternion / norm)
    if len(rotations) < 2:
        add_unique_warning(warnings, "head_rotation_unavailable")
        return None
    total = 0.0
    for previous, current in zip(rotations, rotations[1:]):
        dot = float(np.clip(abs(np.dot(previous, current)), -1.0, 1.0))
        total += float(2.0 * np.degrees(np.arccos(dot)))
    return total


def aggregate_trial_locating_features(
    search_intervals: Sequence[SearchInterval],
    focus_episodes: Sequence[FocusEpisode],
    tracker: pd.DataFrame,
) -> LocatingFeatureAggregation:
    """Aggregate locating, irrelevant-focus and headset-turning measurements."""
    warnings: list[str] = []
    ordered_intervals = sorted(
        search_intervals,
        key=lambda interval: interval.qualifying_grab_start_seconds,
    )
    ordered_focus = sorted(
        focus_episodes,
        key=lambda episode: episode.focus_start_seconds,
    )
    target_names = {
        _focus_name_key(interval.canonical_product_name)
        for interval in ordered_intervals
    }
    locating_values: list[float] = []
    irrelevant_values: list[float] = []
    head_turning_values: list[float] = []
    for interval in ordered_intervals:
        if not interval.is_valid:
            if interval.search_start_seconds is None or interval.first_target_focus_seconds is None:
                add_unique_warning(warnings, "locating_time_unavailable")
            continue
        start = interval.search_start_seconds
        target = interval.first_target_focus_seconds
        if start is None or target is None or not np.isfinite(start) or not np.isfinite(target):
            add_unique_warning(warnings, "locating_time_unavailable")
            continue
        locating_duration = float(target - start)
        if locating_duration < 0:
            add_unique_warning(warnings, "negative_locating_duration")
            continue
        locating_values.append(locating_duration)

        irrelevant_duration = 0.0
        seen_focus_labels: set[str] = set()
        for episode in ordered_focus:
            overlap_start = max(start, episode.focus_start_seconds)
            overlap_end = min(target, episode.focus_end_seconds)
            if overlap_end < overlap_start:
                continue
            if not _is_explicitly_irrelevant_focus(episode):
                if _is_excluded_focus(episode, target_names):
                    continue
                label = _focus_name_key(episode.canonical_focus_name) or _focus_name_key(
                    episode.cleaned_focus_tag
                )
                if label and label not in seen_focus_labels:
                    add_unique_warning(warnings, f"unclassifiable_focus:{label}")
                    seen_focus_labels.add(label)
                continue
            irrelevant_duration += float(overlap_end - overlap_start)
        irrelevant_values.append(irrelevant_duration)

        turning = _head_rotation_path_degrees(tracker, float(start), float(target), warnings)
        if turning is not None:
            head_turning_values.append(turning)

    return LocatingFeatureAggregation(
        median_time_to_target_seconds=(
            float(np.median(locating_values)) if locating_values else None
        ),
        median_irrelevant_focus_duration_seconds=(
            float(np.median(irrelevant_values)) if irrelevant_values else None
        ),
        median_head_turning_degrees=(
            float(np.median(head_turning_values)) if head_turning_values else None
        ),
        warnings=tuple(warnings),
    )


def _reach_path_ratio(
    tracker: pd.DataFrame,
    hand: str,
    reach_start: float,
    grab_start: float,
    minimum_straight_distance_meters: float,
    warnings: list[str],
) -> float | None:
    """Calculate one observed 3-D reach path ratio without bridging gaps."""
    motion = prepare_hand_motion(tracker, hand)
    for warning in motion.warnings:
        add_unique_warning(warnings, warning)

    samples = motion.samples
    windowed = [
        (index, sample)
        for index, sample in enumerate(samples)
        if sample.timestamp_seconds is not None
        and reach_start <= sample.timestamp_seconds <= grab_start
    ]
    valid_windowed = [(index, sample) for index, sample in windowed if sample.is_valid]
    if len(valid_windowed) < 2:
        add_unique_warning(warnings, f"insufficient_reach_positions:{hand}")
        return None

    if any(not sample.is_valid for _, sample in windowed):
        add_unique_warning(warnings, f"reach_position_gap:{hand}")

    first_sample = valid_windowed[0][1]
    last_sample = valid_windowed[-1][1]
    first_position = np.array(
        [first_sample.x_meters, first_sample.y_meters, first_sample.z_meters],
        dtype=float,
    )
    last_position = np.array(
        [last_sample.x_meters, last_sample.y_meters, last_sample.z_meters],
        dtype=float,
    )
    if not np.isfinite(first_position).all() or not np.isfinite(last_position).all():
        add_unique_warning(warnings, f"reach_endpoint_position_missing:{hand}")
        return None

    straight_distance = float(np.linalg.norm(last_position - first_position))
    if not np.isfinite(straight_distance):
        add_unique_warning(warnings, f"reach_endpoint_position_missing:{hand}")
        return None
    if straight_distance < minimum_straight_distance_meters:
        add_unique_warning(warnings, "straight_distance_too_small")
        return None

    path_steps: list[float] = []
    for previous_index, current_index in zip(range(len(samples) - 1), range(1, len(samples))):
        previous = samples[previous_index]
        current = samples[current_index]
        if not previous.is_valid or not current.is_valid:
            continue
        if previous.timestamp_seconds is None or current.timestamp_seconds is None:
            continue
        if not (
            reach_start <= previous.timestamp_seconds <= grab_start
            and reach_start <= current.timestamp_seconds <= grab_start
        ):
            continue
        if current.step_distance_meters is not None and np.isfinite(current.step_distance_meters):
            path_steps.append(float(current.step_distance_meters))

    if not path_steps:
        add_unique_warning(warnings, f"insufficient_reach_path:{hand}")
        return None

    path_length = float(sum(path_steps))
    ratio = path_length / straight_distance
    if not np.isfinite(ratio):
        add_unique_warning(warnings, f"reach_path_ratio_unavailable:{hand}")
        return None
    return max(1.0, ratio)


def aggregate_trial_reach_features(
    reach_intervals: Sequence[ReachInterval],
    tracker: pd.DataFrame,
    minimum_straight_distance_meters: float = 0.02,
) -> ReachFeatureAggregation:
    """Aggregate median reach duration and observed 3-D path ratio."""
    if minimum_straight_distance_meters < 0:
        raise ValueError("minimum_straight_distance_meters cannot be negative")

    warnings: list[str] = []
    durations: list[float] = []
    path_ratios: list[float] = []
    ordered_intervals = sorted(
        reach_intervals,
        key=lambda interval: (
            interval.grab_start_seconds
            if interval.grab_start_seconds is not None
            and np.isfinite(interval.grab_start_seconds)
            else float("inf")
        ),
    )

    for interval in ordered_intervals:
        start = interval.reach_start_seconds
        grab = interval.grab_start_seconds
        if not interval.is_valid or start is None or grab is None:
            add_unique_warning(warnings, "reach_duration_unavailable")
            continue
        if not np.isfinite(start) or not np.isfinite(grab):
            add_unique_warning(warnings, "reach_duration_unavailable")
            continue
        duration = float(grab - start)
        if duration < 0:
            add_unique_warning(warnings, "negative_reach_duration")
            continue
        durations.append(duration)

        ratio = _reach_path_ratio(
            tracker,
            interval.hand,
            float(start),
            float(grab),
            minimum_straight_distance_meters,
            warnings,
        )
        if ratio is not None:
            path_ratios.append(ratio)

    return ReachFeatureAggregation(
        median_reach_duration_seconds=(float(np.median(durations)) if durations else None),
        median_reach_path_ratio=(float(np.median(path_ratios)) if path_ratios else None),
        warnings=tuple(warnings),
    )


def _product_event_key(product: object, grab_start: object) -> tuple[str, float] | None:
    """Build an exact canonical-product/recorded-start matching key."""
    if product is None or grab_start is None:
        return None
    try:
        numeric_start = float(grab_start)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(numeric_start):
        return None
    return (_focus_name_key(product), numeric_start)


def build_product_grab_features(
    trial_metadata: Mapping[str, object],
    grab_events: Sequence[ProductGrabEvent],
    search_intervals: Sequence[SearchInterval],
    reach_intervals: Sequence[ReachInterval],
    tracker: pd.DataFrame,
) -> ProductGrabOutputResult:
    """Build one auditable product-grab table without writing a file."""
    warnings: list[str] = []
    search_by_key: dict[tuple[str, float], SearchInterval] = {}
    for interval in search_intervals:
        key = _product_event_key(
            interval.canonical_product_name,
            interval.qualifying_grab_start_seconds,
        )
        if key is None:
            add_unique_warning(warnings, "search_interval_key_unavailable")
        elif key in search_by_key:
            add_unique_warning(warnings, f"duplicate_search_interval:{key[0]}:{key[1]}")
        else:
            search_by_key[key] = interval

    reach_by_key: dict[tuple[str, float], ReachInterval] = {}
    for interval in reach_intervals:
        key = _product_event_key(interval.canonical_product_name, interval.grab_start_seconds)
        if key is None:
            add_unique_warning(warnings, "reach_interval_key_unavailable")
        elif key in reach_by_key:
            add_unique_warning(warnings, f"duplicate_reach_interval:{key[0]}:{key[1]}")
        else:
            reach_by_key[key] = interval

    rows: list[dict[str, object]] = []
    ordered_events = sorted(
        grab_events,
        key=lambda event: (event.grab_start_seconds, event.hand),
    )
    metadata_keys = (
        "participant_id",
        "session_id",
        "source_tracker_csv_filename",
        "participant_group",
        "condition_name",
        "difficulty_level",
        "trial_order",
        "language",
    )
    for event in ordered_events:
        row_warnings: list[str] = []
        key = _product_event_key(event.canonical_product_name, event.grab_start_seconds)
        search = search_by_key.get(key) if key is not None and event.is_first_time_on_list else None
        reach = reach_by_key.get(key) if key is not None and event.is_first_time_on_list else None

        search_start = None
        target_focus = None
        search_valid = None
        if event.is_first_time_on_list:
            if search is None:
                warning = f"search_interval_unmatched:{event.canonical_product_name}:{event.grab_start_seconds}"
                add_unique_warning(row_warnings, warning)
                add_unique_warning(warnings, warning)
            else:
                search_start = search.search_start_seconds
                target_focus = search.first_target_focus_seconds
                search_valid = search.is_valid

        reach_start = None
        reach_duration = None
        reach_path_ratio = None
        reach_valid = None
        if event.is_first_time_on_list:
            if reach is None:
                warning = f"reach_interval_unmatched:{event.canonical_product_name}:{event.grab_start_seconds}"
                add_unique_warning(row_warnings, warning)
                add_unique_warning(warnings, warning)
            else:
                reach_start = reach.reach_start_seconds
                reach_valid = reach.is_valid
                if reach.is_valid and reach_start is not None and np.isfinite(reach_start):
                    duration = float(event.grab_start_seconds - reach_start)
                    if duration >= 0 and np.isfinite(duration):
                        reach_duration = duration
                    else:
                        warning = f"negative_reach_duration:{event.hand}"
                        add_unique_warning(row_warnings, warning)
                        add_unique_warning(warnings, warning)
                    if reach_duration is not None:
                        reach_path_ratio = _reach_path_ratio(
                            tracker,
                            event.hand,
                            float(reach_start),
                            float(event.grab_start_seconds),
                            0.02,
                            row_warnings,
                        )
                        for warning in row_warnings:
                            add_unique_warning(warnings, warning)

        row: dict[str, object] = {key_name: trial_metadata.get(key_name) for key_name in metadata_keys}
        row.update(
            {
                "canonical_product_name": event.canonical_product_name,
                "raw_product_label": event.raw_product_label,
                "hand": event.hand,
                "grab_start_seconds": event.grab_start_seconds,
                "grab_release_seconds": event.grab_release_seconds,
                "grab_duration_seconds": event.grab_duration_seconds,
                "is_on_list": event.is_on_list,
                "is_first_time_on_list": event.is_first_time_on_list,
                "is_repeated_on_list": event.is_on_list and not event.is_first_time_on_list,
                "is_off_list": not event.is_on_list,
                "overlaps_other_hand_grab": event.overlaps_other_hand_grab,
                "search_start_seconds": search_start,
                "first_target_focus_seconds": target_focus,
                "search_interval_valid": search_valid,
                "reach_start_seconds": reach_start,
                "reach_duration_seconds": reach_duration,
                "reach_path_ratio": reach_path_ratio,
                "reach_interval_valid": reach_valid,
                "processing_warnings": ";".join(row_warnings),
            }
        )
        rows.append(row)

    table = pd.DataFrame(rows, columns=PRODUCT_GRAB_OUTPUT_COLUMNS)
    return ProductGrabOutputResult(table=table, warnings=tuple(warnings))


def write_product_grab_features(table: pd.DataFrame, path: Path) -> None:
    """Write a product-grab table using the fixed schema and blank missing cells."""
    output = table.copy()
    for column in PRODUCT_GRAB_OUTPUT_COLUMNS:
        if column not in output.columns:
            output[column] = pd.NA
    output = output.loc[:, PRODUCT_GRAB_OUTPUT_COLUMNS]
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(destination, index=False, encoding="utf-8", na_rep="")


TRIAL_VALIDITY_COUNT_COLUMNS = (
    "real_product_grab_count",
    "first_time_on_list_grab_count",
    "list_visit_count",
    "valid_search_interval_count",
    "valid_reach_duration_count",
    "valid_reach_path_ratio_count",
)


def _coerce_trial_validity_count(
    value: object,
    key: str,
    warnings: list[str],
) -> int | float | None:
    """Normalize an optional trial-validity count without inventing values."""
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        add_unique_warning(warnings, f"invalid_validity_count:{key}")
        return None
    if not np.isfinite(numeric) or numeric < 0:
        add_unique_warning(warnings, f"invalid_validity_count:{key}")
        return None
    return int(numeric) if numeric.is_integer() else numeric


def build_trial_features(
    trial_metadata: Mapping[str, object],
    pace_and_rechecks: PaceAndListRecheckAggregation | None,
    locating_features: LocatingFeatureAggregation | None,
    reach_features: ReachFeatureAggregation | None,
    error_outcome: ErrorOutcomeRecord | None = None,
    validity_counts: Mapping[str, object] | None = None,
) -> TrialFeatureOutputResult:
    """Build one fixed-schema trial-feature row without writing a file."""
    warnings: list[str] = []
    for aggregate in (locating_features, reach_features):
        for warning in getattr(aggregate, "warnings", ()):
            add_unique_warning(warnings, warning)

    row: dict[str, object] = {
        key: trial_metadata.get(key) for key in TRIAL_FEATURE_COLUMNS
    }
    row.update(
        {
            "performance": trial_metadata.get("performance"),
            "mental_demand_score_0_to_10": trial_metadata.get(
                "mental_demand_score_0_to_10"
            ),
            "median_time_between_qualifying_grabs_seconds": getattr(
                pace_and_rechecks,
                "median_time_between_qualifying_grabs_seconds",
                None,
            ),
            "list_recheck_count": getattr(
                pace_and_rechecks, "list_recheck_count", None
            ),
            "total_list_recheck_duration_seconds": getattr(
                pace_and_rechecks,
                "total_list_recheck_duration_seconds",
                None,
            ),
            "median_time_to_target_seconds": getattr(
                locating_features, "median_time_to_target_seconds", None
            ),
            "median_irrelevant_focus_duration_seconds": getattr(
                locating_features,
                "median_irrelevant_focus_duration_seconds",
                None,
            ),
            "median_head_turning_degrees": getattr(
                locating_features, "median_head_turning_degrees", None
            ),
            "median_reach_duration_seconds": getattr(
                reach_features, "median_reach_duration_seconds", None
            ),
            "median_reach_path_ratio": getattr(
                reach_features, "median_reach_path_ratio", None
            ),
        }
    )

    error_columns = (
        "errors_missing",
        "errors_wrong_order",
        "errors_duplicate",
        "errors_not_in_list",
        "total_error_count",
        "error_change_from_d0",
    )
    for key in error_columns:
        row[key] = getattr(error_outcome, key, None) if error_outcome is not None else None

    for key in TRIAL_VALIDITY_COUNT_COLUMNS:
        value = validity_counts.get(key) if validity_counts is not None else None
        row[key] = _coerce_trial_validity_count(value, key, warnings)

    row["processing_warnings"] = ";".join(warnings)
    table = pd.DataFrame([row], columns=TRIAL_FEATURE_COLUMNS)
    return TrialFeatureOutputResult(table=table, warnings=tuple(warnings))


def write_trial_features(table: pd.DataFrame, path: Path) -> None:
    """Write the fixed-schema trial-feature table with blank missing cells."""
    output = table.copy()
    for column in TRIAL_FEATURE_COLUMNS:
        if column not in output.columns:
            output[column] = pd.NA
    output = output.loc[:, TRIAL_FEATURE_COLUMNS]
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(destination, index=False, encoding="utf-8", na_rep="")


def detect_search_intervals(
    tracker: pd.DataFrame,
    grab_events: Sequence[ProductGrabEvent],
    list_visits: Sequence[ListVisit],
    focus_episodes: Sequence[FocusEpisode],
) -> SearchIntervalResult:
    """Build shared search records for qualifying first-time-on-list grabs."""
    if "time" not in tracker:
        raise ValueError("Tracker table is missing required column: time")

    warnings: list[str] = []
    times = pd.to_numeric(tracker["time"], errors="coerce").to_numpy(float)
    usable_activity_start: float | None = None
    blackout_column = "enableBlackout"
    if blackout_column in tracker and not tracker[blackout_column].isna().all():
        blackout = parse_boolean_series(tracker[blackout_column]).to_numpy(bool)
        valid_starts = np.flatnonzero(np.isfinite(times) & ~blackout)
        if len(valid_starts):
            usable_activity_start = float(times[valid_starts[0]])
    if usable_activity_start is None:
        add_unique_warning(warnings, "usable_activity_start_unavailable")

    qualifying_grabs = sorted(
        (
            event
            for event in grab_events
            if event.is_on_list and event.is_first_time_on_list
        ),
        key=lambda event: (
            event.grab_start_seconds,
            event.grab_release_seconds,
        ),
    )
    ordered_visits = sorted(
        list_visits,
        key=lambda visit: (
            visit.list_visit_start_seconds,
            visit.list_visit_end_seconds,
        ),
    )
    ordered_focus = sorted(
        focus_episodes,
        key=lambda episode: (
            episode.focus_start_seconds,
            episode.focus_end_seconds,
        ),
    )

    intervals: list[SearchInterval] = []
    previous_qualifying_grab_start: float | None = None
    for event in qualifying_grabs:
        grab_start = event.grab_start_seconds
        completed_visit_ends = [
            visit.list_visit_end_seconds
            for visit in ordered_visits
            if visit.list_visit_end_seconds <= grab_start
        ]
        latest_visit_end = max(completed_visit_ends, default=None)

        if previous_qualifying_grab_start is None:
            if usable_activity_start is None:
                intervals.append(
                    SearchInterval(
                        canonical_product_name=event.canonical_product_name,
                        qualifying_grab_start_seconds=grab_start,
                        search_start_seconds=None,
                        first_target_focus_seconds=None,
                        is_valid=False,
                    )
                )
                previous_qualifying_grab_start = grab_start
                continue
            search_start = max(
                usable_activity_start,
                latest_visit_end if latest_visit_end is not None else usable_activity_start,
            )
        else:
            search_start = max(
                previous_qualifying_grab_start,
                latest_visit_end
                if latest_visit_end is not None
                else previous_qualifying_grab_start,
            )

        if search_start > grab_start:
            add_unique_warning(warnings, "search_start_after_grab")
            intervals.append(
                SearchInterval(
                    canonical_product_name=event.canonical_product_name,
                    qualifying_grab_start_seconds=grab_start,
                    search_start_seconds=None,
                    first_target_focus_seconds=None,
                    is_valid=False,
                )
            )
            previous_qualifying_grab_start = grab_start
            continue

        target_focus_candidates = [
            max(search_start, episode.focus_start_seconds)
            for episode in ordered_focus
            if episode.canonical_focus_name == event.canonical_product_name
            and episode.focus_start_seconds <= grab_start
            and episode.focus_end_seconds >= search_start
        ]
        target_focus_candidates = [
            candidate for candidate in target_focus_candidates if candidate <= grab_start
        ]
        first_target_focus = (
            min(target_focus_candidates) if target_focus_candidates else None
        )
        if first_target_focus is None:
            add_unique_warning(warnings, "target_focus_not_found")
        intervals.append(
            SearchInterval(
                canonical_product_name=event.canonical_product_name,
                qualifying_grab_start_seconds=grab_start,
                search_start_seconds=search_start,
                first_target_focus_seconds=first_target_focus,
                is_valid=first_target_focus is not None,
            )
        )
        previous_qualifying_grab_start = grab_start

    return SearchIntervalResult(
        intervals=tuple(intervals), warnings=tuple(warnings)
    )


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
