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
import hashlib
import json
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

MENTAL_DEMAND_COLUMN_ALIASES = {
    "participant": "participant",
    "participant_id": "participant",
    "group": "group",
    "participant_group": "group",
    "condition": "condition",
    "condition_name": "condition",
    "trial": "trial_order",
    "trial_order": "trial_order",
    "level": "difficulty",
    "difficulty": "difficulty",
    "difficulty_level": "difficulty",
    "md": "mental_demand",
    "mental demand": "mental_demand",
    "mental_demand": "mental_demand",
    "mental_demand_score_0_to_10": "mental_demand",
}

PERFORMANCE_PERCENT_ALIASES = {
    "performance_percent",
    "performance_percentage",
    "performance_pct",
    "percent_performance",
}

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


class DescriptivePlotResult(NamedTuple):
    """Generated descriptive plot paths and unique plotting warnings."""

    paths: tuple[Path, ...]
    warnings: tuple[str, ...]


class FullExtractionResult(NamedTuple):
    """Completed FE-01 tables, generated files, and unique run warnings."""

    product_grab_table: pd.DataFrame
    trial_feature_table: pd.DataFrame
    qc_table: pd.DataFrame
    output_paths: tuple[Path, ...]
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
    "correct_products_collected_count",
    "performance_percent",
    "performance_change_from_d0_percentage_points",
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


QC_COLUMNS = (
    "participant_id",
    "session_id",
    "source_tracker_csv_filename",
    "participant_group",
    "condition_name",
    "difficulty_level",
    "trial_order",
    "language",
    "tracker_path",
    "participant_details_path",
    "tracker_file_exists",
    "participant_details_file_exists",
    "tracker_loaded",
    "tracker_row_count",
    "finite_time_row_count",
    "participant_hand",
    "malformed_time_header_repaired",
    "processing_status",
    "real_product_grab_count",
    "first_time_on_list_grab_count",
    "list_visit_count",
    "valid_search_interval_count",
    "valid_reach_duration_count",
    "valid_reach_path_ratio_count",
    "missing_behavioural_measure_count",
    "mental_demand_available",
    "warning_count",
    "processing_warnings",
)


BEHAVIOURAL_MEASURE_COLUMNS = (
    "median_time_between_qualifying_grabs_seconds",
    "list_recheck_count",
    "total_list_recheck_duration_seconds",
    "median_time_to_target_seconds",
    "median_irrelevant_focus_duration_seconds",
    "median_head_turning_degrees",
    "median_reach_duration_seconds",
    "median_reach_path_ratio",
)


DESCRIPTIVE_PLOT_CONDITIONS = ("Visual", "Auditory", "Cognitive")
DESCRIPTIVE_PLOT_DIFFICULTIES = (0, 2, 6, 10)
DESCRIPTIVE_PLOT_DIFFICULTY_LABELS = tuple(
    f"D{difficulty}" for difficulty in DESCRIPTIVE_PLOT_DIFFICULTIES
)
DESCRIPTIVE_PLOT_SPECS = (
    (
        "01_time_between_correct_grabs.png",
        "median_time_between_qualifying_grabs_seconds",
        "Time between qualifying correct-product grabs (seconds)",
        "Time between qualifying correct-product grabs",
    ),
    (
        "02_list_recheck_count.png",
        "list_recheck_count",
        "List rechecks (count)",
        "List recheck count",
    ),
    (
        "03_total_list_recheck_time.png",
        "total_list_recheck_duration_seconds",
        "Total list recheck duration (seconds)",
        "Total list recheck duration",
    ),
    (
        "04_time_to_locate_target.png",
        "median_time_to_target_seconds",
        "Time to locate target (seconds)",
        "Time to locate target",
    ),
    (
        "05_irrelevant_focus_time.png",
        "median_irrelevant_focus_duration_seconds",
        "Irrelevant focus duration (seconds)",
        "Irrelevant focus duration",
    ),
    (
        "06_reach_time.png",
        "median_reach_duration_seconds",
        "Reach duration (seconds)",
        "Reach duration",
    ),
    (
        "07_reach_path_ratio.png",
        "median_reach_path_ratio",
        "Reach path ratio (dimensionless)",
        "Reach path ratio",
    ),
    (
        "08_head_turning.png",
        "median_head_turning_degrees",
        "Headset angular movement (degrees)",
        "Headset angular movement",
    ),
    (
        "09_total_errors.png",
        "total_error_count",
        "Total errors (count)",
        "Total errors",
    ),
    (
        "10_error_change_from_D0.png",
        "error_change_from_d0",
        "Error change from D0 (count)",
        "Error change from D0",
    ),
    (
        "11_relative_performance_change.png",
        "__relative_performance_change",
        "Relative performance change (percentage points)",
        "Relative performance change",
    ),
    (
        "12_subjective_mental_demand.png",
        "mental_demand_score_0_to_10",
        "Subjective mental demand (0–10)",
        "Subjective mental demand",
    ),
)
DESCRIPTIVE_COVERAGE_FEATURES = tuple(spec[1] for spec in DESCRIPTIVE_PLOT_SPECS[:8])


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


class _TrialSignalCache:
    """Private cache whose lifetime is limited to one tracker trial."""

    def __init__(self, tracker: pd.DataFrame):
        self.tracker = tracker
        self._time_seconds: np.ndarray | None = None
        self._blackout_values: np.ndarray | None = None
        self._blackout_prepared = False
        self._hand_motion: dict[str, HandMotionSeries] = {}
        self._head_rotations: tuple[tuple[float, np.ndarray], ...] | None = None
        self._head_rotation_prepared = False
        self._head_rotation_columns_missing = False
        self._focus_source_id: int | None = None
        self._sorted_focus: tuple[FocusEpisode, ...] = ()

    def time_seconds(self) -> np.ndarray:
        if self._time_seconds is None:
            if "time" not in self.tracker.columns:
                raise ValueError("Tracker table missing required column: time")
            self._time_seconds = pd.to_numeric(
                self.tracker["time"], errors="coerce"
            ).to_numpy(float)
        return self._time_seconds

    def blackout_values(self) -> np.ndarray | None:
        if not self._blackout_prepared:
            column = self.tracker.get("enableBlackout")
            if column is None or column.isna().all():
                self._blackout_values = None
            else:
                self._blackout_values = parse_boolean_series(column).to_numpy(bool)
            self._blackout_prepared = True
        return self._blackout_values

    def hand_motion(self, hand: str) -> HandMotionSeries:
        if hand not in self._hand_motion:
            self._hand_motion[hand] = prepare_hand_motion(self.tracker, hand)
        return self._hand_motion[hand]

    def head_rotations(
        self, warnings: list[str]
    ) -> tuple[tuple[float, np.ndarray], ...]:
        if not self._head_rotation_prepared:
            required = ("hmd_rot_x", "hmd_rot_y", "hmd_rot_z", "hmd_rot_w")
            if any(column not in self.tracker.columns for column in required):
                self._head_rotation_columns_missing = True
                self._head_rotations = ()
            else:
                times = self.time_seconds()
                components = [
                    pd.to_numeric(self.tracker[column], errors="coerce").to_numpy(float)
                    for column in required
                ]
                rotations: list[tuple[float, np.ndarray]] = []
                for index, timestamp in enumerate(times):
                    if not np.isfinite(timestamp):
                        continue
                    quaternion = np.asarray(
                        [component[index] for component in components], dtype=float
                    )
                    norm = float(np.linalg.norm(quaternion))
                    if np.isfinite(norm) and norm > 0:
                        rotations.append((float(timestamp), quaternion / norm))
                self._head_rotations = tuple(rotations)
            self._head_rotation_prepared = True
        if self._head_rotation_columns_missing:
            add_unique_warning(warnings, "head_rotation_unavailable")
        return self._head_rotations or ()

    def sorted_focus(self, focus_episodes: Sequence[FocusEpisode]) -> tuple[FocusEpisode, ...]:
        source_id = id(focus_episodes)
        if self._focus_source_id != source_id:
            self._sorted_focus = tuple(
                sorted(
                    focus_episodes,
                    key=lambda episode: episode.focus_start_seconds,
                )
            )
            self._focus_source_id = source_id
        return self._sorted_focus


class ReachDetectionConfig(NamedTuple):
    """Provisional historical settings for movement-based reach onset."""

    absolute_speed_threshold_meters_per_second: float = 0.05
    peak_speed_fraction: float = 0.05
    minimum_movement_duration_seconds: float = 0.10


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


def _normalise_group(value: object) -> str | None:
    """Return a case-insensitive participant-group key."""
    if value is None or pd.isna(value):
        return None
    text = str(value).strip().casefold()
    return text or None


def _performance_key(row: Mapping[str, object]) -> tuple[str, str]:
    """Build the same-participant/same-condition performance baseline key."""
    return (str(row["participant_id"]), str(row["condition_name"]))


def _derive_performance_metrics(
    trials: pd.DataFrame,
) -> tuple[pd.DataFrame, tuple[str, ...]]:
    """Add explicit count, percentage, and D0-change performance columns.

    The inspected ``performance_1.csv`` stores correct-product counts in its
    ``performance`` column.  Explicit percentage-named columns take priority
    when present so an already-percent source is never multiplied twice.
    """
    output = trials.copy()
    warnings: list[str] = []
    explicit_percent_column = next(
        (
            column
            for column in output.columns
            if str(column).strip().casefold() in PERFORMANCE_PERCENT_ALIASES
        ),
        None,
    )
    output["correct_products_collected_count"] = np.nan
    output["performance_percent"] = np.nan
    output["performance_change_from_d0_percentage_points"] = np.nan
    if explicit_percent_column is not None:
        percent_values = pd.to_numeric(
            output[explicit_percent_column], errors="coerce"
        )
        invalid = output[explicit_percent_column].notna() & ~np.isfinite(
            percent_values.to_numpy(float)
        )
        if bool(invalid.any()):
            add_unique_warning(warnings, "invalid_performance_percent_values")
        out_of_range = percent_values.notna() & ~percent_values.between(0, 100)
        if bool(out_of_range.any()):
            add_unique_warning(warnings, "performance_percent_out_of_range")
        output["performance_percent"] = percent_values.where(~out_of_range)
        if "performance" in output.columns:
            counts = pd.to_numeric(output["performance"], errors="coerce")
            output["correct_products_collected_count"] = counts.where(
                counts.between(0, 20)
            )
    elif "performance" in output.columns:
        counts = pd.to_numeric(output["performance"], errors="coerce")
        finite_counts = counts.notna() & np.isfinite(counts.to_numpy(float))
        valid_counts = finite_counts & counts.between(0, 20)
        if bool((finite_counts & ~valid_counts).any()):
            add_unique_warning(warnings, "performance_units_unresolved")
        output["correct_products_collected_count"] = counts.where(valid_counts)
        output["performance_percent"] = (counts * 5).where(valid_counts)
        add_unique_warning(warnings, "performance_interpreted_as_correct_product_count")
    else:
        add_unique_warning(warnings, "performance_source_unavailable")

    baseline_values: dict[tuple[str, str], list[float]] = {}
    for row in output.to_dict(orient="records"):
        if row.get("difficulty_level") == 0:
            value = _finite_numeric_or_none(row.get("performance_percent"))
            if value is not None:
                baseline_values.setdefault(_performance_key(row), []).append(value)
    baselines: dict[tuple[str, str], float] = {}
    for key, values in baseline_values.items():
        if len(values) > 1:
            add_unique_warning(
                warnings, f"duplicate_performance_d0_baseline:{key[0]}:{key[1]}"
            )
        baselines[key] = float(np.median(values))

    unavailable_keys: set[tuple[str, str]] = set()
    for index, row in output.iterrows():
        difficulty = row.get("difficulty_level")
        if difficulty == 0:
            continue
        if difficulty not in (2, 6, 10):
            add_unique_warning(warnings, f"unsupported_performance_difficulty:{difficulty}")
            continue
        key = _performance_key(row)
        baseline = baselines.get(key)
        current = _finite_numeric_or_none(row.get("performance_percent"))
        if baseline is None or current is None:
            unavailable_keys.add(key)
            continue
        output.at[index, "performance_change_from_d0_percentage_points"] = (
            baseline - current
        )
    for participant, condition in sorted(unavailable_keys):
        add_unique_warning(
            warnings,
            f"performance_baseline_or_value_unavailable:{participant}:{condition}",
        )
    return output, tuple(warnings)


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
    warnings: list[str] = []
    if "time" not in table.columns:
        repaired_columns = [
            column
            for column in table.columns
            if re.fullmatch(r"\d+time", str(column), flags=re.IGNORECASE)
        ]
        if len(repaired_columns) == 1:
            original_column = repaired_columns[0]
            table = table.rename(columns={original_column: "time"})
            warnings.append(f"repaired_time_header:{original_column}")
        else:
            raise ValueError("Tracker table is missing required column: time")
    table = table.copy()
    table["time"] = pd.to_numeric(table["time"], errors="coerce")
    if table["time"].notna().sum() == 0:
        raise ValueError("Tracker table has no usable recorded time values")

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


def _detect_reach_intervals_with_cache(
    tracker: pd.DataFrame,
    grab_events: Sequence[ProductGrabEvent],
    config: ReachDetectionConfig | None = None,
    *,
    search_intervals: Sequence[SearchInterval] = (),
    cache: _TrialSignalCache,
) -> ReachDetectionResult:
    """Detect the final sustained movement block before each qualifying grab.

    Reach onset is the beginning of the final above-threshold movement block
    in the accepted search-start-to-grab interval.  The absolute threshold,
    peak-speed fraction, and sustained duration are provisional historical
    settings and are not validated here.
    """
    config = config or ReachDetectionConfig()
    if config.absolute_speed_threshold_meters_per_second < 0:
        raise ValueError("absolute_speed_threshold_meters_per_second cannot be negative")
    if not 0 <= config.peak_speed_fraction <= 1:
        raise ValueError("peak_speed_fraction must be between 0 and 1")
    if config.minimum_movement_duration_seconds < 0:
        raise ValueError("minimum_movement_duration_seconds cannot be negative")

    warnings: list[str] = []
    intervals: list[ReachInterval] = []
    search_by_key: dict[tuple[str, float], SearchInterval] = {}
    for search in search_intervals or ():
        try:
            key = (
                clean_product_name(search.canonical_product_name),
                float(search.qualifying_grab_start_seconds),
            )
        except (TypeError, ValueError):
            add_unique_warning(warnings, "reach_search_interval_key_unavailable")
            continue
        if not np.isfinite(key[1]):
            add_unique_warning(warnings, "reach_search_interval_key_unavailable")
            continue
        if key in search_by_key:
            add_unique_warning(
                warnings,
                f"duplicate_reach_search_interval:{key[0]}:{key[1]}",
            )
        else:
            search_by_key[key] = search

    qualifying_grabs = sorted(
        (
            event
            for event in grab_events
            if event.is_on_list and event.is_first_time_on_list
        ),
        key=lambda event: event.grab_start_seconds,
    )
    for event in qualifying_grabs:
        hand = actual_grab_hand(event)
        try:
            grab_start = float(event.grab_start_seconds)
        except (TypeError, ValueError):
            grab_start = float("nan")
        search = search_by_key.get(
            (clean_product_name(event.canonical_product_name), grab_start)
        )
        if search is None or not search.is_valid:
            add_unique_warning(warnings, f"reach_search_interval_unavailable:{hand}")
            intervals.append(
                ReachInterval(
                    canonical_product_name=event.canonical_product_name,
                    hand=hand,
                    reach_start_seconds=None,
                    grab_start_seconds=grab_start,
                    is_valid=False,
                )
            )
            continue
        try:
            search_start = float(search.search_start_seconds)
        except (TypeError, ValueError):
            search_start = float("nan")
        if (
            not np.isfinite(search_start)
            or not np.isfinite(grab_start)
            or search_start > grab_start
        ):
            add_unique_warning(warnings, f"reach_interval_boundaries_invalid:{hand}")
            intervals.append(
                ReachInterval(
                    canonical_product_name=event.canonical_product_name,
                    hand=hand,
                    reach_start_seconds=None,
                    grab_start_seconds=grab_start,
                    is_valid=False,
                )
            )
            continue

        motion = cache.hand_motion(hand)
        for warning in motion.warnings:
            add_unique_warning(warnings, warning)
        candidate_samples = [
            sample
            for sample in motion.samples
            if sample.timestamp_seconds is not None
            and search_start <= sample.timestamp_seconds <= grab_start
        ]
        finite_speeds = [
            float(sample.step_speed_meters_per_second)
            for sample in candidate_samples
            if sample.is_valid
            and sample.step_speed_meters_per_second is not None
            and np.isfinite(sample.step_speed_meters_per_second)
        ]
        if not finite_speeds:
            add_unique_warning(warnings, f"reach_onset_not_found:{hand}")
            intervals.append(
                ReachInterval(
                    canonical_product_name=event.canonical_product_name,
                    hand=hand,
                    reach_start_seconds=None,
                    grab_start_seconds=grab_start,
                    is_valid=False,
                )
            )
            continue

        peak_speed = max(finite_speeds)
        threshold = max(
            config.absolute_speed_threshold_meters_per_second,
            config.peak_speed_fraction * peak_speed,
        )
        movement_blocks: list[tuple[list[HandPositionSample], float | None]] = []
        active_block: list[HandPositionSample] = []
        active_block_onset: float | None = None
        previous_timestamp: float | None = None
        sample_positions = {
            id(sample): index for index, sample in enumerate(motion.samples)
        }

        def close_block() -> None:
            nonlocal active_block, active_block_onset
            if active_block:
                movement_blocks.append((active_block, active_block_onset))
            active_block = []
            active_block_onset = None

        for sample in candidate_samples:
            timestamp = sample.timestamp_seconds
            speed = sample.step_speed_meters_per_second
            if timestamp is None or not sample.is_valid or speed is None:
                close_block()
                previous_timestamp = None
                continue
            if not np.isfinite(timestamp) or not np.isfinite(speed):
                close_block()
                previous_timestamp = None
                continue
            if previous_timestamp is not None and timestamp <= previous_timestamp:
                add_unique_warning(warnings, f"non_monotonic_reach_time:{hand}")
                close_block()
            if float(speed) >= threshold:
                if not active_block:
                    sample_index = sample_positions.get(id(sample))
                    predecessor = (
                        motion.samples[sample_index - 1]
                        if sample_index is not None and sample_index > 0
                        else None
                    )
                    predecessor_time = (
                        predecessor.timestamp_seconds
                        if predecessor is not None and predecessor.is_valid
                        else None
                    )
                    if predecessor_time is None or not np.isfinite(predecessor_time):
                        add_unique_warning(
                            warnings,
                            f"reach_onset_predecessor_unavailable:{hand}",
                        )
                        active_block_onset = None
                    else:
                        active_block_onset = max(
                            float(search_start),
                            float(predecessor_time),
                        )
                active_block.append(sample)
            else:
                close_block()
            previous_timestamp = float(timestamp)
        close_block()

        qualifying_blocks = [
            (block, onset)
            for block, onset in movement_blocks
            if onset is not None
            and block[-1].timestamp_seconds - block[0].timestamp_seconds
            + 1e-12
            >= config.minimum_movement_duration_seconds
        ]
        if not qualifying_blocks:
            add_unique_warning(warnings, f"reach_onset_not_found:{hand}")
            intervals.append(
                ReachInterval(
                    canonical_product_name=event.canonical_product_name,
                    hand=hand,
                    reach_start_seconds=None,
                    grab_start_seconds=grab_start,
                    is_valid=False,
                )
            )
            continue
        reach_start = float(qualifying_blocks[-1][1])
        intervals.append(
            ReachInterval(
                canonical_product_name=event.canonical_product_name,
                hand=hand,
                reach_start_seconds=reach_start,
                grab_start_seconds=grab_start,
                is_valid=True,
            )
        )
    return ReachDetectionResult(intervals=tuple(intervals), warnings=tuple(warnings))


def detect_reach_intervals(
    tracker: pd.DataFrame,
    grab_events: Sequence[ProductGrabEvent],
    config: ReachDetectionConfig | None = None,
    *,
    search_intervals: Sequence[SearchInterval] = (),
) -> ReachDetectionResult:
    """Detect reach intervals using a fresh trial cache."""
    return _detect_reach_intervals_with_cache(
        tracker,
        grab_events,
        config,
        search_intervals=search_intervals,
        cache=_TrialSignalCache(tracker),
    )


def _complete_grab_segments_for_hand(
    tracker: pd.DataFrame,
    hand: str,
    catalog: ProductCatalog,
    warnings: list[str],
    *,
    time_seconds: np.ndarray | None = None,
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
    times = (
        time_seconds
        if time_seconds is not None
        else pd.to_numeric(tracker["time"], errors="coerce").to_numpy(float)
    )

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


def _detect_product_grab_events_with_cache(
    tracker: pd.DataFrame,
    catalog: ProductCatalog,
    difficulty: int,
    language: str,
    config: GrabDetectionConfig | None = None,
    *,
    cache: _TrialSignalCache,
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
            tracker,
            hand,
            catalog,
            warnings,
            time_seconds=cache.time_seconds(),
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


def detect_product_grab_events(
    tracker: pd.DataFrame,
    catalog: ProductCatalog,
    difficulty: int,
    language: str,
    config: GrabDetectionConfig | None = None,
) -> GrabDetectionResult:
    """Detect complete real-product grabs using a fresh trial cache."""
    return _detect_product_grab_events_with_cache(
        tracker,
        catalog,
        difficulty,
        language,
        config,
        cache=_TrialSignalCache(tracker),
    )


def _detect_focus_episodes_with_cache(
    tracker: pd.DataFrame,
    catalog: ProductCatalog,
    *,
    cache: _TrialSignalCache,
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

    times = cache.time_seconds()
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


def detect_focus_episodes(
    tracker: pd.DataFrame,
    catalog: ProductCatalog,
) -> FocusDetectionResult:
    """Detect complete focus episodes using a fresh trial cache."""
    return _detect_focus_episodes_with_cache(
        tracker,
        catalog,
        cache=_TrialSignalCache(tracker),
    )


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


def _is_excluded_focus(
    episode: FocusEpisode,
    target_names: set[str],
    catalog: ProductCatalog | None = None,
) -> bool:
    name = _focus_name_key(episode.canonical_focus_name)
    tag = _focus_name_key(episode.cleaned_focus_tag)
    if not name and not tag:
        return True
    if name in target_names:
        return True
    if catalog is not None:
        if catalog.canonicalize(name) in catalog.all_canonical_products:
            return True
        if catalog.canonicalize(tag) in catalog.all_canonical_products:
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


def _head_rotation_path_degrees_from_prepared(
    rotations: Sequence[tuple[float, np.ndarray]],
    start: float,
    end: float,
    warnings: list[str],
) -> float | None:
    """Calculate the existing head-rotation path from prepared samples."""
    windowed = [
        quaternion
        for timestamp, quaternion in rotations
        if start <= timestamp <= end
    ]
    if len(windowed) < 2:
        add_unique_warning(warnings, "head_rotation_unavailable")
        return None
    total = 0.0
    for previous, current in zip(windowed, windowed[1:]):
        dot = float(np.clip(abs(np.dot(previous, current)), -1.0, 1.0))
        total += float(2.0 * np.degrees(np.arccos(dot)))
    return total if np.isfinite(total) else None


def _aggregate_trial_locating_features_with_cache(
    search_intervals: Sequence[SearchInterval],
    focus_episodes: Sequence[FocusEpisode],
    tracker: pd.DataFrame,
    catalog: ProductCatalog | None = None,
    *,
    cache: _TrialSignalCache,
) -> LocatingFeatureAggregation:
    """Aggregate locating, irrelevant-focus and headset-turning measurements."""
    warnings: list[str] = []
    ordered_intervals = sorted(
        search_intervals,
        key=lambda interval: interval.qualifying_grab_start_seconds,
    )
    ordered_focus = cache.sorted_focus(focus_episodes)
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
            if episode.focus_start_seconds > target:
                break
            overlap_start = max(start, episode.focus_start_seconds)
            overlap_end = min(target, episode.focus_end_seconds)
            if overlap_end < overlap_start:
                continue
            if not _is_explicitly_irrelevant_focus(episode):
                if _is_excluded_focus(episode, target_names, catalog):
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

        turning = _head_rotation_path_degrees_from_prepared(
            cache.head_rotations(warnings),
            float(start),
            float(target),
            warnings,
        )
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


def aggregate_trial_locating_features(
    search_intervals: Sequence[SearchInterval],
    focus_episodes: Sequence[FocusEpisode],
    tracker: pd.DataFrame,
    catalog: ProductCatalog | None = None,
) -> LocatingFeatureAggregation:
    """Aggregate locating features using a fresh trial cache."""
    return _aggregate_trial_locating_features_with_cache(
        search_intervals,
        focus_episodes,
        tracker,
        catalog,
        cache=_TrialSignalCache(tracker),
    )


def _reach_path_ratio(
    tracker: pd.DataFrame,
    hand: str,
    reach_start: float,
    grab_start: float,
    minimum_straight_distance_meters: float,
    warnings: list[str],
    *,
    motion: HandMotionSeries | None = None,
) -> float | None:
    """Calculate one observed 3-D reach path ratio without bridging gaps."""
    motion = motion if motion is not None else prepare_hand_motion(tracker, hand)
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


def _aggregate_trial_reach_features_with_cache(
    reach_intervals: Sequence[ReachInterval],
    tracker: pd.DataFrame,
    minimum_straight_distance_meters: float = 0.02,
    *,
    cache: _TrialSignalCache,
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
            motion=cache.hand_motion(interval.hand),
        )
        if ratio is not None:
            path_ratios.append(ratio)

    return ReachFeatureAggregation(
        median_reach_duration_seconds=(float(np.median(durations)) if durations else None),
        median_reach_path_ratio=(float(np.median(path_ratios)) if path_ratios else None),
        warnings=tuple(warnings),
    )


def aggregate_trial_reach_features(
    reach_intervals: Sequence[ReachInterval],
    tracker: pd.DataFrame,
    minimum_straight_distance_meters: float = 0.02,
) -> ReachFeatureAggregation:
    """Aggregate reach features using a fresh trial cache."""
    return _aggregate_trial_reach_features_with_cache(
        reach_intervals,
        tracker,
        minimum_straight_distance_meters,
        cache=_TrialSignalCache(tracker),
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


def _build_product_grab_features_with_cache(
    trial_metadata: Mapping[str, object],
    grab_events: Sequence[ProductGrabEvent],
    search_intervals: Sequence[SearchInterval],
    reach_intervals: Sequence[ReachInterval],
    tracker: pd.DataFrame,
    *,
    cache: _TrialSignalCache,
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
                motion=cache.hand_motion(event.hand),
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


def build_product_grab_features(
    trial_metadata: Mapping[str, object],
    grab_events: Sequence[ProductGrabEvent],
    search_intervals: Sequence[SearchInterval],
    reach_intervals: Sequence[ReachInterval],
    tracker: pd.DataFrame,
) -> ProductGrabOutputResult:
    """Build product-grab features using a fresh trial cache."""
    return _build_product_grab_features_with_cache(
        trial_metadata,
        grab_events,
        search_intervals,
        reach_intervals,
        tracker,
        cache=_TrialSignalCache(tracker),
    )


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


def _plot_value_is_missing(value: object) -> bool:
    """Return whether a scalar plotting value is missing."""
    try:
        missing = pd.isna(value)
    except (TypeError, ValueError):
        return False
    return isinstance(missing, (bool, np.bool_)) and bool(missing)


def _plot_difficulty(value: object) -> int | None:
    """Normalize a plotting difficulty token to one of the accepted levels."""
    if value is None or _plot_value_is_missing(value):
        return None
    text = str(value).strip().casefold()
    if text.startswith("d"):
        text = text[1:].strip()
    try:
        numeric = float(text)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(numeric) or not numeric.is_integer():
        return None
    difficulty = int(numeric)
    return difficulty if difficulty in DESCRIPTIVE_PLOT_DIFFICULTIES else None


def _plot_condition(value: object) -> str | None:
    """Normalize a plotting condition without raising on unknown labels."""
    if value is None or _plot_value_is_missing(value):
        return None
    text = str(value).strip().casefold()
    return {
        "visual": "Visual",
        "auditory": "Auditory",
        "cognitive": "Cognitive",
    }.get(text)


def _plot_participant_key(value: object) -> str | None:
    """Return a stable participant key for same-participant baselines."""
    if value is None or _plot_value_is_missing(value) or not str(value).strip():
        return None
    try:
        return normalize_participant(value)
    except (TypeError, ValueError):
        return str(value).strip().casefold()


def _plot_numeric_column(
    frame: pd.DataFrame,
    column: str,
    warnings: list[str],
) -> pd.Series:
    """Coerce one metric to finite numeric values and report bad entries."""
    if column not in frame.columns:
        add_unique_warning(warnings, f"missing metric column: {column}")
        return pd.Series(np.nan, index=frame.index, dtype=float)

    raw = frame[column]
    numeric = pd.to_numeric(raw, errors="coerce")
    numeric = pd.Series(np.asarray(numeric, dtype=float), index=frame.index)
    finite = np.isfinite(numeric.to_numpy(float))
    present = raw.notna() & raw.astype("string").str.strip().ne("")
    if bool((present & ~pd.Series(finite, index=frame.index)).any()):
        add_unique_warning(warnings, f"invalid numeric values: {column}")
    numeric.loc[~pd.Series(finite, index=frame.index)] = np.nan
    return numeric


def _prepare_descriptive_plot_frame(
    trial_features: pd.DataFrame,
    warnings: list[str],
) -> pd.DataFrame:
    """Copy and normalize identifiers and all metrics used by FE-01.14 plots."""
    if not isinstance(trial_features, pd.DataFrame):
        raise TypeError("trial_features must be a pandas DataFrame")
    frame = trial_features.copy().reset_index(drop=True)
    required_identifiers = (
        "participant_id",
        "session_id",
        "source_tracker_csv_filename",
        "participant_group",
        "condition_name",
        "difficulty_level",
        "trial_order",
        "language",
    )
    for column in required_identifiers:
        if column not in frame.columns:
            add_unique_warning(warnings, f"missing required identifier column: {column}")
            frame[column] = pd.NA
        missing_values = frame[column].isna() | frame[column].astype("string").str.strip().eq("")
        if bool(missing_values.any()):
            add_unique_warning(warnings, f"missing identifier value: {column}")

    normalized_conditions: list[str | None] = []
    for value in frame["condition_name"]:
        condition = _plot_condition(value)
        normalized_conditions.append(condition)
        if condition is None:
            if not (value is None or _plot_value_is_missing(value) or not str(value).strip()):
                add_unique_warning(warnings, f"unknown condition: {str(value).strip()}")
    frame["__plot_condition"] = normalized_conditions

    normalized_difficulties: list[int | None] = []
    for value in frame["difficulty_level"]:
        difficulty = _plot_difficulty(value)
        normalized_difficulties.append(difficulty)
        if difficulty is None:
            if not (value is None or _plot_value_is_missing(value) or not str(value).strip()):
                add_unique_warning(warnings, f"unknown difficulty: {str(value).strip()}")
    frame["__plot_difficulty"] = normalized_difficulties
    frame["__plot_participant"] = frame["participant_id"].map(_plot_participant_key)

    metric_columns = {
        "performance",
        "mental_demand_score_0_to_10",
        "total_error_count",
        "error_change_from_d0",
        *(spec[1] for spec in DESCRIPTIVE_PLOT_SPECS if not spec[1].startswith("__")),
    }
    if "performance_percent" in frame.columns:
        metric_columns.add("performance_percent")
    for column in sorted(metric_columns):
        frame[f"__plot_{column}"] = _plot_numeric_column(frame, column, warnings)
    return frame


def _relative_performance_values(
    frame: pd.DataFrame,
    warnings: list[str],
) -> pd.Series:
    """Compute same-participant/same-condition D0-minus-current changes.

    Full extraction uses ``performance_percent``.  The legacy ``performance``
    fallback keeps earlier synthetic FE-01.14 tables readable without changing
    their values.
    """
    if "__plot_performance_percent" in frame.columns:
        performance = frame["__plot_performance_percent"]
        if performance.notna().sum() == 0:
            performance = frame["__plot_performance"]
    else:
        performance = frame["__plot_performance"]

    baselines: dict[tuple[str, str], list[float]] = {}
    for index in frame.index:
        participant = frame.at[index, "__plot_participant"]
        condition = frame.at[index, "__plot_condition"]
        difficulty = frame.at[index, "__plot_difficulty"]
        value = performance.at[index]
        if participant is None or condition is None or difficulty != 0:
            continue
        if pd.notna(value) and np.isfinite(float(value)):
            baselines.setdefault((participant, condition), []).append(float(value))

    baseline_medians: dict[tuple[str, str], float] = {}
    for key, values in baselines.items():
        if len(values) > 1:
            add_unique_warning(
                warnings,
                f"duplicate D0 performance baseline: {key[0]}/{key[1]}",
            )
        baseline_medians[key] = float(np.median(values))

    changes = pd.Series(np.nan, index=frame.index, dtype=float)
    unavailable_keys: set[tuple[str | None, str | None]] = set()
    for index in frame.index:
        participant = frame.at[index, "__plot_participant"]
        condition = frame.at[index, "__plot_condition"]
        difficulty = frame.at[index, "__plot_difficulty"]
        if difficulty is None or difficulty == 0:
            continue
        key = (participant, condition)
        baseline = baseline_medians.get(key)
        value = performance.at[index]
        if baseline is None or pd.isna(value) or not np.isfinite(float(value)):
            unavailable_keys.add(key)
            continue
        changes.at[index] = baseline - float(value)

    for participant, condition in sorted(
        unavailable_keys, key=lambda item: (str(item[0]), str(item[1]))
    ):
        participant_label = participant if participant is not None else "missing participant_id"
        condition_label = condition if condition is not None else "missing condition_name"
        add_unique_warning(
            warnings,
            f"performance baseline unavailable: {participant_label}/{condition_label}",
        )
    return changes


def _draw_descriptive_metric_plot(
    frame: pd.DataFrame,
    metric_column: str,
    filename: str,
    ylabel: str,
    title: str,
    destination: Path,
    plt,
    *,
    exclude_d0: bool = False,
    zero_line: bool = False,
    y_limits: tuple[float, float] | None = None,
) -> None:
    """Draw one three-panel observation-plus-median descriptive figure."""
    figure, axes = plt.subplots(1, 3, figsize=(13, 4.2), sharex=True)
    axes = np.atleast_1d(axes)
    for axis, condition in zip(axes, DESCRIPTIVE_PLOT_CONDITIONS):
        median_x: list[int] = []
        median_y: list[float] = []
        for position, difficulty in enumerate(DESCRIPTIVE_PLOT_DIFFICULTIES):
            if exclude_d0 and difficulty == 0:
                continue
            mask = (
                (frame["__plot_condition"] == condition)
                & (frame["__plot_difficulty"] == difficulty)
            )
            values = frame.loc[mask, metric_column].dropna().astype(float)
            values = values[np.isfinite(values.to_numpy(float))]
            if values.empty:
                continue
            jitter = np.linspace(-0.08, 0.08, len(values)) if len(values) > 1 else np.array([0.0])
            axis.scatter(
                np.full(len(values), position, dtype=float) + jitter,
                values.to_numpy(float),
                color="#8aa4c8",
                alpha=0.65,
                s=22,
                label="Trial observations" if position == 0 else "_nolegend_",
            )
            median_x.append(position)
            median_y.append(float(np.median(values.to_numpy(float))))
        if median_x:
            axis.plot(
                median_x,
                median_y,
                color="#c44e52",
                marker="o",
                linewidth=2,
                label="Condition/difficulty median",
            )
        else:
            axis.text(0.5, 0.5, "No finite measurements", ha="center", va="center", transform=axis.transAxes)
        if zero_line:
            axis.axhline(0.0, color="#555555", linestyle="--", linewidth=1, label="Zero reference" if condition == DESCRIPTIVE_PLOT_CONDITIONS[0] else "_nolegend_")
        axis.set_title(condition)
        axis.set_xticks(range(len(DESCRIPTIVE_PLOT_DIFFICULTIES)))
        axis.set_xticklabels(DESCRIPTIVE_PLOT_DIFFICULTY_LABELS)
        axis.set_xlabel("Difficulty")
        axis.set_ylabel(ylabel)
        if y_limits is not None:
            axis.set_ylim(*y_limits)
        axis.grid(True, axis="y", alpha=0.25)
    figure.suptitle(title)
    handles: list[object] = []
    labels: list[str] = []
    for axis in axes:
        panel_handles, panel_labels = axis.get_legend_handles_labels()
        for handle, label in zip(panel_handles, panel_labels):
            if label not in labels:
                handles.append(handle)
                labels.append(label)
    if handles:
        figure.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.02), ncol=3, frameon=False)
    figure.tight_layout(rect=(0, 0.08, 1, 0.94))
    figure.savefig(destination, format="png", dpi=120)
    plt.close(figure)


def _draw_coverage_plot(
    frame: pd.DataFrame,
    destination: Path,
    plt,
) -> None:
    """Draw valid-measurement coverage by feature, condition and difficulty."""
    figure, axes = plt.subplots(1, 3, figsize=(14, 4.5), sharex=True, sharey=True)
    axes = np.atleast_1d(axes)
    for axis, condition in zip(axes, DESCRIPTIVE_PLOT_CONDITIONS):
        for feature in DESCRIPTIVE_COVERAGE_FEATURES:
            metric_column = f"__plot_{feature}"
            percentages: list[float] = []
            for difficulty in DESCRIPTIVE_PLOT_DIFFICULTIES:
                mask = (
                    (frame["__plot_condition"] == condition)
                    & (frame["__plot_difficulty"] == difficulty)
                )
                denominator = int(mask.sum())
                if denominator == 0 or feature not in frame.columns:
                    percentages.append(np.nan)
                    continue
                values = frame.loc[mask, metric_column]
                finite_count = int(values.notna().sum())
                percentages.append(100.0 * finite_count / denominator)
            if any(np.isfinite(value) for value in percentages):
                axis.plot(
                    range(len(DESCRIPTIVE_PLOT_DIFFICULTIES)),
                    percentages,
                    marker="o",
                    linewidth=1.5,
                    label=feature,
                )
        axis.set_title(condition)
        axis.set_xticks(range(len(DESCRIPTIVE_PLOT_DIFFICULTIES)))
        axis.set_xticklabels(DESCRIPTIVE_PLOT_DIFFICULTY_LABELS)
        axis.set_xlabel("Difficulty")
        axis.set_ylabel("Valid measurements (%)")
        axis.set_ylim(0, 100)
        axis.grid(True, axis="y", alpha=0.25)
    handles: list[object] = []
    labels: list[str] = []
    for axis in axes:
        panel_handles, panel_labels = axis.get_legend_handles_labels()
        for handle, label in zip(panel_handles, panel_labels):
            if label not in labels:
                handles.append(handle)
                labels.append(label)
    if handles:
        figure.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.02), ncol=2, frameon=False)
    figure.suptitle("Valid measurements by feature, condition, and difficulty")
    figure.tight_layout(rect=(0, 0.18, 1, 0.94))
    figure.savefig(destination, format="png", dpi=120)
    plt.close(figure)


def write_descriptive_feature_plots(
    trial_features: pd.DataFrame,
    output_dir: Path,
) -> DescriptivePlotResult:
    """Write the thirteen FE-01.14 descriptive feature and outcome plots."""
    warnings: list[str] = []
    frame = _prepare_descriptive_plot_frame(trial_features, warnings)
    frame["__plot_relative_performance_change"] = _relative_performance_values(frame, warnings)

    try:
        import matplotlib

        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover - environment dependency
        raise RuntimeError("FE-01.14 requires Matplotlib") from exc

    destination_root = Path(output_dir) / "plots"
    destination_root.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for filename, metric, ylabel, title in DESCRIPTIVE_PLOT_SPECS:
        path = destination_root / filename
        _draw_descriptive_metric_plot(
            frame,
            f"__plot_{metric[2:]}" if metric.startswith("__") else f"__plot_{metric}",
            filename,
            ylabel,
            title,
            path,
            plt,
            exclude_d0=metric in {"error_change_from_d0", "__relative_performance_change"},
            zero_line=metric in {"error_change_from_d0", "__relative_performance_change"},
            y_limits=(0.0, 10.0) if metric == "mental_demand_score_0_to_10" else None,
        )
        paths.append(path)

    coverage_path = destination_root / "13_valid_measurements_by_difficulty.png"
    _draw_coverage_plot(frame, coverage_path, plt)
    paths.append(coverage_path)
    return DescriptivePlotResult(paths=tuple(paths), warnings=tuple(warnings))


def _detect_search_intervals_with_cache(
    tracker: pd.DataFrame,
    grab_events: Sequence[ProductGrabEvent],
    list_visits: Sequence[ListVisit],
    focus_episodes: Sequence[FocusEpisode],
    *,
    cache: _TrialSignalCache,
) -> SearchIntervalResult:
    """Build shared search records for qualifying first-time-on-list grabs."""
    if "time" not in tracker:
        raise ValueError("Tracker table is missing required column: time")

    warnings: list[str] = []
    times = cache.time_seconds()
    usable_activity_start: float | None = None
    blackout = cache.blackout_values()
    if blackout is not None:
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
    ordered_focus = cache.sorted_focus(focus_episodes)

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

        target_focus_candidates: list[float] = []
        for episode in ordered_focus:
            if episode.focus_start_seconds > grab_start:
                break
            if (
                episode.canonical_focus_name == event.canonical_product_name
                and episode.focus_end_seconds >= search_start
            ):
                target_focus_candidates.append(
                    max(search_start, episode.focus_start_seconds)
                )
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


def detect_search_intervals(
    tracker: pd.DataFrame,
    grab_events: Sequence[ProductGrabEvent],
    list_visits: Sequence[ListVisit],
    focus_episodes: Sequence[FocusEpisode],
) -> SearchIntervalResult:
    """Build search intervals using a fresh trial cache."""
    return _detect_search_intervals_with_cache(
        tracker,
        grab_events,
        list_visits,
        focus_episodes,
        cache=_TrialSignalCache(tracker),
    )


def _finite_numeric_or_none(value: object) -> float | None:
    """Return a finite numeric value or ``None`` without fabricating data."""
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    return numeric if np.isfinite(numeric) else None


def _read_mental_demand_values(
    path: Path,
) -> tuple[dict[tuple[str, str | None, str, str, int | float], list[float]], tuple[str, ...], bool]:
    """Read and normalize the optional per-trial mental-demand workbook."""
    warnings: list[str] = []
    path = Path(path)
    if not path.is_file():
        return {}, ("mental_demand_source_unavailable",), False
    try:
        source = pd.read_excel(path)
    except (OSError, ValueError, ImportError) as error:
        return {}, (f"mental_demand_read_error:{type(error).__name__}",), False

    rename = {
        column: MENTAL_DEMAND_COLUMN_ALIASES[str(column).strip().casefold()]
        for column in source.columns
        if str(column).strip().casefold() in MENTAL_DEMAND_COLUMN_ALIASES
    }
    table = source.rename(columns=rename).copy()
    required = {"participant", "condition", "trial_order", "difficulty", "mental_demand"}
    missing = sorted(required - set(table.columns))
    if missing:
        add_unique_warning(
            warnings, "mental_demand_columns_missing:" + ",".join(missing)
        )
        return {}, tuple(warnings), False

    has_group = "group" in table.columns
    values: dict[tuple[str, str | None, str, str, int | float], list[float]] = {}
    for row_number, row in table.iterrows():
        try:
            participant = normalize_participant(row["participant"])
            condition = normalize_condition(row["condition"])
            trial_order = normalize_trial_order(row["trial_order"])
            difficulty = normalize_difficulty(row["difficulty"])
        except (TypeError, ValueError):
            add_unique_warning(warnings, f"mental_demand_invalid_identifier_row:{row_number + 2}")
            continue
        group = _normalise_group(row["group"]) if has_group else None
        score = _finite_numeric_or_none(row["mental_demand"])
        if score is None:
            add_unique_warning(warnings, f"mental_demand_invalid_value_row:{row_number + 2}")
            continue
        if score < 0 or score > 10:
            add_unique_warning(warnings, f"mental_demand_out_of_range_row:{row_number + 2}")
            continue
        key = (participant, group, condition, trial_order, difficulty)
        values.setdefault(key, []).append(score)
    return values, tuple(warnings), True


def _mental_demand_for_trial(
    row: Mapping[str, object],
    values: Mapping[tuple[str, str | None, str, str, int | float], Sequence[float]],
    source_available: bool,
) -> tuple[float | None, tuple[str, ...]]:
    """Return one exact mental-demand match and trial-scoped warnings."""
    if not source_available:
        return None, ("mental_demand_source_unavailable",)
    key = (
        str(row["participant_id"]),
        _normalise_group(row.get("participant_group")),
        str(row["condition_name"]),
        str(row["trial_order"]),
        row["difficulty_level"],
    )
    matched = list(values.get(key, ()))
    if not matched:
        return None, (
            "mental_demand_match_missing:"
            f"{key[0]}:{key[2]}:{key[3]}:D{key[4]}",
        )
    if len(matched) > 1:
        return float(np.median(matched)), (
            "duplicate_mental_demand_match:"
            f"{key[0]}:{key[2]}:{key[3]}:D{key[4]}",
        )
    return float(matched[0]), ()


def _error_outcome_key(record: ErrorOutcomeRecord) -> tuple[object, ...]:
    """Build the trial identity used to attach normalized error outcomes."""
    return (
        record.participant_id,
        record.session_id,
        record.condition_name,
        record.difficulty_level,
        record.trial_order,
    )


def _concat_fixed_tables(
    tables: Sequence[pd.DataFrame],
    columns: Sequence[str],
) -> pd.DataFrame:
    """Concatenate per-trial tables while preserving an explicit schema."""
    if not tables:
        return pd.DataFrame(columns=list(columns))
    return pd.concat(tables, ignore_index=True).reindex(columns=list(columns))


def _write_feature_extraction_manifest(
    path: Path,
    performance_path: Path,
    raw_root: Path,
    correct_lists_path: Path,
    output_names: Sequence[str],
    trial_table: pd.DataFrame,
    product_table: pd.DataFrame,
    qc_table: pd.DataFrame,
    warnings: Sequence[str],
    mental_demand_path: Path | None = None,
    mental_demand_available_count: int = 0,
    performance_units: str = "unknown",
) -> None:
    """Write stable provenance for one full extraction run."""
    extractor_bytes = Path(__file__).read_bytes()
    manifest = {
        "schema_version": "feature_extraction_v2",
        "extractor": {
            "file": Path(__file__).name,
            "sha256": hashlib.sha256(extractor_bytes).hexdigest(),
        },
        "inputs": {
            "performance": str(Path(performance_path)),
            "raw_root": str(Path(raw_root)),
            "correct_lists": str(Path(correct_lists_path)),
            "mental_demand": str(Path(mental_demand_path))
            if mental_demand_path is not None
            else None,
        },
        "settings": {
            "grab_minimum_duration_seconds": GrabDetectionConfig().minimum_duration_seconds,
            "grab_merge_gap_seconds": GrabDetectionConfig().merge_gap_seconds,
            "reach_absolute_speed_threshold_meters_per_second": ReachDetectionConfig().absolute_speed_threshold_meters_per_second,
            "reach_peak_speed_fraction": ReachDetectionConfig().peak_speed_fraction,
            "reach_minimum_movement_duration_seconds": ReachDetectionConfig().minimum_movement_duration_seconds,
            "minimum_straight_distance_meters": 0.02,
            "performance_units": performance_units,
        },
        "outputs": list(output_names),
        "counts": {
            "trial_rows": int(len(trial_table)),
            "product_grab_rows": int(len(product_table)),
            "qc_rows": int(len(qc_table)),
            "failed_trials": int((qc_table["processing_status"] == "failed").sum())
            if "processing_status" in qc_table
            else 0,
            "repaired_time_headers": int(
                qc_table["malformed_time_header_repaired"].fillna(False).astype(bool).sum()
            )
            if "malformed_time_header_repaired" in qc_table
            else 0,
        },
        "mental_demand_available_count": int(mental_demand_available_count),
        "warnings": list(warnings),
    }
    Path(path).write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def run_full_extraction(
    performance_path: Path,
    raw_root: Path,
    correct_lists_path: Path,
    output_dir: Path,
    limit: int | None = None,
    mental_demand_path: Path | None = None,
) -> FullExtractionResult:
    """Run FE-01.1--FE-01.14 for every normalized trial and write outputs.

    Each trial is isolated: a missing or unusable tracker produces a failed
    QC row and an otherwise complete trial row with unavailable behavioural
    measurements.  Raw input files are read only; the one known ``digits+time``
    header typo is repaired in the in-memory tracker table by
    :func:`load_tracker_table`.
    """
    if limit is not None and limit < 1:
        raise ValueError("--limit must be at least 1")

    performance_path = Path(performance_path)
    raw_root = Path(raw_root)
    correct_lists_path = Path(correct_lists_path)
    output_dir = Path(output_dir)

    trials = standardize_trial_table(read_table(performance_path))
    source_has_explicit_percentage = any(
        str(column).strip().casefold() in PERFORMANCE_PERCENT_ALIASES
        for column in trials.columns
    )
    if limit is not None:
        trials = trials.head(limit).copy()
    catalog = parse_correct_lists(correct_lists_path)
    trials, performance_warnings = _derive_performance_metrics(trials)
    error_result = derive_error_outcomes(trials)

    resolved_mental_demand_path = (
        Path(mental_demand_path)
        if mental_demand_path is not None
        else correct_lists_path.parent / "mental_demand.xlsx"
    )
    mental_demand_values, mental_demand_warnings, mental_demand_source_available = (
        _read_mental_demand_values(resolved_mental_demand_path)
    )

    run_warnings: list[str] = []
    for warning in performance_warnings:
        add_unique_warning(run_warnings, warning)
    for warning in error_result.warnings:
        add_unique_warning(run_warnings, warning)
    for warning in mental_demand_warnings:
        add_unique_warning(run_warnings, warning)

    error_by_key: dict[tuple[object, ...], ErrorOutcomeRecord] = {}
    for record in error_result.records:
        key = _error_outcome_key(record)
        if key in error_by_key:
            add_unique_warning(run_warnings, f"duplicate_error_outcome:{key}")
        else:
            error_by_key[key] = record

    product_tables: list[pd.DataFrame] = []
    trial_tables: list[pd.DataFrame] = []
    qc_rows: list[dict[str, object]] = []

    for source_row in trials.to_dict(orient="records"):
        metadata = dict(source_row)
        trial_warnings: list[str] = []
        mental_demand_value, mental_demand_trial_warnings = _mental_demand_for_trial(
            source_row,
            mental_demand_values,
            mental_demand_source_available,
        )
        metadata["mental_demand_score_0_to_10"] = mental_demand_value
        for warning in mental_demand_trial_warnings:
            add_unique_warning(trial_warnings, warning)
        paths = build_trial_paths(source_row, raw_root)
        tracker_exists = paths.tracker.is_file()
        details_exists = paths.participant_details.is_file()
        participant_hand, hand_warnings = read_participant_hand(paths.participant_details)
        for warning in hand_warnings:
            add_unique_warning(trial_warnings, warning)

        tracker_loaded = False
        tracker_row_count: int | None = None
        finite_time_row_count: int | None = None
        repaired_time_header = False
        tracker: pd.DataFrame | None = None
        grab_events: tuple[ProductGrabEvent, ...] = ()
        search_intervals: tuple[SearchInterval, ...] = ()
        reach_intervals: tuple[ReachInterval, ...] = ()
        list_visits: tuple[ListVisit, ...] = ()
        pace: PaceAndListRecheckAggregation | None = None
        locating: LocatingFeatureAggregation | None = None
        reach_features: ReachFeatureAggregation | None = None
        product_result: ProductGrabOutputResult | None = None
        validity_counts: dict[str, object] | None = None

        if not tracker_exists:
            add_unique_warning(trial_warnings, "tracker_file_missing")
        else:
            try:
                tracker_result = load_tracker_table(paths.tracker)
                tracker = tracker_result.table
                signal_cache = _TrialSignalCache(tracker)
                tracker_loaded = True
                tracker_row_count = int(len(tracker))
                finite_time_row_count = int(np.isfinite(signal_cache.time_seconds()).sum())
                for warning in tracker_result.warnings:
                    add_unique_warning(trial_warnings, warning)
                    if warning.startswith("repaired_time_header:"):
                        repaired_time_header = True

                grab_result = _detect_product_grab_events_with_cache(
                    tracker,
                    catalog,
                    source_row["difficulty_level"],
                    source_row["language"],
                    cache=signal_cache,
                )
                grab_events = grab_result.events
                for warning in grab_result.warnings:
                    add_unique_warning(trial_warnings, warning)

                focus_result = _detect_focus_episodes_with_cache(
                    tracker,
                    catalog,
                    cache=signal_cache,
                )
                for warning in focus_result.warnings:
                    add_unique_warning(trial_warnings, warning)
                list_visits = detect_list_visits(focus_result.episodes)

                search_result = _detect_search_intervals_with_cache(
                    tracker,
                    grab_events,
                    list_visits,
                    focus_result.episodes,
                    cache=signal_cache,
                )
                search_intervals = search_result.intervals
                for warning in search_result.warnings:
                    add_unique_warning(trial_warnings, warning)

                reach_result = _detect_reach_intervals_with_cache(
                    tracker,
                    grab_events,
                    search_intervals=search_intervals,
                    cache=signal_cache,
                )
                reach_intervals = reach_result.intervals
                for warning in reach_result.warnings:
                    add_unique_warning(trial_warnings, warning)

                pace = aggregate_trial_pace_and_list_rechecks(grab_events, list_visits)
                locating = _aggregate_trial_locating_features_with_cache(
                    search_intervals,
                    focus_result.episodes,
                    tracker,
                    catalog,
                    cache=signal_cache,
                )
                for warning in locating.warnings:
                    add_unique_warning(trial_warnings, warning)
                reach_features = _aggregate_trial_reach_features_with_cache(
                    reach_intervals,
                    tracker,
                    cache=signal_cache,
                )
                for warning in reach_features.warnings:
                    add_unique_warning(trial_warnings, warning)

                product_result = _build_product_grab_features_with_cache(
                    metadata,
                    grab_events,
                    search_intervals,
                    reach_intervals,
                    tracker,
                    cache=signal_cache,
                )
                for warning in product_result.warnings:
                    add_unique_warning(trial_warnings, warning)
                if not product_result.table.empty:
                    for warning_cell in product_result.table["processing_warnings"].fillna(""):
                        for warning in str(warning_cell).split(";"):
                            if warning:
                                add_unique_warning(trial_warnings, warning)
                product_tables.append(product_result.table)

                validity_counts = {
                    "real_product_grab_count": len(grab_events),
                    "first_time_on_list_grab_count": sum(
                        event.is_on_list and event.is_first_time_on_list
                        for event in grab_events
                    ),
                    "list_visit_count": len(list_visits),
                    "valid_search_interval_count": sum(
                        interval.is_valid for interval in search_intervals
                    ),
                    "valid_reach_duration_count": int(
                        product_result.table["reach_duration_seconds"].notna().sum()
                    ),
                    "valid_reach_path_ratio_count": int(
                        product_result.table["reach_path_ratio"].notna().sum()
                    ),
                }
            except (OSError, ValueError, pd.errors.ParserError) as error:
                add_unique_warning(
                    trial_warnings,
                    f"tracker_processing_failed:{type(error).__name__}:{error}",
                )

        error_key = (
            str(source_row["participant_id"]),
            str(source_row["session_id"]),
            str(source_row["condition_name"]),
            source_row["difficulty_level"],
            str(source_row["trial_order"]),
        )
        error_record = error_by_key.get(error_key)
        if error_record is None:
            add_unique_warning(trial_warnings, "error_outcome_unmatched")

        trial_result = build_trial_features(
            metadata,
            pace,
            locating,
            reach_features,
            error_record,
            validity_counts,
        )
        trial_table = trial_result.table.copy()
        for warning in trial_result.warnings:
            add_unique_warning(trial_warnings, warning)
        trial_table.loc[0, "processing_warnings"] = ";".join(trial_warnings)
        trial_tables.append(trial_table)

        missing_measure_count: int | None
        if tracker_loaded:
            missing_measure_count = int(
                sum(pd.isna(trial_table.loc[0, column]) for column in BEHAVIOURAL_MEASURE_COLUMNS)
            )
        else:
            missing_measure_count = None
        status = "failed" if not tracker_loaded else ("warning" if trial_warnings else "ok")
        qc_rows.append(
            {
                **{key: source_row.get(key) for key in REQUIRED_MASTER_COLUMNS + ("language",)},
                "tracker_path": str(paths.tracker),
                "participant_details_path": str(paths.participant_details),
                "tracker_file_exists": tracker_exists,
                "participant_details_file_exists": details_exists,
                "tracker_loaded": tracker_loaded,
                "tracker_row_count": tracker_row_count,
                "finite_time_row_count": finite_time_row_count,
                "participant_hand": participant_hand,
                "malformed_time_header_repaired": repaired_time_header,
                "processing_status": status,
                "real_product_grab_count": (
                    validity_counts.get("real_product_grab_count")
                    if validity_counts is not None
                    else None
                ),
                "first_time_on_list_grab_count": (
                    validity_counts.get("first_time_on_list_grab_count")
                    if validity_counts is not None
                    else None
                ),
                "list_visit_count": (
                    validity_counts.get("list_visit_count")
                    if validity_counts is not None
                    else None
                ),
                "valid_search_interval_count": (
                    validity_counts.get("valid_search_interval_count")
                    if validity_counts is not None
                    else None
                ),
                "valid_reach_duration_count": (
                    validity_counts.get("valid_reach_duration_count")
                    if validity_counts is not None
                    else None
                ),
                "valid_reach_path_ratio_count": (
                    validity_counts.get("valid_reach_path_ratio_count")
                    if validity_counts is not None
                    else None
                ),
                "missing_behavioural_measure_count": missing_measure_count,
                "mental_demand_available": mental_demand_value is not None,
                "warning_count": len(trial_warnings),
                "processing_warnings": ";".join(trial_warnings),
            }
        )
        for warning in trial_warnings:
            add_unique_warning(run_warnings, warning)

    product_table = _concat_fixed_tables(product_tables, PRODUCT_GRAB_OUTPUT_COLUMNS)
    trial_table = _concat_fixed_tables(trial_tables, TRIAL_FEATURE_COLUMNS)
    qc_table = _concat_fixed_tables([pd.DataFrame(qc_rows)], QC_COLUMNS)

    output_dir.mkdir(parents=True, exist_ok=True)
    product_path = output_dir / "product_grab_features_v2.csv"
    trial_path = output_dir / "trial_features_v2.csv"
    qc_path = output_dir / "feature_extraction_qc.csv"
    manifest_path = output_dir / "feature_extraction_manifest.json"
    write_product_grab_features(product_table, product_path)
    write_trial_features(trial_table, trial_path)
    qc_table.to_csv(qc_path, index=False, encoding="utf-8", na_rep="")

    plot_result = write_descriptive_feature_plots(trial_table, output_dir)
    for warning in plot_result.warnings:
        add_unique_warning(run_warnings, warning)

    output_names = [
        product_path.name,
        trial_path.name,
        qc_path.name,
        manifest_path.name,
        *(path.relative_to(output_dir).as_posix() for path in plot_result.paths),
    ]
    _write_feature_extraction_manifest(
        manifest_path,
        performance_path,
        raw_root,
        correct_lists_path,
        output_names,
        trial_table,
        product_table,
        qc_table,
        run_warnings,
        mental_demand_path=resolved_mental_demand_path,
        mental_demand_available_count=int(
            trial_table["mental_demand_score_0_to_10"].notna().sum()
        ),
        performance_units=(
            "explicit_percentage"
            if source_has_explicit_percentage
            else "correct_product_count"
        ),
    )
    output_paths = (product_path, trial_path, qc_path, manifest_path, *plot_result.paths)
    return FullExtractionResult(
        product_grab_table=product_table,
        trial_feature_table=trial_table,
        qc_table=qc_table,
        output_paths=tuple(output_paths),
        warnings=tuple(run_warnings),
    )


def build_argument_parser() -> argparse.ArgumentParser:
    """Create the read-only FE-01.1 command-line interface."""
    parser = argparse.ArgumentParser(
        description="Validate and normalize inputs for selected-feature extraction."
    )
    parser.add_argument("--performance", type=Path, required=True)
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--correct-lists", type=Path, required=True)
    parser.add_argument(
        "--mental-demand",
        type=Path,
        default=None,
        help="Optional mental-demand workbook; defaults beside CorrectLists.txt.",
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--run-full-extraction",
        action="store_true",
        help="Run FE-01.1--FE-01.14 and write the agreed output files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs") / "feature_extraction_v2",
        help="Output directory used with --run-full-extraction.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Load requested trials and print a summary without writing outputs."""
    arguments = build_argument_parser().parse_args(argv)
    if arguments.limit is not None and arguments.limit < 1:
        raise ValueError("--limit must be at least 1")

    if arguments.run_full_extraction:
        result = run_full_extraction(
            arguments.performance,
            arguments.raw_root,
            arguments.correct_lists,
            arguments.output_dir,
            limit=arguments.limit,
            mental_demand_path=arguments.mental_demand,
        )
        print(f"Loaded trial rows: {len(result.trial_feature_table)}")
        print(f"Product-grab rows: {len(result.product_grab_table)}")
        print(f"QC rows: {len(result.qc_table)}")
        print(f"Output files written: {len(result.output_paths)}")
        print(f"Run warnings: {len(result.warnings)}")
        return 0

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
