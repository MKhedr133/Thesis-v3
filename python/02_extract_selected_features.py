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
