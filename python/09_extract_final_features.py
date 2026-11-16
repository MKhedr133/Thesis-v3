#!/usr/bin/env python3
"""Extract ten descriptive trial features without replacing historical outputs."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

fe01 = importlib.import_module("02_extract_selected_features")

# Keep recorded codes for catalogue lookup and model provenance. Use study
# levels in the public-facing tables and figures.
STUDY_LEVEL = {0: 0, 2: 1, 6: 2, 10: 3}
TRIAL_ID = (
    "participant_id", "session_id", "source_tracker_csv_filename",
    "condition_name", "difficulty_level", "trial_order",
)
FEATURE_COLUMNS = {
    "performance_percent": "correct_products_collected_percent",
    "mental_demand_score_0_to_10": "self_reported_mental_demand",
    "list_recheck_count": "shopping_list_recheck_count",
    "total_list_recheck_duration_seconds": "shopping_list_recheck_total_seconds",
    "median_time_to_target_seconds": "median_product_search_duration_seconds",
    "median_irrelevant_focus_duration_seconds": "median_non_task_focus_seconds",
    "median_head_turning_degrees": "median_head_rotation_during_search_degrees",
    "median_reach_duration_seconds": "median_reach_start_to_grab_seconds",
    "median_reach_path_ratio": "median_hand_travel_to_straight_distance_ratio",
}
PACE = "median_time_between_grabs_seconds"
FEATURE_ORDER = [*list(FEATURE_COLUMNS.values())[:2], PACE, *list(FEATURE_COLUMNS.values())[2:]]


def display_levels(table: pd.DataFrame) -> pd.DataFrame:
    """Expose study levels while retaining the original catalogue code."""
    codes = table["difficulty_level"]
    if not codes.isin(STUDY_LEVEL).all():
        raise ValueError("Unrecognised recorded difficulty code in FE-02 input.")
    table = table.copy()
    table["recorded_difficulty_code"] = codes.astype("Int64")
    table["difficulty_level"] = codes.map(STUDY_LEVEL).astype("Int64")
    return table


def trial_feature_table(trials: pd.DataFrame, grabs: pd.DataFrame) -> pd.DataFrame:
    """Use the trial percentage and calculate pace from every valid grab."""
    if trials.duplicated(list(TRIAL_ID)).any():
        raise ValueError("Duplicate trial keys in FE-02 input.")
    starts = grabs.sort_values("grab_start_seconds").copy()
    starts["gap"] = starts.groupby(list(TRIAL_ID), dropna=False)["grab_start_seconds"].diff()
    medians = starts.groupby(list(TRIAL_ID), dropna=False)["gap"].median().rename(PACE)
    result = trials.loc[:, [*TRIAL_ID, *FEATURE_COLUMNS]].rename(columns=FEATURE_COLUMNS)
    result = result.join(medians, on=list(TRIAL_ID), validate="one_to_one")
    return display_levels(result.loc[:, [*TRIAL_ID, *FEATURE_ORDER]])


def _same_trial(metadata, selector: tuple[str, str, str, int]) -> bool:
    participant, session, filename, _ = selector
    return (
        metadata["participant_id"], metadata["session_id"],
        metadata["source_tracker_csv_filename"],
    ) == (participant, session, filename)


def _make_worked_trace(metadata, cache, visits, grabs, grab_number: int, *, all_grabs: bool = False) -> dict:
    """Save exact extracted event boundaries and their recorded hand samples."""
    if not 1 <= grab_number <= len(grabs):
        raise ValueError("Worked example grab number is outside the valid-grab table.")
    grab = grabs.iloc[grab_number - 1]
    required = ("search_start_seconds", "first_target_focus_seconds", "reach_start_seconds",
                "grab_start_seconds", "reach_path_ratio")
    if not all(pd.notna(grab[key]) and np.isfinite(grab[key]) for key in required):
        raise ValueError("Choose a worked example grab with valid search and reach values.")
    search, focus, reach, end, ratio = (float(grab[key]) for key in required)
    motion = cache.hand_motion(grab["hand"])
    previous = grabs.iloc[:grab_number - 1]
    if all_grabs:
        previous = previous.loc[previous["grab_start_seconds"] < end]
    else:
        previous = previous.loc[previous["is_first_time_on_list"]]
    previous_grab = float(previous.iloc[-1]["grab_start_seconds"]) if len(previous) else None
    last_visit = max(
        (v for v in visits if v.list_visit_end_seconds <= end),
        key=lambda v: v.list_visit_end_seconds,
        default=None,
    )
    # Only show a list visit if it could have contributed to this search start.
    if last_visit is not None and previous_grab is not None:
        if last_visit.list_visit_end_seconds < previous_grab:
            last_visit = None
    left = min(search, reach, previous_grab if previous_grab is not None else search,
               last_visit.list_visit_start_seconds if last_visit is not None else search)
    samples = [s for s in motion.samples if s.timestamp_seconds is not None
               and left <= s.timestamp_seconds <= end]
    path = [s for s in samples if reach <= s.timestamp_seconds <= end]
    if len(path) < 2 or any(not s.is_valid for s in path):
        raise ValueError("Worked example requires a continuously observed reach path.")

    def number(value):
        return float(value) if value is not None and np.isfinite(value) else None

    def relative(value):
        return float(value - left) if value is not None else None

    config = fe01.ReachDetectionConfig()
    speeds = [s.step_speed_meters_per_second for s in samples
              if s.timestamp_seconds >= search and s.is_valid
              and number(s.step_speed_meters_per_second) is not None]
    return {
        "schema_version": "fe02_worked_trace_v1",
        "condition": metadata["condition_name"],
        "difficulty_level": STUDY_LEVEL[metadata["difficulty_level"]],
        "product": str(grab["canonical_product_name"]),
        "hand": str(grab["hand"]),
        "events": {
            "previous_grab_start": relative(previous_grab),
            "list_check_start": relative(last_visit.list_visit_start_seconds) if last_visit else None,
            "list_check_end": relative(last_visit.list_visit_end_seconds) if last_visit else None,
            "search_start": relative(search),
            "first_target_focus": relative(focus),
            "reach_start": relative(reach),
            "grab_start": relative(end),
        },
        "speed_threshold_m_s": (max(config.absolute_speed_threshold_meters_per_second,
                                    config.peak_speed_fraction * max(speeds)) if speeds else None),
        "hand_speed": [[relative(s.timestamp_seconds), number(s.step_speed_meters_per_second)]
                       for s in samples],
        "reach_path_xyz_m": [[float(s.x_meters), float(s.y_meters), float(s.z_meters)]
                             for s in path],
        "reach_path_ratio": ratio,
    }


def run_extraction(
    performance: Path, raw_root: Path, correct_lists: Path, output_dir: Path,
    *, mental_demand: Path | None = None, limit: int | None = None,
    worked_example: tuple[str, str, str, int] | None = None,
    all_grabs: bool = False,
) -> tuple[Path, ...]:
    """Write an isolated descriptive run. Never overwrite an earlier run."""
    output_dir = Path(output_dir)
    if output_dir.exists():
        raise FileExistsError(f"Choose a new descriptive output directory: {output_dir}")
    trace = None

    missing_focus_keys = []
    list_visit_rows = []

    def collect(metadata, cache, episodes, visits, grabs):
        nonlocal trace
        for visit in visits:
            list_visit_rows.append({**{key: metadata[key] for key in TRIAL_ID}, **visit._asdict()})
        if not episodes:
            missing_focus_keys.append(tuple(metadata[key] for key in TRIAL_ID))
        if worked_example is not None and _same_trial(metadata, worked_example):
            if trace is not None:
                raise ValueError("Worked example selector matched more than one trial.")
            trace = _make_worked_trace(
                metadata, cache, visits, grabs, worked_example[3], all_grabs=all_grabs)

    result = fe01.run_full_extraction(
        performance, raw_root, correct_lists, None, limit=limit,
        mental_demand_path=mental_demand, all_grabs=all_grabs, on_extracted_trial=collect,
    )
    if worked_example is not None and trace is None:
        raise ValueError("No processed trial matched the worked example selector.")

    trials = result.trial_feature_table.copy()
    qc = result.qc_table.copy()
    qc["list_focus_usable"] = pd.array(
        [True if loaded else pd.NA for loaded in qc["tracker_loaded"]], dtype="boolean"
    )
    if missing_focus_keys:
        unresolved = pd.MultiIndex.from_tuples(missing_focus_keys, names=TRIAL_ID)
        unavailable = pd.MultiIndex.from_frame(trials.loc[:, list(TRIAL_ID)]).isin(unresolved)
        trials.loc[unavailable, ["list_recheck_count", "total_list_recheck_duration_seconds"]] = np.nan
        qc_unavailable = pd.MultiIndex.from_frame(qc.loc[:, list(TRIAL_ID)]).isin(unresolved)
        qc.loc[qc_unavailable, "list_focus_usable"] = False
        qc.loc[qc_unavailable, "list_visit_count"] = pd.NA

    features = trial_feature_table(trials, result.product_grab_table)
    grabs = display_levels(result.product_grab_table)
    qc = display_levels(qc)
    qc["missing_behavioural_measure_count"] = (
        features[FEATURE_ORDER[2:]].isna().sum(axis=1).where(qc["tracker_loaded"])
    )
    list_visits = display_levels(pd.DataFrame(
        list_visit_rows, columns=[*TRIAL_ID, *fe01.ListVisit._fields]
    ))
    outputs = {
        "trial_features.csv": features,
        "product_grab_events.csv": grabs,
        "shopping_list_visits.csv": list_visits,
        "feature_extraction_qc.csv": qc,
    }
    files = [*outputs, "feature_extraction_manifest.json"]
    if trace is not None:
        files.append("worked_trace.json")
    manifest = {
        "schema_version": "fe04_all_grabs_v1" if all_grabs else "fe02_v1",
        "settings_status": "historical_settings_not_scientifically_validated",
        "recorded_code_to_study_level": STUDY_LEVEL,
        "grab_settings": fe01.GrabDetectionConfig()._asdict(),
        "reach_settings": fe01.ReachDetectionConfig()._asdict(),
        "list_focus_minimum_seconds": None,
        "minimum_straight_distance_m": 0.02,
        "performance_definition": "official_percentage_or_correct_count_out_of_20",
        "catalogue_uses_recorded_difficulty": True,
        "intended_use": "descriptive_extraction_not_a_replacement_for_frozen_model_inputs",
        "trial_limit": limit,
        "input_sha256": {
            label: hashlib.sha256(Path(path).read_bytes()).hexdigest()
            for label, path in {"performance": performance, "correct_lists": correct_lists,
                                "mental_demand": mental_demand or Path(correct_lists).parent / "mental_demand.xlsx"}.items()
            if Path(path).is_file()
        },
        "code_sha256": {Path(path).name: hashlib.sha256(Path(path).read_bytes()).hexdigest()
                        for path in (__file__, fe01.__file__)},
        "trial_count": len(features),
        "grab_count": len(grabs),
        "trials_without_complete_focus_episodes": len(missing_focus_keys),
        "warnings": list(result.warnings),
        "files": files,
    }
    output_dir.mkdir(parents=True, exist_ok=False)
    for name, table in outputs.items():
        table.to_csv(output_dir / name, index=False, encoding="utf-8", na_rep="")
    if all_grabs:
        manifest.update(
            search_and_reach_grab_scope="all_valid_product_grabs",
            search_boundary="previous_valid_grab_start_or_later_completed_list_visit",
            reach_requires_target_focus=True,
            trial_features_sha256=hashlib.sha256((output_dir / "trial_features.csv").read_bytes()).hexdigest(),
        )
    for name, data in [("worked_trace.json", trace), ("feature_extraction_manifest.json", manifest)]:
        if data is not None:
            (output_dir / name).write_text(
                json.dumps(data, indent=2, allow_nan=False) + "\n", encoding="utf-8"
            )
    return tuple(output_dir / name for name in files)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--performance", type=Path, required=True)
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--correct-lists", type=Path, required=True)
    parser.add_argument("--mental-demand", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--all-grabs", action="store_true",
                        help="Include repeat and off-list grabs in descriptive search and reach measures.")
    parser.add_argument("--worked-example", nargs=4, metavar=("PARTICIPANT", "SESSION", "TRACKER", "GRAB_NUMBER"),
                        help="Optional trace for one trial and a 1-based valid grab number.")
    args = parser.parse_args(argv)
    selector = None
    if args.worked_example:
        participant, session, tracker, number = args.worked_example
        try:
            number = int(number)
        except ValueError:
            parser.error("GRAB_NUMBER must be a positive integer.")
        selector = (participant, session, tracker, number)
    files = run_extraction(args.performance, args.raw_root, args.correct_lists, args.output_dir,
                           mental_demand=args.mental_demand, limit=args.limit, worked_example=selector,
                           all_grabs=args.all_grabs)
    for path in files:
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
