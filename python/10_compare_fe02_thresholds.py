#!/usr/bin/env python3
"""Run development-only FE-02 sensitivity checks without selecting a threshold."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, replace
import hashlib
import importlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

fe02 = importlib.import_module("09_extract_final_features")
fe01 = fe02.fe01

CONDITIONS = ("Visual", "Auditory", "Cognitive")
TRIAL_KEY = list(fe02.TRIAL_ID)
FEATURES = fe02.FEATURE_ORDER[2:]


@dataclass(frozen=True)
class Settings:
    grab_duration_s: float = 0.20
    grab_merge_s: float = 0.10
    list_focus_s: float = 0.0
    target_focus_s: float = 0.0
    reach_floor_m_s: float = 0.05
    reach_peak_fraction: float = 0.05
    reach_sustained_s: float = 0.10
    path_displacement_m: float = 0.02
    use_previous_grab_release: bool = False


BASELINE = Settings()
OPTIONS = {
    "grab_duration_s": (0.10, 0.30),
    "grab_merge_s": (0.06, 0.20),
    "list_focus_s": (0.06, 0.10, 0.20),
    "target_focus_s": (0.06, 0.10, 0.20),
    "reach_floor_m_s": (0.03, 0.08),
    "reach_peak_fraction": (0.025, 0.10),
    "reach_sustained_s": (0.06, 0.20),
    "path_displacement_m": (0.01, 0.03),
    "use_previous_grab_release": (True,),
}
LABELS = {
    "grab_duration_s": "Minimum grab duration (s)",
    "grab_merge_s": "Maximum grab merge gap (s)",
    "list_focus_s": "Minimum list focus duration (s)",
    "target_focus_s": "Minimum target focus duration (s)",
    "reach_floor_m_s": "Absolute reach speed floor (m/s)",
    "reach_peak_fraction": "Fraction of peak hand speed",
    "reach_sustained_s": "Minimum sustained movement (s)",
    "path_displacement_m": "Minimum straight-line displacement (m)",
    "use_previous_grab_release": "Previous qualifying grab boundary",
}
FEATURE_LABELS = dict(zip(FEATURES, (
    "Median time between grabs (s)",
    "Number of shopping list rechecks",
    "Total time rechecking the shopping list (s)",
    "Product search duration (s)",
    "Focus duration on non-task objects (s)",
    "Head rotation during search (°)",
    "Time from reach start to grab (s)",
    "Hand travel relative to straight-line distance",
)))


def development_trials(performance: Path, split_path: Path) -> pd.DataFrame:
    """Validate the existing MOD-09 partition before any tracker is opened."""
    split = pd.read_csv(split_path, dtype=str)
    if not {"participant_id", "split"} <= set(split):
        raise ValueError("MOD-09 split requires participant_id and split columns.")
    if not split["participant_id"].fillna("").str.strip().str.fullmatch(r"[Pp]?[0-9]+").all():
        raise ValueError("MOD-09 split has invalid participant IDs.")
    ids = split["participant_id"].map(fe01.normalize_participant)
    labels = split["split"].str.strip().str.lower()
    if ids.duplicated().any() or labels.value_counts(dropna=False).to_dict() != {"train": 26, "test": 6}:
        raise ValueError("Expected the existing MOD-09 manifest with 26 unique train and 6 unique test IDs.")
    trials = fe01.standardize_trial_table(fe01.read_table(performance))
    if not set(trials["participant_id"]) <= set(ids):
        raise ValueError("Trial table contains a participant absent from the split.")
    dev_ids = set(ids[labels.eq("train")])
    dev = trials.loc[trials["participant_id"].isin(dev_ids)].copy()
    cells = list(zip(dev["participant_id"], dev["condition_name"], dev["difficulty_level"]))
    expected = {(p, c, d) for p in dev_ids for c in CONDITIONS for d in fe02.STUDY_LEVEL}
    if len(cells) != len(expected) or set(cells) != expected:
        raise ValueError("Expected exactly one trial for each development participant, condition and difficulty (312 trials).")
    return dev


def scenarios():
    yield "baseline", "baseline", None, BASELINE, ()
    for setting, values in OPTIONS.items():
        for value in values:
            yield (f"{setting}={value}", setting, value, replace(BASELINE, **{setting: value}), (setting,))
    # A short event can pass duration filtering after merging. Reach speed
    # floor and peak fraction likewise interact via the detector's max rule.
    for first, second in (("grab_duration_s", "grab_merge_s"),
                          ("reach_floor_m_s", "reach_peak_fraction")):
        for x in OPTIONS[first]:
            for y in OPTIONS[second]:
                yield (f"{first}={x},{second}={y}", "interaction", None,
                       replace(BASELINE, **{first: x, second: y}), (first, second))


def median(values):
    return float(np.median(values)) if len(values) else np.nan


def minimum_duration(episodes, threshold):
    return tuple(e for e in episodes if e.focus_duration_seconds >= threshold - 1e-9)


def measure_trial(meta, tracker, catalogue, labelled_reaches=None):
    """Read each development tracker once and reuse unchanged signal stages."""
    cache = fe01._TrialSignalCache(tracker)
    focus = fe01._detect_focus_episodes_with_cache(tracker, catalogue, cache=cache)
    grab_cache, visit_cache, target_cache, search_cache, locating_cache, reach_cache, path_cache = (
        {} for _ in range(7)
    )
    for name, family, value, settings, _ in scenarios():
        g_key = (settings.grab_duration_s, settings.grab_merge_s)
        if g_key not in grab_cache:
            grab_cache[g_key] = fe01._detect_product_grab_events_with_cache(
                tracker, catalogue, meta["difficulty_level"], meta["language"],
                fe01.GrabDetectionConfig(*g_key), cache=cache,
            )
        grab_result = grab_cache[g_key]
        grabs = grab_result.events
        if settings.list_focus_s not in visit_cache:
            visit_cache[settings.list_focus_s] = fe01.detect_list_visits(
                minimum_duration(focus.episodes, settings.list_focus_s)
            )
        visits = visit_cache[settings.list_focus_s]
        if settings.target_focus_s not in target_cache:
            target_cache[settings.target_focus_s] = minimum_duration(
                focus.episodes, settings.target_focus_s
            )
        s_key = (g_key, settings.list_focus_s, settings.target_focus_s,
                 settings.use_previous_grab_release)
        if s_key not in search_cache:
            search_cache[s_key] = fe01._detect_search_intervals_with_cache(
                tracker, grabs, visits, target_cache[settings.target_focus_s], cache=cache,
                use_previous_grab_release=settings.use_previous_grab_release,
            )
            # Target filtering changes search boundaries only, not which
            # non-task focus episodes contribute within those boundaries.
            locating_cache[s_key] = fe01._aggregate_trial_locating_features_with_cache(
                search_cache[s_key].intervals, focus.episodes, tracker, catalogue, cache=cache
            )
        search, locating = search_cache[s_key], locating_cache[s_key]
        reach_params = (settings.reach_floor_m_s, settings.reach_peak_fraction,
                        settings.reach_sustained_s)
        r_key = (s_key, reach_params)
        if r_key not in reach_cache:
            reach_cache[r_key] = fe01._detect_reach_intervals_with_cache(
                tracker, grabs, fe01.ReachDetectionConfig(*reach_params),
                search_intervals=search.intervals, cache=cache,
            )
        reaches = reach_cache[r_key]
        path_key = (r_key, settings.path_displacement_m)
        if path_key not in path_cache:
            warnings = []
            durations, ratios = [], []
            for e in reaches.intervals:
                if e.is_valid:
                    durations.append(e.grab_start_seconds - e.reach_start_seconds)
                    ratio = fe01._reach_path_ratio(
                        tracker, e.hand, e.reach_start_seconds, e.grab_start_seconds,
                        settings.path_displacement_m, warnings, motion=cache.hand_motion(e.hand)
                    )
                    if ratio is not None:
                        ratios.append(ratio)
                else:
                    fe01.add_unique_warning(warnings, "reach_duration_unavailable")
            path_cache[path_key] = (median(durations), median(ratios), len(ratios), warnings)
        duration, ratio, n_ratios, path_warnings = path_cache[path_key]
        annotations = [] if labelled_reaches is None else labelled_reaches
        reach_errors = []
        for product, hand, start, grab, tolerance in annotations:
            matches = [e for e in reaches.intervals
                       if e.canonical_product_name == product and e.hand == hand
                       and abs(e.grab_start_seconds - grab) <= tolerance]
            error = (matches[0].reach_start_seconds - start
                     if len(matches) == 1 and matches[0].is_valid else None)
            reach_errors.append({"product": product, "hand": hand, "reference_grab_s": grab,
                                 "reach_onset_error_s": error})
        rechecks = [v for v in visits if not v.is_initial_view]
        values = (
            median(np.diff([g.grab_start_seconds for g in grabs])),
            len(rechecks) if focus.episodes else np.nan,
            sum(v.list_visit_duration_seconds for v in rechecks) if focus.episodes else np.nan,
            locating.median_time_to_target_seconds, locating.median_irrelevant_focus_duration_seconds,
            locating.median_head_turning_degrees, duration, ratio,
        )
        warnings = (*focus.warnings, *grab_result.warnings, *search.warnings, *reaches.warnings,
                    *locating.warnings, *path_warnings)
        qc = {
            "valid_grabs": len(grabs), "list_visits": len(visits) if focus.episodes else np.nan,
            "qualifying_grabs": len(search.intervals),
            "valid_searches": sum(e.is_valid for e in search.intervals),
            "valid_reaches": sum(e.is_valid for e in reaches.intervals),
            "valid_path_ratios": n_ratios,
            "warning_codes": ";".join(sorted({w.split(":", 1)[0] for w in warnings})),
        }
        yield name, family, value, dict(zip(FEATURES, values)), qc, reach_errors


def read_reference(path: Path, allowed_trials: pd.DataFrame) -> dict:
    """Read independently annotated reach times already aligned to the tracker clock."""
    ref = pd.read_csv(path)
    keys = ["participant_id", "session_id", "source_tracker_csv_filename"]
    numeric = ["reach_start_seconds", "grab_start_seconds", "match_tolerance_seconds"]
    required = {*keys, "product_name", "hand", *numeric}
    if not required <= set(ref) or ref.empty or ref[list(required)].isna().any().any():
        raise ValueError("Reference requires a nonempty, complete tracker-aligned reach annotation CSV.")
    ref["participant_id"] = ref["participant_id"].map(fe01.normalize_participant)
    ref["hand"] = ref["hand"].str.strip().str.lower()
    for field in numeric:
        ref[field] = pd.to_numeric(ref[field], errors="raise")
    if (not np.isfinite(ref[numeric]).all().all()
            or not ref["hand"].isin(["left", "right"]).all()
            or (ref["match_tolerance_seconds"] <= 0).any()
            or (ref["reach_start_seconds"] >= ref["grab_start_seconds"]).any()):
        raise ValueError("Invalid reach annotation hand, timing or matching tolerance.")
    if allowed_trials.duplicated(keys).any():
        raise ValueError("Tracker identity is ambiguous in the development trials.")
    allowed = set(allowed_trials[keys].itertuples(index=False, name=None))
    if not set(ref[keys].itertuples(index=False, name=None)) <= allowed:
        raise ValueError("Reach references must belong to the selected development trials only.")
    return {
        key: list(group[["product_name", "hand", *numeric]].itertuples(index=False, name=None))
        for key, group in ref.groupby(keys, sort=False)
    }


def compare(trial_rows: pd.DataFrame) -> pd.DataFrame:
    """Pair each scenario to the baseline using the original trial identity."""
    keys = [*TRIAL_KEY]
    baseline = trial_rows.loc[trial_rows.scenario.eq("baseline"), [*keys, *FEATURES]]
    rows = []
    for (scenario, condition), group in trial_rows.groupby(["scenario", "condition_name"], sort=False):
        paired = group.merge(baseline, on=keys, suffixes=("", "_baseline"), validate="one_to_one")
        for feature in FEATURES:
            before, after = paired[feature + "_baseline"], paired[feature]
            valid = before.notna() & after.notna()
            delta = (after - before)[valid]
            rows.append({
                "scenario": scenario, "family": group["family"].iloc[0],
                "setting_value": group["setting_value"].iloc[0], "condition": condition,
                "feature": feature, "n_trials": len(group), "baseline_valid": int(before.notna().sum()),
                "setting_valid": int(after.notna().sum()), "paired_n": int(valid.sum()),
                "lost_n": int((before.notna() & after.isna()).sum()),
                "gained_n": int((before.isna() & after.notna()).sum()),
                "median_change": delta.median() if len(delta) else np.nan,
                "q25_change": delta.quantile(.25) if len(delta) else np.nan,
                "q75_change": delta.quantile(.75) if len(delta) else np.nan,
                "median_absolute_change": delta.abs().median() if len(delta) else np.nan,
                "n_changed": int((delta.abs() > 1e-9).sum()),
            })
    return pd.DataFrame(rows)


def summarise_references(reaches: list[dict]) -> pd.DataFrame:
    frame = pd.DataFrame(reaches)
    if frame.empty:
        return frame
    rows = []
    for (scenario, condition), g in frame.groupby(["scenario", "condition_name"], sort=False):
        error = g["reach_onset_error_s"].dropna()
        rows.append({
            "scenario": scenario, "condition": condition, "annotated_n": len(g),
            "matched_n": len(error), "unmatched_or_ambiguous_n": len(g) - len(error),
            "median_signed_error_s": error.median() if len(error) else np.nan,
            "median_absolute_error_s": error.abs().median() if len(error) else np.nan,
        })
    return pd.DataFrame(rows)


def run(performance, raw_root, correct_lists, split, output_dir, reach_reference=None):
    output_dir = Path(output_dir)
    if output_dir.exists():
        raise FileExistsError("Choose a new sensitivity output directory.")
    dev = development_trials(performance, split)
    reference = read_reference(reach_reference, dev) if reach_reference else {}
    catalogue = fe01.parse_correct_lists(correct_lists)
    configs = list(scenarios())
    rows, reference_rows = [], []
    empty_qc = {k: np.nan for k in
                ("valid_grabs", "list_visits", "qualifying_grabs",
                 "valid_searches", "valid_reaches", "valid_path_ratios")}
    for i, meta in enumerate(dev.to_dict("records"), 1):
        tracker_path = fe01.build_trial_paths(meta, raw_root).tracker
        identity = {k: meta[k] for k in TRIAL_KEY}
        lookup = tuple(meta[k] for k in TRIAL_KEY[:3])
        labelled = [(catalogue.canonicalize(p), h, start, grab, tol)
                    for p, h, start, grab, tol in reference.get(lookup, ())]
        for j, (p, h, _, grab, tol) in enumerate(labelled):
            if any((p, h) == (p2, h2) and abs(grab - g2) <= tol+t2
                   for p2, h2, _, g2, t2 in labelled[:j]):
                raise ValueError("Two annotations for one product/hand have overlapping matching windows.")
        try:
            tracker = fe01.load_tracker_table(tracker_path)
        except (OSError, ValueError, pd.errors.ParserError) as err:
            for name, family, value, _, _ in configs:
                rows.append({**identity, "scenario": name, "family": family,
                             "setting_value": value, **dict.fromkeys(FEATURES, np.nan),
                             **empty_qc, "tracker_loaded": False,
                             "warning_codes": type(err).__name__})
                reference_rows.extend({**identity, "scenario": name, "product": p,
                                       "hand": h, "reference_grab_s": g, "reach_onset_error_s": None}
                                      for p, h, _, g, _ in labelled)
        else:
            for name, family, value, features, qc, errors in measure_trial(
                    meta, tracker.table, catalogue, labelled):
                qc["warning_codes"] = ";".join(sorted(
                    set(filter(None, qc["warning_codes"].split(";"))) |
                    {w.split(":", 1)[0] for w in tracker.warnings}
                ))
                rows.append({**identity, "scenario": name, "family": family, "setting_value": value,
                             **features, **qc, "tracker_loaded": True})
                reference_rows.extend({**identity, "scenario": name, **error} for error in errors)
        if i % 26 == 0:
            print(f"Development trials processed: {i}/{len(dev)}", flush=True)
    trial_rows = fe02.display_levels(pd.DataFrame(rows))
    combined = compare(trial_rows)
    reference_summary = summarise_references(reference_rows)
    output_dir.mkdir(parents=True, exist_ok=False)
    trial_rows.to_csv(output_dir / "trial_sensitivity.csv", index=False)
    combined.to_csv(output_dir / "paired_feature_sensitivity.csv", index=False)
    qc_columns = [*empty_qc, "tracker_loaded"]
    qc = trial_rows.groupby(["scenario", "condition_name"], sort=False)[qc_columns].sum(min_count=1)
    qc.columns = [*list(empty_qc), "trackers_loaded"]
    qc.to_csv(output_dir / "event_qc_summary.csv")
    if reference_rows:
        fe02.display_levels(pd.DataFrame(reference_rows)).to_csv(
            output_dir / "reach_reference_matches.csv", index=False
        )
        reference_summary.to_csv(output_dir / "reach_reference_summary.csv", index=False)
    render_plots(combined, reference_summary, output_dir)
    render_qc(qc.reset_index(), configs, output_dir)
    inputs = {"performance": performance, "split": split, "correct_lists": correct_lists}
    if reach_reference:
        inputs["reach_reference"] = reach_reference
    manifest = {
        "participant_partition": "MOD-09 train only",
        "development_participants": 26, "trials": len(dev),
        "reserved_tracker_reads": 0,
        "threshold_selected": False, "detector_validation": "manual onset comparison" if reference_rows else "not performed",
        "note": "List and target options filter complete Unity episodes independently. A threshold of zero reproduces FE-02.",
        "scenarios": {name: asdict(settings) for name, _, _, settings, _ in configs},
        "input_sha256": {label: hashlib.sha256(Path(path).read_bytes()).hexdigest()
                         for label, path in inputs.items()},
        "code_sha256": {Path(path).name: hashlib.sha256(Path(path).read_bytes()).hexdigest()
                       for path in (__file__, fe01.__file__, fe02.__file__)},
    }
    (output_dir / "sensitivity_manifest.json").write_text(json.dumps(manifest, indent=2)+"\n", encoding="utf-8")
    print(f"Sensitivity outputs: {output_dir}")


def render_plots(summary, reference, output_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages

    def setting_label(value):
        return ("Release" if value in ("True", True) else f"{float(value):g}")

    markers = ("o", "s", "^")
    colours = ("#1d668f", "#99471c", "#387255")
    with PdfPages(output_dir / "feature_sensitivity.pdf") as pdf:
        for family in OPTIONS:
            part = summary.loc[summary.family.eq(family)]
            active = part.loc[(part.n_changed > 0) |
                              (part.lost_n > 0) | (part.gained_n > 0), "feature"]
            features = [f for f in FEATURES if f in set(active)]
            # Always put count beside duration for the list-focus comparison.
            if family == "list_focus_s":
                features = [f for f in FEATURES if f in {*features, *FEATURES[1:3]}]
            if not features:
                features = [FEATURES[3] if family == "use_previous_grab_release" else FEATURES[-2]
                            if family.startswith("reach") else FEATURES[-1]
                            if family == "path_displacement_m" else FEATURES[0]]
            for i in range(0, len(features), 4):
                shown = features[i:i+4]
                fig, axes = plt.subplots(len(shown), 2, figsize=(8.2, 2.15*len(shown)+1.05),
                                         squeeze=False, layout="constrained")
                fig.suptitle(f"{LABELS[family]}\nChange from FE-02 baseline · 26 development participants", fontsize=12)
                for row, feature in enumerate(shown):
                    f = part.loc[part.feature.eq(feature)]
                    left, right = axes[row]
                    for j, condition in enumerate(CONDITIONS):
                        g = f.loc[f.condition.eq(condition)]
                        x = np.arange(len(g)) + (j-1)*.17
                        left.vlines(x, g.q25_change, g.q75_change, color=colours[j], linewidth=1.2)
                        left.scatter(x, g.median_change, marker=markers[j], color=colours[j], s=20, label=condition)
                        right.scatter(x, g.lost_n, marker=markers[j], color=colours[j], s=20,
                                      label=condition, zorder=3)
                        right.scatter(x, -g.gained_n, marker=markers[j], facecolors="none",
                                      edgecolors=colours[j], s=26, zorder=3)
                    for ax in (left, right):
                        ax.set_xticks(range(len(g)), [setting_label(v) for v in g.setting_value])
                        ax.axhline(0, color="0.6", linewidth=.75, zorder=0)
                        ax.tick_params(labelsize=8)
                    if row == len(shown) - 1:
                        for ax in (left, right):
                            ax.set_xlabel("Candidate setting", fontsize=9)
                    left.set_title(FEATURE_LABELS[feature], fontsize=9, loc="left")
                    left.set_ylabel("Paired change", fontsize=8)
                    right.set_ylabel("Trials lost (+) / gained (−)", fontsize=8)
                handles, labels = axes[0, 0].get_legend_handles_labels()
                fig.legend(handles, labels, ncol=3, loc="outside lower center", frameon=False, fontsize=9)
                pdf.savefig(fig)
                plt.close(fig)
    if not reference.empty:
        for condition in CONDITIONS:
            g = reference.loc[reference.condition.eq(condition)].reset_index(drop=True)
            if g.empty:
                continue
            fig, (left, right) = plt.subplots(1, 2, figsize=(10, .29*len(g)+1.35),
                                             sharey=True, layout="constrained")
            y = np.arange(len(g))
            left.set_yticks(y, [s.replace("_", " ") for s in g.scenario], fontsize=7)
            left.invert_yaxis()
            left.scatter(g.median_absolute_error_s, y, label="Median absolute", s=15)
            left.scatter(g.median_signed_error_s, y, facecolors="none", edgecolors="black",
                         label="Median signed", s=24)
            left.axvline(0, color="0.6", linewidth=.75)
            left.set_xlabel("Reach onset error (s)")
            left.legend(frameon=False, fontsize=8)
            right.barh(y, g.unmatched_or_ambiguous_n, color="#99471c")
            right.set_xlabel("Unmatched or ambiguous references")
            fig.suptitle(f"{condition} · independently annotated reach onsets")
            fig.savefig(output_dir / f"reach_reference_{condition.lower()}.pdf")
            plt.close(fig)



def render_qc(qc, configs, output_dir):
    """Show changes in valid events and in omitted events, by condition."""
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages

    short = {
        "grab_duration_s": "Grab minimum", "grab_merge_s": "Merge gap",
        "list_focus_s": "List focus", "target_focus_s": "Target focus",
        "reach_floor_m_s": "Reach floor", "reach_peak_fraction": "Reach peak fraction",
        "reach_sustained_s": "Reach sustain", "path_displacement_m": "Path displacement",
        "use_previous_grab_release": "Search boundary",
    }
    labels = {
        name: " / ".join(f"{short[k]}: {getattr(settings, k):g}" if k != "use_previous_grab_release"
                         else "Search boundary: release" for k in changed)
        for name, _, _, settings, changed in configs if changed
    }
    variants = [name for name, _, _, _, changed in configs if changed]
    metrics = ("valid_grabs", "list_visits", "valid_searches", "valid_reaches", "valid_path_ratios")
    with PdfPages(output_dir / "event_qc_sensitivity.pdf") as pdf:
        for condition in CONDITIONS:
            g = qc.loc[qc.condition_name.eq(condition)].set_index("scenario")
            raw = g.loc[variants, list(metrics)]
            delta = (raw - g.loc["baseline", list(metrics)].astype(float)).to_numpy(float)
            finite = np.abs(delta[np.isfinite(delta)])
            limit = max(1., float(finite.max())) if finite.size else 1.
            fig, ax = plt.subplots(figsize=(9.5, 7.4), layout="constrained")
            im = ax.imshow(delta, aspect="auto", cmap="RdBu_r", vmin=-limit, vmax=limit)
            ax.set_title(f"{condition} · Change in valid events from baseline", loc="left")
            ax.set_yticks(np.arange(len(variants)), [labels[n] for n in variants], fontsize=8)
            ax.set_xticks(np.arange(len(metrics)),
                          ["Grabs", "List visits", "Searches", "Reaches", "Path ratios"])
            ax.tick_params(length=0)
            for i, values in enumerate(delta):
                for j, value in enumerate(values):
                    label = str(int(value)) if np.isfinite(value) else "NA"
                    ax.text(j, i, label, ha="center", va="center", fontsize=7,
                            color="white" if np.isfinite(value) and abs(value)/limit > .7 else "black")
            fig.colorbar(im, ax=ax, label="Event count difference")
            pdf.savefig(fig)
            plt.close(fig)

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("performance", "raw-root", "correct-lists", "holdout-split", "output-dir"):
        parser.add_argument("--"+name, type=Path, required=True)
    parser.add_argument("--reach-reference", type=Path, help="Optional independently annotated tracker-clock CSV.")
    a = parser.parse_args()
    run(a.performance, a.raw_root, a.correct_lists, a.holdout_split, a.output_dir, a.reach_reference)


if __name__ == "__main__":
    main()
