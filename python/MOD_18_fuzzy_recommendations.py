#!/usr/bin/env python3
"""MOD-18: candidate fuzzy rules for prospective difficulty recommendations.

Independent of MOD-16 SHAP and MOD-17 retrospective fuzzy MLR reconstruction.
No patient outcome or treatment benefit is predicted by this module.
"""
from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import MOD_14_direct_linear_regression_evaluation as MOD14

VERSION = "candidate-v0.1"
THRESHOLDS = (0.10, 0.20)
LEVELS = (0, 2, 6, 10)

# Shorthand is used only in the versioned rule registry.
FEATURES = {
    "M": "mental_demand_score_0_to_10",
    "P": "median_time_between_qualifying_grabs_seconds",
    "T": "median_time_to_target_seconds",
    "F": "median_irrelevant_focus_duration_seconds",
    "H": "median_head_turning_degrees",
    "LD": "total_list_recheck_duration_seconds",
    "LC": "list_recheck_count",
    "R": "median_reach_duration_seconds",
    "RP": "median_reach_path_ratio",
    "G": "performance_change_from_d0_percentage_points",
}
LABELS = {
    "M": ("Mental demand", "/10"),
    "P": ("Grab interval", "s"),
    "T": ("Target time", "s"),
    "F": ("Irrelevant focus", "s"),
    "H": ("Head turning", "deg"),
    "LD": ("List recheck time", "s"),
    "LC": ("List recheck count", ""),
    "R": ("Reach time", "s"),
    "RP": ("Reach path ratio", ""),
    "G": ("D0 performance drop", "pp"),
}
BY_FEATURE = {v: k for k, v in FEATURES.items()}

# Each action uses explicitly hypothesised antecedents, never MLR predictions.
# Tuple (feature shortcut, category): L=Low, M=Medium, H=High.
RULES = {
    "visual": {
        "D1": (("M", "H"), ("F", "H")),
        "D2": (("M", "H"), ("P", "H")),
        "D3": (("M", "H"), ("T", "H")),
        "D4": (("F", "H"), ("P", "H")),
        "D5": (("M", "H"), ("H", "H")),
        "I1": (("M", "L"), ("F", "L"), ("T", "L")),
        "I2": (("M", "L"), ("F", "L"), ("P", "L")),
        "I3": (("M", "L"), ("T", "L"), ("P", "L")),
        "I4": (("M", "L"), ("F", "L"), ("H", "L")),
        "I5": (("M", "L"), ("P", "L"), ("LD", "L")),
    },
    "auditory": {
        "D1": (("M", "H"), ("P", "H")),
        "D2": (("M", "H"), ("T", "H")),
        "D3": (("P", "H"), ("T", "H")),
        "D4": (("M", "H"), ("H", "H")),
        "D5": (("P", "H"), ("H", "H")),
        "I1": (("M", "L"), ("P", "L"), ("T", "L")),
        "I2": (("M", "L"), ("P", "L"), ("H", "L")),
        "I3": (("M", "L"), ("T", "L"), ("H", "L")),
        "I4": (("M", "L"), ("T", "L"), ("P", "M")),
        "I5": (("M", "L"), ("P", "L"), ("H", "M")),
    },
    "cognitive_primary": {
        "D1": (("M", "H"), ("P", "H")),
        "D2": (("M", "H"), ("T", "H")),
        "D3": (("P", "H"), ("T", "H")),
        "D4": (("M", "H"), ("LC", "L")),
        "D5": (("M", "H"), ("LD", "L")),
        "I1": (("M", "L"), ("P", "L"), ("T", "L")),
        "I2": (("M", "L"), ("P", "L"), ("F", "L")),
        "I3": (("M", "L"), ("P", "L"), ("H", "L")),
        "I4": (("M", "L"), ("P", "L"), ("LC", "H")),
        "I5": (("M", "L"), ("P", "L"), ("LD", "H")),
    },
    "cognitive_later": {
        "D1": (("G", "H"), ("M", "H")),
        "D2": (("G", "H"), ("P", "H")),
        "D3": (("M", "H"), ("P", "H")),
        "D4": (("G", "H"), ("T", "H")),
        "D5": (("G", "H"), ("LD", "L")),
        "I1": (("G", "L"), ("M", "L"), ("P", "L")),
        "I2": (("G", "L"), ("M", "L"), ("T", "L")),
        "I3": (("G", "L"), ("P", "L"), ("T", "L")),
        "I4": (("G", "L"), ("M", "L"), ("LC", "H")),
        "I5": (("G", "L"), ("M", "L"), ("LD", "H")),
    },
}

# Short, hypothesis-level explanations. Not clinical endorsements.
RULE_REASONS = {
    "visual": {
        "D1": "Demand and irrelevant gaze rise together",
        "D2": "Demand coincides with slow shopping",
        "D3": "Demand coincides with slow target search",
        "D4": "Irrelevant gaze and slower progression coincide",
        "D5": "Demand coincides with extensive scanning",
        "I1": "Low demand, little distraction and quick search",
        "I2": "Low demand, little distraction and fast progress",
        "I3": "Low demand and fast search and progress",
        "I4": "Low demand, little distraction and limited scanning",
        "I5": "Low demand, quick progress and little list checking",
    },
    "auditory": {
        "D1": "Demand coincides with slow shopping",
        "D2": "Demand coincides with slow product search",
        "D3": "Shopping and search times are both prolonged",
        "D4": "Demand coincides with extensive scanning",
        "D5": "Slow shopping accompanies extensive scanning",
        "I1": "Low demand and fast shopping and search",
        "I2": "Low demand, fast shopping and limited scanning",
        "I3": "Low demand, fast search and limited scanning",
        "I4": "Low demand, fast search and typical shopping pace",
        "I5": "Low demand, fast shopping and typical scanning",
    },
    "cognitive_primary": {
        "D1": "Demand and slow shopping coincide",
        "D2": "Demand and slow target search coincide",
        "D3": "Slow shopping and target search coincide",
        "D4": "High demand accompanies limited list checks",
        "D5": "High demand accompanies little list consultation",
        "I1": "Low demand and fast shopping and search",
        "I2": "Low demand, fast progress and little irrelevant gaze",
        "I3": "Low demand, fast progress and limited scanning",
        "I4": "Low demand, fast progress and frequent list checks",
        "I5": "Low demand, fast progress and prolonged list use",
    },
    "cognitive_later": {
        "D1": "Product performance fell with high demand",
        "D2": "Product performance fell with slower shopping",
        "D3": "High demand accompanies slow shopping",
        "D4": "Product performance fell with slower target search",
        "D5": "Product performance fell with little list use",
        "I1": "Lower relative deterioration, demand and grab time",
        "I2": "Lower relative deterioration, demand and search time",
        "I3": "Lower relative deterioration with faster task pace",
        "I4": "Low relative deterioration, demand and frequent list use",
        "I5": "Low relative deterioration, demand and prolonged list use",
    },
}
FEATURE_WARNINGS = {
    "M": "Perceived effort is subjective",
    "P": "Timing includes movement, planning and list use",
    "T": "Search time depends on visual layout",
    "F": "Focus covers selected irrelevant-object labels",
    "H": "Scanning may be appropriate for the layout",
    "LC": "List checking may be successful compensation",
    "LD": "List use may support correct product collection",
    "G": "Stage-relative Low can conceal a large absolute decline",
    "R": "Reach duration is affected by motor factors",
    "RP": "Reach path ratio is affected by movement geometry",
}

EXPLORATORY = {
    "visual": {"D5", "I4", "I5"},
    "auditory": {"D4", "D5", "I2", "I3", "I4", "I5"},
    "cognitive_primary": {"D4", "D5", "I2", "I3", "I4", "I5"},
    "cognitive_later": {"D5", "I3", "I4", "I5"},
}


class RecommendationError(ValueError):
    """Invalid calibration, policy, participant or source record."""


@dataclass(frozen=True)
class Anchors:
    low: float
    median: float
    high: float
    count: int


def validate_rule_registry(registry=None) -> None:
    """Fail on undocumented model features, malformed rules or unequal counts."""
    registry = registry or MOD14.build_model_registry()
    if set(registry) != set(RULES):
        raise RecommendationError("Rule library must match the four frozen models")
    for model, rules in RULES.items():
        allowed = set(registry[model].predictors)
        if set(rules) != {*(f"D{i}" for i in range(1, 6)),
                          *(f"I{i}" for i in range(1, 6))}:
            raise RecommendationError(f"{model}: expected exactly D1-D5 and I1-I5")
        for rule_id, antecedents in rules.items():
            length = 2 if rule_id.startswith("D") else 3
            if len(antecedents) != length:
                raise RecommendationError(f"{model}/{rule_id}: expected {length} inputs")
            if len({feature for feature, _ in antecedents}) != length:
                raise RecommendationError(f"{model}/{rule_id}: duplicate predictor")
            for feature, category in antecedents:
                if FEATURES.get(feature) not in allowed or category not in "LMH":
                    raise RecommendationError(f"{model}/{rule_id}: forbidden antecedent")


def memberships(value: float, a: Anchors) -> dict[str, float]:
    """Shoulder/triangle/shoulder partition with ties treated explicitly."""
    if not math.isfinite(value):
        raise RecommendationError("Non-finite fuzzy input")
    lo, mid, hi = a.low, a.median, a.high
    x = min(max(float(value), lo), hi)
    if lo == hi:
        return {"L": 0., "M": 1., "H": 0.}
    if mid == lo:
        hi_weight = (x - lo) / (hi - lo)
        return {"L": 0., "M": 1. - hi_weight, "H": hi_weight}
    if mid == hi:
        med_weight = (x - lo) / (hi - lo)
        return {"L": 1. - med_weight, "M": med_weight, "H": 0.}
    if x <= mid:
        w = (x - lo) / (mid - lo)
        return {"L": 1. - w, "M": w, "H": 0.}
    w = (x - mid) / (hi - mid)
    return {"L": 0., "M": 1. - w, "H": w}


def calibrate(development: pd.DataFrame, registry=None):
    """Compute a separate development-only reference for each model/stage."""
    registry = registry or MOD14.build_model_registry()
    refs, records = {}, []
    for model, spec in registry.items():
        complete = MOD14.prepare_model_frame(development, model_id=model)
        for stage in spec.allowed_stages:
            cohort = complete.loc[complete["difficulty_stage"].eq(stage)]
            if cohort["participant_id"].nunique() < 2:
                raise RecommendationError(f"{model}/D{LEVELS[stage]}: insufficient reference")
            for feature in spec.predictors:
                values = cohort[feature].to_numpy(dtype=float)
                if not np.isfinite(values).all():
                    raise RecommendationError("Calibration contains non-finite data")
                a = Anchors(float(values.min()), float(np.median(values)),
                            float(values.max()), len(values))
                refs[(model, int(stage), feature)] = a
                records.append({
                    "model": model, "assigned_stage": int(stage),
                    "difficulty_level": LEVELS[stage], "feature": feature,
                    "minimum": a.low, "median": a.median,
                    "maximum": a.high, "development_rows": a.count,
                    "development_participants": cohort["participant_id"].nunique(),
                })
    return refs, pd.DataFrame(records)


def rule_registry_table() -> pd.DataFrame:
    rows = []
    for model, rules in RULES.items():
        for name, antecedents in rules.items():
            rows.append({
                "model": model, "rule_id": name,
                "action": "Decrease" if name[0] == "D" else "Increase",
                "antecedents": " AND ".join(
                    f"{LABELS[f][0]} {dict(L='Low', M='Medium', H='High')[c]}"
                    for f, c in antecedents
                ),
                "feature_keys": ",".join(f for f, _ in antecedents),
                "status": "exploratory" if name in EXPLORATORY[model] else "candidate",
                "rationale": RULE_REASONS[model][name],
                "measurement_cautions": "; ".join(
                    dict.fromkeys(FEATURE_WARNINGS[f] for f, _ in antecedents)
                ),
                "clinically_approved": False,
            })
    return pd.DataFrame(rows)


def record_memberships(model: str, stage: int, record: pd.Series, refs, registry):
    """Return all eligible feature memberships, plus independent audit flags."""
    measured, audit = {}, []
    for feature in registry[model].predictors:
        key = (model, stage, feature)
        a = refs[key]
        value = pd.to_numeric(pd.Series([record.get(feature)]), errors="coerce").iloc[0]
        if not math.isfinite(float(value)):
            measured[feature] = None
            audit.append("MISSING_OR_INVALID:" + feature)
            continue
        outside = bool(value < a.low or value > a.high)
        if outside:
            audit.append("OUTSIDE_CALIBRATION:" + feature)
        measured[feature] = {
            "value": float(value), "reference": a,
            "memberships": memberships(float(value), a),
            "outside": outside,
        }
    return measured, audit


def evaluate_rules(model: str, measured):
    """Min-AND antecedents and max support within each action group."""
    scores = {}
    for name, antecedents in RULES[model].items():
        weights = [measured[FEATURES[feature]] for feature, _ in antecedents]
        if any(detail is None for detail in weights):
            scores[name] = math.nan
        else:
            scores[name] = min(
                detail["memberships"][category]
                for detail, (_, category) in zip(weights, antecedents)
            )
    actions = {}
    for action, prefix in (("Increase", "I"), ("Decrease", "D")):
        active = [v for k, v in scores.items() if k.startswith(prefix)
                  and math.isfinite(v)]
        actions[action] = max(active) if active else math.nan
    return scores, actions


def decide(model: str, stage: int, threshold: float, scores, actions, audit,
           measured, baseline_status: str = "not_applicable"):
    """Preserve the raw decision separately from the safeguarded proposal."""
    inc, dec = actions["Increase"], actions["Decrease"]
    if not (math.isfinite(inc) and math.isfinite(dec)):
        raw = "Review"
    elif inc >= threshold and dec >= threshold:
        raw = "Review"
    elif inc >= threshold:
        raw = "Increase"
    elif dec >= threshold:
        raw = "Decrease"
    else:
        raw = "Maintain"
    flags = list(audit)
    if raw == "Review" and not flags:
        flags.append("CONFLICTING_ACTIONS")
    if raw == "Decrease" and stage == 0:
        flags.append("LOWER_BOUNDARY")
    if raw == "Increase" and stage == 3:
        flags.append("UPPER_BOUNDARY")
    if baseline_status == "unavailable_before_trial":
        flags.append("D0_BASELINE_NOT_YET_OBSERVED")
    if raw == "Maintain" and not flags:
        demand = measured.get(FEATURES["M"])
        if demand is not None and demand["memberships"]["H"] >= threshold:
            flags.append("UNMATCHED_HIGH_MENTAL_DEMAND")
        if model == "cognitive_later":
            loss = measured.get(FEATURES["G"])
            if loss is not None and loss["memberships"]["H"] >= threshold:
                flags.append("UNMATCHED_HIGH_PERFORMANCE_DROP")
    if model == "cognitive_later" and raw == "Increase":
        loss = measured.get(FEATURES["G"])
        if loss is not None and loss["value"] > 0:
            flags.append("INCREASE_WITH_D0_PERFORMANCE_DROP")
    final = "Review" if flags else raw
    current = LEVELS[stage]
    proposed = {
        "Increase": LEVELS[stage + 1] if stage < 3 else None,
        "Decrease": LEVELS[stage - 1] if stage > 0 else None,
        "Maintain": current,
    }.get(final)
    strongest = []
    if raw in {"Increase", "Decrease"}:
        lead = "I" if raw == "Increase" else "D"
        support = actions[raw]
        strongest = [name for name, score in scores.items()
                     if name.startswith(lead) and math.isfinite(score)
                     and math.isclose(score, support, abs_tol=1e-12)]
    return {
        "raw_action": raw, "final_action": final, "proposed_level": proposed,
        "review_flags": "|".join(dict.fromkeys(flags)),
        "strongest_rules": "|".join(strongest),
    }


def baseline_status_for(row, original: pd.DataFrame, model: str) -> str:
    """Detect retrospective Cognitive D0 values unavailable in trial order."""
    if model != "cognitive_later":
        return "not_applicable"
    if "trial_order" not in original.columns:
        return "unknown_trial_order"
    sub = original.loc[
        original["participant_id"].eq(row["participant_id"])
        & original["condition_name"].eq("Cognitive")
        & pd.to_numeric(original["difficulty_level"], errors="coerce").eq(0)
    ]
    if len(sub) != 1:
        return "unavailable_before_trial"
    prev = pd.to_numeric(sub.iloc[0]["trial_order"], errors="coerce")
    current = pd.to_numeric(row.get("trial_order"), errors="coerce")
    if not math.isfinite(float(prev)) or not math.isfinite(float(current)):
        return "unknown_trial_order"
    return "available" if prev < current else "unavailable_before_trial"


def evaluate_model(model: str, source: pd.DataFrame, refs, registry, thresholds=THRESHOLDS):
    """Evaluate stage records without discarding incomplete observations."""
    spec = registry[model]
    needed = {"participant_id", "condition_name", "difficulty_level", "difficulty_stage",
              *spec.predictors}
    if missing := (needed - set(source.columns)):
        raise RecommendationError(f"{model}: missing source columns {sorted(missing)}")
    frame = source.loc[source["condition_name"].eq(spec.condition)].copy()
    frame["difficulty_stage"] = pd.to_numeric(frame["difficulty_stage"], errors="coerce")
    frame["difficulty_level"] = pd.to_numeric(frame["difficulty_level"], errors="coerce")
    if frame[["difficulty_stage", "difficulty_level"]].isna().any().any():
        raise RecommendationError(f"{model}: invalid assigned stage coding")
    for _, row in frame.iterrows():
        level = int(row["difficulty_level"])
        if level not in LEVELS or int(row["difficulty_stage"]) != LEVELS.index(level):
            raise RecommendationError(f"{model}: invalid stage coding")
    frame = frame.loc[frame["difficulty_stage"].isin(spec.allowed_stages)]
    # Every assigned stage should have one record for every participant.
    # Fail loudly rather than silently generating fewer held-out figures.
    expected_people = set(source["participant_id"].astype(str))
    for stage in spec.allowed_stages:
        observed_people = set(frame.loc[
            frame["difficulty_stage"].eq(stage), "participant_id"].astype(str))
        if observed_people != expected_people:
            raise RecommendationError(
                f"{model}/D{LEVELS[stage]}: missing or unexpected participant-stage records")
    if frame.duplicated(["participant_id", "difficulty_level"]).any():
        raise RecommendationError(f"{model}: duplicate participant-stage trial")
    frame = frame.sort_values(["participant_id", "difficulty_level"], kind="stable")
    recs, membership_rows, activation_rows = [], [], []
    for _, row in frame.iterrows():
        stage = int(row["difficulty_stage"])
        measured, audit = record_memberships(model, stage, row, refs, registry)
        baseline = baseline_status_for(row, source, model)
        if baseline == "unknown_trial_order":
            audit.append("UNVERIFIED_D0_BASELINE_ORDER")
        scores, supports = evaluate_rules(model, measured)
        identity = {"model": model, "participant_id": str(row["participant_id"]),
                    "assigned_stage": stage, "difficulty_level": LEVELS[stage],
                    "baseline_status": baseline}
        for feature, detail in measured.items():
            a = refs[(model, stage, feature)]
            membership_rows.append({
                **identity, "feature": feature, "measured_value":
                detail["value"] if detail else math.nan,
                "reference_min": a.low, "reference_median": a.median,
                "reference_max": a.high,
                "low": detail["memberships"]["L"] if detail else math.nan,
                "medium": detail["memberships"]["M"] if detail else math.nan,
                "high": detail["memberships"]["H"] if detail else math.nan,
                "out_of_calibration": detail["outside"] if detail else None,
            })
        for name, score in scores.items():
            activation_rows.append({
                **identity, "rule_id": name,
                "action": "Increase" if name.startswith("I") else "Decrease",
                "score": score, "status": (
                    "exploratory" if name in EXPLORATORY[model] else "candidate"
                ),
            })
        for threshold in thresholds:
            outcome = decide(model, stage, float(threshold), scores, supports,
                             audit, measured, baseline)
            recs.append({
                **identity, "threshold": float(threshold),
                "increase_support": supports["Increase"],
                "decrease_support": supports["Decrease"], **outcome,
                "policy_version": VERSION,
            })
    return (pd.DataFrame(recs), pd.DataFrame(membership_rows),
            pd.DataFrame(activation_rows))


def rule_coverage(activations: pd.DataFrame, thresholds=THRESHOLDS):
    rows = []
    for model, block in activations.groupby("model", sort=True):
        wide = block.pivot(index=["participant_id", "difficulty_level"],
                           columns="rule_id", values="score")
        for threshold in thresholds:
            fired = wide.ge(threshold)
            for rule in RULES[model]:
                same_action = [k for k in RULES[model]
                               if k != rule and k[0] == rule[0]]
                rows.append({
                    "model": model, "threshold": threshold, "rule_id": rule,
                    "activation_count": int(fired[rule].sum()),
                    "unique_action_coverage": int(
                        (fired[rule] & ~fired[same_action].any(axis=1)).sum()
                    ),
                    "observation_count": len(wide),
                })
    return pd.DataFrame(rows)


def _short_reason(model, strongest: str) -> str:
    ids = strongest.split("|")[:2] if strongest else []
    if not ids:
        return "No qualifying change rule"
    return "; ".join(
        " + ".join(LABELS[key][0] + " " +
                   {"L": "Low", "M": "Medium", "H": "High"}[category]
                   for key, category in RULES[model][rid])
        for rid in ids
    )


def plot_trial(row: pd.Series, member: pd.DataFrame, activations: pd.DataFrame,
               target: Path):
    """A compact single-trial figure: short labels, no limitations text block."""
    model = str(row["model"])
    stage = int(row["difficulty_level"])
    person = str(row["participant_id"])
    m = member.set_index("feature")
    act = activations.set_index("rule_id")["score"]
    n = len(m)
    fig = plt.figure(figsize=(12.0, max(7.6, 4.9 + .37*n)), facecolor="white")
    gs = fig.add_gridspec(3, 2, height_ratios=[0.95, 4.6, 1.25],
                          width_ratios=[1.08, 1.0], hspace=.25, wspace=.2,
                          left=.055, right=.975, top=.93, bottom=.065)
    ax0 = fig.add_subplot(gs[0, :])
    ax1 = fig.add_subplot(gs[1, 0])
    ax2 = fig.add_subplot(gs[1, 1])
    ax3 = fig.add_subplot(gs[2, :])
    for ax in (ax0, ax1, ax2, ax3):
        ax.set_axis_off()
    result = str(row["final_action"])
    proposed = row["proposed_level"]
    destination = f"D{int(proposed)}" if pd.notna(proposed) else "—"
    stage_str = f"D{stage}  →  {destination}" if result != "Review" else f"D{stage}  →  REVIEW"
    ax0.text(0, .91, f"{model.replace('_', ' ').title()}  |  {person}  |  D{stage}",
             size=12, color="#495B6C", va="top", transform=ax0.transAxes)
    ax0.text(0, .16, stage_str, size=22, weight="bold",
             color="#17324D", va="bottom", transform=ax0.transAxes)
    ax0.text(.99, .2, f"{result.upper()}  ·  τ {row['threshold']:.2f}",
             ha="right", size=12, color="#3A596F", transform=ax0.transAxes)
    ax1.text(0, 1.01, "MEASURED / STAGE MEDIAN / LOW·MED·HIGH",
             size=9, weight="bold", color="#385269", transform=ax1.transAxes)
    for i, (feature, v) in enumerate(m.iterrows()):
        key = BY_FEATURE[feature]
        name, unit = LABELS[key]
        y = .95 - (i+.45) * (.87 / max(n, 1))
        value = v["measured_value"]
        unit_str = f" {unit}" if unit else ""
        observed = "—" if pd.isna(value) else f"{value:.3g}{unit_str}"
        ref = f"{v['reference_median']:.3g}{unit_str}"
        weights = (
            "—" if pd.isna(v["low"]) else
            f"{v['low']:.2f} · {v['medium']:.2f} · {v['high']:.2f}"
        )
        ax1.text(.0, y, f"{key}  {name}", fontsize=9.4, va="center",
                 transform=ax1.transAxes)
        ax1.text(.52, y, f"{observed} / {ref}", fontsize=9.1,
                 va="center", transform=ax1.transAxes)
        ax1.text(.99, y, weights, fontsize=8.9, ha="right",
                 va="center", transform=ax1.transAxes, color="#486783")
    ax2.text(0, 1.01, "RULE ACTIVATION  ·  MIN-AND",
             size=9, weight="bold", color="#385269", transform=ax2.transAxes)
    y_positions = np.linspace(.94, .12, 10)
    for i, name in enumerate((*[f"D{j}" for j in range(1,6)],
                              *[f"I{j}" for j in range(1,6)])):
        score = float(act[name])
        y = y_positions[i]
        ax2.text(0, y, name, transform=ax2.transAxes, va="center",
                 fontsize=9, weight="bold")
        ax2.text(.085, y, "+".join(f"{k}{cat}" for k,cat in RULES[model][name]),
                 transform=ax2.transAxes, va="center", fontsize=8.6)
        if math.isfinite(score):
            ax2.barh(y, max(0, score) * .32, left=.52, height=.045,
                     transform=ax2.transAxes, color="#557D9A" if name[0]=="D"
                     else "#91A7B5", zorder=2)
            ax2.text(.98, y, f"{score:.2f}", ha="right", va="center",
                     transform=ax2.transAxes, size=9)
    ax3.text(0, .90, "ACTION SUPPORT  ·  MAX-OR", fontsize=9, weight="bold",
             transform=ax3.transAxes, color="#385269")
    inc = float(row["increase_support"])
    dec = float(row["decrease_support"])
    ax3.text(0, .51, f"Increase {inc:.2f}" if math.isfinite(inc) else "Increase —",
             fontsize=11, transform=ax3.transAxes)
    ax3.text(.31, .51, f"Decrease {dec:.2f}" if math.isfinite(dec) else "Decrease —",
             fontsize=11, transform=ax3.transAxes)
    ax3.text(.61, .51, f"Threshold {row['threshold']:.2f}",
             fontsize=10, transform=ax3.transAxes)
    brief = (
        "Review: " + str(row["review_flags"]).split("|")[0]
        if result == "Review"
        else "Why: " + _short_reason(model, row["strongest_rules"])
    )
    ax3.text(0, .19, brief[:115], fontsize=9.6, transform=ax3.transAxes)
    ax3.text(0, -.055, f"MOD-18 {VERSION}  ·  development-stage calibration  ·  "
             "candidate rules; therapist approval required",
             fontsize=8.2, transform=ax3.transAxes, color="#718294")
    target.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(target, dpi=165, bbox_inches="tight", facecolor="white")
    plt.close(fig)



def plot_development_summaries(
    model: str, recommendations: pd.DataFrame,
    activations: pd.DataFrame, output_dir: Path,
    plot_threshold: float,
) -> None:
    """Five aggregate development plots; no participant-level figures."""
    output_dir.mkdir(parents=True, exist_ok=True)
    model_label = model.replace("_", " ").title()
    levels = sorted(recommendations["difficulty_level"].unique())
    actions = ("Increase", "Decrease", "Maintain", "Review")
    colors = {
        "Increase": "#587b98", "Decrease": "#b07b58",
        "Maintain": "#8b9b96", "Review": "#b56b75",
    }

    # Heatmap: the mean rule firing for each assigned development stage.
    order = [f"D{i}" for i in range(1, 6)] + [
        f"I{i}" for i in range(1, 6)]
    heat = activations.pivot_table(
        index="rule_id", columns="difficulty_level",
        values="score", aggfunc="mean",
    ).reindex(index=order, columns=levels)
    fig, ax = plt.subplots(figsize=(7.5, 5))
    image = ax.imshow(
        heat.to_numpy(dtype=float), vmin=0, vmax=1,
        cmap="Blues", aspect="auto",
    )
    ax.set_xticks(range(len(levels)), [f"D{n}" for n in levels])
    ax.set_yticks(range(len(order)), order)
    ax.set_xlabel("Assigned stage")
    ax.set_title(f"{model_label} · Mean rule activation")
    for row in range(len(order)):
        for col in range(len(levels)):
            value = heat.iloc[row, col]
            if pd.notna(value):
                ax.text(
                    col, row, f"{value:.2f}", ha="center", va="center",
                    fontsize=9, color="white" if value > .56 else "#293a4a",
                )
    fig.colorbar(image, ax=ax, label="Mean activation", shrink=.8)
    fig.tight_layout()
    fig.savefig(output_dir / "rule_activation_heatmap.png", dpi=165)
    plt.close(fig)

    # Threshold comparison: final action counts, identical denominator.
    count = pd.crosstab(
        recommendations["threshold"], recommendations["final_action"]
    ).reindex(index=THRESHOLDS, columns=actions, fill_value=0)
    fig, ax = plt.subplots(figsize=(7.4, 4.3))
    bottom = np.zeros(len(count), dtype=float)
    for action in actions:
        values = count[action].to_numpy(dtype=float)
        ax.bar(
            np.arange(len(count)), values, bottom=bottom,
            label=action, color=colors[action], width=.62,
        )
        bottom += values
    ax.set_xticks(range(len(count)), [f"{t:.2f}" for t in count.index])
    ax.set_xlabel("Activation threshold")
    ax.set_ylabel("Development trials")
    ax.set_title(f"{model_label} · Threshold comparison")
    ax.legend(frameon=False, ncol=4, loc="upper center", fontsize=8)
    fig.tight_layout()
    fig.savefig(output_dir / "threshold_comparison.png", dpi=165)
    plt.close(fig)

    selected = recommendations.loc[
        np.isclose(recommendations["threshold"], plot_threshold)
    ].copy()
    stage_counts = pd.crosstab(
        selected["difficulty_level"], selected["final_action"]
    ).reindex(index=levels, columns=actions, fill_value=0)
    fig, ax = plt.subplots(figsize=(7.4, 4.3))
    bottom = np.zeros(len(levels), dtype=float)
    for action in actions:
        values = stage_counts[action].to_numpy(dtype=float)
        ax.bar(
            range(len(levels)), values, bottom=bottom,
            width=.68, label=action, color=colors[action],
        )
        bottom += values
    ax.set_xticks(range(len(levels)), [f"D{n}" for n in levels])
    ax.set_xlabel("Assigned stage")
    ax.set_ylabel("Development trials")
    ax.set_title(f"{model_label} · Recommendations (τ={plot_threshold:.2f})")
    ax.legend(frameon=False, ncol=4, loc="upper center", fontsize=8)
    fig.tight_layout()
    fig.savefig(output_dir / "recommendation_distribution.png", dpi=165)
    plt.close(fig)

    # Review reasons are counted separately; one case may have several flags.
    review = selected.loc[selected["final_action"].eq("Review"), "review_flags"]
    flags = [
        item for value in review.fillna("")
        for item in str(value).split("|") if item
    ]
    counts = pd.Series(flags, dtype="string").value_counts()
    fig, ax = plt.subplots(
        figsize=(8.0, max(3.0, 1.4 + .34 * max(len(counts), 1)))
    )
    if counts.empty:
        ax.text(.5, .5, "No review flags", ha="center", va="center",
                transform=ax.transAxes, fontsize=11)
        ax.set_axis_off()
    else:
        ax.barh(
            np.arange(len(counts)), counts.to_numpy(),
            color=colors["Review"], height=.68,
        )
        ax.set_yticks(range(len(counts)), counts.index.to_list(), fontsize=8)
        ax.invert_yaxis()
        ax.set_xlabel("Occurrences")
    ax.set_title(f"{model_label} · Review flags (τ={plot_threshold:.2f})")
    fig.tight_layout()
    fig.savefig(output_dir / "review_flag_summary.png", dpi=165)
    plt.close(fig)

    # Both supports are rule compatibilities, not confidence probabilities.
    fig, ax = plt.subplots(figsize=(6.3, 5.2))
    for action in actions:
        rows = selected.loc[selected["final_action"].eq(action)]
        ax.scatter(
            rows["increase_support"], rows["decrease_support"],
            s=23, alpha=.75, color=colors[action], label=action,
            edgecolors="none",
        )
    ax.axvline(plot_threshold, color="#56616d", linestyle="--", linewidth=1)
    ax.axhline(plot_threshold, color="#56616d", linestyle="--", linewidth=1)
    ax.set(xlim=(-.015, 1.015), ylim=(-.015, 1.015),
           xlabel="Increase support", ylabel="Decrease support")
    ax.set_title(f"{model_label} · Action support")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(output_dir / "action_support_scatter.png", dpi=165)
    plt.close(fig)


def write_outputs(destination: Path, scope: str, calibration: pd.DataFrame,
                  registry_table: pd.DataFrame, results, render_plots: bool,
                  plot_threshold: float, policy_id: str | None,
                  render_summary_plots: bool = False):
    """Only generated exports; never write data or mutate an existing output."""
    if render_plots and scope != "test":
        raise RecommendationError(
            "Individual recommendation plots are reserved for test scope")
    if render_summary_plots and scope != "development":
        raise RecommendationError(
            "Aggregate research plots are development-only")
    if render_plots and render_summary_plots:
        raise RecommendationError("Choose one plotting mode per run")
    if destination.exists() and any(destination.iterdir()):
        raise RecommendationError(f"Refusing to overwrite: {destination}")
    destination.mkdir(parents=True, exist_ok=True)
    calibration.to_csv(destination / "fuzzy_calibration.csv", index=False)
    registry_table.to_csv(destination / "rule_registry.csv", index=False)
    routed, overview = [], []
    for model, (recs, members, activations) in results.items():
        folder = destination / model
        folder.mkdir()
        for name, frame in (("recommendations", recs), ("memberships", members),
                            ("rule_activations", activations),
                            ("rule_coverage", rule_coverage(activations))):
            frame.to_csv(folder / (name + ".csv"), index=False)
        tally = (recs.groupby(["threshold", "final_action"], dropna=False)
                 .size().rename("count").reset_index())
        tally.insert(0, "model", model)
        tally.to_csv(folder / "model_summary.csv", index=False)
        review_cases = recs.loc[recs["final_action"].eq("Review")].copy()
        review_cases.to_csv(folder / "review_flags.csv", index=False)
        # Compare the two thresholds on the same participant-stage row.
        paired = recs.pivot(
            index=["participant_id", "difficulty_level"],
            columns="threshold", values=["raw_action", "final_action"])
        paired.columns = [
            f"{action}_at_{threshold:.2f}" for action, threshold in paired.columns]
        paired = paired.reset_index()
        paired["raw_changed"] = (
            paired["raw_action_at_0.10"] != paired["raw_action_at_0.20"])
        paired["safeguarded_changed"] = (
            paired["final_action_at_0.10"] != paired["final_action_at_0.20"])
        paired.to_csv(folder / "threshold_comparison.csv", index=False)
        overview.append(tally)
        if render_summary_plots:
            plot_development_summaries(
                model, recs, activations,
                folder / "summary_figures", plot_threshold,
            )
        for _, r in recs.iterrows():
            if (r["model"] != "cognitive_primary"
                    or int(r["difficulty_level"]) == 0):
                routed.append(r.to_dict())
        if render_plots:
            plot_rows = recs.loc[np.isclose(recs["threshold"], plot_threshold)]
            for _, row in plot_rows.iterrows():
                level = int(row["difficulty_level"])
                person = str(row["participant_id"])
                subset_m = members.loc[
                    members["participant_id"].eq(person)
                    & members["difficulty_level"].eq(level)]
                subset_a = activations.loc[
                    activations["participant_id"].eq(person)
                    & activations["difficulty_level"].eq(level)]
                target = folder / "figures" / f"D{level}" / f"participant_{person}_D{level}.png"
                plot_trial(row, subset_m, subset_a, target)
    pd.concat(overview, ignore_index=True).to_csv(
        destination / "cross_model_summary.csv", index=False)
    pd.DataFrame(routed).to_csv(
        destination / "routed_recommendations.csv", index=False)
    with (destination / "validation_report.json").open("w", encoding="utf-8") as f:
        json.dump({
            "scope": scope,
            "models": {
                model: {
                    "unique_trial_count": int(
                        recs[["participant_id", "difficulty_level"]].drop_duplicates().shape[0]),
                    "review_decisions": int(recs["final_action"].eq("Review").sum()),
                    "stages": sorted(map(int, recs["difficulty_level"].unique())),
                }
                for model, (recs, _, _) in results.items()
            },
            "interpretation": "counts of candidate rule behaviour, not clinical accuracy",
        }, f, indent=2)
    manifest = {
        "module": "MOD-18", "rule_version": VERSION, "scope": scope,
        "policy_id": policy_id, "calibration": "development-only, per model-stage",
        "inference": "min AND, max per action, threshold gate",
        "thresholds": list(THRESHOLDS), "plot_threshold": plot_threshold,
        "plot_mode": (
            "individual_test" if render_plots else
            "aggregate_development" if render_summary_plots else "none"
        ),
        "rule_status": "research candidate, not clinically validated",
        "separation": "independent of SHAP and MOD-17 retrospective MLR explanation",
        "prospective_cognitive_rule": "D0 uses primary; later uses same-condition "
                                     "prior D0 when available",
        "heldout_design_rule": "no rule/threshold/plot tuning on test participants",
    }
    (destination / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--modeling-data", type=Path, required=True)
    p.add_argument("--holdout-split", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--scope", choices=("development", "test"), default="development")
    p.add_argument("--enable-locked-test-evaluation", action="store_true")
    p.add_argument("--frozen-policy-id", default=None)
    p.add_argument(
        "--render-plots", action="store_true",
        help="Individual test participant/stage plots (test scope only)",
    )
    p.add_argument(
        "--render-summary-plots", action="store_true",
        help="Five aggregate research plots per model (development scope only)",
    )
    p.add_argument("--plot-threshold", type=float, choices=THRESHOLDS, default=.20)
    return p


def main(argv=None):
    args = parser().parse_args(argv)
    if args.render_plots and args.scope != "test":
        raise RecommendationError(
            "--render-plots requires --scope test")
    if args.render_summary_plots and args.scope != "development":
        raise RecommendationError(
            "--render-summary-plots requires --scope development")
    if args.render_plots and args.render_summary_plots:
        raise RecommendationError(
            "Cannot request both summary and individual plots")
    if args.scope == "test" and (
        not args.enable_locked_test_evaluation or not args.frozen_policy_id
    ):
        raise RecommendationError(
            "Held-out evaluation requires --enable-locked-test-evaluation and "
            "--frozen-policy-id after policy approval and freeze")
    validate_rule_registry()
    raw = pd.read_csv(args.modeling_data)
    split = pd.read_csv(args.holdout_split)
    development, test, _, _ = MOD14.prepare_split_data(
        modeling_data=raw, holdout_split=split)
    refs, calibration = calibrate(development)
    chosen = development if args.scope == "development" else test
    registry = MOD14.build_model_registry()
    results = {}
    for model in registry:
        results[model] = evaluate_model(model, chosen, refs, registry)
    write_outputs(args.output_dir, args.scope, calibration,
                  rule_registry_table(), results, args.render_plots,
                  args.plot_threshold, args.frozen_policy_id,
                  render_summary_plots=args.render_summary_plots)
    print(f"MOD-18 {args.scope} exports: {args.output_dir}")


if __name__ == "__main__":
    main()
