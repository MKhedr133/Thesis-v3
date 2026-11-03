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

# MOD-18 routing only; the frozen MOD-14 model registry remains unchanged.
MOD18_STAGES = {
    "visual": (0, 1, 2, 3),
    "auditory": (0, 1, 2, 3),
    "cognitive_primary": (0,),
    "cognitive_later": (1, 2, 3),
}


def eligible_stages(model: str, registry) -> tuple[int, ...]:
    """Prospective-stage eligibility, independent of the frozen MLR workflow."""
    stages = MOD18_STAGES[model]
    if not set(stages).issubset(registry[model].allowed_stages):
        raise RecommendationError(
            f"{model}: MOD-18 stages must belong to the frozen model registry"
        )
    return stages


def rule_label(rule_id: str) -> str:
    """Display label without changing the stable rule identifier."""
    if len(rule_id) != 2 or rule_id[0] not in "DI" or rule_id[1] not in "12345":
        raise RecommendationError(f"Unknown fuzzy rule identifier: {rule_id}")
    action = "Decrease" if rule_id.startswith("D") else "Increase"
    return f"{action} rule {rule_id[1]} ({rule_id})"


def stage_label(level: int) -> str:
    """Presentation-only label; underlying difficulty codes stay unchanged."""
    if level not in LEVELS:
        raise RecommendationError(f"Unrecognized difficulty level: {level}")
    return f"Stage {LEVELS.index(level)}"


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
        for stage in eligible_stages(model, registry):
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
                "rule_name": rule_label(name),
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
    stages = eligible_stages(model, registry)
    frame = frame.loc[frame["difficulty_stage"].isin(stages)]
    # Every assigned stage should have one record for every participant.
    # Fail loudly rather than silently generating fewer held-out figures.
    expected_people = set(source["participant_id"].astype(str))
    for stage in stages:
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
                "rule_name": rule_label(name),
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


def readable_rule(model: str, rule_id: str) -> str:
    """Expand a versioned rule into its complete human-readable IF–THEN form."""
    if rule_id not in RULES[model]:
        raise RecommendationError(f"Unrecognized rule: {model}/{rule_id}")
    antecedents = [
        f"{LABELS[feature][0].replace('D0', 'Stage 0')} is "
        f"{dict(L='Low', M='Medium', H='High')[membership]}"
        for feature, membership in RULES[model][rule_id]
    ]
    action = "Decrease" if rule_id.startswith("D") else "Increase"
    return "IF " + " AND ".join(antecedents) + f"\nTHEN {action} difficulty"


def _short_reason(model, strongest: str) -> str:
    """Legacy compact text for downstream consumers of this module."""
    ids = strongest.split("|")[:2] if strongest else []
    return "; ".join(readable_rule(model, rid).replace("\n", " ")
                     for rid in ids) if ids else "No qualifying change rule"


def plot_trial(row: pd.Series, member: pd.DataFrame, activations: pd.DataFrame,
               target: Path):
    """Compact therapist-facing recommendation: stage, action, rule, strengths.

    Maintain has no dedicated 5-rule class: it is the threshold fallback.
    Review is a safeguard, not an actionable recommendation.
    """
    import textwrap

    model = str(row["model"])
    person = str(row["participant_id"])
    stage = int(row["difficulty_level"])
    final = str(row["final_action"])
    raw = str(row["raw_action"])
    threshold = float(row["threshold"])
    proposed = row["proposed_level"]
    act = activations.set_index("rule_id")["score"].reindex(RULES[model])
    color = {"Increase": "#246c70", "Decrease": "#a85b39",
             "Maintain": "#526f91", "Review": "#9a5260"}[final]
    next_stage = f"→ {stage_label(int(proposed))}" if pd.notna(proposed) else ""
    fig = plt.figure(figsize=(11.8, 6.35), facecolor="#fbfcfd")
    title_ax = fig.add_axes([.06, .72, .88, .23])
    rule_ax = fig.add_axes([.06, .41, .88, .27])
    bars_ax = fig.add_axes([.22, .08, .71, .29])
    for ax in (title_ax, rule_ax):
        ax.set_axis_off()

    title_ax.text(0, .95,
                  f"{model.replace('_', ' ').title()}  ·  {person}  ·  τ {threshold:.2f}",
                  fontsize=10.5, color="#5a6b7a", va="top",
                  transform=title_ax.transAxes)
    title_ax.text(0, .48, "ASSIGNED STAGE", fontsize=9, color="#697b8a",
                  transform=title_ax.transAxes)
    title_ax.text(0, .12, stage_label(stage), fontsize=24, weight="bold",
                  color="#183448", transform=title_ax.transAxes)
    title_ax.text(.37, .48,
                  "STATUS" if final == "Review" else "RECOMMENDATION",
                  fontsize=9, color="#697b8a", transform=title_ax.transAxes)
    display_action = (
        "REVIEW REQUIRED" if final == "Review"
        else f"{final.upper()}  {next_stage}"
    )
    title_ax.text(.37, .12, display_action,
                  fontsize=22, weight="bold", color=color,
                  transform=title_ax.transAxes)
    title_ax.plot([0, 1], [.01, .01], transform=title_ax.transAxes,
                  color="#dce4eb", linewidth=1)

    finite = [(rid, float(value)) for rid, value in act.items()
              if pd.notna(value) and math.isfinite(float(value))]
    strongest = str(row.get("strongest_rules", "")).split("|")[0]
    if final in ("Increase", "Decrease"):
        rule_group = final
        selected = strongest if strongest in RULES[model] else ""
    elif final == "Review" and raw in ("Increase", "Decrease"):
        rule_group = raw
        selected = strongest if strongest in RULES[model] else ""
    elif final == "Review" and finite:
        # With competing supports, show the strongest *provisional* IF rule.
        selected = max(finite, key=lambda item: (item[1], item[0]))[0]
        rule_group = "Increase" if selected.startswith("I") else "Decrease"
    else:
        rule_group = None
        selected = ""

    if rule_group:
        prefix = "I" if rule_group == "Increase" else "D"
        group = sorted(
            ((rid, value) for rid, value in finite if rid.startswith(prefix)),
            key=lambda item: (-item[1], item[0]),
        )
        if not selected and group:
            selected = group[0][0]
        rule_ax.text(0, .97,
                     "STRONGEST RULE" if final != "Review"
                     else "PROVISIONAL RULE · REVIEW REQUIRED",
                     fontsize=9, weight="bold", color="#526777",
                     transform=rule_ax.transAxes, va="top")
        if final == "Review":
            flag = str(row.get("review_flags", "")).split("|")[0]
            rule_ax.text(.99, .97, flag.replace("_", " ").replace("D0", "Stage 0")[:36],
                         fontsize=8, color="#9a5260", ha="right",
                         transform=rule_ax.transAxes, va="top")
        if selected:
            strength = float(act[selected])
            rule_ax.text(0, .79,
                         f"{rule_label(selected)}  ·  activation {strength:.2f}",
                         fontsize=12, weight="bold", color=color,
                         transform=rule_ax.transAxes, va="top")
            antecedents, consequent = readable_rule(model, selected).split("\n")
            # Wrap only if necessary; never drop the antecedents.
            antecedent_lines = textwrap.wrap(
                antecedents, width=86, subsequent_indent="   "
            )
            rule_ax.text(0, .60, "\n".join(antecedent_lines),
                         fontsize=11.2, color="#213b4f",
                         transform=rule_ax.transAxes, va="top",
                         linespacing=1.35)
            rule_ax.text(0, .11, consequent, fontsize=11.3, weight="bold",
                         color=color, transform=rule_ax.transAxes, va="bottom")
        heading = (f"OTHER {rule_group.upper()} RULES" if final != "Review"
                   else f"OTHER PROVISIONAL {rule_group.upper()} RULES")
        shown = [(rid, value) for rid, value in group if rid != selected]
    elif final == "Maintain":
        inc = float(row["increase_support"])
        dec = float(row["decrease_support"])
        rule_ax.text(0, .97, "THRESHOLD DECISION", fontsize=9,
                     weight="bold", color="#526777",
                     transform=rule_ax.transAxes, va="top")
        rule_ax.text(
            0, .70,
            f"IF Increase support {inc:.2f} < {threshold:.2f} AND "
            f"Decrease support {dec:.2f} < {threshold:.2f}",
            fontsize=11.2, color="#213b4f", transform=rule_ax.transAxes,
            va="top")
        rule_ax.text(0, .23, "THEN Maintain difficulty",
                     fontsize=11.3, weight="bold", color=color,
                     transform=rule_ax.transAxes, va="top")
        heading = "STRONGEST NON-QUALIFYING RULES"
        shown = sorted(finite, key=lambda item: (-item[1], item[0]))[:5]
    else:
        rule_ax.text(0, .97, "REVIEW REQUIRED", fontsize=9, weight="bold",
                     color=color, transform=rule_ax.transAxes, va="top")
        flag = str(row.get("review_flags", "")).split("|")[0]
        if not flag:
            flag = "Conflicting or unavailable rule evidence"
        rule_ax.text(0, .69, flag.replace("_", " ").replace(":", ": ").replace("D0", "Stage 0"),
                     fontsize=11, color="#213b4f",
                     transform=rule_ax.transAxes, va="top")
        rule_ax.text(0, .27, "No automatic stage recommendation",
                     fontsize=11, weight="bold", color=color,
                     transform=rule_ax.transAxes, va="top")
        heading = "AVAILABLE RULE STRENGTHS"
        shown = sorted(finite, key=lambda item: (-item[1], item[0]))[:5]

    bars_ax.set_title(heading, loc="left", fontsize=9, color="#526777",
                      weight="bold", pad=10)
    if shown:
        rule_ids = [rid for rid, _ in shown]
        vals = [value for _, value in shown]
        y = np.arange(len(shown))
        bar_colors = [color if rid == selected else "#9bafbf" for rid in rule_ids]
        bars_ax.barh(y, vals, color=bar_colors, height=.55)
        bars_ax.set_yticks(y, labels=[rule_label(rid) for rid in rule_ids])
        bars_ax.invert_yaxis()
        bars_ax.set_xlim(0, 1.10)
        bars_ax.set_xticks([0, .5, 1], ["0", "0.5", "1"])
        bars_ax.tick_params(axis="both", length=0, labelsize=9, colors="#596d7b")
        for yi, value in zip(y, vals):
            bars_ax.text(min(value + .025, 1.04), yi, f"{value:.2f}",
                         va="center", fontsize=9, color="#244155")
    else:
        bars_ax.text(.5, .45, "No valid rule activations",
                     ha="center", va="center", transform=bars_ax.transAxes,
                     color="#687986")
        bars_ax.set_xlim(0, 1)
        bars_ax.set_yticks([])
        bars_ax.set_xticks([])
    for edge in ("top", "right", "left"):
        bars_ax.spines[edge].set_visible(False)
    bars_ax.spines["bottom"].set_color("#d7e1e9")
    target.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(target, dpi=170, facecolor=fig.get_facecolor())
    plt.close(fig)


def plot_rule_activation_heatmap(
    model: str, activations: pd.DataFrame, output_file: Path,
) -> None:
    """Show numeric activations per observation, grouped into stage panels.

    Each observation is a column and each rule is a row. Separate stage panels
    keep the values readable instead of compressing up to 104 columns.
    """
    rule_ids = list(RULES[model])
    required = {"participant_id", "difficulty_level", "rule_id", "score"}
    if not required.issubset(activations.columns):
        raise RecommendationError(f"{model}: incomplete heatmap input")
    frame = activations.copy()
    if frame.duplicated(["participant_id", "difficulty_level", "rule_id"]).any():
        raise RecommendationError(f"{model}: duplicate rule scores")
    stages = sorted(frame["difficulty_level"].unique())
    if not stages or any(int(level) not in LEVELS for level in stages):
        raise RecommendationError(f"{model}: invalid heatmap stage")

    fig, axes = plt.subplots(
        len(stages), 1, figsize=(18.0, 4.15 * len(stages) + .7),
        squeeze=False, facecolor="white",
    )
    cmap = plt.get_cmap("Blues").with_extremes(bad="#edf1f4")
    for ax, level in zip(axes[:, 0], stages):
        stage = frame.loc[frame["difficulty_level"].eq(level)]
        people = sorted(stage["participant_id"].astype(str).unique())
        wide = stage.pivot(
            index="rule_id", columns="participant_id", values="score"
        ).reindex(index=rule_ids, columns=people)
        matrix = wide.to_numpy(dtype=float)
        image = ax.imshow(
            matrix, cmap=cmap, vmin=0, vmax=1,
            interpolation="nearest", aspect="auto",
        )
        ax.set_yticks(
            np.arange(len(rule_ids)),
            labels=[rule_label(rid) for rid in rule_ids],
            fontsize=8.5,
        )
        ax.set_xticks([])
        ax.tick_params(length=0)
        ax.set_title(stage_label(int(level)), loc="left",
                     fontsize=11, color="#263e52", pad=9)
        # Matplotlib text is centred on each cell; NaN is a grey dash.
        # Four characters at 7 pt fit inside each of 26 participant columns.
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                value = matrix[i, j]
                ax.text(
                    j, i, f"{value:.2f}" if np.isfinite(value) else "–",
                    ha="center", va="center", fontsize=7.3,
                    color="white" if np.isfinite(value) and value >= .57
                          else "#253d51",
                )
    fig.suptitle(f"{model.replace('_', ' ').title()} · Rule activation",
                 fontsize=15, color="#1c394f", y=.998)
    fig.subplots_adjust(left=.165, right=.92, top=.95,
                        bottom=.035, hspace=.36)
    fig.colorbar(image, ax=list(axes[:, 0]), shrink=.6,
                 label="Activation", pad=.012)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_file, dpi=170, facecolor="white")
    plt.close(fig)



def recommendation_distribution_table(
    development_recs: pd.DataFrame, test_recs: pd.DataFrame,
) -> pd.DataFrame:
    """Three-action summary, with transparently labelled provisional fallback.

    Prefer safeguarded actions. If *all* cases at a threshold are Review,
    display provisional raw actions so the chart is diagnostically useful.
    Provisional counts are not actionable recommendations. Both the basis
    and number of blocked cases are recorded in the exported CSV.
    """
    rows = []
    actions = ("Increase", "Maintain", "Decrease")
    for scope, recs in (("development", development_recs), ("test", test_recs)):
        for threshold in THRESHOLDS:
            block = recs.loc[np.isclose(recs["threshold"], threshold)]
            approved = block.loc[block["final_action"].isin(actions)]
            review_count = int(block["final_action"].eq("Review").sum())
            if approved.empty and not block.empty:
                basis = "provisional"
                eligible = block.loc[block["raw_action"].isin(actions)]
                field = "raw_action"
            else:
                basis = "safeguarded"
                eligible = approved
                field = "final_action"
            denominator = len(eligible)
            counts = eligible[field].value_counts()
            for action in actions:
                count = int(counts.get(action, 0))
                rows.append({
                    "scope": scope, "threshold": threshold, "action": action,
                    "count": count, "denominator": denominator,
                    "percentage": (
                        100 * count / denominator if denominator else math.nan
                    ),
                    "basis": basis,
                    "review_count": review_count,
                    "observation_count": int(len(block)),
                })
    return pd.DataFrame(rows)


def plot_recommendation_distribution(
    model: str, development_recs: pd.DataFrame, test_recs: pd.DataFrame,
    output_file: Path,
) -> None:
    """Two bar panels; provisional fallback is visibly non-actionable."""
    summary = recommendation_distribution_table(development_recs, test_recs)
    actions = ("Increase", "Maintain", "Decrease")
    x = np.arange(3)
    width = .34
    colors = ("#2e5e83", "#90b0c7")
    fig, axes = plt.subplots(
        1, 2, figsize=(12.8, 4.9), facecolor="white",
        gridspec_kw={"wspace": .26}
    )
    for ax, (scope, participants) in zip(
        axes, (("development", 26), ("test", 6))
    ):
        scope_data = summary.loc[summary["scope"].eq(scope)]
        peak = max(1, int(scope_data["count"].max()))
        label_space = max(1.6, peak * .22)
        provisional_notes = []
        for idx, threshold in enumerate(THRESHOLDS):
            subset = scope_data.loc[np.isclose(
                scope_data["threshold"], threshold
            )]
            values = [
                int(subset.loc[subset["action"].eq(action), "count"].iloc[0])
                for action in actions
            ]
            percents = [
                float(subset.loc[subset["action"].eq(action), "percentage"].iloc[0])
                for action in actions
            ]
            basis = str(subset["basis"].iloc[0])
            review = int(subset["review_count"].iloc[0])
            total = int(subset["observation_count"].iloc[0])
            coordinates = x + (idx - .5) * width
            ax.bar(
                coordinates, values, width=width * .92,
                label=f"τ {threshold:.2f}", color=colors[idx],
                alpha=.62 if basis == "provisional" else 1.,
                hatch="//" if basis == "provisional" else None,
            )
            for px, count, percent in zip(coordinates, values, percents):
                annotation = (
                    f"{count} · {percent:.0f}%" if math.isfinite(percent)
                    else f"{count} · n/a"
                )
                ax.text(px, count + label_space * .14, annotation,
                        fontsize=8.8, ha="center", color="#254257")
            if basis == "provisional":
                provisional_notes.append(
                    f"τ {threshold:.2f}: {review}/{total} Review · provisional"
                )
        ax.set_ylim(0, peak + label_space)
        ax.set_xticks(x, actions, fontsize=10)
        ax.set_ylabel("Action counts", fontsize=9)
        ax.set_title(
            f"{scope.title()} · {participants} participants",
            fontsize=11, loc="left", pad=12, color="#254257"
        )
        if provisional_notes:
            # Keep the note outside the plotting area; never disguise
            # unverified raw actions as final recommendations.
            ax.text(
                .5, -.25, "\n".join(provisional_notes),
                ha="center", va="top", transform=ax.transAxes,
                color="#875d3e", fontsize=9,
            )
        ax.spines[["top", "right"]].set_visible(False)
        ax.spines[["left", "bottom"]].set_color("#d7e1e8")
        ax.tick_params(length=0, colors="#526777")
        ax.grid(axis="y", color="#e8edf2", linewidth=.8)
        ax.set_axisbelow(True)
    fig.suptitle(
        f"{model.replace('_', ' ').title()} · Recommendation distribution",
        fontsize=14, color="#1c394f", weight="bold", y=.99
    )
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(.5, .94),
               frameon=False, ncol=2)
    fig.subplots_adjust(left=.08, right=.98, bottom=.26, top=.80, wspace=.25)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_file, dpi=170, facecolor="white")
    plt.close(fig)


def plot_development_summaries(
    model: str, recommendations: pd.DataFrame,
    activations: pd.DataFrame, output_dir: Path,
    plot_threshold: float,
) -> None:
    """Development-only exports: observation heatmap and two-threshold counts.

    The comparison involving held-out participants is *only* produced in the
    locked, combined figure-set mode after the policy has been frozen.
    """
    plot_rule_activation_heatmap(
        model, activations, output_dir / f"{model}_rule_activation_heatmap.png"
    )
    # Retain a development-only threshold comparison for legacy CLI users.
    counts = recommendation_distribution_table(recommendations, recommendations)
    counts = counts.loc[counts["scope"].eq("development")]
    fig, ax = plt.subplots(figsize=(7, 4.6), facecolor="white")
    x = np.arange(3)
    provisional_notes = []
    for i, threshold in enumerate(THRESHOLDS):
        subset = counts.loc[np.isclose(counts["threshold"], threshold)]
        nums = [int(subset.loc[subset["action"].eq(action), "count"].iloc[0])
                for action in ("Increase", "Maintain", "Decrease")]
        basis = str(subset["basis"].iloc[0])
        ax.bar(x + (i - .5) * .34, nums, width=.31,
               color=("#2e5e83", "#90b0c7")[i],
               alpha=.62 if basis == "provisional" else 1.,
               hatch="//" if basis == "provisional" else None,
               label=f"τ {threshold:.2f}")
        if basis == "provisional":
            review = int(subset["review_count"].iloc[0])
            total = int(subset["observation_count"].iloc[0])
            provisional_notes.append(
                f"τ {threshold:.2f}: {review}/{total} Review · provisional"
            )
    ax.set_xticks(x, ["Increase", "Maintain", "Decrease"])
    ax.set_ylabel("Action counts")
    ax.set_title(f"{model.replace('_', ' ').title()} · Development thresholds",
                 loc="left", fontsize=12)
    ax.legend(frameon=False)
    if provisional_notes:
        fig.text(.5, .055, "\n".join(provisional_notes),
                 ha="center", va="center", fontsize=8.5, color="#875d3e")
        fig.subplots_adjust(bottom=.24)
    else:
        fig.tight_layout()
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / f"{model}_threshold_comparison.png",
                dpi=170, facecolor="white")
    plt.close(fig)


def write_figure_set(
    destination: Path,
    development_results: dict,
    test_results: dict,
    plot_threshold: float,
    calibration: pd.DataFrame,
    policy_id: str,
) -> int:
    """Export 80 figures and their numerical results under one output root.

    Scope-specific CSV/JSON exports are kept separate. Neither calibration
    nor the policy is modified by the descriptive held-out visualisations.
    """
    if destination.exists():
        if not destination.is_dir() or any(destination.iterdir()):
            raise RecommendationError(
                f"Refusing to overwrite an existing output: {destination}"
            )
    if plot_threshold not in THRESHOLDS:
        raise RecommendationError("Unknown plot threshold")
    registry = MOD14.build_model_registry()
    if set(development_results) != set(registry) or set(test_results) != set(registry):
        raise RecommendationError("Figure-set models must match the frozen registry")

    # Preflight counts and participant boundaries before creating any files.
    expected_test = {}
    for model, spec in registry.items():
        dev_recs, _, dev_rules = development_results[model]
        test_recs, test_members, test_rules = test_results[model]
        dev_ids = set(dev_recs["participant_id"].astype(str))
        test_ids = set(test_recs["participant_id"].astype(str))
        if dev_ids & test_ids or len(dev_ids) != 26 or len(test_ids) != 6:
            raise RecommendationError(f"{model}: invalid 26/6 participant boundary")
        if not np.all(np.isin(THRESHOLDS, dev_recs["threshold"].unique())):
            raise RecommendationError(f"{model}: missing development thresholds")
        selected = test_recs.loc[
            np.isclose(test_recs["threshold"], plot_threshold)
        ]
        expected_rows = len(eligible_stages(model, registry)) * 6
        if len(selected) != expected_rows or selected.duplicated(
            ["participant_id", "difficulty_level"]
        ).any():
            raise RecommendationError(
                f"{model}: expected {expected_rows} unique held-out trials"
            )
        for stage in eligible_stages(model, registry):
            stage_rows = selected.loc[selected["assigned_stage"].eq(stage)]
            if set(stage_rows["participant_id"].astype(str)) != test_ids:
                raise RecommendationError(
                    f"{model}/D{LEVELS[stage]}: incomplete held-out coverage"
                )
        if not test_members["participant_id"].isin(test_ids).all():
            raise RecommendationError(f"{model}: invalid membership participant")
        if not test_rules["participant_id"].isin(test_ids).all():
            raise RecommendationError(f"{model}: invalid activation participant")
        expected_test[model] = selected

    destination.mkdir(parents=True, exist_ok=True)
    dev_dir = destination / "summary"
    test_dir = destination / "test"
    for model in registry:
        dev_recs, _, dev_activations = development_results[model]
        plot_rule_activation_heatmap(
            model, dev_activations,
            dev_dir / f"{model}_rule_activation_heatmap.png"
        )
        test_recs, members, activations = test_results[model]
        plot_recommendation_distribution(
            model, dev_recs, test_recs,
            dev_dir / f"{model}_recommendation_distribution.png"
        )
        for _, row in expected_test[model].iterrows():
            person = str(row["participant_id"])
            level = int(row["difficulty_level"])
            # IDs are written as filenames, never interpreted as subpaths.
            if not person or any(c in person for c in r'\/:*?"<>|'):
                raise RecommendationError("Invalid participant ID for filename")
            same_member = members.loc[
                members["participant_id"].astype(str).eq(person)
                & members["difficulty_level"].eq(level)
            ]
            same_rules = activations.loc[
                activations["participant_id"].astype(str).eq(person)
                & activations["difficulty_level"].eq(level)
            ]
            if len(same_member) != len(registry[model].predictors) or len(same_rules) != 10:
                raise RecommendationError(
                    f"{model}/{person}/D{level}: incomplete rule explanation"
                )
            target = test_dir / model / f"participant_{person}_D{level}.png"
            plot_trial(row, same_member, same_rules, target)

    actual = list(destination.rglob("*.png"))
    if len(actual) != 80:
        raise RecommendationError(
            f"Figure set incomplete: expected 80 PNGs, got {len(actual)}"
        )

    # Reuse the regular, separately scoped exports. Each contains both
    # thresholds, measured memberships, rule strengths, review flags and
    # source-calibration metadata; no redundant PNGs are generated here.
    data_dir = destination / "data"
    registry_table = rule_registry_table()
    for scope, scoped_results in (
        ("development", development_results), ("test", test_results),
    ):
        write_outputs(
            data_dir / scope, scope, calibration, registry_table,
            scoped_results, render_plots=False,
            plot_threshold=plot_threshold, policy_id=policy_id,
        )

    # Exact counts behind the two-panel distribution figures.
    distribution_tables = []
    for model in registry:
        summary = recommendation_distribution_table(
            development_results[model][0], test_results[model][0]
        )
        summary.insert(0, "model", model)
        distribution_tables.append(summary)
    pd.concat(distribution_tables, ignore_index=True).to_csv(
        data_dir / "recommendation_distribution.csv", index=False
    )
    return len(actual)



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
        help="Two aggregate plots per model (development scope only)",
    )
    p.add_argument(
        "--render-figure-set", action="store_true",
        help="One folder: 80 PNGs plus development/test numerical CSV/JSON results",
    )
    p.add_argument("--plot-threshold", type=float, choices=THRESHOLDS, default=.20)
    return p


def main(argv=None):
    args = parser().parse_args(argv)
    if args.render_figure_set:
        if args.render_plots or args.render_summary_plots:
            raise RecommendationError(
                "--render-figure-set cannot be combined with other plot switches")
        if args.scope != "development":
            raise RecommendationError(
                "--render-figure-set includes both scopes; omit --scope test")
        if not args.enable_locked_test_evaluation or not args.frozen_policy_id:
            raise RecommendationError(
                "Combined figures require --enable-locked-test-evaluation and "
                "--frozen-policy-id after actual policy approval and freeze")
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
    if args.render_figure_set:
        development_results = {
            model: evaluate_model(model, development, refs, registry)
            for model in registry
        }
        test_results = {
            model: evaluate_model(model, test, refs, registry)
            for model in registry
        }
        count = write_figure_set(
            args.output_dir, development_results, test_results,
            args.plot_threshold, calibration, args.frozen_policy_id
        )
        print(f"MOD-18 figures and data: {count} PNGs, development/test CSVs in {args.output_dir}")
        return
    write_outputs(args.output_dir, args.scope, calibration,
                  rule_registry_table(), results, args.render_plots,
                  args.plot_threshold, args.frozen_policy_id,
                  render_summary_plots=args.render_summary_plots)
    print(f"MOD-18 {args.scope} exports: {args.output_dir}")


if __name__ == "__main__":
    main()
