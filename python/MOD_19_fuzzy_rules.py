#!/usr/bin/env python3
"""MOD-19 stage 2: traceable conversion of Auditory CART paths to fuzzy rules.

Romanov et al. (2025), DOI 10.3390/axioms14030196, motivates the stages.
This is an experimental, development-only adaptation, NOT a Mamdani predictor.
No test observations, therapeutic actions or clinical thresholds are used.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import MOD_14_direct_linear_regression_evaluation as MOD14
import MOD_19_tree_mamdani as MOD19


TERMS = {
    3: ("Low", "Medium", "High"),
    5: ("Very low", "Low", "Medium", "High", "Very high"),
}


class ConversionError(ValueError):
    """Invalid source evidence, paths or fuzzy conversion."""


@dataclass(frozen=True)
class NumericRule:
    leaf_ids: tuple[int, ...]
    sample_count: int
    output: float
    bounds: tuple[tuple[str, float | None, float | None], ...]


def normalise(rule: MOD19.RawTreeRule) -> NumericRule:
    """Retain the strongest > lower and <= upper bounds per input feature.

    Unlike a lossy rule simplification, this preserves crisp CART routes.
    """
    bounds: dict[str, list[float | None]] = {}
    for condition in rule.conditions:
        if (condition.feature not in MOD14.AUDITORY_PREDICTORS
                or condition.operator not in ("<=", ">")
                or not np.isfinite(condition.threshold)):
            raise ConversionError("Unsupported or non-finite crisp tree condition")
        interval = bounds.setdefault(condition.feature, [None, None])
        if condition.operator == ">":
            interval[0] = (condition.threshold if interval[0] is None
                           else max(interval[0], condition.threshold))
        else:
            interval[1] = (condition.threshold if interval[1] is None
                           else min(interval[1], condition.threshold))
    for feature, (lower, upper) in bounds.items():
        if lower is not None and upper is not None and lower >= upper:
            raise ConversionError(f"Empty crisp interval for {feature}")
    return NumericRule(
        leaf_ids=(int(rule.leaf_id),),
        sample_count=int(rule.sample_count),
        output=float(rule.prediction),
        bounds=tuple(
            (feature, *bounds[feature])
            for feature in MOD14.AUDITORY_PREDICTORS if feature in bounds
        ),
    )


def reduce_identical_numeric(rules: tuple[NumericRule, ...]) -> tuple[NumericRule, ...]:
    """Paper's step 3: collapse identical antecedents, averaging consequents."""
    groups: dict[tuple, list[NumericRule]] = {}
    for rule in rules:
        groups.setdefault(rule.bounds, []).append(rule)
    reduced = []
    for group in groups.values():
        reduced.append(NumericRule(
            leaf_ids=tuple(i for rule in group for i in rule.leaf_ids),
            sample_count=sum(rule.sample_count for rule in group),
            output=float(np.mean([rule.output for rule in group])),
            bounds=group[0].bounds,
        ))
    return tuple(reduced)


def membership(value: float, centres: np.ndarray) -> np.ndarray:
    """Overlapping triangular interior and shoulder endpoint memberships."""
    if not np.isfinite(value) or len(centres) < 2 or np.any(np.diff(centres) <= 0):
        raise ConversionError("Membership requires finite value and distinct centres")
    weights = np.zeros(len(centres), dtype=float)
    if value <= centres[0]:
        weights[0] = 1.0
    elif value >= centres[-1]:
        weights[-1] = 1.0
    else:
        right = int(np.searchsorted(centres, value, side="right"))
        left = right - 1
        weights[right] = (value - centres[left]) / (centres[right] - centres[left])
        weights[left] = 1.0 - weights[right]
    return weights


def simplify_representatives(
    rule: NumericRule, ranges: dict[str, tuple[float, float]]
) -> tuple[tuple[str, float], ...]:
    """Paper-inspired step 4: interval midpoint or available extrema."""
    representatives = []
    for feature, lower, upper in rule.bounds:
        if feature not in ranges:
            raise ConversionError(f"Missing development calibration for {feature}")
        minimum, maximum = ranges[feature]
        if lower is not None and upper is not None:
            representative = (lower + upper) / 2
        elif lower is not None:
            representative = maximum
        elif upper is not None:
            representative = minimum
        else:
            raise ConversionError("Unbounded predictor must be omitted from the rule")
        if not np.isfinite(representative):
            raise ConversionError("Non-finite representative")
        if ((lower is not None and not representative > lower)
                or (upper is not None and not representative <= upper)):
            raise ConversionError("Representative lies outside its crisp interval")
        representatives.append((feature, float(representative)))
    return tuple(representatives)


def fuzzify(
    numeric: tuple[NumericRule, ...],
    ranges: dict[str, tuple[float, float]],
    *,
    number_terms: int,
) -> tuple[list[dict], dict[str, np.ndarray]]:
    """Paper-inspired step 5: map representatives/outputs to strongest terms."""
    if number_terms not in TERMS:
        raise ConversionError("Only provisional 3- or 5-term partitions supported")
    centres = {
        feature: np.linspace(low, high, number_terms)
        for feature, (low, high) in ranges.items()
    }
    for feature, values in centres.items():
        if not np.isfinite(values).all() or not (values[-1] > values[0]):
            raise ConversionError(f"Cannot fuzzify constant/invalid {feature}")
    raw = []
    for rule in numeric:
        selected = []
        degrees = []
        for feature, value in simplify_representatives(rule, ranges):
            weights = membership(value, centres[feature])
            index = int(np.argmax(weights))  # explicit deterministic lower-index tie
            selected.append((feature, TERMS[number_terms][index]))
            degrees.append(float(weights[index]))
        y_weights = membership(rule.output, centres["frozen_mlr_prediction"])
        output_index = int(np.argmax(y_weights))
        raw.append({
            "source_leaf_ids": list(rule.leaf_ids),
            "sample_count": rule.sample_count,
            "tree_output": rule.output,
            "antecedent": selected,
            "consequent": TERMS[number_terms][output_index],
            "membership_sum": float(sum(degrees)),
            "representatives": list(simplify_representatives(rule, ranges)),
            "centres_of_selected_terms": [
                float(centres[feature][TERMS[number_terms].index(term)])
                for feature, term in selected
            ],
        })
    return raw, centres


def reduce_identical_fuzzy(raw: list[dict]) -> tuple[list[dict], list[dict]]:
    """Paper's step 6: for equal antecedents choose minimum membership sum.

    All discarded mappings remain visible in the collision audit.
    """
    groups: dict[tuple, list[dict]] = {}
    for candidate in raw:
        groups.setdefault(tuple(map(tuple, candidate["antecedent"])), []).append(candidate)
    final, audits = [], []
    for antecedent, group in groups.items():
        chosen = min(
            group,
            key=lambda r: (r["membership_sum"], min(r["source_leaf_ids"])),
        )
        final.append(chosen)
        if len(group) > 1:
            audits.append({
                "antecedent": list(antecedent),
                "candidate_leaves": [r["source_leaf_ids"] for r in group],
                "candidate_consequents": [r["consequent"] for r in group],
                "candidate_membership_sums": [r["membership_sum"] for r in group],
                "chosen_leaves": chosen["source_leaf_ids"],
                "rule": "Romanov Algorithm 5 minimum antecedent membership sum",
            })
    return final, audits


def load_crisp_rules(table: pd.DataFrame) -> tuple[MOD19.RawTreeRule, ...]:
    needed = {"leaf_id", "development_row_count", "tree_output", "conditions_json"}
    if not needed.issubset(table.columns):
        raise ConversionError("Missing MOD-19 Stage 1 rule columns")
    if table["leaf_id"].duplicated().any() or table.empty:
        raise ConversionError("Crisp rules require unique, nonempty leaf IDs")
    rules = []
    for record in table.to_dict("records"):
        conditions = json.loads(record["conditions_json"])
        rules.append(MOD19.RawTreeRule(
            leaf_id=int(record["leaf_id"]),
            sample_count=int(record["development_row_count"]),
            prediction=float(record["tree_output"]),
            conditions=tuple(MOD19.TreeCondition(**item) for item in conditions),
        ))
    return tuple(rules)


def verify_crisp_source(
    frame: pd.DataFrame,
    rules: tuple[NumericRule, ...],
    predictions: pd.DataFrame,
) -> None:
    """Check normalisation preserves exact Stage-1 trial routing."""
    needed = {
        "participant_id", "difficulty_level", "difficulty_stage",
        "frozen_mlr_prediction", "provisional_tree_prediction",
    }
    if not needed.issubset(predictions.columns) or len(frame) != len(predictions):
        raise ConversionError("Stage-1 development rows/columns mismatch")
    for column in ("participant_id", "difficulty_level", "difficulty_stage"):
        if not np.array_equal(frame[column].to_numpy(), predictions[column].to_numpy()):
            raise ConversionError(f"Stage-1 development row order mismatch: {column}")
    if not np.allclose(
        frame["frozen_mlr_prediction"].to_numpy(dtype=float),
        predictions["frozen_mlr_prediction"].to_numpy(dtype=float),
        atol=1e-8, rtol=0,
    ):
        raise ConversionError("Frozen MLR targets differ from Stage 1")
    counts = {leaf: 0 for rule in rules for leaf in rule.leaf_ids}
    for idx, (_, row) in enumerate(frame.iterrows()):
        matching = []
        for rule in rules:
            if all(
                (lower is None or row[feature] > lower)
                and (upper is None or row[feature] <= upper)
                for feature, lower, upper in rule.bounds
            ):
                matching.append(rule)
        if len(matching) != 1:
            raise ConversionError("Normalised crisp rules must cover each row exactly once")
        match = matching[0]
        if len(match.leaf_ids) != 1:
            raise ConversionError("Source verification must precede numeric reduction")
        counts[match.leaf_ids[0]] += 1
        if not np.isclose(
            match.output, float(predictions.iloc[idx]["provisional_tree_prediction"]),
            atol=1e-8, rtol=0,
        ):
            raise ConversionError("Crisp rule no longer reproduces Stage-1 tree output")
    for rule in rules:
        if counts[rule.leaf_ids[0]] != rule.sample_count:
            raise ConversionError("Leaf coverage count differs from Stage 1")


def build_conversion(
    frame: pd.DataFrame,
    raw_rules: tuple[MOD19.RawTreeRule, ...],
    predictions: pd.DataFrame,
    *,
    number_terms: int,
) -> dict:
    """Convert verified development-only tree paths to provisional fuzzy labels."""
    normalised = tuple(normalise(rule) for rule in raw_rules)
    verify_crisp_source(frame, normalised, predictions)
    reduced = reduce_identical_numeric(normalised)
    ranges = {
        feature: (
            float(frame[feature].min()), float(frame[feature].max())
        )
        for feature in MOD14.AUDITORY_PREDICTORS
    }
    ranges["frozen_mlr_prediction"] = (
        float(frame["frozen_mlr_prediction"].min()),
        float(frame["frozen_mlr_prediction"].max()),
    )
    fuzzy_raw, centres = fuzzify(reduced, ranges, number_terms=number_terms)
    fuzzy_final, collisions = reduce_identical_fuzzy(fuzzy_raw)
    return {
        "normalised": normalised,
        "numeric": reduced,
        "fuzzy_raw": fuzzy_raw,
        "fuzzy_final": fuzzy_final,
        "collisions": collisions,
        "centres": centres,
        "ranges": ranges,
        "number_terms": number_terms,
    }


def write_conversion(result: dict, output_dir: Path, *, frame: pd.DataFrame) -> None:
    output_dir = Path(output_dir)
    if output_dir.exists() and (not output_dir.is_dir() or any(output_dir.iterdir())):
        raise ConversionError(f"Output directory is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    def save_csv(name: str, rows: list[dict], columns: list[str]) -> None:
        pd.DataFrame(rows, columns=columns).to_csv(output_dir / name, index=False)

    numeric_rows = [
        {
            "source_leaf_ids": json.dumps(rule.leaf_ids),
            "sample_count": rule.sample_count,
            "tree_output": rule.output,
            "bounds_json": json.dumps(rule.bounds),
            "representatives_json": json.dumps(
                simplify_representatives(rule, result["ranges"])
            ),
        }
        for rule in result["numeric"]
    ]
    save_csv(
        "normalised_numerical_rules.csv", numeric_rows,
        ["source_leaf_ids", "sample_count", "tree_output",
         "bounds_json", "representatives_json"],
    )

    def fuzzy_rows(rules: list[dict]) -> list[dict]:
        return [{
            "source_leaf_ids": json.dumps(rule["source_leaf_ids"]),
            "sample_count": rule["sample_count"],
            "tree_output": rule["tree_output"],
            "antecedent_json": json.dumps(rule["antecedent"]),
            "consequent": rule["consequent"],
            "representatives_json": json.dumps(rule["representatives"]),
            "membership_sum_at_representatives": rule["membership_sum"],
            "readable_rule": (
                "IF " + " AND ".join(f"{f} is {t}" for f, t in rule["antecedent"])
                + f" THEN predicted difficulty is {rule['consequent']}"
            ),
        } for rule in rules]

    fuzzy_columns = [
        "source_leaf_ids", "sample_count", "tree_output", "antecedent_json",
        "consequent", "representatives_json",
        "membership_sum_at_representatives", "readable_rule",
    ]
    save_csv("fuzzy_rules_before_reduction.csv",
             fuzzy_rows(result["fuzzy_raw"]), fuzzy_columns)
    save_csv("fuzzy_rules.csv",
             fuzzy_rows(result["fuzzy_final"]), fuzzy_columns)
    save_csv(
        "fuzzy_collisions.csv",
        [{
            "antecedent_json": json.dumps(a["antecedent"]),
            "candidate_leaves_json": json.dumps(a["candidate_leaves"]),
            "candidate_consequents_json": json.dumps(a["candidate_consequents"]),
            "candidate_membership_sums_json": json.dumps(a["candidate_membership_sums"]),
            "chosen_leaves_json": json.dumps(a["chosen_leaves"]),
            "selection_rule": a["rule"],
        } for a in result["collisions"]],
        ["antecedent_json", "candidate_leaves_json", "candidate_consequents_json",
         "candidate_membership_sums_json", "chosen_leaves_json", "selection_rule"],
    )
    partitions = []
    for feature, centres in result["centres"].items():
        for term, centre in zip(TERMS[result["number_terms"]], centres):
            partitions.append({
                "feature": feature, "term": term, "centre": float(centre),
                "development_min": result["ranges"][feature][0],
                "development_max": result["ranges"][feature][1],
            })
    save_csv(
        "membership_partitions.csv", partitions,
        ["feature", "term", "centre", "development_min", "development_max"],
    )
    fig, axes = plt.subplots(3, 2, figsize=(13, 11))
    for ax, (feature, centres) in zip(axes.flat, result["centres"].items()):
        xs = np.linspace(centres[0], centres[-1], 250)
        values = np.array([membership(float(x), centres) for x in xs])
        for i, term in enumerate(TERMS[result["number_terms"]]):
            ax.plot(xs, values[:, i], label=term)
        ax.set_title(feature.replace("_", " "))
        ax.set_ylabel("Membership")
        ax.legend(fontsize=7)
    for ax in axes.flat[len(result["centres"]):]:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(output_dir / "membership_functions.png", dpi=160)
    plt.close(fig)
    manifest = {
        "module": "MOD-19", "stage": "2: provisional fuzzy-rule conversion",
        "research_status": "development-only; Mamdani inference NOT implemented",
        "model_id": MOD19.MODEL_ID,
        "source": "Romanov et al. 2025 DOI 10.3390/axioms14030196; adapted",
        "numeric_normalisation": "tightest > and <= bounds; exact crisp route preserved",
        "numeric_duplicate_consequents": "unweighted average (paper algorithm 2)",
        "numeric_representatives": "midpoint, development min or development max",
        "fuzzy_memberships": "uniform overlapping triangular interior/shoulder endpoints",
        "fuzzy_terms": list(TERMS[result["number_terms"]]),
        "fuzzy_duplicate_selection": "minimum antecedent membership sum (paper algorithm 5)",
        "fuzzy_ties": "prefer lower term index, then smaller source leaf ID",
        "number_terms_provisional": result["number_terms"],
        "development_participants": int(frame["participant_id"].nunique()),
        "development_rows": len(frame), "test_participants_used": 0,
        "raw_numeric_rules": len(result["normalised"]),
        "reduced_numeric_rules": len(result["numeric"]),
        "fuzzy_rules_before_reduction": len(result["fuzzy_raw"]),
        "fuzzy_rules_after_reduction": len(result["fuzzy_final"]),
        "fuzzy_collision_groups": len(result["collisions"]),
        "limitations": [
            "Generated terms are relative to development ranges, not clinical cutoffs.",
            "Linguistic mapping and duplicate reduction may change tree behaviour.",
            "No Mamdani prediction, centroid, grouped selection, or test evaluation yet.",
        ],
    }
    (output_dir / "conversion_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--modeling-data", type=Path, required=True)
    parser.add_argument("--holdout-split", type=Path, required=True)
    parser.add_argument("--mod14-output-dir", type=Path, required=True)
    parser.add_argument("--stage1-output-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--terms", type=int, required=True, choices=tuple(TERMS))
    args = parser.parse_args()

    source_manifest = json.loads(
        (args.stage1_output_dir / "manifest.json").read_text(encoding="utf-8")
    )
    if (source_manifest.get("model_id") != MOD19.MODEL_ID
            or source_manifest.get("stage") != "1: crisp tree extraction only"
            or source_manifest.get("development_participants") != 26
            or source_manifest.get("test_participants_used") != 0):
        raise ConversionError("Source Stage-1 manifest does not match Auditory contract")
    frame = MOD19.prepare_development_teacher(
        modeling_data=pd.read_csv(args.modeling_data),
        holdout_split=pd.read_csv(args.holdout_split),
        regression_coefficients=pd.read_csv(
            args.mod14_output_dir / "regression_coefficients.csv"),
        training_predictions=pd.read_csv(
            args.mod14_output_dir / "training_predictions.csv"),
    )
    raw_rules = load_crisp_rules(pd.read_csv(
        args.stage1_output_dir / "numerical_rules.csv"
    ))
    if (len(raw_rules) != source_manifest["raw_rule_count"]
            or len(frame) != source_manifest["development_rows"]):
        raise ConversionError("Stage-1 manifest counts disagree with source")
    result = build_conversion(
        frame, raw_rules,
        pd.read_csv(args.stage1_output_dir / "development_predictions.csv"),
        number_terms=args.terms,
    )
    write_conversion(result, args.output_dir, frame=frame)


if __name__ == "__main__":
    main()
