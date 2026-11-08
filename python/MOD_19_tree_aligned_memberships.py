#!/usr/bin/env python3
"""MOD-19 Stage 2B: tree-threshold-aligned Auditory fuzzy antecedents.

Preserves the original uniform Stage-2 experiment. This alternative derives
region-specific input terms from exact Stage-1 CART split thresholds; a crisp
condition spanning several tree regions becomes an explicit OR of terms.
The output remains three development-relative terms (Low/Medium/High).
No Mamdani inference, grouped model selection or test-participant usage.
This is a thesis-specific extension of Romanov et al. (2025), not a claim
of verbatim implementation or numerical equivalence to the original tree.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import MOD_14_direct_linear_regression_evaluation as MOD14
import MOD_19_tree_mamdani as MOD19
import MOD_19_fuzzy_rules as BASE


TERM_LABELS = {
    1: ("Any",),
    2: ("Low", "High"),
    3: ("Low", "Medium", "High"),
    4: ("Very low", "Low", "Medium", "High"),
}
OUTPUT_TERMS = ("Low", "Medium", "High")


@dataclass(frozen=True)
class Partition:
    feature: str
    labels: tuple[str, ...]
    thresholds: tuple[float, ...]
    minimum: float
    maximum: float
    knots: tuple[tuple[float, tuple[float, ...]], ...]

    def degrees(self, value: float) -> np.ndarray:
        if not np.isfinite(value):
            raise BASE.ConversionError(f"Non-finite {self.feature}")
        positions = np.array([k[0] for k in self.knots], dtype=float)
        weights = np.array([k[1] for k in self.knots], dtype=float)
        return np.array(
            [np.interp(value, positions, weights[:, j])
             for j in range(len(self.labels))],
            dtype=float,
        )


def make_partition(
    feature: str, minimum: float, maximum: float, thresholds: tuple[float, ...]
) -> Partition:
    """Construct overlapping piecewise-linear sets crossing at 0.5 at each split."""
    boundaries = tuple(sorted(set(float(v) for v in thresholds)))
    if (not np.isfinite([minimum, maximum, *boundaries]).all()
            or minimum >= maximum
            or any(not (minimum < t < maximum) for t in boundaries)):
        raise BASE.ConversionError(f"Invalid development range/splits: {feature}")
    count = len(boundaries) + 1
    labels = TERM_LABELS.get(
        count, tuple(f"Region {j + 1}" for j in range(count))
    )

    def one(index: int) -> tuple[float, ...]:
        return tuple(float(j == index) for j in range(count))

    def crossover(index: int) -> tuple[float, ...]:
        return tuple(0.5 if j in (index, index + 1) else 0.0
                     for j in range(count))

    knots = [(float(minimum), one(0))]
    for index, threshold in enumerate(boundaries):
        knots.append((threshold, crossover(index)))
        if index + 1 < len(boundaries):
            midpoint = (threshold + boundaries[index + 1]) / 2
            knots.append((midpoint, one(index + 1)))
    knots.append((float(maximum), one(count - 1)))
    if any(b[0] <= a[0] for a, b in zip(knots, knots[1:])):
        raise BASE.ConversionError(f"Nonincreasing fuzzy knot positions: {feature}")
    return Partition(
        feature, labels, boundaries, float(minimum), float(maximum), tuple(knots)
    )


def build_partitions(
    frame: pd.DataFrame, numeric_rules: tuple[BASE.NumericRule, ...]
) -> dict[str, Partition]:
    """Find every exact CART split for each independent Auditory input."""
    partitions = {}
    for feature in MOD14.AUDITORY_PREDICTORS:
        if feature not in frame:
            raise BASE.ConversionError(f"Missing Auditory feature: {feature}")
        values = pd.to_numeric(frame[feature], errors="coerce").to_numpy(dtype=float)
        if not np.isfinite(values).all():
            raise BASE.ConversionError(f"Non-finite Auditory feature: {feature}")
        thresholds = sorted({
            float(bound)
            for rule in numeric_rules
            for f, lower, upper in rule.bounds if f == feature
            for bound in (lower, upper) if bound is not None
        })
        partitions[feature] = make_partition(
            feature, float(values.min()), float(values.max()), tuple(thresholds)
        )
    return partitions


def route_terms(
    lower: float | None, upper: float | None, partition: Partition
) -> tuple[str, ...]:
    """Represent a crisp interval by the OR of *all* covered tree regions.

    Every internal CART threshold is represented exactly once in the partition.
    Crisp regions are (t[i-1], t[i]], with outer tails. This retains coverage
    without claiming that continuous fuzzy membership equals a crisp route.
    """
    boundaries = partition.thresholds
    for endpoint in (lower, upper):
        if endpoint is not None and endpoint not in boundaries:
            raise BASE.ConversionError(
                f"Crisp bound not present in learned split thresholds: {endpoint}"
            )
    selected = []
    for index, name in enumerate(partition.labels):
        left = boundaries[index - 1] if index else None
        right = boundaries[index] if index < len(boundaries) else None
        if ((lower is None or (left is not None and left >= lower))
                and (upper is None or (right is not None and right <= upper))):
            selected.append(name)
    if not selected:
        raise BASE.ConversionError(f"No fuzzy region covers numeric interval: {partition.feature}")
    return tuple(selected)


def antecedent_degree(
    antecedent: tuple[tuple[str, tuple[str, ...]], ...],
    partitions: dict[str, Partition],
    row: pd.Series,
) -> float:
    """MAX for alternatives within each input; MIN across different inputs."""
    degrees = []
    for feature, options in antecedent:
        partition = partitions[feature]
        strengths = partition.degrees(float(row[feature]))
        degrees.append(max(
            float(strengths[partition.labels.index(name)]) for name in options
        ))
    return min(degrees) if degrees else 1.0


def aligned_conversion(
    frame: pd.DataFrame,
    raw_rules: tuple[MOD19.RawTreeRule, ...],
    stage1_predictions: pd.DataFrame,
) -> dict:
    normalised = tuple(BASE.normalise(rule) for rule in raw_rules)
    BASE.verify_crisp_source(frame, normalised, stage1_predictions)
    numeric = BASE.reduce_identical_numeric(normalised)
    if len(numeric) != len(normalised):
        raise BASE.ConversionError(
            "Tree-aligned Stage 2B requires unique numerical paths; inspect reduction first"
        )
    partitions = build_partitions(frame, numeric)
    output_min = float(frame["frozen_mlr_prediction"].min())
    output_max = float(frame["frozen_mlr_prediction"].max())
    if not output_min < output_max:
        raise BASE.ConversionError("Constant output cannot support three fuzzy terms")
    output_centres = np.linspace(output_min, output_max, 3)
    rules = []
    for rule in numeric:
        antecedent = tuple(
            (feature, route_terms(lower, upper, partitions[feature]))
            for feature, lower, upper in rule.bounds
        )
        output_weights = BASE.membership(rule.output, output_centres)
        consequent = OUTPUT_TERMS[int(np.argmax(output_weights))]
        clauses = [
            f"{feature} is " + (" OR ".join(options) if len(options) == 1
                               else "(" + " OR ".join(options) + ")")
            for feature, options in antecedent
        ]
        rules.append({
            "source_leaf_ids": list(rule.leaf_ids),
            "sample_count": rule.sample_count,
            "tree_output": rule.output,
            "antecedent": antecedent,
            "consequent": consequent,
            "readable_rule": "IF " + " AND ".join(clauses)
                             + f" THEN predicted difficulty is {consequent}",
        })

    collisions: dict[tuple, list[dict]] = {}
    for rule in rules:
        collisions.setdefault(rule["antecedent"], []).append(rule)
    audits = [{
        "antecedent": antecedent,
        "source_leaf_ids": [r["source_leaf_ids"] for r in group],
        "consequents": [r["consequent"] for r in group],
        "resolution": "PRESERVED: no silent duplicate/conflict removal",
    } for antecedent, group in collisions.items() if len(group) > 1]

    # Antecedent coverage, not Mamdani inference and not predictive fidelity.
    diagnostics = []
    for _, row in frame.iterrows():
        source = [
            rule for rule in numeric
            if all(
                (lower is None or row[feature] > lower)
                and (upper is None or row[feature] <= upper)
                for feature, lower, upper in rule.bounds
            )
        ]
        if len(source) != 1:
            raise BASE.ConversionError("Exactly one original CART rule must match")
        strengths = [
            antecedent_degree(rule["antecedent"], partitions, row)
            for rule in rules
        ]
        matched = next(i for i, rule in enumerate(rules)
                       if rule["source_leaf_ids"] == list(source[0].leaf_ids))
        if strengths[matched] < 0.5 - 1e-10 or max(strengths) <= 0:
            raise BASE.ConversionError("Tree-aligned fuzzy route has a coverage hole")
        diagnostics.append({
            "participant_id": row["participant_id"],
            "difficulty_stage": int(row["difficulty_stage"]),
            "source_leaf_ids": json.dumps(source[0].leaf_ids),
            "source_rule_activation": float(strengths[matched]),
            "max_rule_activation": float(max(strengths)),
            "active_rule_count": sum(v > 0 for v in strengths),
        })
    return {
        "partitions": partitions, "output_centres": output_centres,
        "rules": rules, "collision_audit": audits, "diagnostics": diagnostics,
        "development_rows": len(frame),
        "development_participants": int(frame["participant_id"].nunique()),
    }


def write_outputs(result: dict, destination: Path, *, source_dir: Path) -> None:
    destination = Path(destination)
    if destination.exists() and (not destination.is_dir() or any(destination.iterdir())):
        raise BASE.ConversionError(f"Output directory is not empty: {destination}")
    destination.mkdir(parents=True, exist_ok=True)

    def csv(name: str, records: list[dict], columns: list[str]) -> None:
        pd.DataFrame(records, columns=columns).to_csv(destination / name, index=False)

    partitions = result["partitions"]
    records = []
    for feature, partition in partitions.items():
        for index, label in enumerate(partition.labels):
            records.append({
                "feature": feature, "term": label,
                "region_index": index,
                "split_thresholds_json": json.dumps(partition.thresholds),
                "knots_json": json.dumps(partition.knots),
                "development_min": partition.minimum,
                "development_max": partition.maximum,
            })
    for j, label in enumerate(OUTPUT_TERMS):
        records.append({
            "feature": "frozen_mlr_prediction", "term": label,
            "region_index": j, "split_thresholds_json": "[]",
            "knots_json": "[]",
            "development_min": float(result["output_centres"][0]),
            "development_max": float(result["output_centres"][-1]),
        })
    csv(
        "tree_aligned_membership_partitions.csv", records,
        ["feature", "term", "region_index", "split_thresholds_json",
         "knots_json", "development_min", "development_max"],
    )
    csv(
        "tree_aligned_fuzzy_rules.csv",
        [{
            "source_leaf_ids": json.dumps(r["source_leaf_ids"]),
            "sample_count": r["sample_count"],
            "tree_output": r["tree_output"],
            "antecedent_or_groups_json": json.dumps(r["antecedent"]),
            "consequent": r["consequent"],
            "readable_rule": r["readable_rule"],
        } for r in result["rules"]],
        ["source_leaf_ids", "sample_count", "tree_output",
         "antecedent_or_groups_json", "consequent", "readable_rule"],
    )
    csv(
        "tree_aligned_collision_audit.csv",
        [{
            "antecedent_or_groups_json": json.dumps(r["antecedent"]),
            "source_leaf_ids_json": json.dumps(r["source_leaf_ids"]),
            "consequents_json": json.dumps(r["consequents"]),
            "resolution": r["resolution"],
        } for r in result["collision_audit"]],
        ["antecedent_or_groups_json", "source_leaf_ids_json",
         "consequents_json", "resolution"],
    )
    csv(
        "development_antecedent_coverage.csv", result["diagnostics"],
        ["participant_id", "difficulty_stage", "source_leaf_ids",
         "source_rule_activation", "max_rule_activation", "active_rule_count"],
    )

    fig, axes = plt.subplots(3, 2, figsize=(13, 11))
    for ax, (feature, partition) in zip(axes.flat, partitions.items()):
        grid = np.linspace(partition.minimum, partition.maximum, 400)
        membership_grid = np.array([partition.degrees(float(x)) for x in grid])
        for index, label in enumerate(partition.labels):
            ax.plot(grid, membership_grid[:, index], label=label)
        for threshold in partition.thresholds:
            ax.axvline(threshold, linestyle=":", linewidth=0.9, alpha=0.6)
        ax.set_title(feature.replace("_", " "))
        ax.set_ylabel("Membership")
        ax.legend(fontsize=7)
    output_axis = axes.flat[4]
    output_grid = np.linspace(
        result["output_centres"][0], result["output_centres"][-1], 400
    )
    output_memberships = np.array([
        BASE.membership(float(x), result["output_centres"])
        for x in output_grid
    ])
    for j, label in enumerate(OUTPUT_TERMS):
        output_axis.plot(output_grid, output_memberships[:, j], label=label)
    output_axis.set_title("frozen MLR prediction (3 uniform output terms)")
    output_axis.set_ylabel("Membership")
    output_axis.legend(fontsize=7)
    axes.flat[5].axis("off")
    fig.tight_layout()
    fig.savefig(destination / "tree_aligned_membership_functions.png", dpi=160)
    plt.close(fig)

    source_hashes = {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in (
            source_dir / "manifest.json",
            source_dir / "numerical_rules.csv",
            source_dir / "development_predictions.csv",
        )
    }
    manifest = {
        "module": "MOD-19", "stage": "2B: tree-aligned membership conversion",
        "research_status": "provisional development-only; NOT Mamdani fitted",
        "model_id": MOD19.MODEL_ID,
        "method_reference": "Romanov et al. 2025, DOI 10.3390/axioms14030196",
        "adaptation": (
            "Exact CART splits become 0.5 membership crossovers; antecedent "
            "conditions spanning regions become OR-groups. This additional "
            "alignment method is a thesis-specific design, not a published "
            "Romanov algorithm reproduced verbatim."
        ),
        "membership_rule": (
            "piecewise-linear partition of unity, shoulder endpoints, "
            "0.5 crossovers at CART split thresholds"
        ),
        "antecedent_connectives_for_future_inference": (
            "OR=max within feature; AND=min across features"
        ),
        "output_terms": list(OUTPUT_TERMS),
        "input_terms": {
            feature: list(partition.labels)
            for feature, partition in partitions.items()
        },
        "input_thresholds": {
            feature: list(partition.thresholds)
            for feature, partition in partitions.items()
        },
        "development_participants": result["development_participants"],
        "development_rows": result["development_rows"],
        "source_rule_count": len(result["rules"]),
        "fuzzy_rule_count": len(result["rules"]),
        "collision_groups_preserved": len(result["collision_audit"]),
        "antecedent_coverage_rows": len(result["diagnostics"]),
        "test_participants_used": 0,
        "stage1_file_sha256": source_hashes,
        "limitations": [
            "Input category counts derive from provisional tree; grouped selection pending.",
            "Extra OR alternatives may affect interpretability and rule overlap.",
            "Only antecedent coverage verified, NOT Mamdani predictions or fidelity.",
            "Output terms remain uniformly spaced; no clinical calibration.",
            "Previously inspected reserved-test cases cannot support untouched validation.",
        ],
    }
    (destination / "tree_aligned_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--modeling-data", type=Path, required=True)
    parser.add_argument("--holdout-split", type=Path, required=True)
    parser.add_argument("--mod14-output-dir", type=Path, required=True)
    parser.add_argument("--stage1-output-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    stage1_manifest = json.loads(
        (args.stage1_output_dir / "manifest.json").read_text(encoding="utf-8")
    )
    if (stage1_manifest.get("model_id") != MOD19.MODEL_ID
            or stage1_manifest.get("stage") != "1: crisp tree extraction only"
            or stage1_manifest.get("development_participants") != 26
            or stage1_manifest.get("test_participants_used") != 0):
        raise BASE.ConversionError("Stage-1 Auditory development manifest mismatch")
    frame = MOD19.prepare_development_teacher(
        modeling_data=pd.read_csv(args.modeling_data),
        holdout_split=pd.read_csv(args.holdout_split),
        regression_coefficients=pd.read_csv(
            args.mod14_output_dir / "regression_coefficients.csv"
        ),
        training_predictions=pd.read_csv(
            args.mod14_output_dir / "training_predictions.csv"
        ),
    )
    raw_rules = BASE.load_crisp_rules(
        pd.read_csv(args.stage1_output_dir / "numerical_rules.csv")
    )
    if (len(raw_rules) != stage1_manifest["raw_rule_count"]
            or len(frame) != stage1_manifest["development_rows"]):
        raise BASE.ConversionError("Stage-1 rule/row count mismatch")
    stage1_predictions = pd.read_csv(
        args.stage1_output_dir / "development_predictions.csv"
    )
    result = aligned_conversion(frame, raw_rules, stage1_predictions)
    write_outputs(result, args.output_dir, source_dir=args.stage1_output_dir)


if __name__ == "__main__":
    main()
