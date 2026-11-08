#!/usr/bin/env python3
"""MOD-19 Stage 3: fixed Mamdani inference for the Auditory tree-aligned rules.

This experiment *imitates* frozen MOD-14 Auditory MLR predictions. It does not
train another predictor, change existing fuzzy partitions, or recommend therapy.
No patient test observations are used. All fidelity metrics here are in-sample
development diagnostics, not participant-grouped cross-validation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import MOD_14_direct_linear_regression_evaluation as MOD14
import MOD_19_tree_mamdani as MOD19
import MOD_19_fuzzy_rules as BASE
import MOD_19_tree_aligned_memberships as ALIGNED


OUTPUT_GRID_SIZE = 4001
STATUS_OK = "OK"
STATUS_UNAVAILABLE = "Unavailable"
OUTPUT_TERMS = ALIGNED.OUTPUT_TERMS


def file_sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def centroid(x: np.ndarray, membership: np.ndarray) -> tuple[float, float]:
    """Trapezoidal area and first moment; no fallback for an empty output."""
    x = np.asarray(x, dtype=float)
    membership = np.asarray(membership, dtype=float)
    if (x.ndim != 1 or membership.shape != x.shape or len(x) < 3
            or not np.isfinite(x).all() or not np.isfinite(membership).all()
            or np.any(np.diff(x) <= 0)
            or np.any((membership < -1e-12) | (membership > 1 + 1e-12))):
        raise BASE.ConversionError("Invalid centroid grid or membership values")
    widths = np.diff(x)
    area = float(np.sum(widths * (membership[:-1] + membership[1:]) / 2))
    moment = float(np.sum(
        widths * (
            x[:-1] * membership[:-1] + x[1:] * membership[1:]
        ) / 2
    ))
    if area <= 1e-12:
        return float("nan"), area
    return moment / area, area


def output_grid(
    centres: np.ndarray, *, grid_size: int = OUTPUT_GRID_SIZE,
) -> tuple[np.ndarray, np.ndarray]:
    """Sample the saved three-term output universe, without clamping inputs."""
    centres = np.asarray(centres, dtype=float)
    if (centres.shape != (3,) or not np.isfinite(centres).all()
            or np.any(np.diff(centres) <= 0)
            or isinstance(grid_size, bool) or grid_size < 101
            or grid_size % 2 != 1):
        raise BASE.ConversionError("Invalid output centres or integration resolution")
    x = np.linspace(centres[0], centres[-1], grid_size)
    memberships = np.array([BASE.membership(v, centres) for v in x])
    return x, memberships


def aggregate_consequents(
    activations: np.ndarray, rules: list[dict], output_memberships: np.ndarray
) -> np.ndarray:
    """Mamdani MIN implication, MAX aggregation; do not normalise strengths."""
    activations = np.asarray(activations, dtype=float)
    if (activations.shape != (len(rules),)
            or output_memberships.ndim != 2
            or output_memberships.shape[1] != 3
            or not np.isfinite(activations).all()
            or np.any((activations < 0) | (activations > 1))):
        raise BASE.ConversionError("Invalid rule activations/output memberships")
    aggregated = np.zeros(output_memberships.shape[0], dtype=float)
    for activation, rule in zip(activations, rules):
        consequent = rule["consequent"]
        if consequent not in OUTPUT_TERMS:
            raise BASE.ConversionError(f"Unknown Mamdani consequent: {consequent}")
        if activation > 0:
            index = OUTPUT_TERMS.index(consequent)
            aggregated = np.maximum(
                aggregated,
                np.minimum(activation, output_memberships[:, index]),
            )
    return aggregated


def rule_strength(
    row: pd.Series, rule: dict, partitions: dict[str, ALIGNED.Partition]
) -> tuple[float, list[dict]]:
    """MAX within one feature's OR terms; MIN across feature antecedents."""
    strengths = []
    trace = []
    for feature, options in rule["antecedent"]:
        partition = partitions[feature]
        degrees = partition.degrees(float(row[feature]))
        terms = [
            {"term": label, "degree": float(degrees[partition.labels.index(label)])}
            for label in options
        ]
        combined = max(item["degree"] for item in terms)
        strengths.append(combined)
        trace.append({
            "feature": feature, "alternatives": terms,
            "or_activation": combined,
        })
    return (min(strengths) if strengths else 1.0), trace


def infer_observation(
    row: pd.Series,
    rules: list[dict],
    partitions: dict[str, ALIGNED.Partition],
    x: np.ndarray,
    output_memberships: np.ndarray,
) -> tuple[dict, list[dict], np.ndarray]:
    if not rules:
        raise BASE.ConversionError("Mamdani rule base is empty")
    activations = []
    traces = []
    for rule in rules:
        activation, trace = rule_strength(row, rule, partitions)
        activations.append(activation)
        traces.append({
            "source_leaf_ids": json.dumps(rule["source_leaf_ids"]),
            "consequent": rule["consequent"],
            "activation": activation,
            "antecedent_trace_json": json.dumps(trace),
        })
    activations_arr = np.asarray(activations, dtype=float)
    aggregate = aggregate_consequents(activations_arr, rules, output_memberships)
    prediction, area = centroid(x, aggregate)
    status = STATUS_OK if np.isfinite(prediction) else STATUS_UNAVAILABLE
    return ({
        "mamdani_prediction": prediction,
        "status": status,
        "output_area": area,
        "maximum_rule_activation": float(activations_arr.max()),
        "active_rules": int(np.count_nonzero(activations_arr > 0)),
    }, traces, aggregate)


def participant_balanced_mae(
    frame: pd.DataFrame, *, reference: str, prediction: str
) -> float:
    if frame.empty or not {"participant_id", reference, prediction}.issubset(frame):
        raise BASE.ConversionError("Missing participant-balanced MAE inputs")
    if frame["participant_id"].isna().any():
        raise BASE.ConversionError("Missing participant identifier")
    if not np.isfinite(frame[[reference, prediction]].to_numpy(dtype=float)).all():
        raise BASE.ConversionError("Non-finite fidelity values")
    absolute = (frame[reference] - frame[prediction]).abs()
    return float(absolute.groupby(frame["participant_id"], sort=False).mean().mean())


def pooled_r2(
    frame: pd.DataFrame, *, reference: str, prediction: str
) -> float:
    """Pooled R2 uses the *reference predictions* as the denominator."""
    if frame.empty or not {reference, prediction}.issubset(frame):
        raise BASE.ConversionError("Missing R2 inputs")
    y = frame[reference].to_numpy(dtype=float)
    y_hat = frame[prediction].to_numpy(dtype=float)
    if not np.isfinite(y).all() or not np.isfinite(y_hat).all():
        raise BASE.ConversionError("Non-finite R2 inputs")
    denominator = float(np.sum((y - np.mean(y)) ** 2))
    return (float("nan") if denominator <= 1e-14 else
            float(1 - np.sum((y - y_hat) ** 2) / denominator))


def evaluate_fixed_rules(frame: pd.DataFrame, converted: dict) -> tuple[
    pd.DataFrame, pd.DataFrame, pd.DataFrame, dict
]:
    """Evaluate fixed Stage-2B rules without fitting or selecting parameters."""
    x, output_memberships = output_grid(converted["output_centres"])
    rows, activations = [], []
    illustration = None
    for _, source in frame.iterrows():
        result, traces, aggregate = infer_observation(
            source, converted["rules"], converted["partitions"],
            x, output_memberships,
        )
        new_row = {
            "participant_id": source["participant_id"],
            "difficulty_level": source["difficulty_level"],
            "difficulty_stage": source["difficulty_stage"],
            "frozen_mlr_prediction": float(source["frozen_mlr_prediction"]),
            **result,
        }
        rows.append(new_row)
        for trace in traces:
            activations.append({
                "participant_id": source["participant_id"],
                "difficulty_level": source["difficulty_level"],
                "difficulty_stage": source["difficulty_stage"],
                **trace,
            })
        if illustration is None:
            illustration = {
                "x": x, "output_memberships": output_memberships,
                "aggregate": aggregate,
                "mamdani": result["mamdani_prediction"],
                "mlr": float(source["frozen_mlr_prediction"]),
            }
    predictions = pd.DataFrame(rows)
    if len(predictions) != len(frame):
        raise BASE.ConversionError("Missing development inference rows")
    valid = predictions["status"].eq(STATUS_OK)
    predictions["absolute_mlr_mamdani_error"] = (
        predictions["frozen_mlr_prediction"] - predictions["mamdani_prediction"]
    ).abs()
    metrics = []
    for name, reference, predicted in (
        ("MLR vs Mamdani", "frozen_mlr_prediction", "mamdani_prediction"),
    ):
        subset = predictions.loc[valid]
        metrics.append({
            "comparison": name,
            "reference": reference,
            "prediction": predicted,
            "n_available": len(subset),
            "n_unavailable": int((~valid).sum()),
            "development_participants": int(subset["participant_id"].nunique()),
            "participant_balanced_mae": (
                participant_balanced_mae(subset, reference=reference,
                                         prediction=predicted)
                if not subset.empty else float("nan")
            ),
            "pooled_r2": (
                pooled_r2(subset, reference=reference, prediction=predicted)
                if not subset.empty else float("nan")
            ),
            "maximum_absolute_error": (
                float(subset["absolute_mlr_mamdani_error"].max())
                if not subset.empty else float("nan")
            ),
            "evaluation": "development, in-sample surrogate fidelity only",
        })
    return predictions, pd.DataFrame(activations), pd.DataFrame(metrics), illustration


def saved_knots_match(serialized: str, expected: tuple) -> bool:
    """Compare ragged (position, membership-vector) knots without 2-D coercion."""
    try:
        recorded = json.loads(serialized)
        if not isinstance(recorded, list) or len(recorded) != len(expected):
            return False
        for recorded_knot, (position, degrees) in zip(recorded, expected):
            if (not isinstance(recorded_knot, list)
                    or len(recorded_knot) != 2
                    or not isinstance(recorded_knot[1], list)
                    or len(recorded_knot[1]) != len(degrees)):
                return False
            if not np.isclose(
                float(recorded_knot[0]), position, atol=1e-9, rtol=0
            ):
                return False
            if not np.allclose(
                np.asarray(recorded_knot[1], dtype=float),
                np.asarray(degrees, dtype=float),
                atol=1e-9, rtol=0,
            ):
                return False
        return True
    except (TypeError, ValueError, OverflowError):
        return False


def verify_sources(
    *,
    stage1_dir: Path, stage2b_dir: Path,
    stage1_manifest: dict, stage2b_manifest: dict,
    raw_rules: tuple[MOD19.RawTreeRule, ...],
    stage1_predictions: pd.DataFrame, frame: pd.DataFrame,
) -> dict:
    """Reconstruct and compare the exact saved Stage-2B rule base and partitions."""
    if (stage1_manifest.get("stage") != "1: crisp tree extraction only"
            or stage2b_manifest.get("stage") != "2B: tree-aligned membership conversion"
            or stage1_manifest.get("model_id") != MOD19.MODEL_ID
            or stage2b_manifest.get("model_id") != MOD19.MODEL_ID
            or stage1_manifest.get("development_participants") != 26
            or stage2b_manifest.get("development_participants") != 26
            or stage1_manifest.get("test_participants_used") != 0
            or stage2b_manifest.get("test_participants_used") != 0
            or len(frame) != stage1_manifest.get("development_rows")
            or len(frame) != stage2b_manifest.get("development_rows")
            or len(raw_rules) != stage1_manifest.get("raw_rule_count")):
        raise BASE.ConversionError("Stage-1/2B manifests disagree with frozen contract")
    for filename, recorded_hash in stage2b_manifest["stage1_file_sha256"].items():
        if filename not in (
            "manifest.json", "numerical_rules.csv", "development_predictions.csv"
        ) or file_sha256(stage1_dir / filename) != recorded_hash:
            raise BASE.ConversionError("Stage-1 source SHA256 changed since Stage 2B")

    result = ALIGNED.aligned_conversion(frame, raw_rules, stage1_predictions)
    if (len(result["rules"]) != stage2b_manifest["fuzzy_rule_count"]
            or result["development_participants"] != 26):
        raise BASE.ConversionError("Stage-2B rule count mismatch")

    saved_rules = pd.read_csv(stage2b_dir / "tree_aligned_fuzzy_rules.csv")
    if len(saved_rules) != len(result["rules"]):
        raise BASE.ConversionError("Saved Stage-2B rules incomplete")
    for saved, expected in zip(saved_rules.to_dict("records"), result["rules"]):
        if (json.loads(saved["source_leaf_ids"]) != expected["source_leaf_ids"]
                or json.loads(saved["antecedent_or_groups_json"])
                != [[feature, list(options)]
                    for feature, options in expected["antecedent"]]
                or saved["consequent"] != expected["consequent"]
                or not np.isclose(float(saved["tree_output"]),
                                  expected["tree_output"], atol=1e-9, rtol=0)):
            raise BASE.ConversionError("Stage-2B saved linguistic rules changed")
    saved_partitions = pd.read_csv(
        stage2b_dir / "tree_aligned_membership_partitions.csv"
    )
    if len(saved_partitions) != sum(
        len(partition.labels) for partition in result["partitions"].values()
    ) + len(OUTPUT_TERMS):
        raise BASE.ConversionError("Saved Stage-2B membership partition count mismatch")
    for feature, partition in result["partitions"].items():
        subset = saved_partitions[saved_partitions["feature"].eq(feature)]
        if (list(subset["term"]) != list(partition.labels)
                or not all(json.loads(item) == list(partition.thresholds)
                           for item in subset["split_thresholds_json"])
                or not all(
                    saved_knots_match(item, partition.knots)
                    for item in subset["knots_json"]
                )):
            raise BASE.ConversionError(f"Saved membership differs: {feature}")
    if stage2b_manifest["input_terms"] != {
        feature: list(partition.labels)
        for feature, partition in result["partitions"].items()
    }:
        raise BASE.ConversionError("Stage-2B linguistic terms changed")
    return result


def write_outputs(
    output_dir: Path,
    predictions: pd.DataFrame, activation_table: pd.DataFrame,
    metrics: pd.DataFrame, illustration: dict,
    *, frame: pd.DataFrame, rule_count: int,
    stage1_dir: Path, stage2b_dir: Path, mod14_dir: Path,
) -> None:
    output_dir = Path(output_dir)
    if output_dir.exists() and (
        not output_dir.is_dir() or any(output_dir.iterdir())
    ):
        raise BASE.ConversionError(f"Output directory is not empty: {output_dir}")
    if (rule_count < 1 or len(predictions) != len(frame)
            or len(activation_table) != len(frame) * rule_count
            or not activation_table.groupby(
                ["participant_id", "difficulty_level"], dropna=False
            ).size().eq(rule_count).all()):
        raise BASE.ConversionError("Incomplete Mamdani development results")
    output_dir.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(output_dir / "mamdani_predictions.csv", index=False)
    activation_table.to_csv(output_dir / "rule_activations.csv", index=False)
    metrics.to_csv(output_dir / "fidelity_metrics.csv", index=False)

    fig, ax = plt.subplots(figsize=(10, 5.5))
    x = illustration["x"]
    for j, label in enumerate(OUTPUT_TERMS):
        ax.plot(x, illustration["output_memberships"][:, j],
                linestyle="--", alpha=0.55, label=f"{label} output membership")
    ax.plot(x, illustration["aggregate"], linewidth=2.3,
            label="Aggregated clipped output")
    if np.isfinite(illustration["mamdani"]):
        ax.axvline(illustration["mamdani"], color="black", linestyle="-",
                   label="Mamdani centroid")
    ax.axvline(illustration["mlr"], color="grey", linestyle=":",
               label="Frozen MLR")
    ax.set_xlabel("Predicted difficulty stage (continuous)")
    ax.set_ylabel("Membership degree")
    ax.set_title("Illustrative development observation: Mamdani aggregation")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output_dir / "example_mamdani_aggregation.png", dpi=160)
    plt.close(fig)
    files = {
        "mod14_regression_coefficients.csv": mod14_dir / "regression_coefficients.csv",
        "mod14_training_predictions.csv": mod14_dir / "training_predictions.csv",
        "stage1_manifest.json": stage1_dir / "manifest.json",
        "stage1_numerical_rules.csv": stage1_dir / "numerical_rules.csv",
        "stage1_development_predictions.csv": stage1_dir / "development_predictions.csv",
        "stage2b_manifest.json": stage2b_dir / "tree_aligned_manifest.json",
        "stage2b_fuzzy_rules.csv": stage2b_dir / "tree_aligned_fuzzy_rules.csv",
        "stage2b_partitions.csv": stage2b_dir / "tree_aligned_membership_partitions.csv",
    }
    manifest = {
        "module": "MOD-19", "stage": "3: fixed Mamdani development baseline",
        "model_id": MOD19.MODEL_ID,
        "teacher": "frozen MOD-14 Auditory MLR predictions",
        "research_status": "provisional development-only, in-sample descriptive fidelity",
        "development_participants": int(frame["participant_id"].nunique()),
        "development_rows": len(frame), "test_participants_used": 0,
        "rule_count": rule_count,
        "inference": {
            "antecedent_or": "MAX within one input's alternatives",
            "antecedent_and": "MIN across inputs",
            "implication": "MIN clipping",
            "aggregation": "MAX across all rule outputs",
            "defuzzification": "centroid, composite trapezoidal integration",
            "output_universe_points": OUTPUT_GRID_SIZE,
            "output_universe": "frozen development MLR prediction min/max",
            "zero_activation": "Unavailable, no fabricated fallback prediction",
        },
        "n_available": int(predictions["status"].eq(STATUS_OK).sum()),
        "n_unavailable": int(predictions["status"].eq(STATUS_UNAVAILABLE).sum()),
        "source_sha256": {
            name: file_sha256(path) for name, path in files.items()
        },
        "limitations": [
            "In-sample surrogate fidelity; NOT participant-grouped validation.",
            "Crisp tree hyperparameters/fuzzy terms remain provisional.",
            "Fuzzy predictions approximate the frozen MLR, not the assigned difficulty.",
            "Output universe is development MLR range; no clinical calibration.",
            "No prospective adaptation, SHAP attribution or therapeutic prescription.",
            "Previously inspected reserved-test cases cannot support untouched validation.",
        ],
    }
    (output_dir / "mamdani_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--modeling-data", required=True, type=Path)
    parser.add_argument("--holdout-split", required=True, type=Path)
    parser.add_argument("--mod14-output-dir", required=True, type=Path)
    parser.add_argument("--stage1-output-dir", required=True, type=Path)
    parser.add_argument("--stage2b-output-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()

    stage1 = json.loads(
        (args.stage1_output_dir / "manifest.json").read_text(encoding="utf-8")
    )
    stage2b = json.loads(
        (args.stage2b_output_dir / "tree_aligned_manifest.json").read_text(
            encoding="utf-8"
        )
    )
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
    raw = BASE.load_crisp_rules(
        pd.read_csv(args.stage1_output_dir / "numerical_rules.csv")
    )
    original = pd.read_csv(
        args.stage1_output_dir / "development_predictions.csv"
    )
    converted = verify_sources(
        stage1_dir=args.stage1_output_dir,
        stage2b_dir=args.stage2b_output_dir,
        stage1_manifest=stage1, stage2b_manifest=stage2b,
        raw_rules=raw, stage1_predictions=original, frame=frame,
    )
    predictions, activations, metrics, illustration = evaluate_fixed_rules(
        frame, converted
    )
    if len(predictions) != 104 or len(activations) != 104 * len(converted["rules"]):
        raise BASE.ConversionError("Unexpected frozen Auditory development coverage")
    predictions["provisional_tree_prediction"] = original[
        "provisional_tree_prediction"
    ].to_numpy(dtype=float)
    predictions["absolute_tree_mamdani_error"] = (
        predictions["provisional_tree_prediction"] - predictions["mamdani_prediction"]
    ).abs()
    # Add both fidelity links for a three-way MLR / tree / Mamdani comparison.
    for comparison, ref, target in (
        ("MLR vs crisp tree", "frozen_mlr_prediction",
         "provisional_tree_prediction"),
        ("Crisp tree vs Mamdani", "provisional_tree_prediction",
         "mamdani_prediction"),
    ):
        valid = predictions.loc[predictions["status"].eq(STATUS_OK)]
        metrics.loc[len(metrics)] = {
            "comparison": comparison,
            "reference": ref, "prediction": target,
            "n_available": len(valid),
            "n_unavailable": len(predictions) - len(valid),
            "development_participants": valid["participant_id"].nunique(),
            "participant_balanced_mae": (
                participant_balanced_mae(valid, reference=ref, prediction=target)
                if not valid.empty else float("nan")
            ),
            "pooled_r2": (
                pooled_r2(valid, reference=ref, prediction=target)
                if not valid.empty else float("nan")
            ),
            "maximum_absolute_error": (
                float((valid[ref] - valid[target]).abs().max())
                if not valid.empty else float("nan")
            ),
            "evaluation": "development, in-sample surrogate fidelity only",
        }
    write_outputs(
        args.output_dir, predictions, activations, metrics, illustration,
        frame=frame, rule_count=len(converted["rules"]),
        stage1_dir=args.stage1_output_dir,
        stage2b_dir=args.stage2b_output_dir,
        mod14_dir=args.mod14_output_dir,
    )


if __name__ == "__main__":
    main()
