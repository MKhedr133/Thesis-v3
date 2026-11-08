#!/usr/bin/env python3
"""MOD-19 Stage 4A: fixed 3-vs-5 output-term sensitivity experiment.

Only the output vocabulary, membership centres and resulting linguistic
consequents differ. The frozen Auditory MLR, crisp tree, Stage-2B input
partitions, rule antecedents, MIN/MAX operators and centroid are unchanged.

Descriptive IN-SAMPLE development diagnostics; NOT participant-grouped
validation, optimisation, or an evaluation on six reserved test participants.
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

import MOD_19_tree_mamdani as MOD19
import MOD_19_fuzzy_rules as BASE
import MOD_19_tree_aligned_memberships as ALIGNED
import MOD_19_mamdani_inference as MAMDANI


TERM_COUNTS = (3, 5)


def output_rule_labels(
    rules: list[dict], low: float, high: float, term_count: int,
) -> tuple[tuple[str, ...], np.ndarray, list[str]]:
    """Re-label only the frozen numerical tree-leaf predictions."""
    if term_count not in TERM_COUNTS or not np.isfinite([low, high]).all() or high <= low:
        raise BASE.ConversionError("Invalid output vocabulary or development range")
    vocabulary = BASE.TERMS[term_count]
    centres = np.linspace(low, high, term_count)
    labels = []
    for rule in rules:
        value = float(rule["tree_output"])
        if not np.isfinite(value) or not low - 1e-8 <= value <= high + 1e-8:
            raise BASE.ConversionError("Tree leaf is outside the frozen output universe")
        weights = BASE.membership(value, centres)
        labels.append(vocabulary[int(np.argmax(weights))])
    return vocabulary, centres, labels


def compression(labels: list[str], vocabulary: tuple[str, ...]) -> dict:
    counts = {term: labels.count(term) for term in vocabulary}
    return {
        "distinct_consequents_used": sum(value > 0 for value in counts.values()),
        "rules_per_consequent": counts,
        "pairs_sharing_consequent": sum(count * (count - 1) // 2
                                        for count in counts.values()),
        "largest_consequent_group": max(counts.values()),
    }


def calculate_memberships(
    x: np.ndarray, centres: np.ndarray,
) -> np.ndarray:
    """Use the same linear triangular/shoulder evaluator as Stage 3."""
    return np.array([BASE.membership(float(value), centres) for value in x])


def infer_with_labels(
    strengths: np.ndarray,
    labels: list[str],
    vocabulary: tuple[str, ...],
    x: np.ndarray,
    membership_grid: np.ndarray,
) -> tuple[float, str, float]:
    """MIN implication, MAX accumulation, same Stage-3 centroid integration."""
    if (len(strengths) != len(labels) or membership_grid.shape != (len(x), len(vocabulary))
            or not np.isfinite(strengths).all()
            or np.any((strengths < 0) | (strengths > 1))):
        raise BASE.ConversionError("Invalid output comparison rule strengths")
    aggregated = np.zeros(len(x), dtype=float)
    for strength, term in zip(strengths, labels):
        if term not in vocabulary:
            raise BASE.ConversionError(f"Unexpected output term: {term}")
        if strength > 0:
            aggregated = np.maximum(
                aggregated,
                np.minimum(float(strength),
                           membership_grid[:, vocabulary.index(term)]),
            )
    prediction, area = MAMDANI.centroid(x, aggregated)
    return prediction, (
        MAMDANI.STATUS_OK if np.isfinite(prediction)
        else MAMDANI.STATUS_UNAVAILABLE
    ), area


def compute_comparison(
    frame: pd.DataFrame, converted: dict, crisp_predictions: np.ndarray,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    """No refitting, no parameter selection and no data-driven winner."""
    rules = converted["rules"]
    if not rules or len(frame) != len(crisp_predictions):
        raise BASE.ConversionError("Incomplete fixed development inputs")
    low, high = [float(v) for v in
                 (converted["output_centres"][0],
                  converted["output_centres"][-1])]
    x = np.linspace(low, high, MAMDANI.OUTPUT_GRID_SIZE)
    vocabularies = {}
    strengths = []
    for _, observation in frame.iterrows():
        strengths.append([
            MAMDANI.rule_strength(
                observation, rule, converted["partitions"]
            )[0] for rule in rules
        ])
    strengths = np.asarray(strengths, dtype=float)
    if not np.isfinite(strengths).all() or (strengths.max(axis=1) <= 0).any():
        raise BASE.ConversionError("Zero/invalid development rule coverage")
    result = frame.loc[:, [
        "participant_id", "difficulty_level", "difficulty_stage",
        "frozen_mlr_prediction",
    ]].copy().reset_index(drop=True)
    result["crisp_tree_prediction"] = np.asarray(
        crisp_predictions, dtype=float
    )
    mapping_records = []
    metrics = []
    for term_count in TERM_COUNTS:
        vocabulary, centres, labels = output_rule_labels(
            rules, low, high, term_count
        )
        grid = calculate_memberships(x, centres)
        predicted = [
            infer_with_labels(row, labels, vocabulary, x, grid)
            for row in strengths
        ]
        key = f"output_{term_count}_terms"
        result[f"{key}_prediction"] = [v[0] for v in predicted]
        result[f"{key}_status"] = [v[1] for v in predicted]
        result[f"{key}_output_area"] = [v[2] for v in predicted]
        valid = result[f"{key}_status"].eq(MAMDANI.STATUS_OK)
        usable = result.loc[valid]
        numerical = f"{key}_prediction"
        metrics.append({
            "configuration": key,
            "number_output_terms": term_count,
            "n_available": int(valid.sum()),
            "n_unavailable": int((~valid).sum()),
            "development_participants": int(usable["participant_id"].nunique()),
            "participant_balanced_mae_vs_mlr": (
                MAMDANI.participant_balanced_mae(
                    usable, reference="frozen_mlr_prediction",
                    prediction=numerical
                ) if not usable.empty else float("nan")
            ),
            "pooled_r2_vs_mlr": (
                MAMDANI.pooled_r2(
                    usable, reference="frozen_mlr_prediction",
                    prediction=numerical
                ) if not usable.empty else float("nan")
            ),
            "participant_balanced_mae_vs_tree": (
                MAMDANI.participant_balanced_mae(
                    usable, reference="crisp_tree_prediction",
                    prediction=numerical
                ) if not usable.empty else float("nan")
            ),
            "maximum_absolute_error_vs_mlr": (
                float((usable[numerical] -
                       usable["frozen_mlr_prediction"]).abs().max())
                if not usable.empty else float("nan")
            ),
            "evaluation": "development in-sample surrogate fidelity only",
        })
        for rule, label in zip(rules, labels):
            mapping_records.append({
                "configuration": key,
                "source_leaf_ids": json.dumps(rule["source_leaf_ids"]),
                "source_tree_prediction": rule["tree_output"],
                "development_leaf_rows": rule["sample_count"],
                "output_consequent": label,
                "output_centres_json": json.dumps(centres.tolist()),
                "antecedent_or_groups_json": json.dumps(rule["antecedent"]),
            })
        vocabularies[key] = {
            "vocabulary": list(vocabulary),
            "centres": centres.tolist(),
            "compression": compression(labels, vocabulary),
        }
    result["absolute_error_3_terms"] = (
        result["frozen_mlr_prediction"] -
        result["output_3_terms_prediction"]
    ).abs()
    result["absolute_error_5_terms"] = (
        result["frozen_mlr_prediction"] -
        result["output_5_terms_prediction"]
    ).abs()
    return (
        result, pd.DataFrame(metrics), pd.DataFrame(mapping_records),
        vocabularies
    )


def verify_stage3_baseline(
    comparison: pd.DataFrame, saved: pd.DataFrame,
) -> None:
    """Three-term comparison must exactly reproduce saved Stage-3 inference."""
    if len(comparison) != len(saved):
        raise BASE.ConversionError("Stage-3 baseline row count differs")
    for column in ("participant_id", "difficulty_level", "difficulty_stage"):
        if (column not in saved
                or not np.array_equal(
                    comparison[column].to_numpy(), saved[column].to_numpy()
                )):
            raise BASE.ConversionError(f"Stage-3 baseline row order differs: {column}")
    for new, original in (
        ("frozen_mlr_prediction", "frozen_mlr_prediction"),
        ("crisp_tree_prediction", "provisional_tree_prediction"),
        ("output_3_terms_prediction", "mamdani_prediction"),
        ("output_3_terms_output_area", "output_area"),
    ):
        if original not in saved or not np.allclose(
            comparison[new].to_numpy(dtype=float),
            saved[original].to_numpy(dtype=float),
            rtol=0, atol=1e-8, equal_nan=True,
        ):
            raise BASE.ConversionError(
                f"Three-term baseline does not reproduce Stage 3: {original}"
            )
    if ("status" not in saved or not np.array_equal(
        comparison["output_3_terms_status"].to_numpy(),
        saved["status"].to_numpy()
    )):
        raise BASE.ConversionError("Stage-3 baseline status mismatch")


def source_paths(
    mod14: Path, stage1: Path, stage2b: Path,
) -> dict[str, Path]:
    return {
        "mod14_regression_coefficients.csv": mod14 / "regression_coefficients.csv",
        "mod14_training_predictions.csv": mod14 / "training_predictions.csv",
        "stage1_manifest.json": stage1 / "manifest.json",
        "stage1_numerical_rules.csv": stage1 / "numerical_rules.csv",
        "stage1_development_predictions.csv": stage1 / "development_predictions.csv",
        "stage2b_manifest.json": stage2b / "tree_aligned_manifest.json",
        "stage2b_fuzzy_rules.csv": stage2b / "tree_aligned_fuzzy_rules.csv",
        "stage2b_partitions.csv": stage2b / "tree_aligned_membership_partitions.csv",
    }


def verify_stage3_provenance(
    saved_manifest: dict, paths: dict[str, Path],
    *, development_rows: int,
) -> None:
    if (saved_manifest.get("module") != "MOD-19"
            or saved_manifest.get("stage") != "3: fixed Mamdani development baseline"
            or saved_manifest.get("model_id") != MOD19.MODEL_ID
            or saved_manifest.get("development_participants") != 26
            or saved_manifest.get("development_rows") != development_rows
            or saved_manifest.get("test_participants_used") != 0
            or set(saved_manifest.get("source_sha256", {})) != set(paths)):
        raise BASE.ConversionError("Stage-3 baseline manifest mismatch")
    for label, path in paths.items():
        if MAMDANI.file_sha256(path) != saved_manifest["source_sha256"][label]:
            raise BASE.ConversionError(f"Stage-3 provenance changed: {label}")


def save_comparison(
    output_dir: Path, predictions: pd.DataFrame, metrics: pd.DataFrame,
    mappings: pd.DataFrame, vocabularies: dict,
    *, provenance: dict[str, Path],
    modeling_data: Path, holdout_split: Path, stage3: Path,
) -> None:
    if output_dir.exists() and (
        not output_dir.is_dir() or any(output_dir.iterdir())
    ):
        raise BASE.ConversionError(f"Output already exists: {output_dir}")
    if (len(predictions) != 104 or len(metrics) != 2
            or len(mappings) != 16
            or predictions["participant_id"].nunique() != 26):
        raise BASE.ConversionError("Incomplete MOD-19 Stage 4A output")
    output_dir.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(output_dir / "output_term_predictions.csv", index=False)
    metrics.to_csv(output_dir / "output_term_fidelity.csv", index=False)
    mappings.to_csv(output_dir / "consequent_mapping.csv", index=False)

    fig, ax = plt.subplots(figsize=(9.5, 5.2))
    low = vocabularies["output_3_terms"]["centres"][0]
    high = vocabularies["output_3_terms"]["centres"][-1]
    x = np.linspace(low, high, 500)
    for count in TERM_COUNTS:
        info = vocabularies[f"output_{count}_terms"]
        centres = np.asarray(info["centres"], dtype=float)
        memberships = calculate_memberships(x, centres)
        for j, label in enumerate(info["vocabulary"]):
            ax.plot(x, memberships[:, j], linestyle="-" if count == 3 else "--",
                    label=f"{count} terms: {label}", alpha=0.8)
    ax.set_xlabel("Frozen MLR continuous prediction (stage points)")
    ax.set_ylabel("Output membership")
    ax.set_title("MOD-19: fixed 3-term vs 5-term output partitions")
    ax.legend(ncol=2, fontsize=8)
    fig.tight_layout()
    fig.savefig(output_dir / "output_membership_comparison.png", dpi=160)
    plt.close(fig)

    sources = {
        **{name: MAMDANI.file_sha256(path) for name, path in provenance.items()},
        "modeling_data.csv": MAMDANI.file_sha256(modeling_data),
        "participant_holdout_split.csv": MAMDANI.file_sha256(holdout_split),
        "stage3_mamdani_manifest.json":
            MAMDANI.file_sha256(stage3 / "mamdani_manifest.json"),
        "stage3_mamdani_predictions.csv":
            MAMDANI.file_sha256(stage3 / "mamdani_predictions.csv"),
    }
    manifest = {
        "module": "MOD-19", "stage": "4A: output vocabulary sensitivity",
        "model_id": MOD19.MODEL_ID,
        "status": "descriptive in-sample; not model selection",
        "development_participants": 26, "development_rows": 104,
        "test_participants_used": 0,
        "candidate_output_counts": list(TERM_COUNTS),
        "fixed": [
            "frozen MOD-14 Auditory MLR predictions",
            "eight CART source rules, thresholds and tree outputs",
            "Stage-2B tree-aligned input partitions and antecedents",
            "MAX-OR, MIN-AND, MIN clipping, MAX aggregation",
            f"{MAMDANI.OUTPUT_GRID_SIZE}-point centroid integration",
        ],
        "changed": [
            "number of evenly spaced output membership functions",
            "numerical tree leaf to linguistic consequent mapping",
        ],
        "three_term_reproduces_stage3": True,
        "output_vocabulary": vocabularies,
        "source_sha256": sources,
        "limitations": [
            "Results describe same development observations used to fit CART.",
            "Not participant-grouped cross-validation; no final choice made.",
            "Only output vocabulary differs; output term labels are relative.",
            "No reserved-test observations used; later reserved evaluation exploratory.",
            "No therapist preference, clinical decision or adaptation validity tested.",
        ],
    }
    (output_dir / "output_term_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--modeling-data", type=Path, required=True)
    parser.add_argument("--holdout-split", type=Path, required=True)
    parser.add_argument("--mod14-output-dir", type=Path, required=True)
    parser.add_argument("--stage1-output-dir", type=Path, required=True)
    parser.add_argument("--stage2b-output-dir", type=Path, required=True)
    parser.add_argument("--stage3-output-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    stage1 = json.loads((args.stage1_output_dir / "manifest.json").read_text())
    stage2b = json.loads(
        (args.stage2b_output_dir / "tree_aligned_manifest.json").read_text()
    )
    stage3 = json.loads(
        (args.stage3_output_dir / "mamdani_manifest.json").read_text()
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
    raw_rules = BASE.load_crisp_rules(
        pd.read_csv(args.stage1_output_dir / "numerical_rules.csv")
    )
    original = pd.read_csv(
        args.stage1_output_dir / "development_predictions.csv"
    )
    converted = MAMDANI.verify_sources(
        stage1_dir=args.stage1_output_dir,
        stage2b_dir=args.stage2b_output_dir,
        stage1_manifest=stage1, stage2b_manifest=stage2b,
        raw_rules=raw_rules, stage1_predictions=original, frame=frame,
    )
    paths = source_paths(
        args.mod14_output_dir, args.stage1_output_dir, args.stage2b_output_dir
    )
    verify_stage3_provenance(stage3, paths, development_rows=len(frame))
    result, metrics, mappings, categories = compute_comparison(
        frame, converted,
        original["provisional_tree_prediction"].to_numpy(dtype=float),
    )
    verify_stage3_baseline(
        result, pd.read_csv(args.stage3_output_dir / "mamdani_predictions.csv")
    )
    save_comparison(
        args.output_dir, result, metrics, mappings, categories,
        provenance=paths, modeling_data=args.modeling_data,
        holdout_split=args.holdout_split, stage3=args.stage3_output_dir,
    )


if __name__ == "__main__":
    main()
