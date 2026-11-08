#!/usr/bin/env python3
"""MOD-19 Stage 4B: participant-grouped Auditory surrogate evaluation.

Leave one of the 26 development participants out; refit the provisional
CART and tree-aligned memberships on the other 25; assess 3 vs 5 output
terms and crisp CART on four unseen participant observations. The teacher
is the FROZEN full-development MOD-14 MLR, so this evaluates conditional
surrogate generalisation, NOT independent MLR predictive performance.
No reserved-test observations are used and no model is selected here.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

import MOD_14_direct_linear_regression_evaluation as MOD14
import MOD_19_tree_mamdani as TREE
import MOD_19_fuzzy_rules as BASE
import MOD_19_tree_aligned_memberships as ALIGNED
import MOD_19_mamdani_inference as MAMDANI
import MOD_19_output_term_sensitivity as SENSITIVITY

TREE_PARAMETERS = {"max_depth": 3, "min_samples_leaf": 4, "random_state": 42}
OUTPUT_COUNTS = (3, 5)
STAGES = {(0, 0), (2, 1), (6, 2), (10, 3)}


def verify_development_frame(frame: pd.DataFrame) -> tuple[str, ...]:
    """The only CV population is the frozen 26-person Auditory development set."""
    required = {
        "participant_id", "difficulty_level", "difficulty_stage",
        "frozen_mlr_prediction", *MOD14.AUDITORY_PREDICTORS,
    }
    if not required.issubset(frame.columns) or len(frame) != 104:
        raise BASE.ConversionError("Expected 104 eligible Auditory development rows")
    if frame["participant_id"].isna().any():
        raise BASE.ConversionError("Missing participant identifier")
    count = frame.groupby("participant_id", dropna=False).size()
    if len(count) != 26 or not count.eq(4).all():
        raise BASE.ConversionError("Expected 26 independent participants with four trials each")
    if frame.duplicated(["participant_id", "difficulty_level"]).any():
        raise BASE.ConversionError("Repeated participant-stage observation")
    pairs = set(zip(frame["difficulty_level"], frame["difficulty_stage"]))
    if pairs != STAGES:
        raise BASE.ConversionError("Auditory D0/D2/D6/D10 stage coding differs")
    if not np.isfinite(
        frame[["frozen_mlr_prediction", *MOD14.AUDITORY_PREDICTORS]]
        .to_numpy(dtype=float)
    ).all():
        raise BASE.ConversionError("Non-finite Auditory development inputs")
    return tuple(sorted(frame["participant_id"].astype(str).unique()))


def check_source_provenance(
    *,
    stage4a_manifest: dict, stage1_manifest: dict,
    modeling_data: Path, holdout_split: Path, mod14_dir: Path,
    stage1_dir: Path,
) -> dict[str, str]:
    """Match Stage-4A source hashes; lock the already-studied CART settings."""
    expected = {
        "modeling_data.csv": modeling_data,
        "participant_holdout_split.csv": holdout_split,
        "mod14_regression_coefficients.csv":
            mod14_dir / "regression_coefficients.csv",
        "mod14_training_predictions.csv": mod14_dir / "training_predictions.csv",
        "stage1_manifest.json": stage1_dir / "manifest.json",
    }
    recorded = stage4a_manifest.get("source_sha256", {})
    if (
        stage4a_manifest.get("module") != "MOD-19"
        or stage4a_manifest.get("stage") != "4A: output vocabulary sensitivity"
        or stage4a_manifest.get("model_id") != TREE.MODEL_ID
        or stage4a_manifest.get("development_participants") != 26
        or stage4a_manifest.get("development_rows") != 104
        or stage4a_manifest.get("test_participants_used") != 0
        or stage4a_manifest.get("candidate_output_counts") != [3, 5]
        or stage4a_manifest.get("three_term_reproduces_stage3") is not True
        or stage1_manifest.get("stage") != "1: crisp tree extraction only"
        or stage1_manifest.get("model_id") != TREE.MODEL_ID
        or stage1_manifest.get("development_participants") != 26
        or stage1_manifest.get("test_participants_used") != 0
    ):
        raise BASE.ConversionError("Stage 1/4A provenance contract mismatch")
    if any(stage1_manifest.get("tree_parameters", {}).get(key) != value
           for key, value in TREE_PARAMETERS.items()):
        raise BASE.ConversionError("Original provisional CART settings changed")
    hashes = {name: MAMDANI.file_sha256(path) for name, path in expected.items()}
    for name, actual in hashes.items():
        if recorded.get(name) != actual:
            raise BASE.ConversionError(f"Stage-4A source changed: {name}")
    return hashes


def train_one_fold(
    training: pd.DataFrame,
) -> tuple[object, dict]:
    """Fit ALL learned surrogate components on training participants alone."""
    tree = TREE.fit_development_tree(
        training, max_depth=TREE_PARAMETERS["max_depth"],
        min_samples_leaf=TREE_PARAMETERS["min_samples_leaf"],
    )
    features = training.loc[:, MOD14.AUDITORY_PREDICTORS].to_numpy(dtype=float)
    source = training.loc[:, [
        "participant_id", "difficulty_level", "difficulty_stage",
        "frozen_mlr_prediction",
    ]].copy()
    source["provisional_tree_prediction"] = tree.predict(features)
    converted = ALIGNED.aligned_conversion(
        training, TREE.extract_raw_rules(tree), source,
    )
    return tree, converted


def run_grouped_validation(
    frame: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """26 LOPO folds; validation observations never influence fitted surrogate."""
    participant_ids = verify_development_frame(frame)
    rows = []
    fold_rows = []
    for fold_index, participant_id in enumerate(participant_ids, start=1):
        is_validation = frame["participant_id"].astype(str).eq(participant_id)
        training = frame.loc[~is_validation].copy().reset_index(drop=True)
        validation = frame.loc[is_validation].copy().reset_index(drop=True)
        training_ids = set(training["participant_id"].astype(str))
        validation_ids = set(validation["participant_id"].astype(str))
        if (training_ids & validation_ids or len(training_ids) != 25
                or len(validation_ids) != 1
                or len(training) != 100 or len(validation) != 4):
            raise BASE.ConversionError("Participant leakage or incomplete LOPO fold")
        tree, converted = train_one_fold(training)
        rules = converted["rules"]
        low = float(training["frozen_mlr_prediction"].min())
        high = float(training["frozen_mlr_prediction"].max())
        x = np.linspace(low, high, MAMDANI.OUTPUT_GRID_SIZE)
        output_variants = {}
        for count in OUTPUT_COUNTS:
            vocabulary, centres, labels = SENSITIVITY.output_rule_labels(
                rules, low, high, count
            )
            output_variants[count] = (
                vocabulary, labels,
                SENSITIVITY.calculate_memberships(x, centres),
            )
            details = SENSITIVITY.compression(labels, vocabulary)
            fold_rows.append({
                "fold": fold_index,
                "validation_participant": participant_id,
                "output_terms": count,
                "n_train_participants": len(training_ids),
                "n_validation_participants": len(validation_ids),
                "n_training_rows": len(training),
                "n_validation_rows": len(validation),
                "n_tree_rules": len(rules),
                "train_mlr_min": low,
                "train_mlr_max": high,
                **{name: value for name, value in details.items()
                   if name != "rules_per_consequent"},
                "rules_per_consequent_json":
                    json.dumps(details["rules_per_consequent"]),
            })
        tree_predictions = tree.predict(
            validation.loc[:, MOD14.AUDITORY_PREDICTORS].to_numpy(dtype=float)
        )
        for index, (_, observation) in enumerate(validation.iterrows()):
            activations = np.array([
                MAMDANI.rule_strength(
                    observation, rule, converted["partitions"]
                )[0] for rule in rules
            ], dtype=float)
            row = {
                "fold": fold_index,
                "participant_id": participant_id,
                "difficulty_level": observation["difficulty_level"],
                "difficulty_stage": observation["difficulty_stage"],
                "frozen_mlr_prediction":
                    float(observation["frozen_mlr_prediction"]),
                "crisp_tree_prediction": float(tree_predictions[index]),
                "active_rules": int(np.count_nonzero(activations > 0)),
                "maximum_rule_activation": float(activations.max()),
                "inputs_outside_training_range": int(sum(
                    not (
                        partition.minimum <= float(observation[feature])
                        <= partition.maximum
                    ) for feature, partition in converted["partitions"].items()
                )),
            }
            for count in OUTPUT_COUNTS:
                vocabulary, labels, memberships = output_variants[count]
                predicted, status, area = SENSITIVITY.infer_with_labels(
                    activations, labels, vocabulary, x, memberships,
                )
                row[f"output_{count}_terms_prediction"] = predicted
                row[f"output_{count}_terms_status"] = status
                row[f"output_{count}_terms_area"] = area
            rows.append(row)
    predictions = pd.DataFrame(rows)
    fold_diagnostics = pd.DataFrame(fold_rows)
    if (len(predictions) != 104
            or len(fold_diagnostics) != 26 * len(OUTPUT_COUNTS)
            or predictions.duplicated(["participant_id", "difficulty_level"]).any()
            or predictions["participant_id"].nunique() != 26):
        raise BASE.ConversionError("Missing or duplicated grouped validation rows")

    definitions = (
        ("crisp_tree", "crisp_tree_prediction", None),
        ("output_3_terms", "output_3_terms_prediction",
         "output_3_terms_status"),
        ("output_5_terms", "output_5_terms_prediction",
         "output_5_terms_status"),
    )
    metrics = []
    for label, column, status in definitions:
        valid = (
            predictions[status].eq(MAMDANI.STATUS_OK)
            if status else pd.Series(True, index=predictions.index)
        )
        subset = predictions.loc[valid]
        metrics.append({
            "model": label,
            "evaluation": "leave-one-participant-out development surrogate fidelity",
            "n_available": int(valid.sum()),
            "n_unavailable": int((~valid).sum()),
            "validation_participants": int(subset["participant_id"].nunique()),
            "coverage_percent": float(100 * valid.mean()),
            "participant_balanced_mae_vs_frozen_mlr": (
                MAMDANI.participant_balanced_mae(
                    subset, reference="frozen_mlr_prediction", prediction=column
                ) if not subset.empty else float("nan")
            ),
            "pooled_r2_vs_frozen_mlr": (
                MAMDANI.pooled_r2(
                    subset, reference="frozen_mlr_prediction", prediction=column
                ) if not subset.empty else float("nan")
            ),
            "max_abs_error_vs_frozen_mlr": (
                float((subset["frozen_mlr_prediction"]
                       - subset[column]).abs().max())
                if not subset.empty else float("nan")
            ),
        })
    participant_errors = predictions[["participant_id"]].drop_duplicates().copy()
    for label, column, status in definitions:
        errors = (predictions["frozen_mlr_prediction"] - predictions[column]).abs()
        if status:
            errors = errors.where(predictions[status].eq(MAMDANI.STATUS_OK))
        by_person = errors.groupby(predictions["participant_id"]).mean()
        participant_errors[f"{label}_mae"] = (
            participant_errors["participant_id"].map(by_person)
        )
        if status:
            counts = predictions[status].eq(MAMDANI.STATUS_OK).groupby(
                predictions["participant_id"]
            ).sum()
            participant_errors[f"{label}_available"] = (
                participant_errors["participant_id"].map(counts)
            )
    return predictions, pd.DataFrame(metrics), fold_diagnostics, participant_errors


def write_results(
    output_dir: Path, predictions: pd.DataFrame, metrics: pd.DataFrame,
    folds: pd.DataFrame, participant_errors: pd.DataFrame,
    *, source_hashes: dict[str, str], stage4a_manifest: Path,
) -> None:
    if output_dir.exists() and (
        not output_dir.is_dir() or any(output_dir.iterdir())
    ):
        raise BASE.ConversionError(f"Output already exists: {output_dir}")
    if (len(predictions) != 104 or len(metrics) != 3
            or len(folds) != 52 or len(participant_errors) != 26):
        raise BASE.ConversionError("Incomplete LOPO validation outputs")
    output_dir.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(output_dir / "grouped_predictions.csv", index=False)
    metrics.to_csv(output_dir / "grouped_fidelity.csv", index=False)
    folds.to_csv(output_dir / "fold_diagnostics.csv", index=False)
    participant_errors.to_csv(
        output_dir / "participant_error_comparison.csv", index=False
    )
    manifest = {
        "module": "MOD-19",
        "stage": "4B: participant-grouped output vocabulary validation",
        "model_id": TREE.MODEL_ID,
        "evaluation": "LOPO development surrogate fidelity to frozen MLR teacher",
        "teacher_training_boundary":
            "frozen MOD-14 teacher originally fitted on all 26 development participants",
        "not_equivalent_to": "independent validation of MLR prediction accuracy",
        "development_participants": 26,
        "development_rows": 104,
        "reserved_test_participants_used": 0,
        "n_folds": 26,
        "training_participants_per_fold": 25,
        "validation_participants_per_fold": 1,
        "training_rows_per_fold": 100,
        "validation_rows_per_fold": 4,
        "tree_parameters": TREE_PARAMETERS,
        "output_vocabulary_counts": list(OUTPUT_COUNTS),
        "fold_fitting": [
            "CART trained against frozen MLR prediction using fold-training rows",
            "CART paths and tree thresholds extracted from that fold tree",
            "All input feature partitions learned from fold-training rows",
            "Output centres calibrated to fold-training MLR min/max",
            "MIN-AND MAX-OR MIN implication MAX accumulation centroid unchanged",
        ],
        "source_sha256": {
            **source_hashes,
            "stage4a_manifest.json": MAMDANI.file_sha256(stage4a_manifest),
        },
        "limitations": [
            "Frozen teacher includes validation participants in its original fit;",
            "LOPO evaluates surrogate transfer conditional on that teacher only.",
            "Fixed provisional CART settings and 3/5 vocabulary; not selected here.",
            "No independent reserved-test inference; no clinical adaptation claim.",
            "An unavailable fuzzy output is preserved, never imputed.",
            "Metrics on available observations require coverage alongside fidelity.",
        ],
    }
    (output_dir / "grouped_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--modeling-data", type=Path, required=True)
    parser.add_argument("--holdout-split", type=Path, required=True)
    parser.add_argument("--mod14-output-dir", type=Path, required=True)
    parser.add_argument("--stage1-output-dir", type=Path, required=True)
    parser.add_argument("--stage4a-output-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    destination = args.output_dir
    if destination.exists() and (
        not destination.is_dir() or any(destination.iterdir())
    ):
        raise BASE.ConversionError(f"Output already exists: {destination}")
    stage1_manifest = json.loads(
        (args.stage1_output_dir / "manifest.json").read_text(encoding="utf-8")
    )
    stage4a_path = args.stage4a_output_dir / "output_term_manifest.json"
    stage4a_manifest = json.loads(stage4a_path.read_text(encoding="utf-8"))
    hashes = check_source_provenance(
        stage4a_manifest=stage4a_manifest,
        stage1_manifest=stage1_manifest,
        modeling_data=args.modeling_data,
        holdout_split=args.holdout_split,
        mod14_dir=args.mod14_output_dir,
        stage1_dir=args.stage1_output_dir,
    )
    frame = TREE.prepare_development_teacher(
        modeling_data=pd.read_csv(args.modeling_data),
        holdout_split=pd.read_csv(args.holdout_split),
        regression_coefficients=pd.read_csv(
            args.mod14_output_dir / "regression_coefficients.csv"
        ),
        training_predictions=pd.read_csv(
            args.mod14_output_dir / "training_predictions.csv"
        ),
    )
    predictions, metrics, folds, paired = run_grouped_validation(frame)
    write_results(
        destination, predictions, metrics, folds, paired,
        source_hashes=hashes, stage4a_manifest=stage4a_path,
    )


if __name__ == "__main__":
    main()
