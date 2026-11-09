#!/usr/bin/env python3
"""Development-only comparison of two Auditory fuzzy explanations.

Use the established 26-participant development split. Refit K=3 zero-order
FCM–Sugeno for each left-out participant; read existing independently
calculated Mamdani 3-/5-term training and participant-grouped predictions.

Report participant-balanced MAE, pooled R² and strict MOD-14 tolerance accuracy
(|error| < 0.25, 0.50, 0.75 difficulty-stage units). Keep accuracy against the
observed stage separate from fidelity to the original Auditory regression.
Never read or evaluate reserved-test prediction records in this script.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

import MOD_19_fcm_sugeno_stage1 as FCM
import MOD_19_fcm_sugeno_stage2a as STAGE2A
import MOD_19_fcm_sugeno_stage2b as SUGENO

FEATURES = FCM.MOD14.AUDITORY_PREDICTORS
KEYS = ("participant_id", "difficulty_level", "difficulty_stage")
THRESHOLDS = (0.25, 0.50, 0.75)
K = 3


class ComparisonError(ValueError):
    """Inputs do not meet the Auditory development-only comparison contract."""


def check_keys_and_teacher(data: pd.DataFrame, original: pd.DataFrame) -> pd.DataFrame:
    """Demand exactly the original participant-stage identities and teacher."""
    required = {*KEYS, "frozen_mlr_prediction"}
    if not required.issubset(data.columns) or len(data) != len(original):
        raise ComparisonError("Prediction table must contain all 104 development records")
    if data[list(KEYS)].isna().any().any() or data.duplicated(list(KEYS)).any():
        raise ComparisonError("Missing or duplicated participant-stage identity")
    ordered = data.sort_values(list(KEYS)).reset_index(drop=True)
    expected = original.sort_values(list(KEYS)).reset_index(drop=True)
    if not ordered[list(KEYS)].astype(str).equals(
        expected[list(KEYS)].astype(str)
    ):
        raise ComparisonError("Prediction table includes wrong participant/stage identities")
    teacher = pd.to_numeric(ordered["frozen_mlr_prediction"], errors="coerce").to_numpy()
    reference = expected["frozen_mlr_prediction"].to_numpy(dtype=float)
    if not np.isfinite(teacher).all() or not np.allclose(
        teacher, reference, atol=1e-8, rtol=0,
    ):
        raise ComparisonError("Saved original regression predictions differ")
    return ordered


def check_manifest(
    path: Path, source_paths: dict[str, Path], *,
    stage: str, candidate: str,
) -> dict:
    manifest = json.loads(Path(path).read_text(encoding="utf-8"))
    if (manifest.get("module") != "MOD-19"
            or manifest.get("stage") != stage
            or manifest.get("development_participants") != 26
            or manifest.get("development_rows") != 104
            or manifest.get("reserved_test_participants_used",
                            manifest.get("test_participants_used")) != 0):
        raise ComparisonError(f"{candidate}: development-only manifest mismatch")
    prior = manifest.get("source_sha256", {})
    for name, path in source_paths.items():
        if prior.get(name) != FCM.sha256(Path(path)):
            raise ComparisonError(f"{candidate}: source hash changed for {name}")
    return manifest


def grouped_sugeno(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """All fitted transformations learned in 25-person training folds only."""
    FCM.validate_development_frame(frame)
    participant_ids = tuple(sorted(frame["participant_id"].astype(str).unique()))
    prediction_rows, fold_rows = [], []
    for fold, left_out in enumerate(participant_ids, 1):
        holdout = frame["participant_id"].astype(str).eq(left_out)
        train = frame.loc[~holdout].reset_index(drop=True)
        validation = frame.loc[holdout].reset_index(drop=True)
        if (len(train) != 100 or len(validation) != 4
                or train["participant_id"].nunique() != 25
                or validation["participant_id"].nunique() != 1
                or left_out in set(train["participant_id"].astype(str))):
            raise ComparisonError("Participant leakage or incomplete validation fold")
        x_train, maxima = FCM.scale_by_training_max(
            train[list(FEATURES)].to_numpy(dtype=float)
        )
        fitted = FCM.fit_fcm(x_train, clusters=K)
        if not fitted.converged:
            raise ComparisonError(f"FCM did not converge for fold {fold}")
        constants, effective_n = SUGENO.fit_zero_order_constants(
            train["frozen_mlr_prediction"].to_numpy(dtype=float), fitted.memberships
        )
        model = SUGENO.ZeroOrderAuditory(fitted.centres, maxima, constants)
        predictions, memberships, contributions = model.infer(
            validation[list(FEATURES)].to_numpy(dtype=float)
        )
        if (not np.isfinite(predictions).all()
                or not np.allclose(memberships.sum(axis=1), 1, atol=1e-10, rtol=0)
                or not np.allclose(contributions.sum(axis=1), predictions, atol=1e-10)):
            raise ComparisonError("Invalid fold inference output")
        fold_rows.append({
            "fold": fold, "training_participants": 25,
            "validation_participants": 1, "training_observations": 100,
            "validation_observations": 4,
            "fcm_iterations": fitted.iterations,
            "fcm_objective": fitted.objective,
            "k": K, "fuzzifier": FCM.FUZZIFIER,
            "minimum_effective_cluster_n": float(effective_n.min()),
        })
        for row_index, (_, row) in enumerate(validation.iterrows()):
            prediction_rows.append({
                "fold": fold,
                "participant_id": row["participant_id"],
                "difficulty_level": row["difficulty_level"],
                "difficulty_stage": row["difficulty_stage"],
                "frozen_mlr_prediction": float(row["frozen_mlr_prediction"]),
                "zero_order_sugeno_prediction": float(predictions[row_index]),
            })
    predictions_frame = pd.DataFrame(prediction_rows)
    if (len(predictions_frame) != 104
            or predictions_frame["participant_id"].nunique() != 26
            or predictions_frame.duplicated(list(KEYS)).any()):
        raise ComparisonError("Incomplete participant-grouped Sugeno outputs")
    return predictions_frame, pd.DataFrame(fold_rows)


def record_metrics(
    frame: pd.DataFrame, *, method: str, subset: str, prediction: str,
    status: str | None = None,
) -> list[dict]:
    """One row per reference, preserving missing-output coverage explicitly."""
    if prediction not in frame or (status is not None and status not in frame):
        raise ComparisonError(f"Missing {method} prediction or status")
    if status is not None:
        states = frame[status].astype(str)
        if not states.isin(("OK", "Unavailable")).all():
            raise ComparisonError(f"Unrecognised {method} availability status")
        valid = states.eq("OK")
    else:
        valid = pd.Series(True, index=frame.index)
    data = frame.loc[valid].copy()
    values = pd.to_numeric(data[prediction], errors="coerce").to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ComparisonError(f"Available {method} predictions must be finite")
    data[prediction] = values
    rows = []
    for reference_name, reference_col in (
        ("observed_difficulty_stage", "difficulty_stage"),
        ("original_regression_prediction", "frozen_mlr_prediction"),
    ):
        target = data[reference_col].to_numpy(dtype=float)
        if not np.isfinite(target).all():
            raise ComparisonError("Invalid evaluation reference")
        result = {
            "method": method, "evaluation": subset, "reference": reference_name,
            "total_observations": int(len(frame)),
            "available_observations": int(len(data)),
            "unavailable_observations": int(len(frame) - len(data)),
            "available_participants": int(data["participant_id"].nunique()),
            "coverage_percent": float(100 * len(data) / len(frame)),
            "participant_balanced_mae": float("nan"),
            "pooled_r2": float("nan"),
        }
        for t in THRESHOLDS:
            token = f"{t:.2f}".replace(".", "p")
            result[f"accuracy_within_{token}_percent"] = (
                100.0 * float(np.mean(np.abs(target - values) < t))
                if len(data) else float("nan")
            )
        if len(data):
            result["participant_balanced_mae"] = FCM.MOD14.participant_balanced_mae(
                participant_ids=data["participant_id"],
                observed=pd.Series(target), predicted=pd.Series(values),
            )
            result["pooled_r2"] = FCM.MOD14.pooled_r2(
                observed=pd.Series(target), predicted=pd.Series(values),
            )
        rows.append(result)
    return rows


def compare(
    frame: pd.DataFrame, sugeno_training: pd.DataFrame,
    sugeno_grouped: pd.DataFrame, mamdani_training: pd.DataFrame,
    mamdani_grouped: pd.DataFrame,
) -> pd.DataFrame:
    """Compare identical development observations, separately for two references."""
    FCM.validate_development_frame(frame)
    sources = {
        ("FCM–Sugeno", "training"): check_keys_and_teacher(sugeno_training, frame),
        ("FCM–Sugeno", "participant-excluded"): check_keys_and_teacher(sugeno_grouped, frame),
        ("Mamdani", "training"): check_keys_and_teacher(mamdani_training, frame),
        ("Mamdani", "participant-excluded"): check_keys_and_teacher(mamdani_grouped, frame),
    }
    report = []
    for (method, evaluation), table in sources.items():
        table = table.copy()
        if method == "FCM–Sugeno":
            report += record_metrics(
                table, method="FCM–Sugeno (3 clusters)", subset=evaluation,
                prediction="zero_order_sugeno_prediction",
            )
        else:
            for terms in (3, 5):
                report += record_metrics(
                    table, method=f"Mamdani ({terms} output terms)", subset=evaluation,
                    prediction=f"output_{terms}_terms_prediction",
                    status=f"output_{terms}_terms_status",
                )
    baseline = frame.copy()
    baseline["original_prediction"] = baseline["frozen_mlr_prediction"]
    report += record_metrics(
        baseline, method="Original Auditory regression",
        subset="training reference", prediction="original_prediction",
    )[:1]  # Baseline only against the observed stage; teacher self-fit is trivial.
    return pd.DataFrame(report)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--modeling-data", type=Path, required=True)
    parser.add_argument("--holdout-split", type=Path, required=True)
    parser.add_argument("--mod14-output-dir", type=Path, required=True)
    parser.add_argument("--fcm-stage1-dir", type=Path, required=True)
    parser.add_argument("--sugeno-training-dir", type=Path, required=True)
    parser.add_argument("--mamdani-training-dir", type=Path, required=True)
    parser.add_argument("--mamdani-grouped-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise ComparisonError(f"Output already exists: {args.output_dir}")
    sources = {
        "modeling_data.csv": args.modeling_data,
        "participant_holdout_split.csv": args.holdout_split,
        "mod14_regression_coefficients.csv":
            args.mod14_output_dir / "regression_coefficients.csv",
        "mod14_training_predictions.csv":
            args.mod14_output_dir / "training_predictions.csv",
    }
    frame = FCM.prepare_frozen_development(
        modeling_data=pd.read_csv(args.modeling_data),
        holdout_split=pd.read_csv(args.holdout_split),
        regression_coefficients=pd.read_csv(sources["mod14_regression_coefficients.csv"]),
        training_predictions=pd.read_csv(sources["mod14_training_predictions.csv"]),
    )
    STAGE2A.read_verified_stage1(frame, args.fcm_stage1_dir, sources)
    sugeno_manifest = check_manifest(
        args.sugeno_training_dir / "zero_order_manifest.json", sources,
        stage="2B: exploratory zero-order FCM-Sugeno", candidate="Sugeno",
    )
    mamdani_manifest = check_manifest(
        args.mamdani_training_dir / "output_term_manifest.json", sources,
        stage="4A: output vocabulary sensitivity", candidate="Mamdani training",
    )
    grouped_manifest = check_manifest(
        args.mamdani_grouped_dir / "grouped_manifest.json", sources,
        stage="4B: participant-grouped output vocabulary validation",
        candidate="Mamdani grouped",
    )
    if (sugeno_manifest.get("cluster_count") != K
            or mamdani_manifest.get("candidate_output_counts") != [3, 5]
            or grouped_manifest.get("output_vocabulary_counts") != [3, 5]
            or grouped_manifest.get("n_folds") != 26):
        raise ComparisonError("Fuzzy method settings differ from the comparison plan")
    training = pd.read_csv(
        args.sugeno_training_dir / "development_predictions.csv",
        dtype={"participant_id": str},
    )
    # Recheck the training model's saved predictions and constants against Stage 1.
    memberships, stage1_manifest = STAGE2A.read_verified_stage1(
        frame, args.fcm_stage1_dir, sources,
    )
    constants, _ = SUGENO.fit_zero_order_constants(
        frame["frozen_mlr_prediction"].to_numpy(dtype=float), memberships
    )
    saved_rules = pd.read_csv(args.sugeno_training_dir / "zero_order_rules.csv")
    if (saved_rules["cluster"].tolist() != [1, 2, 3]
            or not np.allclose(
                saved_rules["constant_difficulty_stage"].to_numpy(dtype=float),
                constants, atol=1e-10, rtol=0
            )):
        raise ComparisonError("Saved zero-order constants differ from learned values")
    saved_model = SUGENO.model_from_saved_stage1(
        args.fcm_stage1_dir, stage1_manifest, constants
    )
    reconstructed, _, _ = saved_model.infer(
        frame[list(FEATURES)].to_numpy(dtype=float)
    )
    aligned = check_keys_and_teacher(training, frame)
    expected = frame.copy()
    expected["zero_order_sugeno_prediction"] = reconstructed
    expected = expected.sort_values(list(KEYS)).reset_index(drop=True)
    if not np.allclose(
        aligned["zero_order_sugeno_prediction"].to_numpy(dtype=float),
        expected["zero_order_sugeno_prediction"].to_numpy(dtype=float),
        atol=1e-8, rtol=0,
    ):
        raise ComparisonError("Saved zero-order training predictions do not reconstruct")
    sugeno_grouped, folds = grouped_sugeno(frame)
    metrics = compare(
        frame, training, sugeno_grouped,
        pd.read_csv(
            args.mamdani_training_dir / "output_term_predictions.csv",
            dtype={"participant_id": str},
        ),
        pd.read_csv(
            args.mamdani_grouped_dir / "grouped_predictions.csv",
            dtype={"participant_id": str},
        ),
    )
    # Validate that each Mamdani validation fold belongs to one person only.
    mamdani_rows = pd.read_csv(args.mamdani_grouped_dir / "grouped_predictions.csv")
    if (not mamdani_rows.groupby("fold")["participant_id"].nunique().eq(1).all()
            or len(mamdani_rows["fold"].unique()) != 26
            or not mamdani_rows.groupby("fold").size().eq(4).all()):
        raise ComparisonError("Mamdani grouped validation fold identities changed")
    args.output_dir.mkdir(parents=True)
    metrics.to_csv(args.output_dir / "development_metrics.csv", index=False)
    sugeno_grouped.to_csv(args.output_dir / "sugeno_grouped_predictions.csv", index=False)
    folds.to_csv(args.output_dir / "sugeno_fold_diagnostics.csv", index=False)
    manifest = {
        "analysis": "Auditory fuzzy development comparison",
        "split": "26 development participants, 6 reserved participants untouched",
        "development_rows": 104, "reserved_test_participants_used": 0,
        "evaluation": ["training", "participant-excluded"],
        "original_regression_training_boundary": "original regression trained on all 26 development participants",
        "accuracy": "strict abs(reference - predicted) < 0.25, 0.50, 0.75 difficulty-stage units; not percentages of stage range",
        "mae": "participant-balanced", "r2": "pooled",
        "models": ["FCM–Sugeno K=3", "Mamdani 3 terms", "Mamdani 5 terms"],
        "method_selection": "none; this report does not evaluate reserved test",
        "source_sha256": {
            **{name: FCM.sha256(path) for name, path in sources.items()},
            "sugeno_training_manifest": FCM.sha256(
                args.sugeno_training_dir / "zero_order_manifest.json"
            ),
            "mamdani_training_manifest": FCM.sha256(
                args.mamdani_training_dir / "output_term_manifest.json"
            ),
            "mamdani_grouped_manifest": FCM.sha256(
                args.mamdani_grouped_dir / "grouped_manifest.json"
            ),
        },
        "limitations": [
            "Original regression trained on all development participants; grouped results assess fuzzy-model transfer conditional on that reference.",
            "Mamdani Unavailable predictions are preserved; corresponding metrics are calculated only on available rows and include coverage.",
            "Clustering/rule partitions are independent methods; no model selected or modified from test results.",
            "Zero-order constants and centres are fitted anew on each Sugeno training fold.",
            "Therapist interface, clinical adaptation and six-person final test are separate stages.",
        ],
    }
    (args.output_dir / "comparison_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(metrics.to_string(index=False, float_format=lambda x: f"{x:.5g}"))
    print("Reserved-test participants evaluated: 0")


if __name__ == "__main__":
    main()
