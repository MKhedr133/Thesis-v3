#!/usr/bin/env python3
"""MOD-19 FCM–Sugeno alternative: Stage 1, Auditory input clustering only.

Construct deterministic fuzzy C-means clusters from the *development-only*
Auditory features. Validate frozen MOD-14 MLR targets but do not use those
targets in clustering. This stage does not fit Sugeno rules or make predictions.
It is a thesis-specific adaptation, not an exact replication of Kusumadewi et
al. (2023). The number of clusters is a required exploratory input, not chosen
from reserved-test results.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

import MOD_14_direct_linear_regression_evaluation as MOD14

MODEL_ID = "auditory"
FUZZIFIER = 2.0
SEED = 42
MAX_ITER = 500
TOLERANCE = 1e-8


class FcmStage1Error(ValueError):
    """Invalid development input, clustering setting, or output contract."""


@dataclass(frozen=True)
class FcmResult:
    centres: np.ndarray
    memberships: np.ndarray
    objective: float
    iterations: int
    converged: bool


def prepare_frozen_development(
    modeling_data: pd.DataFrame,
    holdout_split: pd.DataFrame,
    regression_coefficients: pd.DataFrame,
    training_predictions: pd.DataFrame,
) -> pd.DataFrame:
    """Reconstruct and verify the frozen teacher; never return reserved rows."""
    development, _, dev_ids, _ = MOD14.prepare_split_data(
        modeling_data=modeling_data, holdout_split=holdout_split,
    )
    frame = MOD14.prepare_model_frame(development, model_id=MODEL_ID)
    predictors = MOD14.build_model_registry()[MODEL_ID].predictors
    if "participant_id" not in frame.columns:
        raise FcmStage1Error("Auditory participant identifier is missing")
    if set(frame["participant_id"]) != set(dev_ids):
        raise FcmStage1Error("Auditory development participants differ from frozen split")
    if not {"model", "predictor", "coefficient"}.issubset(regression_coefficients.columns):
        raise FcmStage1Error("Missing MOD-14 coefficient columns")
    source = regression_coefficients.loc[
        regression_coefficients["model"].eq(MODEL_ID)
    ]
    expected_names = ("Intercept", *predictors)
    if (len(source) != len(expected_names)
            or source["predictor"].duplicated().any()
            or set(source["predictor"]) != set(expected_names)):
        raise FcmStage1Error("Frozen Auditory MLR coefficient schema changed")
    coefficients = pd.Series(
        pd.to_numeric(source["coefficient"], errors="coerce").to_numpy(dtype=float),
        index=source["predictor"].tolist(),
    ).reindex(expected_names)
    if not np.isfinite(coefficients.to_numpy(dtype=float)).all():
        raise FcmStage1Error("Non-finite MLR coefficients")
    teacher = MOD14.LinearModelFit(
        model_id=MODEL_ID, predictors=predictors, coefficients=coefficients,
    ).predict(frame).to_numpy(dtype=float)
    required = {"model", "participant_id", "predicted_difficulty_stage"}
    if not required.issubset(training_predictions.columns):
        raise FcmStage1Error("Missing MOD-14 prediction columns")
    recorded = training_predictions.loc[
        training_predictions["model"].eq(MODEL_ID)
    ].reset_index(drop=True)
    if (len(recorded) != len(frame)
            or recorded["participant_id"].astype(str).tolist()
            != frame["participant_id"].tolist()):
        raise FcmStage1Error("Frozen Auditory MLR prediction identity/order mismatch")
    stored = pd.to_numeric(
        recorded["predicted_difficulty_stage"], errors="coerce"
    ).to_numpy(dtype=float)
    if (not np.isfinite(stored).all()
            or not np.allclose(teacher, stored, atol=1e-8, rtol=0)):
        raise FcmStage1Error("Frozen Auditory MLR prediction reconstruction failed")
    result = frame.copy()
    result["frozen_mlr_prediction"] = teacher
    validate_development_frame(result)
    return result


def validate_development_frame(frame: pd.DataFrame) -> None:
    """Reject incomplete participants, stage recoding and non-Auditory features."""
    required = {
        "participant_id", "difficulty_level", "difficulty_stage",
        "frozen_mlr_prediction", *MOD14.AUDITORY_PREDICTORS,
    }
    if not required.issubset(frame.columns) or len(frame) != 104:
        raise FcmStage1Error("Expected 104 complete Auditory development rows")
    count = frame.groupby("participant_id", dropna=False).size()
    if len(count) != 26 or not count.eq(4).all() or frame["participant_id"].isna().any():
        raise FcmStage1Error("Expected four rows for each of 26 development participants")
    if frame.duplicated(["participant_id", "difficulty_level"]).any():
        raise FcmStage1Error("Duplicate participant-stage observation")
    expected = {(0, 0), (2, 1), (6, 2), (10, 3)}
    if (set(zip(frame["difficulty_level"], frame["difficulty_stage"])) != expected
            or any(set(zip(group["difficulty_level"], group["difficulty_stage"])) != expected
                   for _, group in frame.groupby("participant_id"))):
        raise FcmStage1Error("D0/D2/D6/D10 stage coding differs")
    if not np.isfinite(
        frame[[*MOD14.AUDITORY_PREDICTORS, "frozen_mlr_prediction"]]
        .to_numpy(dtype=float)
    ).all():
        raise FcmStage1Error("Non-finite Auditory predictor or frozen MLR output")


def scale_by_training_max(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Paper-inspired x/max(x), learned from training rows only, no centring."""
    data = np.asarray(values, dtype=float)
    if (data.ndim != 2 or not data.size or not np.isfinite(data).all()
            or np.any(data < 0)):
        raise FcmStage1Error("Input must be a complete nonnegative numeric matrix")
    maxima = np.max(data, axis=0)
    if np.any(maxima <= 0):
        raise FcmStage1Error("Each feature needs a positive training maximum")
    return data / maxima, maxima


def membership_from_centres(
    x: np.ndarray, centres: np.ndarray, *, fuzzifier: float = FUZZIFIER,
) -> np.ndarray:
    """Standard FCM distance memberships, including exact-centre ties."""
    data = np.asarray(x, dtype=float)
    prototypes = np.asarray(centres, dtype=float)
    if (data.ndim != 2 or prototypes.ndim != 2
            or data.shape[1] != prototypes.shape[1]
            or prototypes.shape[0] < 2
            or not np.isfinite(data).all() or not np.isfinite(prototypes).all()
            or not np.isfinite(fuzzifier) or fuzzifier <= 1):
        raise FcmStage1Error("Invalid FCM input/centres or fuzzifier")
    dist2 = np.sum((data[:, None, :] - prototypes[None, :, :]) ** 2, axis=2)
    memberships = np.zeros_like(dist2)
    exact = dist2 == 0
    exact_rows = exact.any(axis=1)
    if exact_rows.any():
        hits = exact[exact_rows]
        memberships[exact_rows] = hits / hits.sum(axis=1, keepdims=True)
    if (~exact_rows).any():
        safe = dist2[~exact_rows]
        # Each nonzero row is rescaled before exponentiation for stability.
        relative = safe / safe.min(axis=1, keepdims=True)
        weights = relative ** (-1.0 / (fuzzifier - 1.0))
        memberships[~exact_rows] = weights / weights.sum(axis=1, keepdims=True)
    return memberships


def fit_fcm(
    x: np.ndarray, *, clusters: int, seed: int = SEED,
    fuzzifier: float = FUZZIFIER, max_iter: int = MAX_ITER,
    tolerance: float = TOLERANCE,
) -> FcmResult:
    """Fit conventional FCM to scaled input features, without target labels."""
    data = np.asarray(x, dtype=float)
    if (data.ndim != 2 or len(data) < 3 or not np.isfinite(data).all()
            or not np.isfinite(fuzzifier) or fuzzifier <= 1):
        raise FcmStage1Error("FCM requires finite rows and fuzzifier > 1")
    if (isinstance(clusters, bool) or not isinstance(clusters, int)
            or clusters < 2 or clusters > len(np.unique(data, axis=0))):
        raise FcmStage1Error("Cluster count must be 2..number of distinct training rows")
    if (not isinstance(seed, int) or isinstance(seed, bool)
            or not isinstance(max_iter, int) or max_iter < 1
            or not np.isfinite(tolerance) or tolerance <= 0):
        raise FcmStage1Error("Invalid FCM algorithm parameters")
    rng = np.random.default_rng(seed)
    u = rng.dirichlet(np.ones(clusters), size=len(data))
    converged = False
    iterations = max_iter
    for iteration in range(1, max_iter + 1):
        power = u ** fuzzifier
        denom = power.sum(axis=0)
        if np.any(denom <= 0):
            raise FcmStage1Error("Empty fuzzy cluster")
        centres = (power.T @ data) / denom[:, None]
        updated = membership_from_centres(data, centres, fuzzifier=fuzzifier)
        difference = float(np.max(np.abs(updated - u)))
        u = updated
        if difference <= tolerance:
            converged = True
            iterations = iteration
            break
    power = u ** fuzzifier
    centres = (power.T @ data) / power.sum(axis=0)[:, None]
    memberships = membership_from_centres(data, centres, fuzzifier=fuzzifier)
    dist2 = np.sum((data[:, None, :] - centres[None, :, :]) ** 2, axis=2)
    objective = float(np.sum((memberships ** fuzzifier) * dist2))
    if not np.isfinite(objective):
        raise FcmStage1Error("Non-finite FCM objective")
    return FcmResult(centres, memberships, objective, iterations, converged)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_outputs(
    output_dir: Path, frame: pd.DataFrame, fcm: FcmResult,
    feature_maxima: np.ndarray, *, clusters: int,
    sources: dict[str, Path],
) -> None:
    """Write local-only diagnostics, refusing to overwrite existing outputs."""
    output_dir = Path(output_dir)
    if output_dir.exists():
        raise FcmStage1Error(f"Output directory already exists: {output_dir}")
    validate_development_frame(frame)
    features = MOD14.AUDITORY_PREDICTORS
    if (fcm.centres.shape != (clusters, len(features))
            or fcm.memberships.shape != (len(frame), clusters)
            or not fcm.converged
            or not np.isfinite(fcm.centres).all()
            or not np.allclose(fcm.memberships.sum(axis=1), 1.0, atol=1e-10)
            or feature_maxima.shape != (len(features),)):
        raise FcmStage1Error("Incomplete, unconverged or invalid FCM fit")
    centres = pd.DataFrame(fcm.centres, columns=features)
    centres.insert(0, "cluster", np.arange(1, clusters + 1))
    unscaled = fcm.centres * feature_maxima
    for idx, name in enumerate(features):
        centres[name + "_original_units"] = unscaled[:, idx]
    membership = frame[["participant_id", "difficulty_level", "difficulty_stage"]].copy()
    for c in range(clusters):
        membership[f"cluster_{c + 1}_membership"] = fcm.memberships[:, c]
    fingerprints = {label: sha256(path) for label, path in sources.items()}
    manifest = {
        "module": "MOD-19", "method": "FCM-Sugeno alternative", "stage": "1: Auditory FCM clustering only",
        "status": "exploratory development-only; NO Sugeno inference or model selection",
        "target": "frozen MOD-14 Auditory MLR (verified, not used for input clustering)",
        "model": MODEL_ID, "features": list(features),
        "development_participants": 26, "development_rows": 104,
        "reserved_test_participants_used": 0,
        "fcm": {"clusters": clusters, "fuzzifier": FUZZIFIER,
                "seed": SEED, "max_iter": MAX_ITER, "tolerance": TOLERANCE,
                "iterations": fcm.iterations, "converged": fcm.converged,
                "objective": fcm.objective},
        "scaling": "each nonnegative input divided by training-development maximum",
        "training_feature_maxima": dict(zip(features, feature_maxima.astype(float).tolist())),
        "source_sha256": fingerprints,
        "limitations": [
            "Cluster count is explicitly supplied; not selected or validated.",
            "Only fuzzy clustering; no Gaussian antecedent rule or Sugeno consequent yet.",
            "No use of frozen MLR predictions in clustering, only in provenance checks.",
            "No reserved-test observations or clinical adaptation policy used.",
            "Cluster IDs are arbitrary and not therapist-readable labels.",
        ],
    }
    output_dir.mkdir(parents=True)
    centres.to_csv(output_dir / "cluster_centres.csv", index=False)
    membership.to_csv(output_dir / "development_memberships.csv", index=False)
    (output_dir / "cluster_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--modeling-data", type=Path, required=True)
    parser.add_argument("--holdout-split", type=Path, required=True)
    parser.add_argument("--mod14-output-dir", type=Path, required=True)
    parser.add_argument("--clusters", type=int, required=True,
                        help="Exploratory cluster count; NO automated model selection")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FcmStage1Error(f"Output directory already exists: {args.output_dir}")
    sources = {
        "modeling_data.csv": args.modeling_data,
        "participant_holdout_split.csv": args.holdout_split,
        "mod14_regression_coefficients.csv": args.mod14_output_dir / "regression_coefficients.csv",
        "mod14_training_predictions.csv": args.mod14_output_dir / "training_predictions.csv",
    }
    frame = prepare_frozen_development(
        modeling_data=pd.read_csv(args.modeling_data),
        holdout_split=pd.read_csv(args.holdout_split),
        regression_coefficients=pd.read_csv(sources["mod14_regression_coefficients.csv"]),
        training_predictions=pd.read_csv(sources["mod14_training_predictions.csv"]),
    )
    scaled, maxima = scale_by_training_max(
        frame.loc[:, MOD14.AUDITORY_PREDICTORS].to_numpy(dtype=float)
    )
    fit = fit_fcm(scaled, clusters=args.clusters)
    write_outputs(args.output_dir, frame, fit, maxima,
                  clusters=args.clusters, sources=sources)


if __name__ == "__main__":
    main()
