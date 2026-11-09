#!/usr/bin/env python3
"""MOD-19 Stage 2B: exploratory zero-order Auditory FCM–Sugeno surrogate.

The antecedent is the *joint, distance-based FCM cluster membership*, not
four independent Gaussian terms. Each rule consequent is one development-
fitted constant, learned as a membership-squared weighted teacher mean.
Inference uses normalised FCM memberships to combine the constants.
This method is a deliberately simpler thesis-specific alternative to the
first-order FCM–Sugeno architecture of Kusumadewi et al. (2023).

Only the frozen MOD-14 Auditory MLR is approximated. Stage 2B outputs are
in-sample development diagnostics, NOT participant-grouped validation,
independent-test results, or therapist adaptation recommendations.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

import MOD_19_fcm_sugeno_stage1 as STAGE1
import MOD_19_fcm_sugeno_stage2a as STAGE2A

FEATURES = STAGE1.MOD14.AUDITORY_PREDICTORS


class Stage2BError(ValueError):
    """Invalid Auditory FCM, teacher, rule-fitting or output contract."""


def validate_memberships(memberships: np.ndarray, *, observations: int) -> np.ndarray:
    """Reject invalid fuzzy activations instead of silently normalising them."""
    u = np.asarray(memberships, dtype=float)
    if (u.ndim != 2 or u.shape[0] != observations or u.shape[1] < 2
            or not np.isfinite(u).all() or (u < 0).any() or (u > 1).any()
            or not np.allclose(u.sum(axis=1), 1.0, atol=1e-10, rtol=0)):
        raise Stage2BError("Expected finite FCM memberships in [0,1], summing to one")
    return u


def fit_zero_order_constants(
    mlr_predictions: np.ndarray, memberships: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Learn local constant teacher means using the existing FCM m=2 weights.

    This is per-cluster weighted least squares, NOT a globally optimised fit
    to the final blended output. It is independent of participant test data.
    """
    y = np.asarray(mlr_predictions, dtype=float)
    if y.ndim != 1 or not len(y) or not np.isfinite(y).all():
        raise Stage2BError("Frozen MLR teacher targets must be a finite vector")
    u = validate_memberships(memberships, observations=len(y))
    weights = u ** STAGE1.FUZZIFIER
    sums = weights.sum(axis=0)
    if (sums <= 0).any():
        raise Stage2BError("An FCM cluster has no consequent training support")
    constants = weights.T @ y / sums
    effective_n = sums ** 2 / np.sum(weights ** 2, axis=0)
    if not np.isfinite(constants).all() or not np.isfinite(effective_n).all():
        raise Stage2BError("Invalid zero-order consequent or effective sample size")
    return constants, effective_n


def combine_rule_outputs(
    memberships: np.ndarray, constants: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Apply normalised FCM rule firing to constant Sugeno consequents."""
    c = np.asarray(constants, dtype=float)
    if c.ndim != 1 or c.size < 2 or not np.isfinite(c).all():
        raise Stage2BError("Expected finite scalar consequent for each cluster")
    u = validate_memberships(memberships, observations=len(memberships))
    if u.shape[1] != c.size:
        raise Stage2BError("Rule count differs between membership and consequents")
    contributions = u * c[None, :]
    predictions = contributions.sum(axis=1) / u.sum(axis=1)
    if not np.isfinite(predictions).all():
        raise Stage2BError("Non-finite zero-order Sugeno prediction")
    return predictions, contributions


@dataclass(frozen=True)
class ZeroOrderAuditory:
    """Exploratory fitted surrogate; feature order and scaling are immutable."""

    centres: np.ndarray
    training_maxima: np.ndarray
    constants: np.ndarray

    def infer(self, aud_features: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Infer from four original-unit Auditory features without refitting."""
        x = np.asarray(aud_features, dtype=float)
        if (x.ndim != 2 or x.shape[1] != len(FEATURES)
                or not np.isfinite(x).all() or (x < 0).any()
                or self.training_maxima.shape != (len(FEATURES),)
                or not np.isfinite(self.training_maxima).all()
                or (self.training_maxima <= 0).any()
                or self.centres.shape != (len(self.constants), len(FEATURES))
                or not np.isfinite(self.centres).all()):
            raise Stage2BError("Invalid four-feature Auditory inference input/model")
        u = STAGE1.membership_from_centres(
            x / self.training_maxima, self.centres,
            fuzzifier=STAGE1.FUZZIFIER,
        )
        prediction, contributions = combine_rule_outputs(u, self.constants)
        return prediction, u, contributions


def model_from_saved_stage1(
    stage1_dir: Path, manifest: dict, constants: np.ndarray
) -> ZeroOrderAuditory:
    """Recover original Stage 1 centres and training scaling without re-fitting."""
    k = manifest["fcm"]["clusters"]
    centres_table = pd.read_csv(Path(stage1_dir) / "cluster_centres.csv")
    if (list(centres_table.columns[:1]) != ["cluster"]
            or len(centres_table) != k
            or centres_table["cluster"].tolist() != list(range(1, k + 1))):
        raise Stage2BError("Stage 1 centre ordering changed")
    centres = centres_table[list(FEATURES)].to_numpy(dtype=float)
    maxima = np.array([manifest["training_feature_maxima"][n] for n in FEATURES],
                      dtype=float)
    if len(constants) != k or not np.isfinite(constants).all():
        raise Stage2BError("Stage 1 cluster count and fitted constants differ")
    return ZeroOrderAuditory(centres, maxima, constants)


def write_outputs(
    output_dir: Path, *, frame: pd.DataFrame, model: ZeroOrderAuditory,
    memberships: np.ndarray, predictions: np.ndarray,
    effective_n: np.ndarray, stage1_dir: Path,
    source_paths: dict[str, Path],
) -> tuple[float, float]:
    """Save local development-only artefacts; refuse to overwrite any directory."""
    output_dir = Path(output_dir)
    if output_dir.exists():
        raise Stage2BError(f"Output directory already exists: {output_dir}")
    STAGE1.validate_development_frame(frame)
    u = validate_memberships(memberships, observations=len(frame))
    y = frame["frozen_mlr_prediction"].to_numpy(dtype=float)
    p = np.asarray(predictions, dtype=float)
    if (p.shape != y.shape or not np.isfinite(p).all()
            or model.centres.shape != (u.shape[1], len(FEATURES))
            or model.constants.shape != (u.shape[1],)
            or effective_n.shape != (u.shape[1],)
            or not np.isfinite(model.constants).all()
            or not np.isfinite(effective_n).all()):
        raise Stage2BError("Invalid model or in-sample development predictions")
    inferred, inferred_u, contributions = model.infer(
        frame[list(FEATURES)].to_numpy(dtype=float)
    )
    if (not np.allclose(u, inferred_u, atol=1e-7, rtol=0)
            or not np.allclose(p, inferred, atol=1e-10, rtol=0)):
        raise Stage2BError("Prediction differs from saved FCM inference")
    # Reuse MOD-14's exact thesis-specific metrics, not sklearn defaults.
    mae = STAGE1.MOD14.participant_balanced_mae(
        participant_ids=frame["participant_id"],
        observed=frame["frozen_mlr_prediction"],
        predicted=pd.Series(p),
    )
    r2 = STAGE1.MOD14.pooled_r2(
        observed=frame["frozen_mlr_prediction"],
        predicted=pd.Series(p),
    )
    rules = pd.DataFrame({
        "cluster": np.arange(1, len(model.constants) + 1),
        "constant_difficulty_stage": model.constants,
        "effective_sample_size_u_squared": effective_n,
    })
    for idx, feature in enumerate(FEATURES):
        rules[feature + "_centre_original_units"] = (
            model.centres[:, idx] * model.training_maxima[idx]
        )
    development = frame[
        ["participant_id", "difficulty_level", "difficulty_stage"]
    ].copy()
    development["frozen_mlr_prediction"] = y
    development["zero_order_sugeno_prediction"] = p
    development["absolute_surrogate_error"] = np.abs(y - p)
    for k in range(u.shape[1]):
        development[f"cluster_{k + 1}_membership"] = u[:, k]
        development[f"rule_{k + 1}_weighted_contribution"] = contributions[:, k]

    hashes = {name: STAGE1.sha256(path) for name, path in source_paths.items()}
    for name in ("cluster_manifest.json", "cluster_centres.csv",
                 "development_memberships.csv"):
        hashes["stage1/" + name] = STAGE1.sha256(Path(stage1_dir) / name)
    manifest = {
        "module": "MOD-19", "model": "auditory",
        "stage": "2B: exploratory zero-order FCM-Sugeno",
        "status": "in-sample development diagnostics only, NOT grouped CV/test",
        "development_participants": 26, "development_rows": 104,
        "reserved_test_participants_used": 0,
        "input_features": list(FEATURES),
        "cluster_count": len(model.constants),
        "fcm_fuzzifier": STAGE1.FUZZIFIER,
        "antecedent": "joint distance-based FCM membership, NOT Gaussian feature terms",
        "consequent": "one constant per cluster; local (u^2)-weighted MLR teacher mean",
        "inference": "FCM membership-weighted mean of constant consequents",
        "target": "frozen MOD-14 Auditory MLR predictions, NOT observed difficulty",
        "development_in_sample_participant_balanced_MAE": float(mae),
        "development_in_sample_pooled_R2": float(r2),
        "consequent_range": [float(min(model.constants)), float(max(model.constants))],
        "source_sha256": hashes,
        "limitations": [
            "K is exploratory, not selected; no participant-grouped validation yet.",
            "Constants are local weighted means, not globally optimal blended parameters.",
            "FCM joint memberships are not Gaussian per-feature fuzzy antecedents.",
            "In-sample fidelity does not establish generalisation or therapist comprehension.",
            "Rule activations are not causal explanations or adaptation instructions.",
            "Only development participants supplied targets; reserved test not evaluated.",
        ],
    }
    output_dir.mkdir(parents=True)
    rules.to_csv(output_dir / "zero_order_rules.csv", index=False)
    development.to_csv(output_dir / "development_predictions.csv", index=False)
    (output_dir / "zero_order_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return float(mae), float(r2)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--modeling-data", type=Path, required=True)
    parser.add_argument("--holdout-split", type=Path, required=True)
    parser.add_argument("--mod14-output-dir", type=Path, required=True)
    parser.add_argument("--stage1-output-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise Stage2BError(f"Output directory already exists: {args.output_dir}")
    sources = {
        "modeling_data.csv": args.modeling_data,
        "participant_holdout_split.csv": args.holdout_split,
        "mod14_regression_coefficients.csv": args.mod14_output_dir / "regression_coefficients.csv",
        "mod14_training_predictions.csv": args.mod14_output_dir / "training_predictions.csv",
    }
    frame = STAGE1.prepare_frozen_development(
        modeling_data=pd.read_csv(args.modeling_data),
        holdout_split=pd.read_csv(args.holdout_split),
        regression_coefficients=pd.read_csv(sources["mod14_regression_coefficients.csv"]),
        training_predictions=pd.read_csv(sources["mod14_training_predictions.csv"]),
    )
    memberships, fcm_manifest = STAGE2A.read_verified_stage1(
        frame, args.stage1_output_dir, sources
    )
    constants, effective_n = fit_zero_order_constants(
        frame["frozen_mlr_prediction"].to_numpy(dtype=float), memberships
    )
    model = model_from_saved_stage1(args.stage1_output_dir, fcm_manifest, constants)
    predictions, recomputed_u, _ = model.infer(
        frame[list(FEATURES)].to_numpy(dtype=float)
    )
    if not np.allclose(memberships, recomputed_u, atol=1e-7, rtol=0):
        raise Stage2BError("New inference memberships do not match Stage 1")
    mae, r2 = write_outputs(
        args.output_dir, frame=frame, model=model,
        memberships=memberships, predictions=predictions,
        effective_n=effective_n, stage1_dir=args.stage1_output_dir,
        source_paths=sources,
    )
    print("Stage 2B: zero-order Auditory FCM-Sugeno (exploratory development only)")
    print("Cluster constants:", ", ".join(f"{v:.8g}" for v in constants))
    print(f"In-sample participant-balanced surrogate MAE: {mae:.8g}")
    print(f"In-sample pooled surrogate R²: {r2:.8g}")
    print("Grouped development validation and reserved-test assessment: NOT performed")


if __name__ == "__main__":
    main()
