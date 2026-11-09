#!/usr/bin/env python3
"""MOD-19 Stage 2A: development-only diagnostic of Sugeno consequent degeneracy.

Fit separate unregularised first-order weighted linear consequents against the
frozen Auditory MLR teacher using Stage 1 FCM memberships. This is a diagnostic,
NOT a fitted Gaussian-antecedent Sugeno system, a clinical policy, or a model
selection. Never read or use reserved-test observations beyond the frozen split
validation already performed by the Stage 1 preparation routine.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

import MOD_19_fcm_sugeno_stage1 as STAGE1

FEATURES = STAGE1.MOD14.AUDITORY_PREDICTORS
COEFFICIENT_NAMES = ("Intercept", *FEATURES)
DIAGNOSTIC_TOLERANCE = 1e-7


class Stage2AError(ValueError):
    """An input identity, membership, or regression contract is invalid."""


@dataclass(frozen=True)
class ConsequentDiagnostic:
    coefficients: np.ndarray
    rows: tuple[dict, ...]
    maximum_teacher_error: float
    maximum_consequent_difference: float
    degenerate_identical_consequents: bool


def read_verified_stage1(
    frame: pd.DataFrame, stage1_dir: Path, source_paths: dict[str, Path]
) -> tuple[np.ndarray, dict]:
    """Verify Stage 1 manifest, membership identity and fitted centre equations."""
    STAGE1.validate_development_frame(frame)
    stage1_dir = Path(stage1_dir)
    manifest_path = stage1_dir / "cluster_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (manifest.get("model") != "auditory"
            or manifest.get("development_participants") != 26
            or manifest.get("development_rows") != 104
            or manifest.get("reserved_test_participants_used") != 0
            or manifest.get("features") != list(FEATURES)):
        raise Stage2AError("Stage 1 development-only Auditory provenance mismatch")
    fcm = manifest.get("fcm", {})
    k = fcm.get("clusters")
    if (type(k) is not int or k < 2 or fcm.get("converged") is not True
            or fcm.get("fuzzifier") != 2.0):
        raise Stage2AError("Stage 1 FCM did not converge or changed its parameters")
    previous_hashes = manifest.get("source_sha256", {})
    if set(source_paths) != set(previous_hashes):
        raise Stage2AError("Stage 1 source manifest keys differ from current inputs")
    for name, path in source_paths.items():
        if STAGE1.sha256(Path(path)) != previous_hashes[name]:
            raise Stage2AError(f"Stage 1 source changed: {name}")

    memberships_csv = pd.read_csv(
        stage1_dir / "development_memberships.csv", dtype={"participant_id": str}
    )
    id_cols = ["participant_id", "difficulty_level", "difficulty_stage"]
    weight_cols = [f"cluster_{i}_membership" for i in range(1, k + 1)]
    if list(memberships_csv.columns) != [*id_cols, *weight_cols]:
        raise Stage2AError("Unexpected Stage 1 membership schema")
    if (len(memberships_csv) != len(frame)
            or memberships_csv[id_cols].astype(str).to_numpy().tolist()
            != frame[id_cols].astype(str).to_numpy().tolist()):
        raise Stage2AError("Stage 1 participant/stage membership row alignment changed")
    u = memberships_csv[weight_cols].to_numpy(dtype=float)
    if (not np.isfinite(u).all() or np.any(u < 0) or np.any(u > 1)
            or not np.allclose(u.sum(axis=1), 1.0, atol=1e-10, rtol=0)):
        raise Stage2AError("Stage 1 memberships are invalid")

    centres_table = pd.read_csv(stage1_dir / "cluster_centres.csv")
    if (len(centres_table) != k or centres_table["cluster"].tolist()
            != list(range(1, k + 1))):
        raise Stage2AError("Stage 1 centres or cluster ordering changed")
    centres = centres_table[list(FEATURES)].to_numpy(dtype=float)
    maxima = np.array([manifest["training_feature_maxima"][f] for f in FEATURES], dtype=float)
    if (not np.isfinite(centres).all() or not np.isfinite(maxima).all()
            or np.any(maxima <= 0)):
        raise Stage2AError("Stage 1 centres/scaling are invalid")
    scaled = frame[list(FEATURES)].to_numpy(dtype=float) / maxima
    if not np.allclose(scaled.max(axis=0), 1.0, atol=1e-10, rtol=0):
        raise Stage2AError("Stage 1 scaling maxima do not match development data")
    expected_u = STAGE1.membership_from_centres(scaled, centres, fuzzifier=2.0)
    if not np.allclose(u, expected_u, atol=DIAGNOSTIC_TOLERANCE, rtol=0):
        raise Stage2AError("Saved memberships do not match Stage 1 FCM centres")
    weights = u ** 2
    expected_centres = weights.T @ scaled / weights.sum(axis=0)[:, None]
    if not np.allclose(centres, expected_centres, atol=DIAGNOSTIC_TOLERANCE, rtol=0):
        raise Stage2AError("Stage 1 centres do not match fuzzy weighted averages")
    return u, manifest


def frozen_teacher_coefficients(table: pd.DataFrame) -> np.ndarray:
    """Read existing MOD-14 Auditory coefficients in the frozen feature order."""
    required = {"model", "predictor", "coefficient"}
    if not required.issubset(table.columns):
        raise Stage2AError("MOD-14 coefficient schema differs")
    source = table.loc[table["model"].eq("auditory")]
    if (len(source) != len(COEFFICIENT_NAMES)
            or source["predictor"].duplicated().any()
            or set(source["predictor"]) != set(COEFFICIENT_NAMES)):
        raise Stage2AError("Frozen Auditory coefficient names differ")
    series = pd.Series(
        pd.to_numeric(source["coefficient"], errors="coerce").to_numpy(dtype=float),
        index=source["predictor"],
    ).reindex(COEFFICIENT_NAMES)
    coefficients = series.to_numpy(dtype=float)
    if not np.isfinite(coefficients).all():
        raise Stage2AError("Non-finite frozen Auditory coefficients")
    return coefficients


def diagnose_local_consequents(
    frame: pd.DataFrame, memberships: np.ndarray, teacher_coefficients: np.ndarray
) -> ConsequentDiagnostic:
    """Fit K separate weighted OLS equations and test algebraic teacher copying.

    The FCM degree squared is used as a diagnostic least-squares weight;
    these are *not* final Gaussian-rule firing strengths. No penalty, tuning,
    or held-out assessment is performed here.
    """
    STAGE1.validate_development_frame(frame)
    x = frame[list(FEATURES)].to_numpy(dtype=float)
    design = np.column_stack([np.ones(len(frame)), x])
    teacher = np.asarray(teacher_coefficients, dtype=float)
    u = np.asarray(memberships, dtype=float)
    if (teacher.shape != (design.shape[1],) or not np.isfinite(teacher).all()
            or u.ndim != 2 or u.shape[0] != len(frame) or u.shape[1] < 2
            or not np.isfinite(u).all() or (u < 0).any() or (u > 1).any()
            or not np.allclose(u.sum(axis=1), 1.0, atol=1e-10, rtol=0)):
        raise Stage2AError("Invalid frozen coefficients or FCM memberships")
    y = frame["frozen_mlr_prediction"].to_numpy(dtype=float)
    if not np.allclose(y, design @ teacher, atol=1e-8, rtol=0):
        raise Stage2AError("Frozen teacher predictions do not reconstruct")
    coefficient_rows = []
    diagnostic_rows = []
    for k in range(u.shape[1]):
        weights = u[:, k] ** 2
        sqrt_w = np.sqrt(weights)
        weighted_x = design * sqrt_w[:, None]
        weighted_y = y * sqrt_w
        coefficients, _, rank, singular_values = np.linalg.lstsq(
            weighted_x, weighted_y, rcond=None
        )
        if rank != design.shape[1]:
            raise Stage2AError(f"Cluster {k + 1} has rank-deficient weighted design")
        fitted = design @ coefficients
        deviation = float(np.max(np.abs(coefficients - teacher)))
        residual = float(np.max(np.abs(fitted - y)))
        row = {
            "cluster": k + 1,
            "weighted_design_rank": int(rank),
            "weighted_design_condition_number": float(singular_values[0] / singular_values[-1]),
            "effective_sample_size": float(weights.sum() ** 2 / np.sum(weights ** 2)),
            "maximum_coefficient_difference_from_frozen_mlr": deviation,
            "maximum_prediction_difference_from_frozen_mlr": residual,
        }
        row.update({name: float(value) for name, value in zip(COEFFICIENT_NAMES, coefficients)})
        diagnostic_rows.append(row)
        coefficient_rows.append(coefficients)
    coefficients = np.vstack(coefficient_rows)
    rule_values = design @ coefficients.T
    # Only an algebraic demonstration: these FCM weights are NOT Sugeno Gaussians.
    averaged = np.sum(u * rule_values, axis=1) / np.sum(u, axis=1)
    err = float(np.max(np.abs(averaged - y)))
    coefficient_spread = float(np.max(np.ptp(coefficients, axis=0)))
    degenerate = bool(
        max(row["maximum_coefficient_difference_from_frozen_mlr"]
            for row in diagnostic_rows) <= DIAGNOSTIC_TOLERANCE
        and coefficient_spread <= DIAGNOSTIC_TOLERANCE
        and err <= DIAGNOSTIC_TOLERANCE
    )
    return ConsequentDiagnostic(
        coefficients, tuple(diagnostic_rows), err, coefficient_spread, degenerate
    )


def write_diagnostic(
    output_dir: Path, diagnostic: ConsequentDiagnostic,
    source_paths: dict[str, Path], stage1_dir: Path,
) -> None:
    """Write local aggregate diagnostics only; refuse overwrites."""
    output_dir = Path(output_dir)
    if output_dir.exists():
        raise Stage2AError(f"Output directory already exists: {output_dir}")
    if not diagnostic.rows or not np.isfinite(diagnostic.coefficients).all():
        raise Stage2AError("Invalid local consequent diagnostic")
    source_hashes = {key: STAGE1.sha256(Path(value)) for key, value in source_paths.items()}
    for name in ("cluster_manifest.json", "cluster_centres.csv", "development_memberships.csv"):
        source_hashes["stage1/" + name] = STAGE1.sha256(Path(stage1_dir) / name)
    summary = {
        "module": "MOD-19",
        "stage": "2A: Auditory FCM-Sugeno local consequent degeneracy diagnostic",
        "status": "development-only diagnostic; not fitted Sugeno inference or validated fidelity",
        "model": "auditory", "development_participants": 26,
        "development_rows": 104, "reserved_test_participants_used": 0,
        "number_of_clusters": len(diagnostic.rows),
        "consequent_type": "separate first-order weighted linear least-squares fits",
        "weighting": "squared Stage 1 FCM membership for each cluster; not Gaussian firing",
        "comparison": "frozen MOD-14 Auditory MLR coefficients and development predictions",
        "tolerance": DIAGNOSTIC_TOLERANCE,
        "maximum_FCM_weighted_prediction_difference_from_frozen_MLR": diagnostic.maximum_teacher_error,
        "maximum_coefficient_spread_between_clusters": diagnostic.maximum_consequent_difference,
        "identical_MLR_consequent_degeneracy_detected": diagnostic.degenerate_identical_consequents,
        "source_sha256": source_hashes,
        "limitations": [
            "This is a diagnostic, not a fitted Gaussian-antecedent Sugeno inference system.",
            "Cluster count K=3 is exploratory, not selected by grouped fidelity.",
            "Exact reproduction of a linear teacher can be algebraic degeneracy, not explanatory quality.",
            "No participant-level records saved here; no reserved-test data used in the fitting.",
        ],
    }
    output_dir.mkdir(parents=True)
    pd.DataFrame(diagnostic.rows).to_csv(output_dir / "consequent_diagnostics.csv", index=False)
    (output_dir / "degeneracy_manifest.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--modeling-data", type=Path, required=True)
    parser.add_argument("--holdout-split", type=Path, required=True)
    parser.add_argument("--mod14-output-dir", type=Path, required=True)
    parser.add_argument("--stage1-output-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise Stage2AError(f"Output directory already exists: {args.output_dir}")
    sources = {
        "modeling_data.csv": args.modeling_data,
        "participant_holdout_split.csv": args.holdout_split,
        "mod14_regression_coefficients.csv": args.mod14_output_dir / "regression_coefficients.csv",
        "mod14_training_predictions.csv": args.mod14_output_dir / "training_predictions.csv",
    }
    coeff_table = pd.read_csv(sources["mod14_regression_coefficients.csv"])
    frame = STAGE1.prepare_frozen_development(
        modeling_data=pd.read_csv(args.modeling_data),
        holdout_split=pd.read_csv(args.holdout_split),
        regression_coefficients=coeff_table,
        training_predictions=pd.read_csv(sources["mod14_training_predictions.csv"]),
    )
    u, _ = read_verified_stage1(frame, args.stage1_output_dir, sources)
    diagnostic = diagnose_local_consequents(
        frame, u, frozen_teacher_coefficients(coeff_table)
    )
    write_diagnostic(args.output_dir, diagnostic, sources, args.stage1_output_dir)
    print("FCM-Sugeno Stage 2A (development-only diagnostic)")
    print(f"Cluster consequents: {len(diagnostic.rows)}")
    print(f"Identical MLR consequents detected: {diagnostic.degenerate_identical_consequents}")
    print(f"Max difference from frozen MLR: {diagnostic.maximum_teacher_error:.12g}")
    print(f"Max coefficient spread across clusters: {diagnostic.maximum_consequent_difference:.12g}")


if __name__ == "__main__":
    main()
