#!/usr/bin/env python3
"""MOD-07 — Training-only RI versus RI+RS comparison with participant bootstrap.

This module compares random-intercept (RI) and random-intercept plus random-slope
(RI+RS) structures under the difficulty-only M0 fixed-effects model. It uses only
the 26 participants assigned to training by the frozen MOD-06 split. The six test
participants are excluded from all fitting and bootstrap resampling.

The participant bootstrap resamples the 26 training participant IDs with
replacement. Every sampled occurrence receives a distinct temporary cluster ID,
so repeated draws of the same original participant are treated as separate
bootstrap clusters while retaining all repeated trial rows within each draw.

The CLI fixes B=2000. The bootstrap seed must be supplied explicitly because a
project-wide bootstrap seed has not yet been frozen. This module reports evidence
only; it does not automatically select RI or RI+RS and does not impose a bootstrap
confidence-interval method.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from MOD_02_random_effects_structure import (
    CONDITIONS,
    GAUSSIAN_TARGETS,
    PERFORMANCE_DIFFICULTY_COLUMN,
    fit_mixedlm_with_fallback,
    prepare_internal_modeling_data,
)
from MOD_06_participant_split import (
    assignment_signature,
    canonical_participants,
    participant_source_signature,
)


SCRIPT_VERSION = "1.0.0"
METHOD_ID = "training_m0_ri_vs_ri_rs_cluster_bootstrap_v1"
BOOTSTRAP_REPLICATES = 2000
BOOTSTRAP_BIT_GENERATOR = "PCG64"
EXPECTED_TRAIN = 26
EXPECTED_TEST = 6
PARTICIPANT_COLUMN = "participant_id"
PARTITION_COLUMN = "partition"
BOOTSTRAP_CLUSTER_COLUMN = "bootstrap_cluster_id"

TRAINING_COMPARISON_FILENAME = "training_random_effects_comparison.csv"
TRAINING_AUDIT_FILENAME = "training_random_effects_audit.csv"
BOOTSTRAP_DRAWS_FILENAME = "bootstrap_participant_draws.csv"
BOOTSTRAP_REPLICATES_FILENAME = "bootstrap_random_effects_replicates.csv"
BOOTSTRAP_SUMMARY_FILENAME = "bootstrap_random_effects_summary.csv"
MANIFEST_FILENAME = "mod07_manifest.json"


class Mod07Error(ValueError):
    """Raised when frozen split or training-only invariants are violated."""


@dataclass(frozen=True)
class TargetSpec:
    target: str
    difficulty_column: str
    formula: str


@dataclass(frozen=True)
class Mod07Result:
    training_comparison: pd.DataFrame
    training_audit: pd.DataFrame
    bootstrap_draws: pd.DataFrame
    bootstrap_replicates: pd.DataFrame
    bootstrap_summary: pd.DataFrame
    manifest: dict[str, Any]


def build_m0_target_specs(
    targets: Iterable[str] = GAUSSIAN_TARGETS,
) -> dict[str, TargetSpec]:
    """Return difficulty-only M0 formulas for all requested Gaussian targets."""
    specs: dict[str, TargetSpec] = {}
    for target in targets:
        difficulty_column = (
            PERFORMANCE_DIFFICULTY_COLUMN
            if target == "performance_change_from_d0_percentage_points"
            else "difficulty_stage"
        )
        specs[target] = TargetSpec(
            target=target,
            difficulty_column=difficulty_column,
            formula=f"{target} ~ {difficulty_column}",
        )
    return specs


def _load_split_manifest(path: Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    if not isinstance(manifest, dict):
        raise Mod07Error("Split manifest must contain a JSON object")
    return manifest


def validate_and_filter_training_data(
    modeling_data: pd.DataFrame,
    assignments: pd.DataFrame,
    split_manifest: dict[str, Any],
) -> tuple[pd.DataFrame, tuple[str, ...], tuple[str, ...]]:
    """Validate the frozen MOD-06 split and return only training rows."""
    required_assignment = {PARTICIPANT_COLUMN, "participant_group", PARTITION_COLUMN}
    missing = sorted(required_assignment - set(assignments.columns))
    if missing:
        raise Mod07Error(f"Split assignment is missing required columns: {missing}")

    if assignments[PARTICIPANT_COLUMN].isna().any():
        raise Mod07Error("Split assignment contains missing participant_id")
    if assignments[PARTICIPANT_COLUMN].duplicated().any():
        raise Mod07Error("Split assignment contains duplicate participant_id values")

    assignment_copy = assignments.copy(deep=True)
    assignment_copy[PARTICIPANT_COLUMN] = (
        assignment_copy[PARTICIPANT_COLUMN].astype(str).str.strip()
    )
    if set(assignment_copy[PARTITION_COLUMN]) != {"train", "test"}:
        raise Mod07Error("Split assignment must contain only train and test partitions")

    partition_counts = assignment_copy[PARTITION_COLUMN].value_counts().to_dict()
    if partition_counts != {"train": EXPECTED_TRAIN, "test": EXPECTED_TEST}:
        raise Mod07Error(
            f"Expected train/test participant counts 26/6, found {partition_counts}"
        )

    expected_assignment_hash = split_manifest.get("assignment_sha256")
    actual_assignment_hash = assignment_signature(assignment_copy)
    if expected_assignment_hash != actual_assignment_hash:
        raise Mod07Error(
            "Split assignment SHA-256 does not match participant_split_manifest.json"
        )

    participants = canonical_participants(modeling_data)
    expected_source_hash = split_manifest.get("participant_source_sha256")
    actual_source_hash = participant_source_signature(participants)
    if expected_source_hash != actual_source_hash:
        raise Mod07Error(
            "Modeling-data participant-source SHA-256 does not match the MOD-06 manifest"
        )

    assignment_ids = set(assignment_copy[PARTICIPANT_COLUMN])
    modeling_ids = set(participants[PARTICIPANT_COLUMN].astype(str))
    if assignment_ids != modeling_ids:
        raise Mod07Error("Modeling data and split assignment contain different participants")

    if "participant_group" in modeling_data.columns:
        model_groups = (
            modeling_data[[PARTICIPANT_COLUMN, "participant_group"]]
            .drop_duplicates()
            .copy()
        )
        if model_groups[PARTICIPANT_COLUMN].duplicated().any():
            raise Mod07Error("Modeling data contain inconsistent participant_group values")
        merged = assignment_copy[[PARTICIPANT_COLUMN, "participant_group"]].merge(
            model_groups,
            on=PARTICIPANT_COLUMN,
            how="left",
            suffixes=("_split", "_model"),
            validate="one_to_one",
        )
        mismatch = merged["participant_group_split"].ne(merged["participant_group_model"])
        if mismatch.any():
            bad = sorted(merged.loc[mismatch, PARTICIPANT_COLUMN].astype(str))
            raise Mod07Error(f"participant_group mismatch for participants: {bad}")

    training_ids = tuple(
        sorted(
            assignment_copy.loc[
                assignment_copy[PARTITION_COLUMN].eq("train"), PARTICIPANT_COLUMN
            ].astype(str)
        )
    )
    test_ids = tuple(
        sorted(
            assignment_copy.loc[
                assignment_copy[PARTITION_COLUMN].eq("test"), PARTICIPANT_COLUMN
            ].astype(str)
        )
    )

    prepared = prepare_internal_modeling_data(modeling_data)
    training = prepared.loc[
        prepared[PARTICIPANT_COLUMN].astype(str).isin(training_ids)
    ].copy()
    observed_training_ids = set(training[PARTICIPANT_COLUMN].astype(str))
    if observed_training_ids != set(training_ids):
        raise Mod07Error("Not all 26 training participants are present in modeling data")
    if set(training[PARTICIPANT_COLUMN].astype(str)) & set(test_ids):
        raise Mod07Error("Held-out test participants leaked into the training frame")

    return training, training_ids, test_ids


def _prepare_target_frame(
    training_data: pd.DataFrame,
    condition: str,
    spec: TargetSpec,
) -> tuple[pd.DataFrame, int]:
    frame = training_data.loc[
        training_data["condition_name"].eq(condition)
    ].copy()
    structural_excluded = 0

    if spec.target == "performance_change_from_d0_percentage_points":
        d0_mask = pd.to_numeric(frame["difficulty_level"], errors="coerce").eq(0)
        structural_excluded = int(d0_mask.sum())
        frame = frame.loc[~d0_mask].copy()

    required = [PARTICIPANT_COLUMN, spec.target, spec.difficulty_column]
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise Mod07Error(
            f"{condition}/{spec.target}: missing required columns {missing}"
        )
    frame = frame.loc[frame[required].notna().all(axis=1)].copy()
    if frame.empty:
        raise Mod07Error(f"{condition}/{spec.target}: no eligible rows")
    return frame, structural_excluded


def _row_signature(frame: pd.DataFrame) -> str:
    columns = [
        column
        for column in (
            PARTICIPANT_COLUMN,
            "condition_name",
            "difficulty_level",
            "difficulty_stage",
            PERFORMANCE_DIFFICULTY_COLUMN,
            "trial_order",
            "session_id",
        )
        if column in frame.columns
    ]
    payload_rows = (
        frame[columns]
        .astype("string")
        .fillna("<NA>")
        .agg("|".join, axis=1)
        .sort_values(kind="mergesort")
        .tolist()
    )
    return sha256("\n".join(payload_rows).encode("utf-8")).hexdigest()


def _safe_float(value: Any) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return np.nan
    return numeric if np.isfinite(numeric) else np.nan


def _extract_fit_metrics(result: Any, structure: str) -> dict[str, Any]:
    if result is None:
        return {
            "converged": False,
            "log_likelihood": np.nan,
            "aic": np.nan,
            "bic": np.nan,
            "random_intercept_variance": np.nan,
            "random_slope_variance": np.nan,
            "intercept_slope_covariance": np.nan,
            "intercept_slope_correlation": np.nan,
        }

    converged = bool(getattr(result, "converged", False))
    covariance_matrix = np.asarray(getattr(result, "cov_re", np.empty((0, 0))))
    ri_variance = (
        _safe_float(covariance_matrix[0, 0])
        if covariance_matrix.ndim == 2 and covariance_matrix.shape[0] >= 1
        else np.nan
    )
    rs_variance = np.nan
    covariance = np.nan
    correlation = np.nan
    if structure == "RI_RS" and covariance_matrix.ndim == 2 and covariance_matrix.shape[0] >= 2:
        rs_variance = _safe_float(covariance_matrix[1, 1])
        covariance = _safe_float(covariance_matrix[0, 1])
        if (
            np.isfinite(ri_variance)
            and np.isfinite(rs_variance)
            and ri_variance > 0
            and rs_variance > 0
            and np.isfinite(covariance)
        ):
            correlation = covariance / np.sqrt(ri_variance * rs_variance)

    return {
        "converged": converged,
        "log_likelihood": _safe_float(getattr(result, "llf", np.nan)),
        "aic": _safe_float(getattr(result, "aic", np.nan)),
        "bic": _safe_float(getattr(result, "bic", np.nan)),
        "random_intercept_variance": ri_variance,
        "random_slope_variance": rs_variance,
        "intercept_slope_covariance": covariance,
        "intercept_slope_correlation": correlation,
    }


def _fit_pair(
    fit_data: pd.DataFrame,
    spec: TargetSpec,
    *,
    participant_column: str = PARTICIPANT_COLUMN,
) -> dict[str, Any]:
    pair: dict[str, Any] = {}
    for prefix, structure in (("ri", "RI"), ("ri_rs", "RI_RS")):
        result, optimizer, fit_warnings, fit_errors = fit_mixedlm_with_fallback(
            spec.formula,
            fit_data,
            structure,
            spec.difficulty_column,
            participant_column=participant_column,
        )
        metrics = _extract_fit_metrics(result, structure)
        for key, value in metrics.items():
            pair[f"{prefix}_{key}"] = value
        pair[f"{prefix}_optimizer"] = optimizer
        pair[f"{prefix}_warnings"] = " | ".join(fit_warnings)
        pair[f"{prefix}_errors"] = " | ".join(fit_errors)

    pair["paired_converged"] = bool(
        pair["ri_converged"] and pair["ri_rs_converged"]
    )
    if pair["paired_converged"]:
        pair["delta_aic_ri_rs_minus_ri"] = (
            pair["ri_rs_aic"] - pair["ri_aic"]
            if np.isfinite(pair["ri_rs_aic"]) and np.isfinite(pair["ri_aic"])
            else np.nan
        )
        pair["delta_bic_ri_rs_minus_ri"] = (
            pair["ri_rs_bic"] - pair["ri_bic"]
            if np.isfinite(pair["ri_rs_bic"]) and np.isfinite(pair["ri_bic"])
            else np.nan
        )
    else:
        pair["delta_aic_ri_rs_minus_ri"] = np.nan
        pair["delta_bic_ri_rs_minus_ri"] = np.nan
    return pair


def compare_training_random_effects(
    training_data: pd.DataFrame,
    *,
    targets: Iterable[str] = GAUSSIAN_TARGETS,
    conditions: Iterable[str] = CONDITIONS,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[tuple[str, str], pd.DataFrame]]:
    """Fit the M0 RI/RI+RS pair once on the original 26-person training set."""
    specs = build_m0_target_specs(targets)
    comparison_rows: list[dict[str, Any]] = []
    audit_rows: list[dict[str, Any]] = []
    prepared_frames: dict[tuple[str, str], pd.DataFrame] = {}

    for condition in conditions:
        for target in targets:
            spec = specs[target]
            try:
                fit_data, structural_excluded = _prepare_target_frame(
                    training_data, condition, spec
                )
                prepared_frames[(condition, target)] = fit_data
                signature = _row_signature(fit_data)
                pair = _fit_pair(fit_data, spec)
                comparison_rows.append(
                    {
                        "condition_name": condition,
                        "target_name": target,
                        "fixed_effects_model": "M0_difficulty_only",
                        "fixed_effects_formula": spec.formula,
                        "difficulty_source_column": spec.difficulty_column,
                        "participant_count": int(
                            fit_data[PARTICIPANT_COLUMN].nunique()
                        ),
                        "observation_count": int(len(fit_data)),
                        "excluded_structural_d0_count": structural_excluded,
                        "included_row_signature": signature,
                        "selection_status": "evidence_only_no_automatic_selection",
                        **pair,
                    }
                )
                audit_rows.append(
                    {
                        "condition_name": condition,
                        "target_name": target,
                        "status": "comparable" if pair["paired_converged"] else "fit_problem",
                        "participant_count": int(
                            fit_data[PARTICIPANT_COLUMN].nunique()
                        ),
                        "observation_count": int(len(fit_data)),
                        "included_row_signature": signature,
                        "message": (
                            "RI and RI+RS used identical M0 rows and difficulty coding."
                            if pair["paired_converged"]
                            else "One or both paired fits did not converge."
                        ),
                    }
                )
            except Exception as exc:
                audit_rows.append(
                    {
                        "condition_name": condition,
                        "target_name": target,
                        "status": "failed",
                        "participant_count": np.nan,
                        "observation_count": np.nan,
                        "included_row_signature": "",
                        "message": f"{type(exc).__name__}: {exc}",
                    }
                )

    return (
        pd.DataFrame(comparison_rows),
        pd.DataFrame(audit_rows),
        prepared_frames,
    )


def generate_bootstrap_draws(
    training_ids: Iterable[str],
    *,
    bootstrap_replicates: int,
    bootstrap_seed: int,
) -> pd.DataFrame:
    """Generate deterministic participant draws from the frozen training IDs."""
    ids = np.asarray(sorted(str(value) for value in training_ids), dtype=str)
    if len(ids) != EXPECTED_TRAIN or len(set(ids.tolist())) != EXPECTED_TRAIN:
        raise Mod07Error("Bootstrap requires exactly 26 unique training participants")
    if bootstrap_replicates <= 0:
        raise Mod07Error("bootstrap_replicates must be positive")

    rng = np.random.Generator(np.random.PCG64(int(bootstrap_seed)))
    rows: list[dict[str, Any]] = []
    for replicate in range(1, bootstrap_replicates + 1):
        sampled = rng.choice(ids, size=EXPECTED_TRAIN, replace=True)
        for draw_position, participant_id in enumerate(sampled, start=1):
            rows.append(
                {
                    "bootstrap_replicate": replicate,
                    "draw_position": draw_position,
                    PARTICIPANT_COLUMN: str(participant_id),
                }
            )
    return pd.DataFrame(rows)


def _materialize_bootstrap_sample(
    fit_data: pd.DataFrame,
    draw_ids: Iterable[str],
) -> pd.DataFrame:
    pieces: list[pd.DataFrame] = []
    for draw_position, participant_id in enumerate(draw_ids, start=1):
        piece = fit_data.loc[
            fit_data[PARTICIPANT_COLUMN].astype(str).eq(str(participant_id))
        ].copy()
        if piece.empty:
            continue
        piece[BOOTSTRAP_CLUSTER_COLUMN] = (
            f"draw_{draw_position:02d}_{participant_id}"
        )
        pieces.append(piece)
    if not pieces:
        raise Mod07Error("Bootstrap sample contains no eligible rows")
    return pd.concat(pieces, ignore_index=True)


def _draw_signature(draw_ids: Iterable[str]) -> str:
    payload = "\n".join(
        f"{position}|{participant_id}"
        for position, participant_id in enumerate(draw_ids, start=1)
    )
    return sha256(payload.encode("utf-8")).hexdigest()


def bootstrap_random_effects(
    prepared_frames: dict[tuple[str, str], pd.DataFrame],
    bootstrap_draws: pd.DataFrame,
    *,
    targets: Iterable[str] = GAUSSIAN_TARGETS,
    conditions: Iterable[str] = CONDITIONS,
) -> pd.DataFrame:
    """Fit RI/RI+RS M0 pairs for every bootstrap draw and condition × target."""
    specs = build_m0_target_specs(targets)
    rows: list[dict[str, Any]] = []
    grouped_draws = bootstrap_draws.groupby("bootstrap_replicate", sort=True)

    for replicate, draw_frame in grouped_draws:
        ordered_draw = (
            draw_frame.sort_values("draw_position", kind="mergesort")
            [PARTICIPANT_COLUMN]
            .astype(str)
            .tolist()
        )
        draw_signature = _draw_signature(ordered_draw)
        unique_original_count = len(set(ordered_draw))

        for condition in conditions:
            for target in targets:
                spec = specs[target]
                base_frame = prepared_frames.get((condition, target))
                if base_frame is None:
                    rows.append(
                        {
                            "bootstrap_replicate": int(replicate),
                            "condition_name": condition,
                            "target_name": target,
                            "bootstrap_draw_signature": draw_signature,
                            "unique_original_participants_drawn": unique_original_count,
                            "bootstrap_cluster_count": 0,
                            "observation_count": 0,
                            "paired_converged": False,
                            "fit_status": "missing_training_pair",
                        }
                    )
                    continue

                sample = _materialize_bootstrap_sample(base_frame, ordered_draw)
                pair = _fit_pair(
                    sample,
                    spec,
                    participant_column=BOOTSTRAP_CLUSTER_COLUMN,
                )
                rows.append(
                    {
                        "bootstrap_replicate": int(replicate),
                        "condition_name": condition,
                        "target_name": target,
                        "fixed_effects_model": "M0_difficulty_only",
                        "fixed_effects_formula": spec.formula,
                        "difficulty_source_column": spec.difficulty_column,
                        "bootstrap_draw_signature": draw_signature,
                        "unique_original_participants_drawn": unique_original_count,
                        "bootstrap_cluster_count": int(
                            sample[BOOTSTRAP_CLUSTER_COLUMN].nunique()
                        ),
                        "observation_count": int(len(sample)),
                        "fit_status": (
                            "paired_converged"
                            if pair["paired_converged"]
                            else "fit_problem"
                        ),
                        **pair,
                    }
                )

    return pd.DataFrame(rows)


def summarize_bootstrap_replicates(
    replicate_results: pd.DataFrame,
) -> pd.DataFrame:
    """Return descriptive stability summaries without selecting a winner or CI."""
    rows: list[dict[str, Any]] = []
    for (condition, target), group in replicate_results.groupby(
        ["condition_name", "target_name"], sort=True
    ):
        paired = group.loc[group["paired_converged"].eq(True)].copy()
        aic = paired.loc[
            np.isfinite(pd.to_numeric(paired["delta_aic_ri_rs_minus_ri"], errors="coerce"))
        ].copy()
        bic = paired.loc[
            np.isfinite(pd.to_numeric(paired["delta_bic_ri_rs_minus_ri"], errors="coerce"))
        ].copy()
        slope = pd.to_numeric(
            paired["ri_rs_random_slope_variance"], errors="coerce"
        ).dropna()
        delta_aic = pd.to_numeric(
            aic["delta_aic_ri_rs_minus_ri"], errors="coerce"
        ).dropna()
        delta_bic = pd.to_numeric(
            bic["delta_bic_ri_rs_minus_ri"], errors="coerce"
        ).dropna()

        rows.append(
            {
                "condition_name": condition,
                "target_name": target,
                "bootstrap_replicates_requested": int(len(group)),
                "paired_converged_count": int(len(paired)),
                "paired_converged_percentage": (
                    100.0 * len(paired) / len(group) if len(group) else np.nan
                ),
                "valid_aic_pair_count": int(len(delta_aic)),
                "ri_rs_lower_aic_count": int((delta_aic < 0).sum()),
                "ri_rs_lower_aic_percentage_of_valid": (
                    100.0 * (delta_aic < 0).mean() if len(delta_aic) else np.nan
                ),
                "delta_aic_median": (
                    float(delta_aic.median()) if len(delta_aic) else np.nan
                ),
                "delta_aic_q25": (
                    float(delta_aic.quantile(0.25)) if len(delta_aic) else np.nan
                ),
                "delta_aic_q75": (
                    float(delta_aic.quantile(0.75)) if len(delta_aic) else np.nan
                ),
                "valid_bic_pair_count": int(len(delta_bic)),
                "ri_rs_lower_bic_count": int((delta_bic < 0).sum()),
                "ri_rs_lower_bic_percentage_of_valid": (
                    100.0 * (delta_bic < 0).mean() if len(delta_bic) else np.nan
                ),
                "delta_bic_median": (
                    float(delta_bic.median()) if len(delta_bic) else np.nan
                ),
                "delta_bic_q25": (
                    float(delta_bic.quantile(0.25)) if len(delta_bic) else np.nan
                ),
                "delta_bic_q75": (
                    float(delta_bic.quantile(0.75)) if len(delta_bic) else np.nan
                ),
                "ri_rs_random_slope_variance_median": (
                    float(slope.median()) if len(slope) else np.nan
                ),
                "selection_status": "evidence_only_no_automatic_selection",
                "confidence_interval_status": "not_frozen_not_computed",
            }
        )
    return pd.DataFrame(rows)


def build_manifest(
    *,
    modeling_data_path: Path,
    split_assignments_path: Path,
    split_manifest_path: Path,
    split_manifest: dict[str, Any],
    training_ids: Iterable[str],
    test_ids: Iterable[str],
    bootstrap_seed: int,
    bootstrap_replicates: int,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "script_version": SCRIPT_VERSION,
        "method_id": METHOD_ID,
        "fixed_effects_model": "M0_difficulty_only",
        "random_structures_compared": ["RI", "RI_RS"],
        "estimation_method": "maximum_likelihood",
        "optimizer_rule": (
            "run lbfgs and powell; retain converged fit with highest finite log likelihood"
        ),
        "bootstrap_unit": "participant",
        "bootstrap_replicates": int(bootstrap_replicates),
        "bootstrap_seed": int(bootstrap_seed),
        "bootstrap_seed_status": "explicit_execution_parameter_not_yet_project_frozen",
        "bootstrap_bit_generator": BOOTSTRAP_BIT_GENERATOR,
        "training_participant_count": len(tuple(training_ids)),
        "test_participant_count": len(tuple(test_ids)),
        "training_participants": list(training_ids),
        "held_out_test_participants": list(test_ids),
        "test_participants_used_for_fitting": False,
        "test_participants_used_for_bootstrap": False,
        "automatic_random_structure_selection": False,
        "confidence_interval_method": None,
        "confidence_interval_status": "not_yet_frozen",
        "mod06_assignment_sha256": split_manifest.get("assignment_sha256"),
        "mod06_participant_source_sha256": split_manifest.get(
            "participant_source_sha256"
        ),
        "modeling_data_path": str(modeling_data_path),
        "split_assignments_path": str(split_assignments_path),
        "split_manifest_path": str(split_manifest_path),
    }


def run_mod07(
    modeling_data: pd.DataFrame,
    assignments: pd.DataFrame,
    split_manifest: dict[str, Any],
    *,
    bootstrap_seed: int,
    bootstrap_replicates: int = BOOTSTRAP_REPLICATES,
    targets: Iterable[str] = GAUSSIAN_TARGETS,
    conditions: Iterable[str] = CONDITIONS,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    tuple[str, ...],
    tuple[str, ...],
]:
    training, training_ids, test_ids = validate_and_filter_training_data(
        modeling_data, assignments, split_manifest
    )
    comparison, audit, prepared_frames = compare_training_random_effects(
        training, targets=targets, conditions=conditions
    )
    draws = generate_bootstrap_draws(
        training_ids,
        bootstrap_replicates=bootstrap_replicates,
        bootstrap_seed=bootstrap_seed,
    )
    replicates = bootstrap_random_effects(
        prepared_frames,
        draws,
        targets=targets,
        conditions=conditions,
    )
    summary = summarize_bootstrap_replicates(replicates)
    return comparison, audit, draws, replicates, summary, training_ids, test_ids


def write_outputs(result: Mod07Result, output_dir: Path) -> dict[str, Path]:
    """Write all MOD-07 evidence without overwriting a previous official run."""
    output_dir = Path(output_dir)
    paths = {
        "training_comparison": output_dir / TRAINING_COMPARISON_FILENAME,
        "training_audit": output_dir / TRAINING_AUDIT_FILENAME,
        "bootstrap_draws": output_dir / BOOTSTRAP_DRAWS_FILENAME,
        "bootstrap_replicates": output_dir / BOOTSTRAP_REPLICATES_FILENAME,
        "bootstrap_summary": output_dir / BOOTSTRAP_SUMMARY_FILENAME,
        "manifest": output_dir / MANIFEST_FILENAME,
    }
    existing = [path for path in paths.values() if path.exists()]
    if existing:
        raise FileExistsError(
            "Refusing to overwrite existing MOD-07 outputs: "
            + ", ".join(str(path) for path in existing)
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    result.training_comparison.to_csv(paths["training_comparison"], index=False)
    result.training_audit.to_csv(paths["training_audit"], index=False)
    result.bootstrap_draws.to_csv(paths["bootstrap_draws"], index=False)
    result.bootstrap_replicates.to_csv(paths["bootstrap_replicates"], index=False)
    result.bootstrap_summary.to_csv(paths["bootstrap_summary"], index=False)
    with paths["manifest"].open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(result.manifest, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return paths


def build_cli_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--modeling-data", type=Path, required=True)
    parser.add_argument("--split-assignments", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--bootstrap-seed", type=int, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def main() -> int:
    args = build_cli_parser().parse_args()
    modeling_data = pd.read_csv(args.modeling_data)
    assignments = pd.read_csv(args.split_assignments)
    split_manifest = _load_split_manifest(args.split_manifest)

    (
        comparison,
        audit,
        draws,
        replicates,
        summary,
        training_ids,
        test_ids,
    ) = run_mod07(
        modeling_data,
        assignments,
        split_manifest,
        bootstrap_seed=args.bootstrap_seed,
        bootstrap_replicates=BOOTSTRAP_REPLICATES,
    )
    manifest = build_manifest(
        modeling_data_path=args.modeling_data,
        split_assignments_path=args.split_assignments,
        split_manifest_path=args.split_manifest,
        split_manifest=split_manifest,
        training_ids=training_ids,
        test_ids=test_ids,
        bootstrap_seed=args.bootstrap_seed,
        bootstrap_replicates=BOOTSTRAP_REPLICATES,
    )
    result = Mod07Result(
        training_comparison=comparison,
        training_audit=audit,
        bootstrap_draws=draws,
        bootstrap_replicates=replicates,
        bootstrap_summary=summary,
        manifest=manifest,
    )
    paths = write_outputs(result, args.output_dir)

    paired_training = int(comparison["paired_converged"].sum()) if len(comparison) else 0
    total_training = int(len(comparison))
    paired_bootstrap = int(replicates["paired_converged"].sum()) if len(replicates) else 0
    total_bootstrap = int(len(replicates))
    print(f"Wrote MOD-07 outputs to: {args.output_dir}")
    print(f"Training participants used: {len(training_ids)}")
    print(f"Held-out test participants used: 0 of {len(test_ids)}")
    print(f"Training RI/RI+RS pairs converged: {paired_training}/{total_training}")
    print(f"Bootstrap replicates per condition-target: {BOOTSTRAP_REPLICATES}")
    print(f"Bootstrap paired fits converged: {paired_bootstrap}/{total_bootstrap}")
    print(f"Manifest: {paths['manifest']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
