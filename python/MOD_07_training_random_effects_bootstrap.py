#!/usr/bin/env python3
"""MOD-07 — Training-only RI versus RI+RS comparison and selective bootstrap.

The immediate MOD-07 random-effects evidence base covers all 10 accepted
Gaussian targets (relative performance, mental demand, and eight continuous
behavioural features) across the three experimental conditions. The ordinary
training-only comparison is run separately from the participant bootstrap so
bootstrap stability evidence is generated only for explicitly selected
condition × target pairs.

Frozen methodological rules:
- use only the 26 participants assigned to training by MOD-06;
- never use the six held-out test participants for fitting or bootstrap;
- compare RI and RI+RS under the same difficulty-only M0 fixed-effects model;
- use ML estimation through the existing MOD-02 fitting helper;
- run L-BFGS and Powell and retain the converged fit with highest finite log
  likelihood;
- participant bootstrap uses B=2000, NumPy PCG64, seed 20260919;
- repeated draws of one original participant receive distinct temporary cluster
  IDs;
- evidence is reported without automatically selecting RI or RI+RS.

Bootstrap draws are generated centrally before worker processes start. This
makes the samples independent of worker count and task scheduling. Bootstrap
fitting supports process-based parallelism, parent-process checkpointing, and
resume after interruption.
"""
from __future__ import annotations

import os

# The bootstrap uses multiple Python worker processes. Force numerical libraries
# to one thread per worker to avoid nested BLAS/OpenMP oversubscription.
for _name in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ[_name] = "1"

import argparse
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

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


SCRIPT_VERSION = "2.1.0"
METHOD_ID = "training_m0_ri_vs_ri_rs_full_compare_selective_bootstrap_v3"
BOOTSTRAP_REPLICATES = 2000
BOOTSTRAP_SEED = 20260919
BOOTSTRAP_BIT_GENERATOR = "PCG64"

COMPARE_TARGETS = tuple(GAUSSIAN_TARGETS)

EXPECTED_TRAIN = 26
EXPECTED_TEST = 6
PARTICIPANT_COLUMN = "participant_id"
PARTITION_COLUMN = "partition"
BOOTSTRAP_CLUSTER_COLUMN = "bootstrap_cluster_id"

TRAINING_COMPARISON_FILENAME = "training_random_effects_comparison.csv"
TRAINING_AUDIT_FILENAME = "training_random_effects_audit.csv"
SELECTED_PAIRS_FILENAME = "bootstrap_selected_pairs.csv"
BOOTSTRAP_DRAWS_FILENAME = "bootstrap_participant_draws.csv"
BOOTSTRAP_REPLICATES_FILENAME = "bootstrap_random_effects_replicates.csv"
BOOTSTRAP_SUMMARY_FILENAME = "bootstrap_random_effects_summary.csv"
MANIFEST_FILENAME = "mod07_manifest.json"
CHECKPOINT_FILENAME = "bootstrap_random_effects_replicates.checkpoint.csv"
CHECKPOINT_MANIFEST_FILENAME = "mod07_bootstrap_checkpoint_manifest.json"


class Mod07Error(ValueError):
    """Raised when frozen MOD-07 invariants or run configuration are violated."""


@dataclass(frozen=True)
class TargetSpec:
    target: str
    difficulty_column: str
    formula: str


Pair = tuple[str, str]


def default_worker_count() -> int:
    """Return a conservative process count for CPU-bound bootstrap fitting."""
    logical = os.cpu_count() or 1
    return max(1, min(8, logical // 2 if logical > 1 else 1))


def build_m0_target_specs(
    targets: Iterable[str] = COMPARE_TARGETS,
) -> dict[str, TargetSpec]:
    """Return difficulty-only M0 formulas for requested Gaussian targets."""
    requested = tuple(targets)
    unknown = sorted(set(requested) - set(GAUSSIAN_TARGETS))
    if unknown:
        raise Mod07Error(f"Unsupported Gaussian targets: {unknown}")

    specs: dict[str, TargetSpec] = {}
    for target in requested:
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


def comparison_pairs() -> tuple[Pair, ...]:
    """Return all 30 condition × Gaussian-target pairs for Stage A."""
    return tuple(
        (condition, target)
        for condition in CONDITIONS
        for target in COMPARE_TARGETS
    )


def _load_json(path: Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise Mod07Error(f"{path} must contain a JSON object")
    return value


def validate_and_filter_training_data(
    modeling_data: pd.DataFrame,
    assignments: pd.DataFrame,
    split_manifest: Mapping[str, Any],
) -> tuple[pd.DataFrame, tuple[str, ...], tuple[str, ...]]:
    """Validate the frozen MOD-06 split and return only training rows."""
    required_assignment = {
        PARTICIPANT_COLUMN,
        "participant_group",
        PARTITION_COLUMN,
    }
    missing = sorted(required_assignment - set(assignments.columns))
    if missing:
        raise Mod07Error(f"Split assignment is missing required columns: {missing}")

    assignment_copy = assignments.copy(deep=True)
    if assignment_copy[PARTICIPANT_COLUMN].isna().any():
        raise Mod07Error("Split assignment contains missing participant_id")
    assignment_copy[PARTICIPANT_COLUMN] = (
        assignment_copy[PARTICIPANT_COLUMN].astype(str).str.strip()
    )
    if assignment_copy[PARTICIPANT_COLUMN].duplicated().any():
        raise Mod07Error("Split assignment contains duplicate participant_id values")
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
        mismatch = merged["participant_group_split"].ne(
            merged["participant_group_model"]
        )
        if mismatch.any():
            bad = sorted(merged.loc[mismatch, PARTICIPANT_COLUMN].astype(str))
            raise Mod07Error(f"participant_group mismatch for participants: {bad}")

    training_ids = tuple(
        sorted(
            assignment_copy.loc[
                assignment_copy[PARTITION_COLUMN].eq("train"),
                PARTICIPANT_COLUMN,
            ].astype(str)
        )
    )
    test_ids = tuple(
        sorted(
            assignment_copy.loc[
                assignment_copy[PARTITION_COLUMN].eq("test"),
                PARTICIPANT_COLUMN,
            ].astype(str)
        )
    )

    prepared = prepare_internal_modeling_data(modeling_data)
    training = prepared.loc[
        prepared[PARTICIPANT_COLUMN].astype(str).isin(training_ids)
    ].copy()

    if set(training[PARTICIPANT_COLUMN].astype(str)) != set(training_ids):
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
        d0_mask = pd.to_numeric(
            frame["difficulty_level"], errors="coerce"
        ).eq(0)
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
    rows = (
        frame[columns]
        .astype("string")
        .fillna("<NA>")
        .agg("|".join, axis=1)
        .sort_values(kind="mergesort")
        .tolist()
    )
    return sha256("\n".join(rows).encode("utf-8")).hexdigest()


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

    covariance_matrix = np.asarray(
        getattr(result, "cov_re", np.empty((0, 0)))
    )
    ri_variance = (
        _safe_float(covariance_matrix[0, 0])
        if covariance_matrix.ndim == 2 and covariance_matrix.shape[0] >= 1
        else np.nan
    )
    rs_variance = np.nan
    covariance = np.nan
    correlation = np.nan

    if (
        structure == "RI_RS"
        and covariance_matrix.ndim == 2
        and covariance_matrix.shape[0] >= 2
    ):
        rs_variance = _safe_float(covariance_matrix[1, 1])
        covariance = _safe_float(covariance_matrix[0, 1])
        if (
            np.isfinite(ri_variance)
            and np.isfinite(rs_variance)
            and ri_variance > 0
            and rs_variance > 0
            and np.isfinite(covariance)
        ):
            correlation = covariance / np.sqrt(
                ri_variance * rs_variance
            )

    return {
        "converged": bool(getattr(result, "converged", False)),
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
        for key, value in _extract_fit_metrics(result, structure).items():
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
    pairs: Sequence[Pair] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[Pair, pd.DataFrame]]:
    """Fit ordinary M0 RI/RI+RS pairs on the original training set."""
    selected = tuple(pairs if pairs is not None else comparison_pairs())
    targets = tuple(dict.fromkeys(target for _, target in selected))
    specs = build_m0_target_specs(targets)

    comparison_rows: list[dict[str, Any]] = []
    audit_rows: list[dict[str, Any]] = []
    prepared_frames: dict[Pair, pd.DataFrame] = {}

    for condition, target in selected:
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
                    "status": (
                        "comparable"
                        if pair["paired_converged"]
                        else "fit_problem"
                    ),
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
    bootstrap_replicates: int = BOOTSTRAP_REPLICATES,
    bootstrap_seed: int = BOOTSTRAP_SEED,
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


def _draws_signature(draws: pd.DataFrame) -> str:
    canonical = draws.sort_values(
        ["bootstrap_replicate", "draw_position"], kind="mergesort"
    )
    payload = "\n".join(
        f"{int(row.bootstrap_replicate)}|{int(row.draw_position)}|{row.participant_id}"
        for row in canonical.itertuples(index=False)
    )
    return sha256(payload.encode("utf-8")).hexdigest()


def _materialize_bootstrap_sample(
    fit_data: pd.DataFrame,
    draw_ids: Iterable[str],
) -> pd.DataFrame:
    pieces: list[pd.DataFrame] = []
    for draw_position, participant_id in enumerate(draw_ids, start=1):
        piece = fit_data.loc[
            fit_data[PARTICIPANT_COLUMN].astype(str).eq(
                str(participant_id)
            )
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


def load_selected_pairs(
    path: Path,
    *,
    allowed_targets: Iterable[str] = COMPARE_TARGETS,
    allowed_conditions: Iterable[str] = CONDITIONS,
) -> tuple[Pair, ...]:
    """Read and strictly validate selected condition × target pairs."""
    frame = pd.read_csv(path)
    required = ["condition_name", "target_name"]
    if list(frame.columns) != required:
        raise Mod07Error(
            "Selected-pairs CSV must contain exactly: "
            "condition_name,target_name"
        )
    if frame.empty:
        raise Mod07Error("Selected-pairs CSV contains no pairs")
    if frame[required].isna().any(axis=None):
        raise Mod07Error("Selected-pairs CSV contains missing values")

    frame = frame.astype(str)
    bad_conditions = sorted(
        set(frame["condition_name"]) - set(allowed_conditions)
    )
    bad_targets = sorted(
        set(frame["target_name"]) - set(allowed_targets)
    )
    if bad_conditions:
        raise Mod07Error(
            f"Selected-pairs CSV contains invalid conditions: {bad_conditions}"
        )
    if bad_targets:
        raise Mod07Error(
            "Selected-pairs CSV contains targets outside the accepted "
            f"Gaussian MixedLM scope: {bad_targets}"
        )
    if frame.duplicated(required).any():
        raise Mod07Error("Selected-pairs CSV contains duplicate pairs")

    return tuple(
        (row.condition_name, row.target_name)
        for row in frame.itertuples(index=False)
    )


def _prepare_frames_for_pairs(
    training_data: pd.DataFrame,
    pairs: Sequence[Pair],
) -> tuple[dict[Pair, pd.DataFrame], dict[str, TargetSpec]]:
    targets = tuple(dict.fromkeys(target for _, target in pairs))
    specs = build_m0_target_specs(targets)
    frames: dict[Pair, pd.DataFrame] = {}
    for condition, target in pairs:
        frame, _ = _prepare_target_frame(
            training_data, condition, specs[target]
        )
        frames[(condition, target)] = frame
    return frames, specs


_WORKER_FRAMES: dict[Pair, pd.DataFrame] = {}
_WORKER_SPECS: dict[str, TargetSpec] = {}


def _bootstrap_worker_initializer(
    frames: dict[Pair, pd.DataFrame],
    specs: dict[str, TargetSpec],
) -> None:
    global _WORKER_FRAMES, _WORKER_SPECS
    _WORKER_FRAMES = frames
    _WORKER_SPECS = specs


def _bootstrap_worker(
    task: tuple[int, str, str, tuple[str, ...]],
) -> dict[str, Any]:
    replicate, condition, target, draw_ids = task
    try:
        base_frame = _WORKER_FRAMES[(condition, target)]
        spec = _WORKER_SPECS[target]
        sample = _materialize_bootstrap_sample(base_frame, draw_ids)
        pair = _fit_pair(
            sample,
            spec,
            participant_column=BOOTSTRAP_CLUSTER_COLUMN,
        )
        return {
            "bootstrap_replicate": int(replicate),
            "condition_name": condition,
            "target_name": target,
            "fixed_effects_model": "M0_difficulty_only",
            "fixed_effects_formula": spec.formula,
            "difficulty_source_column": spec.difficulty_column,
            "bootstrap_draw_signature": _draw_signature(draw_ids),
            "unique_original_participants_drawn": len(set(draw_ids)),
            "bootstrap_cluster_count": int(
                sample[BOOTSTRAP_CLUSTER_COLUMN].nunique()
            ),
            "observation_count": int(len(sample)),
            "fit_status": (
                "paired_converged"
                if pair["paired_converged"]
                else "fit_problem"
            ),
            "task_error": "",
            **pair,
        }
    except Exception as exc:
        return {
            "bootstrap_replicate": int(replicate),
            "condition_name": condition,
            "target_name": target,
            "bootstrap_draw_signature": _draw_signature(draw_ids),
            "unique_original_participants_drawn": len(set(draw_ids)),
            "bootstrap_cluster_count": np.nan,
            "observation_count": np.nan,
            "paired_converged": False,
            "fit_status": "failed_exception",
            "task_error": f"{type(exc).__name__}: {exc}",
        }


def _bootstrap_tasks(
    draws: pd.DataFrame,
    pairs: Sequence[Pair],
) -> list[tuple[int, str, str, tuple[str, ...]]]:
    draw_map: dict[int, tuple[str, ...]] = {}
    for replicate, group in draws.groupby(
        "bootstrap_replicate", sort=True
    ):
        ordered = tuple(
            group.sort_values("draw_position", kind="mergesort")[
                PARTICIPANT_COLUMN
            ].astype(str)
        )
        draw_map[int(replicate)] = ordered

    return [
        (replicate, condition, target, draw_map[replicate])
        for replicate in sorted(draw_map)
        for condition, target in pairs
    ]


def _task_key(row: Mapping[str, Any]) -> tuple[int, str, str]:
    return (
        int(row["bootstrap_replicate"]),
        str(row["condition_name"]),
        str(row["target_name"]),
    )


def _write_csv_atomic(frame: pd.DataFrame, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    try:
        frame.to_csv(tmp, index=False)
        tmp.replace(path)
    finally:
        tmp.unlink(missing_ok=True)


def _write_json_atomic(value: Mapping[str, Any], path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    try:
        with tmp.open("w", encoding="utf-8", newline="\n") as handle:
            json.dump(dict(value), handle, indent=2, sort_keys=True)
            handle.write("\n")
        tmp.replace(path)
    finally:
        tmp.unlink(missing_ok=True)


def _config_sha256(value: Mapping[str, Any]) -> str:
    payload = json.dumps(
        dict(value),
        sort_keys=True,
        separators=(",", ":"),
    )
    return sha256(payload.encode("utf-8")).hexdigest()


def build_bootstrap_configuration(
    *,
    pairs: Sequence[Pair],
    frames: Mapping[Pair, pd.DataFrame],
    training_ids: Sequence[str],
    split_manifest: Mapping[str, Any],
    draws: pd.DataFrame,
    bootstrap_replicates: int,
    bootstrap_seed: int,
) -> dict[str, Any]:
    """Build resume-critical configuration independent of worker count."""
    configuration = {
        "schema_version": 1,
        "method_id": METHOD_ID,
        "script_version": SCRIPT_VERSION,
        "fixed_effects_model": "M0_difficulty_only",
        "random_structures_compared": ["RI", "RI_RS"],
        "bootstrap_unit": "participant",
        "bootstrap_replicates": int(bootstrap_replicates),
        "bootstrap_seed": int(bootstrap_seed),
        "bootstrap_bit_generator": BOOTSTRAP_BIT_GENERATOR,
        "training_participants": list(training_ids),
        "selected_pairs": [
            {"condition_name": condition, "target_name": target}
            for condition, target in pairs
        ],
        "pair_row_signatures": {
            f"{condition}|{target}": _row_signature(
                frames[(condition, target)]
            )
            for condition, target in pairs
        },
        "draws_sha256": _draws_signature(draws),
        "mod06_assignment_sha256": split_manifest.get("assignment_sha256"),
        "mod06_participant_source_sha256": split_manifest.get(
            "participant_source_sha256"
        ),
        "automatic_random_structure_selection": False,
    }
    configuration["configuration_sha256"] = _config_sha256(
        configuration
    )
    return configuration


def _validate_resume_files(
    *,
    output_dir: Path,
    configuration: Mapping[str, Any],
    draws: pd.DataFrame,
) -> pd.DataFrame:
    checkpoint_manifest_path = (
        output_dir / CHECKPOINT_MANIFEST_FILENAME
    )
    draws_path = output_dir / BOOTSTRAP_DRAWS_FILENAME
    checkpoint_path = output_dir / CHECKPOINT_FILENAME

    if not checkpoint_manifest_path.exists() or not draws_path.exists():
        raise Mod07Error(
            "Resume requires the checkpoint manifest and bootstrap draw file"
        )

    existing_manifest = _load_json(checkpoint_manifest_path)
    if existing_manifest.get("configuration_sha256") != configuration.get(
        "configuration_sha256"
    ):
        raise Mod07Error(
            "Existing checkpoint configuration does not match this run"
        )

    existing_draws = pd.read_csv(draws_path)
    if _draws_signature(existing_draws) != _draws_signature(draws):
        raise Mod07Error(
            "Existing bootstrap draws do not match the frozen run configuration"
        )

    if not checkpoint_path.exists():
        return pd.DataFrame()
    checkpoint = pd.read_csv(checkpoint_path)
    if checkpoint.empty:
        return checkpoint

    required = {
        "bootstrap_replicate",
        "condition_name",
        "target_name",
    }
    if not required.issubset(checkpoint.columns):
        raise Mod07Error("Checkpoint is missing task identity columns")
    if checkpoint.duplicated(list(required)).any():
        raise Mod07Error("Checkpoint contains duplicate completed tasks")
    return checkpoint


def _initialise_bootstrap_outputs(
    *,
    output_dir: Path,
    pairs: Sequence[Pair],
    draws: pd.DataFrame,
    configuration: Mapping[str, Any],
    resume: bool,
) -> pd.DataFrame:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    protected = [
        output_dir / SELECTED_PAIRS_FILENAME,
        output_dir / BOOTSTRAP_DRAWS_FILENAME,
        output_dir / BOOTSTRAP_REPLICATES_FILENAME,
        output_dir / BOOTSTRAP_SUMMARY_FILENAME,
        output_dir / MANIFEST_FILENAME,
        output_dir / CHECKPOINT_FILENAME,
        output_dir / CHECKPOINT_MANIFEST_FILENAME,
    ]

    if resume:
        if (output_dir / BOOTSTRAP_REPLICATES_FILENAME).exists():
            raise Mod07Error(
                "Final bootstrap output already exists; nothing to resume"
            )
        return _validate_resume_files(
            output_dir=output_dir,
            configuration=configuration,
            draws=draws,
        )

    existing = [path for path in protected if path.exists()]
    if existing:
        raise FileExistsError(
            "Refusing to overwrite existing MOD-07 bootstrap outputs: "
            + ", ".join(str(path) for path in existing)
        )

    selected = pd.DataFrame(
        pairs, columns=["condition_name", "target_name"]
    )
    _write_csv_atomic(selected, output_dir / SELECTED_PAIRS_FILENAME)
    _write_csv_atomic(draws, output_dir / BOOTSTRAP_DRAWS_FILENAME)
    _write_json_atomic(
        configuration,
        output_dir / CHECKPOINT_MANIFEST_FILENAME,
    )
    return pd.DataFrame()


def run_bootstrap_tasks(
    *,
    frames: dict[Pair, pd.DataFrame],
    specs: dict[str, TargetSpec],
    draws: pd.DataFrame,
    pairs: Sequence[Pair],
    workers: int,
    existing_results: pd.DataFrame | None = None,
    checkpoint_path: Path | None = None,
    checkpoint_every: int = 50,
) -> pd.DataFrame:
    """Run missing bootstrap tasks serially or with process-based parallelism."""
    if workers < 1:
        raise Mod07Error("workers must be at least 1")
    if checkpoint_every < 1:
        raise Mod07Error("checkpoint_every must be at least 1")

    completed = (
        existing_results.copy(deep=True)
        if existing_results is not None
        else pd.DataFrame()
    )
    completed_keys: set[tuple[int, str, str]] = set()
    if not completed.empty:
        completed_keys = {
            _task_key(row)
            for row in completed.to_dict(orient="records")
        }

    tasks = [
        task
        for task in _bootstrap_tasks(draws, pairs)
        if (task[0], task[1], task[2]) not in completed_keys
    ]
    new_rows: list[dict[str, Any]] = []

    def checkpoint() -> None:
        if checkpoint_path is None:
            return
        pieces = [
            frame
            for frame in (completed, pd.DataFrame(new_rows))
            if not frame.empty
        ]
        combined = (
            pd.concat(pieces, ignore_index=True)
            if pieces
            else pd.DataFrame()
        )
        if not combined.empty:
            combined = combined.sort_values(
                ["bootstrap_replicate", "condition_name", "target_name"],
                kind="mergesort",
            ).reset_index(drop=True)
        _write_csv_atomic(combined, checkpoint_path)

    if workers == 1:
        _bootstrap_worker_initializer(frames, specs)
        iterator = map(_bootstrap_worker, tasks)
        for index, row in enumerate(iterator, start=1):
            new_rows.append(row)
            if index % checkpoint_every == 0:
                checkpoint()
    else:
        with ProcessPoolExecutor(
            max_workers=workers,
            initializer=_bootstrap_worker_initializer,
            initargs=(frames, specs),
        ) as executor:
            iterator = executor.map(
                _bootstrap_worker,
                tasks,
                chunksize=1,
            )
            for index, row in enumerate(iterator, start=1):
                new_rows.append(row)
                if index % checkpoint_every == 0:
                    checkpoint()

    checkpoint()

    pieces = [
        frame
        for frame in (completed, pd.DataFrame(new_rows))
        if not frame.empty
    ]
    combined = (
        pd.concat(pieces, ignore_index=True)
        if pieces
        else pd.DataFrame()
    )
    if not combined.empty:
        combined = combined.sort_values(
            ["bootstrap_replicate", "condition_name", "target_name"],
            kind="mergesort",
        ).reset_index(drop=True)
        if combined.duplicated(
            ["bootstrap_replicate", "condition_name", "target_name"]
        ).any():
            raise Mod07Error(
                "Bootstrap execution produced duplicate task rows"
            )
    return combined


def summarize_bootstrap_replicates(
    replicate_results: pd.DataFrame,
) -> pd.DataFrame:
    """Return descriptive stability summaries without selecting a winner or CI."""
    rows: list[dict[str, Any]] = []
    for (condition, target), group in replicate_results.groupby(
        ["condition_name", "target_name"], sort=True
    ):
        paired = group.loc[
            group["paired_converged"].eq(True)
        ].copy()
        delta_aic = pd.to_numeric(
            paired["delta_aic_ri_rs_minus_ri"],
            errors="coerce",
        ).dropna()
        delta_bic = pd.to_numeric(
            paired["delta_bic_ri_rs_minus_ri"],
            errors="coerce",
        ).dropna()
        slope = pd.to_numeric(
            paired["ri_rs_random_slope_variance"],
            errors="coerce",
        ).dropna()

        rows.append(
            {
                "condition_name": condition,
                "target_name": target,
                "bootstrap_replicates_requested": int(len(group)),
                "paired_converged_count": int(len(paired)),
                "paired_converged_percentage": (
                    100.0 * len(paired) / len(group)
                    if len(group)
                    else np.nan
                ),
                "valid_aic_pair_count": int(len(delta_aic)),
                "ri_rs_lower_aic_count": int((delta_aic < 0).sum()),
                "ri_rs_lower_aic_percentage_of_valid": (
                    100.0 * (delta_aic < 0).mean()
                    if len(delta_aic)
                    else np.nan
                ),
                "delta_aic_median": (
                    float(delta_aic.median())
                    if len(delta_aic)
                    else np.nan
                ),
                "delta_aic_q25": (
                    float(delta_aic.quantile(0.25))
                    if len(delta_aic)
                    else np.nan
                ),
                "delta_aic_q75": (
                    float(delta_aic.quantile(0.75))
                    if len(delta_aic)
                    else np.nan
                ),
                "valid_bic_pair_count": int(len(delta_bic)),
                "ri_rs_lower_bic_count": int((delta_bic < 0).sum()),
                "ri_rs_lower_bic_percentage_of_valid": (
                    100.0 * (delta_bic < 0).mean()
                    if len(delta_bic)
                    else np.nan
                ),
                "delta_bic_median": (
                    float(delta_bic.median())
                    if len(delta_bic)
                    else np.nan
                ),
                "delta_bic_q25": (
                    float(delta_bic.quantile(0.25))
                    if len(delta_bic)
                    else np.nan
                ),
                "delta_bic_q75": (
                    float(delta_bic.quantile(0.75))
                    if len(delta_bic)
                    else np.nan
                ),
                "ri_rs_random_slope_variance_median": (
                    float(slope.median()) if len(slope) else np.nan
                ),
                "selection_status": (
                    "evidence_only_no_automatic_selection"
                ),
                "confidence_interval_status": (
                    "not_frozen_not_computed"
                ),
            }
        )
    return pd.DataFrame(rows)


def build_manifest(
    *,
    mode: str,
    modeling_data_path: Path,
    split_assignments_path: Path,
    split_manifest_path: Path,
    split_manifest: Mapping[str, Any],
    training_ids: Sequence[str],
    test_ids: Sequence[str],
    pairs: Sequence[Pair],
    workers: int | None = None,
    bootstrap_replicates: int | None = None,
    bootstrap_seed: int | None = None,
    configuration_sha256: str | None = None,
) -> dict[str, Any]:
    return {
        "schema_version": 2,
        "script_version": SCRIPT_VERSION,
        "method_id": METHOD_ID,
        "mode": mode,
        "fixed_effects_model": "M0_difficulty_only",
        "random_structures_compared": ["RI", "RI_RS"],
        "estimation_method": "maximum_likelihood",
        "optimizer_rule": (
            "run lbfgs and powell; retain converged fit with highest finite "
            "log likelihood"
        ),
        "analysis_pairs": [
            {"condition_name": condition, "target_name": target}
            for condition, target in pairs
        ],
        "bootstrap_unit": (
            "participant" if mode == "bootstrap-selected" else None
        ),
        "bootstrap_replicates": bootstrap_replicates,
        "bootstrap_seed": bootstrap_seed,
        "bootstrap_seed_status": (
            "frozen" if bootstrap_seed is not None else None
        ),
        "bootstrap_bit_generator": (
            BOOTSTRAP_BIT_GENERATOR
            if bootstrap_seed is not None
            else None
        ),
        "workers": workers,
        "blas_threads_per_worker_target": 1,
        "training_participant_count": len(training_ids),
        "test_participant_count": len(test_ids),
        "training_participants": list(training_ids),
        "held_out_test_participants": list(test_ids),
        "test_participants_used_for_fitting": False,
        "test_participants_used_for_bootstrap": False,
        "automatic_random_structure_selection": False,
        "confidence_interval_method": None,
        "confidence_interval_status": "not_yet_frozen",
        "configuration_sha256": configuration_sha256,
        "mod06_assignment_sha256": split_manifest.get("assignment_sha256"),
        "mod06_participant_source_sha256": split_manifest.get(
            "participant_source_sha256"
        ),
        "modeling_data_path": str(modeling_data_path),
        "split_assignments_path": str(split_assignments_path),
        "split_manifest_path": str(split_manifest_path),
    }


def write_compare_outputs(
    *,
    comparison: pd.DataFrame,
    audit: pd.DataFrame,
    manifest: Mapping[str, Any],
    output_dir: Path,
) -> dict[str, Path]:
    output_dir = Path(output_dir)
    paths = {
        "comparison": output_dir / TRAINING_COMPARISON_FILENAME,
        "audit": output_dir / TRAINING_AUDIT_FILENAME,
        "manifest": output_dir / MANIFEST_FILENAME,
    }
    existing = [path for path in paths.values() if path.exists()]
    if existing:
        raise FileExistsError(
            "Refusing to overwrite existing MOD-07 compare-only outputs: "
            + ", ".join(str(path) for path in existing)
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv_atomic(comparison, paths["comparison"])
    _write_csv_atomic(audit, paths["audit"])
    _write_json_atomic(manifest, paths["manifest"])
    return paths


def run_compare_only(
    *,
    modeling_data: pd.DataFrame,
    assignments: pd.DataFrame,
    split_manifest: Mapping[str, Any],
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    tuple[str, ...],
    tuple[str, ...],
]:
    training, training_ids, test_ids = validate_and_filter_training_data(
        modeling_data, assignments, split_manifest
    )
    comparison, audit, _ = compare_training_random_effects(
        training, pairs=comparison_pairs()
    )
    return comparison, audit, training_ids, test_ids


def run_bootstrap_selected(
    *,
    modeling_data: pd.DataFrame,
    assignments: pd.DataFrame,
    split_manifest: Mapping[str, Any],
    pairs: Sequence[Pair],
    output_dir: Path,
    workers: int,
    resume: bool,
    checkpoint_every: int = 50,
    bootstrap_replicates: int = BOOTSTRAP_REPLICATES,
    bootstrap_seed: int = BOOTSTRAP_SEED,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    dict[str, Any],
    tuple[str, ...],
    tuple[str, ...],
]:
    if int(bootstrap_seed) != BOOTSTRAP_SEED:
        raise Mod07Error(
            f"Bootstrap seed is frozen at {BOOTSTRAP_SEED}"
        )

    training, training_ids, test_ids = validate_and_filter_training_data(
        modeling_data, assignments, split_manifest
    )
    frames, specs = _prepare_frames_for_pairs(training, pairs)
    draws = generate_bootstrap_draws(
        training_ids,
        bootstrap_replicates=bootstrap_replicates,
        bootstrap_seed=bootstrap_seed,
    )
    configuration = build_bootstrap_configuration(
        pairs=pairs,
        frames=frames,
        training_ids=training_ids,
        split_manifest=split_manifest,
        draws=draws,
        bootstrap_replicates=bootstrap_replicates,
        bootstrap_seed=bootstrap_seed,
    )
    existing = _initialise_bootstrap_outputs(
        output_dir=output_dir,
        pairs=pairs,
        draws=draws,
        configuration=configuration,
        resume=resume,
    )

    checkpoint_path = Path(output_dir) / CHECKPOINT_FILENAME
    replicates = run_bootstrap_tasks(
        frames=frames,
        specs=specs,
        draws=draws,
        pairs=pairs,
        workers=workers,
        existing_results=existing,
        checkpoint_path=checkpoint_path,
        checkpoint_every=checkpoint_every,
    )

    expected_tasks = bootstrap_replicates * len(pairs)
    if len(replicates) != expected_tasks:
        raise Mod07Error(
            f"Expected {expected_tasks} bootstrap task rows, "
            f"found {len(replicates)}"
        )

    summary = summarize_bootstrap_replicates(replicates)
    _write_csv_atomic(
        replicates,
        Path(output_dir) / BOOTSTRAP_REPLICATES_FILENAME,
    )
    _write_csv_atomic(
        summary,
        Path(output_dir) / BOOTSTRAP_SUMMARY_FILENAME,
    )

    manifest = {
        "configuration_sha256": configuration["configuration_sha256"],
        "selected_pairs": configuration["selected_pairs"],
    }
    return replicates, summary, manifest, training_ids, test_ids


def build_cli_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        choices=("compare-only", "bootstrap-selected"),
        required=True,
    )
    parser.add_argument("--modeling-data", type=Path, required=True)
    parser.add_argument("--split-assignments", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--selected-pairs",
        type=Path,
        help=(
            "CSV with exactly condition_name,target_name; required for "
            "bootstrap-selected."
        ),
    )
    parser.add_argument(
        "--bootstrap-seed",
        type=int,
        default=BOOTSTRAP_SEED,
        help=(
            f"Frozen bootstrap seed. Only {BOOTSTRAP_SEED} is accepted."
        ),
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=default_worker_count(),
        help=(
            "Bootstrap worker processes. Default uses a conservative fraction "
            "of available logical CPUs."
        ),
    )
    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=50,
        help="Parent-process checkpoint frequency in completed bootstrap tasks.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume a compatible interrupted bootstrap-selected run.",
    )
    return parser


def main() -> int:
    args = build_cli_parser().parse_args()
    if args.bootstrap_seed != BOOTSTRAP_SEED:
        raise Mod07Error(
            f"Bootstrap seed is frozen at {BOOTSTRAP_SEED}; "
            f"received {args.bootstrap_seed}"
        )
    if args.mode == "compare-only" and args.resume:
        raise Mod07Error("--resume is only valid with bootstrap-selected")
    if args.mode == "bootstrap-selected" and args.selected_pairs is None:
        raise Mod07Error(
            "--selected-pairs is required with bootstrap-selected"
        )

    modeling_data = pd.read_csv(args.modeling_data)
    assignments = pd.read_csv(args.split_assignments)
    split_manifest = _load_json(args.split_manifest)

    if args.mode == "compare-only":
        comparison, audit, training_ids, test_ids = run_compare_only(
            modeling_data=modeling_data,
            assignments=assignments,
            split_manifest=split_manifest,
        )
        manifest = build_manifest(
            mode="compare-only",
            modeling_data_path=args.modeling_data,
            split_assignments_path=args.split_assignments,
            split_manifest_path=args.split_manifest,
            split_manifest=split_manifest,
            training_ids=training_ids,
            test_ids=test_ids,
            pairs=comparison_pairs(),
        )
        paths = write_compare_outputs(
            comparison=comparison,
            audit=audit,
            manifest=manifest,
            output_dir=args.output_dir,
        )
        paired = (
            int(comparison["paired_converged"].sum())
            if len(comparison)
            else 0
        )
        print(f"Wrote MOD-07 compare-only outputs to: {args.output_dir}")
        print(f"Gaussian condition-target pairs: {len(comparison_pairs())}")
        print(f"Training participants used: {len(training_ids)}")
        print(f"Held-out test participants used: 0 of {len(test_ids)}")
        print(f"RI/RI+RS pairs converged: {paired}/{len(comparison)}")
        print(f"Manifest: {paths['manifest']}")
        return 0

    pairs = load_selected_pairs(args.selected_pairs)
    (
        replicates,
        summary,
        bootstrap_manifest,
        training_ids,
        test_ids,
    ) = run_bootstrap_selected(
        modeling_data=modeling_data,
        assignments=assignments,
        split_manifest=split_manifest,
        pairs=pairs,
        output_dir=args.output_dir,
        workers=args.workers,
        resume=args.resume,
        checkpoint_every=args.checkpoint_every,
        bootstrap_replicates=BOOTSTRAP_REPLICATES,
        bootstrap_seed=BOOTSTRAP_SEED,
    )

    manifest = build_manifest(
        mode="bootstrap-selected",
        modeling_data_path=args.modeling_data,
        split_assignments_path=args.split_assignments,
        split_manifest_path=args.split_manifest,
        split_manifest=split_manifest,
        training_ids=training_ids,
        test_ids=test_ids,
        pairs=pairs,
        workers=args.workers,
        bootstrap_replicates=BOOTSTRAP_REPLICATES,
        bootstrap_seed=BOOTSTRAP_SEED,
        configuration_sha256=bootstrap_manifest["configuration_sha256"],
    )
    manifest["selected_pairs_path"] = str(args.selected_pairs)
    _write_json_atomic(
        manifest,
        args.output_dir / MANIFEST_FILENAME,
    )

    # Successful finalization: checkpoint state is no longer needed.
    (args.output_dir / CHECKPOINT_FILENAME).unlink(missing_ok=True)
    (args.output_dir / CHECKPOINT_MANIFEST_FILENAME).unlink(missing_ok=True)

    paired = (
        int(replicates["paired_converged"].sum())
        if len(replicates)
        else 0
    )
    print(f"Wrote MOD-07 bootstrap outputs to: {args.output_dir}")
    print(f"Selected condition-outcome pairs: {len(pairs)}")
    print(f"Bootstrap replicates per selected pair: {BOOTSTRAP_REPLICATES}")
    print(f"Worker processes: {args.workers}")
    print(f"Training participants used: {len(training_ids)}")
    print(f"Held-out test participants used: 0 of {len(test_ids)}")
    print(f"Bootstrap paired fits converged: {paired}/{len(replicates)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
