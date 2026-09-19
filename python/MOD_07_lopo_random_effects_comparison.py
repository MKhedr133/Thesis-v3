#!/usr/bin/env python3
"""MOD-07 — LOPO comparison of RI versus RI+RS under M0.

This module implements the revised random-effects selection workflow.

Method
------
- use all 32 participants;
- compare only the difficulty-only M0 fixed-effects model;
- evaluate RI and RI+RS with leave-one-participant-out (LOPO) prediction;
- in each fold, fit on 31 participants and predict the held-out participant
  from fixed effects only;
- report participant-balanced LOPO MAE and pooled held-out LOPO R²;
- fit RI and RI+RS once on all 32 participants to report AIC, BIC,
  convergence, optimizer, warnings, and random-effects covariance estimates;
- report signed deltas without automatically selecting a structure.

Delta conventions
-----------------
delta_lopo_mae_ri_rs_minus_ri = MAE(RI+RS) - MAE(RI)
    Negative values favour RI+RS.

delta_lopo_r2_ri_rs_minus_ri = R²(RI+RS) - R²(RI)
    Positive values favour RI+RS.

delta_aic_ri_rs_minus_ri = AIC(RI+RS) - AIC(RI)
delta_bic_ri_rs_minus_ri = BIC(RI+RS) - BIC(RI)
    Negative values favour RI+RS.

The purpose of MOD-07 is evidence generation only. The script does not choose
RI or RI+RS automatically.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

import MOD_02_random_effects_structure as MOD02
import MOD_05_lopo_predictive_comparison as MOD05


SCRIPT_VERSION = "3.0.0"
METHOD_ID = "all32_m0_lopo_ri_vs_ri_rs_v1"
EXPECTED_PARTICIPANTS = 32
PARTICIPANT_COLUMN = "participant_id"
COMPARE_TARGETS = tuple(MOD02.GAUSSIAN_TARGETS)
RANDOM_STRUCTURES = ("RI", "RI_RS")

COMPARISON_FILENAME = "lopo_random_effects_comparison.csv"
AUDIT_FILENAME = "lopo_random_effects_audit.csv"
PREDICTIONS_FILENAME = "lopo_predictions.csv"
PARTICIPANT_ERRORS_FILENAME = "lopo_participant_errors.csv"
LOPO_CHECKS_FILENAME = "lopo_checks.csv"
MANIFEST_FILENAME = "mod07_manifest.json"


class Mod07Error(ValueError):
    """Raised when the revised MOD-07 analysis contract is violated."""


@dataclass(frozen=True)
class TargetSpec:
    target: str
    difficulty_column: str
    formula: str


Pair = tuple[str, str]


def build_m0_target_specs(
    targets: Iterable[str] = COMPARE_TARGETS,
) -> dict[str, TargetSpec]:
    """Return difficulty-only M0 formulas for requested Gaussian targets."""
    requested = tuple(targets)
    unknown = sorted(set(requested) - set(MOD02.GAUSSIAN_TARGETS))
    if unknown:
        raise Mod07Error(f"Unsupported Gaussian targets: {unknown}")

    specs: dict[str, TargetSpec] = {}
    for target in requested:
        difficulty_column = (
            MOD02.PERFORMANCE_DIFFICULTY_COLUMN
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
    """Return all 30 condition × Gaussian-target pairs."""
    return tuple(
        (condition, target)
        for condition in MOD02.CONDITIONS
        for target in COMPARE_TARGETS
    )


def validate_all_participants(modeling_data: pd.DataFrame) -> tuple[str, ...]:
    """Require the complete 32-participant modelling dataset."""
    if PARTICIPANT_COLUMN not in modeling_data.columns:
        raise Mod07Error(f"Missing required column: {PARTICIPANT_COLUMN}")

    ids = tuple(
        sorted(
            modeling_data[PARTICIPANT_COLUMN]
            .dropna()
            .astype(str)
            .str.strip()
            .unique()
            .tolist()
        )
    )
    if len(ids) != EXPECTED_PARTICIPANTS:
        raise Mod07Error(
            "MOD-07 LOPO requires all 32 participants; "
            f"found {len(ids)} unique participants"
        )
    if any(not participant_id for participant_id in ids):
        raise Mod07Error("participant_id contains blank values")
    return ids


def _prepare_target_frame(
    prepared_data: pd.DataFrame,
    condition: str,
    spec: TargetSpec,
) -> tuple[pd.DataFrame, int]:
    """Create the exact full-data M0 frame for one condition and target."""
    frame = prepared_data.loc[
        prepared_data["condition_name"].eq(condition)
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

    participant_count = int(frame[PARTICIPANT_COLUMN].nunique())
    if participant_count != EXPECTED_PARTICIPANTS:
        raise Mod07Error(
            f"{condition}/{spec.target}: LOPO requires 32 participants, "
            f"but the eligible target frame contains {participant_count}"
        )
    return frame, structural_excluded


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
            correlation = covariance / np.sqrt(ri_variance * rs_variance)

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


def fit_full_data_m0(
    modeling_data: pd.DataFrame,
    *,
    pairs: Sequence[Pair] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fit full-data M0 under RI and RI+RS for AIC/BIC and diagnostics."""
    validate_all_participants(modeling_data)
    prepared = MOD02.prepare_internal_modeling_data(modeling_data)
    selected = tuple(pairs if pairs is not None else comparison_pairs())
    targets = tuple(dict.fromkeys(target for _, target in selected))
    specs = build_m0_target_specs(targets)

    result_rows: list[dict[str, Any]] = []
    audit_rows: list[dict[str, Any]] = []

    for condition, target in selected:
        spec = specs[target]
        try:
            fit_data, structural_excluded = _prepare_target_frame(
                prepared, condition, spec
            )
            row: dict[str, Any] = {
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
            }

            paired_converged = True
            for prefix, structure in (("ri", "RI"), ("ri_rs", "RI_RS")):
                result, optimizer, fit_warnings, fit_errors = (
                    MOD02.fit_mixedlm_with_fallback(
                        spec.formula,
                        fit_data,
                        structure,
                        spec.difficulty_column,
                    )
                )
                metrics = _extract_fit_metrics(result, structure)
                paired_converged = paired_converged and bool(
                    metrics["converged"]
                )
                for key, value in metrics.items():
                    row[f"{prefix}_{key}"] = value
                row[f"{prefix}_optimizer"] = optimizer
                row[f"{prefix}_warnings"] = " | ".join(fit_warnings)
                row[f"{prefix}_errors"] = " | ".join(fit_errors)

            row["paired_converged"] = paired_converged
            if paired_converged:
                row["delta_aic_ri_rs_minus_ri"] = (
                    row["ri_rs_aic"] - row["ri_aic"]
                    if np.isfinite(row["ri_rs_aic"])
                    and np.isfinite(row["ri_aic"])
                    else np.nan
                )
                row["delta_bic_ri_rs_minus_ri"] = (
                    row["ri_rs_bic"] - row["ri_bic"]
                    if np.isfinite(row["ri_rs_bic"])
                    and np.isfinite(row["ri_bic"])
                    else np.nan
                )
            else:
                row["delta_aic_ri_rs_minus_ri"] = np.nan
                row["delta_bic_ri_rs_minus_ri"] = np.nan

            result_rows.append(row)
            audit_rows.append(
                {
                    "condition_name": condition,
                    "target_name": target,
                    "status": (
                        "pass" if paired_converged else "fit_problem"
                    ),
                    "participant_count": row["participant_count"],
                    "observation_count": row["observation_count"],
                    "message": (
                        "Full-data RI and RI+RS M0 fits used identical rows."
                        if paired_converged
                        else "One or both full-data M0 fits did not converge."
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
                    "message": f"{type(exc).__name__}: {exc}",
                }
            )

    return pd.DataFrame(result_rows), pd.DataFrame(audit_rows)


def _structure_summary(
    lopo_summary: pd.DataFrame,
    structure: str,
    prefix: str,
) -> pd.DataFrame:
    """Return M0 LOPO summary columns for one random structure."""
    required = {
        "condition_name",
        "target_name",
        "random_structure",
        "model_id",
        "participant_count",
        "participant_balanced_mae",
        "lopo_r2",
    }
    missing = sorted(required - set(lopo_summary.columns))
    if missing:
        raise Mod07Error(f"LOPO summary missing required columns: {missing}")

    subset = lopo_summary.loc[
        lopo_summary["random_structure"].eq(structure)
        & lopo_summary["model_id"].eq("M0")
    ].copy()
    if "prediction_count" not in subset.columns:
        subset["prediction_count"] = np.nan
    if "participant_balanced_rmse" not in subset.columns:
        subset["participant_balanced_rmse"] = np.nan

    keep = [
        "condition_name",
        "target_name",
        "participant_count",
        "prediction_count",
        "participant_balanced_mae",
        "participant_balanced_rmse",
        "lopo_r2",
    ]
    subset = subset[keep].rename(
        columns={
            "participant_count": f"{prefix}_lopo_participant_count",
            "prediction_count": f"{prefix}_lopo_prediction_count",
            "participant_balanced_mae": f"{prefix}_lopo_mae",
            "participant_balanced_rmse": f"{prefix}_lopo_rmse",
            "lopo_r2": f"{prefix}_lopo_r2",
        }
    )
    return subset


def combine_full_fit_and_lopo(
    full_fit: pd.DataFrame,
    lopo_summary: pd.DataFrame,
) -> pd.DataFrame:
    """Combine full-data information criteria with out-of-fold metrics."""
    ri = _structure_summary(lopo_summary, "RI", "ri")
    ri_rs = _structure_summary(lopo_summary, "RI_RS", "ri_rs")

    combined = full_fit.merge(
        ri,
        on=["condition_name", "target_name"],
        how="left",
        validate="one_to_one",
    ).merge(
        ri_rs,
        on=["condition_name", "target_name"],
        how="left",
        validate="one_to_one",
    )

    combined["delta_lopo_mae_ri_rs_minus_ri"] = (
        combined["ri_rs_lopo_mae"] - combined["ri_lopo_mae"]
    )
    combined["delta_lopo_r2_ri_rs_minus_ri"] = (
        combined["ri_rs_lopo_r2"] - combined["ri_lopo_r2"]
    )
    combined["selection_status"] = "evidence_only_no_automatic_selection"
    combined["lopo_prediction_scope"] = "fixed_effect_population_only"
    return combined


def build_audit(
    full_fit_audit: pd.DataFrame,
    lopo_checks: pd.DataFrame,
    comparison: pd.DataFrame,
) -> pd.DataFrame:
    """Create one audit row per condition × target pair."""
    rows: list[dict[str, Any]] = []
    for condition, target in comparison_pairs():
        full_rows = full_fit_audit.loc[
            full_fit_audit["condition_name"].eq(condition)
            & full_fit_audit["target_name"].eq(target)
        ]
        check_rows = lopo_checks.loc[
            lopo_checks["condition_name"].eq(condition)
            & lopo_checks["target_name"].eq(target)
            & lopo_checks["random_structure"].isin(RANDOM_STRUCTURES)
        ]
        comparison_rows = comparison.loc[
            comparison["condition_name"].eq(condition)
            & comparison["target_name"].eq(target)
        ]

        full_ok = (
            len(full_rows) == 1
            and str(full_rows.iloc[0]["status"]) == "pass"
        )
        lopo_ok = (
            len(check_rows) == 2
            and check_rows["status"].eq("pass").all()
        )
        complete_summary = False
        if len(comparison_rows) == 1:
            row = comparison_rows.iloc[0]
            complete_summary = (
                int(row.get("ri_lopo_participant_count", 0))
                == EXPECTED_PARTICIPANTS
                and int(row.get("ri_rs_lopo_participant_count", 0))
                == EXPECTED_PARTICIPANTS
            )

        status = "pass" if full_ok and lopo_ok and complete_summary else "problem"
        rows.append(
            {
                "condition_name": condition,
                "target_name": target,
                "status": status,
                "full_data_pair_ok": full_ok,
                "ri_lopo_check_ok": bool(
                    (
                        check_rows.loc[
                            check_rows["random_structure"].eq("RI"),
                            "status",
                        ]
                        == "pass"
                    ).all()
                )
                if len(check_rows.loc[
                    check_rows["random_structure"].eq("RI")
                ])
                else False,
                "ri_rs_lopo_check_ok": bool(
                    (
                        check_rows.loc[
                            check_rows["random_structure"].eq("RI_RS"),
                            "status",
                        ]
                        == "pass"
                    ).all()
                )
                if len(check_rows.loc[
                    check_rows["random_structure"].eq("RI_RS")
                ])
                else False,
                "complete_32_participant_lopo": complete_summary,
                "message": (
                    "Full-data comparison and both 32-fold LOPO structures completed."
                    if status == "pass"
                    else "Review full-data or LOPO diagnostics before interpretation."
                ),
            }
        )
    return pd.DataFrame(rows)


def run_mod07(
    modeling_data: pd.DataFrame,
    *,
    jobs: int = 1,
    checkpoint_dir: Path | None = None,
    resume: bool = True,
    progress: bool = False,
    progress_every: int = 1,
) -> tuple[pd.DataFrame, pd.DataFrame, MOD05.LopoResult, tuple[str, ...]]:
    """Run the all-32 M0 RI-versus-RI+RS LOPO comparison."""
    participant_ids = validate_all_participants(modeling_data)

    lopo = MOD05.run_lopo_prediction(
        modeling_data,
        model_rhs_registry={"M0": "{D}"},
        use_proposed_registry=False,
        targets=COMPARE_TARGETS,
        conditions=tuple(MOD02.CONDITIONS),
        random_structures=RANDOM_STRUCTURES,
        jobs=jobs,
        checkpoint_dir=checkpoint_dir,
        resume=resume,
        progress=progress,
        progress_every=progress_every,
    )

    full_fit, full_fit_audit = fit_full_data_m0(modeling_data)
    comparison = combine_full_fit_and_lopo(
        full_fit,
        lopo.model_summary,
    )
    audit = build_audit(
        full_fit_audit,
        lopo.checks,
        comparison,
    )
    return comparison, audit, lopo, participant_ids


def build_manifest(
    *,
    modeling_data_path: Path,
    participant_ids: Sequence[str],
    jobs: int,
    checkpoint_dir: Path | None,
) -> dict[str, Any]:
    """Return the MOD-07 method/audit manifest."""
    return {
        "script_version": SCRIPT_VERSION,
        "method_id": METHOD_ID,
        "analysis_role": "random_effects_structure_evidence",
        "participant_count": len(participant_ids),
        "participant_ids": list(participant_ids),
        "uses_all_32_participants": len(participant_ids) == EXPECTED_PARTICIPANTS,
        "cross_validation": "leave_one_participant_out",
        "lopo_fold_count": EXPECTED_PARTICIPANTS,
        "participants_fit_per_fold": EXPECTED_PARTICIPANTS - 1,
        "heldout_participants_per_fold": 1,
        "heldout_prediction_scope": "fixed_effect_population_only",
        "fixed_effects_model": "M0_difficulty_only",
        "random_structures": list(RANDOM_STRUCTURES),
        "gaussian_targets": list(COMPARE_TARGETS),
        "condition_target_pair_count": len(comparison_pairs()),
        "full_data_information_criteria": ["AIC", "BIC"],
        "predictive_metrics": [
            "participant_balanced_mae",
            "pooled_heldout_r2",
        ],
        "lopo_r2_definition": "1-SSE/SST across pooled held-out row predictions",
        "delta_conventions": {
            "MAE": "RI+RS minus RI; negative favours RI+RS",
            "R2": "RI+RS minus RI; positive favours RI+RS",
            "AIC": "RI+RS minus RI; negative favours RI+RS",
            "BIC": "RI+RS minus RI; negative favours RI+RS",
        },
        "automatic_random_structure_selection": False,
        "bootstrap_used": False,
        "train_test_split_used": False,
        "modeling_data_path": str(modeling_data_path),
        "jobs": int(jobs),
        "checkpoint_dir": (
            str(checkpoint_dir) if checkpoint_dir is not None else None
        ),
    }


def _refuse_existing_outputs(output_dir: Path) -> None:
    filenames = (
        COMPARISON_FILENAME,
        AUDIT_FILENAME,
        PREDICTIONS_FILENAME,
        PARTICIPANT_ERRORS_FILENAME,
        LOPO_CHECKS_FILENAME,
        MANIFEST_FILENAME,
    )
    existing = [
        output_dir / filename
        for filename in filenames
        if (output_dir / filename).exists()
    ]
    if existing:
        names = ", ".join(path.name for path in existing)
        raise FileExistsError(
            f"Refusing to overwrite existing MOD-07 output(s): {names}"
        )


def write_outputs(
    *,
    comparison: pd.DataFrame,
    audit: pd.DataFrame,
    lopo: MOD05.LopoResult,
    manifest: Mapping[str, Any],
    output_dir: Path,
) -> dict[str, Path]:
    """Write final MOD-07 outputs without overwriting existing results."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    _refuse_existing_outputs(output_dir)

    paths = {
        "comparison": output_dir / COMPARISON_FILENAME,
        "audit": output_dir / AUDIT_FILENAME,
        "predictions": output_dir / PREDICTIONS_FILENAME,
        "participant_errors": output_dir / PARTICIPANT_ERRORS_FILENAME,
        "lopo_checks": output_dir / LOPO_CHECKS_FILENAME,
        "manifest": output_dir / MANIFEST_FILENAME,
    }

    comparison.to_csv(paths["comparison"], index=False, encoding="utf-8")
    audit.to_csv(paths["audit"], index=False, encoding="utf-8")
    lopo.predictions.to_csv(paths["predictions"], index=False, encoding="utf-8")
    lopo.participant_errors.to_csv(
        paths["participant_errors"], index=False, encoding="utf-8"
    )
    lopo.checks.to_csv(paths["lopo_checks"], index=False, encoding="utf-8")
    paths["manifest"].write_text(
        json.dumps(dict(manifest), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return paths


def build_cli_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--modeling-data", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--jobs",
        type=int,
        default=0,
        help=(
            "Parallel condition-target-structure families. "
            "0=automatic using MOD-05's conservative cap; 1=serial."
        ),
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        help=(
            "LOPO checkpoint directory. Defaults to "
            "OUTPUT_DIR/_lopo_checkpoints."
        ),
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Ignore matching LOPO checkpoints and refit all folds.",
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Suppress LOPO progress messages.",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=1,
        help="Print progress after this many completed LOPO families.",
    )
    return parser


def main() -> int:
    args = build_cli_parser().parse_args()
    if args.progress_every < 1:
        raise Mod07Error("--progress-every must be >= 1")

    modeling_data = pd.read_csv(args.modeling_data)
    checkpoint_dir = (
        args.checkpoint_dir
        if args.checkpoint_dir is not None
        else args.output_dir / "_lopo_checkpoints"
    )

    comparison, audit, lopo, participant_ids = run_mod07(
        modeling_data,
        jobs=args.jobs,
        checkpoint_dir=checkpoint_dir,
        resume=not args.no_resume,
        progress=not args.no_progress,
        progress_every=args.progress_every,
    )

    manifest = build_manifest(
        modeling_data_path=args.modeling_data,
        participant_ids=participant_ids,
        jobs=MOD05.resolve_job_count(args.jobs),
        checkpoint_dir=checkpoint_dir,
    )
    paths = write_outputs(
        comparison=comparison,
        audit=audit,
        lopo=lopo,
        manifest=manifest,
        output_dir=args.output_dir,
    )

    print(
        f"[MOD-07] completed {len(comparison)} condition-target comparisons "
        f"using {len(participant_ids)} participants."
    )
    print(
        f"[MOD-07] LOPO errors: {len(lopo.errors)}; "
        f"warnings: {len(lopo.warnings)}."
    )
    for name, path in paths.items():
        print(f"[MOD-07] {name}: {path}")
    return 0 if audit["status"].eq("pass").all() else 1


if __name__ == "__main__":
    raise SystemExit(main())
