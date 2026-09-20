#!/usr/bin/env python3
"""MOD-08 — Condition-specific Age/TMT-B fixed-effects model comparison.

This module compares the seven prespecified fixed-effect candidates after the
random-effects structure has been frozen separately for each condition:

    Visual    -> RI
    Auditory  -> RI
    Cognitive -> RI_RS

For each of the 10 accepted Gaussian targets, the seven candidate models are
compared using:
- 32-fold leave-one-participant-out (LOPO) prediction;
- participant-balanced LOPO MAE;
- pooled held-out LOPO R²;
- full-data ML AIC and BIC;
- convergence/optimizer/warning diagnostics;
- parsimony-supporting nested model contrasts.

The script generates evidence only. It does not automatically select a winning
fixed-effects model and it does not use p-values as model-selection gates.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

import MOD_02_random_effects_structure as MOD02
import MOD_04_age_tmt_associations as MOD04
import MOD_05_lopo_predictive_comparison as MOD05


SCRIPT_VERSION = "1.0.0"
METHOD_ID = "condition_specific_random_effects_all32_fixed_effects_lopo_v1"
EXPECTED_PARTICIPANTS = 32
PARTICIPANT_COLUMN = "participant_id"

CONDITION_RANDOM_STRUCTURES = {
    "Visual": "RI",
    "Auditory": "RI",
    "Cognitive": "RI_RS",
}

MODEL_RHS = dict(MOD04.PROPOSED_MODEL_RHS)
MODEL_ORDER = ("M0", "MA", "MT", "MAT", "MDA", "MDT", "MDAT")
TARGETS = tuple(MOD02.GAUSSIAN_TARGETS)

NESTED_COMPARISONS = (
    ("MA", "M0", "add_age_main_effect"),
    ("MT", "M0", "add_tmt_main_effect"),
    ("MAT", "MA", "add_tmt_to_age_model"),
    ("MAT", "MT", "add_age_to_tmt_model"),
    ("MDA", "MA", "add_difficulty_by_age_interaction"),
    ("MDT", "MT", "add_difficulty_by_tmt_interaction"),
    ("MDAT", "MAT", "add_both_difficulty_interactions"),
    ("MDAT", "MDA", "add_tmt_main_and_difficulty_by_tmt"),
    ("MDAT", "MDT", "add_age_main_and_difficulty_by_age"),
)

COMPARISON_FILENAME = "fixed_effect_model_comparison.csv"
NESTED_FILENAME = "fixed_effect_nested_comparisons.csv"
AUDIT_FILENAME = "fixed_effect_model_audit.csv"
PREDICTIONS_FILENAME = "lopo_predictions.csv"
PARTICIPANT_ERRORS_FILENAME = "lopo_participant_errors.csv"
LOPO_CHECKS_FILENAME = "lopo_checks.csv"
MANIFEST_FILENAME = "mod08_manifest.json"


class Mod08Error(ValueError):
    """Raised when the frozen MOD-08 method contract is violated."""


@dataclass(frozen=True)
class Mod08Result:
    comparison: pd.DataFrame
    nested_comparisons: pd.DataFrame
    audit: pd.DataFrame
    predictions: pd.DataFrame
    participant_errors: pd.DataFrame
    lopo_checks: pd.DataFrame
    errors: tuple[str, ...]
    warnings: tuple[str, ...]
    participant_ids: tuple[str, ...]


def validate_method_contract() -> None:
    """Validate the frozen condition structures and seven-model registry."""
    if tuple(MODEL_RHS) != MODEL_ORDER:
        raise Mod08Error(
            "Candidate model registry order/content differs from the frozen "
            "M0, MA, MT, MAT, MDA, MDT, MDAT registry"
        )
    if set(CONDITION_RANDOM_STRUCTURES) != set(MOD02.CONDITIONS):
        raise Mod08Error(
            "Condition-specific random-effects map must cover exactly "
            f"{tuple(MOD02.CONDITIONS)}"
        )
    invalid = sorted(
        set(CONDITION_RANDOM_STRUCTURES.values()) - {"RI", "RI_RS"}
    )
    if invalid:
        raise Mod08Error(f"Unsupported frozen random structure(s): {invalid}")


def validate_all_participants(modeling_data: pd.DataFrame) -> tuple[str, ...]:
    """Require the complete 32-participant modelling dataset."""
    if PARTICIPANT_COLUMN not in modeling_data.columns:
        raise Mod08Error(f"Missing required column: {PARTICIPANT_COLUMN}")
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
        raise Mod08Error(
            "MOD-08 requires all 32 participants; "
            f"found {len(ids)} unique participants"
        )
    if any(not value for value in ids):
        raise Mod08Error("participant_id contains blank values")
    return ids


def difficulty_columns() -> dict[str, str]:
    return {
        target: spec.difficulty_column
        for target, spec in MOD02.build_internal_registry().items()
    }


def _prepare_target_frame(
    prepared_data: pd.DataFrame,
    *,
    condition: str,
    target: str,
    difficulty_column: str,
) -> tuple[pd.DataFrame, int]:
    """Return one common eligible frame used by all seven candidates."""
    frame = prepared_data.loc[
        prepared_data["condition_name"].eq(condition)
    ].copy()
    structural_excluded = 0

    if target == "performance_change_from_d0_percentage_points":
        if difficulty_column == "difficulty_stage":
            raise Mod08Error("Performance cannot silently use difficulty_stage")
        d0_mask = pd.to_numeric(
            frame["difficulty_level"], errors="coerce"
        ).eq(0)
        structural_excluded = int(d0_mask.sum())
        frame = frame.loc[~d0_mask].copy()

    required = [
        PARTICIPANT_COLUMN,
        "participant_group",
        "tmt_b_seconds",
        target,
        difficulty_column,
    ]
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise Mod08Error(
            f"{condition}/{target}: missing required columns {missing}"
        )

    frame["tmt_b_seconds"] = pd.to_numeric(
        frame["tmt_b_seconds"], errors="coerce"
    )
    frame = frame.loc[frame[required].notna().all(axis=1)].copy()
    if frame.empty:
        raise Mod08Error(f"{condition}/{target}: no eligible rows")
    if not np.isfinite(frame["tmt_b_seconds"].to_numpy(float)).all():
        raise Mod08Error(f"{condition}/{target}: TMT-B contains non-finite values")

    groups = set(frame["participant_group"].astype(str))
    if not {"Young", "Old"}.issubset(groups):
        raise Mod08Error(
            f"{condition}/{target}: age groups must include Young and Old"
        )

    participant_count = int(frame[PARTICIPANT_COLUMN].nunique())
    if participant_count != EXPECTED_PARTICIPANTS:
        raise Mod08Error(
            f"{condition}/{target}: eligible frame contains "
            f"{participant_count} participants rather than 32"
        )

    return frame, structural_excluded


def _row_signature(
    frame: pd.DataFrame,
    *,
    target: str,
    difficulty_column: str,
) -> str:
    """Hash the exact analysis rows/covariates used by every candidate."""
    columns = list(
        dict.fromkeys(
            [
                PARTICIPANT_COLUMN,
                "participant_group",
                "tmt_b_seconds",
                "condition_name",
                "difficulty_level",
                difficulty_column,
                target,
            ]
        )
    )
    stable = frame[columns].copy()
    stable = stable.sort_values(
        [PARTICIPANT_COLUMN, "difficulty_level"],
        kind="mergesort",
    )
    payload = stable.to_csv(index=False, float_format="%.17g")
    return sha256(payload.encode("utf-8")).hexdigest()


def _safe_float(value: Any) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return np.nan
    return numeric if np.isfinite(numeric) else np.nan


def fit_full_data_candidates(
    modeling_data: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fit all seven candidates once on all participants using frozen structures."""
    validate_method_contract()
    validate_all_participants(modeling_data)

    prepared = MOD02.prepare_internal_modeling_data(modeling_data)
    dcols = difficulty_columns()
    fit_rows: list[dict[str, Any]] = []
    audit_rows: list[dict[str, Any]] = []

    for condition in MOD02.CONDITIONS:
        structure = CONDITION_RANDOM_STRUCTURES[condition]
        for target in TARGETS:
            dcol = dcols.get(target)
            if not dcol:
                audit_rows.append({
                    "condition_name": condition,
                    "target_name": target,
                    "random_structure": structure,
                    "status": "failed",
                    "message": "Missing difficulty column mapping.",
                })
                continue

            try:
                frame, structural_excluded = _prepare_target_frame(
                    prepared,
                    condition=condition,
                    target=target,
                    difficulty_column=dcol,
                )
                signature = _row_signature(
                    frame,
                    target=target,
                    difficulty_column=dcol,
                )
                family_rows: list[dict[str, Any]] = []

                for model_id in MODEL_ORDER:
                    rhs = MODEL_RHS[model_id].format(D=dcol)
                    formula = f"{target} ~ {rhs}"
                    result, optimizer, fit_warnings, fit_errors = (
                        MOD02.fit_mixedlm_with_fallback(
                            formula,
                            frame,
                            structure,
                            dcol,
                        )
                    )
                    converged = bool(
                        result is not None
                        and getattr(result, "converged", False)
                    )
                    if converged:
                        fixed_count = int(len(result.fe_params))
                        total_count = int(
                            getattr(result, "df_modelwc", len(result.params))
                        ) + 1
                        llf = _safe_float(result.llf)
                        aic = _safe_float(result.aic)
                        bic = _safe_float(result.bic)
                    else:
                        fixed_count = np.nan
                        total_count = np.nan
                        llf = np.nan
                        aic = np.nan
                        bic = np.nan

                    family_rows.append({
                        "condition_name": condition,
                        "target_name": target,
                        "random_structure": structure,
                        "model_id": model_id,
                        "formula": formula,
                        "difficulty_source_column": dcol,
                        "participant_count": int(
                            frame[PARTICIPANT_COLUMN].nunique()
                        ),
                        "observation_count": int(len(frame)),
                        "excluded_structural_d0_count": structural_excluded,
                        "included_row_signature": signature,
                        "fixed_effect_count": fixed_count,
                        "total_parameter_count": total_count,
                        "log_likelihood": llf,
                        "aic": aic,
                        "bic": bic,
                        "convergence_status": (
                            "converged"
                            if converged
                            else (
                                "failed"
                                if result is None
                                else "non_converged"
                            )
                        ),
                        "optimizer": optimizer,
                        "warnings": " | ".join(fit_warnings),
                        "fit_errors": " | ".join(fit_errors),
                    })

                family = pd.DataFrame(family_rows)
                m0_rows = family.loc[family["model_id"].eq("M0")]
                m0_aic = (
                    _safe_float(m0_rows.iloc[0]["aic"])
                    if len(m0_rows) == 1
                    else np.nan
                )
                m0_bic = (
                    _safe_float(m0_rows.iloc[0]["bic"])
                    if len(m0_rows) == 1
                    else np.nan
                )
                family["delta_aic_vs_m0"] = family["aic"] - m0_aic
                family["delta_bic_vs_m0"] = family["bic"] - m0_bic
                family["delta_ic_convention"] = (
                    "candidate_minus_M0; negative means lower information criterion"
                )
                fit_rows.extend(family.to_dict("records"))

                signatures = set(family["included_row_signature"])
                complete = (
                    len(family) == len(MODEL_ORDER)
                    and set(family["model_id"]) == set(MODEL_ORDER)
                )
                all_converged = complete and family[
                    "convergence_status"
                ].eq("converged").all()
                audit_rows.append({
                    "condition_name": condition,
                    "target_name": target,
                    "random_structure": structure,
                    "status": (
                        "pass"
                        if all_converged and len(signatures) == 1
                        else "problem"
                    ),
                    "participant_count": int(
                        frame[PARTICIPANT_COLUMN].nunique()
                    ),
                    "observation_count": int(len(frame)),
                    "included_row_signature": signature,
                    "candidate_model_count": int(len(family)),
                    "all_candidates_use_identical_rows": len(signatures) == 1,
                    "all_candidates_converged": bool(all_converged),
                    "message": (
                        "All seven candidates fitted on identical eligible rows "
                        "under the frozen condition-specific random structure."
                        if all_converged and len(signatures) == 1
                        else "Review candidate convergence or row comparability."
                    ),
                })
            except Exception as exc:
                audit_rows.append({
                    "condition_name": condition,
                    "target_name": target,
                    "random_structure": structure,
                    "status": "failed",
                    "participant_count": np.nan,
                    "observation_count": np.nan,
                    "included_row_signature": "",
                    "candidate_model_count": 0,
                    "all_candidates_use_identical_rows": False,
                    "all_candidates_converged": False,
                    "message": f"{type(exc).__name__}: {exc}",
                })

    return pd.DataFrame(fit_rows), pd.DataFrame(audit_rows)


def run_condition_specific_lopo(
    modeling_data: pd.DataFrame,
    *,
    jobs: int = 1,
    checkpoint_dir: Path | None = None,
    resume: bool = True,
    progress: bool = False,
    progress_every: int = 1,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    tuple[str, ...],
    tuple[str, ...],
]:
    """Run seven-model LOPO under one frozen random structure per condition."""
    summaries: list[pd.DataFrame] = []
    predictions: list[pd.DataFrame] = []
    participant_errors: list[pd.DataFrame] = []
    checks: list[pd.DataFrame] = []
    errors: list[str] = []
    warnings_out: list[str] = []

    for condition in MOD02.CONDITIONS:
        structure = CONDITION_RANDOM_STRUCTURES[condition]
        condition_checkpoint = (
            Path(checkpoint_dir) / condition.lower()
            if checkpoint_dir is not None
            else None
        )
        result = MOD05.run_lopo_prediction(
            modeling_data,
            model_rhs_registry=MODEL_RHS,
            use_proposed_registry=False,
            targets=TARGETS,
            conditions=(condition,),
            random_structures=(structure,),
            jobs=jobs,
            checkpoint_dir=condition_checkpoint,
            resume=resume,
            progress=progress,
            progress_every=progress_every,
        )
        summaries.append(result.model_summary)
        predictions.append(result.predictions)
        participant_errors.append(result.participant_errors)
        checks.append(result.checks)
        errors.extend(result.errors)
        warnings_out.extend(result.warnings)

    def concat(frames: list[pd.DataFrame]) -> pd.DataFrame:
        nonempty = [frame for frame in frames if len(frame)]
        return (
            pd.concat(nonempty, ignore_index=True)
            if nonempty
            else pd.DataFrame()
        )

    return (
        concat(summaries),
        concat(predictions),
        concat(participant_errors),
        concat(checks),
        tuple(errors),
        tuple(warnings_out),
    )


def combine_full_fit_and_lopo(
    full_fit: pd.DataFrame,
    lopo_summary: pd.DataFrame,
) -> pd.DataFrame:
    """Create the primary 210-row model-comparison evidence table."""
    required = {
        "condition_name",
        "target_name",
        "random_structure",
        "model_id",
        "participant_count",
        "prediction_count",
        "participant_balanced_mae",
        "participant_balanced_rmse",
        "lopo_r2",
        "delta_mae_vs_m0",
    }
    missing = sorted(required - set(lopo_summary.columns))
    if missing:
        raise Mod08Error(f"LOPO summary missing required columns: {missing}")

    lopo = lopo_summary.rename(columns={
        "participant_count": "lopo_participant_count",
        "prediction_count": "lopo_prediction_count",
        "participant_balanced_mae": "lopo_mae",
        "participant_balanced_rmse": "lopo_rmse",
    }).copy()

    keep = [
        "condition_name",
        "target_name",
        "random_structure",
        "model_id",
        "lopo_participant_count",
        "lopo_prediction_count",
        "lopo_mae",
        "lopo_rmse",
        "lopo_r2",
        "delta_mae_vs_m0",
    ]
    combined = full_fit.merge(
        lopo[keep],
        on=[
            "condition_name",
            "target_name",
            "random_structure",
            "model_id",
        ],
        how="left",
        validate="one_to_one",
    )

    combined["mae_improvement_percent_vs_m0"] = np.nan
    combined["delta_lopo_r2_vs_m0"] = np.nan

    group_keys = ["condition_name", "target_name", "random_structure"]
    for _, index in combined.groupby(group_keys, sort=False).groups.items():
        idx = list(index)
        family = combined.loc[idx]
        m0 = family.loc[family["model_id"].eq("M0")]
        if len(m0) != 1:
            continue
        m0_mae = _safe_float(m0.iloc[0]["lopo_mae"])
        m0_r2 = _safe_float(m0.iloc[0]["lopo_r2"])
        if np.isfinite(m0_mae) and m0_mae != 0:
            combined.loc[idx, "mae_improvement_percent_vs_m0"] = (
                combined.loc[idx, "delta_mae_vs_m0"] / m0_mae * 100.0
            )
        if np.isfinite(m0_r2):
            combined.loc[idx, "delta_lopo_r2_vs_m0"] = (
                combined.loc[idx, "lopo_r2"] - m0_r2
            )

    combined["delta_mae_convention"] = (
        "MAE_M0_minus_MAE_candidate; positive means candidate predicts better"
    )
    combined["delta_r2_convention"] = (
        "R2_candidate_minus_R2_M0; positive means candidate predicts better"
    )
    combined["selection_status"] = "evidence_only_no_automatic_selection"

    condition_order = {name: i for i, name in enumerate(MOD02.CONDITIONS)}
    target_order = {name: i for i, name in enumerate(TARGETS)}
    model_order = {name: i for i, name in enumerate(MODEL_ORDER)}
    combined["_condition_order"] = combined["condition_name"].map(
        condition_order
    )
    combined["_target_order"] = combined["target_name"].map(target_order)
    combined["_model_order"] = combined["model_id"].map(model_order)
    combined = combined.sort_values(
        ["_condition_order", "_target_order", "_model_order"],
        kind="mergesort",
    ).drop(
        columns=["_condition_order", "_target_order", "_model_order"]
    ).reset_index(drop=True)

    return combined


def build_nested_comparisons(
    comparison: pd.DataFrame,
) -> pd.DataFrame:
    """Create prespecified incremental comparisons among nested candidates."""
    rows: list[dict[str, Any]] = []
    keys = ["condition_name", "target_name", "random_structure"]

    for family_key, family in comparison.groupby(keys, sort=False):
        condition, target, structure = family_key
        by_model = {
            str(row["model_id"]): row
            for _, row in family.iterrows()
        }
        for candidate, reference, purpose in NESTED_COMPARISONS:
            if candidate not in by_model or reference not in by_model:
                continue
            cand = by_model[candidate]
            ref = by_model[reference]
            cand_mae = _safe_float(cand.get("lopo_mae"))
            ref_mae = _safe_float(ref.get("lopo_mae"))
            cand_r2 = _safe_float(cand.get("lopo_r2"))
            ref_r2 = _safe_float(ref.get("lopo_r2"))
            cand_aic = _safe_float(cand.get("aic"))
            ref_aic = _safe_float(ref.get("aic"))
            cand_bic = _safe_float(cand.get("bic"))
            ref_bic = _safe_float(ref.get("bic"))

            rows.append({
                "condition_name": condition,
                "target_name": target,
                "random_structure": structure,
                "candidate_model": candidate,
                "reference_model": reference,
                "comparison_purpose": purpose,
                "delta_mae_candidate_minus_reference": (
                    cand_mae - ref_mae
                    if np.isfinite(cand_mae) and np.isfinite(ref_mae)
                    else np.nan
                ),
                "delta_r2_candidate_minus_reference": (
                    cand_r2 - ref_r2
                    if np.isfinite(cand_r2) and np.isfinite(ref_r2)
                    else np.nan
                ),
                "delta_aic_candidate_minus_reference": (
                    cand_aic - ref_aic
                    if np.isfinite(cand_aic) and np.isfinite(ref_aic)
                    else np.nan
                ),
                "delta_bic_candidate_minus_reference": (
                    cand_bic - ref_bic
                    if np.isfinite(cand_bic) and np.isfinite(ref_bic)
                    else np.nan
                ),
                "interpretation_convention": (
                    "MAE/AIC/BIC negative favours candidate; "
                    "R2 positive favours candidate"
                ),
            })

    return pd.DataFrame(rows)


def build_audit(
    full_fit_audit: pd.DataFrame,
    lopo_checks: pd.DataFrame,
    comparison: pd.DataFrame,
) -> pd.DataFrame:
    """Combine full-data and LOPO completeness checks by condition × target."""
    rows: list[dict[str, Any]] = []
    for condition in MOD02.CONDITIONS:
        structure = CONDITION_RANDOM_STRUCTURES[condition]
        for target in TARGETS:
            full = full_fit_audit.loc[
                full_fit_audit["condition_name"].eq(condition)
                & full_fit_audit["target_name"].eq(target)
            ]
            checks = lopo_checks.loc[
                lopo_checks["condition_name"].eq(condition)
                & lopo_checks["target_name"].eq(target)
                & lopo_checks["random_structure"].eq(structure)
            ]
            family = comparison.loc[
                comparison["condition_name"].eq(condition)
                & comparison["target_name"].eq(target)
                & comparison["random_structure"].eq(structure)
            ]

            full_ok = (
                len(full) == 1
                and str(full.iloc[0]["status"]) == "pass"
            )
            lopo_ok = (
                len(checks) == 1
                and str(checks.iloc[0]["status"]) == "pass"
            )
            model_set_ok = (
                len(family) == len(MODEL_ORDER)
                and set(family["model_id"]) == set(MODEL_ORDER)
            )
            lopo_participant_counts = pd.to_numeric(
                family.get("lopo_participant_count", pd.Series(dtype=float)),
                errors="coerce",
            )
            complete_lopo = (
                model_set_ok
                and len(lopo_participant_counts) == len(MODEL_ORDER)
                and lopo_participant_counts.eq(EXPECTED_PARTICIPANTS).all()
            )
            status = (
                "pass"
                if full_ok and lopo_ok and model_set_ok and complete_lopo
                else "problem"
            )

            rows.append({
                "condition_name": condition,
                "target_name": target,
                "random_structure": structure,
                "status": status,
                "full_data_family_ok": full_ok,
                "lopo_family_ok": lopo_ok,
                "seven_candidate_models_present": model_set_ok,
                "all_models_have_32_lopo_participants": complete_lopo,
                "message": (
                    "Full-data and 32-participant LOPO evidence complete for "
                    "all seven candidates."
                    if status == "pass"
                    else "Review full-data or LOPO completeness before selection."
                ),
            })
    return pd.DataFrame(rows)


def run_mod08(
    modeling_data: pd.DataFrame,
    *,
    jobs: int = 1,
    checkpoint_dir: Path | None = None,
    resume: bool = True,
    progress: bool = False,
    progress_every: int = 1,
) -> Mod08Result:
    """Run the complete condition-specific seven-model comparison."""
    validate_method_contract()
    participant_ids = validate_all_participants(modeling_data)

    (
        lopo_summary,
        predictions,
        participant_errors,
        lopo_checks,
        lopo_errors,
        lopo_warnings,
    ) = run_condition_specific_lopo(
        modeling_data,
        jobs=jobs,
        checkpoint_dir=checkpoint_dir,
        resume=resume,
        progress=progress,
        progress_every=progress_every,
    )

    full_fit, full_fit_audit = fit_full_data_candidates(modeling_data)
    comparison = combine_full_fit_and_lopo(full_fit, lopo_summary)
    nested = build_nested_comparisons(comparison)
    audit = build_audit(full_fit_audit, lopo_checks, comparison)

    return Mod08Result(
        comparison=comparison,
        nested_comparisons=nested,
        audit=audit,
        predictions=predictions,
        participant_errors=participant_errors,
        lopo_checks=lopo_checks,
        errors=lopo_errors,
        warnings=lopo_warnings,
        participant_ids=participant_ids,
    )


def build_manifest(
    *,
    modeling_data_path: Path,
    participant_ids: Sequence[str],
    jobs: int,
    checkpoint_dir: Path | None,
) -> dict[str, Any]:
    """Return the auditable MOD-08 method manifest."""
    return {
        "script_version": SCRIPT_VERSION,
        "method_id": METHOD_ID,
        "analysis_role": "fixed_effects_candidate_model_comparison",
        "participant_count": len(participant_ids),
        "participant_ids": list(participant_ids),
        "uses_all_32_participants": len(participant_ids) == EXPECTED_PARTICIPANTS,
        "cross_validation": "leave_one_participant_out",
        "lopo_fold_count": EXPECTED_PARTICIPANTS,
        "participants_fit_per_fold": EXPECTED_PARTICIPANTS - 1,
        "heldout_participants_per_fold": 1,
        "heldout_prediction_scope": "fixed_effect_population_only",
        "condition_random_structures": dict(CONDITION_RANDOM_STRUCTURES),
        "candidate_models": {
            model_id: MODEL_RHS[model_id]
            for model_id in MODEL_ORDER
        },
        "candidate_model_count": len(MODEL_ORDER),
        "gaussian_targets": list(TARGETS),
        "condition_target_pair_count": len(MOD02.CONDITIONS) * len(TARGETS),
        "total_candidate_family_rows": (
            len(MOD02.CONDITIONS) * len(TARGETS) * len(MODEL_ORDER)
        ),
        "predictive_metrics": [
            "participant_balanced_mae",
            "pooled_heldout_r2",
        ],
        "information_criteria": ["AIC", "BIC"],
        "fit_estimation": "ML_reml_false_via_MOD02",
        "tmt_b_representation": "raw_seconds",
        "p_values_used_for_model_selection": False,
        "automatic_fixed_effects_selection": False,
        "nested_comparisons": [
            {
                "candidate": candidate,
                "reference": reference,
                "purpose": purpose,
            }
            for candidate, reference, purpose in NESTED_COMPARISONS
        ],
        "modeling_data_path": str(modeling_data_path),
        "jobs": int(jobs),
        "checkpoint_dir": (
            str(checkpoint_dir) if checkpoint_dir is not None else None
        ),
    }


def _refuse_existing_outputs(output_dir: Path) -> None:
    filenames = (
        COMPARISON_FILENAME,
        NESTED_FILENAME,
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
        raise FileExistsError(
            "Refusing to overwrite existing MOD-08 output(s): "
            + ", ".join(path.name for path in existing)
        )


def write_outputs(
    *,
    result: Mod08Result,
    manifest: Mapping[str, Any],
    output_dir: Path,
) -> dict[str, Path]:
    """Write MOD-08 evidence files without overwriting prior results."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    _refuse_existing_outputs(output_dir)

    paths = {
        "comparison": output_dir / COMPARISON_FILENAME,
        "nested": output_dir / NESTED_FILENAME,
        "audit": output_dir / AUDIT_FILENAME,
        "predictions": output_dir / PREDICTIONS_FILENAME,
        "participant_errors": output_dir / PARTICIPANT_ERRORS_FILENAME,
        "lopo_checks": output_dir / LOPO_CHECKS_FILENAME,
        "manifest": output_dir / MANIFEST_FILENAME,
    }
    result.comparison.to_csv(paths["comparison"], index=False, encoding="utf-8")
    result.nested_comparisons.to_csv(
        paths["nested"], index=False, encoding="utf-8"
    )
    result.audit.to_csv(paths["audit"], index=False, encoding="utf-8")
    result.predictions.to_csv(
        paths["predictions"], index=False, encoding="utf-8"
    )
    result.participant_errors.to_csv(
        paths["participant_errors"], index=False, encoding="utf-8"
    )
    result.lopo_checks.to_csv(
        paths["lopo_checks"], index=False, encoding="utf-8"
    )
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
            "Parallel condition-target families inside each condition. "
            "0=automatic (MOD-05 cap); 1=serial."
        ),
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        help=(
            "LOPO checkpoint root. Defaults to "
            "OUTPUT_DIR/_lopo_checkpoints."
        ),
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Ignore matching checkpoints and refit all LOPO folds.",
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
        raise Mod08Error("--progress-every must be >= 1")

    modeling_data = pd.read_csv(args.modeling_data)
    checkpoint_dir = (
        args.checkpoint_dir
        if args.checkpoint_dir is not None
        else args.output_dir / "_lopo_checkpoints"
    )
    result = run_mod08(
        modeling_data,
        jobs=args.jobs,
        checkpoint_dir=checkpoint_dir,
        resume=not args.no_resume,
        progress=not args.no_progress,
        progress_every=args.progress_every,
    )
    manifest = build_manifest(
        modeling_data_path=args.modeling_data,
        participant_ids=result.participant_ids,
        jobs=MOD05.resolve_job_count(args.jobs),
        checkpoint_dir=checkpoint_dir,
    )
    paths = write_outputs(
        result=result,
        manifest=manifest,
        output_dir=args.output_dir,
    )

    print(
        f"[MOD-08] comparison rows={len(result.comparison)}; "
        f"nested rows={len(result.nested_comparisons)}; "
        f"audit pass={int(result.audit['status'].eq('pass').sum())}/"
        f"{len(result.audit)}"
    )
    print(
        f"[MOD-08] LOPO errors={len(result.errors)}; "
        f"warnings={len(result.warnings)}"
    )
    for name, path in paths.items():
        print(f"[MOD-08] {name}: {path}")

    return 0 if result.audit["status"].eq("pass").all() else 2


if __name__ == "__main__":
    raise SystemExit(main())
