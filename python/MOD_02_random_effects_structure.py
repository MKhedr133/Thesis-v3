#!/usr/bin/env python3
"""MOD-02 — Compare and persist random-effects structure evidence.

The same RI/RI+RS comparison is intended to underlie later difficulty association,
Age/TMT models, and LOPO prediction for a given condition × target.

This file covers the two primary outcomes plus nine task/behaviour features.
Eight feature targets use the Gaussian MixedLM RI/RI+RS pathway.
``total_error_count`` is the ninth feature and is registered in the same target
registry, but its RI/RI+RS fit is blocked until a count-response mixed-model
family is approved. No random structure is auto-selected.

The CLI is intentionally self-contained: target formulas and difficulty columns
are defined in this module rather than supplied through JSON files.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Mapping
import warnings as pywarnings

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

CONDITIONS = ("Visual", "Auditory", "Cognitive")
CONTINUOUS_FEATURE_TARGETS = (
    "median_time_between_qualifying_grabs_seconds",
    "list_recheck_count",
    "total_list_recheck_duration_seconds",
    "median_time_to_target_seconds",
    "median_irrelevant_focus_duration_seconds",
    "median_head_turning_degrees",
    "median_reach_duration_seconds",
    "median_reach_path_ratio",
)
PENDING_COUNT_TARGETS = ("total_error_count",)
FEATURE_TARGETS = (*CONTINUOUS_FEATURE_TARGETS, *PENDING_COUNT_TARGETS)
# Compatibility alias: behavioural/task features now explicitly total nine.
BEHAVIOURAL_TARGETS = FEATURE_TARGETS
GAUSSIAN_TARGETS = (
    "performance_change_from_d0_percentage_points",
    "mental_demand_score_0_to_10",
    *CONTINUOUS_FEATURE_TARGETS,
)
ALL_TARGETS = (*GAUSSIAN_TARGETS, *PENDING_COUNT_TARGETS)

PERFORMANCE_DIFFICULTY_COLUMN = "performance_difficulty_stage"
AGE_TERM = "C(participant_group, Treatment(reference='Young'))"
FIXED_EFFECTS_TEMPLATE = "{D} * " + AGE_TERM + " + {D} * tmt_z"


@dataclass(frozen=True)
class TargetSpec:
    target: str
    target_role: str
    model_family: str
    difficulty_column: str
    formula: str
    fit_status: str


def build_internal_registry() -> dict[str, TargetSpec]:
    """Return the frozen MOD-02 target registry used by the CLI.

    Performance uses only D2/D6/D10 and receives an explicit post-D0 ordered
    stage 0/1/2. Mental demand and the eight continuous feature targets use the
    approved four-stage ``difficulty_stage`` 0/1/2/3. ``total_error_count`` is
    retained as the ninth feature but blocked pending a count mixed-model family.
    """
    registry: dict[str, TargetSpec] = {}
    for target in GAUSSIAN_TARGETS:
        dcol = (
            PERFORMANCE_DIFFICULTY_COLUMN
            if target == "performance_change_from_d0_percentage_points"
            else "difficulty_stage"
        )
        formula = f"{target} ~ " + FIXED_EFFECTS_TEMPLATE.format(D=dcol)
        registry[target] = TargetSpec(
            target=target,
            target_role=_target_role(target),
            model_family="gaussian_mixedlm",
            difficulty_column=dcol,
            formula=formula,
            fit_status="fit_ri_and_ri_rs",
        )
    registry["total_error_count"] = TargetSpec(
        target="total_error_count",
        target_role="secondary_feature",
        model_family="count_mixed_model_pending",
        difficulty_column="difficulty_stage",
        formula="",
        fit_status="blocked_pending_count_model",
    )
    return registry


def prepare_internal_modeling_data(data: pd.DataFrame) -> pd.DataFrame:
    """Add deterministic internal predictors needed by the frozen registry."""
    out = data.copy(deep=True)
    if "difficulty_level" not in out.columns:
        raise ValueError("difficulty_level is required")
    mapping = {2: 0.0, 6: 1.0, 10: 2.0}
    numeric = pd.to_numeric(out["difficulty_level"], errors="coerce")
    out[PERFORMANCE_DIFFICULTY_COLUMN] = numeric.map(mapping)
    return out


@dataclass(frozen=True)
class RandomEffectsRunResult:
    valid: bool
    errors: tuple[str, ...]
    warnings: tuple[str, ...]
    comparison: pd.DataFrame
    pair_audit: pd.DataFrame


def _row_signature(frame: pd.DataFrame) -> str:
    cols = [c for c in (
        "participant_id", "session_id", "source_tracker_csv_filename",
        "condition_name", "difficulty_level", "trial_order",
    ) if c in frame.columns]
    if cols:
        rows = frame[cols].astype("string").fillna("<NA>").agg("|".join, axis=1).tolist()
    else:
        rows = [str(i) for i in frame.index]
    return sha256("\n".join(sorted(rows)).encode("utf-8")).hexdigest()


def add_full_data_tmt_z(data: pd.DataFrame) -> pd.DataFrame:
    """Add full-data TMT z-score from unique participants (association/RI stage only)."""
    out = data.copy(deep=True)
    if "tmt_b_seconds" not in out.columns:
        raise ValueError("tmt_b_seconds is required")
    participant = out[["participant_id", "tmt_b_seconds"]].drop_duplicates("participant_id")
    values = pd.to_numeric(participant["tmt_b_seconds"], errors="coerce")
    if values.isna().any() or len(values) < 2:
        raise ValueError("TMT-B must be finite for at least two unique participants")
    mean = float(values.mean())
    sd = float(values.std(ddof=1))
    if not np.isfinite(sd) or sd <= 0:
        raise ValueError("TMT-B standard deviation must be positive")
    tmt_map = dict(zip(participant["participant_id"].astype(str), ((values - mean) / sd).astype(float)))
    out["tmt_z"] = out["participant_id"].astype(str).map(tmt_map)
    return out


def fit_mixedlm_with_fallback(
    formula: str,
    data: pd.DataFrame,
    random_structure: str,
    difficulty_column: str,
    *,
    participant_column: str = "participant_id",
    maxiter: int = 1000,
    methods: tuple[str, ...] = ("lbfgs", "powell"),
):
    """Fit one statsmodels MixedLM using ML, preserving all optimizer evidence."""
    if random_structure not in {"RI", "RI_RS"}:
        raise ValueError(f"Unsupported random_structure={random_structure!r}")
    re_formula = "1" if random_structure == "RI" else f"1 + {difficulty_column}"
    model = smf.mixedlm(
        formula,
        data=data,
        groups=data[participant_column],
        re_formula=re_formula,
        missing="drop",
    )
    warnings_out: list[str] = []
    errors: list[str] = []
    last = None
    used = ""
    for method in methods:
        used = method
        with pywarnings.catch_warnings(record=True) as caught:
            pywarnings.simplefilter("always")
            try:
                result = model.fit(reml=False, method=method, maxiter=maxiter, disp=False)
                last = result
            except Exception as exc:  # statsmodels optimizer failures vary by backend
                errors.append(f"{method}: {type(exc).__name__}: {exc}")
                result = None
            warnings_out.extend(f"{method}: {w.category.__name__}: {w.message}" for w in caught)
        if result is not None and bool(getattr(result, "converged", False)):
            return result, used, tuple(warnings_out), tuple(errors)
    return last, used, tuple(warnings_out), tuple(errors)


def _target_role(target: str) -> str:
    return "primary_outcome" if target in {
        "performance_change_from_d0_percentage_points", "mental_demand_score_0_to_10"
    } else "secondary_feature"


def _prepare_target_frame(
    data: pd.DataFrame,
    condition: str,
    target: str,
    difficulty_column: str,
    formula: str,
) -> tuple[pd.DataFrame, int]:
    frame = data.loc[data["condition_name"].eq(condition)].copy()
    structural_excluded = 0
    if target == "performance_change_from_d0_percentage_points":
        if difficulty_column == "difficulty_stage":
            raise ValueError("Performance difficulty coding must be explicitly supplied; no difficulty_stage fallback is allowed.")
        d0 = frame["difficulty_level"].eq(0)
        structural_excluded = int(d0.sum())
        frame = frame.loc[~d0].copy()
    elif target == "mental_demand_score_0_to_10" and difficulty_column != "difficulty_stage":
        raise ValueError("Mental demand must use difficulty_stage")
    required = ["participant_id", target, difficulty_column]
    missing = [c for c in required if c not in frame.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    frame = frame.loc[frame[required].notna().all(axis=1)].copy()
    if frame.empty:
        raise ValueError("No eligible rows after target-specific exclusions")
    # Patsy may exclude formula-variable missingness; build model once to recover exact row labels.
    probe = smf.mixedlm(formula, data=frame, groups=frame["participant_id"], re_formula="1", missing="drop")
    labels = list(getattr(probe.data, "row_labels", frame.index))
    return frame.loc[labels].copy(), structural_excluded


def compare_random_effects(
    modeling_data: pd.DataFrame,
    executable_formulas: Mapping[str, str],
    difficulty_columns: Mapping[str, str],
    *,
    targets: tuple[str, ...] = GAUSSIAN_TARGETS,
    conditions: tuple[str, ...] = CONDITIONS,
) -> RandomEffectsRunResult:
    """Fit paired RI/RI+RS models and return evidence without selecting a winner."""
    data = add_full_data_tmt_z(modeling_data)
    rows: list[dict[str, object]] = []
    audits: list[dict[str, object]] = []
    errors: list[str] = []
    warnings: list[str] = []

    for condition in conditions:
        for target in targets:
            formula = executable_formulas.get(target, "").strip()
            dcol = difficulty_columns.get(target, "").strip()
            if not formula or not dcol:
                msg = f"{condition}/{target}: missing caller-supplied formula or difficulty column"
                errors.append(msg)
                audits.append({"condition_name": condition, "target_name": target, "status": "failed", "message": msg})
                continue
            try:
                fit_data, structural_excluded = _prepare_target_frame(data, condition, target, dcol, formula)
                signature = _row_signature(fit_data)
                pair_rows = []
                for structure in ("RI", "RI_RS"):
                    result, optimizer, fit_warnings, fit_errors = fit_mixedlm_with_fallback(
                        formula, fit_data, structure, dcol
                    )
                    warnings.extend(f"{condition}/{target}/{structure}: {w}" for w in fit_warnings)
                    converged = bool(result is not None and getattr(result, "converged", False))
                    if result is None or not converged:
                        errors.append(f"{condition}/{target}/{structure}: fit failed/non-converged")
                    if result is not None:
                        cov_re = np.asarray(result.cov_re)
                        if structure == "RI":
                            ri_var = float(cov_re[0, 0]) if cov_re.size else np.nan
                            rs_var = np.nan
                            covariance = np.nan
                        else:
                            ri_var = float(cov_re[0, 0]) if cov_re.size else np.nan
                            rs_var = float(cov_re[1, 1]) if cov_re.shape[0] >= 2 else np.nan
                            covariance = float(cov_re[0, 1]) if cov_re.shape[0] >= 2 else np.nan
                        ll = float(result.llf) if np.isfinite(result.llf) else np.nan
                        aic = float(result.aic) if np.isfinite(result.aic) else np.nan
                        bic = float(result.bic) if np.isfinite(result.bic) else np.nan
                        fixed_count = int(len(result.fe_params))
                        total_count = int(getattr(result, "df_modelwc", len(result.params))) + 1
                    else:
                        ri_var = rs_var = covariance = ll = aic = bic = np.nan
                        fixed_count = total_count = np.nan
                    pair_rows.append({
                        "condition_name": condition,
                        "target_name": target,
                        "target_role": _target_role(target),
                        "model_id": structure,
                        "random_structure": structure,
                        "fixed_effects_formula": formula,
                        "difficulty_source_column": dcol,
                        "included_row_signature": signature,
                        "participant_count": int(fit_data["participant_id"].nunique()),
                        "observation_count": int(len(fit_data)),
                        "excluded_structural_d0_count": structural_excluded,
                        "fixed_effect_count": fixed_count,
                        "total_parameter_count": total_count,
                        "log_likelihood": ll,
                        "aic": aic,
                        "bic": bic,
                        "delta_aic": np.nan,
                        "delta_bic": np.nan,
                        "likelihood_ratio_statistic": np.nan,
                        "likelihood_ratio_p_value": np.nan,
                        "likelihood_ratio_role": "supporting_only",
                        "estimation_method": "maximum_likelihood",
                        "reml": False,
                        "convergence_status": "converged" if converged else ("non_converged" if result is not None else "failed"),
                        "optimizer": optimizer,
                        "warnings": " | ".join(fit_warnings),
                        "fit_errors": " | ".join(fit_errors),
                        "random_intercept_variance": ri_var,
                        "random_slope_variance": rs_var,
                        "intercept_slope_covariance": covariance,
                        "boundary_flag": np.nan,
                        "singularity_flag": np.nan,
                        "selection_status": "pending_supervisor_rule",
                    })
                if len(pair_rows) == 2:
                    aics = [r["aic"] for r in pair_rows]
                    bics = [r["bic"] for r in pair_rows]
                    if all(np.isfinite(x) for x in aics):
                        amin = min(aics)
                        for r in pair_rows:
                            r["delta_aic"] = r["aic"] - amin
                    if all(np.isfinite(x) for x in bics):
                        bmin = min(bics)
                        for r in pair_rows:
                            r["delta_bic"] = r["bic"] - bmin
                    if all(np.isfinite(r["log_likelihood"]) for r in pair_rows):
                        lr = 2.0 * (pair_rows[1]["log_likelihood"] - pair_rows[0]["log_likelihood"])
                        for r in pair_rows:
                            r["likelihood_ratio_statistic"] = lr
                        if lr < 0:
                            warnings.append(f"{condition}/{target}: negative raw LR statistic preserved")
                rows.extend(pair_rows)
                comparable = all(r["convergence_status"] == "converged" for r in pair_rows)
                audits.append({
                    "condition_name": condition,
                    "target_name": target,
                    "status": "comparable" if comparable else "failed",
                    "included_row_signature": signature,
                    "observation_count": int(len(fit_data)),
                    "participant_count": int(fit_data["participant_id"].nunique()),
                    "message": "RI and RI+RS use identical rows/formula/difficulty coding." if comparable else "One or both paired fits failed/non-converged.",
                })
            except Exception as exc:
                msg = f"{condition}/{target}: {type(exc).__name__}: {exc}"
                errors.append(msg)
                audits.append({"condition_name": condition, "target_name": target, "status": "failed", "message": msg})

    for target in PENDING_COUNT_TARGETS:
        for condition in conditions:
            audits.append({
                "condition_name": condition,
                "target_name": target,
                "status": "blocked",
                "message": "Count-response model family is not approved; no Gaussian RI/RI+RS comparison was run.",
            })
    comparison = pd.DataFrame(rows)
    audit = pd.DataFrame(audits)
    valid = not errors and (audit.loc[audit["status"].ne("blocked"), "status"].eq("comparable").all() if len(audit) else False)
    return RandomEffectsRunResult(valid, tuple(errors), tuple(warnings), comparison, audit)


def write_random_effects_outputs(result: RandomEffectsRunResult, output_dir: Path) -> tuple[Path, Path]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    comparison_path = output_dir / "random_effects_comparison.csv"
    audit_path = output_dir / "random_effects_pair_audit.csv"
    result.comparison.to_csv(comparison_path, index=False, encoding="utf-8", na_rep="")
    result.pair_audit.to_csv(audit_path, index=False, encoding="utf-8", na_rep="")
    return comparison_path, audit_path


def write_selected_random_structures(
    selection_map: Mapping[tuple[str, str], str],
    output_path: Path,
    *,
    rationale: str = "Explicitly supplied after supervisor/student review.",
) -> Path:
    """Persist a human-approved structure map; never infer a selection."""
    rows = []
    for condition in CONDITIONS:
        for target in GAUSSIAN_TARGETS:
            structure = selection_map.get((condition, target))
            if structure not in {"RI", "RI_RS"}:
                raise ValueError(f"Missing/invalid approved random structure for {condition}/{target}")
            rows.append({
                "condition_name": condition,
                "target_name": target,
                "random_structure": structure,
                "selection_status": "approved_explicit_selection",
                "decision_rationale": rationale,
            })
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False, encoding="utf-8")
    return path


def build_cli_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--modeling-data", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def main() -> int:
    args = build_cli_parser().parse_args()
    raw = pd.read_csv(args.modeling_data)
    data = prepare_internal_modeling_data(raw)
    registry = build_internal_registry()
    executable_formulas = {
        target: spec.formula
        for target, spec in registry.items()
        if spec.fit_status == "fit_ri_and_ri_rs"
    }
    difficulty_columns = {
        target: spec.difficulty_column
        for target, spec in registry.items()
        if spec.fit_status == "fit_ri_and_ri_rs"
    }
    result = compare_random_effects(
        data,
        executable_formulas,
        difficulty_columns,
        targets=GAUSSIAN_TARGETS,
    )
    write_random_effects_outputs(result, args.output_dir)
    print(
        "comparison_valid="
        f"{result.valid}; errors={len(result.errors)}; warnings={len(result.warnings)}; "
        f"gaussian_targets={len(GAUSSIAN_TARGETS)}; features={len(FEATURE_TARGETS)}; "
        "total_error_count=blocked_pending_count_model"
    )
    return 0 if result.valid else 2


if __name__ == "__main__":
    raise SystemExit(main())
