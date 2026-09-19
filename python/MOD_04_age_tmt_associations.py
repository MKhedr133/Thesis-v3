#!/usr/bin/env python3
"""MOD-04 — Age/TMT-B association and moderation candidate models.

Every eligible condition × target × fixed-effect candidate is fitted under BOTH
candidate random-effects structures retained in the current analysis:

    RI     : random intercept only
    RI_RS  : random intercept + random difficulty slope

No random structure is selected in MOD-04. Candidate fixed-effect models are
never selected by p-value. The seven-model registry below remains explicitly
*proposed*; real analysis requires either an explicit registry JSON or the
``--use-proposed-registry`` flag.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import importlib.util
import json
import math
from pathlib import Path
import sys
from typing import Mapping

import numpy as np
import pandas as pd
from scipy.stats import norm

HERE = Path(__file__).resolve().parent
DEFAULT_RANDOM_STRUCTURES = ("RI", "RI_RS")


def _load_mod02():
    spec = importlib.util.spec_from_file_location(
        "mod02_for_age_tmt", HERE / "MOD_02_random_effects_structure.py"
    )
    if spec is None or spec.loader is None:
        raise ImportError("MOD_02_random_effects_structure.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules.setdefault(spec.name, mod)
    spec.loader.exec_module(mod)
    return mod


MOD02 = _load_mod02()
AGE_TERM = "C(participant_group, Treatment(reference='Young'))"
PROPOSED_MODEL_RHS = {
    "M0": "{D}",
    "MA": f"{{D}} + {AGE_TERM}",
    "MT": "{D} + tmt_z",
    "MAT": f"{{D}} + {AGE_TERM} + tmt_z",
    "MDA": f"{{D}} + {AGE_TERM} + {{D}}:{AGE_TERM}",
    "MDT": "{D} + tmt_z + {D}:tmt_z",
    "MDAT": f"{{D}} + {AGE_TERM} + tmt_z + {{D}}:{AGE_TERM} + {{D}}:tmt_z",
}
PROPOSED_REGISTRY_APPROVED = False


@dataclass(frozen=True)
class AgeTmtAssociationResult:
    valid: bool
    errors: tuple[str, ...]
    warnings: tuple[str, ...]
    model_fit: pd.DataFrame
    coefficients: pd.DataFrame
    checks: pd.DataFrame


def _registry(
    registry: Mapping[str, str] | None,
    use_proposed_registry: bool,
) -> dict[str, str]:
    if registry is not None:
        return dict(registry)
    if use_proposed_registry:
        return dict(PROPOSED_MODEL_RHS)
    raise ValueError(
        "No approved Age/TMT model registry supplied. "
        "The built-in seven-model registry remains proposed."
    )


def _difficulty_columns() -> dict[str, str]:
    return {
        target: spec.difficulty_column
        for target, spec in MOD02.build_internal_registry().items()
    }


def _eligible_frame(
    data: pd.DataFrame,
    condition: str,
    target: str,
    dcol: str,
) -> pd.DataFrame:
    frame = data.loc[data["condition_name"].eq(condition)].copy()
    if target == "performance_change_from_d0_percentage_points":
        if dcol == "difficulty_stage":
            raise ValueError("Performance cannot silently use difficulty_stage")
        frame = frame.loc[~frame["difficulty_level"].eq(0)].copy()

    required = ["participant_id", "participant_group", "tmt_b_seconds", target, dcol]
    missing = [c for c in required if c not in frame.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    frame = frame.loc[frame[required].notna().all(axis=1)].copy()
    if frame.empty:
        raise ValueError("No eligible rows")

    groups = set(frame["participant_group"].astype(str))
    if not {"Young", "Old"}.issubset(groups):
        raise ValueError(f"Age groups must include Young and Old; found {sorted(groups)}")

    participant = frame[["participant_id", "tmt_b_seconds"]].drop_duplicates("participant_id")
    tmt = pd.to_numeric(participant["tmt_b_seconds"], errors="coerce")
    mean = float(tmt.mean())
    sd = float(tmt.std(ddof=1))
    if not np.isfinite(sd) or sd <= 0:
        raise ValueError("Full-data TMT-B SD must be positive")
    zmap = dict(
        zip(
            participant["participant_id"].astype(str),
            ((tmt - mean) / sd).astype(float),
        )
    )
    frame["tmt_z"] = frame["participant_id"].astype(str).map(zmap)
    frame.attrs["tmt_mean"] = mean
    frame.attrs["tmt_sd"] = sd
    return frame


def fit_age_tmt_models(
    modeling_data: pd.DataFrame,
    *,
    model_rhs_registry: Mapping[str, str] | None = None,
    use_proposed_registry: bool = False,
    targets: tuple[str, ...] = MOD02.GAUSSIAN_TARGETS,
    blocked_targets: tuple[str, ...] = MOD02.PENDING_COUNT_TARGETS,
    random_structures: tuple[str, ...] = DEFAULT_RANDOM_STRUCTURES,
    difficulty_columns: Mapping[str, str] | None = None,
) -> AgeTmtAssociationResult:
    """Fit every candidate fixed-effect model under RI and RI+RS separately."""
    invalid = [s for s in random_structures if s not in {"RI", "RI_RS"}]
    if invalid:
        raise ValueError(f"Unsupported random structure(s): {invalid}")

    registry = _registry(model_rhs_registry, use_proposed_registry)
    dcols = dict(difficulty_columns or _difficulty_columns())
    data = MOD02.prepare_internal_modeling_data(modeling_data)

    fit_rows: list[dict[str, object]] = []
    coef_rows: list[dict[str, object]] = []
    check_rows: list[dict[str, object]] = []
    errors: list[str] = []
    warnings_out: list[str] = []
    zcrit = float(norm.ppf(0.975))

    for condition in MOD02.CONDITIONS:
        for target in targets:
            if target in blocked_targets:
                check_rows.append({
                    "condition_name": condition,
                    "target_name": target,
                    "random_structure": "",
                    "status": "blocked",
                    "message": "Response model family unresolved.",
                })
                continue

            dcol = dcols.get(target, "")
            if not dcol:
                errors.append(f"{condition}/{target}: missing difficulty column")
                continue
            try:
                frame = _eligible_frame(data, condition, target, dcol)
            except Exception as exc:
                errors.append(f"{condition}/{target}: {exc}")
                check_rows.append({
                    "condition_name": condition,
                    "target_name": target,
                    "random_structure": "",
                    "status": "fail",
                    "message": str(exc),
                })
                continue

            for structure in random_structures:
                family_start = len(fit_rows)
                for model_id, rhs_template in registry.items():
                    rhs = rhs_template.format(D=dcol)
                    formula = f"{target} ~ {rhs}"
                    try:
                        result, optimizer, fit_warnings, fit_errors = MOD02.fit_mixedlm_with_fallback(
                            formula, frame, structure, dcol
                        )
                        warnings_out.extend(
                            f"{condition}/{target}/{structure}/{model_id}: {w}"
                            for w in fit_warnings
                        )
                        converged = bool(
                            result is not None and getattr(result, "converged", False)
                        )
                        if result is None or not converged:
                            errors.append(
                                f"{condition}/{target}/{structure}/{model_id}: "
                                "fit failed/non-converged"
                            )
                            fit_rows.append({
                                "condition_name": condition,
                                "target_name": target,
                                "model_id": model_id,
                                "formula": formula,
                                "random_structure": structure,
                                "participant_count": int(frame["participant_id"].nunique()),
                                "observation_count": int(len(frame)),
                                "fixed_effect_count": np.nan,
                                "total_parameter_count": np.nan,
                                "log_likelihood": np.nan,
                                "aic": np.nan,
                                "delta_aic": np.nan,
                                "bic": np.nan,
                                "delta_bic": np.nan,
                                "convergence_status": (
                                    "failed" if result is None else "non_converged"
                                ),
                                "optimizer": optimizer,
                                "warnings": " | ".join(fit_warnings),
                                "fit_errors": " | ".join(fit_errors),
                                "tmt_mean_seconds": frame.attrs["tmt_mean"],
                                "tmt_sd_seconds": frame.attrs["tmt_sd"],
                            })
                            continue

                        fixed_count = len(result.fe_params)
                        total_count = int(
                            getattr(result, "df_modelwc", len(result.params))
                        ) + 1
                        fit_rows.append({
                            "condition_name": condition,
                            "target_name": target,
                            "model_id": model_id,
                            "formula": formula,
                            "random_structure": structure,
                            "participant_count": int(frame["participant_id"].nunique()),
                            "observation_count": int(len(frame)),
                            "fixed_effect_count": fixed_count,
                            "total_parameter_count": total_count,
                            "log_likelihood": float(result.llf),
                            "aic": float(result.aic),
                            "delta_aic": np.nan,
                            "bic": float(result.bic),
                            "delta_bic": np.nan,
                            "convergence_status": "converged",
                            "optimizer": optimizer,
                            "warnings": " | ".join(fit_warnings),
                            "fit_errors": " | ".join(fit_errors),
                            "tmt_mean_seconds": frame.attrs["tmt_mean"],
                            "tmt_sd_seconds": frame.attrs["tmt_sd"],
                        })

                        cov = result.cov_params().loc[
                            result.fe_params.index, result.fe_params.index
                        ]
                        for term, beta in result.fe_params.items():
                            se = math.sqrt(max(0.0, float(cov.loc[term, term])))
                            z = float(beta) / se if se > 0 else np.nan
                            p = (
                                float(2 * (1 - norm.cdf(abs(z))))
                                if np.isfinite(z)
                                else np.nan
                            )
                            coef_rows.append({
                                "condition_name": condition,
                                "target_name": target,
                                "model_id": model_id,
                                "term": term,
                                "beta": float(beta),
                                "standard_error": se,
                                "ci_lower_95": float(beta) - zcrit * se,
                                "ci_upper_95": float(beta) + zcrit * se,
                                "z": z,
                                "p_value": p,
                                "random_structure": structure,
                                "participant_count": int(frame["participant_id"].nunique()),
                                "observation_count": int(len(frame)),
                                "convergence_status": "converged",
                            })
                    except Exception as exc:
                        errors.append(
                            f"{condition}/{target}/{structure}/{model_id}: "
                            f"{type(exc).__name__}: {exc}"
                        )

                family_idx = [
                    i
                    for i in range(family_start, len(fit_rows))
                    if fit_rows[i]["condition_name"] == condition
                    and fit_rows[i]["target_name"] == target
                    and fit_rows[i]["random_structure"] == structure
                ]
                valid_aic = [
                    fit_rows[i]["aic"]
                    for i in family_idx
                    if np.isfinite(fit_rows[i]["aic"])
                ]
                valid_bic = [
                    fit_rows[i]["bic"]
                    for i in family_idx
                    if np.isfinite(fit_rows[i]["bic"])
                ]
                if valid_aic:
                    amin = min(valid_aic)
                    for i in family_idx:
                        if np.isfinite(fit_rows[i]["aic"]):
                            fit_rows[i]["delta_aic"] = fit_rows[i]["aic"] - amin
                if valid_bic:
                    bmin = min(valid_bic)
                    for i in family_idx:
                        if np.isfinite(fit_rows[i]["bic"]):
                            fit_rows[i]["delta_bic"] = fit_rows[i]["bic"] - bmin

                all_converged = bool(family_idx) and all(
                    fit_rows[i]["convergence_status"] == "converged"
                    for i in family_idx
                )
                check_rows.append({
                    "condition_name": condition,
                    "target_name": target,
                    "random_structure": structure,
                    "status": "pass" if all_converged else "warning",
                    "message": (
                        "Candidate family fitted on common eligible rows; "
                        "p-values are association evidence, not selection gates."
                    ),
                })

    return AgeTmtAssociationResult(
        not errors,
        tuple(errors),
        tuple(warnings_out),
        pd.DataFrame(fit_rows),
        pd.DataFrame(coef_rows),
        pd.DataFrame(check_rows),
    )


def write_age_tmt_outputs(
    result: AgeTmtAssociationResult,
    output_dir: Path,
) -> dict[str, Path]:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    paths = {
        "fit": out / "candidate_model_fit.csv",
        "coefficients": out / "candidate_model_coefficients.csv",
        "checks": out / "candidate_model_checks.csv",
    }
    result.model_fit.to_csv(paths["fit"], index=False, encoding="utf-8", na_rep="")
    result.coefficients.to_csv(
        paths["coefficients"], index=False, encoding="utf-8", na_rep=""
    )
    result.checks.to_csv(paths["checks"], index=False, encoding="utf-8", na_rep="")
    return paths


def _load_json(path: Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def build_cli_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--modeling-data", type=Path, required=True)
    parser.add_argument("--model-registry-json", type=Path)
    parser.add_argument("--use-proposed-registry", action="store_true")
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def main() -> int:
    args = build_cli_parser().parse_args()
    registry = _load_json(args.model_registry_json) if args.model_registry_json else None
    result = fit_age_tmt_models(
        pd.read_csv(args.modeling_data),
        model_rhs_registry=registry,
        use_proposed_registry=args.use_proposed_registry,
    )
    write_age_tmt_outputs(result, args.output_dir)
    print(
        f"age_tmt_valid={result.valid}; errors={len(result.errors)}; "
        f"warnings={len(result.warnings)}; random_structures=RI,RI_RS"
    )
    return 0 if result.valid else 2


if __name__ == "__main__":
    raise SystemExit(main())
