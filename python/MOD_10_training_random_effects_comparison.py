#!/usr/bin/env python3
"""MOD-10 — Training-only RI versus RI+RS comparison.

MOD-10 uses the participant split created by MOD-09. Only the 26 participants
labelled as training participants may enter model fitting or training-only
cross validation. The 6 final test participants remain completely excluded
from model development.

This first implementation establishes and validates that data boundary.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

import MOD_02_random_effects_structure as MOD02
import MOD_05_lopo_predictive_comparison as MOD05

SCRIPT_VERSION = "1.0.0"
METHOD_ID = "training26_mdat_ri_vs_ri_rs_lopo_v1"

CONDITIONS = tuple(MOD02.CONDITIONS)
TARGETS = tuple(MOD02.GAUSSIAN_TARGETS)
RANDOM_STRUCTURES = ("RI", "RI_RS")

COMPARISON_FILENAME = "training_random_effects_comparison.csv"
AUDIT_FILENAME = "training_random_effects_audit.csv"
LOPO_SUMMARY_FILENAME = "lopo_model_summary.csv"
PREDICTIONS_FILENAME = "lopo_predictions.csv"
PARTICIPANT_ERRORS_FILENAME = "lopo_participant_errors.csv"
LOPO_CHECKS_FILENAME = "lopo_checks.csv"
MANIFEST_FILENAME = "mod10_manifest.json"

PARTICIPANT_COLUMN = "participant_id"
GROUP_COLUMN = "participant_group"
SPLIT_COLUMN = "split"

EXPECTED_PARTICIPANTS = 32
EXPECTED_TRAINING_PARTICIPANTS = 26
EXPECTED_TEST_PARTICIPANTS = 6

EXPECTED_TRAIN_GROUP_COUNTS = {
    "Young": 13,
    "Old": 13,
}

EXPECTED_TEST_GROUP_COUNTS = {
    "Young": 3,
    "Old": 3,
}


class Mod10Error(ValueError):
    """Raised when the MOD-10 training-data contract is violated."""


def build_target_specs() -> dict[str, MOD02.TargetSpec]:
    """Return the 10 Gaussian MOD-10 targets with the MDAT fixed specification.

    MOD-10 deliberately reuses the MOD-02 target registry so that the
    difficulty coding and fixed-effects formulas remain identical to the
    previously defined modelling specification.

    The common fixed-effects structure is MDAT:

        Difficulty
        + Age
        + TMT-B
        + Difficulty × Age
        + Difficulty × TMT-B

    Relative performance uses performance_difficulty_stage. The remaining
    Gaussian targets use difficulty_stage.
    """
    registry = MOD02.build_internal_registry()

    return {
        target: registry[target]
        for target in MOD02.GAUSSIAN_TARGETS
    }


def _normalise_string_column(
    frame: pd.DataFrame,
    column: str,
) -> pd.Series:
    """Return a stripped string column after validating missing/blank values."""
    if column not in frame.columns:
        raise Mod10Error(f"Missing required column: {column}")

    if frame[column].isna().any():
        raise Mod10Error(f"{column} contains missing values")

    values = frame[column].astype(str).str.strip()

    if values.eq("").any():
        raise Mod10Error(f"{column} contains blank values")

    return values


def validate_holdout_split(
    holdout_split: pd.DataFrame,
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Validate the MOD-09 split and return training and test IDs."""
    split = holdout_split.copy(deep=True)

    split[PARTICIPANT_COLUMN] = _normalise_string_column(
        split,
        PARTICIPANT_COLUMN,
    )
    split[GROUP_COLUMN] = _normalise_string_column(
        split,
        GROUP_COLUMN,
    )
    split[SPLIT_COLUMN] = _normalise_string_column(
        split,
        SPLIT_COLUMN,
    )

    if len(split) != EXPECTED_PARTICIPANTS:
        raise Mod10Error(
            "MOD-10 requires exactly 32 rows in the participant holdout split; "
            f"found {len(split)}"
        )

    if split[PARTICIPANT_COLUMN].nunique() != EXPECTED_PARTICIPANTS:
        raise Mod10Error(
            "MOD-10 requires exactly 32 unique participant IDs "
            "in the participant holdout split"
        )

    allowed_groups = {"Young", "Old"}
    observed_groups = set(split[GROUP_COLUMN])

    if observed_groups != allowed_groups:
        raise Mod10Error(
            "participant_group must contain only Young and Old"
        )

    allowed_split_labels = {"train", "test"}
    observed_split_labels = set(split[SPLIT_COLUMN])

    if observed_split_labels != allowed_split_labels:
        raise Mod10Error(
            "split must contain only train and test labels"
        )

    training = split.loc[split[SPLIT_COLUMN].eq("train")].copy()
    test = split.loc[split[SPLIT_COLUMN].eq("test")].copy()

    if len(training) != EXPECTED_TRAINING_PARTICIPANTS:
        raise Mod10Error(
            "MOD-10 requires exactly 26 training participants; "
            f"found {len(training)}"
        )

    if len(test) != EXPECTED_TEST_PARTICIPANTS:
        raise Mod10Error(
            "MOD-10 requires exactly 6 final test participants; "
            f"found {len(test)}"
        )

    training_group_counts = (
        training[GROUP_COLUMN]
        .value_counts()
        .to_dict()
    )
    test_group_counts = (
        test[GROUP_COLUMN]
        .value_counts()
        .to_dict()
    )

    if training_group_counts != EXPECTED_TRAIN_GROUP_COUNTS:
        raise Mod10Error(
            "Training split must contain exactly "
            "13 Young and 13 Old participants"
        )

    if test_group_counts != EXPECTED_TEST_GROUP_COUNTS:
        raise Mod10Error(
            "Test split must contain exactly "
            "3 Young and 3 Old participants"
        )

    training_ids = tuple(
        sorted(training[PARTICIPANT_COLUMN].tolist())
    )
    test_ids = tuple(
        sorted(test[PARTICIPANT_COLUMN].tolist())
    )

    if set(training_ids) & set(test_ids):
        raise Mod10Error(
            "Training and test participant IDs must be disjoint"
        )

    return training_ids, test_ids


def prepare_training_data(
    *,
    modeling_data: pd.DataFrame,
    holdout_split: pd.DataFrame,
) -> tuple[pd.DataFrame, tuple[str, ...], tuple[str, ...]]:
    """Return modelling rows belonging only to the 26 training participants."""
    training_ids, test_ids = validate_holdout_split(holdout_split)

    data = modeling_data.copy(deep=True)

    data[PARTICIPANT_COLUMN] = _normalise_string_column(
        data,
        PARTICIPANT_COLUMN,
    )

    modeling_ids = set(data[PARTICIPANT_COLUMN])
    split_ids = set(training_ids) | set(test_ids)

    missing_from_modeling = sorted(split_ids - modeling_ids)
    unexpected_in_modeling = sorted(modeling_ids - split_ids)

    if missing_from_modeling:
        raise Mod10Error(
            "Participants from the MOD-09 split are missing from modeling_data: "
            + ", ".join(missing_from_modeling)
        )

    if unexpected_in_modeling:
        raise Mod10Error(
            "modeling_data contains participants not present in the MOD-09 split: "
            + ", ".join(unexpected_in_modeling)
        )

    training_data = data.loc[
        data[PARTICIPANT_COLUMN].isin(training_ids)
    ].copy()

    observed_training_ids = set(training_data[PARTICIPANT_COLUMN])

    if observed_training_ids != set(training_ids):
        raise Mod10Error(
            "Filtered MOD-10 data do not contain exactly the "
            "26 expected training participants"
        )

    if not observed_training_ids.isdisjoint(test_ids):
        raise Mod10Error(
            "Final test participants entered MOD-10 training data"
        )

    if training_data[PARTICIPANT_COLUMN].nunique() != (
        EXPECTED_TRAINING_PARTICIPANTS
    ):
        raise Mod10Error(
            "MOD-10 training data must contain exactly 26 participants"
        )

    return training_data, training_ids, test_ids

def _safe_float(value: Any) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return np.nan
    return numeric if np.isfinite(numeric) else np.nan


def _random_effect_correlation(row: pd.Series) -> float:
    ri_var = _safe_float(row.get("random_intercept_variance"))
    rs_var = _safe_float(row.get("random_slope_variance"))
    covariance = _safe_float(row.get("intercept_slope_covariance"))

    if (
        not np.isfinite(ri_var)
        or not np.isfinite(rs_var)
        or not np.isfinite(covariance)
        or ri_var <= 0
        or rs_var <= 0
    ):
        return np.nan

    return covariance / np.sqrt(ri_var * rs_var)


def run_full_training_comparison(
    training_data: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fit MDAT RI and RI+RS models on the 26 training participants."""
    if training_data[PARTICIPANT_COLUMN].nunique() != (
        EXPECTED_TRAINING_PARTICIPANTS
    ):
        raise Mod10Error(
            "Full MOD-10 comparison requires exactly 26 training participants"
        )

    prepared = MOD02.prepare_internal_modeling_data(training_data)
    specs = build_target_specs()

    formulas = {
        target: spec.formula
        for target, spec in specs.items()
    }

    difficulty_columns = {
        target: spec.difficulty_column
        for target, spec in specs.items()
    }

    result = MOD02.compare_random_effects(
        prepared,
        executable_formulas=formulas,
        difficulty_columns=difficulty_columns,
        targets=TARGETS,
        conditions=CONDITIONS,
    )

    comparison = result.comparison.copy()

    # The underlying utility also registers the unresolved count outcome in
    # its audit table. MOD-10 is restricted to the 10 approved Gaussian targets.
    audit = result.pair_audit.loc[
        result.pair_audit["target_name"].isin(TARGETS)
    ].copy()

    if len(comparison):
        participant_counts = pd.to_numeric(
            comparison["participant_count"],
            errors="coerce",
        )
        if not participant_counts.eq(
            EXPECTED_TRAINING_PARTICIPANTS
        ).all():
            raise Mod10Error(
                "At least one full-data RI/RI+RS comparison did not use "
                "all 26 training participants"
            )

    return comparison, audit


def _full_fit_wide(
    full_comparison: pd.DataFrame,
) -> pd.DataFrame:
    """Convert MOD-02 RI/RI+RS evidence to one row per condition × target."""
    rows: list[dict[str, Any]] = []

    for condition in CONDITIONS:
        for target in TARGETS:
            pair = full_comparison.loc[
                full_comparison["condition_name"].eq(condition)
                & full_comparison["target_name"].eq(target)
            ].copy()

            ri_rows = pair.loc[pair["random_structure"].eq("RI")]
            rs_rows = pair.loc[pair["random_structure"].eq("RI_RS")]

            if len(ri_rows) != 1 or len(rs_rows) != 1:
                raise Mod10Error(
                    f"{condition}/{target}: expected exactly one RI and one RI+RS fit"
                )

            ri = ri_rows.iloc[0]
            rs = rs_rows.iloc[0]

            same_signature = (
                str(ri["included_row_signature"])
                == str(rs["included_row_signature"])
            )

            if not same_signature:
                raise Mod10Error(
                    f"{condition}/{target}: RI and RI+RS did not use identical rows"
                )

            row: dict[str, Any] = {
                "condition_name": condition,
                "target_name": target,
                "fixed_effects_model": "MDAT",
                "fixed_effects_formula": ri["fixed_effects_formula"],
                "difficulty_source_column": ri["difficulty_source_column"],
                "participant_count": int(ri["participant_count"]),
                "observation_count": int(ri["observation_count"]),
                "excluded_structural_d0_count": int(
                    ri["excluded_structural_d0_count"]
                ),
                "included_row_signature": ri["included_row_signature"],
            }

            metrics = (
                "log_likelihood",
                "aic",
                "bic",
                "convergence_status",
                "optimizer",
                "warnings",
                "fit_errors",
                "random_intercept_variance",
                "random_slope_variance",
                "intercept_slope_covariance",
            )

            for prefix, source in (("ri", ri), ("ri_rs", rs)):
                for metric in metrics:
                    row[f"{prefix}_{metric}"] = source.get(metric, np.nan)

            row["ri_rs_intercept_slope_correlation"] = (
                _random_effect_correlation(rs)
            )

            ri_aic = _safe_float(ri["aic"])
            rs_aic = _safe_float(rs["aic"])
            ri_bic = _safe_float(ri["bic"])
            rs_bic = _safe_float(rs["bic"])

            row["delta_aic_ri_rs_minus_ri"] = (
                rs_aic - ri_aic
                if np.isfinite(rs_aic) and np.isfinite(ri_aic)
                else np.nan
            )

            row["delta_bic_ri_rs_minus_ri"] = (
                rs_bic - ri_bic
                if np.isfinite(rs_bic) and np.isfinite(ri_bic)
                else np.nan
            )

            row["paired_converged"] = (
                str(ri["convergence_status"]) == "converged"
                and str(rs["convergence_status"]) == "converged"
            )

            rows.append(row)

    return pd.DataFrame(rows)


def run_training_lopo(
    training_data: pd.DataFrame,
    *,
    jobs: int = 1,
    checkpoint_dir: Path | None = None,
    resume: bool = True,
    progress: bool = True,
) -> MOD05.LopoResult:
    """Run 26-fold training-only LOPO for MDAT under RI and RI+RS."""
    participant_count = int(
        training_data[PARTICIPANT_COLUMN].nunique()
    )

    if participant_count != EXPECTED_TRAINING_PARTICIPANTS:
        raise Mod10Error(
            "MOD-10 LOPO requires exactly 26 training participants"
        )

    specs = build_target_specs()

    difficulty_columns = {
        target: spec.difficulty_column
        for target, spec in specs.items()
    }

    # MOD-05 inserts the target on the left-hand side and substitutes {D}.
    mdat_rhs = MOD02.FIXED_EFFECTS_TEMPLATE

    return MOD05.run_lopo_prediction(
        training_data,
        model_rhs_registry={"MDAT": mdat_rhs},
        use_proposed_registry=False,
        targets=TARGETS,
        blocked_targets=(),
        conditions=CONDITIONS,
        random_structures=RANDOM_STRUCTURES,
        difficulty_columns=difficulty_columns,
        jobs=jobs,
        checkpoint_dir=checkpoint_dir,
        resume=resume,
        progress=progress,
        progress_every=1,
    )


def _lopo_wide(
    lopo_summary: pd.DataFrame,
) -> pd.DataFrame:
    """Return RI-versus-RI+RS LOPO evidence per condition × target.

    A completely failed LOPO family may have no model-summary row because
    MOD-05 constructs summaries only from successful folds. Such a failure is
    retained as missing predictive evidence rather than aborting MOD-10.
    """
    summary = lopo_summary.loc[
        lopo_summary["model_id"].eq("MDAT")
    ].copy()

    rows: list[dict[str, Any]] = []

    for condition in CONDITIONS:
        for target in TARGETS:
            family = summary.loc[
                summary["condition_name"].eq(condition)
                & summary["target_name"].eq(target)
            ]

            row: dict[str, Any] = {
                "condition_name": condition,
                "target_name": target,
            }

            for structure, prefix in (
                ("RI", "ri"),
                ("RI_RS", "ri_rs"),
            ):
                structure_rows = family.loc[
                    family["random_structure"].eq(structure)
                ]

                if len(structure_rows) > 1:
                    raise Mod10Error(
                        f"{condition}/{target}/{structure}: "
                        "multiple LOPO summary rows found"
                    )

                if len(structure_rows) == 0:
                    row[f"{prefix}_lopo_status"] = "failed_no_summary"
                    row[f"{prefix}_lopo_participant_count"] = 0
                    row[f"{prefix}_lopo_prediction_count"] = 0
                    row[f"{prefix}_lopo_mae"] = np.nan
                    row[f"{prefix}_lopo_r2"] = np.nan
                    continue

                source = structure_rows.iloc[0]

                participant_count = int(source["participant_count"])
                prediction_count = int(source["prediction_count"])

                row[f"{prefix}_lopo_status"] = (
                    "complete"
                    if participant_count == EXPECTED_TRAINING_PARTICIPANTS
                    else "partial"
                )
                row[f"{prefix}_lopo_participant_count"] = participant_count
                row[f"{prefix}_lopo_prediction_count"] = prediction_count
                row[f"{prefix}_lopo_mae"] = _safe_float(
                    source["participant_balanced_mae"]
                )
                row[f"{prefix}_lopo_r2"] = _safe_float(
                    source["lopo_r2"]
                )

            ri_mae = row["ri_lopo_mae"]
            rs_mae = row["ri_rs_lopo_mae"]
            ri_r2 = row["ri_lopo_r2"]
            rs_r2 = row["ri_rs_lopo_r2"]

            row["delta_lopo_mae_ri_rs_minus_ri"] = (
                rs_mae - ri_mae
                if np.isfinite(rs_mae) and np.isfinite(ri_mae)
                else np.nan
            )

            row["delta_lopo_r2_ri_rs_minus_ri"] = (
                rs_r2 - ri_r2
                if np.isfinite(rs_r2) and np.isfinite(ri_r2)
                else np.nan
            )

            rows.append(row)

    return pd.DataFrame(rows)


def combine_evidence(
    full_comparison: pd.DataFrame,
    lopo_summary: pd.DataFrame,
) -> pd.DataFrame:
    """Combine full-training AIC/BIC evidence with training-only LOPO."""
    full_wide = _full_fit_wide(full_comparison)
    lopo_wide = _lopo_wide(lopo_summary)

    combined = full_wide.merge(
        lopo_wide,
        on=["condition_name", "target_name"],
        how="left",
        validate="one_to_one",
    )

    combined["selection_status"] = (
        "evidence_only_no_automatic_selection"
    )
    combined["lopo_prediction_scope"] = (
        "fixed_effect_population_only"
    )

    return combined


def build_audit(
    *,
    full_audit: pd.DataFrame,
    lopo_checks: pd.DataFrame,
    comparison: pd.DataFrame,
) -> pd.DataFrame:
    """Build one final audit row per condition × target."""
    rows: list[dict[str, Any]] = []

    for condition in CONDITIONS:
        for target in TARGETS:
            full = full_audit.loc[
                full_audit["condition_name"].eq(condition)
                & full_audit["target_name"].eq(target)
            ]

            checks = lopo_checks.loc[
                lopo_checks["condition_name"].eq(condition)
                & lopo_checks["target_name"].eq(target)
                & lopo_checks["random_structure"].isin(
                    RANDOM_STRUCTURES
                )
            ]

            comp = comparison.loc[
                comparison["condition_name"].eq(condition)
                & comparison["target_name"].eq(target)
            ]

            full_ok = (
                len(full) == 1
                and str(full.iloc[0]["status"]) == "comparable"
            )

            lopo_ok = (
                len(checks) == 2
                and checks["status"].eq("pass").all()
            )

            summary_ok = False

            if len(comp) == 1:
                row = comp.iloc[0]

                summary_ok = (
                    int(row["participant_count"]) == 26
                    and row["ri_lopo_status"] == "complete"
                    and row["ri_rs_lopo_status"] == "complete"
                    and int(row["ri_lopo_participant_count"]) == 26
                    and int(row["ri_rs_lopo_participant_count"]) == 26
                )

            status = (
                "pass"
                if full_ok and lopo_ok and summary_ok
                else "problem"
            )

            rows.append(
                {
                    "condition_name": condition,
                    "target_name": target,
                    "status": status,
                    "full_training_pair_ok": full_ok,
                    "ri_lopo_ok": bool(
                        (
                            checks.loc[
                                checks["random_structure"].eq("RI"),
                                "status",
                            ]
                            == "pass"
                        ).all()
                    )
                    if len(
                        checks.loc[
                            checks["random_structure"].eq("RI")
                        ]
                    )
                    else False,
                    "ri_rs_lopo_ok": bool(
                        (
                            checks.loc[
                                checks["random_structure"].eq("RI_RS"),
                                "status",
                            ]
                            == "pass"
                        ).all()
                    )
                    if len(
                        checks.loc[
                            checks["random_structure"].eq("RI_RS")
                        ]
                    )
                    else False,
                    "complete_26_participant_lopo": summary_ok,
                    "final_test_participants_used": 0,
                }
            )

    return pd.DataFrame(rows)


def build_manifest(
    *,
    modeling_data_path: Path,
    split_path: Path,
    jobs: int,
    checkpoint_dir: Path | None,
) -> dict[str, Any]:
    """Return the MOD-10 method and leakage-boundary manifest."""
    return {
        "script_version": SCRIPT_VERSION,
        "method_id": METHOD_ID,
        "analysis_role": "training_only_random_effects_comparison",
        "training_participant_count": 26,
        "final_test_participant_count": 6,
        "final_test_participants_used": 0,
        "fixed_effects_model": "MDAT",
        "fixed_effects_template": MOD02.FIXED_EFFECTS_TEMPLATE,
        "random_structures": ["RI", "RI_RS"],
        "condition_count": len(CONDITIONS),
        "gaussian_target_count": len(TARGETS),
        "condition_target_pair_count": len(CONDITIONS) * len(TARGETS),
        "cross_validation": "leave_one_training_participant_out",
        "lopo_fold_count": 26,
        "participants_fit_per_fold": 25,
        "heldout_training_participants_per_fold": 1,
        "heldout_prediction_scope": "fixed_effect_population_only",
        "full_training_estimation": "maximum_likelihood",
        "reml": False,
        "optimizers": ["lbfgs", "powell"],
        "automatic_random_structure_selection": False,
        "delta_conventions": {
            "LOPO_MAE": (
                "RI+RS minus RI; negative means lower MAE for RI+RS"
            ),
            "LOPO_R2": (
                "RI+RS minus RI; positive means higher R2 for RI+RS"
            ),
            "AIC": (
                "RI+RS minus RI; negative means lower AIC for RI+RS"
            ),
            "BIC": (
                "RI+RS minus RI; negative means lower BIC for RI+RS"
            ),
        },
        "modeling_data_path": str(modeling_data_path),
        "participant_split_path": str(split_path),
        "jobs": int(jobs),
        "checkpoint_dir": (
            str(checkpoint_dir)
            if checkpoint_dir is not None
            else None
        ),
    }


def _refuse_existing_outputs(output_dir: Path) -> None:
    filenames = (
        COMPARISON_FILENAME,
        AUDIT_FILENAME,
        LOPO_SUMMARY_FILENAME,
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
            "Refusing to overwrite existing MOD-10 outputs: "
            + ", ".join(path.name for path in existing)
        )


def write_outputs(
    *,
    comparison: pd.DataFrame,
    audit: pd.DataFrame,
    lopo: MOD05.LopoResult,
    manifest: Mapping[str, Any],
    output_dir: Path,
) -> dict[str, Path]:
    """Write MOD-10 outputs without silently replacing an earlier run."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    _refuse_existing_outputs(output_dir)

    paths = {
        "comparison": output_dir / COMPARISON_FILENAME,
        "audit": output_dir / AUDIT_FILENAME,
        "lopo_summary": output_dir / LOPO_SUMMARY_FILENAME,
        "predictions": output_dir / PREDICTIONS_FILENAME,
        "participant_errors": output_dir / PARTICIPANT_ERRORS_FILENAME,
        "lopo_checks": output_dir / LOPO_CHECKS_FILENAME,
        "manifest": output_dir / MANIFEST_FILENAME,
    }

    comparison.to_csv(paths["comparison"], index=False)
    audit.to_csv(paths["audit"], index=False)
    lopo.model_summary.to_csv(
        paths["lopo_summary"],
        index=False,
    )
    lopo.predictions.to_csv(
        paths["predictions"],
        index=False,
    )
    lopo.participant_errors.to_csv(
        paths["participant_errors"],
        index=False,
    )
    lopo.checks.to_csv(
        paths["lopo_checks"],
        index=False,
    )
    paths["manifest"].write_text(
        json.dumps(
            dict(manifest),
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    return paths


def run_mod10(
    *,
    modeling_data: pd.DataFrame,
    holdout_split: pd.DataFrame,
    jobs: int = 1,
    checkpoint_dir: Path | None = None,
    resume: bool = True,
    progress: bool = True,
):
    """Run complete training-only RI versus RI+RS comparison."""
    training_data, training_ids, test_ids = prepare_training_data(
        modeling_data=modeling_data,
        holdout_split=holdout_split,
    )

    if set(training_data[PARTICIPANT_COLUMN]) & set(test_ids):
        raise Mod10Error(
            "Leakage detected before model fitting"
        )

    full_comparison, full_audit = (
        run_full_training_comparison(training_data)
    )

    lopo = run_training_lopo(
        training_data,
        jobs=jobs,
        checkpoint_dir=checkpoint_dir,
        resume=resume,
        progress=progress,
    )

    comparison = combine_evidence(
        full_comparison,
        lopo.model_summary,
    )

    audit = build_audit(
        full_audit=full_audit,
        lopo_checks=lopo.checks,
        comparison=comparison,
    )

    return (
        comparison,
        audit,
        lopo,
        training_ids,
        test_ids,
    )


def build_cli_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument(
        "--modeling-data",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--participant-split",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--jobs",
        type=int,
        default=0,
        help="Parallel LOPO families. 0 uses the MOD-05 automatic cap.",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
    )

    return parser


def main() -> int:
    args = build_cli_parser().parse_args()

    modeling_data = pd.read_csv(args.modeling_data)
    holdout_split = pd.read_csv(args.participant_split)

    jobs = MOD05.resolve_job_count(args.jobs)

    checkpoint_dir = (
        args.checkpoint_dir
        if args.checkpoint_dir is not None
        else args.output_dir / "_lopo_checkpoints"
    )

    comparison, audit, lopo, training_ids, test_ids = run_mod10(
        modeling_data=modeling_data,
        holdout_split=holdout_split,
        jobs=jobs,
        checkpoint_dir=checkpoint_dir,
        resume=not args.no_resume,
        progress=not args.no_progress,
    )

    manifest = build_manifest(
        modeling_data_path=args.modeling_data,
        split_path=args.participant_split,
        jobs=jobs,
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
        "[MOD-10] completed training-only RI versus RI+RS comparison"
    )
    print(
        f"[MOD-10] training participants={len(training_ids)}; "
        f"final test participants used=0/{len(test_ids)}"
    )
    print(
        f"[MOD-10] condition-target comparisons={len(comparison)}"
    )
    print(
        f"[MOD-10] LOPO errors={len(lopo.errors)}; "
        f"warnings={len(lopo.warnings)}"
    )

    for name, path in paths.items():
        print(f"[MOD-10] {name}: {path}")

    return (
        0
        if len(audit)
        and audit["status"].eq("pass").all()
        and not lopo.errors
        else 1
    )


if __name__ == "__main__":
    raise SystemExit(main())