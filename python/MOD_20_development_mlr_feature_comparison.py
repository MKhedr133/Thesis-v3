#!/usr/bin/env python3
"""Compare three MLR feature-selection procedures on development trials only."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels
import statsmodels.api as sm
import sklearn

import MOD_02_random_effects_structure as MOD02
import MOD_14_direct_linear_regression_evaluation as MOD14
import MOD_15_visual_auditory_alternative_models as MOD15

TARGET = "difficulty_stage"
TRIAL_KEY = ["participant_id", "condition_name", TARGET]
TRIAL_FEATURES = (
    "correct_products_collected_percent",
    "self_reported_mental_demand",
    "median_time_between_grabs_seconds",
    "shopping_list_recheck_count",
    "shopping_list_recheck_total_seconds",
    "median_product_search_duration_seconds",
    "median_non_task_focus_seconds",
    "median_head_rotation_during_search_degrees",
    "median_reach_start_to_grab_seconds",
    "median_hand_travel_to_straight_distance_ratio",
)
ADD_BACK = {
    "Visual": ("median_product_search_duration_seconds",
               "median_head_rotation_during_search_degrees"),
    "Auditory": ("median_time_between_grabs_seconds",
                 "median_product_search_duration_seconds"),
    "Cognitive": ("shopping_list_recheck_count", "shopping_list_recheck_total_seconds"),
}
P_CUTOFF = 0.05
MAE_MARGIN = 0.05
CANDIDATES = ("full", "screened", "add_back")


@dataclass(frozen=True)
class Comparison:
    summary: pd.DataFrame
    rounds: pd.DataFrame
    predictions: pd.DataFrame
    screen: pd.DataFrame
    coefficients: pd.DataFrame


def prepare_development(trials: pd.DataFrame, split: pd.DataFrame) -> pd.DataFrame:
    """Keep only frozen development measurements before validating trial values."""
    development, reserved = MOD14.validate_holdout_split(split)
    required = [*TRIAL_KEY, *TRIAL_FEATURES]
    missing = set(required) - set(trials)
    if missing:
        raise ValueError(f"Missing comparison columns: {sorted(missing)}")
    ids = trials["participant_id"].astype("string").str.strip()
    if not ids.isin([*development, *reserved]).all():
        raise ValueError("Trial input contains a participant outside the frozen split")
    rows = trials.loc[ids.isin(development), required].copy()
    rows["participant_id"] = ids.loc[rows.index]
    rows[TARGET] = pd.to_numeric(rows[TARGET], errors="raise")
    expected = pd.MultiIndex.from_product(
        [development, tuple(ADD_BACK), range(4)], names=TRIAL_KEY,
    )
    observed = pd.MultiIndex.from_frame(rows[TRIAL_KEY])
    if not observed.is_unique or len(observed) != len(expected) or not observed.isin(expected).all():
        raise ValueError("Expected one trial per development participant, condition and stage")
    rows[TARGET] = rows[TARGET].astype(int)
    if "recorded_difficulty_code" in trials:
        recorded = pd.to_numeric(trials.loc[rows.index, "recorded_difficulty_code"], errors="raise")
        expected_stage = recorded.map(MOD14.DIFFICULTY_STAGE_BY_LEVEL)
        if not rows[TARGET].eq(expected_stage).all():
            raise ValueError("Recorded difficulty codes do not match development stages")
    rows[list(TRIAL_FEATURES)] = rows[list(TRIAL_FEATURES)].apply(
        pd.to_numeric, errors="raise",
    )
    if np.isinf(rows[list(TRIAL_FEATURES)].to_numpy(dtype=float, na_value=np.nan)).any():
        raise ValueError("Non-finite predictor values must be represented as missing, not infinity")
    return rows.sort_values(TRIAL_KEY).reset_index(drop=True)


def candidate_features(
    features: tuple[str, ...], pvalues: dict[str, float], condition: str,
) -> dict[str, tuple[str, ...]]:
    """Restore only removed members of the condition's predefined pair."""
    screened = tuple(feature for feature in features if pvalues[feature] < P_CUTOFF)
    add_back = tuple(
        feature for feature in features if feature in screened or feature in ADD_BACK[condition]
    )
    return {"full": features, "screened": screened, "add_back": add_back}


def select_candidate(summary: pd.DataFrame) -> str:
    """Prefer fewer predictors within the fixed MAE margin, then lower MAE."""
    indexed = summary.set_index("candidate")
    full = indexed.loc["full"]
    eligible = []
    for name in CANDIDATES[1:]:
        candidate = indexed.loc[name]
        loss = candidate["participant_balanced_mae"] - full["participant_balanced_mae"]
        # Allow for binary floating-point representation of the inclusive boundary.
        within_margin = loss <= MAE_MARGIN or np.isclose(loss, MAE_MARGIN, rtol=0, atol=1e-12)
        if candidate["mean_predictor_count"] < full["mean_predictor_count"] and within_margin:
            eligible.append((candidate["mean_predictor_count"], candidate["participant_balanced_mae"], name))
    return min(eligible, key=lambda score: score[:2])[2] if eligible else "full"


def _design(rows: pd.DataFrame, features: tuple[str, ...]) -> pd.DataFrame:
    x = rows[list(features)].astype(float).copy()
    x.insert(0, "intercept", 1.0)
    return x


def _fit_ols(rows: pd.DataFrame, features: tuple[str, ...]) -> pd.Series:
    x = _design(rows, features)
    if len(x) <= x.shape[1] or np.linalg.matrix_rank(x.to_numpy()) < x.shape[1]:
        raise ValueError("OLS training design has insufficient rows or is rank deficient")
    coefficients = sm.OLS(rows[TARGET].astype(float), x).fit().params
    if not np.isfinite(coefficients).all():
        raise ValueError("OLS returned non-finite coefficients")
    return coefficients


def _screen(rows: pd.DataFrame, condition: str, label: str) -> tuple[dict[str, float], list[dict]]:
    result, optimizer, fit_warnings, fit_errors = MOD02.fit_mixedlm_with_fallback(
        f"{TARGET} ~ " + " + ".join(TRIAL_FEATURES), rows, "RI", TARGET, reml=True,
    )
    diagnostic = " | ".join((*fit_warnings, *fit_errors))
    if result is None or not result.converged:
        raise ValueError(f"{condition}, {label}: mixed-model screen did not converge. {diagnostic}")
    pvalues = result.pvalues.reindex(TRIAL_FEATURES)
    if not np.isfinite(pvalues).all():
        raise ValueError(f"{condition}, {label}: invalid mixed-model p-values. {diagnostic}")
    audit = [
        {"condition": condition, "round": label, "feature": feature, "p_value": float(p),
         "retained_by_screen": bool(p < P_CUTOFF), "optimizer": optimizer,
         "random_intercept_variance": float(result.cov_re.iloc[0, 0]), "diagnostic": diagnostic}
        for feature, p in pvalues.items()
    ]
    return pvalues.to_dict(), audit


def compare_condition(rows: pd.DataFrame, condition: str) -> Comparison:
    """Compare candidates on identical rows using single-layer grouped CV."""
    rows = rows.reset_index(drop=True)
    if set(rows["condition_name"]) != {condition}:
        raise ValueError("Pass trial rows from exactly the requested condition")
    splits = MOD15.make_participant_cv_splits(rows)
    prediction_blocks = []
    round_audit = []
    screen_audit = []
    counts = {}
    for number, (train_index, validation_index) in enumerate(splits, start=1):
        label = str(number)
        training = rows.iloc[train_index]
        validation = rows.iloc[validation_index]
        train_common = training.dropna(subset=list(TRIAL_FEATURES))
        score_common = validation.dropna(subset=list(TRIAL_FEATURES))
        if set(train_common.participant_id) != set(training.participant_id):
            raise ValueError(f"{condition}, round {label}: a training participant has no complete trial")
        if set(score_common.participant_id) != set(validation.participant_id):
            raise ValueError(f"{condition}, round {label}: a validation participant has no paired trial")

        # Fit all candidates on the same rows to isolate the change in predictors.
        full_coefficients = _fit_ols(train_common, TRIAL_FEATURES)
        pvalues, audit = _screen(train_common, condition, label)
        screen_audit.extend(audit)
        sets = candidate_features(TRIAL_FEATURES, pvalues, condition)
        fitted_by_features = {TRIAL_FEATURES: full_coefficients}
        for name, features in sets.items():
            if features not in fitted_by_features:
                fitted_by_features[features] = _fit_ols(train_common, features)
            model_predictions = _design(score_common, features).dot(fitted_by_features[features])
            if not np.isfinite(model_predictions).all():
                raise ValueError(f"{condition}, round {label}: non-finite predictions for {name}")
            prediction_blocks.append(score_common[TRIAL_KEY].assign(
                round=label, candidate=name, predicted_stage=model_predictions,
            ))
            native_coverage = validation[list(features)].notna().all(axis=1)
            round_audit.append({
                "condition": condition, "round": label, "candidate": name,
                "predictors": ";".join(features), "predictor_count": len(features),
                "training_participants": training.participant_id.nunique(),
                "training_trials": len(train_common),
                "validation_participants": validation.participant_id.nunique(),
                "validation_trials": len(validation),
                "paired_scored_trials": len(score_common),
                "feature_available_trials": int(native_coverage.sum()),
            })
    predictions = pd.concat(prediction_blocks, ignore_index=True)
    rounds = pd.DataFrame(round_audit)
    summary = []
    for candidate in CANDIDATES:
        block = predictions.loc[predictions.candidate.eq(candidate)]
        audits = rounds.loc[rounds.candidate.eq(candidate)]
        summary.append({
            "condition": condition,
            "candidate": candidate,
            "validation_participants": block.participant_id.nunique(),
            "validation_trials": int(audits.validation_trials.sum()),
            "paired_scored_trials": len(block),
            "feature_available_trials": int(audits.feature_available_trials.sum()),
            "mean_predictor_count": float(audits.predictor_count.mean()),
            "participant_balanced_mae": MOD14.participant_balanced_mae(
                participant_ids=block.participant_id, observed=block[TARGET],
                predicted=block.predicted_stage,
            ),
            "pooled_r2": MOD14.pooled_r2(observed=block[TARGET], predicted=block.predicted_stage),
        })
    summary = pd.DataFrame(summary)
    summary["mae_change_vs_full"] = (
        summary.participant_balanced_mae
        - summary.loc[summary.candidate.eq("full"), "participant_balanced_mae"].iloc[0]
    )
    selected = select_candidate(summary)
    summary["selected"] = summary.candidate.eq(selected)

    final_rows = rows.dropna(subset=list(TRIAL_FEATURES))
    features = TRIAL_FEATURES
    if selected != "full":
        final_pvalues, audit = _screen(final_rows, condition, "final")
        screen_audit.extend(audit)
        features = candidate_features(TRIAL_FEATURES, final_pvalues, condition)[selected]
    coefficients = _fit_ols(final_rows, features)
    final = pd.DataFrame({
        "condition": condition, "candidate": selected, "term": coefficients.index,
        "coefficient": coefficients.to_numpy(), "training_trials": len(final_rows),
        "training_participants": final_rows.participant_id.nunique(),
    })
    return Comparison(summary, rounds, predictions, pd.DataFrame(screen_audit), final)


def _read_fe04(path: Path, manifest_path: Path) -> tuple[pd.DataFrame, str]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
    if (manifest.get("schema_version") != "fe04_all_grabs_v1"
            or manifest.get("search_and_reach_grab_scope") != "all_valid_product_grabs"
            or manifest.get("reach_requires_target_focus") is not True):
        raise ValueError("Expected the FE-04 all-grab feature extraction manifest")
    if manifest.get("trial_count") != 384:
        raise ValueError("Expected a complete FE-04 export with 384 trial rows")
    digest = sha256(path.read_bytes()).hexdigest()
    if digest != manifest.get("trial_features_sha256"):
        raise ValueError("Trial CSV SHA-256 does not match the feature extraction manifest")
    rows = pd.read_csv(path, dtype={"participant_id": str})
    if len(rows) != 384 or not {"difficulty_level", "recorded_difficulty_code"} <= set(rows):
        raise ValueError("FE-04 export is missing the trial count or difficulty-code columns")
    if TARGET in rows:
        raise ValueError("FE-04 must not contain a conflicting difficulty stage column")
    return rows.rename(columns={"difficulty_level": TARGET}), digest


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trial-features", type=Path, required=True, help="FE-04 trial_features.csv")
    parser.add_argument("--feature-manifest", type=Path, required=True, help="Matching FE-04 manifest")
    parser.add_argument("--participant-split", type=Path, required=True, help="Frozen MOD-09 split CSV")
    parser.add_argument("--output-dir", type=Path, required=True, help="New local output directory")
    args = parser.parse_args(argv)
    if args.output_dir.exists():
        raise FileExistsError(f"Comparison output already exists: {args.output_dir}")
    trials, feature_hash = _read_fe04(args.trial_features, args.feature_manifest)
    split = pd.read_csv(args.participant_split, dtype={"participant_id": str})
    development = prepare_development(trials, split)
    results = [
        compare_condition(development.loc[development.condition_name.eq(condition)], condition)
        for condition in ADD_BACK
    ]
    outputs = {
        "comparison_summary.csv": "summary",
        "cross_validation_rounds.csv": "rounds",
        "development_predictions.csv": "predictions",
        "mixed_model_screen.csv": "screen",
        "final_coefficients.csv": "coefficients",
    }
    provenance = {
        "evaluation": "development_cross_validation",
        "source_schema": "fe04_all_grabs_v1",
        "source_feature_sha256": feature_hash,
        "source_feature_manifest_sha256": sha256(args.feature_manifest.read_bytes()).hexdigest(),
        "participant_split_sha256": sha256(args.participant_split.read_bytes()).hexdigest(),
        "module_sha256": sha256(Path(__file__).read_bytes()).hexdigest(),
        "development_participants": int(development.participant_id.nunique()),
        "reserved_participants_evaluated": 0,
        "cross_validation_rounds": MOD15.CV_N_SPLITS,
        "p_value_cutoff_exclusive": P_CUTOFF,
        "mae_margin_stage_units_inclusive": MAE_MARGIN,
        "add_back": ADD_BACK,
        "predictors": TRIAL_FEATURES,
        "statsmodels_version": statsmodels.__version__,
        "scikit_learn_version": sklearn.__version__,
        "missing_data": "same complete cases for fitting and scoring all candidates",
    }
    args.output_dir.mkdir(parents=True)
    for filename, name in outputs.items():
        pd.concat([getattr(result, name) for result in results], ignore_index=True).to_csv(
            args.output_dir / filename, index=False,
        )
    (args.output_dir / "comparison_manifest.json").write_text(
        json.dumps(provenance, indent=2) + "\n", encoding="utf-8",
    )


if __name__ == "__main__":
    main()
