#!/usr/bin/env python3
"""MOD-11 — Training-only participant bootstrap model comparison.

MOD-11 compares seven frozen fixed-effects candidate models using only the
26 participants assigned to the MOD-09 training set.

The six final test participants remain completely excluded from fitting,
bootstrap validation, diagnostics, tuning, and model-selection decisions.

This module currently implements the MOD-11 bootstrap foundation. Mixed-model
fitting, OOB prediction, metric aggregation, and output generation are added in
subsequent implementation stages.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path

import MOD_10_training_random_effects_comparison as MOD10


SCRIPT_VERSION = "1.1.0"

EXPECTED_TRAINING_PARTICIPANTS = 26
EXPECTED_TEST_PARTICIPANTS = 6

DEFAULT_BOOTSTRAP_REPLICATES = 500

RANDOM_STRUCTURES = ("RI",)

MODEL_IDS = (
    "M0",
    "MA",
    "MT",
    "MAT",
    "MDA",
    "MDT",
    "MDAT",
)

PARTICIPANT_COLUMN = "participant_id"
BOOTSTRAP_CLUSTER_COLUMN = "bootstrap_cluster_id"
BOOTSTRAP_DRAW_COLUMN = "bootstrap_draw_index"
BOOTSTRAP_REPLICATE_COLUMN = "bootstrap_replicate"

AGE_TERM = (
    "C(participant_group, Treatment(reference='Young'))"
)

MOD02 = MOD10.MOD02

CONDITIONS = tuple(MOD10.CONDITIONS)
TARGETS = tuple(MOD10.TARGETS)

ORIGINAL_ROW_ID_COLUMN = "mod11_original_row_id"

CHECKPOINT_SCHEMA_VERSION = 1
DEFAULT_AUTO_MAX_JOBS = 4

COMPARISON_FILENAME = "mod11_model_comparison.csv"
ORIGINAL_FITS_FILENAME = "mod11_original_training_fits.csv"
BOOTSTRAP_FITS_FILENAME = "mod11_bootstrap_fits.csv.gz"
PREDICTIONS_FILENAME = "mod11_bootstrap_predictions.csv.gz"
ELIGIBILITY_FILENAME = "mod11_bootstrap_eligibility.csv"
AUDIT_FILENAME = "mod11_bootstrap_fit_audit.csv"
MANIFEST_FILENAME = "mod11_manifest.json"


class Mod11Error(ValueError):
    """Raised when a frozen MOD-11 data contract is violated."""


def build_candidate_model_rhs_registry() -> dict[str, str]:
    """Return the seven frozen MOD-11 fixed-effects candidates.

    ``{D}`` is replaced later by the appropriate difficulty variable for the
    target being modelled.

    TMT-B always enters in its original unit through ``tmt_b_seconds``.
    """
    return {
        "M0": "{D}",
        "MA": f"{{D}} + {AGE_TERM}",
        "MT": "{D} + tmt_b_seconds",
        "MAT": f"{{D}} + {AGE_TERM} + tmt_b_seconds",
        "MDA": (
            f"{{D}} + {AGE_TERM} + "
            f"{{D}}:{AGE_TERM}"
        ),
        "MDT": (
            "{D} + tmt_b_seconds + "
            "{D}:tmt_b_seconds"
        ),
        "MDAT": (
            f"{{D}} + {AGE_TERM} + tmt_b_seconds + "
            f"{{D}}:{AGE_TERM} + "
            "{D}:tmt_b_seconds"
        ),
    }


def prepare_training_data(
    *,
    modeling_data: pd.DataFrame,
    holdout_split: pd.DataFrame,
) -> tuple[
    pd.DataFrame,
    tuple[str, ...],
    tuple[str, ...],
]:
    """Return only the locked 26 MOD-09 training participants.

    MOD-10 already implements and tests the frozen MOD-09 split contract.
    MOD-11 reuses that boundary rather than maintaining a second independent
    implementation.
    """
    training_data, training_ids, test_ids = (
        MOD10.prepare_training_data(
            modeling_data=modeling_data,
            holdout_split=holdout_split,
        )
    )

    if len(training_ids) != EXPECTED_TRAINING_PARTICIPANTS:
        raise Mod11Error(
            "MOD-11 requires exactly 26 training participants"
        )

    if len(test_ids) != EXPECTED_TEST_PARTICIPANTS:
        raise Mod11Error(
            "MOD-11 requires exactly 6 final test participants"
        )

    observed_training = set(
        training_data[PARTICIPANT_COLUMN].astype(str)
    )

    if not observed_training.isdisjoint(test_ids):
        raise Mod11Error(
            "Final MOD-09 test participants entered MOD-11"
        )

    return training_data, training_ids, test_ids


def _validate_training_participant_ids(
    participant_ids: Sequence[str],
) -> tuple[str, ...]:
    """Validate and normalise the frozen training participant IDs."""
    ids = tuple(str(value).strip() for value in participant_ids)

    if len(ids) != EXPECTED_TRAINING_PARTICIPANTS:
        raise Mod11Error(
            "MOD-11 bootstrap requires exactly "
            "26 training participant IDs"
        )

    if any(not value for value in ids):
        raise Mod11Error(
            "Training participant IDs must not be blank"
        )

    if len(set(ids)) != EXPECTED_TRAINING_PARTICIPANTS:
        raise Mod11Error(
            "Training participant IDs must be unique"
        )

    return ids


def draw_participant_ids(
    training_participant_ids: Sequence[str],
    *,
    rng: np.random.Generator,
) -> tuple[str, ...]:
    """Draw 26 training participants with replacement.

    The output contains one entry per bootstrap draw. Therefore the same
    original participant may appear multiple times.
    """
    ids = _validate_training_participant_ids(
        training_participant_ids
    )

    sampled = rng.choice(
        np.asarray(ids, dtype=object),
        size=EXPECTED_TRAINING_PARTICIPANTS,
        replace=True,
    )

    return tuple(str(value) for value in sampled.tolist())


def generate_bootstrap_plan(
    training_participant_ids: Sequence[str],
    *,
    n_replicates: int,
    seed: int,
) -> tuple[tuple[str, ...], ...]:
    """Generate one reproducible participant bootstrap plan.

    The resulting plan should be generated once for an analysis and reused
    across all seven candidate models so model comparisons are based on the
    same participant resamples.

    The seed is supplied explicitly rather than silently fixed here because the
    official MOD-11 bootstrap seed has not yet been frozen as a thesis decision.
    """
    ids = _validate_training_participant_ids(
        training_participant_ids
    )

    if isinstance(n_replicates, bool):
        raise Mod11Error(
            "n_replicates must be a positive integer"
        )

    try:
        replicate_count = int(n_replicates)
    except (TypeError, ValueError) as exc:
        raise Mod11Error(
            "n_replicates must be a positive integer"
        ) from exc

    if replicate_count != n_replicates or replicate_count < 1:
        raise Mod11Error(
            "n_replicates must be a positive integer"
        )

    rng = np.random.default_rng(seed)

    return tuple(
        draw_participant_ids(
            ids,
            rng=rng,
        )
        for _ in range(replicate_count)
    )


def identify_oob_participants(
    training_participant_ids: Sequence[str],
    sampled_participant_ids: Sequence[str],
) -> tuple[str, ...]:
    """Return original training participants absent from one bootstrap sample."""
    training_ids = _validate_training_participant_ids(
        training_participant_ids
    )

    sampled = tuple(
        str(value).strip()
        for value in sampled_participant_ids
    )

    if len(sampled) != EXPECTED_TRAINING_PARTICIPANTS:
        raise Mod11Error(
            "A MOD-11 bootstrap replicate must contain "
            "exactly 26 participant draws"
        )

    unexpected = sorted(
        set(sampled) - set(training_ids)
    )

    if unexpected:
        raise Mod11Error(
            "Bootstrap sample contains IDs outside the "
            "locked training set: "
            + ", ".join(unexpected)
        )

    sampled_unique = set(sampled)

    return tuple(
        participant_id
        for participant_id in training_ids
        if participant_id not in sampled_unique
    )


def build_bootstrap_sample(
    training_data: pd.DataFrame,
    sampled_participant_ids: Sequence[str],
    *,
    replicate_index: int,
) -> pd.DataFrame:
    """Materialise one participant-cluster bootstrap training sample.

    All repeated rows belonging to a sampled participant are copied together.

    If an original participant is drawn multiple times, each draw receives a
    different ``bootstrap_cluster_id``. The original participant ID remains in
    ``participant_id`` for traceability, while the bootstrap cluster ID is the
    grouping variable that will later be supplied to the mixed model.
    """
    if PARTICIPANT_COLUMN not in training_data.columns:
        raise Mod11Error(
            f"Missing required column: {PARTICIPANT_COLUMN}"
        )

    sampled = tuple(
        str(value).strip()
        for value in sampled_participant_ids
    )

    if len(sampled) != EXPECTED_TRAINING_PARTICIPANTS:
        raise Mod11Error(
            "A MOD-11 bootstrap replicate must contain "
            "exactly 26 participant draws"
        )

    available_ids = set(
        training_data[PARTICIPANT_COLUMN]
        .astype(str)
        .str.strip()
    )

    unexpected = sorted(
        set(sampled) - available_ids
    )

    if unexpected:
        raise Mod11Error(
            "Bootstrap sample contains participants not present "
            "in training_data: "
            + ", ".join(unexpected)
        )

    copies: list[pd.DataFrame] = []

    participant_values = (
        training_data[PARTICIPANT_COLUMN]
        .astype(str)
        .str.strip()
    )

    for draw_index, participant_id in enumerate(
        sampled,
        start=1,
    ):
        participant_rows = training_data.loc[
            participant_values.eq(participant_id)
        ].copy()

        if participant_rows.empty:
            raise Mod11Error(
                "No repeated rows found for sampled participant "
                f"{participant_id}"
            )

        cluster_id = (
            f"b{int(replicate_index):04d}"
            f"_d{draw_index:02d}"
            f"_{participant_id}"
        )

        participant_rows[
            BOOTSTRAP_CLUSTER_COLUMN
        ] = cluster_id

        participant_rows[
            BOOTSTRAP_DRAW_COLUMN
        ] = draw_index

        participant_rows[
            BOOTSTRAP_REPLICATE_COLUMN
        ] = int(replicate_index)

        copies.append(participant_rows)

    bootstrap = pd.concat(
        copies,
        axis=0,
        ignore_index=True,
    )

    if (
        bootstrap[BOOTSTRAP_CLUSTER_COLUMN].nunique()
        != EXPECTED_TRAINING_PARTICIPANTS
    ):
        raise Mod11Error(
            "Every bootstrap draw must receive a distinct "
            "bootstrap cluster ID"
        )

    return bootstrap

def build_target_specs():
    """Return the frozen MOD-11 Gaussian target specifications."""
    return MOD10.build_target_specs()


def _ensure_original_row_ids(
    data: pd.DataFrame,
) -> pd.DataFrame:
    """Ensure each original training row has a stable internal identifier."""
    out = data.copy(deep=True)

    if ORIGINAL_ROW_ID_COLUMN not in out.columns:
        out[ORIGINAL_ROW_ID_COLUMN] = np.arange(
            len(out),
            dtype=int,
        )

    if out[ORIGINAL_ROW_ID_COLUMN].isna().any():
        raise Mod11Error(
            f"{ORIGINAL_ROW_ID_COLUMN} contains missing values"
        )

    return out


def prepare_target_frame(
    data: pd.DataFrame,
    *,
    condition: str,
    target: str,
) -> tuple[pd.DataFrame, str]:
    """Prepare identical eligible rows for all seven candidate models.

    Relative performance excludes D0 and uses
    ``performance_difficulty_stage``.

    All other Gaussian outcomes use ``difficulty_stage``.

    Age and raw TMT-B are required even for simpler models so all seven
    candidates for the same condition/outcome are compared on identical rows.
    """
    specs = build_target_specs()

    if target not in specs:
        raise Mod11Error(
            f"Unknown MOD-11 target: {target}"
        )

    spec = specs[target]

    prepared = _ensure_original_row_ids(data)
    prepared = MOD02.prepare_internal_modeling_data(
        prepared
    )

    frame = prepared.loc[
        prepared["condition_name"].eq(condition)
    ].copy()

    if target == (
        "performance_change_from_d0_percentage_points"
    ):
        frame = frame.loc[
            ~frame["difficulty_level"].eq(0)
        ].copy()

    required = [
        PARTICIPANT_COLUMN,
        "participant_group",
        "tmt_b_seconds",
        "difficulty_level",
        spec.difficulty_column,
        target,
        ORIGINAL_ROW_ID_COLUMN,
    ]

    missing = [
        column
        for column in required
        if column not in frame.columns
    ]

    if missing:
        raise Mod11Error(
            f"{condition}/{target}: "
            f"missing required columns {missing}"
        )

    frame["tmt_b_seconds"] = pd.to_numeric(
        frame["tmt_b_seconds"],
        errors="coerce",
    )

    frame[spec.difficulty_column] = pd.to_numeric(
        frame[spec.difficulty_column],
        errors="coerce",
    )

    frame[target] = pd.to_numeric(
        frame[target],
        errors="coerce",
    )

    finite_mask = (
        np.isfinite(frame["tmt_b_seconds"])
        & np.isfinite(frame[spec.difficulty_column])
        & np.isfinite(frame[target])
    )

    complete_mask = (
        frame[required]
        .notna()
        .all(axis=1)
    )

    frame = frame.loc[
        finite_mask & complete_mask
    ].copy()

    if frame.empty:
        raise Mod11Error(
            f"{condition}/{target}: no eligible rows"
        )

    if not set(
        frame["participant_group"].astype(str)
    ).issubset({"Young", "Old"}):
        raise Mod11Error(
            f"{condition}/{target}: unexpected age group"
        )

    return frame, spec.difficulty_column


def build_candidate_formula(
    *,
    target: str,
    model_id: str,
) -> str:
    """Build one frozen target-specific fixed-effects formula."""
    registry = build_candidate_model_rhs_registry()
    specs = build_target_specs()

    if model_id not in registry:
        raise Mod11Error(
            f"Unknown MOD-11 model ID: {model_id}"
        )

    if target not in specs:
        raise Mod11Error(
            f"Unknown MOD-11 target: {target}"
        )

    difficulty_column = specs[target].difficulty_column

    rhs = registry[model_id].format(
        D=difficulty_column
    )

    return f"{target} ~ {rhs}"


def _safe_float(value) -> float:
    """Return a finite float or NaN."""
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return np.nan

    return (
        numeric
        if np.isfinite(numeric)
        else np.nan
    )


def fit_ri_candidate(
    frame: pd.DataFrame,
    *,
    target: str,
    model_id: str,
    participant_column: str,
) -> tuple[object | None, dict[str, object]]:
    """Fit one frozen candidate under the MOD-10 RI structure.

    The existing MOD-02 fitter performs both L-BFGS and Powell attempts and
    returns the converged fit with the highest finite log likelihood.
    """
    specs = build_target_specs()

    if target not in specs:
        raise Mod11Error(
            f"Unknown target: {target}"
        )

    if participant_column not in frame.columns:
        raise Mod11Error(
            f"Missing grouping column: {participant_column}"
        )

    difficulty_column = specs[target].difficulty_column

    formula = build_candidate_formula(
        target=target,
        model_id=model_id,
    )

    result, optimizer, fit_warnings, fit_errors = (
        MOD02.fit_mixedlm_with_fallback(
            formula,
            frame,
            "RI",
            difficulty_column,
            participant_column=participant_column,
        )
    )

    llf = (
        _safe_float(getattr(result, "llf", np.nan))
        if result is not None
        else np.nan
    )

    converged = bool(
        result is not None
        and getattr(result, "converged", False)
        and np.isfinite(llf)
    )

    evidence: dict[str, object] = {
        "target_name": target,
        "model_id": model_id,
        "fixed_effects_formula": formula,
        "difficulty_source_column": difficulty_column,
        "random_structure": "RI",
        "estimation_method": "maximum_likelihood",
        "reml": False,
        "participant_group_column": participant_column,
        "participant_count": int(
            frame[participant_column].nunique()
        ),
        "observation_count": int(len(frame)),
        "convergence_status": (
            "converged"
            if converged
            else (
                "non_converged"
                if result is not None
                else "failed"
            )
        ),
        "optimizer": optimizer,
        "log_likelihood": llf,
        "aic": (
            _safe_float(getattr(result, "aic", np.nan))
            if result is not None
            else np.nan
        ),
        "bic": (
            _safe_float(getattr(result, "bic", np.nan))
            if result is not None
            else np.nan
        ),
        "fixed_effect_count": (
            int(len(result.fe_params))
            if (
                result is not None
                and hasattr(result, "fe_params")
            )
            else np.nan
        ),
        "warnings": " | ".join(
            str(value)
            for value in fit_warnings
        ),
        "fit_errors": " | ".join(
            str(value)
            for value in fit_errors
        ),
    }

    return result, evidence


def fit_original_training_models(
    training_data: pd.DataFrame,
    *,
    conditions: Sequence[str] = CONDITIONS,
    targets: Sequence[str] = TARGETS,
    model_ids: Sequence[str] = MODEL_IDS,
) -> pd.DataFrame:
    """Fit all requested candidates once to the original training data.

    These fits provide the MOD-11 AIC and BIC values. Bootstrap fits are not
    used to calculate AIC or BIC for the final candidate comparison.
    """
    data = _ensure_original_row_ids(
        training_data
    )

    rows: list[dict[str, object]] = []

    for condition in conditions:
        for target in targets:
            frame, _ = prepare_target_frame(
                data,
                condition=condition,
                target=target,
            )

            for model_id in model_ids:
                _, evidence = fit_ri_candidate(
                    frame,
                    target=target,
                    model_id=model_id,
                    participant_column=PARTICIPANT_COLUMN,
                )

                evidence.update(
                    {
                        "condition_name": condition,
                        "fit_scope": (
                            "original_26_training_participants"
                        ),
                        "information_criterion_source": (
                            "original_training_fit"
                        ),
                    }
                )

                rows.append(evidence)

    return pd.DataFrame(rows)


def predict_fixed_effects(
    result,
    frame: pd.DataFrame,
) -> np.ndarray:
    """Predict new participants using population/fixed effects only.

    ``statsmodels`` formula MixedLM ``result.predict(frame)`` evaluates the
    fixed-effects mean. MOD-11 does not add participant random effects for OOB
    participants.
    """
    predicted = np.asarray(
        result.predict(frame),
        dtype=float,
    )

    if len(predicted) != len(frame):
        raise Mod11Error(
            "OOB prediction length mismatch"
        )

    if not np.isfinite(predicted).all():
        raise Mod11Error(
            "OOB prediction contains non-finite values"
        )

    return predicted


def run_bootstrap_replicate(
    training_data: pd.DataFrame,
    *,
    training_participant_ids: Sequence[str],
    sampled_participant_ids: Sequence[str],
    replicate_index: int,
    conditions: Sequence[str] = CONDITIONS,
    targets: Sequence[str] = TARGETS,
    model_ids: Sequence[str] = MODEL_IDS,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fit and predict all requested candidates for one bootstrap replicate.

    Returns
    -------
    predictions:
        One row per OOB observation and successfully fitted candidate.

    fits:
        One row per attempted bootstrap candidate fit, including convergence,
        optimizer, warning and failure evidence.
    """
    data = _ensure_original_row_ids(
        training_data
    )

    training_ids = _validate_training_participant_ids(
        training_participant_ids
    )

    sampled = tuple(
        str(value).strip()
        for value in sampled_participant_ids
    )

    oob_ids = identify_oob_participants(
        training_ids,
        sampled,
    )

    bootstrap_data = build_bootstrap_sample(
        data,
        sampled,
        replicate_index=replicate_index,
    )

    oob_data = data.loc[
        data[PARTICIPANT_COLUMN]
        .astype(str)
        .isin(oob_ids)
    ].copy()

    prediction_rows: list[
        dict[str, object]
    ] = []

    fit_rows: list[
        dict[str, object]
    ] = []

    if not oob_ids:
        return (
            pd.DataFrame(prediction_rows),
            pd.DataFrame(fit_rows),
        )

    for condition in conditions:
        for target in targets:
            bootstrap_frame, bootstrap_difficulty = (
                prepare_target_frame(
                    bootstrap_data,
                    condition=condition,
                    target=target,
                )
            )

            oob_frame, oob_difficulty = (
                prepare_target_frame(
                    oob_data,
                    condition=condition,
                    target=target,
                )
            )

            if bootstrap_difficulty != oob_difficulty:
                raise Mod11Error(
                    f"{condition}/{target}: "
                    "bootstrap/OOB difficulty mismatch"
                )

            for model_id in model_ids:
                result, evidence = fit_ri_candidate(
                    bootstrap_frame,
                    target=target,
                    model_id=model_id,
                    participant_column=(
                        BOOTSTRAP_CLUSTER_COLUMN
                    ),
                )

                evidence.update(
                    {
                        "bootstrap_replicate": int(
                            replicate_index
                        ),
                        "condition_name": condition,
                        "fit_scope": (
                            "bootstrap_training_sample"
                        ),
                        "oob_participant_count": int(
                            oob_frame[
                                PARTICIPANT_COLUMN
                            ].nunique()
                        ),
                        "oob_observation_count": int(
                            len(oob_frame)
                        ),
                        "prediction_scope": (
                            "fixed_effect_population_only"
                        ),
                    }
                )

                if (
                    evidence["convergence_status"]
                    != "converged"
                ):
                    evidence["prediction_status"] = (
                        "not_attempted_fit_failure"
                    )

                    fit_rows.append(evidence)
                    continue

                try:
                    predicted = predict_fixed_effects(
                        result,
                        oob_frame,
                    )

                    observed = pd.to_numeric(
                        oob_frame[target],
                        errors="coerce",
                    ).to_numpy(dtype=float)

                    residual = (
                        observed - predicted
                    )

                    evidence["prediction_status"] = (
                        "success"
                    )

                    for local_index, (
                        _,
                        observation,
                    ) in enumerate(
                        oob_frame.iterrows()
                    ):
                        prediction_rows.append(
                            {
                                "bootstrap_replicate": int(
                                    replicate_index
                                ),
                                "condition_name": condition,
                                "target_name": target,
                                "model_id": model_id,
                                "participant_id": str(
                                    observation[
                                        PARTICIPANT_COLUMN
                                    ]
                                ),
                                ORIGINAL_ROW_ID_COLUMN: (
                                    observation[
                                        ORIGINAL_ROW_ID_COLUMN
                                    ]
                                ),
                                "difficulty_level": (
                                    observation.get(
                                        "difficulty_level",
                                        np.nan,
                                    )
                                ),
                                "difficulty_source_column": (
                                    bootstrap_difficulty
                                ),
                                "observed": float(
                                    observed[local_index]
                                ),
                                "predicted": float(
                                    predicted[local_index]
                                ),
                                "residual": float(
                                    residual[local_index]
                                ),
                                "absolute_error": float(
                                    abs(
                                        residual[
                                            local_index
                                        ]
                                    )
                                ),
                                "squared_error": float(
                                    residual[
                                        local_index
                                    ]
                                    ** 2
                                ),
                                "prediction_scope": (
                                    "fixed_effect_population_only"
                                ),
                            }
                        )

                except Exception as exc:
                    evidence["prediction_status"] = (
                        "failed"
                    )

                    existing_error = str(
                        evidence["fit_errors"]
                    ).strip()

                    prediction_error = (
                        f"{type(exc).__name__}: {exc}"
                    )

                    evidence["fit_errors"] = (
                        f"{existing_error} | "
                        f"{prediction_error}"
                        if existing_error
                        else prediction_error
                    )

                fit_rows.append(evidence)

    return (
        pd.DataFrame(prediction_rows),
        pd.DataFrame(fit_rows),
    )

@dataclass(frozen=True)
class Mod11Result:
    """Complete MOD-11 evidence package."""

    original_fits: pd.DataFrame
    bootstrap_predictions: pd.DataFrame
    bootstrap_fits: pd.DataFrame
    eligibility: pd.DataFrame
    comparison: pd.DataFrame
    fit_audit: pd.DataFrame


def build_comparison_eligibility(
    bootstrap_fits: pd.DataFrame,
    predictions: pd.DataFrame,
    *,
    model_ids: Sequence[str] = MODEL_IDS,
) -> pd.DataFrame:
    """Identify bootstrap replicates comparable across all candidates.

    A condition × target × replicate is comparison eligible only when every
    requested candidate converged and produced the same complete OOB row set.

    This prevents candidate deltas from comparing different bootstrap samples.
    """
    required_fit_columns = {
        "bootstrap_replicate",
        "condition_name",
        "target_name",
        "model_id",
        "convergence_status",
        "prediction_status",
    }

    missing = required_fit_columns - set(
        bootstrap_fits.columns
    )

    if missing:
        raise Mod11Error(
            "Bootstrap fit evidence is missing columns: "
            + ", ".join(sorted(missing))
        )

    expected_models = tuple(model_ids)
    expected_set = set(expected_models)

    rows: list[dict[str, object]] = []

    keys = [
        "condition_name",
        "target_name",
        "bootstrap_replicate",
    ]

    for key, family in bootstrap_fits.groupby(
        keys,
        sort=False,
        dropna=False,
    ):
        condition, target, replicate = key

        observed_models = list(
            family["model_id"].astype(str)
        )

        unique_models = set(observed_models)

        one_fit_per_model = (
            len(family) == len(expected_models)
            and len(unique_models) == len(expected_models)
            and unique_models == expected_set
        )

        all_fit_success = bool(
            one_fit_per_model
            and family[
                "convergence_status"
            ].eq("converged").all()
            and family[
                "prediction_status"
            ].eq("success").all()
        )

        family_predictions = predictions.loc[
            predictions["condition_name"].eq(condition)
            & predictions["target_name"].eq(target)
            & predictions["bootstrap_replicate"].eq(replicate)
        ].copy()

        row_sets: list[set[object]] = []

        prediction_sets_complete = all_fit_success

        if prediction_sets_complete:
            for model_id in expected_models:
                model_predictions = family_predictions.loc[
                    family_predictions["model_id"].eq(model_id)
                ]

                row_ids = set(
                    model_predictions[
                        ORIGINAL_ROW_ID_COLUMN
                    ].tolist()
                )

                if not row_ids:
                    prediction_sets_complete = False
                    break

                if (
                    len(row_ids)
                    != len(model_predictions)
                ):
                    prediction_sets_complete = False
                    break

                row_sets.append(row_ids)

        same_prediction_rows = bool(
            prediction_sets_complete
            and row_sets
            and all(
                row_set == row_sets[0]
                for row_set in row_sets[1:]
            )
        )

        eligible = bool(
            all_fit_success
            and same_prediction_rows
        )

        if eligible:
            reason = "all_models_successful_same_oob_rows"
        elif not one_fit_per_model:
            reason = "missing_or_duplicate_model_fit"
        elif not all_fit_success:
            reason = "one_or_more_model_fit_or_prediction_failures"
        else:
            reason = "oob_prediction_row_mismatch"

        rows.append(
            {
                "condition_name": condition,
                "target_name": target,
                "bootstrap_replicate": int(replicate),
                "expected_model_count": len(
                    expected_models
                ),
                "successful_model_count": int(
                    (
                        family[
                            "convergence_status"
                        ].eq("converged")
                        & family[
                            "prediction_status"
                        ].eq("success")
                    ).sum()
                ),
                "comparison_eligible": eligible,
                "eligibility_reason": reason,
                "common_oob_observation_count": (
                    len(row_sets[0])
                    if eligible
                    else 0
                ),
            }
        )

    return pd.DataFrame(rows)


def filter_to_comparison_eligible_predictions(
    predictions: pd.DataFrame,
    eligibility: pd.DataFrame,
) -> pd.DataFrame:
    """Return predictions from common-support bootstrap replicates only."""
    if predictions.empty:
        return predictions.copy()

    eligible_keys = eligibility.loc[
        eligibility["comparison_eligible"].eq(True),
        [
            "condition_name",
            "target_name",
            "bootstrap_replicate",
        ],
    ].copy()

    if eligible_keys.empty:
        return predictions.iloc[0:0].copy()

    return predictions.merge(
        eligible_keys,
        on=[
            "condition_name",
            "target_name",
            "bootstrap_replicate",
        ],
        how="inner",
        validate="many_to_one",
    )


def participant_balanced_loss_first_mae(
    predictions: pd.DataFrame,
    *,
    expected_participants: Sequence[str],
) -> float:
    """Calculate frozen loss-first participant-balanced OOB MAE.

    1. Average row absolute errors within participant × OOB replicate.
    2. Average those replicate errors within each participant.
    3. Average participant-specific errors equally.
    """
    if predictions.empty:
        return np.nan

    required = {
        "participant_id",
        "bootstrap_replicate",
        "absolute_error",
    }

    if not required.issubset(predictions.columns):
        raise Mod11Error(
            "Predictions do not contain the columns required for OOB MAE"
        )

    frame = predictions.copy()

    frame["absolute_error"] = pd.to_numeric(
        frame["absolute_error"],
        errors="coerce",
    )

    frame = frame.loc[
        np.isfinite(frame["absolute_error"])
    ].copy()

    participant_replicate = (
        frame.groupby(
            [
                "participant_id",
                "bootstrap_replicate",
            ],
            sort=False,
        )["absolute_error"]
        .mean()
    )

    participant_mae = (
        participant_replicate
        .groupby(level="participant_id")
        .mean()
    )

    expected = set(
        str(value)
        for value in expected_participants
    )

    observed = set(
        participant_mae.index.astype(str)
    )

    if observed != expected:
        return np.nan

    return float(
        participant_mae.mean()
    )


def pooled_loss_first_oob_r2(
    predictions: pd.DataFrame,
    *,
    original_frame: pd.DataFrame,
    target: str,
) -> float:
    """Calculate pooled OOB R² using mean squared OOB loss per original row."""
    if predictions.empty:
        return np.nan

    required_prediction_columns = {
        ORIGINAL_ROW_ID_COLUMN,
        "squared_error",
    }

    if not required_prediction_columns.issubset(
        predictions.columns
    ):
        raise Mod11Error(
            "Predictions do not contain the columns required for OOB R²"
        )

    if ORIGINAL_ROW_ID_COLUMN not in original_frame:
        raise Mod11Error(
            f"{ORIGINAL_ROW_ID_COLUMN} missing from original frame"
        )

    if target not in original_frame:
        raise Mod11Error(
            f"Target missing from original frame: {target}"
        )

    if original_frame[
        ORIGINAL_ROW_ID_COLUMN
    ].duplicated().any():
        raise Mod11Error(
            "Original row IDs must be unique for pooled OOB R²"
        )

    frame = predictions.copy()

    frame["squared_error"] = pd.to_numeric(
        frame["squared_error"],
        errors="coerce",
    )

    frame = frame.loc[
        np.isfinite(frame["squared_error"])
    ].copy()

    mean_squared_error_by_row = (
        frame.groupby(
            ORIGINAL_ROW_ID_COLUMN,
            sort=False,
        )["squared_error"]
        .mean()
    )

    expected_row_ids = set(
        original_frame[
            ORIGINAL_ROW_ID_COLUMN
        ].tolist()
    )

    observed_row_ids = set(
        mean_squared_error_by_row.index.tolist()
    )

    if observed_row_ids != expected_row_ids:
        return np.nan

    ordered_squared_error = (
        mean_squared_error_by_row.reindex(
            original_frame[
                ORIGINAL_ROW_ID_COLUMN
            ].tolist()
        )
    )

    sse = float(
        ordered_squared_error.sum()
    )

    observed = pd.to_numeric(
        original_frame[target],
        errors="coerce",
    ).to_numpy(dtype=float)

    if not np.isfinite(observed).all():
        return np.nan

    mean_observed = float(
        np.mean(observed)
    )

    sst = float(
        np.sum(
            (observed - mean_observed) ** 2
        )
    )

    if sst <= 0:
        return np.nan

    return float(
        1.0 - sse / sst
    )


def aggregate_oob_metrics(
    training_data: pd.DataFrame,
    bootstrap_predictions: pd.DataFrame,
    bootstrap_fits: pd.DataFrame,
    *,
    conditions: Sequence[str] = CONDITIONS,
    targets: Sequence[str] = TARGETS,
    model_ids: Sequence[str] = MODEL_IDS,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Aggregate primary predictive metrics using common bootstrap support."""
    eligibility = build_comparison_eligibility(
        bootstrap_fits,
        bootstrap_predictions,
        model_ids=model_ids,
    )

    predictions = (
        filter_to_comparison_eligible_predictions(
            bootstrap_predictions,
            eligibility,
        )
    )

    rows: list[dict[str, object]] = []

    for condition in conditions:
        for target in targets:
            original_frame, _ = prepare_target_frame(
                training_data,
                condition=condition,
                target=target,
            )

            expected_participants = tuple(
                sorted(
                    original_frame[
                        PARTICIPANT_COLUMN
                    ]
                    .astype(str)
                    .unique()
                )
            )

            expected_rows = set(
                original_frame[
                    ORIGINAL_ROW_ID_COLUMN
                ].tolist()
            )

            family_eligibility = eligibility.loc[
                eligibility[
                    "condition_name"
                ].eq(condition)
                & eligibility[
                    "target_name"
                ].eq(target)
            ]

            total_replicates = int(
                len(family_eligibility)
            )

            eligible_replicates = int(
                family_eligibility[
                    "comparison_eligible"
                ].sum()
            )

            for model_id in model_ids:
                subset = predictions.loc[
                    predictions[
                        "condition_name"
                    ].eq(condition)
                    & predictions[
                        "target_name"
                    ].eq(target)
                    & predictions[
                        "model_id"
                    ].eq(model_id)
                ].copy()

                participants_with_oob = set(
                    subset[
                        "participant_id"
                    ].astype(str)
                ) if not subset.empty else set()

                rows_with_oob = set(
                    subset[
                        ORIGINAL_ROW_ID_COLUMN
                    ].tolist()
                ) if not subset.empty else set()

                participant_coverage_complete = (
                    participants_with_oob
                    == set(expected_participants)
                )

                observation_coverage_complete = (
                    rows_with_oob
                    == expected_rows
                )

                mae = (
                    participant_balanced_loss_first_mae(
                        subset,
                        expected_participants=(
                            expected_participants
                        ),
                    )
                )

                r2 = pooled_loss_first_oob_r2(
                    subset,
                    original_frame=original_frame,
                    target=target,
                )

                metrics_complete = bool(
                    eligible_replicates > 0
                    and participant_coverage_complete
                    and observation_coverage_complete
                    and np.isfinite(mae)
                    and np.isfinite(r2)
                )

                rows.append(
                    {
                        "condition_name": condition,
                        "target_name": target,
                        "model_id": model_id,
                        "participant_balanced_oob_mae": mae,
                        "pooled_oob_r2": r2,
                        "requested_or_observed_replicate_count": (
                            total_replicates
                        ),
                        "comparison_eligible_replicate_count": (
                            eligible_replicates
                        ),
                        "excluded_replicate_count": (
                            total_replicates
                            - eligible_replicates
                        ),
                        "participants_with_oob_predictions": (
                            len(
                                participants_with_oob
                            )
                        ),
                        "expected_participant_count": (
                            len(
                                expected_participants
                            )
                        ),
                        "observations_with_oob_predictions": (
                            len(rows_with_oob)
                        ),
                        "expected_observation_count": (
                            len(expected_rows)
                        ),
                        "participant_coverage_complete": (
                            participant_coverage_complete
                        ),
                        "observation_coverage_complete": (
                            observation_coverage_complete
                        ),
                        "metric_status": (
                            "complete"
                            if metrics_complete
                            else "incomplete"
                        ),
                        "oob_aggregation_method": (
                            "loss_first_common_support"
                        ),
                        "comparison_support_rule": (
                            "all_seven_models_successful_"
                            "same_oob_rows"
                        ),
                    }
                )

    return (
        pd.DataFrame(rows),
        eligibility,
    )


def combine_comparison_evidence(
    oob_summary: pd.DataFrame,
    original_fits: pd.DataFrame,
) -> pd.DataFrame:
    """Combine predictive metrics with original-fit AIC/BIC and M0 deltas."""
    keys = [
        "condition_name",
        "target_name",
        "model_id",
    ]

    original_columns = keys + [
        "aic",
        "bic",
        "log_likelihood",
        "convergence_status",
        "optimizer",
        "warnings",
        "fit_errors",
    ]

    available_columns = [
        column
        for column in original_columns
        if column in original_fits.columns
    ]

    combined = oob_summary.merge(
        original_fits[available_columns],
        on=keys,
        how="left",
        validate="one_to_one",
    )

    combined["delta_mae_vs_m0"] = np.nan
    combined["delta_oob_r2_vs_m0"] = np.nan
    combined["delta_aic_vs_m0"] = np.nan
    combined["delta_bic_vs_m0"] = np.nan

    group_keys = [
        "condition_name",
        "target_name",
    ]

    for _, index in combined.groupby(
        group_keys,
        sort=False,
    ).groups.items():
        family = combined.loc[index]

        m0_rows = family.loc[
            family["model_id"].eq("M0")
        ]

        if len(m0_rows) != 1:
            continue

        m0 = m0_rows.iloc[0]

        for row_index in index:
            candidate = combined.loc[
                row_index
            ]

            candidate_mae = _safe_float(
                candidate.get(
                    "participant_balanced_oob_mae"
                )
            )

            candidate_r2 = _safe_float(
                candidate.get(
                    "pooled_oob_r2"
                )
            )

            candidate_aic = _safe_float(
                candidate.get("aic")
            )

            candidate_bic = _safe_float(
                candidate.get("bic")
            )

            m0_mae = _safe_float(
                m0.get(
                    "participant_balanced_oob_mae"
                )
            )

            m0_r2 = _safe_float(
                m0.get("pooled_oob_r2")
            )

            m0_aic = _safe_float(
                m0.get("aic")
            )

            m0_bic = _safe_float(
                m0.get("bic")
            )

            if (
                np.isfinite(candidate_mae)
                and np.isfinite(m0_mae)
            ):
                combined.loc[
                    row_index,
                    "delta_mae_vs_m0",
                ] = (
                    m0_mae
                    - candidate_mae
                )

            if (
                np.isfinite(candidate_r2)
                and np.isfinite(m0_r2)
            ):
                combined.loc[
                    row_index,
                    "delta_oob_r2_vs_m0",
                ] = (
                    candidate_r2
                    - m0_r2
                )

            if (
                np.isfinite(candidate_aic)
                and np.isfinite(m0_aic)
            ):
                combined.loc[
                    row_index,
                    "delta_aic_vs_m0",
                ] = (
                    candidate_aic
                    - m0_aic
                )

            if (
                np.isfinite(candidate_bic)
                and np.isfinite(m0_bic)
            ):
                combined.loc[
                    row_index,
                    "delta_bic_vs_m0",
                ] = (
                    candidate_bic
                    - m0_bic
                )

    combined["delta_mae_convention"] = (
        "MAE_M0_minus_MAE_candidate; "
        "positive means candidate predicts better"
    )

    combined["delta_oob_r2_convention"] = (
        "R2_candidate_minus_R2_M0; "
        "positive means candidate predicts better"
    )

    combined["delta_aic_convention"] = (
        "AIC_candidate_minus_AIC_M0; "
        "negative means candidate favoured"
    )

    combined["delta_bic_convention"] = (
        "BIC_candidate_minus_BIC_M0; "
        "negative means candidate favoured"
    )

    combined["selection_status"] = (
        "evidence_only_no_automatic_selection"
    )

    return combined


def build_bootstrap_fit_audit(
    bootstrap_fits: pd.DataFrame,
) -> pd.DataFrame:
    """Summarise convergence and warning evidence per candidate."""
    if bootstrap_fits.empty:
        return pd.DataFrame()

    rows: list[dict[str, object]] = []

    group_keys = [
        "condition_name",
        "target_name",
        "model_id",
    ]

    for key, family in bootstrap_fits.groupby(
        group_keys,
        sort=False,
    ):
        condition, target, model_id = key

        warnings_present = (
            family.get(
                "warnings",
                pd.Series(
                    "",
                    index=family.index,
                ),
            )
            .fillna("")
            .astype(str)
            .str.strip()
            .ne("")
        )

        errors_present = (
            family.get(
                "fit_errors",
                pd.Series(
                    "",
                    index=family.index,
                ),
            )
            .fillna("")
            .astype(str)
            .str.strip()
            .ne("")
        )

        optimizer_counts = (
            family.get(
                "optimizer",
                pd.Series(
                    "",
                    index=family.index,
                ),
            )
            .fillna("")
            .astype(str)
            .value_counts()
            .sort_index()
        )

        optimizer_text = " | ".join(
            f"{optimizer}:{int(count)}"
            for optimizer, count
            in optimizer_counts.items()
            if optimizer
        )

        rows.append(
            {
                "condition_name": condition,
                "target_name": target,
                "model_id": model_id,
                "fit_attempt_count": int(
                    len(family)
                ),
                "converged_fit_count": int(
                    family[
                        "convergence_status"
                    ].eq("converged").sum()
                ),
                "prediction_success_count": int(
                    family[
                        "prediction_status"
                    ].eq("success").sum()
                ),
                "fit_or_prediction_failure_count": int(
                    ~(
                        family[
                            "convergence_status"
                        ].eq("converged")
                        & family[
                            "prediction_status"
                        ].eq("success")
                    )
                .sum()
                ),
                "warning_attempt_count": int(
                    warnings_present.sum()
                ),
                "fit_error_attempt_count": int(
                    errors_present.sum()
                ),
                "optimizer_counts": optimizer_text,
            }
        )

    return pd.DataFrame(rows)


def _json_default(value):
    if isinstance(value, np.integer):
        return int(value)

    if isinstance(value, np.floating):
        return float(value)

    if isinstance(value, Path):
        return str(value)

    raise TypeError(
        f"Cannot JSON-encode {type(value).__name__}"
    )


def _bootstrap_plan_signature(
    training_participant_ids: Sequence[str],
    bootstrap_plan: Sequence[Sequence[str]],
    *,
    conditions: Sequence[str],
    targets: Sequence[str],
    model_ids: Sequence[str],
) -> str:
    """Return a reproducible checkpoint configuration signature."""
    payload = {
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
        "script_version": SCRIPT_VERSION,
        "training_participant_ids": list(
            training_participant_ids
        ),
        "bootstrap_plan": [
            list(draw)
            for draw in bootstrap_plan
        ],
        "conditions": list(conditions),
        "targets": list(targets),
        "model_ids": list(model_ids),
    }

    encoded = json.dumps(
        payload,
        sort_keys=True,
    ).encode("utf-8")

    return hashlib.sha256(
        encoded
    ).hexdigest()


def _checkpoint_path(
    checkpoint_dir: Path,
    replicate_index: int,
) -> Path:
    return (
        Path(checkpoint_dir)
        / f"replicate_{replicate_index:04d}.json"
    )


def _save_replicate_checkpoint(
    path: Path,
    *,
    signature: str,
    predictions: pd.DataFrame,
    fits: pd.DataFrame,
) -> None:
    path = Path(path)
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    payload = {
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
        "signature": signature,
        "predictions": predictions.to_dict(
            orient="records"
        ),
        "fits": fits.to_dict(
            orient="records"
        ),
    }

    temporary = path.with_suffix(
        ".tmp"
    )

    temporary.write_text(
        json.dumps(
            payload,
            allow_nan=True,
            default=_json_default,
        ),
        encoding="utf-8",
    )

    os.replace(
        temporary,
        path,
    )


def _load_replicate_checkpoint(
    path: Path,
    *,
    expected_signature: str,
) -> tuple[pd.DataFrame, pd.DataFrame] | None:
    path = Path(path)

    if not path.exists():
        return None

    try:
        payload = json.loads(
            path.read_text(
                encoding="utf-8"
            )
        )
    except (
        OSError,
        json.JSONDecodeError,
    ):
        return None

    if (
        payload.get("schema_version")
        != CHECKPOINT_SCHEMA_VERSION
    ):
        return None

    if (
        payload.get("signature")
        != expected_signature
    ):
        return None

    return (
        pd.DataFrame(
            payload.get(
                "predictions",
                [],
            )
        ),
        pd.DataFrame(
            payload.get(
                "fits",
                [],
            )
        ),
    )


def _configure_worker_thread_limits() -> None:
    """Avoid nested BLAS threading when bootstrap replicates run in parallel."""
    for variable in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        os.environ.setdefault(
            variable,
            "1",
        )


def resolve_job_count(
    requested: int,
) -> int:
    """Resolve 0 to a conservative automatic process count."""
    if requested < 0:
        raise Mod11Error(
            "jobs must be >= 0"
        )

    if requested > 0:
        return int(requested)

    available = int(
        os.cpu_count() or 1
    )

    return max(
        1,
        min(
            DEFAULT_AUTO_MAX_JOBS,
            available,
        ),
    )


def _execute_bootstrap_task(
    training_data: pd.DataFrame,
    training_participant_ids: Sequence[str],
    sampled_participant_ids: Sequence[str],
    replicate_index: int,
    conditions: Sequence[str],
    targets: Sequence[str],
    model_ids: Sequence[str],
):
    predictions, fits = run_bootstrap_replicate(
        training_data,
        training_participant_ids=(
            training_participant_ids
        ),
        sampled_participant_ids=(
            sampled_participant_ids
        ),
        replicate_index=replicate_index,
        conditions=conditions,
        targets=targets,
        model_ids=model_ids,
    )

    return (
        replicate_index,
        predictions,
        fits,
    )


def run_bootstrap_plan(
    training_data: pd.DataFrame,
    *,
    training_participant_ids: Sequence[str],
    bootstrap_plan: Sequence[Sequence[str]],
    conditions: Sequence[str] = CONDITIONS,
    targets: Sequence[str] = TARGETS,
    model_ids: Sequence[str] = MODEL_IDS,
    jobs: int = 1,
    checkpoint_dir: Path | None = None,
    resume: bool = True,
    progress: bool = True,
    progress_every: int = 10,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run all bootstrap replicates with optional checkpoint resume."""
    if not bootstrap_plan:
        raise Mod11Error(
            "bootstrap_plan must not be empty"
        )

    if progress_every < 1:
        raise Mod11Error(
            "progress_every must be >= 1"
        )

    training_ids = (
        _validate_training_participant_ids(
            training_participant_ids
        )
    )

    resolved_jobs = resolve_job_count(
        jobs
    )

    signature = _bootstrap_plan_signature(
        training_ids,
        bootstrap_plan,
        conditions=conditions,
        targets=targets,
        model_ids=model_ids,
    )

    result_by_replicate: dict[
        int,
        tuple[
            pd.DataFrame,
            pd.DataFrame,
        ],
    ] = {}

    pending: list[
        tuple[int, Sequence[str]]
    ] = []

    for replicate_index, sampled in enumerate(
        bootstrap_plan,
        start=1,
    ):
        checkpoint = None

        if (
            resume
            and checkpoint_dir is not None
        ):
            checkpoint = (
                _load_replicate_checkpoint(
                    _checkpoint_path(
                        checkpoint_dir,
                        replicate_index,
                    ),
                    expected_signature=signature,
                )
            )

        if checkpoint is not None:
            result_by_replicate[
                replicate_index
            ] = checkpoint
        else:
            pending.append(
                (
                    replicate_index,
                    sampled,
                )
            )

    completed = len(
        result_by_replicate
    )

    def record_result(
        replicate_index: int,
        predictions: pd.DataFrame,
        fits: pd.DataFrame,
    ) -> None:
        nonlocal completed

        result_by_replicate[
            replicate_index
        ] = (
            predictions,
            fits,
        )

        if checkpoint_dir is not None:
            _save_replicate_checkpoint(
                _checkpoint_path(
                    checkpoint_dir,
                    replicate_index,
                ),
                signature=signature,
                predictions=predictions,
                fits=fits,
            )

        completed += 1

        if (
            progress
            and (
                completed % progress_every == 0
                or completed
                == len(bootstrap_plan)
            )
        ):
            print(
                "[MOD-11] bootstrap "
                f"{completed}/{len(bootstrap_plan)}",
                flush=True,
            )

    if resolved_jobs == 1:
        for replicate_index, sampled in pending:
            predictions, fits = (
                run_bootstrap_replicate(
                    training_data,
                    training_participant_ids=(
                        training_ids
                    ),
                    sampled_participant_ids=(
                        sampled
                    ),
                    replicate_index=(
                        replicate_index
                    ),
                    conditions=conditions,
                    targets=targets,
                    model_ids=model_ids,
                )
            )

            record_result(
                replicate_index,
                predictions,
                fits,
            )

    else:
        _configure_worker_thread_limits()

        with ProcessPoolExecutor(
            max_workers=resolved_jobs
        ) as executor:
            futures = {
                executor.submit(
                    _execute_bootstrap_task,
                    training_data,
                    training_ids,
                    sampled,
                    replicate_index,
                    tuple(conditions),
                    tuple(targets),
                    tuple(model_ids),
                ): replicate_index
                for replicate_index, sampled
                in pending
            }

            for future in as_completed(
                futures
            ):
                replicate_index = futures[
                    future
                ]

                (
                    returned_index,
                    predictions,
                    fits,
                ) = future.result()

                if (
                    returned_index
                    != replicate_index
                ):
                    raise Mod11Error(
                        "Bootstrap worker returned "
                        "the wrong replicate"
                    )

                record_result(
                    replicate_index,
                    predictions,
                    fits,
                )

    ordered_predictions: list[
        pd.DataFrame
    ] = []

    ordered_fits: list[
        pd.DataFrame
    ] = []

    for replicate_index in range(
        1,
        len(bootstrap_plan) + 1,
    ):
        predictions, fits = (
            result_by_replicate[
                replicate_index
            ]
        )

        if not predictions.empty:
            ordered_predictions.append(
                predictions
            )

        if not fits.empty:
            ordered_fits.append(
                fits
            )

    return (
        (
            pd.concat(
                ordered_predictions,
                ignore_index=True,
            )
            if ordered_predictions
            else pd.DataFrame()
        ),
        (
            pd.concat(
                ordered_fits,
                ignore_index=True,
            )
            if ordered_fits
            else pd.DataFrame()
        ),
    )


def run_mod11_analysis(
    training_data: pd.DataFrame,
    *,
    training_participant_ids: Sequence[str],
    bootstrap_plan: Sequence[Sequence[str]],
    conditions: Sequence[str] = CONDITIONS,
    targets: Sequence[str] = TARGETS,
    model_ids: Sequence[str] = MODEL_IDS,
    jobs: int = 1,
    checkpoint_dir: Path | None = None,
    resume: bool = True,
    progress: bool = True,
) -> Mod11Result:
    """Run the complete MOD-11 training-only comparison."""
    original_fits = (
        fit_original_training_models(
            training_data,
            conditions=conditions,
            targets=targets,
            model_ids=model_ids,
        )
    )

    predictions, bootstrap_fits = (
        run_bootstrap_plan(
            training_data,
            training_participant_ids=(
                training_participant_ids
            ),
            bootstrap_plan=bootstrap_plan,
            conditions=conditions,
            targets=targets,
            model_ids=model_ids,
            jobs=jobs,
            checkpoint_dir=checkpoint_dir,
            resume=resume,
            progress=progress,
        )
    )

    oob_summary, eligibility = (
        aggregate_oob_metrics(
            training_data,
            predictions,
            bootstrap_fits,
            conditions=conditions,
            targets=targets,
            model_ids=model_ids,
        )
    )

    comparison = (
        combine_comparison_evidence(
            oob_summary,
            original_fits,
        )
    )

    audit = build_bootstrap_fit_audit(
        bootstrap_fits
    )

    return Mod11Result(
        original_fits=original_fits,
        bootstrap_predictions=predictions,
        bootstrap_fits=bootstrap_fits,
        eligibility=eligibility,
        comparison=comparison,
        fit_audit=audit,
    )


def write_mod11_outputs(
    result: Mod11Result,
    output_dir: Path,
    *,
    manifest_metadata: dict | None = None,
) -> dict[str, Path]:
    """Write auditable MOD-11 evidence without choosing a model."""
    output = Path(output_dir)

    output.mkdir(
        parents=True,
        exist_ok=True,
    )

    paths = {
        "comparison": (
            output
            / COMPARISON_FILENAME
        ),
        "original_fits": (
            output
            / ORIGINAL_FITS_FILENAME
        ),
        "bootstrap_fits": (
            output
            / BOOTSTRAP_FITS_FILENAME
        ),
        "predictions": (
            output
            / PREDICTIONS_FILENAME
        ),
        "eligibility": (
            output
            / ELIGIBILITY_FILENAME
        ),
        "audit": (
            output
            / AUDIT_FILENAME
        ),
        "manifest": (
            output
            / MANIFEST_FILENAME
        ),
    }

    result.comparison.to_csv(
        paths["comparison"],
        index=False,
        encoding="utf-8",
        na_rep="",
    )

    result.original_fits.to_csv(
        paths["original_fits"],
        index=False,
        encoding="utf-8",
        na_rep="",
    )

    result.bootstrap_fits.to_csv(
        paths["bootstrap_fits"],
        index=False,
        encoding="utf-8",
        na_rep="",
        compression="gzip",
    )

    result.bootstrap_predictions.to_csv(
        paths["predictions"],
        index=False,
        encoding="utf-8",
        na_rep="",
        compression="gzip",
    )

    result.eligibility.to_csv(
        paths["eligibility"],
        index=False,
        encoding="utf-8",
        na_rep="",
    )

    result.fit_audit.to_csv(
        paths["audit"],
        index=False,
        encoding="utf-8",
        na_rep="",
    )

    manifest = {
        "script_version": SCRIPT_VERSION,
        "method": (
            "participant_cluster_bootstrap_"
            "loss_first_common_support"
        ),
        "random_structure": "RI",
        "default_final_bootstrap_replicates": (
            DEFAULT_BOOTSTRAP_REPLICATES
        ),
        "automatic_model_selection": False,
        "test_set_used": False,
        "metadata": (
            manifest_metadata or {}
        ),
    }

    paths["manifest"].write_text(
        json.dumps(
            manifest,
            indent=2,
            default=_json_default,
        ),
        encoding="utf-8",
    )

    return paths


def build_cli_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__
    )

    parser.add_argument(
        "--modeling-data",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--holdout-split",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--bootstrap-replicates",
        type=int,
        default=DEFAULT_BOOTSTRAP_REPLICATES,
    )

    parser.add_argument(
        "--seed",
        type=int,
        required=True,
        help=(
            "Explicit participant-bootstrap RNG seed. "
            "Freeze this before the official B=500 run."
        ),
    )

    parser.add_argument(
        "--jobs",
        type=int,
        default=0,
        help=(
            "Parallel replicate workers. "
            "0=automatic capped at 4."
        ),
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

    raw_data = pd.read_csv(
        args.modeling_data
    )

    holdout_split = pd.read_csv(
        args.holdout_split
    )

    (
        training_data,
        training_ids,
        test_ids,
    ) = prepare_training_data(
        modeling_data=raw_data,
        holdout_split=holdout_split,
    )

    plan = generate_bootstrap_plan(
        training_ids,
        n_replicates=(
            args.bootstrap_replicates
        ),
        seed=args.seed,
    )

    checkpoint_dir = (
        args.checkpoint_dir
        if args.checkpoint_dir is not None
        else (
            args.output_dir
            / "_bootstrap_checkpoints"
        )
    )

    result = run_mod11_analysis(
        training_data,
        training_participant_ids=(
            training_ids
        ),
        bootstrap_plan=plan,
        jobs=args.jobs,
        checkpoint_dir=checkpoint_dir,
        resume=not args.no_resume,
        progress=not args.no_progress,
    )

    write_mod11_outputs(
        result,
        args.output_dir,
        manifest_metadata={
            "bootstrap_replicates": (
                args.bootstrap_replicates
            ),
            "bootstrap_seed": (
                args.seed
            ),
            "training_participant_count": (
                len(training_ids)
            ),
            "final_test_participant_count": (
                len(test_ids)
            ),
        },
    )

    complete_metrics = int(
        result.comparison[
            "metric_status"
        ].eq("complete").sum()
    )

    total_metrics = int(
        len(result.comparison)
    )

    print(
        "MOD-11 complete: "
        f"metric_rows={complete_metrics}/"
        f"{total_metrics}; "
        f"B={args.bootstrap_replicates}; "
        "automatic_selection=false",
        flush=True,
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())