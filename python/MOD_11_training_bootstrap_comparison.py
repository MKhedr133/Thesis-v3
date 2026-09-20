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

import MOD_10_training_random_effects_comparison as MOD10


SCRIPT_VERSION = "1.0.0"

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