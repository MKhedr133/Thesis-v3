"""Define the pre-fit RI/RS comparison registry for the modelling phase.

This module records human-readable model metadata only. It does not import a
model-fitting library, construct executable formula objects, fit models, or
write modelling-result files.
"""

from __future__ import annotations

from dataclasses import dataclass


CONDITIONS = ("Visual", "Auditory", "Cognitive")
TARGETS = (
    "performance_change_from_d0_percentage_points",
    "mental_demand_score_0_to_10",
)
DIFFICULTY_STAGE_MAPPING = ((0, 0), (2, 1), (6, 2), (10, 3))
LIKELIHOOD_RATIO_ROLE = "supporting_only"
SELECTION_STATUS = "pending_supervisor_rule"

COMPARISON_FIELDS = (
    "model_id",
    "condition_name",
    "target_name",
    "target_role",
    "response_coding",
    "difficulty_term",
    "difficulty_source_column",
    "difficulty_coding_status",
    "difficulty_mapping",
    "fixed_effects_formula",
    "fixed_effects_status",
    "random_structure",
    "random_formula_label",
    "participant_count",
    "observation_count",
    "fixed_effect_count",
    "total_parameter_count",
    "log_likelihood",
    "aic",
    "delta_aic",
    "bic",
    "delta_bic",
    "estimation_method",
    "reml",
    "convergence_status",
    "optimizer",
    "warnings",
    "random_intercept_variance",
    "random_slope_variance",
    "intercept_slope_covariance",
    "boundary_flag",
    "singularity_flag",
    "likelihood_ratio_statistic",
    "likelihood_ratio_p_value",
    "likelihood_ratio_role",
    "selection_status",
)


@dataclass(frozen=True)
class DifficultyCodingSpec:
    """Target-specific difficulty term and coding status."""

    term_name: str
    source_column: str | None
    coding_status: str
    mapping: tuple[tuple[int, int], ...] | None


@dataclass(frozen=True)
class RandomEffectsModelSpec:
    """Human-readable random-effects metadata, not an executable formula."""

    model_id: str
    random_structure: str
    random_formula_label: str
    varying_term: str | None


@dataclass(frozen=True)
class RandomEffectsComparisonSpec:
    """One condition-target RI versus RI+RS comparison contract."""

    condition_name: str
    target_name: str
    target_role: str
    response_coding: str
    difficulty_coding: DifficultyCodingSpec
    fixed_effects_formula: str
    fixed_effects_status: str
    models: tuple[RandomEffectsModelSpec, ...]


@dataclass(frozen=True)
class RandomEffectsComparisonRegistry:
    """Complete MOD-02A pre-fit registry."""

    conditions: tuple[str, ...]
    targets: tuple[str, ...]
    difficulty_stage_mapping: tuple[tuple[int, int], ...]
    comparison_fields: tuple[str, ...]
    comparisons: tuple[RandomEffectsComparisonSpec, ...]


def _difficulty_for_target(target_name: str) -> DifficultyCodingSpec:
    if target_name == "mental_demand_score_0_to_10":
        return DifficultyCodingSpec(
            term_name="difficulty_stage",
            source_column="difficulty_stage",
            coding_status="approved",
            mapping=DIFFICULTY_STAGE_MAPPING,
        )
    if target_name == "performance_change_from_d0_percentage_points":
        return DifficultyCodingSpec(
            term_name="D",
            source_column=None,
            coding_status="unapproved",
            mapping=None,
        )
    raise ValueError(f"Unsupported MOD-02A target: {target_name}")


def _models_for_target(
    difficulty_coding: DifficultyCodingSpec,
) -> tuple[RandomEffectsModelSpec, ...]:
    return (
        RandomEffectsModelSpec(
            model_id="RI",
            random_structure="RI",
            random_formula_label="(1 | participant)",
            varying_term=None,
        ),
        RandomEffectsModelSpec(
            model_id="RI_RS",
            random_structure="RI_RS",
            random_formula_label="(1 + D | participant)",
            varying_term=difficulty_coding.term_name,
        ),
    )


def build_random_effects_registry(
    fixed_effects_formula: str,
) -> RandomEffectsComparisonRegistry:
    """Build the six primary, pre-fit RI/RI+RS comparison specifications."""

    if not isinstance(fixed_effects_formula, str) or not fixed_effects_formula.strip():
        raise ValueError("fixed_effects_formula must be a non-empty string")

    comparisons: list[RandomEffectsComparisonSpec] = []
    for condition_name in CONDITIONS:
        for target_name in TARGETS:
            difficulty_coding = _difficulty_for_target(target_name)
            response_coding = (
                "D2/D6/D10; D0 structural zero/reference; not fitted"
                if target_name == "performance_change_from_d0_percentage_points"
                else "D0/D2/D6/D10"
            )
            comparisons.append(
                RandomEffectsComparisonSpec(
                    condition_name=condition_name,
                    target_name=target_name,
                    target_role="primary_outcome",
                    response_coding=response_coding,
                    difficulty_coding=difficulty_coding,
                    fixed_effects_formula=fixed_effects_formula,
                    fixed_effects_status="unapproved",
                    models=_models_for_target(difficulty_coding),
                )
            )

    return RandomEffectsComparisonRegistry(
        conditions=CONDITIONS,
        targets=TARGETS,
        difficulty_stage_mapping=DIFFICULTY_STAGE_MAPPING,
        comparison_fields=COMPARISON_FIELDS,
        comparisons=tuple(comparisons),
    )
