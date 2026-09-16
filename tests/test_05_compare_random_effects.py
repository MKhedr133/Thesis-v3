"""Synthetic MOD-02A RI/RS registry contract tests."""

from __future__ import annotations

from dataclasses import FrozenInstanceError
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import sys
import unittest


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = REPOSITORY_ROOT / "python" / "05_compare_random_effects.py"


def load_registry_module():
    if not MODULE_PATH.exists():
        return None
    specification = spec_from_file_location("compare_random_effects", MODULE_PATH)
    if specification is None or specification.loader is None:
        return None
    module = module_from_spec(specification)
    sys.modules[specification.name] = module
    specification.loader.exec_module(module)
    return module


REGISTRY = load_registry_module()


EXPECTED_FIELDS = (
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


class RandomEffectsRegistryTests(unittest.TestCase):
    def require_registry(self):
        self.assertIsNotNone(
            REGISTRY,
            "python/05_compare_random_effects.py must be implemented",
        )
        return REGISTRY

    def build(self, formula="D * Age + D * TMT_z"):
        registry = self.require_registry()
        return registry.build_random_effects_registry(formula)

    def test_registry_has_six_separate_primary_condition_target_specs(self):
        registry = self.build()

        self.assertEqual(
            registry.conditions,
            ("Visual", "Auditory", "Cognitive"),
        )
        self.assertEqual(
            registry.targets,
            (
                "performance_change_from_d0_percentage_points",
                "mental_demand_score_0_to_10",
            ),
        )
        self.assertEqual(len(registry.comparisons), 6)
        self.assertEqual(
            {(item.condition_name, item.target_name) for item in registry.comparisons},
            {
                (condition, target)
                for condition in registry.conditions
                for target in registry.targets
            },
        )

    def test_each_spec_has_exactly_ri_and_ri_rs_metadata(self):
        registry = self.build()

        for comparison in registry.comparisons:
            self.assertEqual(
                [model.model_id for model in comparison.models],
                ["RI", "RI_RS"],
            )
            ri, ri_rs = comparison.models
            self.assertEqual(ri.random_structure, "RI")
            self.assertEqual(ri.random_formula_label, "(1 | participant)")
            self.assertIsNone(ri.varying_term)
            self.assertEqual(ri_rs.random_structure, "RI_RS")
            self.assertEqual(
                ri_rs.random_formula_label,
                "(1 + D | participant)",
            )
            self.assertEqual(
                ri_rs.varying_term,
                comparison.difficulty_coding.term_name,
            )
            self.assertFalse(hasattr(ri, "formula"))
            self.assertFalse(hasattr(ri_rs, "formula"))

    def test_human_readable_random_labels_are_not_executable_formulas(self):
        registry = self.build()
        labels = {
            model.random_formula_label
            for comparison in registry.comparisons
            for model in comparison.models
        }

        self.assertEqual(labels, {"(1 | participant)", "(1 + D | participant)"})
        self.assertFalse(any("statsmodels" in name.lower() for name in dir(REGISTRY)))

    def test_difficulty_mapping_and_target_specific_coding_are_explicit(self):
        registry = self.build()
        self.assertEqual(
            registry.difficulty_stage_mapping,
            ((0, 0), (2, 1), (6, 2), (10, 3)),
        )

        performance = next(
            item
            for item in registry.comparisons
            if item.target_name == "performance_change_from_d0_percentage_points"
        )
        mental_demand = next(
            item
            for item in registry.comparisons
            if item.target_name == "mental_demand_score_0_to_10"
        )

        self.assertEqual(performance.difficulty_coding.term_name, "D")
        self.assertIsNone(performance.difficulty_coding.source_column)
        self.assertEqual(performance.difficulty_coding.coding_status, "unapproved")
        self.assertIsNone(performance.difficulty_coding.mapping)

        self.assertEqual(
            mental_demand.difficulty_coding.term_name,
            "difficulty_stage",
        )
        self.assertEqual(
            mental_demand.difficulty_coding.source_column,
            "difficulty_stage",
        )
        self.assertEqual(mental_demand.difficulty_coding.coding_status, "approved")
        self.assertEqual(
            mental_demand.difficulty_coding.mapping,
            registry.difficulty_stage_mapping,
        )

    def test_formula_is_preserved_but_remains_unapproved(self):
        formula = "D * Age + D * TMT_z"
        registry = self.build(formula)

        for comparison in registry.comparisons:
            self.assertEqual(comparison.fixed_effects_formula, formula)
            self.assertEqual(comparison.fixed_effects_status, "unapproved")
            self.assertEqual(comparison.target_role, "primary_outcome")

    def test_empty_fixed_effects_formula_is_rejected(self):
        registry = self.require_registry()

        with self.assertRaises(ValueError):
            registry.build_random_effects_registry("")
        with self.assertRaises(ValueError):
            registry.build_random_effects_registry("   ")

    def test_secondary_features_and_total_error_are_not_targets(self):
        registry = self.build()

        for comparison in registry.comparisons:
            self.assertNotIn("total_error", comparison.target_name)
            self.assertNotIn("median_", comparison.target_name)
            self.assertNotIn("list_", comparison.target_name)

    def test_comparison_field_contract_is_complete_and_ordered(self):
        registry = self.build()

        self.assertEqual(registry.comparison_fields, EXPECTED_FIELDS)

    def test_selection_and_likelihood_ratio_statuses_are_non_decisional(self):
        registry_module = self.require_registry()
        registry = self.build()

        self.assertIn("likelihood_ratio_role", registry.comparison_fields)
        self.assertIn("selection_status", registry.comparison_fields)
        self.assertEqual(
            registry_module.LIKELIHOOD_RATIO_ROLE,
            "supporting_only",
        )
        self.assertEqual(
            registry_module.SELECTION_STATUS,
            "pending_supervisor_rule",
        )

    def test_result_dataclasses_are_frozen_and_registry_is_deterministic(self):
        registry_module = self.require_registry()
        first = self.build()
        second = self.build()

        self.assertEqual(first, second)
        self.assertTrue(
            registry_module.DifficultyCodingSpec.__dataclass_params__.frozen
        )
        self.assertTrue(
            registry_module.RandomEffectsModelSpec.__dataclass_params__.frozen
        )
        self.assertTrue(
            registry_module.RandomEffectsComparisonSpec.__dataclass_params__.frozen
        )
        self.assertTrue(
            registry_module.RandomEffectsComparisonRegistry.__dataclass_params__.frozen
        )
        with self.assertRaises(FrozenInstanceError):
            first.conditions = ("Other",)


if __name__ == "__main__":
    unittest.main()
