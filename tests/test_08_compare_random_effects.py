"""Synthetic MOD-02D RI versus RI+RS comparison tests."""

from __future__ import annotations

from dataclasses import FrozenInstanceError, replace
import importlib.util
from pathlib import Path
import sys
import unittest

import pandas as pd


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
PYTHON_ROOT = REPOSITORY_ROOT / "python"
REGISTRY_PATH = PYTHON_ROOT / "05_compare_random_effects.py"
RI_PATH = PYTHON_ROOT / "06_fit_random_intercept_models.py"
RS_PATH = PYTHON_ROOT / "07_fit_random_intercept_plus_slope_models.py"
COMPARE_PATH = PYTHON_ROOT / "08_compare_random_effects.py"


def load_module(path: Path, name: str):
    if not path.exists():
        return None
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        return None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


REGISTRY_MODULE = load_module(REGISTRY_PATH, "mod02a_registry_for_mod02d_tests")
RI_MODULE = load_module(RI_PATH, "mod02b_ri_for_mod02d_tests")
RS_MODULE = load_module(RS_PATH, "mod02c_rs_for_mod02d_tests")
COMPARE_MODULE = load_module(COMPARE_PATH, "mod02d_compare_for_tests")


PERFORMANCE_TARGET = "performance_change_from_d0_percentage_points"
MENTAL_DEMAND_TARGET = "mental_demand_score_0_to_10"
FORMULAS = {
    PERFORMANCE_TARGET: (
        "performance_change_from_d0_percentage_points ~ performance_D"
    ),
    MENTAL_DEMAND_TARGET: "mental_demand_score_0_to_10 ~ difficulty_stage",
}
DIFFICULTY_COLUMNS = {
    PERFORMANCE_TARGET: "performance_D",
    MENTAL_DEMAND_TARGET: "difficulty_stage",
}


def make_registry():
    return REGISTRY_MODULE.build_random_effects_registry("D * Age + D * TMT_z")


def records_to_frame(records, registry):
    columns = tuple(registry.comparison_fields) + RI_MODULE.COMPARISON_EXTRA_FIELDS
    rows = []
    for record in records:
        row = {}
        for field in registry.comparison_fields:
            value = getattr(record, field)
            if field == "warnings":
                value = "; ".join(record.warnings)
            row[field] = value
        for field in RI_MODULE.COMPARISON_EXTRA_FIELDS:
            row[field] = getattr(record, field)
        rows.append(row)
    return pd.DataFrame(rows, columns=columns)


def make_record(comparison, model_id, *, signature, **overrides):
    random_model = next(
        model for model in comparison.models if model.model_id == model_id
    )
    difficulty = comparison.difficulty_coding
    target_name = comparison.target_name
    is_performance = target_name == PERFORMANCE_TARGET
    record = RI_MODULE.RandomInterceptFitRecord(
        condition_name=comparison.condition_name,
        target_name=target_name,
        model_id=model_id,
        target_role=comparison.target_role,
        response_coding=comparison.response_coding,
        difficulty_term=difficulty.term_name,
        difficulty_source_column=DIFFICULTY_COLUMNS[target_name],
        difficulty_coding_status=difficulty.coding_status,
        difficulty_mapping=difficulty.mapping,
        fixed_effects_formula=comparison.fixed_effects_formula,
        executable_formula=FORMULAS[target_name],
        fixed_effects_status=comparison.fixed_effects_status,
        random_structure=random_model.random_structure,
        random_formula_label=random_model.random_formula_label,
        participant_count=4,
        observation_count=12 if is_performance else 16,
        excluded_observation_count=4 if is_performance else 0,
        excluded_participant_count=0,
        fixed_effect_count=2,
        total_parameter_count=3 if model_id == "RI" else 5,
        log_likelihood=-10.0 if model_id == "RI" else -9.0,
        aic=26.0 if model_id == "RI" else 24.0,
        delta_aic=None,
        bic=31.0 if model_id == "RI" else 29.0,
        delta_bic=None,
        estimation_method="maximum_likelihood",
        reml=False,
        convergence_status=True,
        optimizer="lbfgs",
        warnings=(),
        fit_status="fitted",
        fit_error=None,
        random_intercept_variance=0.7,
        random_slope_variance=None if model_id == "RI" else 0.5,
        intercept_slope_covariance=None if model_id == "RI" else 0.1,
        boundary_flag=None,
        singularity_flag=None,
        likelihood_ratio_statistic=None,
        likelihood_ratio_p_value=None,
        likelihood_ratio_role=REGISTRY_MODULE.LIKELIHOOD_RATIO_ROLE,
        selection_status=REGISTRY_MODULE.SELECTION_STATUS,
        included_row_signature=signature,
    )
    return replace(record, **overrides)


def make_records(registry):
    ri_records = []
    rs_records = []
    for index, comparison in enumerate(registry.comparisons):
        signature = f"{index},{index + 10}"
        ri_records.append(make_record(comparison, "RI", signature=signature))
        rs_records.append(make_record(comparison, "RI_RS", signature=signature))
    return ri_records, rs_records


def make_results(registry, ri_records=None, rs_records=None):
    if ri_records is None or rs_records is None:
        ri_records, rs_records = make_records(registry)
    ri_result = RI_MODULE.RandomInterceptFitResult(
        valid=True,
        errors=(),
        warnings=(),
        fit_records=tuple(ri_records),
        fit_table=records_to_frame(ri_records, registry),
    )
    rs_result = RS_MODULE.RandomEffectsFitResult(
        valid=True,
        errors=(),
        warnings=(),
        fit_records=tuple(rs_records),
        fit_table=records_to_frame(rs_records, registry),
    )
    return ri_result, rs_result


def record_index(registry, condition_name, target_name):
    return next(
        index
        for index, comparison in enumerate(registry.comparisons)
        if comparison.condition_name == condition_name
        and comparison.target_name == target_name
    )


class RandomEffectsComparisonTests(unittest.TestCase):
    def require_modules(self):
        self.assertIsNotNone(REGISTRY_MODULE, "MOD-02A registry must be importable")
        self.assertIsNotNone(RI_MODULE, "MOD-02B fitter must be importable")
        self.assertIsNotNone(RS_MODULE, "MOD-02C fitter must be importable")
        self.assertIsNotNone(COMPARE_MODULE, "MOD-02D comparator must be importable")
        return COMPARE_MODULE

    def test_six_converged_pairs_produce_long_common_table(self):
        module = self.require_modules()
        registry = make_registry()
        ri_result, rs_result = make_results(registry)

        result = module.compare_random_effects(ri_result, rs_result, registry)

        self.assertTrue(result.valid)
        self.assertEqual(len(result.pair_audits), 6)
        self.assertEqual(len(result.comparison_table), 12)
        self.assertEqual(
            list(result.comparison_table["model_id"]),
            [model_id for _ in registry.comparisons for model_id in ("RI", "RI_RS")],
        )
        self.assertTrue(
            all(audit.status == "comparable" for audit in result.pair_audits)
        )
        self.assertTrue(
            all(audit.same_observations for audit in result.pair_audits)
        )

    def test_common_schema_and_pending_selection_are_preserved(self):
        module = self.require_modules()
        registry = make_registry()
        ri_result, rs_result = make_results(registry)

        result = module.compare_random_effects(ri_result, rs_result, registry)

        expected_columns = tuple(registry.comparison_fields) + RI_MODULE.COMPARISON_EXTRA_FIELDS
        self.assertEqual(tuple(result.comparison_table.columns), expected_columns)
        self.assertNotIn("executable_random_formula", result.comparison_table.columns)
        self.assertTrue(
            (result.comparison_table["selection_status"] == "pending_supervisor_rule").all()
        )

    def test_deltas_and_supporting_likelihood_ratio_statistic(self):
        module = self.require_modules()
        registry = make_registry()
        ri_result, rs_result = make_results(registry)

        result = module.compare_random_effects(ri_result, rs_result, registry)
        first_pair = result.comparison_table.iloc[:2]

        self.assertEqual(list(first_pair["delta_aic"]), [2.0, 0.0])
        self.assertEqual(list(first_pair["delta_bic"]), [2.0, 0.0])
        self.assertEqual(list(first_pair["likelihood_ratio_statistic"]), [2.0, 2.0])
        self.assertTrue((first_pair["likelihood_ratio_p_value"].isna()).all())
        self.assertTrue((first_pair["likelihood_ratio_role"] == "supporting_only").all())

    def test_unavailable_metrics_warn_without_invalidating_comparable_pair(self):
        module = self.require_modules()
        registry = make_registry()
        ri_records, rs_records = make_records(registry)
        ri_records[0] = replace(ri_records[0], aic=None, log_likelihood=None)
        rs_records[0] = replace(rs_records[0], bic=None)
        ri_result, rs_result = make_results(registry, ri_records, rs_records)

        result = module.compare_random_effects(ri_result, rs_result, registry)
        first_pair = result.comparison_table.iloc[:2]

        self.assertTrue(result.valid)
        self.assertTrue(any("AIC" in warning for warning in result.warnings))
        self.assertTrue(any("BIC" in warning for warning in result.warnings))
        self.assertTrue(any("log likelihood" in warning for warning in result.warnings))
        self.assertTrue(first_pair["delta_aic"].isna().all())
        self.assertTrue(first_pair["delta_bic"].isna().all())
        self.assertTrue(first_pair["likelihood_ratio_statistic"].isna().all())

    def test_convergence_and_fit_status_are_required(self):
        module = self.require_modules()
        registry = make_registry()

        for field, value in (("convergence_status", None), ("convergence_status", False), ("fit_status", "non_converged")):
            with self.subTest(field=field, value=value):
                ri_records, rs_records = make_records(registry)
                rs_records[0] = replace(rs_records[0], **{field: value})
                ri_result, rs_result = make_results(registry, ri_records, rs_records)

                result = module.compare_random_effects(ri_result, rs_result, registry)

                self.assertFalse(result.valid)
                self.assertEqual(result.pair_audits[0].status, "failed")
                self.assertTrue(result.pair_audits[1].status == "comparable")

    def test_difficulty_metadata_must_match_exactly(self):
        module = self.require_modules()
        registry = make_registry()
        fields_and_values = {
            "difficulty_term": "different_D",
            "difficulty_source_column": "different_column",
            "difficulty_coding_status": "approved",
            "difficulty_mapping": ((0, 1), (2, 2), (6, 3), (10, 4)),
        }

        for field, value in fields_and_values.items():
            with self.subTest(field=field):
                ri_records, rs_records = make_records(registry)
                rs_records[0] = replace(rs_records[0], **{field: value})
                ri_result, rs_result = make_results(registry, ri_records, rs_records)

                result = module.compare_random_effects(ri_result, rs_result, registry)

                self.assertFalse(result.valid)
                self.assertEqual(result.pair_audits[0].status, "failed")

    def test_formula_row_signature_and_counts_must_match(self):
        module = self.require_modules()
        registry = make_registry()
        changes = {
            "fixed_effects_formula": "different_formula",
            "executable_formula": "other_response ~ other_difficulty",
            "included_row_signature": "different_rows",
            "observation_count": 99,
            "excluded_observation_count": 99,
        }

        for field, value in changes.items():
            with self.subTest(field=field):
                ri_records, rs_records = make_records(registry)
                rs_records[0] = replace(rs_records[0], **{field: value})
                ri_result, rs_result = make_results(registry, ri_records, rs_records)

            result = module.compare_random_effects(ri_result, rs_result, registry)

            self.assertFalse(result.valid)
            if field in {"included_row_signature", "observation_count"}:
                self.assertFalse(result.pair_audits[0].same_observations)
            else:
                self.assertTrue(result.pair_audits[0].same_observations)
            self.assertEqual(result.pair_audits[0].status, "failed")

    def test_pair_failures_do_not_stop_unrelated_pairs(self):
        module = self.require_modules()
        registry = make_registry()
        ri_records, rs_records = make_records(registry)
        rs_records[0] = replace(rs_records[0], included_row_signature="mismatch")
        ri_result, rs_result = make_results(registry, ri_records, rs_records)

        result = module.compare_random_effects(ri_result, rs_result, registry)

        self.assertFalse(result.valid)
        self.assertEqual(result.pair_audits[0].status, "failed")
        self.assertTrue(
            all(audit.status == "comparable" for audit in result.pair_audits[1:])
        )

    def test_missing_and_duplicate_records_are_preserved_as_diagnostics(self):
        module = self.require_modules()
        registry = make_registry()
        ri_records, rs_records = make_records(registry)
        missing_rs = rs_records.pop(0)
        rs_result_records = rs_records
        ri_records.append(ri_records[1])
        ri_result, rs_result = make_results(registry, ri_records, rs_result_records)

        result = module.compare_random_effects(ri_result, rs_result, registry)

        self.assertFalse(result.valid)
        self.assertEqual(len(result.comparison_table), 12)
        self.assertIsNotNone(missing_rs)
        self.assertTrue(any("record" in error.lower() for error in result.errors))

    def test_result_and_output_are_frozen_and_input_tables_are_independent(self):
        module = self.require_modules()
        registry = make_registry()
        ri_result, rs_result = make_results(registry)
        ri_original = ri_result.fit_table.copy(deep=True)
        rs_original = rs_result.fit_table.copy(deep=True)

        result = module.compare_random_effects(ri_result, rs_result, registry)

        with self.assertRaises(FrozenInstanceError):
            result.valid = False
        result.comparison_table.loc[0, "model_id"] = "changed"
        self.assertTrue(ri_result.fit_table.equals(ri_original))
        self.assertTrue(rs_result.fit_table.equals(rs_original))


if __name__ == "__main__":
    unittest.main()
