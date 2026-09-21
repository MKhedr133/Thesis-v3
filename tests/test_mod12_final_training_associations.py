"""Synthetic tests for MOD-12 final training-only associations."""

from __future__ import annotations

from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import sys
import unittest

import numpy as np
import pandas as pd

from unittest.mock import patch
from tempfile import TemporaryDirectory


ROOT = Path(__file__).resolve().parents[1]
PYTHON_DIR = ROOT / "python"
sys.path.insert(0, str(PYTHON_DIR))

MODULE_PATH = PYTHON_DIR / "MOD_12_final_training_associations.py"

SPEC = spec_from_file_location(
    "mod12_final_training_associations_test",
    MODULE_PATH,
)

if SPEC is None or SPEC.loader is None:
    raise ImportError(MODULE_PATH)

MOD12 = module_from_spec(SPEC)
sys.modules[SPEC.name] = MOD12
SPEC.loader.exec_module(MOD12)


def synthetic_modeling_data() -> pd.DataFrame:
    """Create synthetic repeated data for 32 participants."""
    rows = []

    conditions = ("Visual", "Auditory", "Cognitive")
    difficulties = (
        (0, 0),
        (2, 1),
        (6, 2),
        (10, 3),
    )

    for group, prefix in (("Young", "Y"), ("Old", "O")):
        for number in range(1, 17):
            participant_id = f"{prefix}{number:02d}"

            for condition in conditions:
                for difficulty_level, difficulty_stage in difficulties:
                    value = float(number + difficulty_stage)

                    rows.append(
                        {
                            "participant_id": participant_id,
                            "participant_group": group,
                            "condition_name": condition,
                            "difficulty_level": difficulty_level,
                            "difficulty_stage": difficulty_stage,
                            "performance_change_from_d0_percentage_points": (
                                float(difficulty_stage * 5)
                            ),
                            "mental_demand_score_0_to_10": value,
                            "median_time_between_qualifying_grabs_seconds": value,
                            "list_recheck_count": float(difficulty_stage),
                            "total_list_recheck_duration_seconds": value,
                            "median_time_to_target_seconds": value,
                            "median_irrelevant_focus_duration_seconds": value,
                            "median_head_turning_degrees": value,
                            "median_reach_duration_seconds": value,
                            "median_reach_path_ratio": 1.0 + value / 100.0,
                        }
                    )

    data = pd.DataFrame(rows)

    # Genuine behavioural zero: this must stay zero.
    zero_mask = (
        data["participant_id"].eq("Y04")
        & data["condition_name"].eq("Visual")
        & data["difficulty_level"].eq(2)
    )
    data.loc[zero_mask, "list_recheck_count"] = 0.0

    # One genuine missing measurement.
    missing_mask = (
        data["participant_id"].eq("Y04")
        & data["condition_name"].eq("Visual")
        & data["difficulty_level"].eq(6)
    )
    data.loc[
        missing_mask,
        "median_reach_duration_seconds",
    ] = np.nan

    return data


def synthetic_holdout_split() -> pd.DataFrame:
    """Create a valid MOD-09-style 26/6 participant split."""
    rows = []

    for group, prefix in (("Young", "Y"), ("Old", "O")):
        for number in range(1, 17):
            rows.append(
                {
                    "participant_id": f"{prefix}{number:02d}",
                    "participant_group": group,
                    "split": (
                        "test"
                        if number in {1, 2, 3}
                        else "train"
                    ),
                }
            )

    return pd.DataFrame(rows)

class _FakeAssociationResult:
    def __init__(
        self,
        difficulty_column: str,
        *,
        coefficient: float = 0.75,
        standard_error: float = 0.20,
        p_value: float = 0.01,
        ci_lower: float = 0.35,
        ci_upper: float = 1.15,
        converged: bool = True,
    ):
        self.converged = converged
        self.llf = -100.0

        self.fe_params = pd.Series(
            {
                "Intercept": 99.0,
                difficulty_column: coefficient,
            }
        )

        self.bse = pd.Series(
            {
                "Intercept": 9.0,
                difficulty_column: standard_error,
            }
        )

        self.pvalues = pd.Series(
            {
                "Intercept": 0.99,
                difficulty_column: p_value,
            }
        )

        self._confidence_intervals = pd.DataFrame(
            {
                0: [-999.0, ci_lower],
                1: [999.0, ci_upper],
            },
            index=[
                "Intercept",
                difficulty_column,
            ],
        )

    def conf_int(self):
        return self._confidence_intervals


class AssociationFitTests(unittest.TestCase):
    def setUp(self):
        self.training_data, _, _ = (
            MOD12.prepare_training_data(
                modeling_data=synthetic_modeling_data(),
                holdout_split=synthetic_holdout_split(),
            )
        )

    def test_mod12_requests_reml_and_ri(self):
        target = "mental_demand_score_0_to_10"
        difficulty_column = "difficulty_stage"

        fake_result = _FakeAssociationResult(
            difficulty_column,
        )

        with patch.object(
            MOD12.MOD02,
            "fit_mixedlm_with_fallback",
            return_value=(
                fake_result,
                "lbfgs",
                (),
                (),
            ),
        ) as fitter:
            MOD12.fit_association(
                self.training_data,
                condition="Visual",
                target=target,
            )

        args = fitter.call_args.args
        kwargs = fitter.call_args.kwargs

        self.assertEqual(
            args[0],
            f"{target} ~ difficulty_stage",
        )

        self.assertEqual(
            args[2],
            "RI",
        )

        self.assertEqual(
            args[3],
            difficulty_column,
        )

        self.assertTrue(
            kwargs["reml"]
        )

    def test_extracts_only_difficulty_inference(self):
        target = "mental_demand_score_0_to_10"

        fake_result = _FakeAssociationResult(
            "difficulty_stage",
            coefficient=0.75,
            standard_error=0.20,
            p_value=0.01,
            ci_lower=0.35,
            ci_upper=1.15,
        )

        with patch.object(
            MOD12.MOD02,
            "fit_mixedlm_with_fallback",
            return_value=(
                fake_result,
                "lbfgs",
                (),
                (),
            ),
        ):
            evidence = MOD12.fit_association(
                self.training_data,
                condition="Visual",
                target=target,
            )

        self.assertEqual(
            evidence["difficulty_coefficient"],
            0.75,
        )

        self.assertEqual(
            evidence["standard_error"],
            0.20,
        )

        self.assertEqual(
            evidence["ci_95_lower"],
            0.35,
        )

        self.assertEqual(
            evidence["ci_95_upper"],
            1.15,
        )

        self.assertEqual(
            evidence["p_value"],
            0.01,
        )

        self.assertEqual(
            evidence["retention_status"],
            "retained",
        )

    def test_p_value_at_or_above_005_is_not_retained(self):
        fake_result = _FakeAssociationResult(
            "difficulty_stage",
            p_value=0.05,
        )

        with patch.object(
            MOD12.MOD02,
            "fit_mixedlm_with_fallback",
            return_value=(
                fake_result,
                "powell",
                (),
                (),
            ),
        ):
            evidence = MOD12.fit_association(
                self.training_data,
                condition="Visual",
                target="mental_demand_score_0_to_10",
            )

        self.assertEqual(
            evidence["retention_status"],
            "not_retained",
        )

    def test_failed_fit_is_not_evaluable(self):
        with patch.object(
            MOD12.MOD02,
            "fit_mixedlm_with_fallback",
            return_value=(
                None,
                "",
                (),
                ("lbfgs: fit failed",),
            ),
        ):
            evidence = MOD12.fit_association(
                self.training_data,
                condition="Visual",
                target="mental_demand_score_0_to_10",
            )

        self.assertEqual(
            evidence["convergence_status"],
            "failed",
        )

        self.assertEqual(
            evidence["retention_status"],
            "not_evaluable",
        )

        self.assertTrue(
            np.isnan(
                evidence["p_value"]
            )
        )

    def test_nonconverged_fit_is_not_evaluable(self):
        fake_result = _FakeAssociationResult(
            "difficulty_stage",
            p_value=0.001,
            converged=False,
        )

        with patch.object(
            MOD12.MOD02,
            "fit_mixedlm_with_fallback",
            return_value=(
                fake_result,
                "powell",
                ("did not converge",),
                (),
            ),
        ):
            evidence = MOD12.fit_association(
                self.training_data,
                condition="Visual",
                target="mental_demand_score_0_to_10",
            )

        self.assertEqual(
            evidence["convergence_status"],
            "non_converged",
        )

        self.assertEqual(
            evidence["retention_status"],
            "not_evaluable",
        )

    def test_warnings_and_errors_use_one_column(self):
        fake_result = _FakeAssociationResult(
            "difficulty_stage",
        )

        with patch.object(
            MOD12.MOD02,
            "fit_mixedlm_with_fallback",
            return_value=(
                fake_result,
                "lbfgs",
                ("warning one",),
                ("powell error",),
            ),
        ):
            evidence = MOD12.fit_association(
                self.training_data,
                condition="Visual",
                target="mental_demand_score_0_to_10",
            )

        self.assertEqual(
            evidence["warnings_errors"],
            "warning one | powell error",
        )

class TrainingBoundaryTests(unittest.TestCase):
    def test_only_26_training_participants_enter_mod12(self):
        training_data, training_ids, test_ids = (
            MOD12.prepare_training_data(
                modeling_data=synthetic_modeling_data(),
                holdout_split=synthetic_holdout_split(),
            )
        )

        self.assertEqual(len(training_ids), 26)
        self.assertEqual(len(test_ids), 6)

        self.assertEqual(
            training_data["participant_id"].nunique(),
            26,
        )

        self.assertTrue(
            set(training_data["participant_id"]).isdisjoint(
                test_ids
            )
        )

    def test_test_participant_values_cannot_change_training_data(self):
        data = synthetic_modeling_data()
        split = synthetic_holdout_split()

        baseline, _, test_ids = MOD12.prepare_training_data(
            modeling_data=data,
            holdout_split=split,
        )

        changed = data.copy()

        test_mask = changed["participant_id"].isin(test_ids)

        measurement_columns = [
            target
            for target in MOD12.TARGETS
        ]

        changed.loc[
            test_mask,
            measurement_columns,
        ] = 999999.0

        repeated, _, _ = MOD12.prepare_training_data(
            modeling_data=changed,
            holdout_split=split,
        )

        pd.testing.assert_frame_equal(
            baseline.reset_index(drop=True),
            repeated.reset_index(drop=True),
        )


class ModelSpecificationTests(unittest.TestCase):
    def test_all_10_measurements_are_registered(self):
        specs = MOD12.build_target_specs()

        self.assertEqual(len(specs), 10)
        self.assertEqual(set(specs), set(MOD12.TARGETS))

    def test_every_measurement_uses_m0_only(self):
        specs = MOD12.build_target_specs()

        for target, spec in specs.items():
            expected_formula = (
                f"{target} ~ {spec.difficulty_column}"
            )

            self.assertEqual(
                spec.formula,
                expected_formula,
            )

    def test_random_structure_is_ri_only(self):
        self.assertEqual(
            MOD12.RANDOM_STRUCTURE,
            "RI",
        )

    def test_relative_performance_uses_post_d0_stage(self):
        spec = MOD12.build_target_specs()[
            MOD12.PERFORMANCE_TARGET
        ]

        self.assertEqual(
            spec.difficulty_column,
            "performance_difficulty_stage",
        )

    def test_other_measurements_use_four_stage_difficulty(self):
        specs = MOD12.build_target_specs()

        for target, spec in specs.items():
            if target == MOD12.PERFORMANCE_TARGET:
                continue

            self.assertEqual(
                spec.difficulty_column,
                "difficulty_stage",
            )


class TargetPreparationTests(unittest.TestCase):
    def setUp(self):
        self.training_data, _, _ = (
            MOD12.prepare_training_data(
                modeling_data=synthetic_modeling_data(),
                holdout_split=synthetic_holdout_split(),
            )
        )

    def test_relative_performance_excludes_d0(self):
        frame = MOD12.prepare_target_frame(
            self.training_data,
            condition="Visual",
            target=MOD12.PERFORMANCE_TARGET,
        )

        self.assertEqual(
            set(frame["difficulty_level"]),
            {2, 6, 10},
        )

        self.assertEqual(
            set(frame["performance_difficulty_stage"]),
            {0.0, 1.0, 2.0},
        )

        self.assertEqual(
            frame["participant_id"].nunique(),
            26,
        )

        self.assertEqual(len(frame), 78)

    def test_other_measurements_keep_all_four_stages(self):
        frame = MOD12.prepare_target_frame(
            self.training_data,
            condition="Visual",
            target="mental_demand_score_0_to_10",
        )

        self.assertEqual(
            set(frame["difficulty_stage"]),
            {0, 1, 2, 3},
        )

        self.assertEqual(
            frame["participant_id"].nunique(),
            26,
        )

        self.assertEqual(len(frame), 104)

    def test_missing_measurement_removes_only_that_row(self):
        frame = MOD12.prepare_target_frame(
            self.training_data,
            condition="Visual",
            target="median_reach_duration_seconds",
        )

        self.assertEqual(len(frame), 103)

        self.assertEqual(
            frame["participant_id"].nunique(),
            26,
        )

    def test_genuine_zero_is_preserved(self):
        frame = MOD12.prepare_target_frame(
            self.training_data,
            condition="Visual",
            target="list_recheck_count",
        )

        row = frame.loc[
            frame["participant_id"].eq("Y04")
            & frame["difficulty_level"].eq(2)
        ]

        self.assertEqual(len(row), 1)

        self.assertEqual(
            float(row.iloc[0]["list_recheck_count"]),
            0.0,
        )


class FinalOutputTests(unittest.TestCase):
    def setUp(self):
        self.training_data, _, _ = (
            MOD12.prepare_training_data(
                modeling_data=synthetic_modeling_data(),
                holdout_split=synthetic_holdout_split(),
            )
        )

    def test_runs_exactly_30_associations(self):
        def fake_fit(
            training_data,
            *,
            condition,
            target,
        ):
            return {
                "condition": condition,
                "measurement": target,
                "difficulty_coefficient": 0.5,
                "standard_error": 0.1,
                "ci_95_lower": 0.3,
                "ci_95_upper": 0.7,
                "p_value": 0.01,
                "participant_count": 26,
                "observation_count": 104,
                "convergence_status": "converged",
                "optimizer": "lbfgs",
                "warnings_errors": "",
                "retention_status": "retained",
            }

        with patch.object(
            MOD12,
            "fit_association",
            side_effect=fake_fit,
        ) as fitter:
            evidence = MOD12.run_all_associations(
                self.training_data
            )

        self.assertEqual(
            fitter.call_count,
            30,
        )

        self.assertEqual(
            len(evidence),
            30,
        )

        self.assertEqual(
            set(evidence["condition"]),
            {"Visual", "Auditory", "Cognitive"},
        )

    def test_evidence_has_only_required_columns(self):
        row = {
            "condition": "Visual",
            "measurement": "mental_demand_score_0_to_10",
            "difficulty_coefficient": 0.5,
            "standard_error": 0.1,
            "ci_95_lower": 0.3,
            "ci_95_upper": 0.7,
            "p_value": 0.01,
            "participant_count": 26,
            "observation_count": 104,
            "convergence_status": "converged",
            "optimizer": "lbfgs",
            "warnings_errors": "",
            "retention_status": "retained",
        }

        with patch.object(
            MOD12,
            "fit_association",
            return_value=row,
        ):
            evidence = MOD12.run_all_associations(
                self.training_data
            )

        self.assertEqual(
            list(evidence.columns),
            list(MOD12.EVIDENCE_COLUMNS),
        )

    def test_retained_features_include_only_retained_rows(self):
        evidence = pd.DataFrame(
            [
                {
                    "condition": "Visual",
                    "measurement": "mental_demand_score_0_to_10",
                    "retention_status": "retained",
                },
                {
                    "condition": "Visual",
                    "measurement": "median_reach_duration_seconds",
                    "retention_status": "not_retained",
                },
                {
                    "condition": "Auditory",
                    "measurement": MOD12.PERFORMANCE_TARGET,
                    "retention_status": "retained",
                },
            ]
        )

        retained = MOD12.build_retained_features(
            evidence
        )

        self.assertEqual(len(retained), 2)

        self.assertNotIn(
            "median_reach_duration_seconds",
            set(retained["measurement"]),
        )

    def test_relative_performance_is_later_only(self):
        evidence = pd.DataFrame(
            [
                {
                    "condition": "Visual",
                    "measurement": MOD12.PERFORMANCE_TARGET,
                    "retention_status": "retained",
                },
                {
                    "condition": "Visual",
                    "measurement": "mental_demand_score_0_to_10",
                    "retention_status": "retained",
                },
            ]
        )

        retained = MOD12.build_retained_features(
            evidence
        )

        performance = retained.loc[
            retained["measurement"].eq(
                MOD12.PERFORMANCE_TARGET
            )
        ].iloc[0]

        mental_demand = retained.loc[
            retained["measurement"].eq(
                "mental_demand_score_0_to_10"
            )
        ].iloc[0]

        self.assertEqual(
            performance["prediction_stage"],
            "later_only",
        )

        self.assertEqual(
            mental_demand["prediction_stage"],
            "initial_and_later",
        )

    def test_output_writer_refuses_overwrite(self):
        evidence = pd.DataFrame(
            columns=MOD12.EVIDENCE_COLUMNS
        )

        retained = pd.DataFrame(
            columns=MOD12.RETAINED_COLUMNS
        )

        with TemporaryDirectory() as directory:
            output_dir = Path(directory)

            MOD12.write_outputs(
                evidence=evidence,
                retained=retained,
                output_dir=output_dir,
            )

            with self.assertRaises(MOD12.Mod12Error):
                MOD12.write_outputs(
                    evidence=evidence,
                    retained=retained,
                    output_dir=output_dir,
                )

if __name__ == "__main__":
    unittest.main()