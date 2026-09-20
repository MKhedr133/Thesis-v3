"""Synthetic tests for MOD-11 training-only bootstrap comparison."""

from __future__ import annotations

from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
import tempfile

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
PYTHON_DIR = REPOSITORY_ROOT / "python"
sys.path.insert(0, str(PYTHON_DIR))

MODULE_PATH = PYTHON_DIR / "MOD_11_training_bootstrap_comparison.py"

SPEC = spec_from_file_location(
    "mod11_training_bootstrap_test",
    MODULE_PATH,
)

if SPEC is None or SPEC.loader is None:
    raise ImportError(MODULE_PATH)

MOD11 = module_from_spec(SPEC)
sys.modules[SPEC.name] = MOD11
SPEC.loader.exec_module(MOD11)


def synthetic_modeling_data() -> pd.DataFrame:
    """Create four repeated rows for each of 32 synthetic participants."""
    rows: list[dict[str, object]] = []

    for group, prefix in (("Young", "Y"), ("Old", "O")):
        for number in range(1, 17):
            participant_id = f"{prefix}{number:02d}"

            for difficulty_level, difficulty_stage in (
                (0, 0),
                (2, 1),
                (6, 2),
                (10, 3),
            ):
                rows.append(
                    {
                        "participant_id": participant_id,
                        "participant_group": group,
                        "difficulty_level": difficulty_level,
                        "difficulty_stage": difficulty_stage,
                        "tmt_b_seconds": float(
                            40
                            + number
                            + (20 if group == "Old" else 0)
                        ),
                    }
                )

    return pd.DataFrame(rows)


def synthetic_holdout_split() -> pd.DataFrame:
    """Create the synthetic equivalent of the locked MOD-09 26/6 split."""
    rows: list[dict[str, str]] = []

    for group, prefix in (("Young", "Y"), ("Old", "O")):
        for number in range(1, 17):
            participant_id = f"{prefix}{number:02d}"

            split = (
                "test"
                if number in {1, 2, 3}
                else "train"
            )

            rows.append(
                {
                    "participant_id": participant_id,
                    "participant_group": group,
                    "split": split,
                }
            )

    return pd.DataFrame(rows)


class ModelContractTests(unittest.TestCase):
    def test_registry_contains_exactly_seven_frozen_models(self):
        registry = MOD11.build_candidate_model_rhs_registry()

        self.assertEqual(
            tuple(registry),
            (
                "M0",
                "MA",
                "MT",
                "MAT",
                "MDA",
                "MDT",
                "MDAT",
            ),
        )

    def test_registry_matches_frozen_formulas(self):
        registry = MOD11.build_candidate_model_rhs_registry()

        age = (
            "C(participant_group, "
            "Treatment(reference='Young'))"
        )

        expected = {
            "M0": "{D}",
            "MA": f"{{D}} + {age}",
            "MT": "{D} + tmt_b_seconds",
            "MAT": f"{{D}} + {age} + tmt_b_seconds",
            "MDA": (
                f"{{D}} + {age} + "
                f"{{D}}:{age}"
            ),
            "MDT": (
                "{D} + tmt_b_seconds + "
                "{D}:tmt_b_seconds"
            ),
            "MDAT": (
                f"{{D}} + {age} + tmt_b_seconds + "
                f"{{D}}:{age} + "
                "{D}:tmt_b_seconds"
            ),
        }

        self.assertEqual(registry, expected)

    def test_tmt_models_use_raw_tmt_b_seconds(self):
        registry = MOD11.build_candidate_model_rhs_registry()

        for model_id in ("MT", "MAT", "MDT", "MDAT"):
            rhs = registry[model_id]

            self.assertIn("tmt_b_seconds", rhs)
            self.assertNotIn("tmt_b_z", rhs)
            self.assertNotIn("standard", rhs.lower())

    def test_random_effects_are_frozen_to_ri_only(self):
        self.assertEqual(
            MOD11.RANDOM_STRUCTURES,
            ("RI",),
        )

    def test_final_bootstrap_target_is_500(self):
        self.assertEqual(
            MOD11.DEFAULT_BOOTSTRAP_REPLICATES,
            500,
        )


class TrainingBoundaryTests(unittest.TestCase):
    def test_prepare_training_data_excludes_all_test_participants(self):
        training_data, training_ids, test_ids = (
            MOD11.prepare_training_data(
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

        self.assertEqual(
            set(training_data["participant_id"]),
            set(training_ids),
        )


class BootstrapSamplingTests(unittest.TestCase):
    def setUp(self):
        (
            self.training_data,
            self.training_ids,
            self.test_ids,
        ) = MOD11.prepare_training_data(
            modeling_data=synthetic_modeling_data(),
            holdout_split=synthetic_holdout_split(),
        )

    def test_participant_draw_has_exactly_26_draws(self):
        sampled = MOD11.draw_participant_ids(
            self.training_ids,
            rng=np.random.default_rng(123),
        )

        self.assertEqual(len(sampled), 26)

        self.assertTrue(
            set(sampled).issubset(
                set(self.training_ids)
            )
        )

    def test_participant_sampling_is_with_replacement(self):
        sampled = MOD11.draw_participant_ids(
            self.training_ids,
            rng=np.random.default_rng(123),
        )

        self.assertLess(
            len(set(sampled)),
            26,
        )

    def test_final_test_ids_can_never_enter_bootstrap_draw(self):
        sampled = MOD11.draw_participant_ids(
            self.training_ids,
            rng=np.random.default_rng(123),
        )

        self.assertTrue(
            set(sampled).isdisjoint(
                self.test_ids
            )
        )

    def test_duplicate_draws_receive_distinct_cluster_ids(self):
        sampled = (
            *self.training_ids[:-1],
            self.training_ids[0],
        )

        bootstrap = MOD11.build_bootstrap_sample(
            self.training_data,
            sampled,
            replicate_index=7,
        )

        self.assertEqual(
            bootstrap[
                MOD11.BOOTSTRAP_CLUSTER_COLUMN
            ].nunique(),
            26,
        )

        duplicated = bootstrap.loc[
            bootstrap["participant_id"].eq(
                self.training_ids[0]
            )
        ]

        self.assertEqual(
            duplicated[
                MOD11.BOOTSTRAP_CLUSTER_COLUMN
            ].nunique(),
            2,
        )

    def test_each_bootstrap_cluster_preserves_all_repeated_rows(self):
        sampled = (
            *self.training_ids[:-1],
            self.training_ids[0],
        )

        bootstrap = MOD11.build_bootstrap_sample(
            self.training_data,
            sampled,
            replicate_index=7,
        )

        rows_per_cluster = (
            bootstrap
            .groupby(MOD11.BOOTSTRAP_CLUSTER_COLUMN)
            .size()
        )

        self.assertTrue(
            rows_per_cluster.eq(4).all()
        )

    def test_oob_ids_are_training_participants_not_sampled(self):
        omitted = self.training_ids[-1]

        sampled = (
            *self.training_ids[:-1],
            self.training_ids[0],
        )

        oob = MOD11.identify_oob_participants(
            self.training_ids,
            sampled,
        )

        self.assertEqual(
            oob,
            (omitted,),
        )

        self.assertTrue(
            set(oob).isdisjoint(self.test_ids)
        )


class BootstrapPlanTests(unittest.TestCase):
    def setUp(self):
        _, self.training_ids, _ = (
            MOD11.prepare_training_data(
                modeling_data=synthetic_modeling_data(),
                holdout_split=synthetic_holdout_split(),
            )
        )

    def test_plan_contains_requested_number_of_replicates(self):
        plan = MOD11.generate_bootstrap_plan(
            self.training_ids,
            n_replicates=5,
            seed=12345,
        )

        self.assertEqual(len(plan), 5)

        for sampled in plan:
            self.assertEqual(len(sampled), 26)

    def test_plan_is_reproducible_for_same_seed(self):
        plan_a = MOD11.generate_bootstrap_plan(
            self.training_ids,
            n_replicates=5,
            seed=12345,
        )

        plan_b = MOD11.generate_bootstrap_plan(
            self.training_ids,
            n_replicates=5,
            seed=12345,
        )

        self.assertEqual(plan_a, plan_b)

    def test_different_seed_changes_plan(self):
        plan_a = MOD11.generate_bootstrap_plan(
            self.training_ids,
            n_replicates=5,
            seed=12345,
        )

        plan_b = MOD11.generate_bootstrap_plan(
            self.training_ids,
            n_replicates=5,
            seed=54321,
        )

        self.assertNotEqual(plan_a, plan_b)

    def test_one_plan_can_be_reused_for_all_models(self):
        plan = MOD11.generate_bootstrap_plan(
            self.training_ids,
            n_replicates=3,
            seed=12345,
        )

        candidate_plan_views = {
            model_id: plan
            for model_id in MOD11.MODEL_IDS
        }

        for model_id in MOD11.MODEL_IDS:
            self.assertIs(
                candidate_plan_views[model_id],
                plan,
            )


def synthetic_modeling_data_with_outcomes() -> pd.DataFrame:
    """Extend the Batch-1 synthetic data with two modelling outcomes."""
    data = synthetic_modeling_data()

    data["condition_name"] = "Visual"

    data["mental_demand_score_0_to_10"] = (
        1.0
        + data["difficulty_stage"].astype(float)
        + data["participant_group"].eq("Old").astype(float) * 0.5
    )

    # D0 is a structural reference and will be excluded by MOD-11.
    data["performance_change_from_d0_percentage_points"] = (
        data["difficulty_stage"].astype(float) * 5.0
    )

    return data


class FakeMixedLMResult:
    """Minimal converged result used to test MOD-11 orchestration."""

    def __init__(
        self,
        *,
        prediction_value: float = 5.0,
        llf: float = -100.0,
        aic: float = 210.0,
        bic: float = 220.0,
    ):
        self.converged = True
        self.llf = llf
        self.aic = aic
        self.bic = bic
        self.fe_params = pd.Series(
            [1.0, 2.0],
            index=["Intercept", "difficulty"],
        )
        self.prediction_value = prediction_value
        self.prediction_frames: list[pd.DataFrame] = []

    def predict(self, frame: pd.DataFrame):
        self.prediction_frames.append(frame.copy())

        return np.full(
            len(frame),
            self.prediction_value,
            dtype=float,
        )


def fake_mixedlm_fitter(
    formula,
    data,
    random_structure,
    difficulty_column,
    *,
    participant_column="participant_id",
    **kwargs,
):
    """Return a deterministic fake fit with MOD-02-compatible output."""
    return (
        FakeMixedLMResult(),
        "lbfgs",
        (),
        (),
    )


class TargetPreparationTests(unittest.TestCase):
    def setUp(self):
        self.training_data, self.training_ids, _ = (
            MOD11.prepare_training_data(
                modeling_data=synthetic_modeling_data_with_outcomes(),
                holdout_split=synthetic_holdout_split(),
            )
        )

    def test_performance_excludes_d0_and_uses_post_d0_stage(self):
        frame, difficulty_column = MOD11.prepare_target_frame(
            self.training_data,
            condition="Visual",
            target=(
                "performance_change_from_d0_percentage_points"
            ),
        )

        self.assertEqual(
            difficulty_column,
            "performance_difficulty_stage",
        )

        self.assertEqual(len(frame), 26 * 3)

        self.assertFalse(
            frame["difficulty_level"].eq(0).any()
        )

        self.assertEqual(
            set(frame[difficulty_column]),
            {0.0, 1.0, 2.0},
        )

    def test_mental_demand_keeps_all_four_stages(self):
        frame, difficulty_column = MOD11.prepare_target_frame(
            self.training_data,
            condition="Visual",
            target="mental_demand_score_0_to_10",
        )

        self.assertEqual(
            difficulty_column,
            "difficulty_stage",
        )

        self.assertEqual(len(frame), 26 * 4)

        self.assertEqual(
            set(frame[difficulty_column]),
            {0, 1, 2, 3},
        )

    def test_candidate_formula_uses_target_specific_difficulty(self):
        formula = MOD11.build_candidate_formula(
            target="mental_demand_score_0_to_10",
            model_id="MDT",
        )

        self.assertEqual(
            formula,
            (
                "mental_demand_score_0_to_10 ~ "
                "difficulty_stage + tmt_b_seconds + "
                "difficulty_stage:tmt_b_seconds"
            ),
        )


class OriginalTrainingFitTests(unittest.TestCase):
    def setUp(self):
        self.training_data, _, _ = (
            MOD11.prepare_training_data(
                modeling_data=synthetic_modeling_data_with_outcomes(),
                holdout_split=synthetic_holdout_split(),
            )
        )

    def test_original_fit_uses_ri_and_original_participant_clusters(self):
        with patch.object(
            MOD11.MOD02,
            "fit_mixedlm_with_fallback",
            side_effect=fake_mixedlm_fitter,
        ) as mocked:
            evidence = MOD11.fit_original_training_models(
                self.training_data,
                conditions=("Visual",),
                targets=("mental_demand_score_0_to_10",),
                model_ids=("M0", "MT"),
            )

        self.assertEqual(len(evidence), 2)

        self.assertTrue(
            evidence["convergence_status"]
            .eq("converged")
            .all()
        )

        self.assertTrue(
            evidence["random_structure"]
            .eq("RI")
            .all()
        )

        self.assertTrue(
            evidence["reml"]
            .eq(False)
            .all()
        )

        self.assertTrue(
            evidence["participant_count"]
            .eq(26)
            .all()
        )

        self.assertTrue(
            evidence["aic"]
            .eq(210.0)
            .all()
        )

        self.assertTrue(
            evidence["bic"]
            .eq(220.0)
            .all()
        )

        self.assertEqual(mocked.call_count, 2)

        for call in mocked.call_args_list:
            args = call.args
            kwargs = call.kwargs

            self.assertEqual(
                args[2],
                "RI",
            )

            self.assertEqual(
                kwargs["participant_column"],
                "participant_id",
            )

    def test_original_mt_fit_uses_raw_tmt_seconds(self):
        with patch.object(
            MOD11.MOD02,
            "fit_mixedlm_with_fallback",
            side_effect=fake_mixedlm_fitter,
        ) as mocked:
            MOD11.fit_original_training_models(
                self.training_data,
                conditions=("Visual",),
                targets=("mental_demand_score_0_to_10",),
                model_ids=("MT",),
            )

        formula = mocked.call_args.args[0]

        self.assertIn(
            "tmt_b_seconds",
            formula,
        )

        self.assertNotIn(
            "tmt_b_z",
            formula,
        )


class BootstrapOobPredictionTests(unittest.TestCase):
    def setUp(self):
        (
            self.training_data,
            self.training_ids,
            self.test_ids,
        ) = MOD11.prepare_training_data(
            modeling_data=synthetic_modeling_data_with_outcomes(),
            holdout_split=synthetic_holdout_split(),
        )

        # Leave the final training participant OOB and duplicate the first.
        self.omitted = self.training_ids[-1]

        self.sampled = (
            *self.training_ids[:-1],
            self.training_ids[0],
        )

    def test_bootstrap_fit_groups_by_bootstrap_cluster(self):
        with patch.object(
            MOD11.MOD02,
            "fit_mixedlm_with_fallback",
            side_effect=fake_mixedlm_fitter,
        ) as mocked:
            predictions, fits = (
                MOD11.run_bootstrap_replicate(
                    self.training_data,
                    training_participant_ids=self.training_ids,
                    sampled_participant_ids=self.sampled,
                    replicate_index=7,
                    conditions=("Visual",),
                    targets=("mental_demand_score_0_to_10",),
                    model_ids=("M0",),
                )
            )

        self.assertEqual(mocked.call_count, 1)

        call = mocked.call_args

        self.assertEqual(
            call.args[2],
            "RI",
        )

        self.assertEqual(
            call.kwargs["participant_column"],
            MOD11.BOOTSTRAP_CLUSTER_COLUMN,
        )

        bootstrap_frame = call.args[1]

        self.assertEqual(
            bootstrap_frame[
                MOD11.BOOTSTRAP_CLUSTER_COLUMN
            ].nunique(),
            26,
        )

        self.assertEqual(len(fits), 1)

    def test_oob_predictions_contain_only_omitted_participant(self):
        with patch.object(
            MOD11.MOD02,
            "fit_mixedlm_with_fallback",
            side_effect=fake_mixedlm_fitter,
        ):
            predictions, _ = MOD11.run_bootstrap_replicate(
                self.training_data,
                training_participant_ids=self.training_ids,
                sampled_participant_ids=self.sampled,
                replicate_index=7,
                conditions=("Visual",),
                targets=("mental_demand_score_0_to_10",),
                model_ids=("M0",),
            )

        self.assertEqual(len(predictions), 4)

        self.assertEqual(
            set(predictions["participant_id"]),
            {self.omitted},
        )

        self.assertTrue(
            set(predictions["participant_id"])
            .isdisjoint(self.test_ids)
        )

        self.assertTrue(
            predictions["predicted"]
            .eq(5.0)
            .all()
        )

        self.assertTrue(
            np.isfinite(
                predictions["absolute_error"]
            ).all()
        )

        self.assertTrue(
            np.isfinite(
                predictions["squared_error"]
            ).all()
        )

    def test_oob_prediction_uses_result_predict_only(self):
        """Fake result has no random-effect prediction API."""
        with patch.object(
            MOD11.MOD02,
            "fit_mixedlm_with_fallback",
            side_effect=fake_mixedlm_fitter,
        ):
            predictions, _ = MOD11.run_bootstrap_replicate(
                self.training_data,
                training_participant_ids=self.training_ids,
                sampled_participant_ids=self.sampled,
                replicate_index=7,
                conditions=("Visual",),
                targets=("mental_demand_score_0_to_10",),
                model_ids=("M0",),
            )

        self.assertEqual(
            predictions["prediction_scope"].unique().tolist(),
            ["fixed_effect_population_only"],
        )

    def test_performance_oob_prediction_excludes_d0(self):
        with patch.object(
            MOD11.MOD02,
            "fit_mixedlm_with_fallback",
            side_effect=fake_mixedlm_fitter,
        ):
            predictions, _ = MOD11.run_bootstrap_replicate(
                self.training_data,
                training_participant_ids=self.training_ids,
                sampled_participant_ids=self.sampled,
                replicate_index=7,
                conditions=("Visual",),
                targets=(
                    "performance_change_from_d0_percentage_points",
                ),
                model_ids=("M0",),
            )

        self.assertEqual(len(predictions), 3)

        self.assertFalse(
            predictions["difficulty_level"]
            .eq(0)
            .any()
        )

class CommonSupportTests(unittest.TestCase):
    def _fit_and_prediction_evidence(self):
        fit_rows = []
        prediction_rows = []

        for replicate in (1, 2):
            for model_id in MOD11.MODEL_IDS:
                failed = (
                    replicate == 2
                    and model_id == "MDAT"
                )

                fit_rows.append(
                    {
                        "bootstrap_replicate": replicate,
                        "condition_name": "Visual",
                        "target_name": "mental_demand_score_0_to_10",
                        "model_id": model_id,
                        "convergence_status": (
                            "failed" if failed else "converged"
                        ),
                        "prediction_status": (
                            "not_attempted_fit_failure"
                            if failed
                            else "success"
                        ),
                        "optimizer": "lbfgs",
                        "warnings": "",
                        "fit_errors": (
                            "synthetic failure" if failed else ""
                        ),
                    }
                )

                if not failed:
                    prediction_rows.append(
                        {
                            "bootstrap_replicate": replicate,
                            "condition_name": "Visual",
                            "target_name": "mental_demand_score_0_to_10",
                            "model_id": model_id,
                            "participant_id": "P01",
                            MOD11.ORIGINAL_ROW_ID_COLUMN: replicate,
                            "observed": 10.0,
                            "predicted": 9.0,
                            "absolute_error": 1.0,
                            "squared_error": 1.0,
                        }
                    )

        return (
            pd.DataFrame(fit_rows),
            pd.DataFrame(prediction_rows),
        )

    def test_replicate_is_eligible_only_when_all_models_succeed(self):
        fits, predictions = self._fit_and_prediction_evidence()

        eligibility = MOD11.build_comparison_eligibility(
            fits,
            predictions,
        )

        replicate_1 = eligibility.loc[
            eligibility["bootstrap_replicate"].eq(1)
        ].iloc[0]

        replicate_2 = eligibility.loc[
            eligibility["bootstrap_replicate"].eq(2)
        ].iloc[0]

        self.assertTrue(
            bool(replicate_1["comparison_eligible"])
        )

        self.assertFalse(
            bool(replicate_2["comparison_eligible"])
        )

    def test_failed_replicate_is_excluded_for_all_models(self):
        fits, predictions = self._fit_and_prediction_evidence()

        eligibility = MOD11.build_comparison_eligibility(
            fits,
            predictions,
        )

        filtered = MOD11.filter_to_comparison_eligible_predictions(
            predictions,
            eligibility,
        )

        self.assertEqual(
            set(filtered["bootstrap_replicate"]),
            {1},
        )

        self.assertEqual(
            set(filtered["model_id"]),
            set(MOD11.MODEL_IDS),
        )


class LossFirstMetricTests(unittest.TestCase):
    def test_mae_calculates_errors_before_averaging(self):
        predictions = pd.DataFrame(
            {
                "participant_id": ["P01", "P01"],
                "bootstrap_replicate": [1, 2],
                "absolute_error": [4.0, 4.0],
            }
        )

        mae = MOD11.participant_balanced_loss_first_mae(
            predictions,
            expected_participants=("P01",),
        )

        # True value 10 with predictions 6 and 14:
        # prediction-first error would be 0,
        # but loss-first MAE must be 4.
        self.assertAlmostEqual(mae, 4.0)

    def test_pooled_r2_uses_mean_squared_loss_before_pooling(self):
        original = pd.DataFrame(
            {
                MOD11.ORIGINAL_ROW_ID_COLUMN: [1, 2],
                "mental_demand_score_0_to_10": [0.0, 10.0],
            }
        )

        predictions = pd.DataFrame(
            [
                {
                    MOD11.ORIGINAL_ROW_ID_COLUMN: 1,
                    "bootstrap_replicate": 1,
                    "squared_error": 0.0,
                },
                {
                    MOD11.ORIGINAL_ROW_ID_COLUMN: 1,
                    "bootstrap_replicate": 2,
                    "squared_error": 0.0,
                },
                {
                    MOD11.ORIGINAL_ROW_ID_COLUMN: 2,
                    "bootstrap_replicate": 1,
                    "squared_error": 16.0,
                },
                {
                    MOD11.ORIGINAL_ROW_ID_COLUMN: 2,
                    "bootstrap_replicate": 2,
                    "squared_error": 16.0,
                },
            ]
        )

        r2 = MOD11.pooled_loss_first_oob_r2(
            predictions,
            original_frame=original,
            target="mental_demand_score_0_to_10",
        )

        # SST = (0 - 5)^2 + (10 - 5)^2 = 50
        # loss-first SSE = 0 + 16 = 16
        # R2 = 1 - 16/50 = 0.68
        self.assertAlmostEqual(r2, 0.68)

    def test_negative_oob_r2_is_preserved(self):
        original = pd.DataFrame(
            {
                MOD11.ORIGINAL_ROW_ID_COLUMN: [1, 2],
                "mental_demand_score_0_to_10": [0.0, 1.0],
            }
        )

        predictions = pd.DataFrame(
            {
                MOD11.ORIGINAL_ROW_ID_COLUMN: [1, 2],
                "bootstrap_replicate": [1, 1],
                "squared_error": [100.0, 100.0],
            }
        )

        r2 = MOD11.pooled_loss_first_oob_r2(
            predictions,
            original_frame=original,
            target="mental_demand_score_0_to_10",
        )

        self.assertLess(r2, 0.0)


class DeltaConventionTests(unittest.TestCase):
    def test_all_m0_delta_signs_follow_frozen_convention(self):
        oob_summary = pd.DataFrame(
            [
                {
                    "condition_name": "Visual",
                    "target_name": "mental_demand_score_0_to_10",
                    "model_id": "M0",
                    "participant_balanced_oob_mae": 2.0,
                    "pooled_oob_r2": 0.20,
                },
                {
                    "condition_name": "Visual",
                    "target_name": "mental_demand_score_0_to_10",
                    "model_id": "MT",
                    "participant_balanced_oob_mae": 1.5,
                    "pooled_oob_r2": 0.30,
                },
            ]
        )

        original_fits = pd.DataFrame(
            [
                {
                    "condition_name": "Visual",
                    "target_name": "mental_demand_score_0_to_10",
                    "model_id": "M0",
                    "aic": 100.0,
                    "bic": 110.0,
                    "convergence_status": "converged",
                    "optimizer": "lbfgs",
                    "warnings": "",
                    "fit_errors": "",
                },
                {
                    "condition_name": "Visual",
                    "target_name": "mental_demand_score_0_to_10",
                    "model_id": "MT",
                    "aic": 95.0,
                    "bic": 108.0,
                    "convergence_status": "converged",
                    "optimizer": "lbfgs",
                    "warnings": "",
                    "fit_errors": "",
                },
            ]
        )

        combined = MOD11.combine_comparison_evidence(
            oob_summary,
            original_fits,
        )

        mt = combined.loc[
            combined["model_id"].eq("MT")
        ].iloc[0]

        self.assertAlmostEqual(
            mt["delta_mae_vs_m0"],
            0.5,
        )

        self.assertAlmostEqual(
            mt["delta_oob_r2_vs_m0"],
            0.10,
        )

        self.assertAlmostEqual(
            mt["delta_aic_vs_m0"],
            -5.0,
        )

        self.assertAlmostEqual(
            mt["delta_bic_vs_m0"],
            -2.0,
        )

        self.assertEqual(
            mt["selection_status"],
            "evidence_only_no_automatic_selection",
        )


class BootstrapCheckpointTests(unittest.TestCase):
    def test_completed_replicates_are_reused_from_checkpoint(self):
        training_data, training_ids, _ = (
            MOD11.prepare_training_data(
                modeling_data=synthetic_modeling_data_with_outcomes(),
                holdout_split=synthetic_holdout_split(),
            )
        )

        plan = MOD11.generate_bootstrap_plan(
            training_ids,
            n_replicates=2,
            seed=123,
        )

        fake_predictions = pd.DataFrame(
            [
                {
                    "bootstrap_replicate": 1,
                    "condition_name": "Visual",
                    "target_name": "mental_demand_score_0_to_10",
                    "model_id": "M0",
                }
            ]
        )

        fake_fits = pd.DataFrame(
            [
                {
                    "bootstrap_replicate": 1,
                    "condition_name": "Visual",
                    "target_name": "mental_demand_score_0_to_10",
                    "model_id": "M0",
                }
            ]
        )

        with tempfile.TemporaryDirectory() as tmp:
            checkpoint_dir = Path(tmp)

            with patch.object(
                MOD11,
                "run_bootstrap_replicate",
                return_value=(
                    fake_predictions,
                    fake_fits,
                ),
            ) as first_run:
                MOD11.run_bootstrap_plan(
                    training_data,
                    training_participant_ids=training_ids,
                    bootstrap_plan=plan,
                    conditions=("Visual",),
                    targets=("mental_demand_score_0_to_10",),
                    model_ids=("M0",),
                    jobs=1,
                    checkpoint_dir=checkpoint_dir,
                    resume=True,
                    progress=False,
                )

            self.assertEqual(
                first_run.call_count,
                2,
            )

            with patch.object(
                MOD11,
                "run_bootstrap_replicate",
            ) as second_run:
                MOD11.run_bootstrap_plan(
                    training_data,
                    training_participant_ids=training_ids,
                    bootstrap_plan=plan,
                    conditions=("Visual",),
                    targets=("mental_demand_score_0_to_10",),
                    model_ids=("M0",),
                    jobs=1,
                    checkpoint_dir=checkpoint_dir,
                    resume=True,
                    progress=False,
                )

            self.assertEqual(
                second_run.call_count,
                0,
            )

class BootstrapFitAuditTests(unittest.TestCase):
    def test_failure_count_counts_failed_attempts_without_negative_values(self):
        fits = pd.DataFrame(
            [
                {
                    "condition_name": "Visual",
                    "target_name": "mental_demand_score_0_to_10",
                    "model_id": "M0",
                    "convergence_status": "converged",
                    "prediction_status": "success",
                    "optimizer": "powell",
                    "warnings": "",
                    "fit_errors": "",
                },
                {
                    "condition_name": "Visual",
                    "target_name": "mental_demand_score_0_to_10",
                    "model_id": "M0",
                    "convergence_status": "failed",
                    "prediction_status": "not_attempted_fit_failure",
                    "optimizer": "",
                    "warnings": "",
                    "fit_errors": "synthetic failure",
                },
            ]
        )

        audit = MOD11.build_bootstrap_fit_audit(fits)

        row = audit.iloc[0]

        self.assertEqual(
            row["fit_attempt_count"],
            2,
        )

        self.assertEqual(
            row["prediction_success_count"],
            1,
        )

        self.assertEqual(
            row["fit_or_prediction_failure_count"],
            1,
        )

        self.assertGreaterEqual(
            row["fit_or_prediction_failure_count"],
            0,
        )


class EligibilityRowCoverageTests(unittest.TestCase):
    def test_mismatched_oob_rows_make_replicate_ineligible(self):
        fit_rows = []
        prediction_rows = []

        for model_id in MOD11.MODEL_IDS:
            fit_rows.append(
                {
                    "bootstrap_replicate": 1,
                    "condition_name": "Visual",
                    "target_name": "mental_demand_score_0_to_10",
                    "model_id": model_id,
                    "convergence_status": "converged",
                    "prediction_status": "success",
                }
            )

            row_id = (
                999
                if model_id == "MDAT"
                else 1
            )

            prediction_rows.append(
                {
                    "bootstrap_replicate": 1,
                    "condition_name": "Visual",
                    "target_name": "mental_demand_score_0_to_10",
                    "model_id": model_id,
                    MOD11.ORIGINAL_ROW_ID_COLUMN: row_id,
                }
            )

        eligibility = MOD11.build_comparison_eligibility(
            pd.DataFrame(fit_rows),
            pd.DataFrame(prediction_rows),
        )

        row = eligibility.iloc[0]

        self.assertFalse(
            bool(row["comparison_eligible"])
        )

        self.assertEqual(
            row["eligibility_reason"],
            "oob_prediction_row_mismatch",
        )


if __name__ == "__main__":
    unittest.main()