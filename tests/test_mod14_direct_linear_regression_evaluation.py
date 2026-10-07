"""Synthetic tests for MOD-14 direct linear regression evaluation."""

from __future__ import annotations

from importlib.util import (
    module_from_spec,
    spec_from_file_location,
)
from pathlib import Path
import sys
import unittest

import numpy as np
import pandas as pd

from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
PYTHON_DIR = ROOT / "python"

sys.path.insert(
    0,
    str(PYTHON_DIR),
)

MODULE_PATH = (
    PYTHON_DIR
    / "MOD_14_direct_linear_regression_evaluation.py"
)

SPEC = spec_from_file_location(
    "mod14_direct_linear_regression_evaluation_test",
    MODULE_PATH,
)

if SPEC is None or SPEC.loader is None:
    raise ImportError(MODULE_PATH)

MOD14 = module_from_spec(SPEC)

# Required when dataclasses are used in a module loaded this way.
sys.modules[SPEC.name] = MOD14

SPEC.loader.exec_module(MOD14)


def synthetic_holdout_split() -> pd.DataFrame:
    """Create the frozen-style 26-development / 6-test split."""
    rows: list[dict[str, str]] = []

    for group, prefix in (
        ("Young", "Y"),
        ("Old", "O"),
    ):
        for number in range(1, 17):
            participant_id = (
                f"{prefix}{number:02d}"
            )

            rows.append(
                {
                    "participant_id": participant_id,
                    "participant_group": group,
                    "split": (
                        "test"
                        if number in {1, 2, 3}
                        else "train"
                    ),
                }
            )

    return pd.DataFrame(rows)


def synthetic_modeling_data() -> pd.DataFrame:
    """Create synthetic repeated-trial data for all 32 participants."""
    rows: list[dict[str, object]] = []

    conditions = (
        "Visual",
        "Auditory",
        "Cognitive",
    )

    difficulties = (
        (0, 0),
        (2, 1),
        (6, 2),
        (10, 3),
    )

    for group, prefix in (
        ("Young", "Y"),
        ("Old", "O"),
    ):
        for number in range(1, 17):
            participant_id = (
                f"{prefix}{number:02d}"
            )

            for condition in conditions:
                for (
                    difficulty_level,
                    difficulty_stage,
                ) in difficulties:

                    base_value = float(
                        number
                        + difficulty_stage
                    )

                    rows.append(
                        {
                            "participant_id": (
                                participant_id
                            ),
                            "participant_group": group,
                            "condition_name": condition,
                            "difficulty_level": (
                                difficulty_level
                            ),
                            "difficulty_stage": (
                                difficulty_stage
                            ),
                            (
                                "performance_change_"
                                "from_d0_percentage_points"
                            ): float(
                                difficulty_stage * 5
                            ),
                            (
                                "mental_demand_score_"
                                "0_to_10"
                            ): base_value,
                            (
                                "median_time_between_"
                                "qualifying_grabs_seconds"
                            ): (
                                base_value + 0.1
                            ),
                            "list_recheck_count": float(
                                difficulty_stage
                            ),
                            (
                                "total_list_recheck_"
                                "duration_seconds"
                            ): (
                                base_value + 0.2
                            ),
                            (
                                "median_time_to_"
                                "target_seconds"
                            ): (
                                base_value + 0.3
                            ),
                            (
                                "median_irrelevant_"
                                "focus_duration_seconds"
                            ): (
                                base_value + 0.4
                            ),
                            (
                                "median_head_turning_"
                                "degrees"
                            ): (
                                base_value + 0.5
                            ),
                            (
                                "median_reach_"
                                "duration_seconds"
                            ): (
                                base_value + 0.6
                            ),
                            (
                                "median_reach_path_ratio"
                            ): (
                                1.0
                                + base_value / 100.0
                            ),
                        }
                    )

    data = pd.DataFrame(rows)

    # Structural baseline:
    # relative performance change is zero at D0.
    data.loc[
        data["difficulty_stage"].eq(0),
        (
            "performance_change_"
            "from_d0_percentage_points"
        ),
    ] = 0.0

    # One missing Visual predictor in a development participant.
    # Only this one model-specific row should be removed.
    missing_visual = (
        data["participant_id"].eq("Y04")
        & data["condition_name"].eq("Visual")
        & data["difficulty_level"].eq(6)
    )

    data.loc[
        missing_visual,
        "median_reach_duration_seconds",
    ] = np.nan

    return data


def regression_ready_modeling_data() -> pd.DataFrame:
    """Create deterministic non-collinear data for regression tests."""
    data = (
        synthetic_modeling_data()
        .copy()
    )

    rng = np.random.default_rng(
        12345
    )

    registry = (
        MOD14.build_model_registry()
    )

    predictors = sorted(
        {
            predictor
            for spec in registry.values()
            for predictor in spec.predictors
        }
    )

    stage = (
        data[
            "difficulty_stage"
        ]
        .astype(float)
        .to_numpy()
    )

    for index, predictor in enumerate(
        predictors
    ):
        noise = rng.normal(
            loc=0.0,
            scale=(
                1.0
                + index * 0.05
            ),
            size=len(data),
        )

        data[predictor] = (
            stage
            * (
                0.15
                + index * 0.03
            )
            + noise
        )

    # Relative performance remains structurally zero at D0.
    data.loc[
        data[
            "difficulty_stage"
        ].eq(0),
        MOD14.PERFORMANCE_CHANGE_PREDICTOR,
    ] = 0.0

    return data


def regression_ready_split_data():
    """Return regression-ready development/test data."""
    (
        development,
        test,
        development_ids,
        test_ids,
    ) = MOD14.prepare_split_data(
        modeling_data=(
            regression_ready_modeling_data()
        ),
        holdout_split=(
            synthetic_holdout_split()
        ),
    )

    return (
        development,
        test,
        development_ids,
        test_ids,
    )



class HoldoutSplitTests(
    unittest.TestCase
):
    def test_split_contains_exactly_26_development_and_6_test_participants(
        self,
    ):
        development_ids, test_ids = (
            MOD14.validate_holdout_split(
                synthetic_holdout_split()
            )
        )

        self.assertEqual(
            len(development_ids),
            26,
        )

        self.assertEqual(
            len(test_ids),
            6,
        )

        self.assertTrue(
            set(development_ids).isdisjoint(
                test_ids
            )
        )

    def test_invalid_split_size_is_rejected(
        self,
    ):
        split = synthetic_holdout_split()

        split.loc[
            split["participant_id"].eq("Y04"),
            "split",
        ] = "test"

        with self.assertRaises(
            MOD14.Mod14Error
        ):
            MOD14.validate_holdout_split(
                split
            )

    def test_development_and_test_data_are_separated_by_participant(
        self,
    ):
        (
            development,
            test,
            development_ids,
            test_ids,
        ) = (
            MOD14.prepare_split_data(
                modeling_data=(
                    synthetic_modeling_data()
                ),
                holdout_split=(
                    synthetic_holdout_split()
                ),
            )
        )

        self.assertEqual(
            len(development_ids),
            26,
        )

        self.assertEqual(
            len(test_ids),
            6,
        )

        self.assertEqual(
            development[
                "participant_id"
            ].nunique(),
            26,
        )

        self.assertEqual(
            test[
                "participant_id"
            ].nunique(),
            6,
        )

        self.assertTrue(
            set(
                development["participant_id"]
            ).isdisjoint(
                test["participant_id"]
            )
        )


class ModelRegistryTests(
    unittest.TestCase
):
    def test_target_coding_matches_mod13(
        self,
    ):
        self.assertEqual(
            MOD14.DIFFICULTY_STAGE_BY_LEVEL,
            {
                0: 0,
                2: 1,
                6: 2,
                10: 3,
            },
        )

    def test_registry_contains_exactly_four_models(
        self,
    ):
        registry = (
            MOD14.build_model_registry()
        )

        self.assertEqual(
            tuple(registry),
            (
                "visual",
                "auditory",
                "cognitive_primary",
                "cognitive_later",
            ),
        )

    def test_visual_model_is_frozen(
        self,
    ):
        spec = (
            MOD14
            .build_model_registry()[
                "visual"
            ]
        )

        self.assertEqual(
            spec.condition,
            "Visual",
        )

        self.assertEqual(
            spec.allowed_stages,
            (0, 1, 2, 3),
        )

        self.assertEqual(
            spec.predictors,
            (
                (
                    "mental_demand_score_"
                    "0_to_10"
                ),
                (
                    "median_time_between_"
                    "qualifying_grabs_seconds"
                ),
                (
                    "total_list_recheck_"
                    "duration_seconds"
                ),
                (
                    "median_time_to_"
                    "target_seconds"
                ),
                (
                    "median_irrelevant_"
                    "focus_duration_seconds"
                ),
                (
                    "median_head_turning_"
                    "degrees"
                ),
                (
                    "median_reach_"
                    "duration_seconds"
                ),
            ),
        )

    def test_auditory_model_is_frozen(
        self,
    ):
        spec = (
            MOD14
            .build_model_registry()[
                "auditory"
            ]
        )

        self.assertEqual(
            spec.condition,
            "Auditory",
        )

        self.assertEqual(
            spec.allowed_stages,
            (0, 1, 2, 3),
        )

        self.assertEqual(
            spec.predictors,
            (
                (
                    "mental_demand_score_"
                    "0_to_10"
                ),
                (
                    "median_time_between_"
                    "qualifying_grabs_seconds"
                ),
                (
                    "median_time_to_"
                    "target_seconds"
                ),
                (
                    "median_head_turning_"
                    "degrees"
                ),
            ),
        )

    def test_cognitive_primary_model_is_frozen(
        self,
    ):
        spec = (
            MOD14
            .build_model_registry()[
                "cognitive_primary"
            ]
        )

        self.assertEqual(
            spec.condition,
            "Cognitive",
        )

        self.assertEqual(
            spec.allowed_stages,
            (0, 1, 2, 3),
        )

        self.assertEqual(
            spec.predictors,
            (
                (
                    "mental_demand_score_"
                    "0_to_10"
                ),
                (
                    "median_time_between_"
                    "qualifying_grabs_seconds"
                ),
                "list_recheck_count",
                (
                    "total_list_recheck_"
                    "duration_seconds"
                ),
                (
                    "median_time_to_"
                    "target_seconds"
                ),
                (
                    "median_irrelevant_"
                    "focus_duration_seconds"
                ),
                (
                    "median_head_turning_"
                    "degrees"
                ),
                (
                    "median_reach_"
                    "duration_seconds"
                ),
                "median_reach_path_ratio",
            ),
        )

    def test_cognitive_later_adds_relative_performance(
        self,
    ):
        registry = (
            MOD14.build_model_registry()
        )

        primary = registry[
            "cognitive_primary"
        ]

        later = registry[
            "cognitive_later"
        ]

        self.assertEqual(
            later.condition,
            "Cognitive",
        )

        self.assertEqual(
            later.allowed_stages,
            (1, 2, 3),
        )

        self.assertEqual(
            later.predictors[:-1],
            primary.predictors,
        )

        self.assertEqual(
            later.predictors[-1],
            (
                "performance_change_"
                "from_d0_percentage_points"
            ),
        )


class ModelFrameTests(
    unittest.TestCase
):
    def setUp(self):
        (
            self.development,
            self.test,
            self.development_ids,
            self.test_ids,
        ) = (
            MOD14.prepare_split_data(
                modeling_data=(
                    synthetic_modeling_data()
                ),
                holdout_split=(
                    synthetic_holdout_split()
                ),
            )
        )

    def test_visual_frames_use_complete_model_specific_rows(
        self,
    ):
        development_frame = (
            MOD14.prepare_model_frame(
                self.development,
                model_id="visual",
            )
        )

        test_frame = (
            MOD14.prepare_model_frame(
                self.test,
                model_id="visual",
            )
        )

        # 26 × 4 = 104 development Visual trials,
        # minus the one deliberately missing required value.
        self.assertEqual(
            len(development_frame),
            103,
        )

        self.assertEqual(
            development_frame[
                "participant_id"
            ].nunique(),
            26,
        )

        # 6 × 4 = 24 test Visual trials.
        self.assertEqual(
            len(test_frame),
            24,
        )

        self.assertEqual(
            test_frame[
                "participant_id"
            ].nunique(),
            6,
        )

        self.assertFalse(
            development_frame
            .isna()
            .any()
            .any()
        )

        self.assertFalse(
            test_frame
            .isna()
            .any()
            .any()
        )

    def test_missing_unneeded_variable_does_not_remove_visual_row(
        self,
    ):
        changed = (
            self.development.copy()
        )

        mask = (
            changed[
                "participant_id"
            ].eq("Y05")
            & changed[
                "condition_name"
            ].eq("Visual")
            & changed[
                "difficulty_level"
            ].eq(2)
        )

        changed.loc[
            mask,
            "list_recheck_count",
        ] = np.nan

        frame = (
            MOD14.prepare_model_frame(
                changed,
                model_id="visual",
            )
        )

        # Still only the deliberately missing
        # Visual reach-duration row is excluded.
        self.assertEqual(
            len(frame),
            103,
        )

    def test_cognitive_primary_uses_all_four_stages(
        self,
    ):
        frame = (
            MOD14.prepare_model_frame(
                self.development,
                model_id="cognitive_primary",
            )
        )

        self.assertEqual(
            set(
                frame["difficulty_stage"]
            ),
            {0, 1, 2, 3},
        )

        self.assertEqual(
            len(frame),
            104,
        )

        self.assertNotIn(
            (
                "performance_change_"
                "from_d0_percentage_points"
            ),
            frame.columns,
        )

    def test_cognitive_later_uses_only_stages_1_2_3(
        self,
    ):
        frame = (
            MOD14.prepare_model_frame(
                self.development,
                model_id="cognitive_later",
            )
        )

        self.assertEqual(
            set(
                frame["difficulty_stage"]
            ),
            {1, 2, 3},
        )

        self.assertEqual(
            len(frame),
            78,
        )

        self.assertIn(
            (
                "performance_change_"
                "from_d0_percentage_points"
            ),
            frame.columns,
        )

    def test_real_zero_values_are_preserved(
        self,
    ):
        changed = (
            self.development.copy()
        )

        mask = (
            changed[
                "participant_id"
            ].eq("Y05")
            & changed[
                "condition_name"
            ].eq("Cognitive")
            & changed[
                "difficulty_level"
            ].eq(2)
        )

        changed.loc[
            mask,
            "list_recheck_count",
        ] = 0.0

        frame = (
            MOD14.prepare_model_frame(
                changed,
                model_id="cognitive_primary",
            )
        )

        row = frame.loc[
            frame[
                "participant_id"
            ].eq("Y05")
            & frame[
                "difficulty_level"
            ].eq(2)
        ]

        self.assertEqual(
            len(row),
            1,
        )

        self.assertEqual(
            float(
                row.iloc[0][
                    "list_recheck_count"
                ]
            ),
            0.0,
        )


class DirectFitContractTests(
    unittest.TestCase
):
    def test_bootstrap_repetitions_are_zero(
        self,
    ):
        self.assertEqual(
            MOD14.BOOTSTRAP_REPETITIONS,
            0,
        )

    def test_mod14_defines_no_participant_bootstrap_function(
        self,
    ):
        self.assertFalse(
            hasattr(
                MOD14,
                "draw_participant_bootstrap",
            )
        )

        self.assertFalse(
            hasattr(
                MOD14,
                "run_participant_bootstrap",
            )
        )


class LinearRegressionTests(
    unittest.TestCase
):
    def setUp(self):
        (
            self.development,
            self.test,
            self.development_ids,
            self.test_ids,
        ) = regression_ready_split_data()

    def test_all_four_models_fit_one_direct_ols_regression(
        self,
    ):
        for model_id, spec in (
            MOD14
            .build_model_registry()
            .items()
        ):
            with self.subTest(
                model_id=model_id
            ):
                frame = (
                    MOD14.prepare_model_frame(
                        self.development,
                        model_id=model_id,
                    )
                )

                fitted = (
                    MOD14.fit_linear_model(
                        frame,
                        model_id=model_id,
                    )
                )

                self.assertEqual(
                    fitted.model_id,
                    model_id,
                )

                self.assertEqual(
                    fitted.predictors,
                    spec.predictors,
                )

                self.assertEqual(
                    len(
                        fitted.coefficients
                    ),
                    (
                        1
                        + len(
                            spec.predictors
                        )
                    ),
                )

                self.assertIn(
                    "Intercept",
                    fitted.coefficients.index,
                )

                self.assertTrue(
                    np.isfinite(
                        fitted
                        .coefficients
                        .to_numpy()
                    ).all()
                )

    def test_predictions_remain_continuous(
        self,
    ):
        frame = (
            MOD14.prepare_model_frame(
                self.development,
                model_id="visual",
            )
        )

        fitted = (
            MOD14.fit_linear_model(
                frame,
                model_id="visual",
            )
        )

        predictions = (
            fitted.predict(
                frame
            )
        )

        self.assertEqual(
            len(predictions),
            len(frame),
        )

        self.assertTrue(
            np.isfinite(
                predictions.to_numpy()
            ).all()
        )

        # Predictions must not be rounded.
        self.assertFalse(
            np.allclose(
                predictions,
                np.round(
                    predictions
                ),
            )
        )

    def test_predictions_are_not_clipped_to_valid_stage_range(
        self,
    ):
        spec = (
            MOD14
            .build_model_registry()[
                "auditory"
            ]
        )

        coefficients = pd.Series(
            {
                "Intercept": 10.0,
                **{
                    predictor: 0.0
                    for predictor
                    in spec.predictors
                },
            },
            dtype=float,
        )

        fitted = (
            MOD14.LinearModelFit(
                model_id="auditory",
                predictors=(
                    spec.predictors
                ),
                coefficients=(
                    coefficients
                ),
            )
        )

        frame = (
            MOD14.prepare_model_frame(
                self.test,
                model_id="auditory",
            )
        )

        predictions = (
            fitted.predict(
                frame
            )
        )

        self.assertTrue(
            (
                predictions
                > 3.0
            ).all()
        )

    def test_rank_deficient_design_is_rejected(
        self,
    ):
        frame = (
            MOD14.prepare_model_frame(
                self.development,
                model_id="auditory",
            )
            .copy()
        )

        # Make two predictors exactly identical.
        frame[
            MOD14.AUDITORY_PREDICTORS[1]
        ] = frame[
            MOD14.AUDITORY_PREDICTORS[0]
        ]

        with self.assertRaises(
            MOD14.Mod14Error
        ):
            MOD14.fit_linear_model(
                frame,
                model_id="auditory",
            )


class MetricTests(
    unittest.TestCase
):
    def test_mae_is_participant_balanced(
        self,
    ):
        participant_ids = pd.Series(
            [
                "A",
                "A",
                "A",
                "A",
                "B",
            ]
        )

        observed = pd.Series(
            [
                0.0,
                1.0,
                2.0,
                3.0,
                0.0,
            ]
        )

        predicted = pd.Series(
            [
                0.0,
                1.0,
                2.0,
                3.0,
                2.0,
            ]
        )

        mae = (
            MOD14.participant_balanced_mae(
                participant_ids=(
                    participant_ids
                ),
                observed=observed,
                predicted=predicted,
            )
        )

        # Participant A MAE = 0
        # Participant B MAE = 2
        # Participant-balanced MAE = (0 + 2) / 2 = 1
        self.assertAlmostEqual(
            mae,
            1.0,
        )

    def test_pooled_r2_matches_definition(
        self,
    ):
        observed = pd.Series(
            [
                0.0,
                1.0,
                2.0,
                3.0,
            ]
        )

        predicted = pd.Series(
            [
                0.0,
                1.0,
                2.0,
                3.0,
            ]
        )

        r2 = (
            MOD14.pooled_r2(
                observed=observed,
                predicted=predicted,
            )
        )

        self.assertAlmostEqual(
            r2,
            1.0,
        )

    def test_pooled_r2_can_be_negative(
        self,
    ):
        observed = pd.Series(
            [
                0.0,
                1.0,
                2.0,
                3.0,
            ]
        )

        predicted = pd.Series(
            [
                3.0,
                3.0,
                0.0,
                0.0,
            ]
        )

        r2 = (
            MOD14.pooled_r2(
                observed=observed,
                predicted=predicted,
            )
        )

        self.assertLess(
            r2,
            0.0,
        )


class DirectEvaluationTests(
    unittest.TestCase
):
    def setUp(self):
        (
            self.development,
            self.test,
            self.development_ids,
            self.test_ids,
        ) = regression_ready_split_data()

    def test_model_is_fitted_using_development_data_only(
        self,
    ):
        original_fit = (
            MOD14.fit_linear_model
        )

        fitted_participant_sets = []

        def recording_fit(
            frame,
            *,
            model_id,
        ):
            fitted_participant_sets.append(
                set(
                    frame[
                        "participant_id"
                    ]
                )
            )

            return original_fit(
                frame,
                model_id=model_id,
            )

        with patch.object(
            MOD14,
            "fit_linear_model",
            side_effect=recording_fit,
        ):
            MOD14.evaluate_model(
                development_data=(
                    self.development
                ),
                test_data=self.test,
                model_id="visual",
            )

        self.assertEqual(
            len(
                fitted_participant_sets
            ),
            1,
        )

        self.assertEqual(
            fitted_participant_sets[0],
            set(
                self.development_ids
            ),
        )

        self.assertTrue(
            fitted_participant_sets[0]
            .isdisjoint(
                self.test_ids
            )
        )

    def test_evaluation_returns_training_and_test_metrics(
        self,
    ):
        result = (
            MOD14.evaluate_model(
                development_data=(
                    self.development
                ),
                test_data=self.test,
                model_id="auditory",
            )
        )

        self.assertEqual(
            result.model_id,
            "auditory",
        )

        self.assertEqual(
            result.development_frame[
                "participant_id"
            ].nunique(),
            26,
        )

        self.assertEqual(
            result.test_frame[
                "participant_id"
            ].nunique(),
            6,
        )

        self.assertTrue(
            np.isfinite(
                result.training_mae
            )
        )

        self.assertTrue(
            np.isfinite(
                result.training_r2
            )
        )

        self.assertTrue(
            np.isfinite(
                result.test_mae
            )
        )

        self.assertTrue(
            np.isfinite(
                result.test_r2
            )
        )

    def test_evaluation_stores_row_level_predictions(
        self,
    ):
        result = (
            MOD14.evaluate_model(
                development_data=(
                    self.development
                ),
                test_data=self.test,
                model_id="visual",
            )
        )

        expected_columns = (
            "participant_id",
            "model",
            "condition",
            "observed_difficulty_stage",
            "predicted_difficulty_stage",
            "error",
            "absolute_error",
        )

        self.assertEqual(
            tuple(
                result
                .training_predictions
                .columns
            ),
            expected_columns,
        )

        self.assertEqual(
            tuple(
                result
                .test_predictions
                .columns
            ),
            expected_columns,
        )

        self.assertEqual(
            len(
                result.training_predictions
            ),
            len(
                result.development_frame
            ),
        )

        self.assertEqual(
            len(
                result.test_predictions
            ),
            len(
                result.test_frame
            ),
        )

    def test_prediction_error_is_observed_minus_predicted(
        self,
    ):
        result = (
            MOD14.evaluate_model(
                development_data=(
                    self.development
                ),
                test_data=self.test,
                model_id="visual",
            )
        )

        predictions = (
            result
            .test_predictions
        )

        expected_error = (
            predictions[
                "observed_difficulty_stage"
            ]
            - predictions[
                "predicted_difficulty_stage"
            ]
        )

        np.testing.assert_allclose(
            predictions[
                "error"
            ].to_numpy(),
            expected_error.to_numpy(),
        )

        np.testing.assert_allclose(
            predictions[
                "absolute_error"
            ].to_numpy(),
            np.abs(
                expected_error.to_numpy()
            ),
        )

    def test_out_of_range_predictions_are_counted_not_modified(
        self,
    ):
        result = (
            MOD14.evaluate_model(
                development_data=(
                    self.development
                ),
                test_data=self.test,
                model_id=(
                    "cognitive_primary"
                ),
            )
        )

        training_values = (
            result
            .training_predictions[
                "predicted_difficulty_stage"
            ]
            .to_numpy()
        )

        expected_training_count = int(
            np.sum(
                (
                    training_values
                    < 0.0
                )
                |
                (
                    training_values
                    > 3.0
                )
            )
        )

        self.assertEqual(
            result
            .training_out_of_range_count,
            expected_training_count,
        )

        test_values = (
            result
            .test_predictions[
                "predicted_difficulty_stage"
            ]
            .to_numpy()
        )

        expected_test_count = int(
            np.sum(
                (
                    test_values
                    < 0.0
                )
                |
                (
                    test_values
                    > 3.0
                )
            )
        )

        self.assertEqual(
            result
            .test_out_of_range_count,
            expected_test_count,
        )


class AllModelEvaluationTests(
    unittest.TestCase
):
    def setUp(self):
        (
            self.development,
            self.test,
            self.development_ids,
            self.test_ids,
        ) = regression_ready_split_data()

    def test_all_four_models_are_evaluated_once(
        self,
    ):
        results = (
            MOD14.evaluate_all_models(
                development_data=(
                    self.development
                ),
                test_data=self.test,
            )
        )

        self.assertEqual(
            tuple(results),
            (
                "visual",
                "auditory",
                "cognitive_primary",
                "cognitive_later",
            ),
        )

    def test_train_test_performance_table_contains_required_comparison(
        self,
    ):
        results = (
            MOD14.evaluate_all_models(
                development_data=(
                    self.development
                ),
                test_data=self.test,
            )
        )

        table = (
            MOD14.build_performance_table(
                results
            )
        )

        expected_columns = (
            "model",
            "condition",
            "development_participant_count",
            "development_row_count",
            "test_participant_count",
            "test_row_count",
            "training_mae",
            "test_mae",
            "test_minus_training_mae",
            "training_r2",
            "test_r2",
            "training_minus_test_r2",
            "training_out_of_range_prediction_count",
            "test_out_of_range_prediction_count",
        )

        self.assertEqual(
            tuple(
                table.columns
            ),
            expected_columns,
        )

        self.assertEqual(
            len(table),
            4,
        )

        np.testing.assert_allclose(
            table[
                "test_minus_training_mae"
            ],
            (
                table["test_mae"]
                - table["training_mae"]
            ),
        )

        np.testing.assert_allclose(
            table[
                "training_minus_test_r2"
            ],
            (
                table["training_r2"]
                - table["test_r2"]
            ),
        )

    def test_cognitive_later_counts_only_later_stage_rows(
        self,
    ):
        results = (
            MOD14.evaluate_all_models(
                development_data=(
                    self.development
                ),
                test_data=self.test,
            )
        )

        result = results[
            "cognitive_later"
        ]

        self.assertEqual(
            len(
                result.development_frame
            ),
            78,
        )

        self.assertEqual(
            len(
                result.test_frame
            ),
            18,
        )

        self.assertEqual(
            set(
                result
                .development_frame[
                    "difficulty_stage"
                ]
            ),
            {1, 2, 3},
        )


if __name__ == "__main__":
    unittest.main()