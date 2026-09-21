"""Synthetic tests for MOD-13 difficulty-prediction Batch 1."""

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

from collections import Counter


ROOT = Path(__file__).resolve().parents[1]
PYTHON_DIR = ROOT / "python"

sys.path.insert(
    0,
    str(PYTHON_DIR),
)

MODULE_PATH = (
    PYTHON_DIR
    / "MOD_13_difficulty_prediction.py"
)

SPEC = spec_from_file_location(
    "mod13_difficulty_prediction_test",
    MODULE_PATH,
)

if SPEC is None or SPEC.loader is None:
    raise ImportError(MODULE_PATH)

MOD13 = module_from_spec(SPEC)

# Required for dataclasses when loading the module this way.
sys.modules[SPEC.name] = MOD13

SPEC.loader.exec_module(MOD13)


def synthetic_modeling_data() -> pd.DataFrame:
    """Create repeated synthetic data for 32 participants."""
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

                    value = float(
                        number
                        + difficulty_stage
                    )

                    rows.append(
                        {
                            "participant_id": (
                                participant_id
                            ),
                            "participant_group": (
                                group
                            ),
                            "condition_name": (
                                condition
                            ),
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
                            ): value,
                            (
                                "median_time_between_"
                                "qualifying_grabs_seconds"
                            ): value,
                            "list_recheck_count": float(
                                difficulty_stage
                            ),
                            (
                                "total_list_recheck_"
                                "duration_seconds"
                            ): value,
                            (
                                "median_time_to_"
                                "target_seconds"
                            ): value,
                            (
                                "median_irrelevant_"
                                "focus_duration_seconds"
                            ): value,
                            (
                                "median_head_turning_"
                                "degrees"
                            ): value,
                            (
                                "median_reach_"
                                "duration_seconds"
                            ): value,
                            (
                                "median_reach_"
                                "path_ratio"
                            ): (
                                1.0
                                + value / 100.0
                            ),
                        }
                    )

    data = pd.DataFrame(rows)

    # Genuine behavioural zero.
    zero_mask = (
        data["participant_id"].eq("Y05")
        & data["condition_name"].eq(
            "Cognitive"
        )
        & data["difficulty_level"].eq(2)
    )

    data.loc[
        zero_mask,
        "list_recheck_count",
    ] = 0.0

    # One genuine missing predictor used by Visual.
    missing_mask = (
        data["participant_id"].eq("Y04")
        & data["condition_name"].eq(
            "Visual"
        )
        & data["difficulty_level"].eq(6)
    )

    data.loc[
        missing_mask,
        "median_reach_duration_seconds",
    ] = np.nan

    return data


def synthetic_holdout_split() -> pd.DataFrame:
    """Create a valid MOD-09-style 26/6 split."""
    rows: list[dict[str, str]] = []

    for group, prefix in (
        ("Young", "Y"),
        ("Old", "O"),
    ):
        for number in range(1, 17):
            rows.append(
                {
                    "participant_id": (
                        f"{prefix}{number:02d}"
                    ),
                    "participant_group": group,
                    "split": (
                        "test"
                        if number in {1, 2, 3}
                        else "train"
                    ),
                }
            )

    return pd.DataFrame(rows)


def regression_ready_training():
    """Create deterministic non-collinear data for regression tests."""
    training, _, test_ids = (
        MOD13.prepare_training_data(
            modeling_data=(
                synthetic_modeling_data()
            ),
            holdout_split=(
                synthetic_holdout_split()
            ),
        )
    )

    data = training.copy()

    rng = np.random.default_rng(
        12345
    )

    registry = (
        MOD13.build_model_registry()
    )

    predictors = sorted(
        {
            predictor
            for spec in registry.values()
            for predictor in spec.predictors
        }
    )

    stage = (
        data["difficulty_stage"]
        .astype(float)
        .to_numpy()
    )

    for index, predictor in enumerate(
        predictors
    ):
        noise = rng.normal(
            loc=0.0,
            scale=1.0 + index * 0.05,
            size=len(data),
        )

        data[predictor] = (
            stage * (
                0.15
                + index * 0.03
            )
            + noise
        )

    # Preserve the structural D0 baseline for relative performance.
    performance = (
        MOD13.PERFORMANCE_CHANGE_PREDICTOR
    )

    d0_mask = (
        data["difficulty_stage"].eq(0)
    )

    data.loc[
        d0_mask,
        performance,
    ] = 0.0

    return (
        data,
        test_ids,
    )


class TrainingBoundaryTests(
    unittest.TestCase
):
    def test_only_26_training_participants_enter_development(
        self,
    ):
        (
            training,
            training_ids,
            test_ids,
        ) = MOD13.prepare_training_data(
            modeling_data=(
                synthetic_modeling_data()
            ),
            holdout_split=(
                synthetic_holdout_split()
            ),
        )

        self.assertEqual(
            len(training_ids),
            26,
        )

        self.assertEqual(
            len(test_ids),
            6,
        )

        self.assertEqual(
            training[
                "participant_id"
            ].nunique(),
            26,
        )

        self.assertTrue(
            set(
                training["participant_id"]
            ).isdisjoint(test_ids)
        )

    def test_test_values_cannot_change_training_data(
        self,
    ):
        data = synthetic_modeling_data()
        split = synthetic_holdout_split()

        (
            baseline,
            _,
            test_ids,
        ) = MOD13.prepare_training_data(
            modeling_data=data,
            holdout_split=split,
        )

        changed = data.copy()

        test_mask = (
            changed["participant_id"]
            .isin(test_ids)
        )

        predictor_columns = sorted(
            {
                predictor
                for spec
                in (
                    MOD13
                    .build_model_registry()
                    .values()
                )
                for predictor
                in spec.predictors
            }
        )

        changed.loc[
            test_mask,
            predictor_columns,
        ] = 999999.0

        repeated, _, _ = (
            MOD13.prepare_training_data(
                modeling_data=changed,
                holdout_split=split,
            )
        )

        pd.testing.assert_frame_equal(
            baseline.reset_index(
                drop=True
            ),
            repeated.reset_index(
                drop=True
            ),
        )

    def test_model_frame_rejects_forbidden_test_participants(
        self,
    ):
        data = synthetic_modeling_data()

        _, test_ids = (
            MOD13.validate_holdout_split(
                synthetic_holdout_split()
            )
        )

        with self.assertRaises(
            MOD13.Mod13Error
        ):
            MOD13.prepare_model_frame(
                data,
                model_id="visual",
                forbidden_participant_ids=(
                    test_ids
                ),
            )


class ModelRegistryTests(
    unittest.TestCase
):
    def test_target_coding_is_frozen(
        self,
    ):
        self.assertEqual(
            MOD13.DIFFICULTY_STAGE_BY_LEVEL,
            {
                0: 0,
                2: 1,
                6: 2,
                10: 3,
            },
        )

    def test_registry_contains_exact_four_models(
        self,
    ):
        registry = (
            MOD13.build_model_registry()
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

    def test_visual_predictors_are_frozen(
        self,
    ):
        spec = (
            MOD13
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

    def test_auditory_predictors_are_frozen(
        self,
    ):
        spec = (
            MOD13
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

    def test_cognitive_primary_is_d0_compatible(
        self,
    ):
        spec = (
            MOD13
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
            len(spec.predictors),
            9,
        )

        self.assertNotIn(
            MOD13.PERFORMANCE_CHANGE_PREDICTOR,
            spec.predictors,
        )

    def test_cognitive_later_adds_relative_performance(
        self,
    ):
        registry = (
            MOD13.build_model_registry()
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
            len(later.predictors),
            10,
        )

        self.assertEqual(
            later.predictors[:-1],
            primary.predictors,
        )

        self.assertEqual(
            later.predictors[-1],
            MOD13.PERFORMANCE_CHANGE_PREDICTOR,
        )


class ModelFrameTests(
    unittest.TestCase
):
    def setUp(self):
        (
            self.training,
            _,
            self.test_ids,
        ) = MOD13.prepare_training_data(
            modeling_data=(
                synthetic_modeling_data()
            ),
            holdout_split=(
                synthetic_holdout_split()
            ),
        )

    def test_visual_complete_rows_drop_only_missing_required_trial(
        self,
    ):
        frame = (
            MOD13.prepare_model_frame(
                self.training,
                model_id="visual",
                forbidden_participant_ids=(
                    self.test_ids
                ),
            )
        )

        self.assertEqual(
            len(frame),
            103,
        )

        self.assertEqual(
            frame[
                "participant_id"
            ].nunique(),
            26,
        )

        self.assertFalse(
            frame.isna().any().any()
        )

    def test_missing_nonregistered_predictor_does_not_remove_visual_row(
        self,
    ):
        changed = (
            self.training.copy()
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
            MOD13.prepare_model_frame(
                changed,
                model_id="visual",
                forbidden_participant_ids=(
                    self.test_ids
                ),
            )
        )

        # Only the existing missing
        # reach-duration row is removed.
        self.assertEqual(
            len(frame),
            103,
        )

    def test_cognitive_primary_uses_all_four_stages(
        self,
    ):
        frame = (
            MOD13.prepare_model_frame(
                self.training,
                model_id=(
                    "cognitive_primary"
                ),
                forbidden_participant_ids=(
                    self.test_ids
                ),
            )
        )

        self.assertEqual(
            len(frame),
            104,
        )

        self.assertEqual(
            set(
                frame["difficulty_stage"]
            ),
            {0, 1, 2, 3},
        )

        self.assertNotIn(
            MOD13.PERFORMANCE_CHANGE_PREDICTOR,
            frame.columns,
        )

    def test_cognitive_later_uses_only_d2_d6_d10(
        self,
    ):
        frame = (
            MOD13.prepare_model_frame(
                self.training,
                model_id=(
                    "cognitive_later"
                ),
                forbidden_participant_ids=(
                    self.test_ids
                ),
            )
        )

        self.assertEqual(
            len(frame),
            78,
        )

        self.assertEqual(
            set(
                frame["difficulty_level"]
            ),
            {2, 6, 10},
        )

        self.assertEqual(
            set(
                frame["difficulty_stage"]
            ),
            {1, 2, 3},
        )

        self.assertIn(
            MOD13.PERFORMANCE_CHANGE_PREDICTOR,
            frame.columns,
        )

    def test_genuine_zero_is_preserved(
        self,
    ):
        frame = (
            MOD13.prepare_model_frame(
                self.training,
                model_id=(
                    "cognitive_primary"
                ),
                forbidden_participant_ids=(
                    self.test_ids
                ),
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

    def test_frame_contains_only_identity_target_and_registered_predictors(
        self,
    ):
        registry = (
            MOD13.build_model_registry()
        )

        for (
            model_id,
            spec,
        ) in registry.items():

            with self.subTest(
                model_id=model_id
            ):
                frame = (
                    MOD13.prepare_model_frame(
                        self.training,
                        model_id=model_id,
                        forbidden_participant_ids=(
                            self.test_ids
                        ),
                    )
                )

                self.assertEqual(
                    tuple(frame.columns),
                    (
                        "participant_id",
                        "condition_name",
                        "difficulty_level",
                        "difficulty_stage",
                        *spec.predictors,
                    ),
                )

    def test_incorrect_target_coding_fails(
        self,
    ):
        changed = (
            self.training.copy()
        )

        mask = (
            changed[
                "participant_id"
            ].eq("Y04")
            & changed[
                "condition_name"
            ].eq("Visual")
            & changed[
                "difficulty_level"
            ].eq(2)
        )

        # Incorrect:
        # D2 should be stage 1.
        changed.loc[
            mask,
            "difficulty_stage",
        ] = 2

        with self.assertRaises(
            MOD13.Mod13Error
        ):
            MOD13.prepare_model_frame(
                changed,
                model_id="visual",
                forbidden_participant_ids=(
                    self.test_ids
                ),
            )

    def test_unknown_model_fails(
        self,
    ):
        with self.assertRaises(
            MOD13.Mod13Error
        ):
            MOD13.prepare_model_frame(
                self.training,
                model_id="not_a_model",
                forbidden_participant_ids=(
                    self.test_ids
                ),
            )


class LinearRegressionTests(
    unittest.TestCase
):
    def setUp(self):
        (
            self.training,
            self.test_ids,
        ) = regression_ready_training()

    def test_all_four_models_can_fit(
        self,
    ):
        for model_id in (
            MOD13
            .build_model_registry()
        ):
            with self.subTest(
                model_id=model_id
            ):
                frame = (
                    MOD13
                    .prepare_model_frame(
                        self.training,
                        model_id=model_id,
                        forbidden_participant_ids=(
                            self.test_ids
                        ),
                    )
                )

                fitted = (
                    MOD13
                    .fit_linear_model(
                        frame,
                        model_id=model_id,
                    )
                )

                expected_terms = (
                    1
                    + len(
                        MOD13
                        .build_model_registry()[
                            model_id
                        ]
                        .predictors
                    )
                )

                self.assertEqual(
                    len(
                        fitted.coefficients
                    ),
                    expected_terms,
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

    def test_prediction_is_continuous_and_unclipped(
        self,
    ):
        frame = (
            MOD13.prepare_model_frame(
                self.training,
                model_id="visual",
                forbidden_participant_ids=(
                    self.test_ids
                ),
            )
        )

        fitted = (
            MOD13.fit_linear_model(
                frame,
                model_id="visual",
            )
        )

        predictions = (
            fitted.predict(frame)
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

        # Batch 2 must not round predictions.
        self.assertFalse(
            np.allclose(
                predictions,
                np.round(predictions),
            )
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
            MOD13
            .participant_balanced_mae(
                participant_ids=(
                    participant_ids
                ),
                observed=observed,
                predicted=predicted,
            )
        )

        # Participant A MAE = 0
        # Participant B MAE = 2
        # Equal participant weighting = 1
        self.assertAlmostEqual(
            mae,
            1.0,
        )

    def test_r2_can_be_negative(
        self,
    ):
        observed = pd.Series(
            [0.0, 1.0, 2.0, 3.0]
        )

        predicted = pd.Series(
            [3.0, 3.0, 0.0, 0.0]
        )

        r2 = MOD13.pooled_r2(
            observed=observed,
            predicted=predicted,
        )

        self.assertLess(
            r2,
            0.0,
        )


class ParticipantBootstrapTests(
    unittest.TestCase
):
    def setUp(self):
        (
            self.training,
            self.test_ids,
        ) = regression_ready_training()

        self.frame = (
            MOD13.prepare_model_frame(
                self.training,
                model_id="visual",
                forbidden_participant_ids=(
                    self.test_ids
                ),
            )
        )

    def test_bootstrap_resamples_whole_participants(
        self,
    ):
        rng = np.random.default_rng(
            42
        )

        (
            inbag,
            oob,
            sampled_ids,
        ) = (
            MOD13
            .draw_participant_bootstrap(
                self.frame,
                rng=rng,
            )
        )

        self.assertEqual(
            len(sampled_ids),
            26,
        )

        source_counts = (
            self.frame[
                "participant_id"
            ]
            .value_counts()
            .to_dict()
        )

        sampled_counts = Counter(
            sampled_ids
        )

        inbag_counts = (
            inbag[
                "participant_id"
            ]
            .value_counts()
            .to_dict()
        )

        for (
            participant_id,
            draw_count,
        ) in sampled_counts.items():

            expected_rows = (
                source_counts[
                    participant_id
                ]
                * draw_count
            )

            self.assertEqual(
                inbag_counts[
                    participant_id
                ],
                expected_rows,
            )

        sampled_unique = set(
            sampled_ids
        )

        expected_oob = (
            set(source_counts)
            - sampled_unique
        )

        self.assertEqual(
            set(
                oob["participant_id"]
            ),
            expected_oob,
        )

    def test_bootstrap_is_reproducible(
        self,
    ):
        first = (
            MOD13
            .run_participant_bootstrap(
                self.frame,
                model_id="visual",
                repetitions=20,
                seed=123,
            )
        )

        second = (
            MOD13
            .run_participant_bootstrap(
                self.frame,
                model_id="visual",
                repetitions=20,
                seed=123,
            )
        )

        pd.testing.assert_frame_equal(
            first.metrics,
            second.metrics,
        )

        pd.testing.assert_frame_equal(
            first.coefficients,
            second.coefficients,
        )

    def test_bootstrap_stores_oob_metrics_and_coefficients(
        self,
    ):
        result = (
            MOD13
            .run_participant_bootstrap(
                self.frame,
                model_id="visual",
                repetitions=20,
                seed=321,
            )
        )

        self.assertEqual(
            len(result.metrics),
            20,
        )

        successful = (
            result.metrics[
                "status"
            ].eq("success")
        )

        self.assertTrue(
            successful.any()
        )

        successful_metrics = (
            result.metrics.loc[
                successful
            ]
        )

        self.assertTrue(
            np.isfinite(
                successful_metrics[
                    "mae"
                ]
            ).all()
        )

        self.assertTrue(
            np.isfinite(
                successful_metrics[
                    "r2"
                ]
            ).all()
        )

        self.assertFalse(
            result.coefficients.empty
        )

        self.assertIn(
            "Intercept",
            set(
                result.coefficients[
                    "predictor"
                ]
            ),
        )


class BootstrapSummaryTests(
    unittest.TestCase
):
    def setUp(self):
        (
            self.training,
            self.test_ids,
        ) = regression_ready_training()

    def test_final_coefficient_comes_from_full_training_fit(
        self,
    ):
        result = (
            MOD13.develop_model(
                self.training,
                model_id="auditory",
                forbidden_participant_ids=(
                    self.test_ids
                ),
                bootstrap_repetitions=20,
                bootstrap_seed=987,
            )
        )

        summary = (
            result.coefficient_summary
            .set_index("predictor")
        )

        for (
            predictor,
            coefficient,
        ) in (
            result
            .final_fit
            .coefficients
            .items()
        ):
            self.assertAlmostEqual(
                summary.loc[
                    predictor,
                    "coefficient",
                ],
                float(coefficient),
            )

    def test_coefficient_ci_uses_2_5_and_97_5_percentiles(
        self,
    ):
        result = (
            MOD13.develop_model(
                self.training,
                model_id="auditory",
                forbidden_participant_ids=(
                    self.test_ids
                ),
                bootstrap_repetitions=20,
                bootstrap_seed=654,
            )
        )

        predictor = (
            result
            .final_fit
            .coefficients
            .index[0]
        )

        draws = (
            result
            .bootstrap
            .coefficients
            .loc[
                lambda frame: (
                    frame["predictor"]
                    .eq(predictor)
                ),
                "coefficient",
            ]
            .to_numpy(dtype=float)
        )

        summary_row = (
            result
            .coefficient_summary
            .loc[
                lambda frame: (
                    frame["predictor"]
                    .eq(predictor)
                )
            ]
            .iloc[0]
        )

        self.assertAlmostEqual(
            summary_row[
                "bootstrap_ci_95_lower"
            ],
            np.percentile(
                draws,
                2.5,
            ),
        )

        self.assertAlmostEqual(
            summary_row[
                "bootstrap_ci_95_upper"
            ],
            np.percentile(
                draws,
                97.5,
            ),
        )

    def test_performance_summary_reports_bootstrap_interval(
        self,
    ):
        result = (
            MOD13.develop_model(
                self.training,
                model_id="visual",
                forbidden_participant_ids=(
                    self.test_ids
                ),
                bootstrap_repetitions=20,
                bootstrap_seed=111,
            )
        )

        summary = (
            result
            .performance_summary
            .iloc[0]
        )

        self.assertEqual(
            summary[
                "bootstrap_repetitions"
            ],
            20,
        )

        self.assertGreater(
            summary[
                "usable_bootstrap_repetitions"
            ],
            0,
        )

        self.assertTrue(
            np.isfinite(
                summary["oob_mae"]
            )
        )

        self.assertLessEqual(
            summary[
                "mae_ci_95_lower"
            ],
            summary[
                "oob_mae"
            ],
        )

        self.assertGreaterEqual(
            summary[
                "mae_ci_95_upper"
            ],
            summary[
                "oob_mae"
            ],
        )

def regression_ready_training():
    """Create deterministic non-collinear data for regression tests."""
    training, _, test_ids = (
        MOD13.prepare_training_data(
            modeling_data=(
                synthetic_modeling_data()
            ),
            holdout_split=(
                synthetic_holdout_split()
            ),
        )
    )

    data = training.copy()

    rng = np.random.default_rng(
        12345
    )

    registry = (
        MOD13.build_model_registry()
    )

    predictors = sorted(
        {
            predictor
            for spec in registry.values()
            for predictor in spec.predictors
        }
    )

    stage = (
        data["difficulty_stage"]
        .astype(float)
        .to_numpy()
    )

    for index, predictor in enumerate(
        predictors
    ):
        noise = rng.normal(
            loc=0.0,
            scale=1.0 + index * 0.05,
            size=len(data),
        )

        data[predictor] = (
            stage * (
                0.15
                + index * 0.03
            )
            + noise
        )

    # Preserve the structural D0 baseline for relative performance.
    performance = (
        MOD13.PERFORMANCE_CHANGE_PREDICTOR
    )

    d0_mask = (
        data["difficulty_stage"].eq(0)
    )

    data.loc[
        d0_mask,
        performance,
    ] = 0.0

    return (
        data,
        test_ids,
    )

if __name__ == "__main__":
    unittest.main()