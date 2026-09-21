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


if __name__ == "__main__":
    unittest.main()