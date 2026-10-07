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


if __name__ == "__main__":
    unittest.main()