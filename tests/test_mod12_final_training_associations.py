"""Synthetic tests for MOD-12 final training-only associations."""

from __future__ import annotations

from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import sys
import unittest

import numpy as np
import pandas as pd


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


if __name__ == "__main__":
    unittest.main()