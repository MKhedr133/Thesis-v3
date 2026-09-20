"""Synthetic tests for MOD-10 training-only random-effects comparison."""

from __future__ import annotations

from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import sys
import unittest

import pandas as pd


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
PYTHON_DIR = REPOSITORY_ROOT / "python"

MODULE_PATH = PYTHON_DIR / "MOD_10_training_random_effects_comparison.py"

SPEC = spec_from_file_location(
    "mod10_training_random_effects_test",
    MODULE_PATH,
)

if SPEC is None or SPEC.loader is None:
    raise ImportError(MODULE_PATH)

MOD10 = module_from_spec(SPEC)
sys.modules[SPEC.name] = MOD10
SPEC.loader.exec_module(MOD10)


def synthetic_modeling_data() -> pd.DataFrame:
    """Create repeated rows for 32 synthetic participants."""
    rows = []

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
                    }
                )

    return pd.DataFrame(rows)


def synthetic_holdout_split() -> pd.DataFrame:
    """Create a valid 26/6 MOD-09 style participant split."""
    rows = []

    for group, prefix in (("Young", "Y"), ("Old", "O")):
        for number in range(1, 17):
            participant_id = f"{prefix}{number:02d}"

            # Three test participants in each age group.
            split = "test" if number in {1, 2, 3} else "train"

            rows.append(
                {
                    "participant_id": participant_id,
                    "participant_group": group,
                    "split": split,
                }
            )

    return pd.DataFrame(rows)


class TrainingBoundaryTests(unittest.TestCase):
    def test_valid_split_returns_only_26_training_participants(self):
        modeling_data = synthetic_modeling_data()
        holdout_split = synthetic_holdout_split()

        training_data, training_ids, test_ids = (
            MOD10.prepare_training_data(
                modeling_data=modeling_data,
                holdout_split=holdout_split,
            )
        )

        self.assertEqual(len(training_ids), 26)
        self.assertEqual(len(test_ids), 6)

        self.assertEqual(
            training_data["participant_id"].nunique(),
            26,
        )

        self.assertTrue(
            set(training_data["participant_id"]).isdisjoint(test_ids)
        )

        self.assertEqual(
            set(training_data["participant_id"]),
            set(training_ids),
        )


if __name__ == "__main__":
    unittest.main()