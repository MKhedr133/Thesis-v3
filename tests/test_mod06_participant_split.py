"""Synthetic tests for MOD-06 deterministic participant splitting."""
from __future__ import annotations

from importlib.util import module_from_spec, spec_from_file_location
import json
from pathlib import Path
import sys
import tempfile
import unittest

import pandas as pd


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = REPOSITORY_ROOT / "python" / "MOD_06_participant_split.py"


def load_module():
    specification = spec_from_file_location("mod06_participant_split_test", MODULE_PATH)
    if specification is None or specification.loader is None:
        raise ImportError(MODULE_PATH)
    module = module_from_spec(specification)
    sys.modules[specification.name] = module
    specification.loader.exec_module(module)
    return module


MOD06 = load_module()


def synthetic_trials() -> pd.DataFrame:
    """Return repeated synthetic rows for 16 Young and 16 Old participants."""
    rows = []
    for group, prefix in (("Young", "Y"), ("Old", "O")):
        for number in range(1, 17):
            participant_id = f"{prefix}{number:02d}"
            for condition_index, condition in enumerate(
                ("Visual", "Auditory", "Cognitive")
            ):
                rows.append(
                    {
                        "participant_id": participant_id,
                        "participant_group": group,
                        "condition_name": condition,
                        "tmt_b_seconds": float(number * 10 + condition_index),
                        "mental_demand_score_0_to_10": float(condition_index + 1),
                    }
                )
    return pd.DataFrame(rows)


class ParticipantSplitTests(unittest.TestCase):
    def test_split_has_required_counts_and_fixed_synthetic_draw(self):
        assignments = MOD06.build_participant_split(synthetic_trials())

        self.assertEqual(len(assignments), 32)
        self.assertEqual(assignments["participant_id"].nunique(), 32)
        self.assertEqual(
            assignments["partition"].value_counts().to_dict(),
            {"train": 26, "test": 6},
        )

        by_group = (
            assignments.groupby(["participant_group", "partition"])
            .size()
            .to_dict()
        )
        self.assertEqual(
            by_group,
            {
                ("Young", "train"): 13,
                ("Young", "test"): 3,
                ("Old", "train"): 13,
                ("Old", "test"): 3,
            },
        )

        held_out = set(
            assignments.loc[
                assignments["partition"].eq("test"), "participant_id"
            ]
        )
        self.assertEqual(
            held_out,
            {"Y08", "Y05", "Y07", "O16", "O05", "O13"},
        )

    def test_repeated_trial_rows_collapse_to_one_participant(self):
        data = synthetic_trials()
        assignments = MOD06.build_participant_split(data)

        self.assertEqual(len(data), 96)
        self.assertEqual(len(assignments), 32)
        self.assertFalse(assignments["participant_id"].duplicated().any())

    def test_participant_assignment_is_shared_across_conditions(self):
        data = synthetic_trials()
        assignments = MOD06.build_participant_split(data)
        joined = data.merge(
            assignments[["participant_id", "partition"]],
            on="participant_id",
            how="left",
            validate="many_to_one",
        )

        self.assertFalse(joined["partition"].isna().any())
        self.assertTrue(
            joined.groupby("participant_id")["partition"].nunique().eq(1).all()
        )
        self.assertTrue(
            joined.groupby("participant_id")["condition_name"].nunique().eq(3).all()
        )

    def test_input_row_order_does_not_change_assignment(self):
        data = synthetic_trials()
        shuffled = data.sample(frac=1.0, random_state=123).reset_index(drop=True)

        first = MOD06.build_participant_split(data)
        second = MOD06.build_participant_split(shuffled)

        pd.testing.assert_frame_equal(first, second)

    def test_non_split_columns_do_not_change_assignment(self):
        first_input = synthetic_trials()
        second_input = first_input.copy(deep=True)
        second_input["tmt_b_seconds"] = (
            second_input["tmt_b_seconds"] * -1000.0
        )
        second_input["mental_demand_score_0_to_10"] = 999.0
        second_input["invented_outcome"] = range(len(second_input))

        first = MOD06.build_participant_split(first_input)
        second = MOD06.build_participant_split(second_input)

        pd.testing.assert_frame_equal(first, second)

    def test_same_input_reproduces_same_assignment_and_signatures(self):
        data = synthetic_trials()
        first = MOD06.build_participant_split(data)
        second = MOD06.build_participant_split(data.copy(deep=True))

        pd.testing.assert_frame_equal(first, second)
        self.assertEqual(
            MOD06.assignment_signature(first),
            MOD06.assignment_signature(second),
        )
        self.assertEqual(
            MOD06.participant_source_signature(
                MOD06.canonical_participants(data)
            ),
            MOD06.participant_source_signature(
                MOD06.canonical_participants(data.copy(deep=True))
            ),
        )

    def test_conflicting_group_assignment_fails(self):
        data = synthetic_trials()
        conflict = data.iloc[[0]].copy()
        conflict["participant_group"] = "Old"
        data = pd.concat([data, conflict], ignore_index=True)

        with self.assertRaisesRegex(
            MOD06.ParticipantSplitError,
            "one consistent participant_group",
        ):
            MOD06.build_participant_split(data)

    def test_unexpected_group_label_fails(self):
        data = synthetic_trials()
        data.loc[data["participant_id"].eq("Y01"), "participant_group"] = "Middle"

        with self.assertRaisesRegex(
            MOD06.ParticipantSplitError,
            "Unexpected participant_group",
        ):
            MOD06.build_participant_split(data)

    def test_wrong_participant_count_fails(self):
        data = synthetic_trials()
        data = data.loc[~data["participant_id"].eq("Y16")].copy()

        with self.assertRaisesRegex(
            MOD06.ParticipantSplitError,
            "Expected 32 unique participants",
        ):
            MOD06.build_participant_split(data)

    def test_wrong_group_count_fails(self):
        data = synthetic_trials()
        data.loc[data["participant_id"].eq("Y16"), "participant_id"] = "O17"
        data.loc[data["participant_id"].eq("O17"), "participant_group"] = "Old"

        with self.assertRaisesRegex(
            MOD06.ParticipantSplitError,
            "Expected participant-group counts",
        ):
            MOD06.build_participant_split(data)

    def test_manifest_records_frozen_method_and_exclusions(self):
        data = synthetic_trials()
        assignments = MOD06.build_participant_split(data)
        manifest = MOD06.build_manifest(data, assignments)

        self.assertEqual(manifest["method_id"], "age_stratified_fixed_holdout_v1")
        self.assertEqual(manifest["random_seed"], 20260919)
        self.assertEqual(manifest["numpy_bit_generator"], "PCG64")
        self.assertEqual(manifest["group_order"], ["Young", "Old"])
        self.assertEqual(manifest["training_participant_count"], 26)
        self.assertEqual(manifest["test_participant_count"], 6)
        self.assertEqual(
            manifest["input_columns_used_for_split"],
            ["participant_id", "participant_group"],
        )
        self.assertFalse(manifest["tmt_b_used_for_split"])
        self.assertFalse(manifest["outcomes_used_for_split"])
        self.assertFalse(manifest["behavioural_features_used_for_split"])
        self.assertFalse(manifest["condition_used_for_split"])
        self.assertEqual(len(manifest["participant_source_sha256"]), 64)
        self.assertEqual(len(manifest["assignment_sha256"]), 64)

    def test_output_writer_refuses_to_overwrite_split(self):
        data = synthetic_trials()
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)
            assignment_path, manifest_path = MOD06.write_split_outputs(
                data, output_dir
            )

            self.assertTrue(assignment_path.exists())
            self.assertTrue(manifest_path.exists())

            assignments = pd.read_csv(assignment_path)
            with manifest_path.open("r", encoding="utf-8") as handle:
                manifest = json.load(handle)

            self.assertEqual(len(assignments), 32)
            self.assertEqual(manifest["random_seed"], 20260919)

            with self.assertRaisesRegex(
                FileExistsError,
                "Refusing to overwrite",
            ):
                MOD06.write_split_outputs(data, output_dir)


if __name__ == "__main__":
    unittest.main()
