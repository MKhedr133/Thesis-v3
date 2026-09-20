"""Synthetic tests for MOD-09 participant holdout splitting."""
from __future__ import annotations

from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
PYDIR = ROOT / "python"
sys.path.insert(0, str(PYDIR))

SPEC = spec_from_file_location(
    "mod09_test",
    PYDIR / "MOD_09_participant_holdout_split.py",
)
if SPEC is None or SPEC.loader is None:
    raise ImportError(PYDIR / "MOD_09_participant_holdout_split.py")
MOD09 = module_from_spec(SPEC)
sys.modules[SPEC.name] = MOD09
SPEC.loader.exec_module(MOD09)


def synthetic_data() -> pd.DataFrame:
    rows = []
    for group, prefix in (("Young", "Y"), ("Old", "O")):
        for number in range(1, 17):
            participant_id = f"{prefix}{number:02d}"
            for stage, difficulty in enumerate((0, 2, 6, 10)):
                rows.append(
                    {
                        "participant_id": participant_id,
                        "participant_group": group,
                        "difficulty_level": difficulty,
                        "difficulty_stage": stage,
                        "irrelevant_outcome": float(number + stage),
                        "tmt_b_seconds": float(
                            30 + number + (15 if group == "Old" else 0)
                        ),
                    }
                )
    return pd.DataFrame(rows)


class Mod09Tests(unittest.TestCase):
    def setUp(self):
        self.data = synthetic_data()

    def test_split_is_reproducible_and_has_exact_group_counts(self):
        first = MOD09.create_holdout_split(
            self.data,
            seed=MOD09.DEFAULT_SEED,
        )
        second = MOD09.create_holdout_split(
            self.data,
            seed=MOD09.DEFAULT_SEED,
        )

        pd.testing.assert_frame_equal(first, second)
        self.assertEqual(len(first), 32)
        self.assertEqual(
            list(first.columns),
            ["participant_id", "participant_group", "split"],
        )
        self.assertEqual(int(first["split"].eq("train").sum()), 26)
        self.assertEqual(int(first["split"].eq("test").sum()), 6)

        counts = (
            first.groupby(["split", "participant_group"])
            .size()
            .to_dict()
        )
        self.assertEqual(counts[("train", "Young")], 13)
        self.assertEqual(counts[("train", "Old")], 13)
        self.assertEqual(counts[("test", "Young")], 3)
        self.assertEqual(counts[("test", "Old")], 3)

    def test_split_does_not_depend_on_row_order_outcomes_or_tmt(self):
        baseline = MOD09.create_holdout_split(self.data, seed=12345)

        changed = self.data.sample(
            frac=1.0,
            random_state=999,
        ).reset_index(drop=True)
        changed["irrelevant_outcome"] = np.arange(len(changed)) * 1000.0
        changed["tmt_b_seconds"] = np.linspace(
            1.0,
            9999.0,
            len(changed),
        )

        repeated = MOD09.create_holdout_split(changed, seed=12345)

        pd.testing.assert_frame_equal(baseline, repeated)

    def test_participant_group_must_be_consistent_within_participant(self):
        bad = self.data.copy()
        mask = bad["participant_id"].eq("Y01")
        bad.loc[bad.index[mask][0], "participant_group"] = "Old"

        with self.assertRaisesRegex(
            MOD09.Mod09Error,
            "inconsistent participant_group",
        ):
            MOD09.create_holdout_split(bad, seed=1)

    def test_requires_exactly_32_participants_and_16_per_group(self):
        missing_participant = self.data.loc[
            ~self.data["participant_id"].eq("Y01")
        ].copy()
        with self.assertRaisesRegex(MOD09.Mod09Error, "32 participants"):
            MOD09.create_holdout_split(missing_participant, seed=1)

        wrong_balance = self.data.copy()
        wrong_balance.loc[
            wrong_balance["participant_id"].eq("Y01"),
            "participant_group",
        ] = "Old"
        with self.assertRaisesRegex(MOD09.Mod09Error, "16 Young and 16 Old"):
            MOD09.create_holdout_split(wrong_balance, seed=1)

    def test_rejects_missing_or_blank_participant_values(self):
        missing_column = self.data.drop(columns=["participant_group"])
        with self.assertRaisesRegex(MOD09.Mod09Error, "participant_group"):
            MOD09.create_holdout_split(missing_column, seed=1)

        blank_id = self.data.copy()
        blank_id.loc[blank_id.index[0], "participant_id"] = " "
        with self.assertRaisesRegex(MOD09.Mod09Error, "blank"):
            MOD09.create_holdout_split(blank_id, seed=1)

    def test_manifest_records_holdout_boundary_and_no_outcome_selection(self):
        split = MOD09.create_holdout_split(
            self.data,
            seed=MOD09.DEFAULT_SEED,
        )
        manifest = MOD09.build_manifest(
            split=split,
            seed=MOD09.DEFAULT_SEED,
            modeling_data_path=Path("outputs/modeling/modeling_data.csv"),
        )

        self.assertEqual(manifest["participant_count"], 32)
        self.assertEqual(manifest["training_participant_count"], 26)
        self.assertEqual(manifest["test_participant_count"], 6)
        self.assertEqual(manifest["training_group_counts"], {"Old": 13, "Young": 13})
        self.assertEqual(manifest["test_group_counts"], {"Old": 3, "Young": 3})
        self.assertEqual(
            manifest["selection_variables"],
            ["participant_id", "participant_group", "seed"],
        )
        self.assertFalse(manifest["outcomes_used_for_split_selection"])
        self.assertFalse(manifest["tmt_b_used_for_split_selection"])
        self.assertTrue(manifest["test_set_locked_for_final_evaluation"])

    def test_output_writer_refuses_overwrite(self):
        split = MOD09.create_holdout_split(self.data, seed=7)
        manifest = MOD09.build_manifest(
            split=split,
            seed=7,
            modeling_data_path=Path("modeling_data.csv"),
        )

        with tempfile.TemporaryDirectory() as directory:
            output_dir = Path(directory)
            paths = MOD09.write_outputs(
                split=split,
                manifest=manifest,
                output_dir=output_dir,
            )
            self.assertTrue(paths["split"].exists())
            self.assertTrue(paths["manifest"].exists())

            with self.assertRaisesRegex(
                FileExistsError,
                "Refusing to overwrite",
            ):
                MOD09.write_outputs(
                    split=split,
                    manifest=manifest,
                    output_dir=output_dir,
                )


if __name__ == "__main__":
    unittest.main()
