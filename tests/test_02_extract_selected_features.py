"""Synthetic tests for FE-01.1 input loading and trial setup."""

from __future__ import annotations

import contextlib
import importlib.util
import io
from pathlib import Path
import tempfile
import unittest

import pandas as pd


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = REPOSITORY_ROOT / "python" / "02_extract_selected_features.py"


def load_extractor_module():
    """Load the numbered script without requiring it to be a Python package."""
    if not MODULE_PATH.exists():
        return None
    specification = importlib.util.spec_from_file_location(
        "extract_selected_features", MODULE_PATH
    )
    if specification is None or specification.loader is None:
        return None
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


EXTRACTOR = load_extractor_module()


class InputAndTrialSetupTests(unittest.TestCase):
    """Check the public input contracts agreed for FE-01.1."""

    def require_extractor(self):
        self.assertIsNotNone(
            EXTRACTOR,
            "python/02_extract_selected_features.py has not been implemented",
        )
        return EXTRACTOR

    def test_semicolon_master_table_is_read_and_normalized(self):
        extractor = self.require_extractor()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "performance_1.csv"
            path.write_text(
                "participant;session;trial;group;condition;trial2;level3;language\n"
                "P1;s002;T004;Young;visual;t3;2;nl\n",
                encoding="utf-8",
            )

            table = extractor.standardize_trial_table(extractor.read_table(path))

        self.assertEqual(table.loc[0, "participant_id"], "P01")
        self.assertEqual(table.loc[0, "session_id"], "S002")
        self.assertEqual(
            table.loc[0, "source_tracker_csv_filename"],
            "data_collector_vr_sample_T004.csv",
        )
        self.assertEqual(table.loc[0, "participant_group"], "Young")
        self.assertEqual(table.loc[0, "condition_name"], "Visual")
        self.assertEqual(table.loc[0, "difficulty_level"], 2)
        self.assertEqual(table.loc[0, "trial_order"], "T3")
        self.assertEqual(table.loc[0, "language"], "NL")

    def test_missing_master_columns_are_reported_together(self):
        extractor = self.require_extractor()
        incomplete = pd.DataFrame({"participant": ["P01"]})

        with self.assertRaises(ValueError) as raised:
            extractor.standardize_trial_table(incomplete)

        message = str(raised.exception)
        self.assertIn("session", message)
        self.assertIn("trial", message)
        self.assertIn("condition", message)
        self.assertIn("difficulty", message)
        self.assertIn("trial_order", message)

    def test_correct_lists_preserve_both_languages_at_each_difficulty(self):
        extractor = self.require_extractor()
        definitions = []
        for difficulty in (0, 2, 6, 10):
            definitions.append(
                f"correctLists.level{difficulty}.EN = {{'English {difficulty} A', "
                f"'English {difficulty} B'}};"
            )
            definitions.append(
                f"correctLists.level{difficulty}.NL = {{'Dutch {difficulty} A', "
                f"'Dutch {difficulty} B'}};"
            )

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "CorrectLists.txt"
            path.write_text("\n".join(definitions), encoding="utf-8")
            catalog = extractor.parse_correct_lists(path)

        for difficulty in (0, 2, 6, 10):
            self.assertEqual(
                catalog.products_for(difficulty, "EN"),
                (f"English {difficulty} A", f"English {difficulty} B"),
            )
            self.assertEqual(
                catalog.products_for(difficulty, "NL"),
                (f"Dutch {difficulty} A", f"Dutch {difficulty} B"),
            )

    def test_trial_paths_keep_participant_and_session_scope(self):
        extractor = self.require_extractor()
        row = {
            "participant_id": "P01",
            "session_id": "S002",
            "source_tracker_csv_filename": "data_collector_vr_sample_T004.csv",
        }
        raw_root = Path("C:/synthetic/raw")

        paths = extractor.build_trial_paths(row, raw_root)

        expected_base = raw_root / "P1" / "P01_HMD_Data" / "P01" / "S002"
        self.assertEqual(
            paths.tracker,
            expected_base / "trackers" / "data_collector_vr_sample_T004.csv",
        )
        self.assertEqual(
            paths.participant_details,
            expected_base / "session_info" / "participant_details.csv",
        )

    def test_tracker_preserves_recorded_times_and_marks_invalid_time_missing(self):
        extractor = self.require_extractor()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "tracker.csv"
            path.write_text(
                "time,focus_object_name,is_grabbing_right\n"
                "0.00,Shelf,false\n"
                "0.03,Product,true\n"
                "invalid,Product,false\n"
                "0.11,Cart,false\n",
                encoding="utf-8",
            )

            result = extractor.load_tracker_table(path)

        self.assertEqual(result.table.loc[0, "time"], 0.00)
        self.assertEqual(result.table.loc[1, "time"], 0.03)
        self.assertTrue(pd.isna(result.table.loc[2, "time"]))
        self.assertEqual(result.table.loc[3, "time"], 0.11)

    def test_tracker_without_time_column_is_rejected(self):
        extractor = self.require_extractor()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "tracker.csv"
            path.write_text("frame,focus_object_name\n1,Shelf\n", encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "time"):
                extractor.load_tracker_table(path)

    def test_tracker_without_any_usable_time_is_rejected(self):
        extractor = self.require_extractor()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "tracker.csv"
            path.write_text("time\ninvalid\nmissing\n", encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "usable.*time"):
                extractor.load_tracker_table(path)

    def test_missing_tracker_columns_are_missing_and_reported(self):
        extractor = self.require_extractor()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "tracker.csv"
            path.write_text("time\n0.00\n0.03\n", encoding="utf-8")

            result = extractor.load_tracker_table(path)

        self.assertIn("focus_object_name", result.table.columns)
        self.assertIn("is_grabbing_right", result.table.columns)
        self.assertTrue(result.table["focus_object_name"].isna().all())
        self.assertTrue(result.table["is_grabbing_right"].isna().all())
        self.assertIn("missing_tracker_column:focus_object_name", result.warnings)
        self.assertIn("missing_tracker_column:is_grabbing_right", result.warnings)

    def test_participant_hand_is_normalized(self):
        extractor = self.require_extractor()
        examples = {"R": "right", "right-handed": "right", "L": "left", "Left": "left"}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "participant_details.csv"
            for source, expected in examples.items():
                with self.subTest(source=source):
                    path.write_text(f"side,skin\n{source},synthetic\n", encoding="utf-8")
                    hand, warnings = extractor.read_participant_hand(path)
                    self.assertEqual(hand, expected)
                    self.assertEqual(warnings, [])

    def test_unavailable_participant_hand_remains_missing(self):
        extractor = self.require_extractor()
        with tempfile.TemporaryDirectory() as directory:
            missing_path = Path(directory) / "participant_details.csv"
            hand, warnings = extractor.read_participant_hand(missing_path)

        self.assertIsNone(hand)
        self.assertEqual(warnings, ["participant_details_file_missing"])

    def test_cli_loads_one_synthetic_trial_without_writing_feature_outputs(self):
        extractor = self.require_extractor()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            performance = root / "performance_1.csv"
            performance.write_text(
                "participant;session;trial;group;condition;trial2;level3;language\n"
                "P1;S002;T004;Young;Visual;T3;2;EN\n",
                encoding="utf-8",
            )
            correct_lists = root / "CorrectLists.txt"
            correct_lists.write_text(
                "\n".join(
                    f"correctLists.level{difficulty}.{language} = "
                    f"{{'{language} product {difficulty}'}};"
                    for difficulty in (0, 2, 6, 10)
                    for language in ("EN", "NL")
                ),
                encoding="utf-8",
            )
            trial_base = (
                root / "raw" / "P1" / "P01_HMD_Data" / "P01" / "S002"
            )
            tracker_directory = trial_base / "trackers"
            details_directory = trial_base / "session_info"
            tracker_directory.mkdir(parents=True)
            details_directory.mkdir(parents=True)
            (tracker_directory / "data_collector_vr_sample_T004.csv").write_text(
                "time,focus_object_name\n0.00,Shelf\n0.03,Product\n",
                encoding="utf-8",
            )
            (details_directory / "participant_details.csv").write_text(
                "side,skin\nR,synthetic\n", encoding="utf-8"
            )

            captured = io.StringIO()
            with contextlib.redirect_stdout(captured):
                exit_code = extractor.main(
                    [
                        "--performance",
                        str(performance),
                        "--raw-root",
                        str(root / "raw"),
                        "--correct-lists",
                        str(correct_lists),
                        "--limit",
                        "1",
                    ]
                )

            self.assertEqual(exit_code, 0)
            self.assertIn("Loaded trial rows: 1", captured.getvalue())
            self.assertIn("Tracker files read: 1", captured.getvalue())
            self.assertFalse((root / "product_grab_features.csv").exists())
            self.assertFalse((root / "trial_features.csv").exists())


if __name__ == "__main__":
    unittest.main()
