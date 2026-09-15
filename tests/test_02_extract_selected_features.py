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


class ProductGrabEventTests(unittest.TestCase):
    """Check FE-01.2 grab boundaries, product identity, and audit flags."""

    def require_grab_api(self):
        self.assertIsNotNone(EXTRACTOR)
        self.assertTrue(
            hasattr(EXTRACTOR, "detect_product_grab_events"),
            "FE-01.2 product-grab detection has not been implemented",
        )
        return EXTRACTOR

    def make_catalog(self):
        extractor = self.require_grab_api()
        return extractor.ProductCatalog(
            {
                (0, "EN"): ("Green Plant", "Red Apple"),
                (0, "NL"): ("Groene Plant", "Rode Appel"),
                (2, "EN"): ("Red Apple", "Blue Cereal Box"),
                (2, "NL"): ("Rode Appel", "Blauwe Doos Cornflakes"),
                (6, "EN"): ("Yellow Banana", "White Beer Box"),
                (6, "NL"): ("Gele Banaan", "Witte Bierkrat"),
                (10, "EN"): ("Green Broccoli", "Blue Water Bottle"),
                (10, "NL"): ("Groene Broccoli", "Blauwe Fles Water"),
            }
        )

    @staticmethod
    def tracker(
        times,
        *,
        right_states=None,
        right_objects=None,
        left_states=None,
        left_objects=None,
    ):
        count = len(times)
        return pd.DataFrame(
            {
                "time": times,
                "is_grabbing_right": right_states or [False] * count,
                "grabbed_object_right": right_objects or [""] * count,
                "is_grabbing_left": left_states or [False] * count,
                "grabbed_object_left": left_objects or [""] * count,
            }
        )

    def detect(self, tracker, *, difficulty=2, language="EN"):
        extractor = self.require_grab_api()
        return extractor.detect_product_grab_events(
            tracker,
            self.make_catalog(),
            difficulty,
            language,
        )

    def test_right_hand_grab_uses_recorded_start_release_and_duration(self):
        tracker = self.tracker(
            [0.00, 0.10, 0.30, 0.31],
            right_states=[False, True, True, False],
            right_objects=["", "Red Apple", "Red Apple", ""],
        )

        event = self.detect(tracker).events[0]

        self.assertEqual(event.hand, "right")
        self.assertEqual(event.raw_product_label, "Red Apple")
        self.assertAlmostEqual(event.grab_start_seconds, 0.10)
        self.assertAlmostEqual(event.grab_release_seconds, 0.31)
        self.assertAlmostEqual(event.grab_duration_seconds, 0.21)

    def test_left_hand_grab_records_actual_hand(self):
        tracker = self.tracker(
            [0.00, 0.20, 0.40],
            left_states=[True, True, False],
            left_objects=["Red Apple", "Red Apple", ""],
        )

        event = self.detect(tracker).events[0]

        self.assertEqual(event.hand, "left")

    def test_dutch_and_english_aliases_share_english_canonical_name(self):
        tracker = self.tracker(
            [0.00, 0.20, 0.31, 0.52],
            right_states=[True, False, True, False],
            right_objects=["Rode Appel", "", "Red Apple", ""],
        )

        events = self.detect(tracker).events

        self.assertEqual([event.canonical_product_name for event in events], ["red apple", "red apple"])

    def test_unity_instance_suffix_maps_to_base_product(self):
        tracker = self.tracker(
            [0.00, 0.25],
            right_states=[True, False],
            right_objects=["Red Apple (3)", ""],
        )

        event = self.detect(tracker).events[0]

        self.assertEqual(event.canonical_product_name, "red apple")
        self.assertEqual(event.raw_product_label, "Red Apple (3)")

    def test_repeated_on_list_product_is_not_first_time_twice(self):
        tracker = self.tracker(
            [0.00, 0.20, 0.40, 0.60],
            right_states=[True, False, True, False],
            right_objects=["Red Apple", "", "Red Apple", ""],
        )

        events = self.detect(tracker).events

        self.assertEqual([event.is_on_list for event in events], [True, True])
        self.assertEqual(
            [event.is_first_time_on_list for event in events], [True, False]
        )

    def test_plausible_product_outside_list_is_retained_for_audit(self):
        tracker = self.tracker(
            [0.00, 0.25],
            right_states=[True, False],
            right_objects=["Orange Santa Bottle", ""],
        )

        event = self.detect(tracker).events[0]

        self.assertEqual(event.canonical_product_name, "orange santa bottle")
        self.assertFalse(event.is_on_list)
        self.assertFalse(event.is_first_time_on_list)

    def test_same_product_segments_at_merge_boundary_are_one_event(self):
        tracker = self.tracker(
            [0.00, 0.20, 0.30, 0.50],
            right_states=[True, False, True, False],
            right_objects=["Red Apple", "", "Red Apple", ""],
        )

        events = self.detect(tracker).events

        self.assertEqual(len(events), 1)
        self.assertAlmostEqual(events[0].grab_start_seconds, 0.00)
        self.assertAlmostEqual(events[0].grab_release_seconds, 0.50)

    def test_same_product_segments_beyond_merge_gap_stay_separate(self):
        tracker = self.tracker(
            [0.00, 0.20, 0.31, 0.52],
            right_states=[True, False, True, False],
            right_objects=["Red Apple", "", "Red Apple", ""],
        )

        events = self.detect(tracker).events

        self.assertEqual(len(events), 2)

    def test_different_products_never_merge_across_short_gap(self):
        tracker = self.tracker(
            [0.00, 0.20, 0.25, 0.50],
            right_states=[True, False, True, False],
            right_objects=["Red Apple", "", "Blue Cereal Box", ""],
        )

        events = self.detect(tracker).events

        self.assertEqual(
            [event.canonical_product_name for event in events],
            ["red apple", "blue cereal box"],
        )

    def test_segment_shorter_than_minimum_duration_is_discarded(self):
        tracker = self.tracker(
            [0.00, 0.19],
            right_states=[True, False],
            right_objects=["Red Apple", ""],
        )

        result = self.detect(tracker)

        self.assertEqual(result.events, ())

    def test_segment_exactly_at_minimum_duration_is_retained(self):
        tracker = self.tracker(
            [0.00, 0.20],
            right_states=[True, False],
            right_objects=["Red Apple", ""],
        )

        result = self.detect(tracker)

        self.assertEqual(len(result.events), 1)
        self.assertAlmostEqual(result.events[0].grab_duration_seconds, 0.20)

    def test_open_ended_grab_is_discarded(self):
        tracker = self.tracker(
            [0.00, 0.30],
            right_states=[True, True],
            right_objects=["Red Apple", "Red Apple"],
        )

        result = self.detect(tracker)

        self.assertEqual(result.events, ())
        self.assertIn("open_ended_grab_discarded:right", result.warnings)

    def test_task_objects_and_placeholders_are_not_product_events(self):
        excluded = [
            "",
            "noObjectGrabbed",
            "notAssigned",
            "Tablet",
            "Cart",
            "Main Shelf",
            "NPC",
            "Cube",
        ]
        for label in excluded:
            with self.subTest(label=label):
                tracker = self.tracker(
                    [0.00, 0.25],
                    right_states=[True, False],
                    right_objects=[label, ""],
                )
                self.assertEqual(self.detect(tracker).events, ())

    def test_overlapping_left_and_right_grabs_remain_separate(self):
        tracker = self.tracker(
            [0.00, 0.30],
            right_states=[True, False],
            right_objects=["Red Apple", ""],
            left_states=[True, False],
            left_objects=["Blue Cereal Box", ""],
        )

        events = self.detect(tracker).events

        self.assertEqual([event.hand for event in events], ["left", "right"])
        self.assertTrue(all(event.overlaps_other_hand_grab for event in events))

    def test_missing_timestamp_breaks_grab_and_adds_warning(self):
        tracker = self.tracker(
            [0.00, 0.10, float("nan"), 0.40],
            right_states=[False, True, True, False],
            right_objects=["", "Red Apple", "Red Apple", ""],
        )

        result = self.detect(tracker)

        self.assertEqual(result.events, ())
        self.assertIn("missing_time_during_grab:right", result.warnings)

    def test_missing_hand_signals_warn_without_fabricating_events(self):
        extractor = self.require_grab_api()
        tracker = pd.DataFrame({"time": [0.00, 0.30]})

        result = extractor.detect_product_grab_events(
            tracker, self.make_catalog(), 2, "EN"
        )

        self.assertEqual(result.events, ())
        self.assertIn("grab_signal_unavailable:left", result.warnings)
        self.assertIn("grab_signal_unavailable:right", result.warnings)


if __name__ == "__main__":
    unittest.main()
