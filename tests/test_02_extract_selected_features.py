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


class FocusEpisodeTests(unittest.TestCase):
    """Check FE-01.3 engine-labelled focus episode boundaries and validity."""

    def require_focus_api(self):
        self.assertIsNotNone(EXTRACTOR)
        self.assertTrue(
            hasattr(EXTRACTOR, "detect_focus_episodes"),
            "FE-01.3 focus-episode detection has not been implemented",
        )
        return EXTRACTOR

    def make_catalog(self):
        extractor = self.require_focus_api()
        return extractor.ProductCatalog(
            {
                (0, "EN"): ("Green Plant",),
                (0, "NL"): ("Groene Plant",),
                (2, "EN"): ("Red Apple",),
                (2, "NL"): ("Rode Appel",),
                (6, "EN"): ("Yellow Banana",),
                (6, "NL"): ("Gele Banaan",),
                (10, "EN"): ("Green Broccoli",),
                (10, "NL"): ("Groene Broccoli",),
            }
        )

    @staticmethod
    def tracker(
        times,
        *,
        names=None,
        tags=None,
        left_blinks=None,
        right_blinks=None,
    ):
        count = len(times)
        data = {
            "time": times,
            "focus_object_name": names or [""] * count,
            "focus_object_tag": tags or [""] * count,
        }
        if left_blinks is not None:
            data["is_left_eye_blinking"] = left_blinks
        if right_blinks is not None:
            data["is_right_eye_blinking"] = right_blinks
        return pd.DataFrame(data)

    def detect(self, tracker):
        extractor = self.require_focus_api()
        return extractor.detect_focus_episodes(tracker, self.make_catalog())

    def test_episode_uses_exact_recorded_start_end_and_duration(self):
        tracker = self.tracker(
            [0.00, 0.03, 0.11],
            names=["Red Apple", "Red Apple", ""],
            tags=["MainShelf", "MainShelf", "notAssigned"],
            left_blinks=[False] * 3,
            right_blinks=[False] * 3,
        )

        event = self.detect(tracker).episodes[0]

        self.assertAlmostEqual(event.focus_start_seconds, 0.00)
        self.assertAlmostEqual(event.focus_end_seconds, 0.11)
        self.assertAlmostEqual(event.focus_duration_seconds, 0.11)

    def test_identical_name_and_tag_rows_form_one_episode(self):
        tracker = self.tracker(
            [0.00, 0.05, 0.10],
            names=["Red Apple", "Red Apple", ""],
            tags=["MainShelf", "MainShelf", ""],
            left_blinks=[False] * 3,
            right_blinks=[False] * 3,
        )

        self.assertEqual(len(self.detect(tracker).episodes), 1)

    def test_label_change_closes_and_starts_at_same_timestamp(self):
        tracker = self.tracker(
            [0.00, 0.20, 0.40],
            names=["Red Apple", "Green Plant", ""],
            tags=["MainShelf", "MainShelf", ""],
            left_blinks=[False] * 3,
            right_blinks=[False] * 3,
        )

        events = self.detect(tracker).episodes

        self.assertEqual(len(events), 2)
        self.assertAlmostEqual(events[0].focus_end_seconds, 0.20)
        self.assertAlmostEqual(events[1].focus_start_seconds, 0.20)

    def test_language_aliases_share_english_canonical_name(self):
        tracker = self.tracker(
            [0.00, 0.10, 0.20],
            names=["Rode Appel", "Red Apple", ""],
            tags=["MainShelf", "MainShelf", ""],
            left_blinks=[False] * 3,
            right_blinks=[False] * 3,
        )

        events = self.detect(tracker).episodes

        self.assertEqual(len(events), 1)
        self.assertEqual(events[0].canonical_focus_name, "red apple")

    def test_unity_suffix_does_not_split_product_episode(self):
        tracker = self.tracker(
            [0.00, 0.10, 0.20],
            names=["Red Apple (3)", "Red Apple", ""],
            tags=["MainShelf", "MainShelf", ""],
            left_blinks=[False] * 3,
            right_blinks=[False] * 3,
        )

        self.assertEqual(len(self.detect(tracker).episodes), 1)

    def test_original_and_cleaned_labels_are_preserved(self):
        tracker = self.tracker(
            [0.00, 0.20],
            names=["Red Apple (3)", ""],
            tags=["MainShelf (1)", ""],
            left_blinks=[False, False],
            right_blinks=[False, False],
        )

        event = self.detect(tracker).episodes[0]

        self.assertEqual(event.raw_focus_name, "Red Apple (3)")
        self.assertEqual(event.canonical_focus_name, "red apple")
        self.assertEqual(event.raw_focus_tag, "MainShelf (1)")
        self.assertEqual(event.cleaned_focus_tag, "mainshelf")

    def test_placeholders_do_not_form_episodes(self):
        for placeholder in ("", "notAssigned", "none", "null", "nan"):
            with self.subTest(placeholder=placeholder):
                tracker = self.tracker(
                    [0.00, 0.20],
                    names=[placeholder, ""],
                    tags=[placeholder, ""],
                    left_blinks=[False, False],
                    right_blinks=[False, False],
                )
                self.assertEqual(self.detect(tracker).episodes, ())

    def test_blink_closes_episode_and_blinking_rows_do_not_start_one(self):
        tracker = self.tracker(
            [0.00, 0.10, 0.20, 0.30],
            names=["Red Apple"] * 4,
            tags=["MainShelf"] * 4,
            left_blinks=[False, True, True, False],
            right_blinks=[False] * 4,
        )

        result = self.detect(tracker)

        self.assertEqual(len(result.episodes), 1)
        self.assertAlmostEqual(result.episodes[0].focus_end_seconds, 0.10)
        self.assertIn("open_ended_focus_discarded", result.warnings)

    def test_either_eye_blink_invalidates_a_row(self):
        for eye in ("left", "right"):
            with self.subTest(eye=eye):
                left = [eye == "left", False]
                right = [eye == "right", False]
                tracker = self.tracker(
                    [0.00, 0.20],
                    names=["Red Apple", ""],
                    tags=["MainShelf", ""],
                    left_blinks=left,
                    right_blinks=right,
                )
                self.assertEqual(self.detect(tracker).episodes, ())

    def test_missing_one_blink_signal_warns_and_uses_available_eye(self):
        tracker = self.tracker(
            [0.00, 0.20],
            names=["Red Apple", ""],
            tags=["MainShelf", ""],
            left_blinks=[False, False],
        )

        result = self.detect(tracker)

        self.assertEqual(len(result.episodes), 1)
        self.assertIn("blink_signal_unavailable:right", result.warnings)

    def test_missing_both_blink_signals_warn_but_keep_labelled_focus(self):
        tracker = self.tracker(
            [0.00, 0.20],
            names=["Red Apple", ""],
            tags=["MainShelf", ""],
        )

        result = self.detect(tracker)

        self.assertEqual(len(result.episodes), 1)
        self.assertIn("blink_signal_unavailable:left", result.warnings)
        self.assertIn("blink_signal_unavailable:right", result.warnings)

    def test_missing_name_or_tag_warns_and_uses_available_field(self):
        for missing, present in (
            ("focus_object_name", "focus_object_tag"),
            ("focus_object_tag", "focus_object_name"),
        ):
            with self.subTest(missing=missing):
                tracker = pd.DataFrame(
                    {
                        "time": [0.00, 0.20],
                        present: ["MainShelf", ""],
                        "is_left_eye_blinking": [False, False],
                        "is_right_eye_blinking": [False, False],
                    }
                )
                result = self.detect(tracker)
                self.assertEqual(len(result.episodes), 1)
                self.assertIn(f"focus_signal_unavailable:{missing}", result.warnings)

    def test_missing_timestamp_breaks_episode_and_warns(self):
        tracker = self.tracker(
            [0.00, float("nan"), 0.20],
            names=["Red Apple", "Red Apple", ""],
            tags=["MainShelf", "MainShelf", ""],
            left_blinks=[False] * 3,
            right_blinks=[False] * 3,
        )

        result = self.detect(tracker)

        self.assertEqual(result.episodes, ())
        self.assertIn("missing_time_during_focus", result.warnings)

    def test_nonmonotonic_boundary_warns_without_negative_episode(self):
        tracker = self.tracker(
            [0.20, 0.10],
            names=["Red Apple", ""],
            tags=["MainShelf", ""],
            left_blinks=[False, False],
            right_blinks=[False, False],
        )

        result = self.detect(tracker)

        self.assertEqual(result.episodes, ())
        self.assertIn("non_monotonic_focus_time", result.warnings)

    def test_open_episode_at_final_row_is_discarded(self):
        tracker = self.tracker(
            [0.00, 0.20],
            names=["Red Apple", "Red Apple"],
            tags=["MainShelf", "MainShelf"],
            left_blinks=[False, False],
            right_blinks=[False, False],
        )

        result = self.detect(tracker)

        self.assertEqual(result.episodes, ())
        self.assertIn("open_ended_focus_discarded", result.warnings)

    def test_zero_duration_complete_episode_is_retained(self):
        tracker = self.tracker(
            [0.10, 0.10],
            names=["Red Apple", ""],
            tags=["MainShelf", ""],
            left_blinks=[False, False],
            right_blinks=[False, False],
        )

        event = self.detect(tracker).episodes[0]

        self.assertEqual(event.focus_duration_seconds, 0.0)

    def test_episode_order_and_warnings_are_unique(self):
        tracker = self.tracker(
            [0.00, 0.10, 0.20, 0.30],
            names=["Red Apple", "", "Green Plant", ""],
            tags=["MainShelf", "", "MainShelf", ""],
        )

        result = self.detect(tracker)

        self.assertEqual(
            [event.focus_start_seconds for event in result.episodes], [0.00, 0.20]
        )
        self.assertEqual(len(result.warnings), len(set(result.warnings)))


class ListVisitTests(unittest.TestCase):
    """Check FE-01.4 tablet/list visits derived from focus episodes only."""

    def require_list_visit_api(self):
        self.assertIsNotNone(EXTRACTOR)
        self.assertTrue(
            hasattr(EXTRACTOR, "detect_list_visits"),
            "FE-01.4 list-visit detection has not been implemented",
        )
        return EXTRACTOR

    def focus_episode(
        self,
        start,
        end,
        *,
        name="",
        tag="",
        raw_name=None,
        raw_tag=None,
    ):
        extractor = self.require_list_visit_api()
        return extractor.FocusEpisode(
            canonical_focus_name=name,
            raw_focus_name=name if raw_name is None else raw_name,
            cleaned_focus_tag=tag,
            raw_focus_tag=tag if raw_tag is None else raw_tag,
            focus_start_seconds=start,
            focus_end_seconds=end,
            focus_duration_seconds=end - start,
        )

    def detect(self, episodes):
        extractor = self.require_list_visit_api()
        return extractor.detect_list_visits(episodes)

    def test_tablet_tag_creates_visit_with_focus_episode_timing(self):
        visit = self.detect(
            [self.focus_episode(0.10, 0.35, name="samsung_tab", tag="tablet")]
        )[0]

        self.assertAlmostEqual(visit.list_visit_start_seconds, 0.10)
        self.assertAlmostEqual(visit.list_visit_end_seconds, 0.35)
        self.assertAlmostEqual(visit.list_visit_duration_seconds, 0.25)
        self.assertTrue(visit.is_initial_view)

    def test_known_tablet_names_work_without_a_tablet_tag(self):
        visits = self.detect(
            [
                self.focus_episode(0.00, 0.10, name="samsung_tab", tag="ui"),
                self.focus_episode(0.20, 0.30, name="tablet", tag="ui"),
            ]
        )

        self.assertEqual(len(visits), 2)
        self.assertEqual([visit.is_initial_view for visit in visits], [True, False])

    def test_non_tablet_focus_episodes_do_not_create_visits(self):
        visits = self.detect(
            [
                self.focus_episode(0.00, 0.10, name="red apple", tag="mainshelf"),
                self.focus_episode(0.20, 0.30, name="cart", tag="cart"),
                self.focus_episode(0.40, 0.50, name="npc", tag="npc"),
            ]
        )

        self.assertEqual(visits, ())

    def test_first_chronological_visit_is_initial_and_later_visits_are_rechecks(self):
        visits = self.detect(
            [
                self.focus_episode(0.10, 0.20, name="samsung_tab", tag="tablet"),
                self.focus_episode(0.40, 0.55, name="samsung_tab", tag="tablet"),
                self.focus_episode(0.70, 0.90, name="samsung_tab", tag="tablet"),
            ]
        )

        self.assertEqual([visit.is_initial_view for visit in visits], [True, False, False])

    def test_recheck_before_any_grab_is_still_retained(self):
        visits = self.detect(
            [
                self.focus_episode(0.00, 0.10, name="samsung_tab", tag="tablet"),
                self.focus_episode(0.15, 0.25, name="samsung_tab", tag="tablet"),
            ]
        )

        self.assertEqual(len(visits), 2)
        self.assertFalse(visits[1].is_initial_view)

    def test_distinct_tablet_focus_episodes_are_not_merged(self):
        visits = self.detect(
            [
                self.focus_episode(0.00, 0.10, name="samsung_tab", tag="tablet"),
                self.focus_episode(0.10, 0.20, name="samsung_tab", tag="tablet"),
            ]
        )

        self.assertEqual(len(visits), 2)
        self.assertEqual(
            [visit.list_visit_duration_seconds for visit in visits], [0.10, 0.10]
        )

    def test_zero_duration_complete_visit_is_retained(self):
        visit = self.detect(
            [self.focus_episode(0.10, 0.10, name="samsung_tab", tag="tablet")]
        )[0]

        self.assertEqual(visit.list_visit_duration_seconds, 0.0)

    def test_no_tablet_focus_episode_returns_empty_tuple(self):
        self.assertEqual(
            self.detect([self.focus_episode(0.00, 0.10, name="red apple")]), ()
        )

    def test_unsorted_focus_episodes_produce_chronological_visits(self):
        visits = self.detect(
            [
                self.focus_episode(0.40, 0.50, name="samsung_tab", tag="tablet"),
                self.focus_episode(0.10, 0.20, name="samsung_tab", tag="tablet"),
            ]
        )

        self.assertEqual(
            [visit.list_visit_start_seconds for visit in visits], [0.10, 0.40]
        )
        self.assertEqual([visit.is_initial_view for visit in visits], [True, False])


class SearchIntervalTests(unittest.TestCase):
    """Check FE-01.5 shared search intervals for qualifying product grabs."""

    def require_search_api(self):
        self.assertIsNotNone(EXTRACTOR)
        self.assertTrue(
            hasattr(EXTRACTOR, "detect_search_intervals"),
            "FE-01.5 search-interval detection has not been implemented",
        )
        return EXTRACTOR

    def grab(self, start, *, product="red apple", release=None, on_list=True, first=True):
        extractor = self.require_search_api()
        release = start + 0.20 if release is None else release
        return extractor.ProductGrabEvent(
            canonical_product_name=product,
            raw_product_label=product,
            hand="right",
            grab_start_seconds=start,
            grab_release_seconds=release,
            grab_duration_seconds=release - start,
            is_on_list=on_list,
            is_first_time_on_list=first,
            overlaps_other_hand_grab=False,
        )

    def focus(self, start, end, *, product="red apple"):
        extractor = self.require_search_api()
        return extractor.FocusEpisode(
            canonical_focus_name=product,
            raw_focus_name=product,
            cleaned_focus_tag="mainshelf",
            raw_focus_tag="MainShelf",
            focus_start_seconds=start,
            focus_end_seconds=end,
            focus_duration_seconds=end - start,
        )

    def visit(self, start, end, *, initial=False):
        extractor = self.require_search_api()
        return extractor.ListVisit(
            list_visit_start_seconds=start,
            list_visit_end_seconds=end,
            list_visit_duration_seconds=end - start,
            is_initial_view=initial,
        )

    @staticmethod
    def tracker(times, blackouts=None):
        data = {"time": times}
        if blackouts is not None:
            data["enableBlackout"] = blackouts
        return pd.DataFrame(data)

    def detect(self, tracker, grabs, visits=(), focus_episodes=()):
        extractor = self.require_search_api()
        return extractor.detect_search_intervals(
            tracker, grabs, visits, focus_episodes
        )

    def test_first_qualifying_grab_starts_at_first_non_blackout_time(self):
        result = self.detect(
            self.tracker([0.00, 0.10, 0.40], [True, False, False]),
            [self.grab(0.50)],
            focus_episodes=[self.focus(0.30, 0.35)],
        )

        interval = result.intervals[0]

        self.assertAlmostEqual(interval.search_start_seconds, 0.10)
        self.assertAlmostEqual(interval.first_target_focus_seconds, 0.30)
        self.assertTrue(interval.is_valid)

    def test_leading_blackout_rows_are_excluded_from_activity_start(self):
        result = self.detect(
            self.tracker([0.00, 0.05, 0.10], [True, True, False]),
            [self.grab(0.40)],
            focus_episodes=[self.focus(0.20, 0.25)],
        )

        self.assertAlmostEqual(result.intervals[0].search_start_seconds, 0.10)

    def test_missing_or_all_blackout_activity_data_invalidates_first_interval(self):
        for tracker in (
            self.tracker([0.00, 0.10]),
            self.tracker([0.00, 0.10], [True, True]),
        ):
            with self.subTest(columns=list(tracker.columns)):
                result = self.detect(
                    tracker,
                    [self.grab(0.40)],
                    focus_episodes=[self.focus(0.20, 0.25)],
                )
                self.assertFalse(result.intervals[0].is_valid)
                self.assertIsNone(result.intervals[0].search_start_seconds)
                self.assertIn("usable_activity_start_unavailable", result.warnings)

    def test_later_interval_uses_previous_qualifying_grab_start_not_release(self):
        result = self.detect(
            self.tracker([0.00], [False]),
            [
                self.grab(0.20, product="red apple", release=0.90),
                self.grab(0.60, product="green plant"),
            ],
            focus_episodes=[
                self.focus(0.10, 0.15, product="red apple"),
                self.focus(0.50, 0.55, product="green plant"),
            ],
        )

        self.assertAlmostEqual(result.intervals[1].search_start_seconds, 0.20)

    def test_repeated_and_off_list_grabs_do_not_create_or_replace_intervals(self):
        result = self.detect(
            self.tracker([0.00], [False]),
            [
                self.grab(0.20, product="red apple"),
                self.grab(0.30, product="red apple", first=False),
                self.grab(0.40, product="orange bottle", on_list=False, first=False),
                self.grab(0.60, product="green plant"),
            ],
            focus_episodes=[
                self.focus(0.10, 0.15, product="red apple"),
                self.focus(0.50, 0.55, product="green plant"),
            ],
        )

        self.assertEqual(len(result.intervals), 2)
        self.assertAlmostEqual(result.intervals[1].search_start_seconds, 0.20)

    def test_later_completed_list_visit_overrides_previous_grab_start(self):
        result = self.detect(
            self.tracker([0.00], [False]),
            [self.grab(0.20), self.grab(0.80, product="green plant")],
            visits=[self.visit(0.40, 0.60)],
            focus_episodes=[
                self.focus(0.10, 0.15),
                self.focus(0.70, 0.75, product="green plant"),
            ],
        )

        self.assertAlmostEqual(result.intervals[1].search_start_seconds, 0.60)

    def test_earlier_list_visit_does_not_override_later_previous_grab(self):
        result = self.detect(
            self.tracker([0.00], [False]),
            [self.grab(0.40), self.grab(0.80, product="green plant")],
            visits=[self.visit(0.10, 0.20)],
            focus_episodes=[
                self.focus(0.30, 0.35),
                self.focus(0.70, 0.75, product="green plant"),
            ],
        )

        self.assertAlmostEqual(result.intervals[1].search_start_seconds, 0.40)

    def test_list_visit_after_current_grab_is_ignored(self):
        result = self.detect(
            self.tracker([0.00], [False]),
            [self.grab(0.20), self.grab(0.70, product="green plant")],
            visits=[self.visit(0.65, 0.75)],
            focus_episodes=[
                self.focus(0.10, 0.15),
                self.focus(0.60, 0.65, product="green plant"),
            ],
        )

        self.assertAlmostEqual(result.intervals[1].search_start_seconds, 0.20)

    def test_earliest_matching_target_focus_ends_interval(self):
        result = self.detect(
            self.tracker([0.00], [False]),
            [self.grab(0.80)],
            focus_episodes=[self.focus(0.50, 0.55), self.focus(0.30, 0.35)],
        )

        self.assertAlmostEqual(result.intervals[0].first_target_focus_seconds, 0.30)

    def test_target_focus_overlapping_search_start_is_clipped(self):
        result = self.detect(
            self.tracker([0.00], [False]),
            [self.grab(0.20), self.grab(0.80, product="green plant")],
            focus_episodes=[
                self.focus(0.10, 0.15),
                self.focus(0.10, 0.70, product="green plant"),
            ],
        )

        interval = result.intervals[1]
        self.assertAlmostEqual(interval.search_start_seconds, 0.20)
        self.assertAlmostEqual(interval.first_target_focus_seconds, 0.20)

    def test_focus_after_grab_does_not_qualify(self):
        result = self.detect(
            self.tracker([0.00], [False]),
            [self.grab(0.40)],
            focus_episodes=[self.focus(0.50, 0.60)],
        )

        self.assertFalse(result.intervals[0].is_valid)
        self.assertIn("target_focus_not_found", result.warnings)

    def test_missing_target_focus_is_invalid_without_fabricated_endpoint(self):
        result = self.detect(self.tracker([0.00], [False]), [self.grab(0.40)])

        interval = result.intervals[0]
        self.assertFalse(interval.is_valid)
        self.assertIsNone(interval.first_target_focus_seconds)
        self.assertIn("target_focus_not_found", result.warnings)

    def test_canonical_product_aliases_match_focus_episodes(self):
        result = self.detect(
            self.tracker([0.00], [False]),
            [self.grab(0.40, product="red apple")],
            focus_episodes=[self.focus(0.20, 0.25, product="red apple")],
        )

        self.assertTrue(result.intervals[0].is_valid)

    def test_intervals_are_ordered_by_qualifying_grab_start(self):
        result = self.detect(
            self.tracker([0.00], [False]),
            [
                self.grab(0.80, product="green plant"),
                self.grab(0.30, product="red apple"),
            ],
            focus_episodes=[
                self.focus(0.20, 0.25, product="red apple"),
                self.focus(0.70, 0.75, product="green plant"),
            ],
        )

        self.assertEqual(
            [interval.qualifying_grab_start_seconds for interval in result.intervals],
            [0.30, 0.80],
        )


class PaceAndListRecheckAggregationTests(unittest.TestCase):
    """Check FE-01.6 pace and list-recheck trial aggregation."""

    def require_aggregation_api(self):
        self.assertIsNotNone(EXTRACTOR)
        self.assertTrue(
            hasattr(EXTRACTOR, "aggregate_trial_pace_and_list_rechecks"),
            "FE-01.6 pace and list-recheck aggregation has not been implemented",
        )
        return EXTRACTOR

    @staticmethod
    def grab(start, *, product="red apple", on_list=True, first=True):
        return EXTRACTOR.ProductGrabEvent(
            canonical_product_name=product,
            raw_product_label=product,
            hand="right",
            grab_start_seconds=start,
            grab_release_seconds=start + 0.20,
            grab_duration_seconds=0.20,
            is_on_list=on_list,
            is_first_time_on_list=first,
            overlaps_other_hand_grab=False,
        )

    @staticmethod
    def visit(start, end, *, initial=False):
        return EXTRACTOR.ListVisit(
            list_visit_start_seconds=start,
            list_visit_end_seconds=end,
            list_visit_duration_seconds=end - start,
            is_initial_view=initial,
        )

    def aggregate(self, grabs=(), visits=()):
        extractor = self.require_aggregation_api()
        return extractor.aggregate_trial_pace_and_list_rechecks(grabs, visits)

    def test_uneven_qualifying_grab_intervals_use_exact_median_pace(self):
        summary = self.aggregate(
            [self.grab(1.00), self.grab(1.30), self.grab(2.20), self.grab(4.00)]
        )

        self.assertAlmostEqual(
            summary.median_time_between_qualifying_grabs_seconds, 0.90
        )

    def test_qualifying_grabs_are_ordered_by_recorded_start_time(self):
        summary = self.aggregate([self.grab(3.00), self.grab(1.00), self.grab(2.00)])

        self.assertAlmostEqual(
            summary.median_time_between_qualifying_grabs_seconds, 1.00
        )

    def test_fewer_than_two_qualifying_grabs_leave_pace_missing(self):
        self.assertIsNone(
            self.aggregate([self.grab(1.00)]).median_time_between_qualifying_grabs_seconds
        )
        self.assertIsNone(
            self.aggregate().median_time_between_qualifying_grabs_seconds
        )

    def test_repeated_and_off_list_grabs_do_not_affect_pace(self):
        summary = self.aggregate(
            [
                self.grab(1.00),
                self.grab(1.20, first=False),
                self.grab(1.40, product="off list product", on_list=False, first=False),
                self.grab(2.00, product="green plant"),
            ]
        )

        self.assertAlmostEqual(
            summary.median_time_between_qualifying_grabs_seconds, 1.00
        )

    def test_initial_visit_only_means_zero_rechecks_and_zero_duration(self):
        summary = self.aggregate(visits=[self.visit(0.10, 0.40, initial=True)])

        self.assertEqual(summary.list_recheck_count, 0)
        self.assertAlmostEqual(summary.total_list_recheck_duration_seconds, 0.0)

    def test_later_visits_each_count_and_contribute_full_duration(self):
        summary = self.aggregate(
            visits=[
                self.visit(0.10, 0.40, initial=True),
                self.visit(1.00, 1.25),
                self.visit(2.00, 2.60),
            ]
        )

        self.assertEqual(summary.list_recheck_count, 2)
        self.assertAlmostEqual(summary.total_list_recheck_duration_seconds, 0.85)

    def test_rechecks_before_any_grab_are_included(self):
        summary = self.aggregate(
            grabs=[self.grab(3.00)],
            visits=[self.visit(0.10, 0.20, initial=True), self.visit(0.30, 0.80)],
        )

        self.assertEqual(summary.list_recheck_count, 1)
        self.assertAlmostEqual(summary.total_list_recheck_duration_seconds, 0.50)

    def test_zero_duration_recheck_counts_but_adds_zero_seconds(self):
        summary = self.aggregate(
            visits=[self.visit(0.10, 0.20, initial=True), self.visit(1.00, 1.00)]
        )

        self.assertEqual(summary.list_recheck_count, 1)
        self.assertAlmostEqual(summary.total_list_recheck_duration_seconds, 0.0)

    def test_unsorted_visits_keep_identity_and_aggregate_all_rechecks(self):
        summary = self.aggregate(
            visits=[
                self.visit(2.00, 2.40),
                self.visit(0.10, 0.20, initial=True),
                self.visit(1.00, 1.30),
            ]
        )

        self.assertEqual(summary.list_recheck_count, 2)
        self.assertAlmostEqual(summary.total_list_recheck_duration_seconds, 0.70)


class LocatingFeatureTests(unittest.TestCase):
    """Check FE-01.7 locating, irrelevant-focus and head-turning measures."""

    def require_api(self):
        self.assertIsNotNone(EXTRACTOR)
        self.assertTrue(
            hasattr(EXTRACTOR, "aggregate_trial_locating_features"),
            "FE-01.7 locating aggregation has not been implemented",
        )
        return EXTRACTOR

    @staticmethod
    def interval(start, target, *, product="red apple", valid=True):
        return EXTRACTOR.SearchInterval(
            canonical_product_name=product,
            qualifying_grab_start_seconds=(start + 0.10 if target is None else target + 0.10),
            search_start_seconds=start,
            first_target_focus_seconds=target,
            is_valid=valid,
        )

    @staticmethod
    def focus(start, end, *, name="red apple", tag="mainshelf"):
        return EXTRACTOR.FocusEpisode(
            canonical_focus_name=name,
            raw_focus_name=name,
            cleaned_focus_tag=tag,
            raw_focus_tag=tag,
            focus_start_seconds=start,
            focus_end_seconds=end,
            focus_duration_seconds=end - start,
        )

    @staticmethod
    def tracker(times, rotations=None):
        data = {"time": times}
        if rotations is not None:
            data.update(
                {
                    "hmd_rot_x": [row[0] for row in rotations],
                    "hmd_rot_y": [row[1] for row in rotations],
                    "hmd_rot_z": [row[2] for row in rotations],
                    "hmd_rot_w": [row[3] for row in rotations],
                }
            )
        return pd.DataFrame(data)

    def aggregate(self, intervals, focus=(), tracker=None):
        extractor = self.require_api()
        if tracker is None:
            tracker = self.tracker([0.0, 1.0])
        return extractor.aggregate_trial_locating_features(intervals, focus, tracker)

    def test_exact_locating_duration_and_median(self):
        summary = self.aggregate(
            [self.interval(1.0, 2.5), self.interval(3.0, 5.0)]
        )

        self.assertAlmostEqual(summary.median_time_to_target_seconds, 1.75)

    def test_missing_target_endpoint_is_excluded_and_reported(self):
        interval = self.interval(1.0, None, valid=False)
        interval = interval._replace(first_target_focus_seconds=None)
        summary = self.aggregate([interval])

        self.assertIsNone(summary.median_time_to_target_seconds)
        self.assertIn("locating_time_unavailable", summary.warnings)

    def test_focus_durations_are_clipped_to_search_boundaries(self):
        summary = self.aggregate(
            [self.interval(2.0, 8.0)],
            [self.focus(0.0, 10.0, name="npc", tag="npc")],
        )

        self.assertAlmostEqual(summary.median_irrelevant_focus_duration_seconds, 6.0)

    def test_target_and_known_scene_labels_are_excluded_from_irrelevant_time(self):
        summary = self.aggregate(
            [self.interval(0.0, 5.0)],
            [
                self.focus(0.0, 1.0, name="red apple"),
                self.focus(1.0, 2.0, name="green plant"),
                self.focus(2.0, 3.0, name="mainshelf", tag="mainshelf"),
                self.focus(3.0, 4.0, name="tablet", tag="tablet"),
                self.focus(4.0, 5.0, name="cart", tag="cart"),
            ],
        )

        self.assertAlmostEqual(summary.median_irrelevant_focus_duration_seconds, 0.0)

    def test_npc_is_explicitly_irrelevant(self):
        summary = self.aggregate(
            [self.interval(0.0, 5.0)],
            [self.focus(1.0, 2.5, name="npc", tag="npc")],
        )

        self.assertAlmostEqual(summary.median_irrelevant_focus_duration_seconds, 1.5)

    def test_unknown_focus_is_excluded_and_warned(self):
        summary = self.aggregate(
            [self.interval(0.0, 5.0)],
            [self.focus(1.0, 3.0, name="mystery object", tag="mystery")],
        )

        self.assertAlmostEqual(summary.median_irrelevant_focus_duration_seconds, 0.0)
        self.assertIn("unclassifiable_focus:mystery object", summary.warnings)

    def test_zero_irrelevant_focus_is_retained_as_zero(self):
        summary = self.aggregate([self.interval(0.0, 1.0)], [])

        self.assertAlmostEqual(summary.median_irrelevant_focus_duration_seconds, 0.0)

    def test_head_turning_accumulates_quaternion_path_in_degrees(self):
        half = 2**-0.5
        summary = self.aggregate(
            [self.interval(0.0, 2.0)],
            tracker=self.tracker(
                [0.0, 1.0, 2.0],
                [(0.0, 0.0, 0.0, 1.0), (0.0, 0.0, half, half), (0.0, 0.0, 1.0, 0.0)],
            ),
        )

        self.assertAlmostEqual(summary.median_head_turning_degrees, 180.0, places=5)

    def test_head_turning_counts_turn_and_return(self):
        half = 2**-0.5
        summary = self.aggregate(
            [self.interval(0.0, 2.0)],
            tracker=self.tracker(
                [0.0, 1.0, 2.0],
                [(0.0, 0.0, 0.0, 1.0), (0.0, 0.0, half, half), (0.0, 0.0, 0.0, 1.0)],
            ),
        )

        self.assertAlmostEqual(summary.median_head_turning_degrees, 180.0, places=5)

    def test_missing_rotation_data_is_missing_with_warning(self):
        summary = self.aggregate([self.interval(0.0, 1.0)])

        self.assertIsNone(summary.median_head_turning_degrees)
        self.assertIn("head_rotation_unavailable", summary.warnings)

    def test_invalid_intervals_do_not_contribute(self):
        summary = self.aggregate(
            [self.interval(0.0, 1.0, valid=False), self.interval(2.0, 4.0)],
            tracker=self.tracker(
                [0.0, 1.0, 2.0, 3.0, 4.0],
                [(0.0, 0.0, 0.0, 1.0)] * 5,
            ),
        )

        self.assertAlmostEqual(summary.median_time_to_target_seconds, 2.0)

    def test_interval_order_does_not_change_medians(self):
        intervals = [self.interval(3.0, 5.0), self.interval(1.0, 2.5)]
        summary = self.aggregate(intervals)

        self.assertAlmostEqual(summary.median_time_to_target_seconds, 1.75)


class HandMotionTests(unittest.TestCase):
    """Check FE-01.8 hand positions, distances and speeds."""

    def require_api(self):
        self.assertIsNotNone(EXTRACTOR)
        self.assertTrue(
            hasattr(EXTRACTOR, "prepare_hand_motion"),
            "FE-01.8 hand-motion preparation has not been implemented",
        )
        return EXTRACTOR

    @staticmethod
    def tracker(rows):
        return pd.DataFrame(rows)

    def prepare(self, tracker, hand="right"):
        extractor = self.require_api()
        return extractor.prepare_hand_motion(tracker, hand)

    def test_exact_times_positions_distances_and_timestamp_based_speeds(self):
        result = self.prepare(
            self.tracker(
                {
                    "time": [0.00, 0.50, 1.50],
                    "right_pos_x": [0.0, 0.3, 0.3],
                    "right_pos_y": [0.0, 0.4, 0.4],
                    "right_pos_z": [0.0, 0.0, 1.4],
                }
            )
        )

        self.assertEqual([sample.timestamp_seconds for sample in result.samples], [0.0, 0.5, 1.5])
        self.assertAlmostEqual(result.samples[1].step_distance_meters, 0.5)
        self.assertAlmostEqual(result.samples[1].step_speed_meters_per_second, 1.0)
        self.assertAlmostEqual(result.samples[2].step_distance_meters, 1.4)
        self.assertAlmostEqual(result.samples[2].step_speed_meters_per_second, 1.4)

    def test_stationary_samples_have_zero_distance_and_speed(self):
        result = self.prepare(
            self.tracker(
                {
                    "time": [0.0, 0.5],
                    "right_pos_x": [1.0, 1.0],
                    "right_pos_y": [2.0, 2.0],
                    "right_pos_z": [3.0, 3.0],
                }
            )
        )

        self.assertEqual(result.samples[1].step_distance_meters, 0.0)
        self.assertEqual(result.samples[1].step_speed_meters_per_second, 0.0)

    def test_left_and_right_hand_columns_are_selected_independently(self):
        tracker = self.tracker(
            {
                "time": [0.0, 1.0],
                "left_pos_x": [0.0, 1.0],
                "left_pos_y": [0.0, 0.0],
                "left_pos_z": [0.0, 0.0],
                "right_pos_x": [0.0, 0.0],
                "right_pos_y": [0.0, 2.0],
                "right_pos_z": [0.0, 0.0],
            }
        )

        self.assertAlmostEqual(self.prepare(tracker, "left").samples[1].step_distance_meters, 1.0)
        self.assertAlmostEqual(self.prepare(tracker, "right").samples[1].step_distance_meters, 2.0)

    def test_missing_coordinate_breaks_step_without_bridging(self):
        result = self.prepare(
            self.tracker(
                {
                    "time": [0.0, 1.0, 2.0],
                    "right_pos_x": [0.0, None, 2.0],
                    "right_pos_y": [0.0, 0.0, 0.0],
                    "right_pos_z": [0.0, 0.0, 0.0],
                }
            )
        )

        self.assertFalse(result.samples[1].is_valid)
        self.assertIsNone(result.samples[1].step_distance_meters)
        self.assertIsNone(result.samples[2].step_distance_meters)

    def test_missing_time_breaks_step_without_bridging(self):
        result = self.prepare(
            self.tracker(
                {
                    "time": [0.0, None, 2.0],
                    "right_pos_x": [0.0, 1.0, 2.0],
                    "right_pos_y": [0.0, 0.0, 0.0],
                    "right_pos_z": [0.0, 0.0, 0.0],
                }
            )
        )

        self.assertFalse(result.samples[1].is_valid)
        self.assertIsNone(result.samples[2].step_distance_meters)

    def test_nonpositive_time_difference_warns_and_speed_is_missing(self):
        result = self.prepare(
            self.tracker(
                {
                    "time": [0.0, 0.0],
                    "right_pos_x": [0.0, 1.0],
                    "right_pos_y": [0.0, 0.0],
                    "right_pos_z": [0.0, 0.0],
                }
            )
        )

        self.assertAlmostEqual(result.samples[1].step_distance_meters, 1.0)
        self.assertIsNone(result.samples[1].step_speed_meters_per_second)
        self.assertIn("nonpositive_time_delta:right", result.warnings)

    def test_missing_hand_columns_return_empty_series_with_warning(self):
        result = self.prepare(pd.DataFrame({"time": [0.0, 1.0]}), "left")

        self.assertEqual(result.samples, ())
        self.assertIn("hand_position_columns_unavailable:left", result.warnings)

    def test_invalid_numeric_strings_become_missing(self):
        result = self.prepare(
            self.tracker(
                {
                    "time": ["0.0", "bad"],
                    "right_pos_x": ["1.0", "2.0"],
                    "right_pos_y": ["0.0", "0.0"],
                    "right_pos_z": ["0.0", "0.0"],
                }
            )
        )

        self.assertTrue(result.samples[0].is_valid)
        self.assertFalse(result.samples[1].is_valid)
        self.assertIsNone(result.samples[1].step_speed_meters_per_second)

    def test_row_order_is_preserved(self):
        result = self.prepare(
            self.tracker(
                {
                    "time": [2.0, 1.0],
                    "right_pos_x": [2.0, 1.0],
                    "right_pos_y": [0.0, 0.0],
                    "right_pos_z": [0.0, 0.0],
                }
            )
        )

        self.assertEqual([sample.timestamp_seconds for sample in result.samples], [2.0, 1.0])
        self.assertIn("nonpositive_time_delta:right", result.warnings)

    def test_both_hands_can_be_prepared_from_one_tracker(self):
        tracker = self.tracker(
            {
                "time": [0.0, 1.0],
                "left_pos_x": [0.0, 1.0],
                "left_pos_y": [0.0, 0.0],
                "left_pos_z": [0.0, 0.0],
                "right_pos_x": [0.0, 0.0],
                "right_pos_y": [0.0, 1.0],
                "right_pos_z": [0.0, 0.0],
            }
        )

        self.assertEqual(self.prepare(tracker, "left").hand, "left")
        self.assertEqual(self.prepare(tracker, "right").hand, "right")

    def test_actual_grab_hand_helper_returns_event_hand(self):
        extractor = self.require_api()
        event = extractor.ProductGrabEvent(
            canonical_product_name="red apple",
            raw_product_label="red apple",
            hand="left",
            grab_start_seconds=1.0,
            grab_release_seconds=1.3,
            grab_duration_seconds=0.3,
            is_on_list=True,
            is_first_time_on_list=True,
            overlaps_other_hand_grab=False,
        )

        self.assertEqual(extractor.actual_grab_hand(event), "left")


class ReachOnsetTests(unittest.TestCase):
    """Check FE-01.9 final sustained pre-grab movement blocks."""

    def require_api(self):
        self.assertIsNotNone(EXTRACTOR)
        self.assertTrue(
            hasattr(EXTRACTOR, "detect_reach_intervals"),
            "FE-01.9 reach-onset detection has not been implemented",
        )
        return EXTRACTOR

    @staticmethod
    def grab(start, *, hand="right", product="red apple", on_list=True, first=True):
        return EXTRACTOR.ProductGrabEvent(
            canonical_product_name=product,
            raw_product_label=product,
            hand=hand,
            grab_start_seconds=start,
            grab_release_seconds=start + 0.20,
            grab_duration_seconds=0.20,
            is_on_list=on_list,
            is_first_time_on_list=first,
            overlaps_other_hand_grab=False,
        )

    @staticmethod
    def tracker(times, positions):
        return pd.DataFrame(
            {
                "time": times,
                "right_pos_x": [position[0] for position in positions],
                "right_pos_y": [position[1] for position in positions],
                "right_pos_z": [position[2] for position in positions],
            }
        )

    def detect(self, tracker, grabs, config=None):
        extractor = self.require_api()
        return extractor.detect_reach_intervals(tracker, grabs, config)

    def test_final_sustained_rest_selects_following_reach_onset(self):
        result = self.detect(
            self.tracker(
                [0.0, 0.1, 0.2, 0.3, 0.6, 1.0],
                [(0.0, 0.0, 0.0), (0.0, 0.0, 0.0), (0.0, 0.0, 0.0),
                 (0.1, 0.0, 0.0), (0.4, 0.0, 0.0), (0.8, 0.0, 0.0)],
            ),
            [self.grab(1.0)],
        )

        interval = result.intervals[0]
        self.assertAlmostEqual(interval.reach_start_seconds, 0.2)
        self.assertAlmostEqual(interval.grab_start_seconds, 1.0)
        self.assertTrue(interval.is_valid)

    def test_multiple_rest_blocks_use_the_final_qualifying_block(self):
        result = self.detect(
            self.tracker(
                [0.0, 0.1, 0.2, 0.4, 0.5, 0.6, 0.7, 0.8, 1.0],
                [(0.0, 0.0, 0.0), (0.0, 0.0, 0.0), (0.0, 0.0, 0.0),
                 (0.2, 0.0, 0.0), (0.3, 0.0, 0.0), (0.4, 0.0, 0.0),
                 (0.4, 0.0, 0.0), (0.4, 0.0, 0.0), (0.8, 0.0, 0.0)],
            ),
            [self.grab(1.0)],
        )

        self.assertAlmostEqual(result.intervals[0].reach_start_seconds, 0.8)

    def test_rest_duration_threshold_is_configurable_and_provisional(self):
        extractor = self.require_api()
        config = extractor.ReachDetectionConfig(
            hand_speed_threshold_meters_per_second=0.05,
            minimum_rest_duration_seconds=0.30,
        )
        result = self.detect(
            self.tracker(
                [0.0, 0.1, 0.2, 0.3, 0.6],
                [(0.0, 0.0, 0.0), (0.0, 0.0, 0.0), (0.0, 0.0, 0.0),
                 (0.2, 0.0, 0.0), (0.5, 0.0, 0.0)],
            ),
            [self.grab(0.6)],
            config,
        )

        self.assertFalse(result.intervals[0].is_valid)

    def test_no_qualifying_rest_produces_invalid_interval_and_warning(self):
        result = self.detect(
            self.tracker(
                [0.0, 0.2, 0.4, 0.8],
                [(0.0, 0.0, 0.0), (0.2, 0.0, 0.0), (0.6, 0.0, 0.0), (1.0, 0.0, 0.0)],
            ),
            [self.grab(0.8)],
        )

        self.assertFalse(result.intervals[0].is_valid)
        self.assertIn("reach_onset_not_found:right", result.warnings)

    def test_missing_hand_motion_keeps_interval_missing(self):
        result = self.detect(
            pd.DataFrame({"time": [0.0, 1.0]}),
            [self.grab(1.0)],
        )

        self.assertFalse(result.intervals[0].is_valid)
        self.assertIn("hand_position_columns_unavailable:right", result.warnings)

    def test_repeated_and_off_list_grabs_do_not_create_reach_intervals(self):
        result = self.detect(
            self.tracker(
                [0.0, 0.1, 0.2, 0.3, 0.6],
                [(0.0, 0.0, 0.0), (0.0, 0.0, 0.0), (0.0, 0.0, 0.0),
                 (0.1, 0.0, 0.0), (0.4, 0.0, 0.0)],
            ),
            [
                self.grab(0.6),
                self.grab(0.7, product="red apple", first=False),
                self.grab(0.8, product="orange bottle", on_list=False, first=False),
            ],
        )

        self.assertEqual(len(result.intervals), 1)

    def test_actual_left_grabbing_hand_selects_left_motion_columns(self):
        extractor = self.require_api()
        tracker = pd.DataFrame(
            {
                "time": [0.0, 0.1, 0.2, 0.3, 0.6],
                "left_pos_x": [0.0, 0.0, 0.0, 0.1, 0.4],
                "left_pos_y": [0.0] * 5,
                "left_pos_z": [0.0] * 5,
            }
        )
        result = extractor.detect_reach_intervals(tracker, [self.grab(0.6, hand="left")])

        self.assertEqual(result.intervals[0].hand, "left")
        self.assertTrue(result.intervals[0].is_valid)


class ReachFeatureAggregationTests(unittest.TestCase):
    """Check FE-01.10 reach duration and path-ratio aggregation."""

    def require_api(self):
        self.assertIsNotNone(EXTRACTOR)
        self.assertTrue(
            hasattr(EXTRACTOR, "aggregate_trial_reach_features"),
            "FE-01.10 reach aggregation not implemented",
        )
        return EXTRACTOR

    @staticmethod
    def interval(start, grab, *, hand="right", valid=True, product="red apple"):
        return EXTRACTOR.ReachInterval(
            canonical_product_name=product,
            hand=hand,
            reach_start_seconds=start,
            grab_start_seconds=grab,
            is_valid=valid,
        )

    @staticmethod
    def tracker(times, positions, *, hand="right"):
        return pd.DataFrame(
            {
                "time": times,
                f"{hand}_pos_x": [position[0] for position in positions],
                f"{hand}_pos_y": [position[1] for position in positions],
                f"{hand}_pos_z": [position[2] for position in positions],
            }
        )

    def aggregate(self, intervals, tracker):
        extractor = self.require_api()
        return extractor.aggregate_trial_reach_features(intervals, tracker)

    def test_exact_duration_and_median_duration(self):
        tracker = self.tracker([0.0, 1.0, 2.0], [(0, 0, 0)] * 3)
        result = self.aggregate(
            [self.interval(0.25, 1.25), self.interval(1.0, 3.0)], tracker
        )
        self.assertAlmostEqual(result.median_reach_duration_seconds, 1.5)

    def test_invalid_or_missing_onset_remains_missing(self):
        tracker = self.tracker([0.0, 1.0], [(0, 0, 0)] * 2)
        result = self.aggregate([self.interval(None, 1.0, valid=False)], tracker)
        self.assertIsNone(result.median_reach_duration_seconds)
        self.assertIn("reach_duration_unavailable", result.warnings)

    def test_negative_boundaries_are_rejected(self):
        tracker = self.tracker([0.0, 1.0], [(0, 0, 0)] * 2)
        result = self.aggregate([self.interval(2.0, 1.0)], tracker)
        self.assertIsNone(result.median_reach_duration_seconds)
        self.assertIn("negative_reach_duration", result.warnings)

    def test_actual_hand_selects_matching_motion_columns(self):
        tracker = pd.DataFrame(
            {
                "time": [0.0, 0.2, 0.4],
                "left_pos_x": [0.0, 0.1, 0.2],
                "left_pos_y": [0.0, 0.0, 0.0],
                "left_pos_z": [0.0, 0.0, 0.0],
                "right_pos_x": [0.0, 0.0, 0.0],
                "right_pos_y": [0.0, 0.0, 0.0],
                "right_pos_z": [0.0, 0.0, 0.0],
            }
        )
        result = self.aggregate([self.interval(0.0, 0.4, hand="left")], tracker)
        self.assertAlmostEqual(result.median_reach_path_ratio, 1.0)

    def test_straight_line_motion_has_ratio_one(self):
        tracker = self.tracker(
            [0.0, 0.1, 0.2], [(0.0, 0, 0), (0.1, 0, 0), (0.2, 0, 0)]
        )
        result = self.aggregate([self.interval(0.0, 0.2)], tracker)
        self.assertAlmostEqual(result.median_reach_path_ratio, 1.0)

    def test_curved_and_returning_motion_accumulates_path(self):
        tracker = self.tracker(
            [0.0, 0.1, 0.2, 0.3],
            [(0, 0, 0), (0.1, 0, 0), (0.1, 0.1, 0), (0.2, 0.1, 0)],
        )
        result = self.aggregate([self.interval(0.0, 0.3)], tracker)
        self.assertGreater(result.median_reach_path_ratio, 1.0)

    def test_small_straight_distance_is_missing_with_warning(self):
        tracker = self.tracker([0.0, 0.1], [(0, 0, 0), (0.01, 0, 0)])
        result = self.aggregate([self.interval(0.0, 0.1)], tracker)
        self.assertIsNone(result.median_reach_path_ratio)
        self.assertIn("straight_distance_too_small", result.warnings)

    def test_missing_position_does_not_bridge_path_gap(self):
        tracker = self.tracker(
            [0.0, 0.1, 0.2, 0.3],
            [(0, 0, 0), (0.1, 0, 0), (None, 0, 0), (0.3, 0, 0)],
        )
        result = self.aggregate([self.interval(0.0, 0.3)], tracker)
        self.assertAlmostEqual(result.median_reach_path_ratio, 1.0)
        self.assertTrue(any(warning.startswith("reach_position_gap") for warning in result.warnings))

    def test_duration_can_be_valid_when_path_ratio_is_missing(self):
        tracker = self.tracker([0.0, 0.1], [(0, 0, 0), (None, 0, 0)])
        result = self.aggregate([self.interval(0.0, 0.1)], tracker)
        self.assertAlmostEqual(result.median_reach_duration_seconds, 0.1)
        self.assertIsNone(result.median_reach_path_ratio)

    def test_unsorted_intervals_do_not_change_medians(self):
        tracker = self.tracker(
            [0.0, 0.1, 0.2, 0.3],
            [(0, 0, 0), (0.1, 0, 0), (0.2, 0, 0), (0.3, 0, 0)],
        )
        result = self.aggregate(
            [self.interval(0.2, 0.3), self.interval(0.0, 0.1)], tracker
        )
        self.assertAlmostEqual(result.median_reach_duration_seconds, 0.1)

    def test_no_valid_intervals_returns_missing_values(self):
        tracker = self.tracker([0.0, 0.1], [(0, 0, 0)] * 2)
        result = self.aggregate([self.interval(None, 0.1, valid=False)], tracker)
        self.assertIsNone(result.median_reach_duration_seconds)
        self.assertIsNone(result.median_reach_path_ratio)


class ErrorOutcomeTests(unittest.TestCase):
    """Check FE-01.11 D0-matched raw and change error outcomes."""

    def require_api(self):
        self.assertIsNotNone(EXTRACTOR)
        self.assertTrue(
            hasattr(EXTRACTOR, "derive_error_outcomes"),
            "FE-01.11 error-outcome derivation not implemented",
        )
        return EXTRACTOR

    @staticmethod
    def row(
        participant="P01",
        session="S001",
        trial="T001",
        condition="Visual",
        difficulty=0,
        trial_order="T1",
        total=0,
        missing=0,
        wrong=0,
        duplicate=0,
        not_in_list=0,
        **extra,
    ):
        value = {
            "participant": participant,
            "session": session,
            "trial": trial,
            "group": "Young",
            "condition": condition,
            "trial2": trial_order,
            "level3": difficulty,
            "errors_total": total,
            "errors_missing": missing,
            "errors_wrongorder": wrong,
            "errors_collectedmorethanonce": duplicate,
            "errors_notinlist": not_in_list,
        }
        value.update(extra)
        return value

    def derive(self, rows):
        return self.require_api().derive_error_outcomes(pd.DataFrame(rows))

    def test_historical_error_aliases_and_metadata_are_normalized(self):
        result = self.derive(
            [
                self.row(
                    participant="p1",
                    session="s002",
                    trial="T004",
                    condition="visual",
                    difficulty="10",
                    trial_order="t3",
                    total=7,
                    missing=1,
                    wrong=2,
                    duplicate=3,
                    not_in_list=1,
                )
            ]
        )
        record = result.records[0]
        self.assertEqual(record.participant_id, "P01")
        self.assertEqual(record.session_id, "S002")
        self.assertEqual(record.condition_name, "Visual")
        self.assertEqual(record.difficulty_level, 10)
        self.assertEqual(record.trial_order, "T3")
        self.assertEqual(record.errors_wrong_order, 2.0)
        self.assertEqual(record.errors_duplicate, 3.0)
        self.assertEqual(record.errors_not_in_list, 1.0)

    def test_recorded_total_is_preserved_as_authoritative(self):
        result = self.derive([self.row(total=99, missing=1, wrong=2, duplicate=3, not_in_list=4)])
        self.assertEqual(result.records[0].total_error_count, 99.0)

    def test_missing_total_falls_back_to_complete_component_sum(self):
        result = self.derive(
            [self.row(total="bad", missing=1, wrong=2, duplicate=3, not_in_list=4)]
        )
        self.assertEqual(result.records[0].total_error_count, 10.0)

    def test_total_component_mismatch_preserves_total_and_warns_once(self):
        result = self.derive([self.row(total=20, missing=1, wrong=2, duplicate=3, not_in_list=4)])
        self.assertEqual(result.records[0].total_error_count, 20.0)
        self.assertEqual(
            sum("error_total_component_mismatch" in warning for warning in result.warnings),
            1,
        )

    def test_incomplete_components_leave_total_missing(self):
        result = self.derive(
            [self.row(total="bad", missing=1, wrong=None, duplicate=3, not_in_list=4)]
        )
        self.assertIsNone(result.records[0].total_error_count)
        self.assertTrue(any("error_total_unavailable" in warning for warning in result.warnings))

    def test_same_participant_condition_baseline_matches_across_sessions(self):
        result = self.derive(
            [
                self.row(session="S001", difficulty=0, total=2),
                self.row(session="S002", difficulty=6, total=7),
            ]
        )
        self.assertEqual(result.records[1].error_change_from_d0, 5.0)

    def test_duplicate_d0_baselines_use_median_and_warn(self):
        result = self.derive(
            [
                self.row(trial="T001", difficulty=0, total=2),
                self.row(trial="T002", difficulty=0, total=6),
                self.row(trial="T003", difficulty=2, total=10),
            ]
        )
        self.assertEqual(result.records[2].error_change_from_d0, 6.0)
        self.assertTrue(any("duplicate_d0_baseline" in warning for warning in result.warnings))

    def test_missing_d0_baseline_leaves_change_missing(self):
        result = self.derive([self.row(difficulty=10, total=4)])
        self.assertIsNone(result.records[0].error_change_from_d0)
        self.assertTrue(any("d0_baseline_unavailable" in warning for warning in result.warnings))

    def test_d0_change_is_missing_and_d2_d6_d10_changes_are_exact(self):
        result = self.derive(
            [
                self.row(trial="T0", difficulty=0, total=3),
                self.row(trial="T2", difficulty=2, total=5),
                self.row(trial="T6", difficulty=6, total=1),
                self.row(trial="T10", difficulty=10, total=8),
            ]
        )
        self.assertIsNone(result.records[0].error_change_from_d0)
        self.assertEqual(
            [record.error_change_from_d0 for record in result.records[1:]],
            [2.0, -2.0, 5.0],
        )

    def test_invalid_trial_errors_remain_missing(self):
        result = self.derive([self.row(total="invalid", missing="bad", wrong=1, duplicate=1, not_in_list=1)])
        self.assertIsNone(result.records[0].total_error_count)
        self.assertIsNone(result.records[0].error_change_from_d0)

    def test_unsupported_difficulty_change_is_missing(self):
        result = self.derive(
            [
                self.row(difficulty=0, total=1),
                self.row(difficulty=4, total=3),
            ]
        )
        self.assertIsNone(result.records[1].error_change_from_d0)
        self.assertTrue(any("unsupported_difficulty" in warning for warning in result.warnings))

    def test_raw_components_and_input_order_are_preserved(self):
        result = self.derive(
            [
                self.row(trial="T002", difficulty=2, total=4, missing=1),
                self.row(trial="T001", difficulty=0, total=2, wrong=2),
            ]
        )
        self.assertEqual([record.trial_order for record in result.records], ["T1", "T1"])
        self.assertEqual(result.records[0].errors_missing, 1.0)
        self.assertEqual(result.records[1].errors_wrong_order, 2.0)

    def test_error_outcome_records_are_immutable(self):
        result = self.derive([self.row()])
        with self.assertRaises(AttributeError):
            result.records[0].total_error_count = 4.0


class ProductGrabOutputTests(unittest.TestCase):
    """Check FE-01.12 auditable product-grab table and writer."""

    EXPECTED_COLUMNS = [
        "participant_id",
        "session_id",
        "source_tracker_csv_filename",
        "participant_group",
        "condition_name",
        "difficulty_level",
        "trial_order",
        "language",
        "canonical_product_name",
        "raw_product_label",
        "hand",
        "grab_start_seconds",
        "grab_release_seconds",
        "grab_duration_seconds",
        "is_on_list",
        "is_first_time_on_list",
        "is_repeated_on_list",
        "is_off_list",
        "overlaps_other_hand_grab",
        "search_start_seconds",
        "first_target_focus_seconds",
        "search_interval_valid",
        "reach_start_seconds",
        "reach_duration_seconds",
        "reach_path_ratio",
        "reach_interval_valid",
        "processing_warnings",
    ]

    def require_api(self):
        self.assertIsNotNone(EXTRACTOR)
        self.assertTrue(
            hasattr(EXTRACTOR, "build_product_grab_features"),
            "FE-01.12 product output builder not implemented",
        )
        self.assertTrue(
            hasattr(EXTRACTOR, "write_product_grab_features"),
            "FE-01.12 product output writer not implemented",
        )
        return EXTRACTOR

    @staticmethod
    def event(product, raw, start, *, on_list=True, first=True, hand="right", overlap=False):
        return EXTRACTOR.ProductGrabEvent(
            canonical_product_name=product,
            raw_product_label=raw,
            hand=hand,
            grab_start_seconds=start,
            grab_release_seconds=start + 0.3,
            grab_duration_seconds=0.3,
            is_on_list=on_list,
            is_first_time_on_list=first,
            overlaps_other_hand_grab=overlap,
        )

    @staticmethod
    def search(product, start, search_start, target, valid=True):
        return EXTRACTOR.SearchInterval(
            canonical_product_name=product,
            qualifying_grab_start_seconds=start,
            search_start_seconds=search_start,
            first_target_focus_seconds=target,
            is_valid=valid,
        )

    @staticmethod
    def reach(product, hand, onset, start, valid=True):
        return EXTRACTOR.ReachInterval(
            canonical_product_name=product,
            hand=hand,
            reach_start_seconds=onset,
            grab_start_seconds=start,
            is_valid=valid,
        )

    @staticmethod
    def tracker():
        return pd.DataFrame(
            {
                "time": [0.0, 0.1, 0.2, 0.3],
                "right_pos_x": [0.0, 0.1, 0.2, 0.3],
                "right_pos_y": [0.0, 0.0, 0.0, 0.0],
                "right_pos_z": [0.0, 0.0, 0.0, 0.0],
            }
        )

    def build(self, events, searches=(), reaches=(), tracker=None):
        extractor = self.require_api()
        metadata = {
            "participant_id": "P01",
            "session_id": "S001",
            "source_tracker_csv_filename": "data_collector_vr_sample_T001.csv",
            "participant_group": "Young",
            "condition_name": "Visual",
            "difficulty_level": 2,
            "trial_order": "T1",
            "language": "EN",
        }
        return extractor.build_product_grab_features(
            metadata, events, searches, reaches, tracker if tracker is not None else self.tracker()
        )

    def test_all_complete_events_and_metadata_are_preserved(self):
        result = self.build(
            [
                self.event("red apple", "Apple(Clone)", 0.3),
                self.event("red apple", "Apple(3)", 0.8, first=False),
                self.event("orange bottle", "Bottle", 1.2, on_list=False, first=False, overlap=True),
            ]
        )
        self.assertEqual(len(result.table), 3)
        self.assertEqual(list(result.table.columns), self.EXPECTED_COLUMNS)
        self.assertEqual(result.table.loc[0, "raw_product_label"], "Apple(Clone)")
        self.assertEqual(result.table.loc[0, "participant_id"], "P01")
        self.assertTrue(bool(result.table.loc[1, "is_repeated_on_list"]))
        self.assertTrue(bool(result.table.loc[2, "is_off_list"]))
        self.assertTrue(bool(result.table.loc[2, "overlaps_other_hand_grab"]))

    def test_search_and_reach_attach_only_to_matching_qualifying_event(self):
        result = self.build(
            [
                self.event("red apple", "Apple", 0.3),
                self.event("red apple", "Apple(2)", 0.8, first=False),
            ],
            [self.search("red apple", 0.3, 0.0, 0.2)],
            [self.reach("red apple", "right", 0.1, 0.3)],
        )
        self.assertAlmostEqual(result.table.loc[0, "search_start_seconds"], 0.0)
        self.assertAlmostEqual(result.table.loc[0, "reach_start_seconds"], 0.1)
        self.assertTrue(pd.isna(result.table.loc[1, "search_start_seconds"]))
        self.assertTrue(pd.isna(result.table.loc[1, "reach_start_seconds"]))

    def test_per_event_reach_duration_and_path_ratio_use_actual_hand(self):
        result = self.build(
            [self.event("red apple", "Apple", 0.3, hand="right")],
            reaches=[self.reach("red apple", "right", 0.0, 0.3)],
        )
        self.assertAlmostEqual(result.table.loc[0, "reach_duration_seconds"], 0.3)
        self.assertAlmostEqual(result.table.loc[0, "reach_path_ratio"], 1.0)
        self.assertTrue(bool(result.table.loc[0, "reach_interval_valid"]))

    def test_unmatched_intervals_are_blank_and_warn_once(self):
        result = self.build(
            [self.event("red apple", "Apple", 0.3)],
            searches=[self.search("orange bottle", 0.9, 0.5, 0.8)],
            reaches=[self.reach("orange bottle", "right", 0.6, 0.9)],
        )
        self.assertTrue(pd.isna(result.table.loc[0, "search_start_seconds"]))
        self.assertTrue(pd.isna(result.table.loc[0, "reach_start_seconds"]))
        self.assertEqual(len(result.warnings), 2)
        self.assertEqual(len(set(result.warnings)), len(result.warnings))

    def test_missing_numeric_values_write_as_blank_cells(self):
        extractor = self.require_api()
        result = self.build([self.event("red apple", "Apple", 0.3)])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "nested" / "product_grab_features_v2.csv"
            extractor.write_product_grab_features(result.table, path)
            text = path.read_text(encoding="utf-8")
        self.assertIn("search_start_seconds", text.splitlines()[0])
        self.assertIn(",,", text)

    def test_empty_event_input_returns_full_schema(self):
        result = self.build([])
        self.assertEqual(len(result.table), 0)
        self.assertEqual(list(result.table.columns), self.EXPECTED_COLUMNS)


class TrialFeatureOutputTests(unittest.TestCase):
    """Check FE-01.13 one-row trial feature aggregation and writer."""

    EXPECTED_COLUMNS = [
        "participant_id",
        "session_id",
        "source_tracker_csv_filename",
        "participant_group",
        "condition_name",
        "difficulty_level",
        "trial_order",
        "language",
        "performance",
        "mental_demand_score_0_to_10",
        "errors_missing",
        "errors_wrong_order",
        "errors_duplicate",
        "errors_not_in_list",
        "total_error_count",
        "error_change_from_d0",
        "median_time_between_qualifying_grabs_seconds",
        "list_recheck_count",
        "total_list_recheck_duration_seconds",
        "median_time_to_target_seconds",
        "median_irrelevant_focus_duration_seconds",
        "median_head_turning_degrees",
        "median_reach_duration_seconds",
        "median_reach_path_ratio",
        "real_product_grab_count",
        "first_time_on_list_grab_count",
        "list_visit_count",
        "valid_search_interval_count",
        "valid_reach_duration_count",
        "valid_reach_path_ratio_count",
        "processing_warnings",
    ]

    def require_api(self):
        self.assertIsNotNone(EXTRACTOR)
        self.assertTrue(
            hasattr(EXTRACTOR, "build_trial_features"),
            "FE-01.13 trial feature builder not implemented",
        )
        self.assertTrue(
            hasattr(EXTRACTOR, "write_trial_features"),
            "FE-01.13 trial feature writer not implemented",
        )
        return EXTRACTOR

    @staticmethod
    def metadata():
        return {
            "participant_id": "P01",
            "session_id": "S001",
            "source_tracker_csv_filename": "data_collector_vr_sample_T001.csv",
            "participant_group": "Young",
            "condition_name": "Visual",
            "difficulty_level": 6,
            "trial_order": "T2",
            "language": "EN",
            "performance": 0.75,
            "mental_demand_score_0_to_10": 6.0,
        }

    @staticmethod
    def aggregates(warnings=()):
        extractor = EXTRACTOR
        pace = extractor.PaceAndListRecheckAggregation(
            median_time_between_qualifying_grabs_seconds=1.25,
            list_recheck_count=2,
            total_list_recheck_duration_seconds=3.5,
        )
        locating = extractor.LocatingFeatureAggregation(
            median_time_to_target_seconds=0.8,
            median_irrelevant_focus_duration_seconds=0.2,
            median_head_turning_degrees=15.0,
            warnings=tuple(warnings),
        )
        reach = extractor.ReachFeatureAggregation(
            median_reach_duration_seconds=0.4,
            median_reach_path_ratio=1.2,
            warnings=tuple(warnings),
        )
        return pace, locating, reach

    @staticmethod
    def error_outcome():
        return EXTRACTOR.ErrorOutcomeRecord(
            participant_id="P01",
            session_id="S001",
            condition_name="Visual",
            difficulty_level=6,
            trial_order="T2",
            errors_missing=1.0,
            errors_wrong_order=2.0,
            errors_duplicate=3.0,
            errors_not_in_list=4.0,
            total_error_count=10.0,
            error_change_from_d0=5.0,
        )

    def build(self, *, error_outcome=None, validity_counts=None, warnings=()):
        extractor = self.require_api()
        pace, locating, reach = self.aggregates(warnings)
        return extractor.build_trial_features(
            self.metadata(),
            pace,
            locating,
            reach,
            error_outcome=error_outcome,
            validity_counts=validity_counts,
        )

    def test_all_metadata_outcomes_measures_and_counts_map_in_fixed_order(self):
        result = self.build(
            error_outcome=self.error_outcome(),
            validity_counts={
                "real_product_grab_count": 5,
                "first_time_on_list_grab_count": 3,
                "list_visit_count": 4,
                "valid_search_interval_count": 3,
                "valid_reach_duration_count": 2,
                "valid_reach_path_ratio_count": 1,
            },
        )
        row = result.table.iloc[0]
        self.assertEqual(list(result.table.columns), self.EXPECTED_COLUMNS)
        self.assertEqual(row["participant_id"], "P01")
        self.assertEqual(row["performance"], 0.75)
        self.assertEqual(row["mental_demand_score_0_to_10"], 6.0)
        self.assertEqual(row["errors_duplicate"], 3.0)
        self.assertEqual(row["error_change_from_d0"], 5.0)
        self.assertEqual(row["median_time_between_qualifying_grabs_seconds"], 1.25)
        self.assertEqual(row["median_reach_path_ratio"], 1.2)
        self.assertEqual(row["valid_reach_path_ratio_count"], 1)

    def test_missing_measurements_remain_missing_and_zero_counts_remain_zero(self):
        extractor = self.require_api()
        pace = extractor.PaceAndListRecheckAggregation(None, 0, 0.0)
        locating = extractor.LocatingFeatureAggregation(None, None, None, ())
        reach = extractor.ReachFeatureAggregation(None, None, ())
        result = extractor.build_trial_features(
            self.metadata(),
            pace,
            locating,
            reach,
            validity_counts={
                "real_product_grab_count": 0,
                "first_time_on_list_grab_count": 0,
                "list_visit_count": 0,
                "valid_search_interval_count": 0,
                "valid_reach_duration_count": 0,
                "valid_reach_path_ratio_count": 0,
            },
        )
        row = result.table.iloc[0]
        self.assertTrue(pd.isna(row["median_time_to_target_seconds"]))
        self.assertEqual(row["list_recheck_count"], 0)
        self.assertEqual(row["total_list_recheck_duration_seconds"], 0.0)
        self.assertEqual(row["real_product_grab_count"], 0)

    def test_aggregate_warnings_are_unique_and_delimited(self):
        result = self.build(warnings=("focus_warning", "focus_warning", "reach_warning"))
        self.assertEqual(result.warnings, ("focus_warning", "reach_warning"))
        self.assertEqual(result.table.loc[0, "processing_warnings"], "focus_warning;reach_warning")

    def test_missing_error_outcome_preserves_blank_error_fields(self):
        result = self.build()
        for column in (
            "errors_missing",
            "errors_wrong_order",
            "errors_duplicate",
            "errors_not_in_list",
            "total_error_count",
            "error_change_from_d0",
        ):
            self.assertTrue(pd.isna(result.table.loc[0, column]))

    def test_result_wrapper_is_immutable(self):
        result = self.build(error_outcome=self.error_outcome())
        with self.assertRaises(AttributeError):
            result.table = pd.DataFrame()

    def test_writer_creates_only_requested_temporary_csv_with_blank_missing_cells(self):
        extractor = self.require_api()
        result = self.build()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "nested" / "trial_features_v2.csv"
            extractor.write_trial_features(result.table, path)
            self.assertTrue(path.exists())
            self.assertEqual(
                sorted(p.relative_to(root).as_posix() for p in root.rglob("*")),
                ["nested", "nested/trial_features_v2.csv"],
            )
            lines = path.read_text(encoding="utf-8").splitlines()
        self.assertEqual(lines[0].split(","), self.EXPECTED_COLUMNS)
        self.assertIn(",,,", lines[1])


if __name__ == "__main__":
    unittest.main()
