"""Synthetic checks for the versioned extractor and worked figure."""
import importlib
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))
extractor = importlib.import_module("09_extract_final_features")
plotter = importlib.import_module("plot_fe02_worked_example")


class FinalExtractorTests(unittest.TestCase):
    def test_all_valid_grabs_and_absolute_performance(self):
        a = ("P01", "S001", "tracker.csv", "Visual", 0, "T1")
        b = ("P01", "S002", "tracker.csv", "Auditory", 6, "T2")
        trials = pd.DataFrame([a, b], columns=extractor.TRIAL_ID)
        for name in extractor.FEATURE_COLUMNS:
            trials[name] = [None, None]
        trials["performance_percent"] = [75.0, 30.0]
        trials["performance_change_from_d0_percentage_points"] = [0.0, -45.0]
        trials["median_time_to_target_seconds"] = [0.2, 0.3]
        trials["median_reach_duration_seconds"] = [0.4, 0.5]
        grabs = pd.DataFrame(
            [(*a, 8.0, True, True), (*a, 1.0, True, True),
             (*a, 2.0, False, False), (*a, 4.0, True, False),
             (*b, 1.0, True, True)],
            columns=(*extractor.TRIAL_ID, "grab_start_seconds", "is_on_list",
                     "is_first_time_on_list"),
        )
        result = extractor.trial_feature_table(trials, grabs)
        self.assertEqual(result[extractor.PACE].iloc[0], 2.0)
        self.assertTrue(pd.isna(result[extractor.PACE].iloc[1]))
        self.assertEqual(result["correct_products_collected_percent"].tolist(), [75.0, 30.0])
        self.assertEqual(result["difficulty_level"].tolist(), [0, 2])
        self.assertEqual(result["recorded_difficulty_code"].tolist(), [0, 6])
        self.assertEqual(result["median_product_search_duration_seconds"].tolist(), [0.2, 0.3])
        self.assertEqual(result["median_reach_start_to_grab_seconds"].tolist(), [0.4, 0.5])
        self.assertNotIn("performance_change_from_d0_percentage_points", result)
        self.assertEqual(len(extractor.FEATURE_ORDER), 10)

    def test_run_and_render_from_saved_events(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            raw = root / "raw"
            performance = root / "performance.csv"
            performance.write_text(
                "participant;session;trial;group;condition;trial2;level3;language;performance\n"
                "P1;S001;T001;Young;Visual;T1;0;EN;15\n"
                "P1;S002;T001;Young;Visual;T1;2;EN;8\n"
                "P1;S003;T001;Young;Cognitive;T1;10;EN;5\n",
                encoding="utf-8",
            )
            catalogue = root / "CorrectLists.txt"
            catalogue.write_text(
                "\n".join(f"correctLists.level{k}.{lang} = {{'Apple'}};"
                          for k in (0, 2, 6, 10) for lang in ("EN", "NL")),
                encoding="utf-8",
            )
            time = np.linspace(0, 1.6, 33)
            grabbing = (time >= 0.9) & (time < 1.25)
            tracker = pd.DataFrame({
                "time": time,
                "enableBlackout": False,
                "focus_object_name": np.where(time < 0.3, "Tablet",
                    np.where(time < 0.45, "Side R", np.where(time < 1.25, "Apple", "Cart"))),
                "focus_object_tag": np.where(time < 0.3, "Tablet", "MainShelf"),
                "is_grabbing_right": grabbing,
                "grabbed_object_right": np.where(grabbing, "Apple", "noObjectGrabbed"),
                "is_grabbing_left": False,
                "grabbed_object_left": "noObjectGrabbed",
                "right_pos_x": np.clip(time - 0.4, 0, 0.5),
                "right_pos_y": np.clip(time - 0.4, 0, 0.5)**2 / 2,
                "right_pos_z": 0,
                "hmd_rot_x": 0, "hmd_rot_y": 0, "hmd_rot_z": 0, "hmd_rot_w": 1,
            })
            filename = "data_collector_vr_sample_T001.csv"
            for session in ("S001", "S002"):
                directory = raw / "P1" / "P01_HMD_Data" / "P01" / session / "trackers"
                directory.mkdir(parents=True)
                table = tracker.copy()
                if session == "S002":
                    table[["focus_object_name", "focus_object_tag"]] = None
                table.to_csv(directory / filename, index=False)
            output = root / "fe02"
            paths = extractor.run_extraction(performance, raw, catalogue, output,
                                            worked_example=("P01", "S001", filename, 1))
            features = pd.read_csv(output / "trial_features.csv")
            self.assertEqual(features["difficulty_level"].tolist(), [0, 1, 3])
            self.assertEqual(features["correct_products_collected_percent"].tolist(), [75., 40., 25.])
            self.assertEqual(features["shopping_list_recheck_count"].iloc[0], 0)
            self.assertTrue(features["shopping_list_recheck_count"].iloc[1:].isna().all())
            self.assertEqual(len(pd.read_csv(output / "shopping_list_visits.csv")), 1)
            qc = pd.read_csv(output / "feature_extraction_qc.csv")
            self.assertEqual(qc["processing_status"].iloc[2], "failed")
            self.assertEqual(qc["list_focus_usable"].iloc[:2].tolist(), [True, False])
            self.assertTrue(pd.isna(qc["list_visit_count"].iloc[1]))
            self.assertEqual(qc["missing_behavioural_measure_count"].iloc[1], 8)
            self.assertEqual(len(paths), 6)
            self.assertFalse((output / "plots").exists())
            trace_path = output / "worked_trace.json"
            trace = json.loads(trace_path.read_text())
            event = trace["events"]
            self.assertNotIn("participant_id", trace)
            self.assertAlmostEqual(event["first_target_focus"] - event["search_start"],
                                   features["median_product_search_duration_seconds"].iloc[0])
            self.assertAlmostEqual(event["grab_start"] - event["reach_start"],
                                   features["median_reach_start_to_grab_seconds"].iloc[0])
            xyz = np.asarray(trace["reach_path_xyz_m"])
            ratio = np.linalg.norm(np.diff(xyz, axis=0), axis=1).sum() / np.linalg.norm(xyz[-1] - xyz[0])
            self.assertAlmostEqual(ratio, trace["reach_path_ratio"])
            figure = root / "worked_example.pdf"
            plotter.render(trace_path, figure)
            self.assertTrue(figure.read_bytes().startswith(b"%PDF-"))
            with self.assertRaises(FileExistsError):
                extractor.run_extraction(performance, raw, catalogue, output)
