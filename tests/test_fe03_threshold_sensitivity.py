"""Synthetic checks for the development-only detector comparisons."""
import contextlib
import importlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))
sensitivity = importlib.import_module("10_compare_fe02_thresholds")


class SensitivityTests(unittest.TestCase):
    def synthetic_tracker(self):
        t = np.arange(80) * 0.02
        moving = np.clip(t - 0.44, 0, 0.46)
        grabbing = (t >= .9) & (t < 1.25)
        return pd.DataFrame({
            "time": t, "enableBlackout": False,
            "focus_object_name": np.select(
                [t < .20, (t >= .22) & (t < .26), (t >= .40) & (t < 1.25)],
                ["Tablet", "Tablet", "Apple"], default="Side R"),
            "focus_object_tag": np.where((t < .20) | ((t >= .22) & (t < .26)), "Tablet", "MainShelf"),
            "is_grabbing_right": grabbing, "grabbed_object_right": np.where(grabbing, "Apple", ""),
            "is_grabbing_left": False, "grabbed_object_left": "",
            "right_pos_x": moving, "right_pos_y": 0, "right_pos_z": moving**2,
            "hmd_rot_x": 0, "hmd_rot_y": 0, "hmd_rot_z": 0, "hmd_rot_w": 1,
        })

    def test_baseline_agrees_and_list_filter_is_separate(self):
        f = sensitivity.fe01
        tracker = self.synthetic_tracker()
        cat = f.ProductCatalog({(0, "EN"): ("Apple",)})
        meta = {"difficulty_level": 0, "language": "EN"}
        results = {name: (features, qc, ref) for name, _, _, features, qc, ref in
                   sensitivity.measure_trial(meta, tracker, cat, [("apple", "right", .44, .9, .02)])}
        baseline, qc, ref = results["baseline"]
        g = f.detect_product_grab_events(tracker, cat, 0, "EN").events
        focus = f.detect_focus_episodes(tracker, cat).episodes
        visits = f.detect_list_visits(focus)
        search = f.detect_search_intervals(tracker, g, visits, focus).intervals
        reach = f.detect_reach_intervals(tracker, g, search_intervals=search).intervals
        agg = f.aggregate_trial_reach_features(reach, tracker)
        self.assertAlmostEqual(baseline[sensitivity.FEATURES[-1]], agg.median_reach_path_ratio)
        self.assertAlmostEqual(baseline[sensitivity.FEATURES[-2]], agg.median_reach_duration_seconds)
        self.assertEqual(qc["valid_reaches"], 1)
        self.assertIsNotNone(ref[0]["reach_onset_error_s"])
        self.assertEqual(baseline[sensitivity.FEATURES[1]], 1)
        self.assertEqual(results["list_focus_s=0.06"][0][sensitivity.FEATURES[1]], 0)
        self.assertEqual(results["target_focus_s=0.06"][0][sensitivity.FEATURES[1]], 1)
        self.assertEqual(len(results), 28)

    def test_split_is_applied_before_tracker_reads_and_outputs_render(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            holdout = tmp/"participant_holdout_split.csv"
            pd.DataFrame({"participant_id": [f"P{i:02d}" for i in range(1, 33)],
                          "split": ["train"]*26+["test"]*6}).to_csv(holdout, index=False)
            trials = tmp/"performance.csv"
            pd.DataFrame(
                [(i, "S001", "T001", "Young", c, "T1", d)
                 for i in range(1, 33) for c in sensitivity.CONDITIONS for d in (0, 2, 6, 10)],
                columns=["participant", "session", "trial", "group", "condition", "trial2", "level3"],
            ).to_csv(trials, index=False)
            catalogue = tmp/"CorrectLists.txt"
            catalogue.write_text("\n".join(f"correctLists.level{k}.{lang} = {{'Apple'}};"
                                  for k in (0, 2, 6, 10) for lang in ("EN", "NL")), encoding="utf-8")
            output = tmp/"sensitivity"
            opened = set()
            tracker = self.synthetic_tracker()

            def read(path):
                opened.add(path.parents[2].name)
                return sensitivity.fe01.TrackerLoadResult(tracker, [])

            # Run the true detector for one representative trial of each
            # condition. Reuse those measurements for other synthetic trials.
            original = sensitivity.measure_trial
            cached = {}
            def measure(meta, *args):
                if meta["condition_name"] not in cached:
                    cached[meta["condition_name"]] = list(original(meta, *args))
                for name, family, value, features, qc, errors in cached[meta["condition_name"]]:
                    yield name, family, value, features, qc.copy(), errors

            with patch.object(sensitivity.fe01, "load_tracker_table", side_effect=read) as load, \
                    patch.object(sensitivity, "measure_trial", side_effect=measure), \
                    contextlib.redirect_stdout(io.StringIO()):
                sensitivity.run(trials, tmp, catalogue, holdout, output)
                self.assertEqual(load.call_count, 312)
            self.assertEqual(opened, {f"P{i:02d}" for i in range(1, 27)})
            manifest = json.loads((output/"sensitivity_manifest.json").read_text())
            self.assertEqual(manifest["reserved_tracker_reads"], 0)
            self.assertFalse(manifest["threshold_selected"])
            self.assertEqual(manifest["detector_validation"], "not performed")
            self.assertEqual(len(pd.read_csv(output/"trial_sensitivity.csv")), 312*28)
            for name in ("feature_sensitivity.pdf", "event_qc_sensitivity.pdf"):
                self.assertTrue((output/name).read_bytes().startswith(b"%PDF"))
            expected = {
                "01_grab_duration.pdf", "02_grab_merge_gap.pdf",
                "03_list_focus.pdf", "04_target_focus.pdf",
                "05_reach_speed_floor.pdf", "06_peak_speed_cutoff.pdf",
                "07_sustained_movement.pdf", "08_path_displacement.pdf",
                "09_search_start.pdf", "threshold_figures.pdf",
            }
            figures = output / "threshold_figures"
            self.assertEqual({f.name for f in figures.glob("*.pdf")}, expected)
            for name in expected:
                self.assertTrue((figures / name).read_bytes().startswith(b"%PDF"))
            captions = (figures / "captions.md").read_text(encoding="utf-8")
            self.assertIn("26 development participants", captions)
            self.assertIn("middle 50%", captions)
            self.assertIn("no minimum", captions.lower())

            refreshed = tmp / "refresh"
            with patch.object(sensitivity.fe01, "load_tracker_table") as load, \
                    contextlib.redirect_stdout(io.StringIO()):
                sensitivity.main(["--figures-from", str(output), "--output-dir", str(refreshed)])
                load.assert_not_called()
            self.assertEqual({f.name for f in refreshed.glob("*.pdf")}, expected)
            self.assertEqual((refreshed / "captions.md").read_text(encoding="utf-8"), captions)

            bad = pd.read_csv(holdout)
            bad.loc[0, "split"] = "test"
            bad.to_csv(holdout, index=False)
            with patch.object(sensitivity.fe01, "load_tracker_table") as load:
                with self.assertRaisesRegex(ValueError, "26 unique"):
                    sensitivity.run(trials, tmp, catalogue, holdout, tmp/"bad")
                load.assert_not_called()
