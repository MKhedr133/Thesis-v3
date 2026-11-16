"""Check all-grab extraction and figure export on synthetic tracker records."""
import contextlib
import importlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))
extractor = importlib.import_module("09_extract_final_features")


class AllGrabTests(unittest.TestCase):
    def test_all_grabs_keep_focus_gate_and_shift_later_windows(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            performance = root / "performance.csv"
            performance.write_text(
                "participant;session;trial;group;condition;trial2;level3;language;performance\n"
                "P1;S001;T001;Young;Visual;T1;0;EN;15\n", encoding="utf-8")
            catalogue = root / "CorrectLists.txt"
            catalogue.write_text("\n".join(
                f"correctLists.level{k}.{lang} = {{'Apple', 'Bread'}};"
                for k in (0, 2, 6, 10) for lang in ("EN", "NL")), encoding="utf-8")
            time = np.arange(501) / 100
            starts = [0.8, 1.6, 2.4, 3.2, 4.0]
            names = ["Apple", "Pear", "Apple", "Bread", "Pear"]
            segments = [(time >= s) & (time < s + .25) for s in starts]
            focus = [(time >= s - .3 - 1e-9) & (time < s + .25) for s in starts[:-1]]
            focus.insert(0, time < .2)
            focus.insert(1, (time >= 1.12) & (time < 1.20))
            delta = sum(np.clip(time - (s - .29), 0, .29) for s in starts)
            tracker = pd.DataFrame({
                "time": time, "enableBlackout": False,
                "focus_object_name": np.select(focus, ["Tablet", "Tablet", *names[:-1]], default="Side R"),
                "focus_object_tag": np.select(focus[:2], ["Tablet", "Tablet"], default="MainShelf"),
                "is_grabbing_right": np.logical_or.reduce(segments),
                "grabbed_object_right": np.select(segments, names, default="noObjectGrabbed"),
                "is_grabbing_left": False, "grabbed_object_left": "noObjectGrabbed",
                "right_pos_x": delta, "right_pos_y": 0., "right_pos_z": 0.,
                "hmd_rot_w": 1., "hmd_rot_x": 0., "hmd_rot_y": 0., "hmd_rot_z": 0.,
            })
            raw = root / "raw"
            trackers = raw / "P1" / "P01_HMD_Data" / "P01" / "S001" / "trackers"
            trackers.mkdir(parents=True)
            filename = "data_collector_vr_sample_T001.csv"
            tracker.to_csv(trackers / filename, index=False)

            legacy = root / "legacy"
            extractor.run_extraction(performance, raw, catalogue, legacy)
            old_grabs = pd.read_csv(legacy / "product_grab_events.csv")
            self.assertEqual(old_grabs.search_interval_valid.notna().tolist(), [True, False, False, True, False])
            self.assertEqual(old_grabs.search_start_seconds.iloc[3], 1.2)
            self.assertEqual(json.loads((legacy / "feature_extraction_manifest.json").read_text())["schema_version"],
                             "fe02_v1")

            revised = root / "revised"
            with contextlib.redirect_stdout(io.StringIO()):
                extractor.main(["--performance", str(performance), "--raw-root", str(raw),
                            "--correct-lists", str(catalogue), "--output-dir", str(revised),
                            "--all-grabs", "--worked-example", "P01", "S001", filename, "4"])
            grabs = pd.read_csv(revised / "product_grab_events.csv")
            self.assertEqual(grabs.canonical_product_name.tolist(), ["apple", "pear", "apple", "bread", "pear"])
            self.assertEqual(grabs.search_interval_valid.tolist(), [True, True, True, True, False])
            self.assertEqual(grabs.reach_interval_valid.tolist(), [True, True, True, True, False])
            np.testing.assert_allclose(grabs.search_start_seconds, [.2, 1.2, 1.6, 2.4, 3.2])
            np.testing.assert_allclose(grabs.first_target_focus_seconds.iloc[:4], [.5, 1.3, 2.1, 2.9])
            self.assertTrue(grabs.loc[4, ["first_target_focus_seconds", "reach_start_seconds",
                                          "reach_duration_seconds", "reach_path_ratio"]].isna().all())
            np.testing.assert_allclose(grabs.reach_path_ratio.iloc[:4], 1.)
            new_qc = pd.read_csv(revised / "feature_extraction_qc.csv").iloc[0]
            self.assertEqual(new_qc.first_time_on_list_grab_count, 2)
            self.assertEqual(new_qc.valid_search_interval_count, 4)
            self.assertEqual(new_qc.valid_reach_duration_count, 4)
            trace = json.loads((revised / "worked_trace.json").read_text())
            self.assertEqual(trace["events"]["previous_grab_start"], trace["events"]["search_start"])
            self.assertAlmostEqual(trace["events"]["grab_start"] - trace["events"]["search_start"], .8)
            before = pd.read_csv(legacy / "trial_features.csv").iloc[0]
            after = pd.read_csv(revised / "trial_features.csv").iloc[0]
            pd.testing.assert_series_equal(before[list(extractor.FEATURE_ORDER[:5])],
                                           after[list(extractor.FEATURE_ORDER[:5])])
            self.assertAlmostEqual(after["median_product_search_duration_seconds"], .4)
            self.assertAlmostEqual(after["median_reach_start_to_grab_seconds"], .29)
            manifest = json.loads((revised / "feature_extraction_manifest.json").read_text())
            self.assertEqual(manifest["search_and_reach_grab_scope"], "all_valid_product_grabs")
            self.assertEqual(manifest["trial_features_sha256"], extractor.hashlib.sha256(
                (revised / "trial_features.csv").read_bytes()).hexdigest())


    def test_simultaneous_grabs_use_previous_earlier_start_and_their_own_hand(self):
        fe = extractor.fe01
        time = np.arange(201) / 100
        tracker = pd.DataFrame({
            "time": time, "enableBlackout": False,
            "right_pos_x": np.clip(time - .65, 0., .45),
            "left_pos_x": np.clip(time - .85, 0., .25),
            "right_pos_y": 0., "right_pos_z": 0., "left_pos_y": 0., "left_pos_z": 0.,
        })
        grabs = tuple(fe.ProductGrabEvent("pear", "Pear", hand, 1.1, 1.4, .3, False, False, True)
                      for hand in ("left", "right"))
        focus = (fe.FocusEpisode("pear", "Pear", "mainshelf", "MainShelf", .5, 1.3, .8),)
        cache = fe._TrialSignalCache(tracker)
        search = fe._detect_search_intervals_with_cache(
            tracker, grabs, (), focus, cache=cache, all_grabs=True).intervals
        self.assertEqual([s.search_start_seconds for s in search], [0., 0.])
        reaches = fe._detect_reach_intervals_with_cache(tracker, grabs,
                      search_intervals=search, cache=cache, all_grabs=True).intervals
        table = fe._build_product_grab_features_with_cache(
            {}, grabs, search, reaches, tracker, cache=cache, all_grabs=True).table
        self.assertEqual(table.hand.tolist(), ["left", "right"])
        np.testing.assert_allclose(table.reach_start_seconds, [.85, .65])

    def test_sustained_target_focus_at_window_start_yields_zero_search_duration(self):
        fe = extractor.fe01
        tracker = pd.DataFrame({"time": [0., 1., 2., 3., 4.], "enableBlackout": False})
        grabs = (
            fe.ProductGrabEvent("pear", "Pear", "left", 1., 1.3, .3, False, False, False),
            fe.ProductGrabEvent("pear", "Pear", "left", 3., 3.4, .4, False, False, False),
        )
        focus = (fe.FocusEpisode("pear", "Pear", "mainshelf", "MainShelf", .5, 3.5, 3.),)
        cache = fe._TrialSignalCache(tracker)
        first, second = fe._detect_search_intervals_with_cache(
            tracker, grabs, (), focus, cache=cache, all_grabs=True).intervals
        self.assertTrue(second.is_valid)
        self.assertEqual(first.first_target_focus_seconds, .5)
        self.assertEqual(second.first_target_focus_seconds, second.search_start_seconds)


    def test_plotter_checks_provenance_and_renders_empty_group_and_wide_values(self):
        plotter = importlib.import_module("11_plot_descriptive_features")
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "trial_features.csv"
            frame = pd.DataFrame(
                [(f"S{n:02d}", condition, level) for n in range(32)
                 for condition in ("Visual", "Auditory", "Cognitive") for level in range(4)],
                columns=["participant_id", "condition_name", "difficulty_level"],
            )
            for feature in extractor.FEATURE_ORDER:
                frame[feature] = 1.0
            focus = "median_product_search_duration_seconds"
            frame.loc[frame.condition_name.eq("Visual") & frame.difficulty_level.eq(0), focus] = np.nan
            frame.loc[1, focus] = 7.5
            frame.to_csv(source, index=False)
            manifest_path = root / "feature_extraction_manifest.json"
            with self.assertRaisesRegex(ValueError, "manifest"):
                plotter.load_trials(source)
            manifest = {"schema_version": "fe04_all_grabs_v1",
                        "search_and_reach_grab_scope": "all_valid_product_grabs",
                        "reach_requires_target_focus": True,
                        "trial_count": 384,
                        "trial_features_sha256": extractor.hashlib.sha256(source.read_bytes()).hexdigest()}
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            figures = root / "figures"
            plotter.main([str(source), str(figures)])
            stats = pd.read_csv(figures / "descriptive_plot_statistics.csv")
            empty = stats.loc[stats.feature.eq(focus) & stats.condition.eq("Visual")
                              & stats.difficulty_level.eq(0)].iloc[0]
            self.assertEqual((empty["n"], empty["missing"]), (0, 32))
            self.assertTrue(pd.isna(empty["median"]))
            self.assertTrue((figures / "descriptive_features_all.pdf").read_bytes().startswith(b"%PDF-"))
            self.assertIn("n = 0", (figures / "figure_captions.md").read_text(encoding="utf-8"))
            self.assertIn("fe04_all_grabs_v1",
                          json.loads((figures / "plot_manifest.json").read_text())["source_schema"])
            data, _, _ = plotter.load_trials(source)
            fig = plotter.draw(data, plotter.FEATURES[5:6], 5)
            try:
                self.assertGreaterEqual(fig.axes[0].get_ylim()[1], 7.5)
                self.assertEqual(fig.axes[0].get_ylim(), fig.axes[2].get_ylim())
                self.assertIn("n = 0", [x.get_text() for x in fig.axes[0].texts])
            finally:
                plotter.plt.close(fig)
            source.write_bytes(source.read_bytes() + b"\n")
            with self.assertRaisesRegex(ValueError, "SHA-256"):
                plotter.load_trials(source)


if __name__ == "__main__":
    unittest.main()
