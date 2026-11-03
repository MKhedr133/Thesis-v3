"""Synthetic checks for prospective fuzzy recommendations (MOD-18)."""
from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
import math
import sys
import unittest

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))
import MOD_18_fuzzy_recommendations as MOD18
import MOD_14_direct_linear_regression_evaluation as MOD14


def make_split():
    rows = []
    for prefix, group in (("Y", "Young"), ("O", "Old")):
        for i in range(1, 17):
            rows.append({
                "participant_id": f"{prefix}{i:02d}", "participant_group": group,
                "split": "test" if i in (1, 2, 3) else "train",
            })
    return pd.DataFrame(rows)


def make_data():
    rows = []
    for prefix, group in (("Y", "Young"), ("O", "Old")):
        for i in range(1, 17):
            pid = f"{prefix}{i:02d}"
            for condition in ("Visual", "Auditory", "Cognitive"):
                for stage, level in enumerate(MOD18.LEVELS):
                    # Stage and participant variations keep all anchors nondegenerate.
                    base = i + (0 if prefix == "Y" else 3) + stage
                    measurements = {
                        "mental_demand_score_0_to_10": float(base / 4),
                        "median_time_between_qualifying_grabs_seconds": base / 5 + 3,
                        "median_time_to_target_seconds": base / 15 + .4,
                        "median_irrelevant_focus_duration_seconds": base / 200,
                        "median_head_turning_degrees": base * 2 + 60,
                        "list_recheck_count": float(20 + base),
                        "total_list_recheck_duration_seconds": base * .6 + 12,
                        "median_reach_duration_seconds": base / 50 + .3,
                        "median_reach_path_ratio": 1 + base / 100,
                        "performance_change_from_d0_percentage_points": stage * (
                            2 + i / 10),
                    }
                    rows.append({
                        "participant_id": pid, "participant_group": group,
                        "condition_name": condition, "difficulty_level": level,
                        "difficulty_stage": stage, "trial_order": stage + 1,
                        "session_id": f"{pid}_{condition}_{level}",
                        **measurements,
                    })
    return pd.DataFrame(rows)


class TestRuleContract(unittest.TestCase):
    def test_four_frozen_registries(self):
        MOD18.validate_rule_registry()
        specs = MOD14.build_model_registry()
        for model, rules in MOD18.RULES.items():
            self.assertEqual(len(rules), 10)
            allowed = set(specs[model].predictors)
            for rid, conditions in rules.items():
                self.assertEqual(len(conditions), 2 if rid[0] == "D" else 3)
                self.assertTrue(all(MOD18.FEATURES[k] in allowed for k, _ in conditions))
        self.assertNotIn("G", {k for r in MOD18.RULES["cognitive_primary"].values()
                               for k, _ in r})

    def test_shoulder_and_median_memberships(self):
        a = MOD18.Anchors(0., 2., 7., 26)
        self.assertEqual(MOD18.memberships(2., a), {"L": 0, "M": 1., "H": 0})
        self.assertAlmostEqual(MOD18.memberships(5., a)["H"], .6)
        self.assertAlmostEqual(MOD18.memberships(0., a)["L"], 1.)
        self.assertEqual(MOD18.memberships(7., a)["H"], 1.)
        self.assertAlmostEqual(sum(MOD18.memberships(3., a).values()), 1.)

    def test_tied_anchors_and_out_of_domain(self):
        self.assertEqual(MOD18.memberships(4., MOD18.Anchors(4,4,4,26)),
                         {"L":0, "M":1., "H":0})
        for a in [MOD18.Anchors(0,0,5,26), MOD18.Anchors(0,5,5,26)]:
            self.assertAlmostEqual(sum(MOD18.memberships(2., a).values()),1.)
        with self.assertRaises(MOD18.RecommendationError):
            MOD18.memberships(float("nan"), MOD18.Anchors(0,2,4,26))

    def test_max_min_not_sum(self):
        measured = {}
        for key, feature in MOD18.FEATURES.items():
            measured[feature] = {"memberships": {"L": 0., "M": 0., "H": 0.}}
        measured[MOD18.FEATURES["M"]]["memberships"]["H"] = .60
        measured[MOD18.FEATURES["F"]]["memberships"]["H"] = .63
        measured[MOD18.FEATURES["P"]]["memberships"]["H"] = .65
        measured[MOD18.FEATURES["T"]]["memberships"]["H"] = .47
        measured[MOD18.FEATURES["H"]]["memberships"]["H"] = .56
        scores, support = MOD18.evaluate_rules("visual", measured)
        self.assertAlmostEqual(scores["D1"], .60)
        self.assertAlmostEqual(scores["D4"], .63)
        self.assertAlmostEqual(support["Decrease"], .63)
        self.assertAlmostEqual(support["Increase"], 0.)

    def test_threshold_boundaries_and_conflict(self):
        empty = {"Increase": 0., "Decrease": .2}
        scores = {k: .2 if k == "D1" else 0. for k in MOD18.RULES["visual"]}
        x = MOD18.decide("visual", 2, .2, scores, empty, [], {})
        self.assertEqual((x["raw_action"], x["final_action"], x["proposed_level"]),
                         ("Decrease", "Decrease", 2))
        x = MOD18.decide("visual", 2, .21, scores, empty, [], {})
        self.assertEqual(x["final_action"], "Maintain")
        x = MOD18.decide("visual", 0, .2, scores, empty, [], {})
        self.assertEqual(x["final_action"], "Review")
        self.assertIn("LOWER_BOUNDARY", x["review_flags"])
        scores["I1"] = .3
        conflict = MOD18.decide("visual", 2, .2, scores,
                                {"Increase": .3, "Decrease": .2}, [], {})
        self.assertEqual(conflict["final_action"], "Review")
        self.assertIn("CONFLICTING", conflict["review_flags"])

    def test_missing_is_not_zero(self):
        values = {k: None for k in MOD18.FEATURES.values()}
        scores, support = MOD18.evaluate_rules("visual", values)
        self.assertTrue(math.isnan(scores["D1"]))
        x = MOD18.decide("visual", 1, .2, scores, support,
                         ["MISSING_OR_INVALID:mental_demand_score_0_to_10"], {})
        self.assertEqual(x["final_action"], "Review")

    def test_positive_performance_deterioration_blocks_increase(self):
        measured = {MOD18.FEATURES["G"]: {"value": 40.,
                                          "memberships": {"H": .1}}}
        x = MOD18.decide("cognitive_later", 2, .2, {},
                         {"Increase": .6, "Decrease": .0}, [], measured, "available")
        self.assertEqual(x["final_action"], "Review")
        self.assertIn("INCREASE_WITH_D0_PERFORMANCE_DROP", x["review_flags"])


class TestPipeline(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.all = make_data()
        cls.dev, cls.test, _, _ = MOD14.prepare_split_data(
            modeling_data=cls.all, holdout_split=make_split())
        cls.registry = MOD14.build_model_registry()
        cls.refs, cls.calibration = MOD18.calibrate(cls.dev)

    def test_dev_only_anchors_and_four_stage_eligibility(self):
        for model, spec in self.registry.items():
            for stage in spec.allowed_stages:
                a = self.refs[(model, stage, spec.predictors[0])]
                self.assertEqual(a.count, 26)
        self.assertNotIn(("cognitive_later", 0,
                          MOD18.FEATURES["M"]), self.refs)

    def test_expected_model_stage_records(self):
        results = {model: MOD18.evaluate_model(model, self.test,
                                                self.refs, self.registry)
                   for model in self.registry}
        expected = {"visual": 24, "auditory": 24,
                    "cognitive_primary": 24, "cognitive_later": 18}
        self.assertEqual(sum(expected.values()), 90)
        for model, (recs, members, scores) in results.items():
            self.assertEqual(len(recs), expected[model] * 2)
            self.assertEqual(len(scores), expected[model] * 10)
            self.assertEqual(len(members),
                             expected[model] * len(self.registry[model].predictors))
            self.assertFalse(recs["participant_id"].isin(
                self.dev["participant_id"]).any())

    def test_later_baseline_future_is_flagged(self):
        modified = self.test.copy()
        bad_id = modified["participant_id"].iloc[0]
        ix = (modified["participant_id"].eq(bad_id)
              & modified["condition_name"].eq("Cognitive")
              & modified["difficulty_level"].eq(0))
        modified.loc[ix, "trial_order"] = 99
        recs, _, _ = MOD18.evaluate_model("cognitive_later", modified,
                                         self.refs, self.registry)
        bad = recs.loc[recs["participant_id"].eq(bad_id)]
        self.assertTrue(bad["review_flags"].str.contains(
            "D0_BASELINE_NOT_YET_OBSERVED").all())
        self.assertTrue(bad["final_action"].eq("Review").all())

    def test_output_exports_and_one_compact_plot(self):
        with TemporaryDirectory() as temp:
            folder = Path(temp) / "mod18"
            results = {model: MOD18.evaluate_model(model, self.test,
                         self.refs, self.registry) for model in self.registry}
            MOD18.write_outputs(folder, "test", self.calibration,
                                MOD18.rule_registry_table(), results,
                                render_plots=False, plot_threshold=.2,
                                policy_id="synthetic-lock")
            self.assertTrue((folder / "cross_model_summary.csv").exists())
            self.assertTrue((folder / "manifest.json").exists())
            routed = pd.read_csv(folder / "routed_recommendations.csv")
            self.assertEqual(len(routed), 72 * 2)
            self.assertEqual(len(pd.read_csv(
                folder / "cognitive_later" / "recommendations.csv")), 36)
            recs, members, scores = results["auditory"]
            one = recs.iloc[0]
            select = members.loc[
                members["participant_id"].eq(one["participant_id"])
                & members["difficulty_level"].eq(one["difficulty_level"])]
            activ = scores.loc[
                scores["participant_id"].eq(one["participant_id"])
                & scores["difficulty_level"].eq(one["difficulty_level"])]
            image_path = folder / "auditory" / "figures" / "D0" / "example.png"
            MOD18.plot_trial(one, select, activ, image_path)
            self.assertTrue(image_path.is_file())
            self.assertGreater(image_path.stat().st_size, 5000)
            with self.assertRaises(MOD18.RecommendationError):
                MOD18.write_outputs(folder, "test", self.calibration,
                                    MOD18.rule_registry_table(), results,
                                    False, .2, "synthetic-lock")

    def test_cli_denies_test_without_lock(self):
        with TemporaryDirectory() as temp:
            base = Path(temp)
            data, split = base / "model.csv", base / "split.csv"
            self.all.to_csv(data, index=False)
            make_split().to_csv(split, index=False)
            args = ["--modeling-data", str(data),
                    "--holdout-split", str(split),
                    "--output-dir", str(base / "output"), "--scope", "test"]
            with self.assertRaises(MOD18.RecommendationError):
                MOD18.main(args)
            self.assertFalse((base / "output").exists())


if __name__ == "__main__":
    unittest.main()
