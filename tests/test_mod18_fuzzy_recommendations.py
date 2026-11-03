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

    def test_development_summary_plots_without_participant_figures(self):
        with TemporaryDirectory() as temp:
            destination = Path(temp) / "development"
            results = {
                model: MOD18.evaluate_model(model, self.dev, self.refs, self.registry)
                for model in self.registry
            }
            MOD18.write_outputs(
                destination, "development", self.calibration,
                MOD18.rule_registry_table(), results,
                render_plots=False, plot_threshold=.2, policy_id=None,
                render_summary_plots=True,
            )
            images = list(destination.rglob("*.png"))
            self.assertEqual(len(images), 8)
            for model in self.registry:
                folder = destination / model / "summary_figures"
                expected = {
                    f"{model}_rule_activation_heatmap.png",
                    f"{model}_threshold_comparison.png",
                }
                self.assertEqual({path.name for path in folder.glob("*.png")},
                                 expected)
                self.assertFalse((destination / model / "figures").exists())
                self.assertTrue(
                    all(path.stat().st_size > 3000 for path in folder.glob("*.png"))
                )
            with (destination / "manifest.json").open(encoding="utf-8") as file:
                import json
                manifest = json.load(file)
            self.assertEqual(manifest["plot_mode"], "aggregate_development")


    def test_distribution_excludes_review_and_uses_actionable_denominator(self):
        dev = pd.DataFrame([
            {"threshold": .1, "final_action": "Increase"},
            {"threshold": .1, "final_action": "Maintain"},
            {"threshold": .1, "final_action": "Decrease"},
            {"threshold": .1, "final_action": "Review"},
            {"threshold": .2, "final_action": "Review"},
            {"threshold": .2, "final_action": "Maintain"},
            {"threshold": .2, "final_action": "Review"},
            {"threshold": .2, "final_action": "Review"},
        ])
        test = pd.DataFrame([
            {"threshold": .1, "final_action": "Increase"},
            {"threshold": .1, "final_action": "Increase"},
            {"threshold": .1, "final_action": "Review"},
            {"threshold": .2, "final_action": "Review"},
            {"threshold": .2, "final_action": "Review"},
            {"threshold": .2, "final_action": "Review"},
        ])
        stats = MOD18.recommendation_distribution_table(dev, test)
        self.assertEqual(len(stats), 12)
        self.assertEqual(set(stats["action"]), {"Increase", "Maintain", "Decrease"})
        def one(scope, threshold, action):
            return stats.loc[
                stats["scope"].eq(scope) &
                np.isclose(stats["threshold"], threshold) &
                stats["action"].eq(action)
            ].iloc[0]

        self.assertEqual(one("development", .1, "Increase")["count"], 1)
        self.assertEqual(one("development", .1, "Increase")["denominator"], 3)
        self.assertAlmostEqual(
            one("development", .1, "Increase")["percentage"], 100 / 3
        )
        self.assertEqual(one("development", .2, "Maintain")["count"], 1)
        self.assertEqual(one("development", .2, "Maintain")["percentage"], 100)
        self.assertEqual(one("test", .1, "Increase")["denominator"], 2)
        self.assertEqual(one("test", .1, "Increase")["percentage"], 100)
        self.assertEqual(one("test", .2, "Increase")["denominator"], 0)
        self.assertTrue(math.isnan(one("test", .2, "Increase")["percentage"]))
        with TemporaryDirectory() as temp:
            image = Path(temp) / "distribution.png"
            MOD18.plot_recommendation_distribution("visual", dev, test, image)
            self.assertTrue(image.exists())
            self.assertGreater(image.stat().st_size, 4000)

    def test_display_stages_do_not_change_data_levels(self):
        self.assertEqual(
            [MOD18.stage_label(level) for level in MOD18.LEVELS],
            ["Stage 0", "Stage 1", "Stage 2", "Stage 3"]
        )
        with self.assertRaises(MOD18.RecommendationError):
            MOD18.stage_label(5)

    def test_explanation_uses_actual_if_then_rule(self):
        d4 = MOD18.readable_rule("visual", "D4")
        self.assertEqual(
            d4,
            "IF Irrelevant focus is High AND Grab interval is High\n"
            "THEN Decrease difficulty",
        )
        i1 = MOD18.readable_rule("cognitive_later", "I1")
        self.assertEqual(i1.count(" AND "), 2)
        self.assertIn("THEN Increase difficulty", i1)
        self.assertIn("Stage 0 performance drop is Low", i1)
        with self.assertRaises(MOD18.RecommendationError):
            MOD18.readable_rule("visual", "INVALID")

    def test_compact_recommendation_card_contains_written_rule_and_other_four(self):
        from unittest.mock import patch
        from matplotlib.axes import Axes
        row = pd.Series({
            "model": "visual", "participant_id": "T01",
            "difficulty_level": 6, "threshold": .2,
            "raw_action": "Decrease", "final_action": "Decrease",
            "proposed_level": 2, "strongest_rules": "D4",
            "increase_support": .05, "decrease_support": .63,
            "review_flags": "",
        })
        activation = pd.DataFrame([
            {"rule_id": rule, "score": (.63 if rule == "D4"
                                       else .41 if rule == "D1" else .06)}
            for rule in MOD18.RULES["visual"]
        ])
        messages = []
        headings = []
        real_text = Axes.text
        real_set_title = Axes.set_title

        def capture_text(axis, x, y, text, *args, **kwargs):
            messages.append(str(text))
            return real_text(axis, x, y, text, *args, **kwargs)

        def capture_title(axis, title, *args, **kwargs):
            # set_title creates a Text artist without calling Axes.text().
            result = real_set_title(axis, title, *args, **kwargs)
            headings.append(axis.get_title(loc=kwargs.get("loc", "center")))
            return result

        with TemporaryDirectory() as temp:
            output = Path(temp) / "example.png"
            with patch.object(Axes, "text", new=capture_text), \
                 patch.object(Axes, "set_title", new=capture_title):
                MOD18.plot_trial(row, pd.DataFrame(), activation, output)
            self.assertTrue(output.exists())
        joined = "\n".join(messages)
        self.assertIn("ASSIGNED STAGE", joined)
        self.assertIn("RECOMMENDATION", joined)
        self.assertIn("Stage 2", joined)
        self.assertIn("DECREASE  → Stage 1", joined)
        self.assertNotIn("D6", joined)
        self.assertIn("IF Irrelevant focus is High", joined)
        self.assertIn("THEN Decrease difficulty", joined)
        self.assertIn("OTHER DECREASE RULES", headings)
        self.assertNotIn("D4", [heading for heading in headings])
        self.assertNotIn("LIMITATIONS", joined)

    def test_development_heatmap_contains_observation_level_columns(self):
        recs, mem, scores = MOD18.evaluate_model(
            "visual", self.dev, self.refs, self.registry
        )
        self.assertEqual(scores["participant_id"].nunique(), 26)
        self.assertEqual(
            scores[["participant_id", "difficulty_level"]].drop_duplicates().shape[0],
            104,
        )
        from unittest.mock import patch
        from matplotlib.axes import Axes
        shapes = []
        stage_ticks = []
        original_imshow = Axes.imshow
        original_set_xticks = Axes.set_xticks

        def capture_image(axis, matrix, *args, **kwargs):
            shapes.append(np.asarray(matrix).shape)
            return original_imshow(axis, matrix, *args, **kwargs)

        def capture_xticks(axis, ticks, labels=None, *args, **kwargs):
            if labels is not None and len(labels) == 4:
                stage_ticks.extend(labels)
            return original_set_xticks(axis, ticks, labels=labels, *args, **kwargs)

        with TemporaryDirectory() as temp:
            output = Path(temp) / "visual_heatmap.png"
            with patch.object(Axes, "imshow", new=capture_image), \
                 patch.object(Axes, "set_xticks", new=capture_xticks):
                MOD18.plot_rule_activation_heatmap("visual", scores, output)
            self.assertTrue(output.exists())
            self.assertGreater(output.stat().st_size, 3000)
        self.assertEqual(shapes, [(10, 104)])
        self.assertEqual(stage_ticks, ["Stage 0", "Stage 1", "Stage 2", "Stage 3"])

    def test_combined_figure_set_includes_98_pngs_and_scoped_csvs(self):
        from unittest.mock import patch

        dev_results = {
            model: MOD18.evaluate_model(model, self.dev, self.refs, self.registry)
            for model in self.registry
        }
        test_results = {
            model: MOD18.evaluate_model(model, self.test, self.refs, self.registry)
            for model in self.registry
        }
        def cheap_trial_renderer(row, members, activations, target):
            # The real trial renderer has its own visual smoke test above.
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(b"PNG-TEST-SENTINEL")

        with TemporaryDirectory() as temp:
            destination = Path(temp) / "MOD_18_png_figures"
            with patch.object(MOD18, "plot_trial", side_effect=cheap_trial_renderer):
                produced = MOD18.write_figure_set(
                    destination, dev_results, test_results, plot_threshold=.20,
                    calibration=self.calibration, policy_id="synthetic-lock"
                )
            self.assertEqual(produced, 98)
            self.assertEqual(len(list((destination / "summary").glob("*.png"))), 8)
            self.assertEqual(len(list((destination / "test").rglob("*.png"))), 90)
            self.assertEqual(len(list(destination.rglob("*.png"))), 98)
            self.assertTrue(
                (destination / "data" / "recommendation_distribution.csv").is_file()
            )
            distribution = pd.read_csv(
                destination / "data" / "recommendation_distribution.csv"
            )
            self.assertEqual(len(distribution), 4 * 2 * 2 * 3)
            self.assertEqual(set(distribution["scope"]), {"development", "test"})
            self.assertEqual(
                set(distribution["action"]), {"Increase", "Maintain", "Decrease"}
            )
            for scope in ("development", "test"):
                self.assertTrue(
                    (destination / "data" / scope / "visual"
                     / "recommendations.csv").is_file()
                )
                self.assertTrue(
                    (destination / "data" / scope / "cognitive_later"
                     / "rule_activations.csv").is_file()
                )
                self.assertTrue(
                    (destination / "data" / scope / "manifest.json").is_file()
                )
            test_recs = pd.read_csv(
                destination / "data" / "test" / "visual" / "recommendations.csv"
            )
            self.assertEqual(set(test_recs["difficulty_level"]), {0, 2, 6, 10})
            self.assertEqual(
                len(pd.read_csv(
                    destination / "data" / "test" / "cognitive_later"
                    / "recommendations.csv"
                )), 36
            )
            self.assertTrue(
                (destination / "summary"
                 / "visual_rule_activation_heatmap.png").exists()
            )
            self.assertTrue(
                (destination / "summary"
                 / "visual_recommendation_distribution.png").exists()
            )
            self.assertFalse(
                (destination / "summary"
                 / "visual_threshold_comparison.png").exists()
            )
            self.assertTrue(
                (destination / "test" / "cognitive_later"
                 / "participant_Y01_D10.png").exists()
            )
            with self.assertRaises(MOD18.RecommendationError):
                MOD18.write_figure_set(
                    destination, dev_results, test_results, plot_threshold=.20,
                    calibration=self.calibration, policy_id="synthetic-lock"
                )

    def test_combined_figure_set_rejects_split_contamination(self):
        dev_results = {
            model: MOD18.evaluate_model(model, self.dev, self.refs, self.registry)
            for model in self.registry
        }
        test_results = {
            model: MOD18.evaluate_model(model, self.test, self.refs, self.registry)
            for model in self.registry
        }
        # Introduce a development ID in a held-out recommendation table.
        recommendations, memberships, activations = test_results["visual"]
        tampered = recommendations.copy()
        tampered.loc[tampered.index[0], "participant_id"] = (
            dev_results["visual"][0]["participant_id"].iloc[0]
        )
        test_results["visual"] = (tampered, memberships, activations)
        with TemporaryDirectory() as temp:
            destination = Path(temp) / "figures"
            with self.assertRaisesRegex(
                MOD18.RecommendationError, "participant boundary"
            ):
                MOD18.write_figure_set(
                    destination, dev_results, test_results, plot_threshold=.20,
                    calibration=self.calibration, policy_id="synthetic-lock"
                )
            self.assertFalse(destination.exists())

    def test_combined_cli_requires_policy_freeze(self):
        with TemporaryDirectory() as temp:
            base = Path(temp)
            args = [
                "--modeling-data", str(base / "unread.csv"),
                "--holdout-split", str(base / "unread_split.csv"),
                "--output-dir", str(base / "output"),
                "--render-figure-set",
            ]
            with self.assertRaisesRegex(
                MOD18.RecommendationError, "actual policy approval"
            ):
                MOD18.main(args)
            self.assertFalse((base / "output").exists())
            flags = MOD18.parser().parse_args([
                *args, "--enable-locked-test-evaluation",
                "--frozen-policy-id", "approved-and-frozen-version"
            ])
            self.assertTrue(flags.render_figure_set)

    def test_plot_modes_reject_wrong_scope_without_writing_files(self):
        results = {
            model: MOD18.evaluate_model(model, self.dev, self.refs, self.registry)
            for model in self.registry
        }
        with TemporaryDirectory() as temp:
            base = Path(temp)
            cases = [
                ("development", True, False),
                ("test", False, True),
                ("development", True, True),
            ]
            for i, (scope, individual, summary) in enumerate(cases):
                dest = base / str(i)
                with self.subTest(scope=scope, individual=individual, summary=summary):
                    with self.assertRaises(MOD18.RecommendationError):
                        MOD18.write_outputs(
                            dest, scope, self.calibration,
                            MOD18.rule_registry_table(), results,
                            render_plots=individual, plot_threshold=.2,
                            policy_id=None, render_summary_plots=summary,
                        )
                    self.assertFalse(dest.exists())

    def test_cli_accepts_summary_option_for_development(self):
        args = MOD18.parser().parse_args([
            "--modeling-data", "model.csv", "--holdout-split", "split.csv",
            "--output-dir", "outputs", "--scope", "development",
            "--render-summary-plots",
        ])
        self.assertTrue(args.render_summary_plots)
        self.assertFalse(args.render_plots)

    def test_main_rejects_development_individual_plots(self):
        with TemporaryDirectory() as temp:
            base = Path(temp)
            args = [
                "--modeling-data", str(base / "missing.csv"),
                "--holdout-split", str(base / "missing_split.csv"),
                "--output-dir", str(base / "output"),
                "--scope", "development", "--render-plots",
            ]
            with self.assertRaisesRegex(
                    MOD18.RecommendationError, "requires --scope test"):
                MOD18.main(args)
            self.assertFalse((base / "output").exists())

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
