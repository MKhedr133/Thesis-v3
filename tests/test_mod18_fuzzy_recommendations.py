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

    def test_only_impossible_transitions_require_review(self):
        scores = {k: 0. for k in MOD18.RULES["visual"]}
        decrease = {"Increase": 0., "Decrease": .20}
        increase = {"Increase": .30, "Decrease": 0.}
        x = MOD18.decide("visual", 2, .20, scores, decrease, [], {})
        self.assertEqual((x["raw_action"], x["final_action"], x["proposed_level"]),
                         ("Decrease", "Decrease", 2))
        x = MOD18.decide("visual", 2, .21, scores, decrease, [], {})
        self.assertEqual(x["final_action"], "Maintain")
        lower = MOD18.decide("visual", 0, .20, scores, decrease, [], {})
        self.assertEqual(lower["final_action"], "Review")
        self.assertEqual(lower["recommendation_status"], "boundary_review")
        self.assertIsNone(lower["proposed_level"])
        self.assertEqual(lower["review_flags"], "LOWER_BOUNDARY")
        upper = MOD18.decide("visual", 3, .20, scores, increase, [], {})
        self.assertEqual(upper["final_action"], "Review")
        self.assertEqual(upper["review_flags"], "UPPER_BOUNDARY")
        self.assertIsNone(upper["proposed_level"])
        at_middle = MOD18.decide("visual", 2, .20, scores, increase, [], {})
        self.assertEqual(at_middle["final_action"], "Increase")
        self.assertEqual(at_middle["proposed_level"], 10)

    def test_conflicting_support_chooses_stronger_action_or_maintain_on_tie(self):
        scores = {k: 0. for k in MOD18.RULES["visual"]}
        cases = (
            (.40, .20, "Increase"),
            (.20, .40, "Decrease"),
            (.30, .30, "Maintain"),
        )
        for inc, dec, expected in cases:
            result = MOD18.decide(
                "visual", 1, .20, scores,
                {"Increase": inc, "Decrease": dec}, [], {}
            )
            self.assertEqual(result["final_action"], expected)
            self.assertEqual(result["review_flags"], "")
            self.assertIn("COMPETING_RULE_SUPPORT", result["diagnostic_flags"])
            if inc == dec:
                self.assertIn("TIED_RULE_SUPPORT", result["diagnostic_flags"])

    def test_valid_outside_calibration_is_diagnostic_not_review(self):
        scores = {k: 0. for k in MOD18.RULES["visual"]}
        result = MOD18.decide(
            "visual", 1, .20, scores,
            {"Increase": .45, "Decrease": .05},
            ["OUTSIDE_CALIBRATION:time"], {}
        )
        self.assertEqual(result["final_action"], "Increase")
        self.assertIn("OUTSIDE_CALIBRATION:time",
                      result["diagnostic_flags"])
        self.assertEqual(result["review_flags"], "")

    def test_missing_is_not_zero_and_not_a_review(self):
        values = {k: None for k in MOD18.FEATURES.values()}
        scores, support = MOD18.evaluate_rules("visual", values)
        self.assertTrue(math.isnan(scores["D1"]))
        result = MOD18.decide(
            "visual", 1, .2, scores, support,
            ["MISSING_OR_INVALID:mental_demand_score_0_to_10"], {}
        )
        self.assertEqual(result["final_action"], "Unavailable")
        self.assertEqual(result["recommendation_status"], "data_unavailable")
        self.assertIsNone(result["proposed_level"])
        self.assertEqual(result["review_flags"], "")
        self.assertIn("MISSING_OR_INVALID", result["diagnostic_flags"])

    def test_positive_performance_deterioration_is_diagnostic_not_review(self):
        measured = {MOD18.FEATURES["G"]: {
            "value": 40., "memberships": {"H": .1}
        }}
        result = MOD18.decide(
            "cognitive_later", 2, .2, {},
            {"Increase": .6, "Decrease": .0}, [], measured,
            "participant_d0_reference"
        )
        self.assertEqual(result["final_action"], "Increase")
        self.assertEqual(result["proposed_level"], 10)
        self.assertIn("INCREASE_WITH_D0_PERFORMANCE_DROP",
                      result["diagnostic_flags"])
        self.assertEqual(result["review_flags"], "")



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
            for stage in MOD18.eligible_stages(model, self.registry):
                a = self.refs[(model, stage, spec.predictors[0])]
                self.assertEqual(a.count, 26)
        self.assertNotIn(("cognitive_later", 0,
                          MOD18.FEATURES["M"]), self.refs)
        self.assertNotIn(("cognitive_primary", 1,
                          MOD18.FEATURES["M"]), self.refs)
        self.assertEqual(MOD18.eligible_stages("cognitive_primary",
                                              self.registry), (0,))
        self.assertEqual(MOD18.eligible_stages("cognitive_later",
                                              self.registry), (1, 2, 3))

    def test_expected_model_stage_records(self):
        results = {model: MOD18.evaluate_model(model, self.test,
                                                self.refs, self.registry)
                   for model in self.registry}
        expected = {"visual": 24, "auditory": 24,
                    "cognitive_primary": 6, "cognitive_later": 18}
        self.assertEqual(sum(expected.values()), 72)
        for model, (recs, members, scores) in results.items():
            self.assertEqual(len(recs), expected[model] * 2)
            self.assertEqual(len(scores), expected[model] * 10)
            self.assertEqual(len(members),
                             expected[model] * len(self.registry[model].predictors))
            self.assertFalse(recs["participant_id"].isin(
                self.dev["participant_id"]).any())
            expected_levels = {
                MOD18.LEVELS[stage]
                for stage in MOD18.eligible_stages(model, self.registry)
            }
            self.assertEqual(set(recs["difficulty_level"]), expected_levels)

    def test_later_baseline_is_participant_specific_without_trial_order(self):
        modified = self.dev.copy()
        person = str(modified["participant_id"].iloc[0])
        d0 = modified.loc[
            modified["participant_id"].eq(person)
            & modified["condition_name"].eq("Cognitive")
            & modified["difficulty_level"].eq(0)
        ].iloc[0]
        later = modified.loc[
            modified["participant_id"].eq(person)
            & modified["condition_name"].eq("Cognitive")
            & modified["difficulty_level"].eq(6)
        ].iloc[0]
        self.assertEqual(
            MOD18.baseline_status_for(later, modified, "cognitive_later"),
            "participant_d0_reference"
        )
        self.assertEqual(d0["performance_change_from_d0_percentage_points"], 0.)
        self.assertGreater(
            later["performance_change_from_d0_percentage_points"], 0.
        )
        # Deliberately place Stage 0 after Stage 2 chronologically.
        mask = (modified["participant_id"].eq(person)
                & modified["condition_name"].eq("Cognitive")
                & modified["difficulty_level"].eq(0))
        modified.loc[mask, "trial_order"] = 99
        no_order = modified.drop(columns="trial_order")
        recs, _, _ = MOD18.evaluate_model(
            "cognitive_later", no_order, self.refs, self.registry
        )
        self.assertEqual(len(recs), 26 * 3 * len(MOD18.THRESHOLDS))
        self.assertTrue(
            recs["baseline_status"].eq("participant_d0_reference").all()
        )
        self.assertFalse(
            recs["diagnostic_flags"].str.contains(
                "UNVERIFIED_D0_BASELINE_ORDER|D0_BASELINE_NOT_YET_OBSERVED",
                regex=True, na=False,
            ).any()
        )
        self.assertTrue(recs["raw_action"].isin(
            {"Increase", "Decrease", "Maintain", "Unavailable"}
        ).all())
        self.assertTrue(
            recs["final_action"].isin(
                {"Increase", "Decrease", "Maintain", "Review", "Unavailable"}
            ).all()
        )
        # A missing participant-matched D0 cannot be silently substituted
        # by a different participant's Stage 0 record.
        no_baseline = modified.drop(d0.name)
        self.assertEqual(
            MOD18.baseline_status_for(later, no_baseline, "cognitive_later"),
            "missing_participant_baseline"
        )
        duplicated = pd.concat([modified, d0.to_frame().T], ignore_index=True)
        self.assertEqual(
            MOD18.baseline_status_for(later, duplicated, "cognitive_later"),
            "ambiguous_participant_baseline"
        )
        result = MOD18.decide(
            "cognitive_later", 1, .2, {},
            {"Increase": .6, "Decrease": .0}, [], {},
            "missing_participant_baseline",
        )
        self.assertEqual(result["final_action"], "Unavailable")
        self.assertIn("MISSING_PARTICIPANT_D0_BASELINE",
                      result["diagnostic_flags"])
        self.assertEqual(result["review_flags"], "")

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
            self.assertTrue(
                (folder / "cognitive_later" / "unavailable_cases.csv").exists()
            )
            self.assertIn("diagnostic_flags", pd.read_csv(
                folder / "cognitive_later" / "recommendations.csv"
            ).columns)
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
            {"threshold": .1, "final_action": "Increase", "raw_action": "Increase"},
            {"threshold": .1, "final_action": "Maintain", "raw_action": "Maintain"},
            {"threshold": .1, "final_action": "Decrease", "raw_action": "Decrease"},
            {"threshold": .1, "final_action": "Review", "raw_action": "Decrease"},
            {"threshold": .2, "final_action": "Review", "raw_action": "Increase"},
            {"threshold": .2, "final_action": "Maintain", "raw_action": "Maintain"},
            {"threshold": .2, "final_action": "Review", "raw_action": "Decrease"},
            {"threshold": .2, "final_action": "Review", "raw_action": "Maintain"},
        ])
        test = pd.DataFrame([
            {"threshold": .1, "final_action": "Increase", "raw_action": "Increase"},
            {"threshold": .1, "final_action": "Increase", "raw_action": "Increase"},
            {"threshold": .1, "final_action": "Review", "raw_action": "Decrease"},
            {"threshold": .2, "final_action": "Review", "raw_action": "Increase"},
            {"threshold": .2, "final_action": "Review", "raw_action": "Maintain"},
            {"threshold": .2, "final_action": "Review", "raw_action": "Decrease"},
        ])
        stats = MOD18.recommendation_distribution_table(dev, test)
        self.assertEqual(len(stats), 12)
        self.assertEqual(
            set(stats["action"]), {"Increase", "Maintain", "Decrease"}
        )

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
        self.assertEqual(one("development", .2, "Maintain")["basis"], "safeguarded")
        self.assertEqual(one("development", .2, "Maintain")["review_count"], 3)
        self.assertEqual(one("test", .1, "Increase")["denominator"], 2)
        self.assertEqual(one("test", .1, "Increase")["percentage"], 100)
        self.assertEqual(one("test", .2, "Increase")["denominator"], 3)
        self.assertAlmostEqual(one("test", .2, "Increase")["percentage"], 100 / 3)
        self.assertEqual(one("test", .2, "Increase")["basis"], "provisional")
        self.assertEqual(one("test", .2, "Increase")["review_count"], 3)
        self.assertEqual(one("test", .2, "Increase")["observation_count"], 3)

        # Even with nonzero bars, the test panel must identify provisional
        # evidence and report that every final recommendation was blocked.
        from unittest.mock import patch
        from matplotlib.axes import Axes
        original_text = Axes.text
        text_labels = []

        def capture_text(axis, x, y, label, *args, **kwargs):
            text_labels.append(str(label))
            return original_text(axis, x, y, label, *args, **kwargs)

        with TemporaryDirectory() as temp:
            image = Path(temp) / "distribution.png"
            with patch.object(Axes, "text", new=capture_text):
                MOD18.plot_recommendation_distribution("visual", dev, test, image)
            self.assertTrue(image.exists())
            self.assertGreater(image.stat().st_size, 4000)
        self.assertTrue(any("3/3 Review" in label and
                            "provisional" in label for label in text_labels))

    def test_distribution_all_review_without_raw_actions_is_not_misrepresented(self):
        # A model can lack both a final and provisional action.
        frame = pd.DataFrame([
            {"threshold": threshold, "final_action": "Review",
             "raw_action": "Review"}
            for threshold in MOD18.THRESHOLDS
        ])
        stats = MOD18.recommendation_distribution_table(frame, frame)
        self.assertTrue(stats["basis"].eq("provisional").all())
        self.assertTrue(stats["count"].eq(0).all())
        self.assertTrue(stats["percentage"].isna().all())
        self.assertTrue(stats["denominator"].eq(0).all())

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

    def test_review_is_shown_as_status_not_a_fourth_recommendation(self):
        from unittest.mock import patch
        from matplotlib.axes import Axes

        row = pd.Series({
            "model": "cognitive_later", "participant_id": "T01",
            "difficulty_level": 10, "threshold": .20,
            "raw_action": "Increase", "final_action": "Review",
            "proposed_level": np.nan, "strongest_rules": "I1",
            "increase_support": .63, "decrease_support": .03,
            "review_flags": "UPPER_BOUNDARY",
        })
        activation = pd.DataFrame([
            {"rule_id": rule, "score": .63 if rule == "I1" else .05}
            for rule in MOD18.RULES["cognitive_later"]
        ])
        labels = []
        actual_text = Axes.text

        def capture(axis, x, y, message, *args, **kwargs):
            labels.append(str(message))
            return actual_text(axis, x, y, message, *args, **kwargs)

        with TemporaryDirectory() as folder:
            target = Path(folder) / "review.png"
            with patch.object(Axes, "text", new=capture):
                MOD18.plot_trial(row, pd.DataFrame(), activation, target)
            self.assertTrue(target.exists())
        self.assertIn("STATUS", labels)
        self.assertIn("REVIEW REQUIRED", labels)
        self.assertNotIn("RECOMMENDATION", labels)
        self.assertTrue(any("PROVISIONAL" in label for label in labels))
        self.assertTrue(any("THEN Increase difficulty" in label
                            for label in labels))

    def test_development_heatmap_shows_values_by_stage(self):
        from unittest.mock import patch
        from matplotlib.axes import Axes

        original_imshow = Axes.imshow
        original_text = Axes.text
        original_title = Axes.set_title
        for model, expected_shapes, expected_stages in (
            ("visual", [(10, 26)] * 4,
             ["Stage 0", "Stage 1", "Stage 2", "Stage 3"]),
            ("cognitive_primary", [(10, 26)], ["Stage 0"]),
            ("cognitive_later", [(10, 26)] * 3,
             ["Stage 1", "Stage 2", "Stage 3"]),
        ):
            _, _, scores = MOD18.evaluate_model(
                model, self.dev, self.refs, self.registry
            )
            shapes, labels, text_cells = [], [], []

            def capture_image(axis, matrix, *args, **kwargs):
                shapes.append(np.asarray(matrix).shape)
                return original_imshow(axis, matrix, *args, **kwargs)

            def capture_title(axis, title, *args, **kwargs):
                if str(title).startswith("Stage "):
                    labels.append(str(title))
                return original_title(axis, title, *args, **kwargs)

            def capture_text(axis, x, y, content, *args, **kwargs):
                # Numeric annotation inside every individual cell.
                if isinstance(content, str) and (
                    content == "–" or content.replace(".", "").isdigit()
                ):
                    text_cells.append(content)
                return original_text(axis, x, y, content, *args, **kwargs)

            with TemporaryDirectory() as temp:
                image = Path(temp) / f"{model}.png"
                with patch.object(Axes, "imshow", new=capture_image), \
                     patch.object(Axes, "text", new=capture_text), \
                     patch.object(Axes, "set_title", new=capture_title):
                    MOD18.plot_rule_activation_heatmap(model, scores, image)
                self.assertTrue(image.is_file())
                self.assertGreater(image.stat().st_size, 5000)
            self.assertEqual(shapes, expected_shapes)
            self.assertEqual(labels, expected_stages)
            self.assertEqual(len(text_cells),
                             26 * len(expected_stages) * 10)
            self.assertTrue(all(len(value) == 4 or value == "–"
                                for value in text_cells))

    def test_data_unavailable_is_not_review_or_maintain(self):
        from unittest.mock import patch
        from matplotlib.axes import Axes
        result = pd.Series({
            "model": "visual", "participant_id": "T01",
            "difficulty_level": 2, "threshold": .20,
            "raw_action": "Unavailable", "final_action": "Unavailable",
            "proposed_level": np.nan, "strongest_rules": "",
            "increase_support": np.nan, "decrease_support": np.nan,
            "review_flags": "",
            "diagnostic_flags": "MISSING_OR_INVALID:mental_demand_score_0_to_10",
        })
        activations = pd.DataFrame([
            {"rule_id": key, "score": np.nan}
            for key in MOD18.RULES["visual"]
        ])
        output_text = []
        real_text = Axes.text

        def capture(ax, x, y, txt, *args, **kwargs):
            output_text.append(str(txt))
            return real_text(ax, x, y, txt, *args, **kwargs)

        with TemporaryDirectory() as folder:
            target = Path(folder) / "unavailable.png"
            with patch.object(Axes, "text", new=capture):
                MOD18.plot_trial(result, pd.DataFrame(), activations, target)
            self.assertTrue(target.exists())
        self.assertIn("STATUS", output_text)
        self.assertIn("DATA UNAVAILABLE", output_text)
        self.assertNotIn("REVIEW REQUIRED", output_text)
        self.assertTrue(any("MISSING OR INVALID" in line
                            for line in output_text))

    def test_unavailable_counts_are_separate_from_review_distribution(self):
        frame = pd.DataFrame([
            {"threshold": tau, "final_action": "Unavailable",
             "raw_action": "Unavailable"}
            for tau in MOD18.THRESHOLDS
        ])
        summary = MOD18.recommendation_distribution_table(frame, frame)
        self.assertTrue(summary["count"].eq(0).all())
        self.assertTrue(summary["denominator"].eq(0).all())
        self.assertTrue(summary["percentage"].isna().all())
        self.assertTrue(summary["review_count"].eq(0).all())
        self.assertTrue(summary["unavailable_count"].eq(1).all())
        self.assertTrue(summary["basis"].eq("safeguarded").all())

    def test_explicit_rule_labels_preserve_rule_ids(self):
        self.assertEqual(MOD18.rule_label("D1"), "Decrease rule 1 (D1)")
        self.assertEqual(MOD18.rule_label("I5"), "Increase rule 5 (I5)")
        with self.assertRaises(MOD18.RecommendationError):
            MOD18.rule_label("D6")
        registry = MOD18.rule_registry_table()
        self.assertEqual(
            registry.loc[registry["rule_id"].eq("D1"), "rule_name"].iloc[0],
            "Decrease rule 1 (D1)"
        )
        _, _, score_table = MOD18.evaluate_model(
            "cognitive_primary", self.test, self.refs, self.registry
        )
        self.assertEqual(score_table["rule_id"].nunique(), 10)
        self.assertIn("Increase rule 1 (I1)", set(score_table["rule_name"]))

    def test_combined_figure_set_includes_80_pngs_and_scoped_csvs(self):
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
            self.assertEqual(produced, 80)
            self.assertEqual(len(list((destination / "summary").glob("*.png"))), 8)
            self.assertEqual(len(list((destination / "test").rglob("*.png"))), 72)
            self.assertEqual(len(list(destination.rglob("*.png"))), 80)
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
            self.assertEqual(len(list(
                (destination / "test" / "cognitive_primary").glob("*.png")
            )), 6)
            self.assertEqual(len(list(
                (destination / "test" / "cognitive_later").glob("*.png")
            )), 18)
            self.assertEqual(set(
                pd.read_csv(destination / "data" / "test" / "cognitive_primary"
                            / "recommendations.csv")["difficulty_level"]
            ), {0})
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
