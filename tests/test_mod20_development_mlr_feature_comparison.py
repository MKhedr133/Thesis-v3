"""Synthetic checks for the development-only full/screened/add-back comparison."""

from hashlib import sha256
import json
import sys
from tempfile import TemporaryDirectory
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))

import MOD_20_development_mlr_feature_comparison as comparison


class DevelopmentComparisonTests(unittest.TestCase):
    @staticmethod
    def split():
        return pd.DataFrame([
            {
                "participant_id": f"S{i:02}",
                "participant_group": "Young" if i % 2 else "Old",
                "split": "train" if i < 26 else "test",
            }
            for i in range(32)
        ])

    @staticmethod
    def trials():
        rng = np.random.default_rng(1729)
        rows = []
        for condition in ("Visual", "Auditory", "Cognitive"):
            for i in range(32):
                person_offset = rng.normal(0, 0.9)
                for stage in range(4):
                    values = rng.normal(10, 1, size=len(comparison.TRIAL_FEATURES))
                    values[0] = 20 + stage + person_offset + rng.normal(0, 0.16)
                    rows.append({
                        "participant_id": f"S{i:02}",
                        "condition_name": condition,
                        "difficulty_stage": stage,
                        **dict(zip(comparison.TRIAL_FEATURES, values)),
                    })
        return pd.DataFrame(rows)

    def test_screen_boundary_and_restoring_only_the_predefined_pair(self):
        search = "median_product_search_duration_seconds"
        rotation = "median_head_rotation_during_search_degrees"
        path = "median_hand_travel_to_straight_distance_ratio"
        mental = "self_reported_mental_demand"
        pvalues = {search: 0.05, rotation: 0.16, path: 0.9, mental: 0.01}
        features = comparison.candidate_features(tuple(pvalues), pvalues, "Visual")
        self.assertEqual(features["screened"], (mental,))
        self.assertEqual(features["add_back"], (search, rotation, mental))

        pvalues[search] = 0.01
        pvalues[rotation] = 0.01
        features = comparison.candidate_features(tuple(pvalues), pvalues, "Visual")
        self.assertEqual(features["screened"], features["add_back"])

    def test_mae_margin_is_inclusive_and_prefers_fewer_features(self):
        scores = pd.DataFrame([
            ("full", 10, 0.70),
            ("screened", 6, 0.75),
            ("add_back", 8, 0.71),
        ], columns=["candidate", "mean_predictor_count", "participant_balanced_mae"])
        self.assertEqual(comparison.select_candidate(scores), "screened")
        scores.loc[1, "participant_balanced_mae"] = 0.751
        self.assertEqual(comparison.select_candidate(scores), "add_back")
        scores.loc[2, "participant_balanced_mae"] = 0.76
        self.assertEqual(comparison.select_candidate(scores), "full")
        scores.loc[2, ["mean_predictor_count", "participant_balanced_mae"]] = [10, 0.69]
        self.assertEqual(comparison.select_candidate(scores), "full")

    def test_identical_scores_and_counts_prefer_screened_over_add_back(self):
        scores = pd.DataFrame([
            ("full", 10, 0.70),
            ("screened", 6, 0.71),
            ("add_back", 6, 0.71),
        ], columns=["candidate", "mean_predictor_count", "participant_balanced_mae"])
        self.assertEqual(comparison.select_candidate(scores), "screened")

    def test_reserved_values_are_never_used_and_stage_grid_is_required(self):
        source = self.trials()
        clean = comparison.prepare_development(source, self.split())
        source.loc[source.participant_id.eq("S31"), "difficulty_stage"] = -99
        source.loc[source.participant_id.eq("S31"), comparison.TRIAL_FEATURES[0]] = np.inf
        pd.testing.assert_frame_equal(clean, comparison.prepare_development(source, self.split()))
        self.assertEqual(len(clean), 26 * 3 * 4)
        self.assertEqual(clean.participant_id.nunique(), 26)
        self.assertNotIn("S31", set(clean.participant_id))

        with self.assertRaisesRegex(ValueError, "one trial"):
            comparison.prepare_development(source.drop(index=0), self.split())

    def test_real_mixed_screen_and_ols_produce_paired_oof_predictions(self):
        source = self.trials()
        source.loc[0, "median_hand_travel_to_straight_distance_ratio"] = np.nan
        development = comparison.prepare_development(source, self.split())
        visual = development.loc[development.condition_name.eq("Visual")]
        result = comparison.compare_condition(visual, "Visual")
        predictions = result.predictions
        self.assertEqual(len(predictions), 103 * 3)
        groups = predictions.groupby(["candidate", "participant_id"])["round"].nunique()
        self.assertTrue(groups.eq(1).all())
        keys = ["participant_id", "difficulty_stage"]
        full = predictions.loc[predictions.candidate.eq("full"), keys].reset_index(drop=True)
        for candidate in ("screened", "add_back"):
            other = predictions.loc[predictions.candidate.eq(candidate), keys].reset_index(drop=True)
            pd.testing.assert_frame_equal(full, other)
        self.assertEqual(set(result.summary["validation_trials"]), {104})
        self.assertEqual(set(result.summary["paired_scored_trials"]), {103})
        self.assertEqual(len(result.screen.query("feature == 'correct_products_collected_percent'")), 6)
        strong_signal = result.screen.loc[result.screen.feature.eq("correct_products_collected_percent")]
        self.assertTrue(strong_signal.p_value.lt(0.05).all())
        coefficient_features = set(result.coefficients["term"])
        self.assertIn("intercept", coefficient_features)
        self.assertIn("correct_products_collected_percent", coefficient_features)

    def test_cli_writes_all_three_conditions_and_rejects_unverified_export(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            trials = root / "trial_features.csv"
            manifest = root / "feature_extraction_manifest.json"
            split = root / "split.csv"
            output = root / "comparison"
            rows = self.trials().rename(columns={"difficulty_stage": "difficulty_level"})
            rows["recorded_difficulty_code"] = rows["difficulty_level"].map({0: 0, 1: 2, 2: 6, 3: 10})
            rows.to_csv(trials, index=False)
            self.split().to_csv(split, index=False)
            source = {
                "schema_version": "fe04_all_grabs_v1",
                "trial_count": 384,
                "search_and_reach_grab_scope": "all_valid_product_grabs",
                "reach_requires_target_focus": True,
                "trial_features_sha256": "wrong hash",
            }
            manifest.write_text(json.dumps(source))
            args = ["--trial-features", str(trials), "--feature-manifest", str(manifest),
                    "--participant-split", str(split), "--output-dir", str(output)]
            with self.assertRaisesRegex(ValueError, "SHA-256"):
                comparison.main(args)
            self.assertFalse(output.exists())

            source["trial_features_sha256"] = sha256(trials.read_bytes()).hexdigest()
            manifest.write_text(json.dumps(source))
            comparison.main(args)
            with self.assertRaises(FileExistsError):
                comparison.main(args)
            results = pd.read_csv(output / "comparison_summary.csv")
            self.assertEqual(len(results), 9)
            self.assertEqual(set(results.condition), {"Visual", "Auditory", "Cognitive"})
            self.assertEqual(results.groupby("condition").selected.sum().tolist(), [1, 1, 1])
            predictions = pd.read_csv(output / "development_predictions.csv")
            self.assertEqual(len(predictions), 3 * 3 * 104)
            self.assertNotIn("S31", set(predictions.participant_id))
            provenance = json.loads((output / "comparison_manifest.json").read_text())
            self.assertEqual(provenance["evaluation"], "development_cross_validation")
            self.assertEqual(provenance["source_schema"], "fe04_all_grabs_v1")

    def test_covariates_are_not_selected_from_participant_metadata(self):
        source = self.trials().assign(age=1, tmt_b_seconds=40)
        prepared = comparison.prepare_development(source, self.split())
        self.assertNotIn("age", prepared.columns)
        self.assertNotIn("tmt_b_seconds", prepared.columns)


if __name__ == "__main__":
    unittest.main()
