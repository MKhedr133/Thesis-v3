"""Synthetic checks for the development-only full/screened/add-back comparison."""

from hashlib import sha256
import json
import sys
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

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

    def test_boundary_with_invalid_p_value_rescreens_all_features_by_participant(self):
        rows = self.trials().query("condition_name == 'Visual'").iloc[:80].copy()
        features = list(comparison.TRIAL_FEATURES)
        mixed_p = pd.Series(0.99, index=features)
        mixed_p["self_reported_mental_demand"] = np.nan
        fit = SimpleNamespace(
            converged=True, pvalues=mixed_p, cov_re=pd.DataFrame([[9e-11]]), scale=0.89,
        )
        # Stub the optimizer-dependent boundary result, but run real clustered OLS.
        with patch.object(comparison.MOD02, "fit_mixedlm_with_fallback",
                          return_value=(fit, "powell", ("boundary warning",), ())):
            pvalues, audit = comparison._screen(rows, "Visual", "2")

        design = comparison.sm.add_constant(rows[features])
        reference = comparison.sm.OLS(rows["difficulty_stage"], design).fit()
        clustered = reference.get_robustcov_results(
            cov_type="cluster", groups=pd.factorize(rows["participant_id"])[0],
            use_t=True, use_correction=True, df_correction=True,
        )
        self.assertEqual(list(pvalues), features)
        np.testing.assert_allclose(list(pvalues.values()), clustered.pvalues[1:])
        self.assertFalse(np.allclose(clustered.pvalues[1:], reference.pvalues.iloc[1:]))
        self.assertEqual({row["screening_method"] for row in audit}, {"ols_participant_clustered"})
        self.assertEqual({row["diagnostic"] for row in audit}, {"boundary warning"})
        self.assertEqual({row["random_intercept_variance"] for row in audit}, {9e-11})
        self.assertEqual(
            [row["feature"] for row in audit if not row["mixed_model_p_value_finite"]],
            ["self_reported_mental_demand"],
        )

    def test_boundary_with_finite_p_values_keeps_the_mixed_screen(self):
        rows = self.trials().query("condition_name == 'Visual'").iloc[:80]
        mixed_p = pd.Series(0.01, index=comparison.TRIAL_FEATURES)
        fit = SimpleNamespace(
            converged=True, pvalues=mixed_p, cov_re=pd.DataFrame([[0.0]]), scale=0.89,
        )
        with patch.object(comparison.MOD02, "fit_mixedlm_with_fallback",
                          return_value=(fit, "powell", (), ())):
            pvalues, audit = comparison._screen(rows, "Visual", "2")
        self.assertEqual(pvalues, mixed_p.to_dict())
        self.assertEqual({row["screening_method"] for row in audit}, {"mixedlm"})

    def test_other_mixed_screen_failures_do_not_trigger_fallback(self):
        rows = self.trials().query("condition_name == 'Visual'").iloc[:80]
        for converged, variance, message in (
            (False, 0.0, "did not converge"),
            (True, 0.01, "invalid mixed-model p-values"),
            (True, -0.01, "invalid mixed-model p-values"),
        ):
            with self.subTest(converged=converged, variance=variance):
                fit = SimpleNamespace(
                    converged=converged, pvalues=pd.Series(np.nan, index=comparison.TRIAL_FEATURES),
                    cov_re=pd.DataFrame([[variance]]), scale=0.89,
                )
                with patch.object(comparison.MOD02, "fit_mixedlm_with_fallback",
                                  return_value=(fit, "powell", (), ())):
                    with self.assertRaisesRegex(ValueError, message):
                        comparison._screen(rows, "Visual", "2")

    def test_invalid_clustered_p_values_still_stop_the_screen(self):
        rows = self.trials().query("condition_name == 'Visual'").iloc[:80].copy()
        rows["difficulty_stage"] = 0
        fit = SimpleNamespace(
            converged=True, pvalues=pd.Series(np.nan, index=comparison.TRIAL_FEATURES),
            cov_re=pd.DataFrame([[0.0]]), scale=0.89,
        )
        with patch.object(comparison.MOD02, "fit_mixedlm_with_fallback",
                          return_value=(fit, "powell", (), ())):
            with self.assertRaisesRegex(ValueError, "invalid participant-clustered OLS p-values"):
                comparison._screen(rows, "Visual", "2")

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
            rows.loc[0, "median_hand_travel_to_straight_distance_ratio"] = np.nan
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
            self.assertEqual(len(predictions), 3 * (103 + 104 + 104))
            self.assertNotIn("S31", set(predictions.participant_id))
            provenance = json.loads((output / "comparison_manifest.json").read_text())
            self.assertEqual(provenance["evaluation"], "development_cross_validation")
            self.assertEqual(provenance["source_schema"], "fe04_all_grabs_v1")
            self.assertEqual(provenance["screening_boundary_icc_max_inclusive"], 1e-8)

            self.assertEqual((output / "comparison_mae_change.pdf").read_bytes()[:5], b"%PDF-")
            table = (output / "comparison_table.tex").read_text(encoding="utf-8")
            self.assertEqual(table.count(r"\textbf{"), 3)
            self.assertEqual(table.count("103/104"), 3)
            self.assertEqual(table.count("104/104"), 6)
            for result in results.itertuples():
                self.assertIn(
                    f"{result.participant_balanced_mae:.3f} & {result.mean_predictor_count:.1f}",
                    table,
                )

    def test_mae_change_figure_uses_paired_deltas_and_marks_full_selection(self):
        scores = pd.DataFrame([
            ("Visual", "full", 0.00, False, 10.0),
            ("Visual", "screened", 0.05, True, 3.2),
            ("Visual", "add_back", -0.02, False, 5.0),
            ("Auditory", "full", 0.00, True, 10.0),
            ("Auditory", "screened", 0.08, False, 3.4),
            ("Auditory", "add_back", 0.08, False, 4.8),
            ("Cognitive", "full", 0.00, False, 10.0),
            ("Cognitive", "screened", 0.06, False, 2.0),
            ("Cognitive", "add_back", 0.01, True, 3.6),
        ], columns=["condition", "candidate", "mae_change_vs_full", "selected", "mean_predictor_count"])
        figure = comparison.mae_change_figure(scores.sample(frac=1, random_state=1))
        try:
            axis = figure.axes[0]
            screened, add_back = axis.collections[:2]
            np.testing.assert_allclose(screened.get_offsets()[:, 0], [0.05, 0.08, 0.06])
            np.testing.assert_allclose(add_back.get_offsets()[:, 0], [-0.02, 0.08, 0.01])
            self.assertNotEqual(screened.get_offsets()[1, 1], add_back.get_offsets()[1, 1])
            self.assertEqual([line.get_xdata() for line in axis.lines], [[0, 0], [0.05, 0.05]])
            self.assertEqual(
                [label.get_text() for label in axis.texts],
                ["3.2", "3.4", "2.0", "5.0", "4.8", "3.6", "10.0"],
            )
            np.testing.assert_allclose(axis.collections[2].get_offsets(), [[0, 1]])
            self.assertFalse(np.array_equal(screened.get_facecolors()[0], screened.get_facecolors()[1]))
            labels = " ".join(" ".join(t.get_text().split()) for t in axis.get_legend().get_texts())
            self.assertIn("keeping those with $p < 0.05$", labels)
            self.assertIn("from the pair chosen in advance for each condition", labels)
            figure.canvas.draw()
            legend = axis.get_legend().get_window_extent(figure.canvas.get_renderer())
            self.assertTrue(figure.bbox.contains(legend.x1, legend.y1))
            for label in axis.texts:
                self.assertLess(label.get_window_extent().x1, figure.bbox.x1)
        finally:
            plt.close(figure)

    def test_covariates_are_not_selected_from_participant_metadata(self):
        source = self.trials().assign(age=1, tmt_b_seconds=40)
        prepared = comparison.prepare_development(source, self.split())
        self.assertNotIn("age", prepared.columns)
        self.assertNotIn("tmt_b_seconds", prepared.columns)


if __name__ == "__main__":
    unittest.main()
