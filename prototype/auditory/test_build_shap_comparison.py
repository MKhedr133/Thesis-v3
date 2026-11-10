"""Synthetic checks for development-only SHAP comparison export (no thesis data)."""
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0,str(Path(__file__).resolve().parent))
from build_abcd_cases import (FEATURES, STAGE_TEXT, ExportError,
                              build_shap_comparison)


class ShapComparisonTests(unittest.TestCase):
    def setUp(self):
        self.coefficients = {"Intercept": 1.0}
        self.coefficients.update({
            name: coefficient for name, coefficient
            in zip(FEATURES, (0.1, 0.05, 0.2, -0.01))
        })
        self.rows = []
        for participant in range(26):
            for index, stage in enumerate((0, 2, 6, 10)):
                x = [
                    2.0 + 0.04 * participant + index,
                    3.0 + 0.01 * participant + 0.2 * index,
                    1.0 + 0.005 * participant + 0.5 * index,
                    45 + 0.3 * participant + 4 * index,
                ]
                row = dict(zip(FEATURES, x))
                row.update(participant_id="Development%02d" % participant,
                           difficulty_level=stage, difficulty_stage=index)
                self.rows.append(row)
        self.frame = pd.DataFrame(self.rows)
        self.legacy = []
        for stage in ("0", "2", "6", "10"):
            group = self.frame.loc[self.frame.difficulty_level == int(stage)]
            means = group[list(FEATURES)].mean()
            reference = self.coefficients["Intercept"] + sum(
                self.coefficients[key] * float(means[key]) for key in FEATURES)
            self.legacy.extend([
                {"DifficultyStage": STAGE_TEXT[stage],
                 "Regression": {"ReferencePrediction": reference}}
                for _ in range(2)
            ])

    def test_export_uses_same_reference_for_all_cross_stage_dots(self):
        result = build_shap_comparison(self.frame, self.coefficients, self.legacy)
        self.assertEqual(result["developmentParticipants"], 26)
        self.assertEqual(result["developmentObservations"], 104)
        self.assertEqual(set(result["byReferenceStage"]), {"0","2","6","10"})
        example = result["byReferenceStage"]["6"]
        self.assertEqual(len(example["observations"]), 104)
        other = next(p for p in example["observations"] if p["stage"] == "0")
        self.assertTrue(np.allclose(
            other["contributions"],
            (np.array(other["values"]) - example["meanFeatures"]) *
            [self.coefficients[key] for key in FEATURES],
            atol=1e-12, rtol=0
        ))
        self.assertNotEqual(
            other["contributions"],
            result["byReferenceStage"]["0"]["observations"][0]["contributions"]
        )

    def test_export_has_no_development_identifiers(self):
        result = build_shap_comparison(self.frame, self.coefficients, self.legacy)
        for stage in result["byReferenceStage"].values():
            self.assertTrue(all(
                set(point) == {"stage", "values", "contributions"}
                for point in stage["observations"]
            ))

    def test_reference_must_equal_original_waterfall(self):
        self.legacy[0]["Regression"]["ReferencePrediction"] += 0.1
        with self.assertRaisesRegex(ExportError, "MOD-16 waterfall"):
            build_shap_comparison(self.frame, self.coefficients, self.legacy)

    def test_reject_missing_development_trial(self):
        with self.assertRaisesRegex(ExportError, "104 trials"):
            build_shap_comparison(self.frame.iloc[:-1], self.coefficients, self.legacy)

    def test_reject_nonfinite_feature(self):
        frame = self.frame.copy()
        frame.loc[0, FEATURES[0]] = np.nan
        with self.assertRaisesRegex(ExportError, "Nonfinite"):
            build_shap_comparison(frame, self.coefficients, self.legacy)


if __name__ == "__main__":
    unittest.main()
