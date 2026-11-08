"""Synthetic MOD-19 stage-1 checks; no real participant records."""

import json
import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))

import MOD_14_direct_linear_regression_evaluation as MOD14
import MOD_19_tree_mamdani as MOD19


class Mod19Stage1Tests(unittest.TestCase):
    @staticmethod
    def example_data():
        splits = []
        rows = []
        names = MOD14.AUDITORY_PREDICTORS
        for p in range(32):
            pid = f"S{p:02d}"
            splits.append({
                "participant_id": pid,
                "participant_group": "Young" if p < 13 or 26 <= p < 29 else "Old",
                "split": "train" if p < 26 else "test",
            })
            for stage, level in enumerate((0, 2, 6, 10)):
                rows.append({
                    "participant_id": pid,
                    "condition_name": "Auditory",
                    "difficulty_level": level,
                    "difficulty_stage": stage,
                    names[0]: (p % 9) + 0.5 * stage,
                    names[1]: 2 + 0.2 * p + 0.3 * stage,
                    names[2]: 1 + 0.1 * p + 0.4 * stage,
                    names[3]: 30 + 2 * p + 6 * stage,
                })
        data = pd.DataFrame(rows)
        split = pd.DataFrame(splits)
        coefficients = pd.Series(
            [0.2, 0.15, 0.08, 0.12, 0.003], index=("Intercept", *names)
        )
        coefficient_table = pd.DataFrame({
            "model": ["auditory"] * len(coefficients),
            "predictor": coefficients.index,
            "coefficient": coefficients.to_numpy(),
        })
        development = data.loc[
            data.participant_id.isin([f"S{i:02d}" for i in range(26)])
        ].copy()
        teacher = MOD14.LinearModelFit("auditory", names, coefficients)
        recorded = teacher.predict(development).to_numpy(dtype=float)
        prediction_table = pd.DataFrame({
            "model": "auditory",
            "participant_id": development.participant_id.to_numpy(),
            "predicted_difficulty_stage": recorded,
        })
        return data, split, coefficient_table, prediction_table

    def make_frame(self):
        modeling, split, coefficients, predictions = self.example_data()
        return MOD19.prepare_development_teacher(
            modeling_data=modeling,
            holdout_split=split,
            regression_coefficients=coefficients,
            training_predictions=predictions,
        )

    def test_teacher_excludes_six_reserved_participants(self):
        frame = self.make_frame()
        self.assertEqual(frame.participant_id.nunique(), 26)
        self.assertEqual(len(frame), 104)
        self.assertFalse(
            set(frame.participant_id) & {f"S{i:02d}" for i in range(26, 32)}
        )

    def test_modified_teacher_prediction_rejected(self):
        modeling, split, coefficients, predictions = self.example_data()
        predictions.loc[0, "predicted_difficulty_stage"] += 0.01
        with self.assertRaisesRegex(MOD19.Mod19Error, "reconstruction"):
            MOD19.prepare_development_teacher(
                modeling_data=modeling,
                holdout_split=split,
                regression_coefficients=coefficients,
                training_predictions=predictions,
            )

    def test_wrong_stage_coding_rejected(self):
        modeling, split, coefficients, predictions = self.example_data()
        modeling.loc[0, "difficulty_stage"] = 3
        with self.assertRaisesRegex(MOD14.Mod14Error, "does not match frozen coding"):
            MOD19.prepare_development_teacher(
                modeling_data=modeling,
                holdout_split=split,
                regression_coefficients=coefficients,
                training_predictions=predictions,
            )

    def test_raw_rules_exactly_reproduce_tree_leaf_predictions(self):
        frame = self.make_frame()
        tree = MOD19.fit_development_tree(
            frame, max_depth=3, min_samples_leaf=4
        )
        rules = MOD19.extract_raw_rules(tree)
        self.assertEqual(len(rules), tree.get_n_leaves())
        self.assertTrue(any(rule.conditions for rule in rules))
        for row in frame.itertuples(index=False):
            values = row._asdict()
            matches = [
                rule for rule in rules
                if all(
                    values[c.feature] <= c.threshold if c.operator == "<="
                    else values[c.feature] > c.threshold
                    for c in rule.conditions
                )
            ]
            self.assertEqual(len(matches), 1)
            predicted = tree.predict([
                [values[name] for name in MOD14.AUDITORY_PREDICTORS]
            ])[0]
            self.assertAlmostEqual(matches[0].prediction, float(predicted))

    def test_invalid_tree_settings_rejected(self):
        frame = self.make_frame()
        with self.assertRaises(MOD19.Mod19Error):
            MOD19.fit_development_tree(
                frame, max_depth=0, min_samples_leaf=4
            )
        with self.assertRaises(MOD19.Mod19Error):
            MOD19.fit_development_tree(
                frame, max_depth=3, min_samples_leaf=0
            )

    def test_output_files_and_overwrite_guard(self):
        frame = self.make_frame()
        tree = MOD19.fit_development_tree(
            frame, max_depth=3, min_samples_leaf=4
        )
        rules = MOD19.extract_raw_rules(tree)
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "out"
            MOD19.write_development_outputs(
                frame=frame, tree=tree, rules=rules, output_dir=target
            )
            self.assertEqual(
                {p.name for p in target.iterdir()},
                {
                    "development_predictions.csv",
                    "numerical_rules.csv",
                    "decision_tree.png",
                    "manifest.json",
                },
            )
            manifest = json.loads((target / "manifest.json").read_text())
            self.assertEqual(manifest["test_participants_used"], 0)
            self.assertEqual(manifest["development_participants"], 26)
            self.assertEqual(
                len(pd.read_csv(target / "numerical_rules.csv")), len(rules)
            )
            with self.assertRaises(MOD19.Mod19Error):
                MOD19.write_development_outputs(
                    frame=frame, tree=tree, rules=rules, output_dir=target
                )


if __name__ == "__main__":
    unittest.main()
