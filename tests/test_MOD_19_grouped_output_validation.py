"""Synthetic verification for MOD-19 Stage 4B; no participant data included."""

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))

import MOD_14_direct_linear_regression_evaluation as MOD14
import MOD_19_fuzzy_rules as BASE
import MOD_19_mamdani_inference as MAMDANI
import MOD_19_grouped_output_validation as GROUPED


def synthetic_development():
    names = MOD14.AUDITORY_PREDICTORS
    rows = []
    for participant in range(26):
        for stage, level in enumerate((0, 2, 6, 10)):
            rows.append({
                "participant_id": f"synthetic-{participant:02d}",
                "difficulty_stage": stage,
                "difficulty_level": level,
                names[0]: (participant + 2 * stage) % 9,
                names[1]: 3.3 + ((participant * 3 + stage) % 14) * 0.4,
                names[2]: 0.7 + ((participant + 3 * stage) % 11) * 0.16,
                names[3]: 49 + ((participant * 7 + stage * 4) % 27) * 5,
            })
    frame = pd.DataFrame(rows)
    frame["frozen_mlr_prediction"] = (
        0.4 + frame[names[0]] * 0.13
        + frame[names[1]] * 0.07
        + frame[names[2]] * 0.20
        + frame[names[3]] * 0.004
    )
    return frame


class GroupedDevelopmentValidationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.frame = synthetic_development()
        cls.train_sets = []
        original_fit = GROUPED.train_one_fold

        def capture(training):
            cls.train_sets.append(set(training["participant_id"]))
            return original_fit(training)

        with patch.object(GROUPED, "train_one_fold", side_effect=capture):
            (cls.predictions, cls.metrics, cls.folds,
             cls.person_errors) = GROUPED.run_grouped_validation(cls.frame)

    def test_frozen_contract_and_four_stages(self):
        groups = GROUPED.verify_development_frame(self.frame)
        self.assertEqual(len(groups), 26)
        self.assertEqual(set(self.frame["difficulty_level"]), {0, 2, 6, 10})

    def test_lo_po_splits_never_train_on_heldout_participant(self):
        self.assertEqual(len(self.train_sets), 26)
        self.assertEqual(len(self.predictions), 104)
        for fold, train_ids in enumerate(self.train_sets, start=1):
            validation = self.predictions.loc[
                self.predictions["fold"].eq(fold)
            ]
            self.assertEqual(len(train_ids), 25)
            self.assertEqual(validation["participant_id"].nunique(), 1)
            self.assertEqual(len(validation), 4)
            self.assertTrue(
                train_ids.isdisjoint(set(validation["participant_id"]))
            )
            self.assertEqual(set(validation["difficulty_stage"]), {0, 1, 2, 3})

    def test_all_models_report_grouped_metrics(self):
        self.assertEqual(
            set(self.metrics["model"]),
            {"crisp_tree", "output_3_terms", "output_5_terms"},
        )
        self.assertTrue(self.metrics["n_available"].between(0, 104).all())
        self.assertTrue(self.metrics["n_unavailable"].between(0, 104).all())
        self.assertTrue(
            (self.metrics["n_available"]
             + self.metrics["n_unavailable"]).eq(104).all()
        )
        self.assertTrue(self.metrics["coverage_percent"].between(0, 100).all())
        self.assertEqual(len(self.person_errors), 26)

    def test_fold_diagnostics_and_rule_count(self):
        self.assertEqual(len(self.folds), 52)
        self.assertEqual(set(self.folds["output_terms"]), {3, 5})
        self.assertTrue(self.folds["n_training_rows"].eq(100).all())
        self.assertTrue(self.folds["n_validation_rows"].eq(4).all())
        self.assertTrue(self.folds["n_train_participants"].eq(25).all())
        self.assertTrue(self.folds["n_validation_participants"].eq(1).all())
        self.assertTrue(self.folds["n_tree_rules"].between(1, 8).all())
        for _, group in self.folds.groupby("fold"):
            self.assertEqual(len(group), 2)
            self.assertEqual(group["n_tree_rules"].nunique(), 1)
            self.assertEqual(group["train_mlr_min"].nunique(), 1)
            self.assertEqual(group["train_mlr_max"].nunique(), 1)

    def test_rule_outputs_are_not_assigned_from_holdout_targets(self):
        original = GROUPED.train_one_fold
        checked = []

        def inspect(training):
            assert len(training) == 100
            checked.append(
                (training["frozen_mlr_prediction"].min(),
                 training["frozen_mlr_prediction"].max())
            )
            return original(training)

        with patch.object(GROUPED, "train_one_fold", side_effect=inspect):
            seen = []
            original_labels = GROUPED.SENSITIVITY.output_rule_labels

            def inspect_labels(rules, low, high, term_count):
                seen.append((low, high, term_count))
                return original_labels(rules, low, high, term_count)

            with patch.object(
                GROUPED.SENSITIVITY, "output_rule_labels",
                side_effect=inspect_labels,
            ):
                GROUPED.run_grouped_validation(self.frame)
        self.assertEqual(len(checked), 26)
        self.assertEqual(len(seen), 52)
        for fold_index, (low, high) in enumerate(checked):
            first, second = seen[2 * fold_index:2 * fold_index + 2]
            self.assertEqual((first[0], first[1]), (low, high))
            self.assertEqual((second[0], second[1]), (low, high))
            self.assertEqual((first[2], second[2]), (3, 5))

    def test_reject_missing_and_duplicate_participant_observations(self):
        invalid = self.frame.drop(index=[0])
        with self.assertRaises(BASE.ConversionError):
            GROUPED.verify_development_frame(invalid)
        duplicate = self.frame.copy()
        duplicate.loc[0, "difficulty_level"] = 2
        with self.assertRaisesRegex(BASE.ConversionError, "Repeated"):
            GROUPED.verify_development_frame(duplicate)

    def test_reject_wrong_stage_coding(self):
        bad = self.frame.copy()
        bad.loc[0, "difficulty_stage"] = 9
        with self.assertRaisesRegex(BASE.ConversionError, "stage coding"):
            GROUPED.verify_development_frame(bad)

    def test_reject_nonfinite_teacher(self):
        bad = self.frame.copy()
        bad.loc[0, "frozen_mlr_prediction"] = np.nan
        with self.assertRaisesRegex(BASE.ConversionError, "Non-finite"):
            GROUPED.verify_development_frame(bad)

    def test_source_provenance_and_tree_parameters(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            mod14 = root / "mod14"
            stage1 = root / "stage1"
            mod14.mkdir()
            stage1.mkdir()
            modeling_data = root / "modeling.csv"
            split = root / "split.csv"
            files = {
                "modeling_data.csv": modeling_data,
                "participant_holdout_split.csv": split,
                "mod14_regression_coefficients.csv":
                    mod14 / "regression_coefficients.csv",
                "mod14_training_predictions.csv":
                    mod14 / "training_predictions.csv",
                "stage1_manifest.json": stage1 / "manifest.json",
            }
            for name, location in files.items():
                location.write_text(name + "\n", encoding="utf-8")
            manifest = {
                "module": "MOD-19",
                "stage": "4A: output vocabulary sensitivity",
                "model_id": "auditory",
                "development_participants": 26,
                "development_rows": 104,
                "test_participants_used": 0,
                "candidate_output_counts": [3, 5],
                "three_term_reproduces_stage3": True,
                "source_sha256": {
                    name: MAMDANI.file_sha256(path)
                    for name, path in files.items()
                },
            }
            stage1_manifest = {
                "stage": "1: crisp tree extraction only",
                "model_id": "auditory",
                "development_participants": 26,
                "test_participants_used": 0,
                "tree_parameters": GROUPED.TREE_PARAMETERS.copy(),
            }
            arguments = {
                "stage4a_manifest": manifest,
                "stage1_manifest": stage1_manifest,
                "modeling_data": modeling_data,
                "holdout_split": split,
                "mod14_dir": mod14,
                "stage1_dir": stage1,
            }
            GROUPED.check_source_provenance(**arguments)
            stage1_manifest["tree_parameters"]["max_depth"] = 4
            with self.assertRaisesRegex(BASE.ConversionError, "CART settings"):
                GROUPED.check_source_provenance(**arguments)
            stage1_manifest["tree_parameters"]["max_depth"] = 3
            modeling_data.write_text("tampered\n", encoding="utf-8")
            with self.assertRaisesRegex(BASE.ConversionError, "source changed"):
                GROUPED.check_source_provenance(**arguments)

    def test_overwrite_guard_keeps_original_file(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            target = root / "prior"
            target.mkdir()
            (target / "keep.txt").write_text("keep")
            with self.assertRaisesRegex(BASE.ConversionError, "already exists"):
                GROUPED.write_results(
                    target, self.predictions, self.metrics, self.folds,
                    self.person_errors, source_hashes={},
                    stage4a_manifest=root / "missing.json",
                )
            self.assertEqual((target / "keep.txt").read_text(), "keep")


if __name__ == "__main__":
    unittest.main()
