"""Synthetic MOD-01B modelling-table construction tests."""

from __future__ import annotations

from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import sys
import tempfile
import unittest

import pandas as pd


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = REPOSITORY_ROOT / "python" / "04_build_modeling_data.py"


def load_builder_module():
    if not MODULE_PATH.exists():
        return None
    specification = spec_from_file_location("build_modeling_data", MODULE_PATH)
    if specification is None or specification.loader is None:
        return None
    module = module_from_spec(specification)
    sys.modules[specification.name] = module
    specification.loader.exec_module(module)
    return module


BUILDER = load_builder_module()


BEHAVIOURAL_COLUMNS = (
    "median_time_between_qualifying_grabs_seconds",
    "list_recheck_count",
    "total_list_recheck_duration_seconds",
    "median_time_to_target_seconds",
    "median_irrelevant_focus_duration_seconds",
    "median_head_turning_degrees",
    "median_reach_duration_seconds",
    "median_reach_path_ratio",
)

ERROR_COMPONENT_COLUMNS = (
    "errors_missing",
    "errors_wrong_order",
    "errors_duplicate",
    "errors_not_in_list",
)


def make_trial_features() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    participants = (
        ("P01", "Young", "Visual"),
        ("P02", "Older", "Auditory"),
        ("P03", "Young", "Cognitive"),
    )
    for participant_id, participant_group, condition_name in participants:
        for trial_order, difficulty_level in enumerate((0, 2, 6, 10), start=1):
            components = (1, 0, 0, 0)
            row: dict[str, object] = {
                "participant_id": participant_id,
                "session_id": f"S{trial_order:02d}",
                "source_tracker_csv_filename": (
                    f"tracker_{participant_id}_{condition_name}_{difficulty_level}.csv"
                ),
                "participant_group": participant_group,
                "condition_name": condition_name,
                "difficulty_level": difficulty_level,
                "trial_order": trial_order,
                "language": "EN",
                "performance": 10.0,
                "correct_products_collected_count": 10,
                "performance_percent": 100.0,
                "performance_change_from_d0_percentage_points": (
                    0.0 if difficulty_level == 0 else 5.0
                ),
                "mental_demand_score_0_to_10": float(difficulty_level) / 2.0,
                "error_change_from_d0": None,
                "total_error_count": sum(components),
                "processing_status": "ok",
                "custom_qc_flag": 0,
            }
            row.update(
                dict(
                    zip(
                        BEHAVIOURAL_COLUMNS,
                        (1.0, 0, 0.0, 2.0, 0.5, 15.0, 1.5, 1.2),
                    )
                )
            )
            row.update(dict(zip(ERROR_COMPONENT_COLUMNS, components)))
            rows.append(row)
    return pd.DataFrame(rows)


def make_covariates() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "participant_id": ["P01", "P02", "P03"],
            "age_years": [25.0, 42.0, 31.0],
            "tmt_b_seconds": [68.0, 91.0, 75.0],
        }
    )


class ModelingDataBuildTests(unittest.TestCase):
    def require_builder(self):
        self.assertIsNotNone(
            BUILDER,
            "python/04_build_modeling_data.py must be implemented",
        )
        return BUILDER

    def build(self, trial_features=None, participant_covariates=None):
        builder = self.require_builder()
        return builder.build_modeling_data(
            make_trial_features() if trial_features is None else trial_features,
            make_covariates()
            if participant_covariates is None
            else participant_covariates,
        )

    def test_valid_table_preserves_rows_columns_and_numeric_difficulty(self):
        trial_features = make_trial_features()
        result = self.build(trial_features=trial_features)

        self.assertTrue(result.valid)
        self.assertEqual(len(result.modeling_data), len(trial_features))
        expected_columns = list(trial_features.columns)
        expected_columns.insert(
            expected_columns.index("difficulty_level") + 1,
            "difficulty_stage",
        )
        expected_columns.extend(["age_years", "tmt_b_seconds"])
        self.assertEqual(list(result.modeling_data.columns), expected_columns)
        for column in trial_features.columns:
            self.assertIn(column, result.modeling_data.columns)
            pd.testing.assert_series_equal(
                result.modeling_data[column],
                trial_features[column],
                check_names=True,
            )
        self.assertEqual(
            result.modeling_data["difficulty_level"].tolist(),
            trial_features["difficulty_level"].tolist(),
        )
        self.assertEqual(
            result.modeling_data["difficulty_stage"].tolist(),
            [0, 1, 2, 3] * 3,
        )
        self.assertNotIn("difficulty_label", result.modeling_data.columns)

    def test_covariates_use_many_to_one_merge_without_row_multiplication(self):
        result = self.build()

        self.assertEqual(len(result.modeling_data), 12)
        self.assertEqual(
            result.modeling_data.groupby("participant_id").size().to_dict(),
            {"P01": 4, "P02": 4, "P03": 4},
        )
        self.assertEqual(
            result.modeling_data.groupby("participant_id")["age_years"]
            .first()
            .to_dict(),
            {"P01": 25.0, "P02": 42.0, "P03": 31.0},
        )

    def test_duplicate_covariates_fail_without_multiplying_rows_or_selecting_value(self):
        covariates = pd.concat(
            [
                make_covariates(),
                pd.DataFrame(
                    {
                        "participant_id": ["P01"],
                        "age_years": [99.0],
                        "tmt_b_seconds": [120.0],
                    }
                ),
            ],
            ignore_index=True,
        )
        result = self.build(participant_covariates=covariates)

        self.assertFalse(result.valid)
        self.assertEqual(len(result.modeling_data), 12)
        p01 = result.modeling_data["participant_id"].eq("P01")
        self.assertTrue(result.modeling_data.loc[p01, "age_years"].isna().all())
        self.assertTrue(
            result.modeling_data.loc[p01, "tmt_b_seconds"].isna().all()
        )
        self.assertTrue(
            any("duplicate_participant_covariates" in error for error in result.errors)
        )

    def test_missing_participant_coverage_preserves_rows_and_missing_covariates(self):
        result = self.build(
            participant_covariates=make_covariates().iloc[:2].copy()
        )

        self.assertFalse(result.valid)
        self.assertEqual(len(result.modeling_data), 12)
        p03 = result.modeling_data["participant_id"].eq("P03")
        self.assertTrue(result.modeling_data.loc[p03, "age_years"].isna().all())
        self.assertTrue(
            result.modeling_data.loc[p03, "tmt_b_seconds"].isna().all()
        )

    def test_existing_trial_columns_are_preserved_without_external_qc_input(self):
        trial_features = make_trial_features()
        result = self.build(trial_features=trial_features)

        self.assertIn("processing_status", result.modeling_data.columns)
        self.assertIn("custom_qc_flag", result.modeling_data.columns)
        self.assertNotIn("qc_path", result.modeling_data.columns)

    def test_warnings_and_missing_values_are_preserved(self):
        trial_features = make_trial_features()
        trial_features.loc[0, "median_reach_duration_seconds"] = None
        trial_features.loc[1, "mental_demand_score_0_to_10"] = None
        result = self.build(trial_features=trial_features)

        self.assertTrue(result.valid)
        self.assertTrue(result.warnings)
        self.assertTrue(
            pd.isna(result.modeling_data.loc[0, "median_reach_duration_seconds"])
        )
        self.assertTrue(
            pd.isna(result.modeling_data.loc[1, "mental_demand_score_0_to_10"])
        )
        self.assertEqual(result.modeling_data.loc[0, "custom_qc_flag"], 0)

    def test_result_container_is_frozen(self):
        builder = self.require_builder()

        self.assertTrue(
            builder.ModelingDataBuildResult.__dataclass_params__.frozen
        )

    def test_checks_table_has_stable_schema_and_validation_records(self):
        result = self.build()

        self.assertEqual(
            list(result.modeling_data_checks.columns),
            ["check_id", "check_name", "status", "count", "message"],
        )
        self.assertGreaterEqual(len(result.modeling_data_checks), 1)
        self.assertTrue(
            set(result.modeling_data_checks["status"]).issubset(
                {"pass", "warning", "fail"}
            )
        )
        self.assertIn(
            "required_trial_columns",
            set(result.modeling_data_checks["check_name"]),
        )

    def test_inputs_are_unchanged_and_returned_frames_are_independent(self):
        trial_features = make_trial_features()
        participant_covariates = make_covariates()
        original_trial_features = trial_features.copy(deep=True)
        original_participant_covariates = participant_covariates.copy(deep=True)

        result = self.build(trial_features, participant_covariates)
        result.modeling_data.loc[0, "custom_qc_flag"] = 99
        result.modeling_data_checks.loc[0, "status"] = "changed"

        pd.testing.assert_frame_equal(trial_features, original_trial_features)
        pd.testing.assert_frame_equal(
            participant_covariates,
            original_participant_covariates,
        )

    def test_writer_creates_only_requested_utf8_index_free_outputs(self):
        builder = self.require_builder()
        result = self.build()
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            modeling_path = root / "modeling_data.csv"
            checks_path = root / "modeling_data_checks.csv"

            builder.write_modeling_outputs(result, modeling_path, checks_path)

            self.assertEqual(
                set(path.name for path in root.iterdir()),
                {"modeling_data.csv", "modeling_data_checks.csv"},
            )
            written = pd.read_csv(modeling_path)
            checks = pd.read_csv(checks_path)
            self.assertNotIn("Unnamed: 0", written.columns)
            self.assertNotIn("Unnamed: 0", checks.columns)
            self.assertEqual(len(written), len(result.modeling_data))
            self.assertEqual(list(checks.columns), list(result.modeling_data_checks.columns))


if __name__ == "__main__":
    unittest.main()
