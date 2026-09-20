"""Synthetic tests for MOD-11 training-only bootstrap comparison."""

from __future__ import annotations

from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import sys
import unittest

import numpy as np
import pandas as pd


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
PYTHON_DIR = REPOSITORY_ROOT / "python"
sys.path.insert(0, str(PYTHON_DIR))

MODULE_PATH = PYTHON_DIR / "MOD_11_training_bootstrap_comparison.py"

SPEC = spec_from_file_location(
    "mod11_training_bootstrap_test",
    MODULE_PATH,
)

if SPEC is None or SPEC.loader is None:
    raise ImportError(MODULE_PATH)

MOD11 = module_from_spec(SPEC)
sys.modules[SPEC.name] = MOD11
SPEC.loader.exec_module(MOD11)


def synthetic_modeling_data() -> pd.DataFrame:
    """Create four repeated rows for each of 32 synthetic participants."""
    rows: list[dict[str, object]] = []

    for group, prefix in (("Young", "Y"), ("Old", "O")):
        for number in range(1, 17):
            participant_id = f"{prefix}{number:02d}"

            for difficulty_level, difficulty_stage in (
                (0, 0),
                (2, 1),
                (6, 2),
                (10, 3),
            ):
                rows.append(
                    {
                        "participant_id": participant_id,
                        "participant_group": group,
                        "difficulty_level": difficulty_level,
                        "difficulty_stage": difficulty_stage,
                        "tmt_b_seconds": float(
                            40
                            + number
                            + (20 if group == "Old" else 0)
                        ),
                    }
                )

    return pd.DataFrame(rows)


def synthetic_holdout_split() -> pd.DataFrame:
    """Create the synthetic equivalent of the locked MOD-09 26/6 split."""
    rows: list[dict[str, str]] = []

    for group, prefix in (("Young", "Y"), ("Old", "O")):
        for number in range(1, 17):
            participant_id = f"{prefix}{number:02d}"

            split = (
                "test"
                if number in {1, 2, 3}
                else "train"
            )

            rows.append(
                {
                    "participant_id": participant_id,
                    "participant_group": group,
                    "split": split,
                }
            )

    return pd.DataFrame(rows)


class ModelContractTests(unittest.TestCase):
    def test_registry_contains_exactly_seven_frozen_models(self):
        registry = MOD11.build_candidate_model_rhs_registry()

        self.assertEqual(
            tuple(registry),
            (
                "M0",
                "MA",
                "MT",
                "MAT",
                "MDA",
                "MDT",
                "MDAT",
            ),
        )

    def test_registry_matches_frozen_formulas(self):
        registry = MOD11.build_candidate_model_rhs_registry()

        age = (
            "C(participant_group, "
            "Treatment(reference='Young'))"
        )

        expected = {
            "M0": "{D}",
            "MA": f"{{D}} + {age}",
            "MT": "{D} + tmt_b_seconds",
            "MAT": f"{{D}} + {age} + tmt_b_seconds",
            "MDA": (
                f"{{D}} + {age} + "
                f"{{D}}:{age}"
            ),
            "MDT": (
                "{D} + tmt_b_seconds + "
                "{D}:tmt_b_seconds"
            ),
            "MDAT": (
                f"{{D}} + {age} + tmt_b_seconds + "
                f"{{D}}:{age} + "
                "{D}:tmt_b_seconds"
            ),
        }

        self.assertEqual(registry, expected)

    def test_tmt_models_use_raw_tmt_b_seconds(self):
        registry = MOD11.build_candidate_model_rhs_registry()

        for model_id in ("MT", "MAT", "MDT", "MDAT"):
            rhs = registry[model_id]

            self.assertIn("tmt_b_seconds", rhs)
            self.assertNotIn("tmt_b_z", rhs)
            self.assertNotIn("standard", rhs.lower())

    def test_random_effects_are_frozen_to_ri_only(self):
        self.assertEqual(
            MOD11.RANDOM_STRUCTURES,
            ("RI",),
        )

    def test_final_bootstrap_target_is_500(self):
        self.assertEqual(
            MOD11.DEFAULT_BOOTSTRAP_REPLICATES,
            500,
        )


class TrainingBoundaryTests(unittest.TestCase):
    def test_prepare_training_data_excludes_all_test_participants(self):
        training_data, training_ids, test_ids = (
            MOD11.prepare_training_data(
                modeling_data=synthetic_modeling_data(),
                holdout_split=synthetic_holdout_split(),
            )
        )

        self.assertEqual(len(training_ids), 26)
        self.assertEqual(len(test_ids), 6)

        self.assertEqual(
            training_data["participant_id"].nunique(),
            26,
        )

        self.assertTrue(
            set(training_data["participant_id"]).isdisjoint(
                test_ids
            )
        )

        self.assertEqual(
            set(training_data["participant_id"]),
            set(training_ids),
        )


class BootstrapSamplingTests(unittest.TestCase):
    def setUp(self):
        (
            self.training_data,
            self.training_ids,
            self.test_ids,
        ) = MOD11.prepare_training_data(
            modeling_data=synthetic_modeling_data(),
            holdout_split=synthetic_holdout_split(),
        )

    def test_participant_draw_has_exactly_26_draws(self):
        sampled = MOD11.draw_participant_ids(
            self.training_ids,
            rng=np.random.default_rng(123),
        )

        self.assertEqual(len(sampled), 26)

        self.assertTrue(
            set(sampled).issubset(
                set(self.training_ids)
            )
        )

    def test_participant_sampling_is_with_replacement(self):
        sampled = MOD11.draw_participant_ids(
            self.training_ids,
            rng=np.random.default_rng(123),
        )

        self.assertLess(
            len(set(sampled)),
            26,
        )

    def test_final_test_ids_can_never_enter_bootstrap_draw(self):
        sampled = MOD11.draw_participant_ids(
            self.training_ids,
            rng=np.random.default_rng(123),
        )

        self.assertTrue(
            set(sampled).isdisjoint(
                self.test_ids
            )
        )

    def test_duplicate_draws_receive_distinct_cluster_ids(self):
        sampled = (
            *self.training_ids[:-1],
            self.training_ids[0],
        )

        bootstrap = MOD11.build_bootstrap_sample(
            self.training_data,
            sampled,
            replicate_index=7,
        )

        self.assertEqual(
            bootstrap[
                MOD11.BOOTSTRAP_CLUSTER_COLUMN
            ].nunique(),
            26,
        )

        duplicated = bootstrap.loc[
            bootstrap["participant_id"].eq(
                self.training_ids[0]
            )
        ]

        self.assertEqual(
            duplicated[
                MOD11.BOOTSTRAP_CLUSTER_COLUMN
            ].nunique(),
            2,
        )

    def test_each_bootstrap_cluster_preserves_all_repeated_rows(self):
        sampled = (
            *self.training_ids[:-1],
            self.training_ids[0],
        )

        bootstrap = MOD11.build_bootstrap_sample(
            self.training_data,
            sampled,
            replicate_index=7,
        )

        rows_per_cluster = (
            bootstrap
            .groupby(MOD11.BOOTSTRAP_CLUSTER_COLUMN)
            .size()
        )

        self.assertTrue(
            rows_per_cluster.eq(4).all()
        )

    def test_oob_ids_are_training_participants_not_sampled(self):
        omitted = self.training_ids[-1]

        sampled = (
            *self.training_ids[:-1],
            self.training_ids[0],
        )

        oob = MOD11.identify_oob_participants(
            self.training_ids,
            sampled,
        )

        self.assertEqual(
            oob,
            (omitted,),
        )

        self.assertTrue(
            set(oob).isdisjoint(self.test_ids)
        )


class BootstrapPlanTests(unittest.TestCase):
    def setUp(self):
        _, self.training_ids, _ = (
            MOD11.prepare_training_data(
                modeling_data=synthetic_modeling_data(),
                holdout_split=synthetic_holdout_split(),
            )
        )

    def test_plan_contains_requested_number_of_replicates(self):
        plan = MOD11.generate_bootstrap_plan(
            self.training_ids,
            n_replicates=5,
            seed=12345,
        )

        self.assertEqual(len(plan), 5)

        for sampled in plan:
            self.assertEqual(len(sampled), 26)

    def test_plan_is_reproducible_for_same_seed(self):
        plan_a = MOD11.generate_bootstrap_plan(
            self.training_ids,
            n_replicates=5,
            seed=12345,
        )

        plan_b = MOD11.generate_bootstrap_plan(
            self.training_ids,
            n_replicates=5,
            seed=12345,
        )

        self.assertEqual(plan_a, plan_b)

    def test_different_seed_changes_plan(self):
        plan_a = MOD11.generate_bootstrap_plan(
            self.training_ids,
            n_replicates=5,
            seed=12345,
        )

        plan_b = MOD11.generate_bootstrap_plan(
            self.training_ids,
            n_replicates=5,
            seed=54321,
        )

        self.assertNotEqual(plan_a, plan_b)

    def test_one_plan_can_be_reused_for_all_models(self):
        plan = MOD11.generate_bootstrap_plan(
            self.training_ids,
            n_replicates=3,
            seed=12345,
        )

        candidate_plan_views = {
            model_id: plan
            for model_id in MOD11.MODEL_IDS
        }

        for model_id in MOD11.MODEL_IDS:
            self.assertIs(
                candidate_plan_views[model_id],
                plan,
            )


if __name__ == "__main__":
    unittest.main()