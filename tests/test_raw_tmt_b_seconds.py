"""Regression tests for raw TMT-B seconds in participant models."""

from __future__ import annotations

from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import sys
import unittest


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
PYTHON_DIR = REPOSITORY_ROOT / "python"
MOD02_PATH = PYTHON_DIR / "MOD_02_random_effects_structure.py"
MOD04_PATH = PYTHON_DIR / "MOD_04_age_tmt_associations.py"
MOD05_PATH = PYTHON_DIR / "MOD_05_lopo_predictive_comparison.py"


def load_module(name: str, path: Path):
    specification = spec_from_file_location(name, path)
    if specification is None or specification.loader is None:
        raise ImportError(path)
    module = module_from_spec(specification)
    sys.modules[name] = module
    specification.loader.exec_module(module)
    return module


MOD02 = load_module("mod02_raw_tmt_test", MOD02_PATH)
MOD04 = load_module("mod04_raw_tmt_test", MOD04_PATH)


class RawTmtSecondsTests(unittest.TestCase):
    def test_candidate_registry_uses_raw_tmt_seconds(self):
        tmt_models = ("MT", "MAT", "MDT", "MDAT")

        for model_id in tmt_models:
            rhs = MOD04.PROPOSED_MODEL_RHS[model_id]
            self.assertIn("tmt_b_seconds", rhs)
            self.assertNotIn("tmt_z", rhs)

    def test_models_without_tmt_are_unchanged_in_scope(self):
        for model_id in ("M0", "MA", "MDA"):
            self.assertNotIn("tmt_b_seconds", MOD04.PROPOSED_MODEL_RHS[model_id])

    def test_mod02_default_formula_uses_raw_tmt_seconds(self):
        self.assertIn("tmt_b_seconds", MOD02.FIXED_EFFECTS_TEMPLATE)
        self.assertNotIn("tmt_z", MOD02.FIXED_EFFECTS_TEMPLATE)

    def test_no_standardized_tmt_predictor_remains_in_pipeline_sources(self):
        for path in (MOD02_PATH, MOD04_PATH, MOD05_PATH):
            source = path.read_text(encoding="utf-8")
            self.assertNotIn("tmt_z", source, path.name)

    def test_lopo_no_longer_records_training_tmt_standardization(self):
        source = MOD05_PATH.read_text(encoding="utf-8")
        self.assertNotIn("training_tmt_mean", source)
        self.assertNotIn("training_tmt_sd", source)


if __name__ == "__main__":
    unittest.main()
