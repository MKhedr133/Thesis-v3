"""Tests for deterministic MixedLM optimizer selection in MOD-02."""

from __future__ import annotations

from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import pandas as pd


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = REPOSITORY_ROOT / "python" / "MOD_02_random_effects_structure.py"


def load_module():
    specification = spec_from_file_location("mod02_optimizer_test", MODULE_PATH)
    if specification is None or specification.loader is None:
        raise ImportError(MODULE_PATH)
    module = module_from_spec(specification)
    sys.modules[specification.name] = module
    specification.loader.exec_module(module)
    return module


MOD02 = load_module()


class _FakeResult:
    def __init__(self, llf: float, converged: bool):
        self.llf = llf
        self.converged = converged


class _FakeModel:
    def __init__(self, results):
        self.results = dict(results)
        self.calls: list[str] = []
        self.reml_calls: list[bool] = []

    def fit(self, *, reml, method, maxiter, disp):
        self.calls.append(method)
        self.reml_calls.append(reml)

        outcome = self.results[method]

        if isinstance(outcome, Exception):
            raise outcome

        return outcome


class OptimizerSelectionTests(unittest.TestCase):
    def _data(self):
        return pd.DataFrame(
            {
                "participant_id": ["P01", "P02"],
                "difficulty_stage": [0.0, 1.0],
                "outcome": [1.0, 2.0],
            }
        )

    def test_runs_all_optimizers_and_selects_highest_log_likelihood(self):
        model = _FakeModel(
            {
                "lbfgs": _FakeResult(llf=-120.0, converged=True),
                "powell": _FakeResult(llf=-100.0, converged=True),
            }
        )

        with patch.object(MOD02.smf, "mixedlm", return_value=model):
            result, optimizer, warnings, errors = MOD02.fit_mixedlm_with_fallback(
                "outcome ~ difficulty_stage",
                self._data(),
                "RI",
                "difficulty_stage",
                methods=("lbfgs", "powell"),
            )

        self.assertEqual(model.calls, ["lbfgs", "powell"])
        self.assertIs(result, model.results["powell"])
        self.assertEqual(optimizer, "powell")
        self.assertEqual(warnings, ())
        self.assertEqual(errors, ())

    def test_nonconverged_fit_cannot_win_on_log_likelihood(self):
        model = _FakeModel(
            {
                "lbfgs": _FakeResult(llf=-120.0, converged=True),
                "powell": _FakeResult(llf=-80.0, converged=False),
            }
        )

        with patch.object(MOD02.smf, "mixedlm", return_value=model):
            result, optimizer, _, _ = MOD02.fit_mixedlm_with_fallback(
                "outcome ~ difficulty_stage",
                self._data(),
                "RI",
                "difficulty_stage",
                methods=("lbfgs", "powell"),
            )

        self.assertEqual(model.calls, ["lbfgs", "powell"])
        self.assertIs(result, model.results["lbfgs"])
        self.assertEqual(optimizer, "lbfgs")

    def test_default_estimation_remains_ml(self):
        model = _FakeModel(
            {
                "lbfgs": _FakeResult(
                    llf=-100.0,
                    converged=True,
                ),
            }
        )

        with patch.object(
            MOD02.smf,
            "mixedlm",
            return_value=model,
        ):
            MOD02.fit_mixedlm_with_fallback(
                "outcome ~ difficulty_stage",
                self._data(),
                "RI",
                "difficulty_stage",
                methods=("lbfgs",),
            )

        self.assertEqual(
            model.reml_calls,
            [False],
        )


    def test_reml_can_be_requested_explicitly(self):
        model = _FakeModel(
            {
                "lbfgs": _FakeResult(
                    llf=-100.0,
                    converged=True,
                ),
            }
        )

        with patch.object(
            MOD02.smf,
            "mixedlm",
            return_value=model,
        ):
            MOD02.fit_mixedlm_with_fallback(
                "outcome ~ difficulty_stage",
                self._data(),
                "RI",
                "difficulty_stage",
                reml=True,
                methods=("lbfgs",),
            )

        self.assertEqual(
            model.reml_calls,
            [True],
        )


if __name__ == "__main__":
    unittest.main()
