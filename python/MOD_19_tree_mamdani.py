#!/usr/bin/env python3
"""MOD-19 (stage 1): development-only Auditory tree rule extraction.

This experimental stage verifies a frozen MOD-14 teacher and extracts crisp
regression-tree rules. No fuzzy membership or Mamdani inference is implemented.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeRegressor, plot_tree

import MOD_14_direct_linear_regression_evaluation as MOD14


MODEL_ID = "auditory"


class Mod19Error(ValueError):
    """Invalid frozen-data contract or provisional tree configuration."""


@dataclass(frozen=True)
class TreeCondition:
    feature: str
    operator: str
    threshold: float


@dataclass(frozen=True)
class RawTreeRule:
    leaf_id: int
    sample_count: int
    prediction: float
    conditions: tuple[TreeCondition, ...]


def prepare_development_teacher(
    *,
    modeling_data: pd.DataFrame,
    holdout_split: pd.DataFrame,
    regression_coefficients: pd.DataFrame,
    training_predictions: pd.DataFrame,
) -> pd.DataFrame:
    """Verify MOD-14 outputs; return only eligible Auditory development rows.

    Reserved-test records are present in modeling_data only to validate the
    canonical 26/6 split. No test features or predictions train the surrogate.
    """
    development, _, development_ids, _ = MOD14.prepare_split_data(
        modeling_data=modeling_data,
        holdout_split=holdout_split,
    )
    frame = MOD14.prepare_model_frame(development, model_id=MODEL_ID)
    if set(frame[MOD14.PARTICIPANT_COLUMN]) != set(development_ids):
        raise Mod19Error("Auditory eligible rows must cover all 26 development participants")

    predictors = MOD14.build_model_registry()[MODEL_ID].predictors
    required = {"model", "predictor", "coefficient"}
    if not required.issubset(regression_coefficients.columns):
        raise Mod19Error("MOD-14 coefficient table lacks required columns")
    source = regression_coefficients.loc[
        regression_coefficients["model"].eq(MODEL_ID)
    ].copy()
    expected_names = ("Intercept", *predictors)
    if len(source) != len(expected_names) or set(source["predictor"]) != set(expected_names):
        raise Mod19Error("Auditory coefficient names/count differ from MOD-14 registry")
    if source["predictor"].duplicated().any():
        raise Mod19Error("Duplicate Auditory regression coefficient")
    coefficients = pd.Series(
        pd.to_numeric(source["coefficient"], errors="coerce").to_numpy(dtype=float),
        index=source["predictor"].tolist(),
    ).reindex(expected_names)
    if not np.isfinite(coefficients.to_numpy(dtype=float)).all():
        raise Mod19Error("Non-finite Auditory regression coefficients")
    original = MOD14.LinearModelFit(
        model_id=MODEL_ID, predictors=predictors, coefficients=coefficients
    ).predict(frame).to_numpy(dtype=float)

    required_predictions = {"model", "participant_id", "predicted_difficulty_stage"}
    if not required_predictions.issubset(training_predictions.columns):
        raise Mod19Error("MOD-14 training prediction table lacks required columns")
    recorded = training_predictions.loc[
        training_predictions["model"].eq(MODEL_ID)
    ].reset_index(drop=True)
    if len(recorded) != len(frame):
        raise Mod19Error("Auditory development prediction row count mismatch")
    if recorded["participant_id"].astype(str).tolist() != frame["participant_id"].tolist():
        raise Mod19Error("MOD-14 Auditory development prediction order/IDs mismatch")
    recorded_values = pd.to_numeric(
        recorded["predicted_difficulty_stage"], errors="coerce"
    ).to_numpy(dtype=float)
    if not np.isfinite(recorded_values).all() or not np.allclose(
        original, recorded_values, rtol=0, atol=1e-8
    ):
        raise Mod19Error("Frozen MOD-14 predictions failed 1e-8 reconstruction check")

    result = frame.copy()
    result["frozen_mlr_prediction"] = original
    return result


def fit_development_tree(
    frame: pd.DataFrame, *, max_depth: int, min_samples_leaf: int
) -> DecisionTreeRegressor:
    """Fit a provisional teacher surrogate; settings have not been selected."""
    if isinstance(max_depth, bool) or not isinstance(max_depth, int) or max_depth < 1:
        raise Mod19Error("max_depth must be a positive integer")
    if (isinstance(min_samples_leaf, bool) or not isinstance(min_samples_leaf, int)
            or min_samples_leaf < 1):
        raise Mod19Error("min_samples_leaf must be a positive integer")
    predictors = MOD14.AUDITORY_PREDICTORS
    if not set((*predictors, "frozen_mlr_prediction")).issubset(frame.columns):
        raise Mod19Error("Auditory feature/MLR columns missing")
    X = frame.loc[:, predictors].to_numpy(dtype=float)
    y = frame["frozen_mlr_prediction"].to_numpy(dtype=float)
    if len(frame) < 2 or not np.isfinite(X).all() or not np.isfinite(y).all():
        raise Mod19Error("Tree data must be complete, finite and contain at least two rows")
    tree = DecisionTreeRegressor(
        max_depth=max_depth, min_samples_leaf=min_samples_leaf, random_state=42
    )
    return tree.fit(X, y)


def extract_raw_rules(tree: DecisionTreeRegressor) -> tuple[RawTreeRule, ...]:
    """Extract every root-to-leaf path with its original crisp inequalities."""
    if not hasattr(tree, "tree_") or tree.n_features_in_ != len(MOD14.AUDITORY_PREDICTORS):
        raise Mod19Error("Expected a fitted Auditory regression tree")
    model = tree.tree_
    predictors = MOD14.AUDITORY_PREDICTORS
    extracted: list[RawTreeRule] = []

    def visit(node_id: int, path: tuple[TreeCondition, ...]) -> None:
        left = int(model.children_left[node_id])
        right = int(model.children_right[node_id])
        if left == right:
            extracted.append(RawTreeRule(
                leaf_id=node_id,
                sample_count=int(model.n_node_samples[node_id]),
                prediction=float(model.value[node_id][0][0]),
                conditions=path,
            ))
            return
        feature = predictors[int(model.feature[node_id])]
        threshold = float(model.threshold[node_id])
        visit(left, (*path, TreeCondition(feature, "<=", threshold)))
        visit(right, (*path, TreeCondition(feature, ">", threshold)))

    visit(0, ())
    if len(extracted) != tree.get_n_leaves():
        raise Mod19Error("Extracted leaf count does not match fitted tree")
    return tuple(extracted)


def write_development_outputs(
    *, frame: pd.DataFrame, tree: DecisionTreeRegressor,
    rules: tuple[RawTreeRule, ...], output_dir: Path,
) -> None:
    """Write development-only, provisional artefacts without overwriting."""
    output_dir = Path(output_dir)
    if output_dir.exists() and (not output_dir.is_dir() or any(output_dir.iterdir())):
        raise Mod19Error(f"Output directory is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    predictors = MOD14.AUDITORY_PREDICTORS
    tree_predictions = tree.predict(frame.loc[:, predictors].to_numpy(dtype=float))
    predictions = pd.DataFrame({
        "participant_id": frame["participant_id"],
        "difficulty_level": frame["difficulty_level"],
        "difficulty_stage": frame["difficulty_stage"],
        "frozen_mlr_prediction": frame["frozen_mlr_prediction"],
        "provisional_tree_prediction": tree_predictions,
        "absolute_tree_fidelity_error": np.abs(
            tree_predictions - frame["frozen_mlr_prediction"].to_numpy(dtype=float)
        ),
    })
    predictions.to_csv(output_dir / "development_predictions.csv", index=False)
    rule_rows = []
    for rule in rules:
        rule_rows.append({
            "leaf_id": rule.leaf_id,
            "development_row_count": rule.sample_count,
            "tree_output": rule.prediction,
            "conditions_json": json.dumps([asdict(c) for c in rule.conditions]),
            "readable_rule": " AND ".join(
                f"{c.feature} {c.operator} {c.threshold:.12g}" for c in rule.conditions
            ) or "TRUE",
        })
    pd.DataFrame(rule_rows).to_csv(output_dir / "numerical_rules.csv", index=False)

    fig, ax = plt.subplots(figsize=(13, 7))
    plot_tree(
        tree, feature_names=list(predictors), filled=False, rounded=True,
        precision=3, fontsize=7, ax=ax,
    )
    fig.tight_layout()
    fig.savefig(output_dir / "decision_tree.png", dpi=180)
    plt.close(fig)
    manifest = {
        "module": "MOD-19", "stage": "1: crisp tree extraction only",
        "research_status": "provisional, development-only; NOT Mamdani fitted",
        "model_id": MODEL_ID,
        "teacher": "frozen MOD-14 Auditory MLR predictions",
        "target_coding": {"D0": 0, "D2": 1, "D6": 2, "D10": 3},
        "predictors": list(predictors),
        "development_participants": int(frame["participant_id"].nunique()),
        "development_rows": len(frame),
        "test_participants_used": 0,
        "tree_parameters": tree.get_params(),
        "raw_rule_count": len(rules),
        "limitations": [
            "Only crisp rules; fuzzy memberships and Mamdani are not implemented.",
            "Hyperparameters supplied explicitly; grouped selection pending.",
            "Development fitting outputs are not cross-validated or test results.",
        ],
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--modeling-data", type=Path, required=True)
    parser.add_argument("--holdout-split", type=Path, required=True)
    parser.add_argument("--mod14-output-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-depth", type=int, required=True)
    parser.add_argument("--min-samples-leaf", type=int, required=True)
    args = parser.parse_args()
    frame = prepare_development_teacher(
        modeling_data=pd.read_csv(args.modeling_data),
        holdout_split=pd.read_csv(args.holdout_split),
        regression_coefficients=pd.read_csv(args.mod14_output_dir / "regression_coefficients.csv"),
        training_predictions=pd.read_csv(args.mod14_output_dir / "training_predictions.csv"),
    )
    tree = fit_development_tree(
        frame, max_depth=args.max_depth, min_samples_leaf=args.min_samples_leaf
    )
    write_development_outputs(
        frame=frame, tree=tree, rules=extract_raw_rules(tree),
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
