"""MOD-17: compact additive Sugeno fuzzy explanations of four frozen MLRs.

Three linguistic sets per feature (Low/Medium/High), calibrated on eligible
development observations only. Fuzzy inference uses no SHAP data or library.
This module describes existing fitted predictions; it never fits a model.
"""
from __future__ import annotations

import argparse
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import MOD_13_difficulty_prediction as MOD13
import MOD_14_direct_linear_regression_evaluation as MOD14

TOLERANCE = 1e-8
CATEGORIES = ("Low", "Medium", "High")
PALETTE = ("#2563eb", "#7c3aed", "#0f766e")
DISPLAY = {
    "mental_demand_score_0_to_10": "Mental demand",
    "median_time_between_qualifying_grabs_seconds": "Time between qualifying grabs",
    "list_recheck_count": "List recheck count",
    "total_list_recheck_duration_seconds": "Total list recheck duration",
    "median_time_to_target_seconds": "Time to locate target",
    "median_irrelevant_focus_duration_seconds": "Irrelevant focus duration",
    "median_head_turning_degrees": "Head turning",
    "median_reach_duration_seconds": "Reach duration",
    "median_reach_path_ratio": "Reach path ratio",
    "performance_change_from_d0_percentage_points": "Relative performance change",
}


class Mod17Error(ValueError):
    """Invalid model, calibration, source prediction or output contract."""


@dataclass(frozen=True)
class FuzzyVariable:
    feature: str
    low: float
    medium: float
    high: float
    development_mean: float
    coefficient: float

    def anchors(self) -> tuple[float, float, float]:
        return (self.low, self.medium, self.high)

    def consequent(self, category_index: int) -> float:
        """Frozen coefficient-centered output at one linguistic anchor."""
        return self.coefficient * (self.anchors()[category_index] - self.development_mean)


def calibrate(feature: str, development: pd.Series, coefficient: float) -> FuzzyVariable:
    """Use only eligible development values; never tune anchors on test data."""
    values = np.asarray(development, dtype=float)
    if values.size == 0 or not np.isfinite(values).all() or not math.isfinite(coefficient):
        raise Mod17Error(f"{feature}: empty/non-finite calibration or coefficient")
    low, medium, high = float(values.min()), float(np.median(values)), float(values.max())
    if not low <= medium <= high:
        raise Mod17Error(f"{feature}: invalid development anchors")
    return FuzzyVariable(feature, low, medium, high, float(values.mean()), float(coefficient))


def memberships(x: float, variable: FuzzyVariable) -> tuple[np.ndarray, float, bool]:
    """Three partition-of-unity sets and an explicit out-of-range residual.

    Values beyond development bounds are clamped ONLY for membership lookup.
    The omitted linear continuation is returned as coefficient * (x - clamped).
    Adding it preserves exact regression fidelity without introducing a fourth
    category or modifying frozen membership anchors. A degenerate category
    (tied anchors) is explicitly inactive, rather than dividing by zero.
    """
    if not math.isfinite(x):
        raise Mod17Error(f"{variable.feature}: non-finite observation")
    lo, mid, hi = variable.anchors()
    clamped = min(max(float(x), lo), hi)
    outside = bool(x < lo or x > hi)
    correction = variable.coefficient * (float(x) - clamped)
    if lo == hi:
        weights = np.array([0., 1., 0.])
    elif mid == lo:
        upper = (clamped - lo) / (hi - lo)
        weights = np.array([0., 1. - upper, upper])
    elif mid == hi:
        upper = (clamped - lo) / (hi - lo)
        weights = np.array([1. - upper, upper, 0.])
    elif clamped <= mid:
        upper = (clamped - lo) / (mid - lo)
        weights = np.array([1. - upper, upper, 0.])
    else:
        upper = (clamped - mid) / (hi - mid)
        weights = np.array([0., 1. - upper, upper])
    if not np.isclose(weights.sum(), 1.0, atol=1e-12, rtol=0):
        raise Mod17Error(f"{variable.feature}: membership partition failed")
    return weights, correction, outside


def evaluate_feature(x: float, variable: FuzzyVariable) -> dict:
    weights, correction, outside = memberships(float(x), variable)
    consequents = np.array([variable.consequent(k) for k in range(3)])
    adjustment = float(weights @ consequents + correction)
    exact = variable.coefficient * (float(x) - variable.development_mean)
    if not math.isclose(adjustment, exact, abs_tol=TOLERANCE, rel_tol=0):
        raise Mod17Error(f"{variable.feature}: fuzzy interpolation mismatch")
    return {
        "value": float(x),
        "low_membership": float(weights[0]),
        "medium_membership": float(weights[1]),
        "high_membership": float(weights[2]),
        "low_consequent": float(consequents[0]),
        "medium_consequent": float(consequents[1]),
        "high_consequent": float(consequents[2]),
        "fuzzy_adjustment": adjustment,
        "out_of_anchor_range": outside,
        "linear_continuation": float(correction),
    }


def _verify_original(frame: pd.DataFrame, original: pd.DataFrame, model_id: str,
                     predicted: np.ndarray, split: str) -> float:
    subset = original.loc[original["model"].eq(model_id)].reset_index(drop=True)
    if len(subset) != len(frame):
        raise Mod17Error(f"{model_id}/{split}: original prediction row count mismatch")
    if not np.array_equal(subset["participant_id"].astype(str), frame["participant_id"].astype(str)):
        raise Mod17Error(f"{model_id}/{split}: original prediction participant/order mismatch")
    if not np.array_equal(subset["observed_difficulty_stage"].to_numpy(dtype=float),
                          frame["difficulty_stage"].to_numpy(dtype=float)):
        raise Mod17Error(f"{model_id}/{split}: original observed stage mismatch")
    mismatch = np.abs(predicted - subset["predicted_difficulty_stage"].to_numpy(dtype=float))
    if not np.isfinite(mismatch).all() or mismatch.max(initial=0.) > TOLERANCE:
        raise Mod17Error(f"{model_id}/{split}: original frozen predictions do not match")
    return float(mismatch.max(initial=0.))


def explain_frame(model_id: str, frame: pd.DataFrame, model: MOD13.LinearModelFit,
                  variables: dict[str, FuzzyVariable], reference: float,
                  split: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compute trial summaries and one feature/category activation record per feature."""
    predictors = tuple(model.predictors)
    predicted = model.predict(frame).to_numpy(dtype=float)
    summaries, activations = [], []
    for position, (_, observation) in enumerate(frame.iterrows()):
        adjustments, outside_features = [], []
        local = []
        for feature in predictors:
            detail = evaluate_feature(float(observation[feature]), variables[feature])
            adjustments.append(detail["fuzzy_adjustment"])
            if detail["out_of_anchor_range"]:
                outside_features.append(feature)
            active = [(cat, detail[f"{cat.lower()}_membership"])
                      for cat in CATEGORIES if detail[f"{cat.lower()}_membership"] > 1e-12]
            matched = ", ".join(f"{cat} {degree:.2f}" for cat, degree in active)
            local.append((feature, detail["fuzzy_adjustment"], matched))
            activations.append({
                "model": model_id, "split": split, "model_row_index": position,
                "participant_id": str(observation["participant_id"]),
                "difficulty_level": int(observation["difficulty_level"]),
                "difficulty_stage": int(observation["difficulty_stage"]),
                "feature": feature, "display_name": DISPLAY.get(feature, feature),
                **detail,
            })
        reconstructed = reference + math.fsum(adjustments)
        if not math.isclose(reconstructed, float(predicted[position]),
                            abs_tol=TOLERANCE, rel_tol=0):
            raise Mod17Error(f"{model_id}/{split}/row{position}: reconstruction failed")
        prominent = sorted(local, key=lambda r: -abs(r[1]))
        descriptions = [
            f"{DISPLAY.get(feature, feature)} ({matched}) "
            f"{'raises' if adjustment >= 0 else 'lowers'} the fitted prediction "
            f"by {abs(adjustment):.3f} stage points"
            for feature, adjustment, matched in prominent[:3]
        ]
        explanation = ("The frozen model predicts the assigned trial's encoded stage, not "
                       "a recommended future stage. " + "; ".join(descriptions) + ".")
        summaries.append({
            "model": model_id, "split": split, "model_row_index": position,
            "participant_id": str(observation["participant_id"]),
            "condition": str(observation["condition_name"]),
            "difficulty_level": int(observation["difficulty_level"]),
            "difficulty_stage": int(observation["difficulty_stage"]),
            "reference_prediction": reference,
            "predicted_difficulty_stage": float(predicted[position]),
            "fuzzy_reconstruction": reconstructed,
            "reconstruction_error": reconstructed - float(predicted[position]),
            "out_of_anchor_range_count": len(outside_features),
            "out_of_anchor_range_features": ";".join(outside_features),
            "explanation": explanation,
        })
    return pd.DataFrame(summaries), pd.DataFrame(activations)


def _short_name(feature: str) -> str:
    return DISPLAY.get(feature, feature)


def _safe_participant_name(value: str) -> str:
    if not re.fullmatch(r"[A-Za-z0-9_-]+", value):
        raise Mod17Error("Unsafe or ambiguous participant ID for plot filename")
    return f"participant_{value}.png"


def plot_memberships(model_id: str, variables: dict[str, FuzzyVariable], output: Path) -> None:
    n = len(variables)
    fig, axes = plt.subplots(math.ceil(n / 2), 2, figsize=(15, 3.25 * math.ceil(n / 2)),
                             squeeze=False)
    for ax, var in zip(axes.ravel(), variables.values()):
        lo, _, hi = var.anchors()
        span = max(hi - lo, 1.)
        values = np.linspace(lo - .10 * span, hi + .10 * span, 160)
        degrees = np.array([memberships(float(x), var)[0] for x in values])
        for k, cat in enumerate(CATEGORIES):
            ax.plot(values, degrees[:, k], label=cat, color=PALETTE[k], lw=2)
        for val in sorted(set(var.anchors())):
            ax.axvline(val, color="#94a3b8", ls=":", lw=0.8)
        ax.set_title(_short_name(var.feature), fontsize=11)
        ax.set_ylabel("Membership")
        ax.set_xlabel("Original measured value")
        ax.set_ylim(-0.03, 1.06)
        ax.grid(alpha=.12)
    for ax in axes.ravel()[n:]:
        ax.axis("off")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False)
    fig.suptitle(f"{MOD14.MODEL_DISPLAY_NAMES[model_id]} — development-calibrated categories",
                 y=1.005, fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, .97))
    fig.savefig(output, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_participant_activations(model_id: str, trial: pd.DataFrame,
                                 activations: pd.DataFrame, predictors: tuple[str, ...],
                                 output: Path) -> None:
    fig, axes = plt.subplots(len(trial), 1,
                             figsize=(9.5, max(3.2 * len(trial), 6)),
                             squeeze=False)
    for ax, (_, row) in zip(axes[:, 0], trial.iterrows()):
        part = activations.loc[
            (activations["participant_id"] == row["participant_id"])
            & (activations["model_row_index"] == row["model_row_index"])
        ].set_index("feature").loc[list(predictors)]
        matrix = part[["low_membership", "medium_membership",
                       "high_membership"]].to_numpy(dtype=float)
        ax.imshow(matrix, vmin=0, vmax=1, aspect="auto", cmap="Blues")
        ax.set_yticks(range(len(predictors)), [_short_name(p) for p in predictors], fontsize=9)
        ax.set_xticks(range(3), CATEGORIES)
        for i in range(len(predictors)):
            for k in range(3):
                ax.text(k, i, f"{matrix[i, k]:.2f}", ha="center", va="center",
                        fontsize=8, color="white" if matrix[i, k] > .55 else "#0f172a")
        ax.set_title(f"D{int(row['difficulty_level'])} | MLR {row['predicted_difficulty_stage']:.3f}",
                     fontsize=11)
    fig.suptitle(f"{MOD14.MODEL_DISPLAY_NAMES[model_id]} — {trial['participant_id'].iloc[0]} | rule activation",
                 fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, .97), h_pad=2.2)
    fig.savefig(output, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_participant_explanations(model_id: str, trial: pd.DataFrame,
                                  activations: pd.DataFrame, predictors: tuple[str, ...],
                                  output: Path) -> None:
    fig, axes = plt.subplots(len(trial), 1,
                             figsize=(14, len(trial) * (2.7 + .34 * len(predictors))),
                             squeeze=False)
    for ax, (_, row) in zip(axes[:, 0], trial.iterrows()):
        ax.axis("off")
        subset = activations.loc[
            (activations["participant_id"] == row["participant_id"])
            & (activations["model_row_index"] == row["model_row_index"])
        ].set_index("feature").loc[list(predictors)]
        body = []
        for _, rec in subset.iterrows():
            active = ", ".join(
                f"{c} {rec[c.lower() + '_membership']:.2f}"
                for c in CATEGORIES if rec[c.lower() + "_membership"] > 1e-12
            )
            label = _short_name(rec["feature"]) + (" *" if rec["out_of_anchor_range"] else "")
            body.append([label, f"{rec['value']:.3f}", active,
                         f"{rec['fuzzy_adjustment']:+.3f}"])
        table = ax.table(cellText=body,
                         colLabels=["Measurement", "Value", "Active rules", "Stage adjustment"],
                         colWidths=[.36, .12, .33, .17],
                         loc="center", cellLoc="left", bbox=[.01, .16, .98, .73])
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        for (r, c), cell in table.get_celld().items():
            cell.set_linewidth(.4)
            cell.set_edgecolor("#cbd5e1")
            if r == 0:
                cell.set_facecolor("#e2e8f0")
                cell.set_text_props(weight="bold")
        ax.set_title(
            f"D{int(row['difficulty_level'])}  |  Reference {row['reference_prediction']:.3f}"
            f" + rules = {row['fuzzy_reconstruction']:.3f}"
            f"  |  Frozen MLR {row['predicted_difficulty_stage']:.3f}"
            f"  |  Assigned {int(row['difficulty_stage'])}",
            loc="left", fontsize=11,
        )
        ax.text(.01, .08, row["explanation"], transform=ax.transAxes, fontsize=8,
                va="center", wrap=True)
        if row["out_of_anchor_range_count"]:
            ax.text(.01, .015,
                    "* Outside development anchor bounds: explicit linear continuation applied",
                    transform=ax.transAxes, fontsize=8, color="#9a3412")
    fig.suptitle(f"{MOD14.MODEL_DISPLAY_NAMES[model_id]} — {trial['participant_id'].iloc[0]} | fuzzy rule explanation",
                 fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, .98), h_pad=2)
    fig.savefig(output, dpi=150, bbox_inches="tight")
    plt.close(fig)


def participant_balanced_usage(activations: pd.DataFrame,
                               predictors: tuple[str, ...]) -> np.ndarray:
    """Mean per participant first; repeated trials are not independent people."""
    means = (activations.groupby(["participant_id", "feature"], sort=False)
             [["low_membership", "medium_membership", "high_membership"]]
             .mean().groupby(level="feature").mean())
    return means.loc[list(predictors)].to_numpy(dtype=float)


def plot_usage(model_id: str, development: pd.DataFrame, test: pd.DataFrame,
               predictors: tuple[str, ...], output: Path) -> None:
    matrices = [participant_balanced_usage(development, predictors),
                participant_balanced_usage(test, predictors)]
    fig, axes = plt.subplots(1, 2, figsize=(13, max(4.5, .53 * len(predictors) + 1.5)),
                             sharey=True)
    for ax, matrix, label in zip(axes, matrices,
                                 ("Development (26 participants)", "Independent test (6 participants)")):
        ax.imshow(matrix, vmin=0, vmax=1, aspect="auto", cmap="Blues")
        ax.set_title(label)
        ax.set_xticks(range(3), CATEGORIES)
        ax.set_yticks(range(len(predictors)), [_short_name(p) for p in predictors], fontsize=9)
        for i in range(len(predictors)):
            for j in range(3):
                ax.text(j, i, f"{matrix[i,j]:.2f}", ha="center", va="center",
                        color="white" if matrix[i,j] > .55 else "#0f172a", fontsize=9)
    fig.suptitle(f"{MOD14.MODEL_DISPLAY_NAMES[model_id]} — participant-balanced rule usage",
                 fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, .96))
    fig.savefig(output, dpi=150, bbox_inches="tight")
    plt.close(fig)


def run_analysis(*, modeling_data: pd.DataFrame, holdout_split: pd.DataFrame,
                 coefficient_table: pd.DataFrame,
                 development_predictions: pd.DataFrame, test_predictions: pd.DataFrame,
                 output_dir: Path) -> pd.DataFrame:
    """Calibrate and verify in memory before writing any outputs."""
    output_dir = Path(output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise Mod17Error(f"Refusing to overwrite nonempty directory: {output_dir}")
    development_data, test_data, dev_ids, test_ids = MOD14.prepare_split_data(
        modeling_data=modeling_data, holdout_split=holdout_split)
    models = MOD13.load_frozen_models(coefficient_table)
    registry = MOD14.build_model_registry()
    if tuple(models) != tuple(registry):
        raise Mod17Error("Frozen model registry differs from MOD-14")

    data = {}
    verification, calibration, rules = [], [], []
    for model_id, spec in registry.items():
        model = models[model_id]
        if tuple(model.predictors) != tuple(spec.predictors):
            raise Mod17Error(f"{model_id}: predictor registry mismatch")
        dev = MOD14.prepare_model_frame(development_data, model_id=model_id)
        test = MOD14.prepare_model_frame(test_data, model_id=model_id)
        if not set(dev["participant_id"]).issubset(dev_ids) or not set(test["participant_id"]).issubset(test_ids):
            raise Mod17Error(f"{model_id}: participant boundary violation")
        if not set(test["difficulty_stage"]).issubset(spec.allowed_stages):
            raise Mod17Error(f"{model_id}: ineligible test stage")
        variables = {}
        for feature in spec.predictors:
            var = calibrate(feature, dev[feature], model.coefficients[feature])
            variables[feature] = var
            calibration.append({
                "model": model_id, "feature": feature, "display_name": _short_name(feature),
                "low_anchor": var.low, "medium_anchor": var.medium,
                "high_anchor": var.high, "development_mean": var.development_mean,
                "frozen_coefficient": var.coefficient,
                "tied_anchors": len(set(var.anchors())) != 3,
                "calibration_split": "development",
            })
            for index, category in enumerate(CATEGORIES):
                rules.append({"model": model_id, "rule_id": f"{model_id}:{feature}:{category.lower()}",
                              "feature": feature, "category": category,
                              "anchor": var.anchors()[index],
                              "consequent_stage_adjustment": var.consequent(index)})
        reference = float(model.coefficients["Intercept"] + math.fsum(
            variables[f].coefficient * variables[f].development_mean
            for f in spec.predictors))
        dev_summary, dev_activation = explain_frame(
            model_id, dev, model, variables, reference, "development")
        test_summary, test_activation = explain_frame(
            model_id, test, model, variables, reference, "test")
        for split, frame, summary, original in (
            ("development", dev, dev_summary, development_predictions),
            ("test", test, test_summary, test_predictions),
        ):
            mismatch = _verify_original(frame, original, model_id,
                                        summary["predicted_difficulty_stage"].to_numpy(), split)
            max_error = float(summary["reconstruction_error"].abs().max())
            verification.append({
                "model": model_id, "split": split,
                "participant_count": int(frame["participant_id"].nunique()),
                "trial_count": len(frame), "reference_prediction": reference,
                "out_of_anchor_trial_count": int((summary["out_of_anchor_range_count"] > 0).sum()),
                "max_absolute_fuzzy_reconstruction_error": max_error,
                "max_absolute_frozen_prediction_mismatch": mismatch,
                "passes": max_error <= TOLERANCE and mismatch <= TOLERANCE,
            })
        data[model_id] = (spec, variables, dev_summary, dev_activation,
                          test_summary, test_activation)

    if not all(row["passes"] for row in verification):
        raise Mod17Error("Numeric verification failed")
    output_dir.mkdir(parents=True, exist_ok=True)
    for model_id, (spec, variables, ds, da, ts, ta) in data.items():
        directory = output_dir / model_id
        directory.mkdir()
        for filename, table in (
            ("explanations_development.csv", ds), ("explanations_test.csv", ts),
            ("activations_development.csv", da), ("activations_test.csv", ta)
        ):
            table.to_csv(directory / filename, index=False, float_format="%.17g")
        plot_memberships(model_id, variables, directory / "membership_functions.png")
        plot_usage(model_id, da, ta, spec.predictors, directory / "rule_usage_development_vs_test.png")
        (directory / "rule_activations_test").mkdir()
        (directory / "fuzzy_explanations_test").mkdir()
        for participant, trials in ts.groupby("participant_id", sort=True):
            filename = _safe_participant_name(str(participant))
            plot_participant_activations(
                model_id, trials, ta, spec.predictors,
                directory / "rule_activations_test" / filename)
            plot_participant_explanations(
                model_id, trials, ta, spec.predictors,
                directory / "fuzzy_explanations_test" / filename)
    pd.DataFrame(calibration).to_csv(output_dir / "fuzzy_calibration.csv", index=False,
                                      float_format="%.17g")
    pd.DataFrame(rules).to_csv(output_dir / "fuzzy_rules.csv", index=False,
                               float_format="%.17g")
    checks = pd.DataFrame(verification)
    checks.to_csv(output_dir / "verification.csv", index=False, float_format="%.17g")
    (output_dir / "manifest.json").write_text(json.dumps({
        "module": "MOD-17", "method": "compact_additive_zero_order_sugeno",
        "categories": list(CATEGORIES),
        "calibration": "min/median/max per original feature and model, eligible development observations only",
        "out_of_anchor_policy": "clamped membership plus explicit coefficient-linear continuation",
        "reference": "frozen MLR evaluated at eligible development feature means",
        "test_used_for_calibration": False, "test_used_for_model_selection": False,
        "frozen_model_refitted": False, "shap_dependency": False,
        "plots": ["membership functions", "test participant rule activation",
                  "test participant fuzzy explanation", "development vs test usage heatmap"],
        "note": "Rules describe frozen assigned-stage regression; no causal or clinical recommendation.",
        "tolerance_absolute": TOLERANCE,
    }, indent=2) + "\n", encoding="utf-8")
    return checks


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--modeling-data", required=True, type=Path)
    parser.add_argument("--holdout-split", required=True, type=Path)
    parser.add_argument("--mod14-output-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args(argv)
    for path in (args.modeling_data, args.holdout_split,
                 args.mod14_output_dir / "regression_coefficients.csv",
                 args.mod14_output_dir / "training_predictions.csv",
                 args.mod14_output_dir / "test_predictions.csv"):
        if not path.is_file():
            raise Mod17Error(f"Missing source file: {path}")
    checks = run_analysis(
        modeling_data=pd.read_csv(args.modeling_data),
        holdout_split=pd.read_csv(args.holdout_split),
        coefficient_table=pd.read_csv(args.mod14_output_dir / "regression_coefficients.csv"),
        development_predictions=pd.read_csv(args.mod14_output_dir / "training_predictions.csv"),
        test_predictions=pd.read_csv(args.mod14_output_dir / "test_predictions.csv"),
        output_dir=args.output_dir,
    )
    print(checks.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
