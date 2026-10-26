"""MOD-16 — independent/interventional SHAP for frozen MOD-14 regressions.

The four MOD-14 equations and the 26/6 participant split are immutable inputs.
SHAP backgrounds contain only complete eligible development observations.
No model fitting, coefficient tuning, prediction clipping or rounding occurs.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

import MOD_13_difficulty_prediction as MOD13
import MOD_14_direct_linear_regression_evaluation as MOD14


RECONSTRUCTION_ATOL = 1e-8
METHOD = "linear_shap_independent_interventional"
DISPLAY_NAMES = {
    "mental_demand_score_0_to_10": "Mental demand (0–10)",
    "median_time_between_qualifying_grabs_seconds": "Time between grabs (s)",
    "total_list_recheck_duration_seconds": "List recheck duration (s)",
    "median_time_to_target_seconds": "Time to target (s)",
    "median_irrelevant_focus_duration_seconds": "Irrelevant focus duration (s)",
    "median_head_turning_degrees": "Head turning (degrees)",
    "median_reach_duration_seconds": "Reach duration (s)",
    "list_recheck_count": "List recheck count",
    "median_reach_path_ratio": "Reach path ratio",
    "performance_change_from_d0_percentage_points": "Performance change from D0 (pp)",
}


class Mod16Error(ValueError):
    """Raised when a frozen-model or attribution contract is violated."""


@dataclass(frozen=True)
class ModelExplanation:
    model_id: str
    reference_prediction: float
    background_means: pd.Series
    development: pd.DataFrame
    test: pd.DataFrame


def explain_linear_frame(
    *,
    model_id: str,
    model: MOD13.LinearModelFit,
    background: pd.DataFrame,
    observations: pd.DataFrame,
    split_name: str,
) -> tuple[pd.DataFrame, float, pd.Series]:
    """Explain observations using all eligible development rows as background.

    SHAP Independent masker retains every background row (no 100-row
    default subsampling). Assert equality to exact coefficient-centering.
    """
    predictors = tuple(model.predictors)
    if not predictors or len(set(predictors)) != len(predictors):
        raise Mod16Error(f"{model_id}: invalid ordered predictors")
    if background.empty or observations.empty:
        raise Mod16Error(f"{model_id}: empty background/observations")
    for name, frame in (("background", background), ("observations", observations)):
        missing = set(predictors) - set(frame.columns)
        if missing:
            raise Mod16Error(f"{model_id}: {name} missing predictors {sorted(missing)}")
        if not np.isfinite(frame.loc[:, predictors].to_numpy(dtype=float)).all():
            raise Mod16Error(f"{model_id}: {name} has non-finite predictors")
    terms = ("Intercept", *predictors)
    if set(model.coefficients.index) != set(terms):
        raise Mod16Error(f"{model_id}: coefficient terms do not match registry")
    beta = model.coefficients.loc[list(predictors)].to_numpy(dtype=float)
    intercept = float(model.coefficients.loc["Intercept"])
    values = observations.loc[:, predictors].to_numpy(dtype=float)
    background_values = background.loc[:, predictors].to_numpy(dtype=float)
    means = pd.Series(background_values.mean(axis=0), index=predictors, name="development_mean")
    expected = float(intercept + np.dot(means.to_numpy(), beta))

    masker = shap.maskers.Independent(
        background_values,
        max_samples=len(background_values),
    )
    explainer = shap.LinearExplainer((beta, intercept), masker=masker)
    shap_result = explainer(values)
    contributions = np.asarray(shap_result.values, dtype=float)
    base_values = np.broadcast_to(
        np.asarray(shap_result.base_values, dtype=float), (len(values),)
    )
    analytic = (values - means.to_numpy()) * beta
    if contributions.shape != analytic.shape or not np.allclose(
        contributions, analytic, atol=RECONSTRUCTION_ATOL, rtol=0.0
    ):
        raise Mod16Error(f"{model_id}: SHAP disagrees with exact linear attributions")
    if not np.allclose(base_values, expected, atol=RECONSTRUCTION_ATOL, rtol=0.0):
        raise Mod16Error(f"{model_id}: SHAP reference mismatch")

    predicted = np.asarray(model.predict(observations), dtype=float)
    reconstructed = base_values + contributions.sum(axis=1)
    if not np.allclose(
        reconstructed, predicted, atol=RECONSTRUCTION_ATOL, rtol=0.0
    ):
        raise Mod16Error(f"{model_id}: attributions do not reconstruct predictions")

    required = ("participant_id", "condition_name", "difficulty_level", "difficulty_stage")
    missing = set(required) - set(observations.columns)
    if missing:
        raise Mod16Error(f"{model_id}: missing observation identities {sorted(missing)}")
    rows = pd.DataFrame({
        "model": model_id,
        "split": split_name,
        "model_row_index": np.arange(len(observations)),
        "participant_id": observations["participant_id"].astype(str).to_numpy(),
        "condition": observations["condition_name"].to_numpy(),
        "difficulty_level": observations["difficulty_level"].to_numpy(),
        "difficulty_stage": observations["difficulty_stage"].to_numpy(),
        "reference_prediction": base_values,
        "predicted_difficulty_stage": predicted,
        "reconstructed_prediction": reconstructed,
        "reconstruction_error": reconstructed - predicted,
    })
    for i, predictor in enumerate(predictors):
        rows[f"value__{predictor}"] = values[:, i]
        rows[f"shap__{predictor}"] = contributions[:, i]
    return rows, expected, means


def verify_saved_predictions(
    attribution: pd.DataFrame, original: pd.DataFrame, *, model_id: str, split_name: str
) -> float:
    """Require exact row alignment with MOD-14's original prediction export."""
    selected = original.loc[original["model"].astype(str).eq(model_id)].reset_index(drop=True)
    if len(selected) != len(attribution):
        raise Mod16Error(f"{model_id}/{split_name}: prediction row count mismatch")
    if not np.array_equal(
        selected["participant_id"].astype(str).to_numpy(),
        attribution["participant_id"].astype(str).to_numpy(),
    ):
        raise Mod16Error(f"{model_id}/{split_name}: prediction participant order mismatch")
    errors = (
        selected["predicted_difficulty_stage"].to_numpy(dtype=float)
        - attribution["predicted_difficulty_stage"].to_numpy(dtype=float)
    )
    if not np.isfinite(errors).all() or np.max(np.abs(errors)) > RECONSTRUCTION_ATOL:
        raise Mod16Error(f"{model_id}/{split_name}: frozen MOD-14 predictions differ")
    return float(np.max(np.abs(errors)))


def _importance(frame: pd.DataFrame, predictors: tuple[str, ...]) -> np.ndarray:
    return np.array([
        float(frame[f"shap__{name}"].abs().mean()) for name in predictors
    ])


def _save_global_plot(model_id: str, result: ModelExplanation, path: Path, predictors: tuple[str, ...]) -> None:
    dev, test = _importance(result.development, predictors), _importance(result.test, predictors)
    order = np.argsort(-dev)
    labels = [DISPLAY_NAMES.get(predictors[i], predictors[i]) for i in order]
    y = np.arange(len(order))
    fig, ax = plt.subplots(figsize=(10, max(4.5, len(order) * 0.58 + 1.6)))
    ax.barh(y - 0.18, dev[order], height=0.35, label="Development (26 participants)", color="#0f766e")
    ax.barh(y + 0.18, test[order], height=0.35, label="Independent test (6 participants)", color="#b45309")
    ax.set_yticks(y, labels)
    ax.invert_yaxis()
    ax.set_xlabel("Mean absolute SHAP contribution (predicted-stage points)")
    ax.set_title(f"{MOD14.MODEL_DISPLAY_NAMES[model_id]} — global SHAP contributions")
    ax.legend(loc="best", fontsize=9)
    ax.grid(axis="x", alpha=0.18)
    ax.set_axisbelow(True)
    fig.text(0.02, 0.012, "Model-specific development reference; test summary is descriptive (six independent participants).", fontsize=8)
    fig.tight_layout(rect=(0, 0.045, 1, 1))
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _save_distribution_plot(model_id: str, result: ModelExplanation, path: Path, predictors: tuple[str, ...]) -> None:
    """Development-only signed contribution distributions; repeated participant trials are not independent."""
    frame = result.development
    order = np.argsort(-_importance(frame, predictors))
    fig, ax = plt.subplots(figsize=(10, max(4.5, len(order) * 0.59 + 1.6)))
    rng = np.random.default_rng(1600)
    for row, idx in enumerate(order):
        vals = frame[f"shap__{predictors[idx]}"].to_numpy(dtype=float)
        y = row + rng.uniform(-0.20, 0.20, len(vals))
        ax.scatter(vals, y, color="#0f766e", alpha=0.5, s=18, edgecolors="none")
    ax.axvline(0, color="black", linewidth=0.9, alpha=0.5)
    ax.set_yticks(range(len(order)), [DISPLAY_NAMES.get(predictors[i], predictors[i]) for i in order])
    ax.invert_yaxis()
    ax.set_xlabel("Signed SHAP contribution (predicted-stage points)")
    ax.set_title(f"{MOD14.MODEL_DISPLAY_NAMES[model_id]} — development contribution distribution")
    ax.grid(axis="x", alpha=0.15)
    fig.text(0.02, 0.012, "Each dot is one eligible development trial; participants may contribute multiple trials.", fontsize=8)
    fig.tight_layout(rect=(0, 0.045, 1, 1))
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _median_prediction_row(frame: pd.DataFrame) -> pd.Series:
    """Predefined exemplar rule: median predicted value, ties by identifiers."""
    chosen = frame.sort_values(
        ["predicted_difficulty_stage", "participant_id", "difficulty_stage", "model_row_index"],
        kind="mergesort",
    )
    return chosen.iloc[(len(chosen) - 1) // 2]


def _save_local_plot(model_id: str, result: ModelExplanation, path: Path, predictors: tuple[str, ...]) -> None:
    examples = (_median_prediction_row(result.development), _median_prediction_row(result.test))
    data = np.array([[float(row[f"shap__{p}"]) for p in predictors] for row in examples])
    common_max = max(0.05, float(np.max(np.abs(data))) * 1.20)
    fig, axs = plt.subplots(1, 2, figsize=(15, max(5.2, len(predictors) * 0.60 + 2.3)), sharex=True)
    for ax, row, split_label, contributions in zip(
        axs, examples, ("Development example", "Unseen test example"), data
    ):
        order = np.argsort(-np.abs(contributions))
        labels = [DISPLAY_NAMES.get(predictors[i], predictors[i]) for i in order]
        values = contributions[order]
        ax.barh(range(len(order)), values, color=np.where(values >= 0, "#0f766e", "#b45309"))
        ax.set_yticks(range(len(order)), labels, fontsize=9)
        ax.invert_yaxis()
        ax.set_xlim(-common_max, common_max)
        ax.axvline(0, color="black", alpha=0.55, linewidth=0.8)
        ax.grid(axis="x", alpha=0.15)
        ax.set_axisbelow(True)
        ax.set_title(
            f"{split_label}: participant {row['participant_id']}, D{int(row['difficulty_level'])}\n"
            f"Reference {float(row['reference_prediction']):.3f} + contributions "
            f"{np.sum(contributions):+.3f} = prediction {float(row['predicted_difficulty_stage']):.3f}",
            fontsize=10,
        )
        ax.set_xlabel("Signed contribution (predicted-stage points)")
    fig.suptitle(f"{MOD14.MODEL_DISPLAY_NAMES[model_id]} — individual SHAP explanations", fontsize=14)
    fig.text(0.02, 0.012, "Both cases use the same development reference. Positive contributions raise, negative contributions lower the fitted prediction.", fontsize=8)
    fig.tight_layout(rect=(0, 0.045, 1, 0.96))
    fig.savefig(path, dpi=180)
    plt.close(fig)


def run_analysis(
    *,
    modeling_data: pd.DataFrame,
    holdout_split: pd.DataFrame,
    coefficient_table: pd.DataFrame,
    development_predictions: pd.DataFrame,
    test_predictions: pd.DataFrame,
    output_dir: Path,
) -> pd.DataFrame:
    """Explain both splits without reading held-out rows into the background."""
    output_dir = Path(output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise Mod16Error(f"Refusing to overwrite existing output directory: {output_dir}")
    development_data, test_data, dev_ids, test_ids = MOD14.prepare_split_data(
        modeling_data=modeling_data, holdout_split=holdout_split
    )
    models = MOD13.load_frozen_models(coefficient_table)
    registry = MOD14.build_model_registry()
    if tuple(models) != tuple(registry):
        raise Mod16Error("MOD-13 and MOD-14 model registries disagree")
    # Build and verify everything in memory before writing any results.
    results: dict[str, ModelExplanation] = {}
    verification = []
    references = []
    for model_id, spec in registry.items():
        model = models[model_id]
        if tuple(model.predictors) != tuple(spec.predictors):
            raise Mod16Error(f"{model_id}: final predictor registry mismatch")
        background = MOD14.prepare_model_frame(development_data, model_id=model_id)
        unseen = MOD14.prepare_model_frame(test_data, model_id=model_id)
        if set(background["participant_id"].astype(str)) - set(dev_ids):
            raise Mod16Error(f"{model_id}: test participant entered background")
        if set(unseen["participant_id"].astype(str)) - set(test_ids):
            raise Mod16Error(f"{model_id}: non-test participant entered test explanations")
        if not set(background["difficulty_stage"]).issubset(spec.allowed_stages):
            raise Mod16Error(f"{model_id}: ineligible stage entered background")
        dev_attr, base, means = explain_linear_frame(
            model_id=model_id, model=model, background=background,
            observations=background, split_name="development",
        )
        test_attr, test_base, test_means = explain_linear_frame(
            model_id=model_id, model=model, background=background,
            observations=unseen, split_name="test",
        )
        if not np.isclose(base, test_base, atol=RECONSTRUCTION_ATOL, rtol=0.0):
            raise Mod16Error(f"{model_id}: split-specific reference mismatch")
        if not np.allclose(means, test_means, atol=RECONSTRUCTION_ATOL, rtol=0.0):
            raise Mod16Error(f"{model_id}: background altered by test observations")
        for split_name, attr, exported in (
            ("development", dev_attr, development_predictions),
            ("test", test_attr, test_predictions),
        ):
            mismatch = verify_saved_predictions(attr, exported, model_id=model_id, split_name=split_name)
            verification.append({
                "model": model_id, "split": split_name, "method": METHOD,
                "participant_count": int(attr["participant_id"].nunique()),
                "observation_count": len(attr),
                "reference_prediction": base,
                "max_absolute_reconstruction_error": float(attr["reconstruction_error"].abs().max()),
                "max_absolute_mod14_prediction_mismatch": mismatch,
                "passes": True,
            })
        for predictor, mean in means.items():
            references.append({
                "model": model_id, "predictor": predictor, "development_mean": float(mean),
                "reference_prediction": base,
                "development_participants": int(background["participant_id"].nunique()),
                "development_observations": len(background),
                "eligible_stages": ",".join(map(str, spec.allowed_stages)),
                "method": METHOD,
            })
        results[model_id] = ModelExplanation(model_id, base, means, dev_attr, test_attr)

    output_dir.mkdir(parents=True, exist_ok=True)
    for model_id, result in results.items():
        directory = output_dir / model_id
        directory.mkdir()
        result.development.to_csv(directory / "attributions_development.csv", index=False, float_format="%.17g")
        result.test.to_csv(directory / "attributions_test.csv", index=False, float_format="%.17g")
        predictors = tuple(registry[model_id].predictors)
        _save_global_plot(model_id, result, directory / "global_importance.png", predictors)
        _save_distribution_plot(model_id, result, directory / "development_distribution.png", predictors)
        _save_local_plot(model_id, result, directory / "local_examples.png", predictors)
    pd.DataFrame(references).to_csv(output_dir / "development_references.csv", index=False, float_format="%.17g")
    checks = pd.DataFrame(verification)
    checks.to_csv(output_dir / "verification.csv", index=False, float_format="%.17g")
    (output_dir / "manifest.json").write_text(
        json.dumps({
            "method": METHOD,
            "masker": "shap.maskers.Independent with all model-eligible development rows",
            "shap_version": shap.__version__,
            "model_ids": list(results),
            "exemplar_rule": "median continuous prediction per split; stable identifier tie-break",
            "reference_definition": "per-model fitted prediction at eligible development predictor means",
            "test_data_in_background": False,
            "tolerance_absolute": RECONSTRUCTION_ATOL,
            "note": "Attributions describe fixed model predictions; they are not causal or prescribing recommendations.",
        }, indent=2) + "\n", encoding="utf-8"
    )
    return checks


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--modeling-data", type=Path, required=True)
    parser.add_argument("--holdout-split", type=Path, required=True)
    parser.add_argument("--mod14-output-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    original = args.mod14_output_dir
    inputs = {
        "modeling_data": args.modeling_data,
        "holdout_split": args.holdout_split,
        "coefficient_table": original / "regression_coefficients.csv",
        "development_predictions": original / "training_predictions.csv",
        "test_predictions": original / "test_predictions.csv",
    }
    for name, path in inputs.items():
        if not path.is_file():
            parser.error(f"Missing {name}: {path}")
    checks = run_analysis(
        **{name: pd.read_csv(path) for name, path in inputs.items()},
        output_dir=args.output_dir,
    )
    print(checks.to_string(index=False))
    print(f"Outputs: {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
