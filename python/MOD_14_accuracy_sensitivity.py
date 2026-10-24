#!/usr/bin/env python3
"""Add threshold-based accuracy plots to frozen MOD-14 MLR predictions.

This is a read-only post-processing analysis: it does NOT fit or modify models.
Usage (from the repository root):
    python python/MOD_14_accuracy_sensitivity.py \
      --mod14-output-dir outputs/MOD_14_direct_linear_regression_evaluation \
      --output-dir outputs/MOD_14_accuracy_sensitivity

Only eligible, original MOD-14 prediction rows are scored. Accuracy at tolerance
`t` is 100 * mean(abs(observed_stage - predicted_stage) < t). No stage rounding,
clipping, missing-row removal, or tuning with independent-test observations.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, PercentFormatter
import numpy as np
import pandas as pd

import MOD_14_direct_linear_regression_evaluation as MOD14

THRESHOLDS: tuple[float, ...] = (0.25, 0.50, 0.75)
MODEL_LABELS: dict[str, str] = {
    "visual": "Visual",
    "auditory": "Auditory",
    "cognitive_primary": "Cognitive\ninitial",
    "cognitive_later": "Cognitive\nlater",
}

# Consistent high-contrast styling, suitable for papers and supervisor slides.
INK = "#17253C"
MUTED = "#596879"
GRID = "#E2E7EE"
DEV_COLOR = "#087F8C"
TEST_COLOR = "#DB7B27"
WHITE = "#FFFFFF"
SERIES_COLORS = ("#087F8C", "#DB7B27", "#7856AB", "#3C6FA6")

REQUIRED_PREDICTION_COLUMNS = (
    "participant_id", "model", "condition", "observed_difficulty_stage",
    "predicted_difficulty_stage", "error", "absolute_error",
)
REQUIRED_PERFORMANCE_COLUMNS = (
    "model", "condition", "development_participant_count",
    "development_row_count", "test_participant_count", "test_row_count",
    "training_mae", "training_r2", "test_mae", "test_r2",
    "training_out_of_range_prediction_count",
    "test_out_of_range_prediction_count",
)


class AccuracyReportError(ValueError):
    """Input records or output paths violate the frozen MOD-14 contract."""


def threshold_token(threshold: float) -> str:
    """Return an unambiguous filename/column token for a threshold."""
    return f"{threshold:.2f}".replace(".", "p")


def exact_tolerance_accuracy(
    observed: np.ndarray | pd.Series,
    predicted: np.ndarray | pd.Series,
    *,
    threshold: float,
) -> tuple[int, int, float]:
    """Return correct count, total count, percentage for |error| < threshold.

    Array positions represent the same trial, with no implicit index join.
    Caller validates record identity; this function rejects non-finite values
    and missing observations instead of reducing the denominator.
    """
    y = np.asarray(observed, dtype=float)
    yhat = np.asarray(predicted, dtype=float)
    if y.ndim != 1 or yhat.ndim != 1 or y.size != yhat.size or not y.size:
        raise AccuracyReportError("Expected nonempty one-dimensional equal-length arrays")
    if not np.isfinite(y).all() or not np.isfinite(yhat).all():
        raise AccuracyReportError("Observed/predicted inputs must be finite")
    if not np.isfinite(threshold) or threshold <= 0:
        raise AccuracyReportError("Threshold must be finite and positive")
    correct = int(np.count_nonzero(np.abs(y - yhat) < threshold))
    return correct, int(y.size), 100.0 * correct / y.size


def _require_columns(frame: pd.DataFrame, columns: tuple[str, ...], label: str) -> None:
    missing = sorted(set(columns) - set(frame.columns))
    if missing:
        raise AccuracyReportError(f"{label}: missing columns {missing}")


def _read_predictions(path: Path, *, split: str) -> pd.DataFrame:
    if not path.is_file():
        raise AccuracyReportError(f"Missing {split} predictions: {path}")
    frame = pd.read_csv(path, dtype={"participant_id": "string"})
    _require_columns(frame, REQUIRED_PREDICTION_COLUMNS, split)
    if frame.empty:
        raise AccuracyReportError(f"{split}: no prediction records")
    if frame["participant_id"].isna().any():
        raise AccuracyReportError(f"{split}: missing participant ID")
    frame["participant_id"] = frame["participant_id"].str.strip()
    if frame["participant_id"].eq("").any():
        raise AccuracyReportError(f"{split}: blank participant ID")
    if frame["model"].isna().any() or frame["condition"].isna().any():
        raise AccuracyReportError(f"{split}: missing model or condition")

    for column in ("observed_difficulty_stage", "predicted_difficulty_stage",
                   "error", "absolute_error"):
        values = pd.to_numeric(frame[column], errors="coerce")
        if not np.isfinite(values.to_numpy(dtype=float)).all():
            raise AccuracyReportError(f"{split}: {column} has invalid/non-finite values")
        frame[column] = values.astype(float)

    registry = MOD14.build_model_registry()  # Single source of stage eligibility.
    if set(frame["model"]) != set(registry):
        raise AccuracyReportError(
            f"{split}: expected exactly models {tuple(registry)}; "
            f"found {tuple(sorted(set(frame['model'])))}"
        )
    if frame.duplicated(["participant_id", "model", "observed_difficulty_stage"]).any():
        raise AccuracyReportError(f"{split}: duplicate participant/model/stage trial key")

    for model_id, spec in registry.items():
        rows = frame.loc[frame["model"].eq(model_id)]
        if rows.empty or not rows["condition"].eq(spec.condition).all():
            raise AccuracyReportError(f"{split}/{model_id}: missing records or incorrect condition")
        if not rows["observed_difficulty_stage"].isin(spec.allowed_stages).all():
            raise AccuracyReportError(
                f"{split}/{model_id}: observed stage outside {spec.allowed_stages}"
            )

    computed_error = (frame["observed_difficulty_stage"]
                      - frame["predicted_difficulty_stage"]).to_numpy(dtype=float)
    if not np.allclose(frame["error"].to_numpy(dtype=float),
                       computed_error, atol=1e-10, rtol=1e-10):
        raise AccuracyReportError(f"{split}: saved signed errors disagree with predictions")
    if not np.allclose(frame["absolute_error"].to_numpy(dtype=float),
                       np.abs(computed_error), atol=1e-10, rtol=1e-10):
        raise AccuracyReportError(f"{split}: saved absolute errors disagree with predictions")
    return frame


def _read_performance(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise AccuracyReportError(f"Missing original performance table: {path}")
    frame = pd.read_csv(path)
    _require_columns(frame, REQUIRED_PERFORMANCE_COLUMNS, "MOD-14 performance")
    if tuple(frame["model"]) != tuple(MOD14.build_model_registry()):
        raise AccuracyReportError("MOD-14 performance: unexpected model order/rows")
    return frame


def _assert_metric(actual: float, saved: float, context: str) -> None:
    if not np.isclose(actual, saved, rtol=1e-9, atol=1e-9, equal_nan=True):
        raise AccuracyReportError(
            f"{context}: source MAE/R² disagrees with saved performance "
            f"(calculated {actual!r}, saved {saved!r})"
        )


def calculate_report(
    training_predictions: pd.DataFrame,
    test_predictions: pd.DataFrame,
    performance: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Validate frozen data and build wide, long, and trial-audit outputs.

    DataFrames must already have passed _read_predictions/_read_performance;
    this also permits validated in-memory synthetic test inputs.
    """
    registry = MOD14.build_model_registry()
    for split, frame, expected_people in (
        ("training", training_predictions, MOD14.EXPECTED_DEVELOPMENT_PARTICIPANTS),
        ("test", test_predictions, MOD14.EXPECTED_TEST_PARTICIPANTS),
    ):
        if frame["participant_id"].nunique() != expected_people:
            raise AccuracyReportError(f"{split}: expected {expected_people} unique participants")
        group_ids = {
            model_id: set(frame.loc[frame["model"].eq(model_id), "participant_id"])
            for model_id in registry
        }
        if any(ids != next(iter(group_ids.values())) for ids in group_ids.values()):
            raise AccuracyReportError(f"{split}: participant membership differs among models")
    dev_ids = set(training_predictions["participant_id"])
    test_ids = set(test_predictions["participant_id"])
    if dev_ids & test_ids:
        raise AccuracyReportError("Development and independent-test participant IDs overlap")

    wide_rows: list[dict[str, object]] = []
    long_rows: list[dict[str, object]] = []
    audit_tables: list[pd.DataFrame] = []
    for model_id, spec in registry.items():
        original = performance.loc[performance["model"].eq(model_id)].iloc[0]
        if original["condition"] != spec.condition:
            raise AccuracyReportError(f"{model_id}: performance condition does not match registry")
        row: dict[str, object] = {
            "model": model_id,
            "condition": spec.condition,
            "development_participant_count": int(original["development_participant_count"]),
            "development_row_count": int(original["development_row_count"]),
            "test_participant_count": int(original["test_participant_count"]),
            "test_row_count": int(original["test_row_count"]),
        }
        for split, predictions, source_prefix, count_prefix in (
            ("training", training_predictions, "training", "development"),
            ("test", test_predictions, "test", "test"),
        ):
            subset = predictions.loc[predictions["model"].eq(model_id)].copy()
            if len(subset) != int(original[f"{count_prefix}_row_count"]):
                raise AccuracyReportError(f"{model_id}/{split}: row count mismatch")
            if subset["participant_id"].nunique() != int(
                original[f"{count_prefix}_participant_count"]
            ):
                raise AccuracyReportError(f"{model_id}/{split}: participant count mismatch")
            observed = subset["observed_difficulty_stage"]
            predicted = subset["predicted_difficulty_stage"]
            mae = MOD14.participant_balanced_mae(
                participant_ids=subset["participant_id"],
                observed=observed, predicted=predicted,
            )
            r2 = MOD14.pooled_r2(observed=observed, predicted=predicted)
            _assert_metric(mae, float(original[f"{source_prefix}_mae"]), f"{model_id}/{split}/MAE")
            _assert_metric(r2, float(original[f"{source_prefix}_r2"]), f"{model_id}/{split}/R²")
            out_of_range = MOD14._count_out_of_range_predictions(predicted)
            if out_of_range != int(original[f"{source_prefix}_out_of_range_prediction_count"]):
                raise AccuracyReportError(f"{model_id}/{split}: out-of-range count mismatch")
            # IMPORTANT: displayed MAE and R² are copied unchanged from MOD-14.
            row[f"{split}_mae"] = float(original[f"{source_prefix}_mae"])
            row[f"{split}_r2"] = float(original[f"{source_prefix}_r2"])
            subset.insert(0, "split", split)
            for threshold in THRESHOLDS:
                correct, n, percent = exact_tolerance_accuracy(
                    observed, predicted, threshold=threshold
                )
                token = threshold_token(threshold)
                row[f"{split}_accuracy_lt_{token}_pct"] = percent
                row[f"{split}_correct_lt_{token}_count"] = correct
                subset[f"correct_lt_{token}"] = (
                    subset["absolute_error"].to_numpy(dtype=float) < threshold
                )
                long_rows.append({
                    "model": model_id, "condition": spec.condition, "split": split,
                    "threshold_stages": threshold, "correct_count": correct,
                    "evaluation_count": n, "participant_count": subset["participant_id"].nunique(),
                    "accuracy_pct": percent,
                })
            audit_tables.append(subset)
        wide_rows.append(row)

    wide = pd.DataFrame(wide_rows)
    long = pd.DataFrame(long_rows)
    audit = pd.concat(audit_tables, ignore_index=True)
    return wide, long, audit


def load_and_calculate(
    *, training_path: Path, test_path: Path, performance_path: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load original saved MOD-14 files and validate all calculations."""
    return calculate_report(
        _read_predictions(training_path, split="training"),
        _read_predictions(test_path, split="test"),
        _read_performance(performance_path),
    )


def _style(ax: plt.Axes) -> None:
    ax.set_facecolor(WHITE)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.grid(axis="y", color=GRID, linewidth=0.9, zorder=0)
    ax.set_axisbelow(True)
    ax.tick_params(axis="both", length=0, labelsize=10, colors=MUTED, pad=9)


def _render_grouped_bars(
    summary: pd.DataFrame,
    *, column_a: str, column_b: str, title: str, subtitle: str,
    ylabel: str, filename: Path, percent: bool = False,
) -> None:
    models = tuple(MOD14.build_model_registry())
    rows = summary.set_index("model").loc[list(models)]
    dev = rows[column_a].to_numpy(dtype=float)
    test = rows[column_b].to_numpy(dtype=float)
    positions = np.arange(len(models))
    fig, ax = plt.subplots(figsize=(11.7, 6.5), facecolor=WHITE)
    fig.subplots_adjust(left=0.105, right=0.96, bottom=0.19, top=0.72)
    fig.text(0.105, 0.91, title, color=INK, fontweight="bold", fontsize=19)
    fig.text(0.105, 0.855, subtitle, color=MUTED, fontsize=10.7)
    fig.text(0.105, 0.811, "MODEL PERFORMANCE  /  MOD-14", color=MUTED,
             fontsize=9, fontweight="bold")
    _style(ax)
    width = 0.33
    b1 = ax.bar(positions - width / 2, dev, width, label="Development · 26 participants",
                color=DEV_COLOR, zorder=3, edgecolor=WHITE, linewidth=0.6)
    b2 = ax.bar(positions + width / 2, test, width, label="Independent test · 6 participants",
                color=TEST_COLOR, zorder=3, edgecolor=WHITE, linewidth=0.6)
    ax.set_xticks(positions, [MODEL_LABELS[m] for m in models],
                  fontweight="medium", color=INK)
    ax.set_ylabel(ylabel, fontsize=11, color=MUTED, labelpad=14)
    maximum = float(max(0.0, np.max(dev), np.max(test)))
    minimum = float(min(0.0, np.min(dev), np.min(test)))
    if percent:
        ax.set_ylim(0, 110)
        ax.set_yticks([0, 20, 40, 60, 80, 100])
        ax.yaxis.set_major_formatter(PercentFormatter(xmax=100, decimals=0))
        labelfmt = "%.1f%%"
        pad = 2.0
    else:
        upper = max(0.1, maximum * 1.25)
        lower = min(0, minimum * 1.25)
        ax.set_ylim(lower, upper)
        labelfmt = "%.3f"
        pad = 0.02 * (upper - lower)
        ax.axhline(0, color=GRID, linewidth=1.1)
    for bars in (b1, b2):
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, h + (pad if h >= 0 else -pad),
                    labelfmt % h, ha="center", va="bottom" if h >= 0 else "top",
                    color=INK, fontweight="semibold", fontsize=10)
    ax.legend(loc="upper left", bbox_to_anchor=(-0.005, 1.11),
              frameon=False, ncol=2, fontsize=10.5, labelcolor=INK)
    count_summary = "  •  ".join(
        f"{MODEL_LABELS[m].replace(chr(10), ' ')} "
        f"{int(rows.loc[m, 'development_row_count'])}/{int(rows.loc[m, 'test_row_count'])}"
        for m in models
    )
    fig.text(0.105, 0.079, "Eligible trials (development/test): " + count_summary,
             color=MUTED, fontsize=9.1)
    fig.text(0.105, 0.043,
             "Cognitive initial covers D0–D10; Cognitive later covers D2–D10.",
             color=MUTED, fontsize=9.1)
    filename.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(filename, dpi=270, facecolor=WHITE)
    fig.savefig(filename.with_suffix(".svg"), facecolor=WHITE)
    plt.close(fig)


def _render_sensitivity_overview(long: pd.DataFrame, *, filename: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13.2, 6.4), facecolor=WHITE,
                             sharey=True)
    fig.subplots_adjust(left=0.075, right=0.97, bottom=0.28, top=0.70, wspace=0.13)
    fig.text(0.075, 0.91, "How accuracy changes with error tolerance",
             fontsize=19, fontweight="bold", color=INK)
    fig.text(0.075, 0.855, "Same fitted MLR predictions, three predefined thresholds · |error| < tolerance",
             fontsize=10.7, color=MUTED)
    models = tuple(MOD14.build_model_registry())
    for ax, split, subtitle in zip(
        axes, ("training", "test"),
        ("DEVELOPMENT · 26 PARTICIPANTS", "INDEPENDENT TEST · 6 PARTICIPANTS"),
    ):
        _style(ax)
        for index, model_id in enumerate(models):
            scores = long.loc[(long["model"] == model_id) & (long["split"] == split)]
            scores = scores.sort_values("threshold_stages")
            ax.plot(scores["threshold_stages"], scores["accuracy_pct"],
                    color=SERIES_COLORS[index], marker="o", linewidth=2.5,
                    markersize=6.8, label=MODEL_LABELS[model_id].replace("\n", " "))
        ax.set_title(subtitle, fontsize=10.5, color=INK, loc="left", pad=17, fontweight="bold")
        ax.set_xticks(THRESHOLDS)
        ax.set_xticklabels([f"{t:.2f}" for t in THRESHOLDS])
        ax.set_xlabel("Error tolerance (difficulty stages)", fontsize=10.2,
                      color=MUTED, labelpad=13)
        ax.set_ylim(0, 104)
        ax.yaxis.set_major_locator(MultipleLocator(20))
        ax.yaxis.set_major_formatter(PercentFormatter(xmax=100))
    axes[0].set_ylabel("Accuracy (%)", fontsize=11, color=MUTED)
    handles, labels = axes[1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4,
               bbox_to_anchor=(0.5, 0.085), frameon=False, fontsize=10.5)
    fig.text(0.075, 0.025,
             "Higher tolerance increases apparent accuracy; models and predictions do not change.",
             fontsize=9.5, color=MUTED)
    filename.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(filename, dpi=270, facecolor=WHITE)
    fig.savefig(filename.with_suffix(".svg"), facecolor=WHITE)
    plt.close(fig)


def expected_output_paths(output_dir: Path) -> tuple[Path, ...]:
    stems = [
        "performance_with_accuracy", "accuracy_by_threshold", "accuracy_trial_audit",
        "analysis_manifest",
    ]
    outputs = [output_dir / (name + (".json" if name == "analysis_manifest" else ".csv"))
               for name in stems]
    figure_stems = [*(f"accuracy_lt_{threshold_token(t)}" for t in THRESHOLDS),
                    "accuracy_sensitivity_overview", "training_vs_test_mae",
                    "training_vs_test_r2"]
    for stem in figure_stems:
        outputs.extend((output_dir / f"{stem}.png", output_dir / f"{stem}.svg"))
    return tuple(outputs)


def create_outputs(
    *, wide: pd.DataFrame, long: pd.DataFrame, audit: pd.DataFrame,
    output_dir: Path, input_paths: tuple[Path, Path, Path], overwrite: bool = False,
) -> tuple[Path, ...]:
    """Write report and 6 charts without modifying frozen MOD-14 source files."""
    output_dir = output_dir.resolve()
    paths = expected_output_paths(output_dir)
    input_resolved = {p.resolve() for p in input_paths}
    if any(p.resolve() in input_resolved for p in paths):
        raise AccuracyReportError("Generated output would overwrite an original MOD-14 input")
    if any(output_dir == path.resolve().parent for path in input_paths):
        raise AccuracyReportError("Use a separate directory, not the frozen MOD-14 output directory")
    existing = [str(p) for p in paths if p.exists()]
    if existing and not overwrite:
        raise AccuracyReportError(
            "Generated outputs already exist (use --overwrite to replace extension outputs): "
            + ", ".join(existing)
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    wide.to_csv(output_dir / "performance_with_accuracy.csv", index=False)
    long.to_csv(output_dir / "accuracy_by_threshold.csv", index=False)
    audit.to_csv(output_dir / "accuracy_trial_audit.csv", index=False)
    for threshold in THRESHOLDS:
        token = threshold_token(threshold)
        _render_grouped_bars(
            wide,
            column_a=f"training_accuracy_lt_{token}_pct",
            column_b=f"test_accuracy_lt_{token}_pct",
            title=f"Accuracy within {threshold:.2f} difficulty stages",
            subtitle="Percentage of trial predictions with absolute error strictly below "
                     f"{threshold:.2f} stages",
            ylabel="Accuracy (%)", percent=True,
            filename=output_dir / f"accuracy_lt_{token}.png",
        )
    _render_sensitivity_overview(
        long, filename=output_dir / "accuracy_sensitivity_overview.png"
    )
    _render_grouped_bars(
        wide, column_a="training_mae", column_b="test_mae",
        title="Mean absolute error · training versus test",
        subtitle="Participant-balanced MAE from frozen MOD-14 performance results",
        ylabel="MAE (difficulty stages)",
        filename=output_dir / "training_vs_test_mae.png",
    )
    _render_grouped_bars(
        wide, column_a="training_r2", column_b="test_r2",
        title="Predictive R² · training versus test",
        subtitle="Pooled R² from frozen MOD-14 performance results",
        ylabel="R² (pooled)",
        filename=output_dir / "training_vs_test_r2.png",
    )
    manifest = {
        "module": "MOD_14_accuracy_sensitivity",
        "definition": "100 * count(abs(observed_stage - predicted_stage) < threshold) / count(all eligible prediction rows)",
        "thresholds_stages": list(THRESHOLDS),
        "round_or_clip_predictions": False,
        "model_fitting_performed": False,
        "training_metric": "fitted-training predictions; not bootstrap/OOB",
        "accuracy_weighting": "trial-pooled (one vote per eligible trial)",
        "mae_weighting": "participant-balanced (unchanged MOD-14)",
        "r2_weighting": "pooled observations (unchanged MOD-14)",
        "inputs": {
            label: {"filename": p.name, "sha256": hashlib.sha256(p.read_bytes()).hexdigest()}
            for label, p in zip(("training_predictions", "test_predictions", "train_test_performance"), input_paths)
        },
    }
    (output_dir / "analysis_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    return paths


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--mod14-output-dir", required=True, type=Path,
                        help="Directory containing frozen MOD-14 prediction and performance CSV files")
    parser.add_argument("--output-dir", required=True, type=Path,
                        help="New separate directory for accuracy reports and figures")
    parser.add_argument("--overwrite", action="store_true",
                        help="Allow replacing only the accuracy extension's generated outputs")
    args = parser.parse_args(argv)
    original = args.mod14_output_dir
    inputs = (
        original / "training_predictions.csv",
        original / "test_predictions.csv",
        original / "train_test_performance.csv",
    )
    wide, long, audit = load_and_calculate(
        training_path=inputs[0], test_path=inputs[1], performance_path=inputs[2],
    )
    paths = create_outputs(
        wide=wide, long=long, audit=audit, output_dir=args.output_dir,
        input_paths=inputs, overwrite=args.overwrite,
    )
    print("Validated frozen MOD-14 prediction pairs; original MAE and R² unchanged.")
    print(wide[["model", *(f"test_accuracy_lt_{threshold_token(t)}_pct" for t in THRESHOLDS)]].
          to_string(index=False, float_format=lambda v: f"{v:.2f}"))
    print(f"Created {len(paths)} output files in {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())