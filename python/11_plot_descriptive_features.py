"""Plot all-grab descriptive trial features without fitting or changing a model."""

import argparse
import json
from hashlib import sha256
from io import BytesIO
from pathlib import Path
from zipfile import ZipFile

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.ticker import FixedLocator, MaxNLocator
import numpy as np
import pandas as pd

CONDITIONS = ("Visual", "Auditory", "Cognitive")
COLOURS = ("#0072B2", "#9C4500", "#007454")
MARKERS = ("o", "s", "^")
LEVELS = (0, 1, 2, 3)
FEATURES = (
    ("correct_products_collected_percent", "Correct products collected", "%", (0, 104), (0, 25, 50, 75, 100)),
    ("self_reported_mental_demand", "Self-reported mental demand", "0–10", (-.35, 10.35), (0, 2, 4, 6, 8, 10)),
    ("median_time_between_grabs_seconds", "Median time between grabs", "s", (0, 35), None),
    ("shopping_list_recheck_count", "Number of shopping list rechecks", "count", (0, 140), None),
    ("shopping_list_recheck_total_seconds", "Total time rechecking the shopping list", "s", (0, 40), None),
    ("median_product_search_duration_seconds", "Product search duration", "s", (0, 4.2), None),
    ("median_non_task_focus_seconds", "Focus duration on non-task objects", "s", (-.015, .50), None),
    ("median_head_rotation_during_search_degrees", "Head rotation during search", "°", (0, 250), None),
    ("median_reach_start_to_grab_seconds", "Time from reach start to grab", "s", (0, 2.8), None),
    ("median_hand_travel_to_straight_distance_ratio", "Hand travel relative to straight-line distance", "unitless", (1, 2.5), None),
)

GROUPS = (
    ("descriptive_overview.pdf", FEATURES[:3]),
    ("descriptive_list_and_search.pdf", FEATURES[3:7]),
    ("descriptive_head_and_reach.pdf", FEATURES[7:]),
)


def load_trials(path):
    manifest_name = "feature_extraction_manifest.json"
    try:
        if path.suffix.lower() == ".zip":
            with ZipFile(path) as package:
                raw = package.read("trial_features.csv")
                manifest = json.loads(package.read(manifest_name))
        else:
            raw = path.read_bytes()
            manifest = json.loads(path.with_name(manifest_name).read_text(encoding="utf-8"))
    except (FileNotFoundError, KeyError) as error:
        raise ValueError("Supply the trial CSV with its extraction manifest or a ZIP containing both.") from error
    if (manifest.get("schema_version") != "fe04_all_grabs_v1"
            or manifest.get("search_and_reach_grab_scope") != "all_valid_product_grabs"
            or manifest.get("reach_requires_target_focus") is not True):
        raise ValueError("Run the extractor with --all-grabs before plotting this version.")
    digest = sha256(raw).hexdigest()
    if manifest.get("trial_features_sha256") != digest:
        raise ValueError("Trial CSV SHA-256 does not match its extraction manifest.")
    data = pd.read_csv(BytesIO(raw))
    if manifest.get("trial_count") != len(data):
        raise ValueError("Trial count does not match the extraction manifest.")
    keys = ["participant_id", "condition_name", "difficulty_level"]
    participants = data.participant_id.unique()
    expected = pd.MultiIndex.from_product((participants, CONDITIONS, LEVELS), names=keys)
    observed = pd.MultiIndex.from_frame(data[keys])
    if (len(participants) != 32 or len(data) != 384 or data.participant_id.isna().any()
            or not observed.is_unique or not observed.difference(expected).empty):
        raise ValueError("Expected one trial for each of 32 participants, three conditions and four levels.")
    data = data.sort_values(keys)
    return data, digest, manifest


def feature_values(data, feature, condition, level):
    values = data.loc[
        data.condition_name.eq(condition) & data.difficulty_level.eq(level), feature
    ].dropna().to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError(f"Non-finite value: {feature}, {condition}, {level}")
    return np.sort(values)


def marks(ax, x, values, colour, marker):
    # Spread observations within each level without changing measured values.
    # The offset has no temporal or participant interpretation.
    offsets = np.random.default_rng(31 + x).permutation(np.linspace(-.22, .22, len(values)))
    q1, med, q3 = np.quantile(values, [.25, .50, .75])
    ax.scatter(x + offsets - .025, values, marker=marker, s=10, linewidths=0,
               color=colour, alpha=.46, zorder=3)
    centre = x + .14
    ax.vlines(centre, q1, q3, color="white", linewidth=3.8, zorder=4)
    ax.vlines(centre, q1, q3, color="#202630", linewidth=1.25, zorder=5)
    ax.plot(centre, med, marker="D", markersize=4.1, color="#202630",
            markeredgecolor="white", markeredgewidth=.55, zorder=6)


def quantiles(values):
    return np.quantile(values, [.25, .50, .75]) if len(values) else (np.nan, np.nan, np.nan)


def draw(data, features, row_offset):
    row_height = 2.0
    height = row_height * len(features) + .32
    fig, axs = plt.subplots(len(features), 3, figsize=(6.88, height), squeeze=False)
    fig.subplots_adjust(left=.113, right=.988, top=1-.70/height,
                        bottom=.52/height, wspace=.26, hspace=.51)
    for row, (feature, label, unit, limits, ticks) in enumerate(features):
        groups = {(condition, level): feature_values(data, feature, condition, level)
                  for condition in CONDITIONS for level in LEVELS}
        observed = np.concatenate(list(groups.values()))
        if len(observed):
            pad = .04 * max(float(np.ptp(observed)), limits[1] - limits[0])
            lo, hi = float(observed.min()), float(observed.max())
            limits = (lo - pad if lo < limits[0] else limits[0],
                      hi + pad if hi > limits[1] else limits[1])
        letter = chr(65 + row_offset + row)
        axs[row, 0].text(0, 1.10, f"{letter}   {label} ({unit})", transform=axs[row, 0].transAxes,
                         ha="left", va="bottom", fontsize=10, fontweight="bold", clip_on=False)
        for j, (condition, colour, marker) in enumerate(zip(CONDITIONS, COLOURS, MARKERS)):
            ax = axs[row, j]
            if row == 0:
                ax.text(.5, 1.40, condition, transform=ax.transAxes, fontsize=9.5, fontweight="bold",
                        color=colour, ha="center", va="bottom", clip_on=False)
            ax.set(xlim=(-.42, 3.42), ylim=limits, xticks=LEVELS)
            ax.yaxis.set_major_locator(FixedLocator(ticks) if ticks is not None else MaxNLocator(4))
            ax.tick_params(axis="y", left=j == 0, labelleft=j == 0, length=2.5, pad=3)
            ax.tick_params(axis="x", length=2.5, pad=2)
            ax.grid(axis="y", linewidth=.55, color="#e0e4e9")
            ax.set_axisbelow(True)
            for spine in ("top", "right"):
                ax.spines[spine].set_visible(False)
            if j:
                ax.spines["left"].set_visible(False)
            for level in LEVELS:
                values = groups[condition, level]
                if len(values):
                    marks(ax, level, values, colour, marker)
                if len(values) < 32:
                    ax.text(level, .97, f"n = {len(values)}", transform=ax.get_xaxis_transform(),
                            fontsize=7.0, va="top", ha="center", color="#202630", zorder=7,
                            bbox={"facecolor": "white", "edgecolor": "none", "pad": .35})
    fig.supxlabel("Difficulty level", fontsize=10, y=.02)
    return fig


def captions():
    shared = ("Each dot represents one trial. Black diamonds are medians and bars span the "
              "25th to 75th percentiles. The bars show variation across trials, not confidence intervals. "
              "All groups contain 32 trials unless a smaller sample is marked. "
              "A group labelled n = 0 has no usable values. Each feature uses the same vertical scale across "
              "the Visual, Auditory and Cognitive conditions.")
    scope = ("Every valid grab is eligible, including repeat grabs and grabs of products not on the "
             "shopping list. Search and reach measures require logged focus on the product actually "
             "grabbed. Reaching also requires detectable hand movement, and the path ratio requires a "
             "usable hand path.")
    return "\n\n".join((
        "# Descriptive feature figures",
        "The figures describe 384 trials from 32 participants at difficulty levels 0–3. "
        "They show feature distributions without testing prediction accuracy.",
        "**A–C. Trial performance, mental demand and pace.** Percentage of correct products collected, "
        "self-reported mental demand and median time between grabs. " + shared,
        "**D–G. Shopping list use and product search.** Number and total duration of shopping list "
        "rechecks, product search duration and focus duration on non-task objects. " + shared + " " + scope,
        "**H–J. Head rotation and reaching.** Head rotation during search, time from reach start to grab "
        "and hand travel relative to straight-line distance. " + shared + " " + scope,
    )) + "\n"


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("source", type=Path, help="All-grab trial_features.csv or ZIP with its extraction manifest")
    p.add_argument("output_dir", type=Path, help="New figure directory")
    args = p.parse_args(argv)
    data, digest, manifest = load_trials(args.source)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    stats = []
    for k, *_ in FEATURES:
        for condition in CONDITIONS:
            for level in LEVELS:
                v = feature_values(data, k, condition, level)
                q1, med, q3 = quantiles(v)
                stats.append((k, condition, level, len(v), 32-len(v), med, q1, q3))
    pd.DataFrame(stats, columns=[
        "feature", "condition", "difficulty_level", "n", "missing", "median", "q25", "q75"
    ]).to_csv(args.output_dir / "descriptive_plot_statistics.csv", index=False)

    style = {"font.family": "DejaVu Sans", "font.size": 8, "axes.linewidth": .65, "pdf.fonttype": 42,
             "text.color": "#202630", "xtick.color": "#202630", "ytick.color": "#202630"}
    with plt.rc_context(style), PdfPages(args.output_dir / "descriptive_features_all.pdf",
                                       metadata={"Title": "Descriptive features from VR supermarket trials"}) as all_pdf:
        pos = 0
        for name, features in GROUPS:
            fig = draw(data, features, pos)
            all_pdf.savefig(fig)
            fig.savefig(args.output_dir / name)
            plt.close(fig)
            pos += len(features)
    (args.output_dir / "figure_captions.md").write_text(captions(), encoding="utf-8")
    provenance = {
        "source_schema": manifest["schema_version"],
        "search_and_reach_grab_scope": manifest["search_and_reach_grab_scope"],
        "reach_requires_target_focus": manifest["reach_requires_target_focus"],
        "trial_features_sha256": digest,
        "extraction_code_sha256": manifest.get("code_sha256", {}),
        "trial_count": len(data),
        "participant_count": int(data.participant_id.nunique()),
        "intended_use": "descriptive_reporting_not_model_evaluation",
        "plot_code_sha256": sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    (args.output_dir / "plot_manifest.json").write_text(
        json.dumps(provenance, indent=2, allow_nan=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
