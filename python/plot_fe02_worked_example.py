#!/usr/bin/env python3
"""Render a thesis-ready worked example from a saved FE-02 trace."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def render(trace_path: Path, output_path: Path) -> None:
    trace = json.loads(Path(trace_path).read_text(encoding="utf-8"))
    if trace.get("schema_version") != "fe02_worked_trace_v1":
        raise ValueError("Expected a worked trace saved by the FE-02 extractor.")
    event = trace["events"]
    times = np.asarray(trace["hand_speed"], dtype=float)
    path = np.asarray(trace["reach_path_xyz_m"], dtype=float)
    if times.ndim != 2 or times.shape[1] != 2 or path.ndim != 2 or path.shape[1] != 3 or len(path) < 2:
        raise ValueError("Worked trace has invalid speed or trajectory samples.")
    end = event["grab_start"]
    fig = plt.figure(figsize=(8.1, 3.55), layout="constrained")
    layout = fig.add_gridspec(1, 2, width_ratios=(1.65, 1))
    axis = fig.add_subplot(layout[0, 0])
    geometry = fig.add_subplot(layout[0, 1], projection="3d")

    axis.set_title("A   Events and hand speed", loc="left", weight="bold")
    axis.axvspan(event["search_start"], event["first_target_focus"], color="#267c96",
                 alpha=0.15, label="Product search")
    axis.axvspan(event["reach_start"], end, facecolor="none", edgecolor="#9a4a16",
                 hatch="///", linewidth=0, alpha=0.45, label="Reach")
    if event["list_check_start"] is not None:
        axis.axvspan(event["list_check_start"], event["list_check_end"],
                     color="#666666", alpha=0.2, label="List check")
    if event["previous_grab_start"] is not None:
        axis.axvline(event["previous_grab_start"], linestyle=":", color="#555555",
                     label="Previous grab start")
    axis.axvline(event["first_target_focus"], linestyle="--", color="#267c96",
                 label="First target focus")
    axis.axvline(event["reach_start"], linestyle="-.", color="#9a4a16",
                 label="Reach start")
    axis.axvline(end, linestyle="-", color="#222222", linewidth=0.9, label="Grab start")
    axis.plot(times[:, 0], times[:, 1], color="#222222", linewidth=1.0, label="Hand speed")
    if trace["speed_threshold_m_s"] is not None:
        axis.axhline(trace["speed_threshold_m_s"], linestyle=(0, (1, 2)), color="#a24380",
                     label="Speed threshold")
    axis.set(xlabel="Time from start of shown interval (s)", ylabel="Hand speed (m/s)",
             xlim=(-0.025 * end, 1.025 * end), ylim=(0, None))
    axis.legend(frameon=False, fontsize=7.4, loc="upper center",
                bbox_to_anchor=(0.5, -0.19), ncol=3, columnspacing=0.9)

    geometry.set_title("B   Reach path", loc="left", weight="bold")
    shifted = path - path[0]
    geometry.plot(*shifted.T, label="Recorded hand path", color="#267c96", linewidth=1.8)
    geometry.plot(*shifted[[0, -1]].T, label="Straight line", color="#222222", linestyle="--")
    geometry.scatter(*shifted[0], marker="o", color="#9a4a16", s=22, label="Reach start")
    geometry.scatter(*shifted[-1], marker="s", color="#222222", s=22, label="Last reach sample")
    span = max(float(np.ptp(shifted, axis=0).max()), 0.01)
    for set_limits, coordinates in zip(
        (geometry.set_xlim, geometry.set_ylim, geometry.set_zlim), shifted.T
    ):
        midpoint = (coordinates.min() + coordinates.max()) / 2
        set_limits(midpoint - span * 0.54, midpoint + span * 0.54)
    geometry.set_box_aspect((1, 1, 1), zoom=0.83)
    geometry.set(xlabel="X (m)", ylabel="Y (m)", zlabel="Z (m)")
    geometry.tick_params(labelsize=7)
    geometry.legend(frameon=False, fontsize=7.4, loc="upper center",
                    bbox_to_anchor=(0.5, -0.09), ncol=2, columnspacing=0.8)
    try:
        output_path = Path(output_path)
        if output_path.suffix.lower() != ".pdf":
            raise ValueError("Use a .pdf output path for the worked figure.")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, format="pdf", bbox_inches="tight", pad_inches=0.12,
                    metadata={"Title": "Worked feature extraction example"})
    finally:
        plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trace", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    render(args.trace, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
