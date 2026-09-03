#!/usr/bin/env python3
"""Plot CU-Multi Stage 1/2 failure diagnostics after a disconnected graph gate."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime
import json
from pathlib import Path
import sys

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from deploy.blackwell_ros2.analyze_cu_multi_tracking import (
    align_similarity,
    associate,
    labeled_paths,
)
from deploy.blackwell_ros2.export_cu_multi_evo_tum import read_utm_groundtruth
from dpvo.loop_closure.tracking_artifact import load_tracking_artifact


COLORS = {"robot0": "#0077BB", "robot1": "#EE7733"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tracking-root", type=Path, required=True)
    parser.add_argument("--groundtruth", action="append", required=True)
    parser.add_argument("--verification", type=Path, required=True)
    parser.add_argument("--telemetry", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def configure() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
            "font.size": 7.5,
            "axes.titlesize": 8.5,
            "axes.labelsize": 7.5,
            "xtick.labelsize": 6.5,
            "ytick.labelsize": 6.5,
            "legend.fontsize": 6.8,
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "pdf.fonttype": 42,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def aligned_trajectories(tracking_root: Path, groundtruth: dict[str, Path]):
    output = {}
    for robot, truth_path in sorted(groundtruth.items()):
        artifact = load_tracking_artifact(tracking_root / robot)
        rows = read_utm_groundtruth(truth_path)
        truth_timestamps = np.asarray([row[0] for row in rows])
        truth_positions = np.asarray([row[1] for row in rows])
        estimate_indices, truth_indices = associate(
            artifact.keyframe_timestamps, truth_timestamps, 0.055
        )
        estimate = artifact.keyframe_poses_xyzw[estimate_indices, :3].astype(float)
        reference = truth_positions[truth_indices]
        aligned, scale, _rotation, _translation = align_similarity(estimate, reference)
        errors = np.linalg.norm(aligned - reference, axis=1)
        time_mask = (
            (truth_timestamps >= artifact.keyframe_timestamps[0] - 0.055)
            & (truth_timestamps <= artifact.keyframe_timestamps[-1] + 0.055)
        )
        output[robot] = {
            "truth": truth_positions[time_mask],
            "aligned": aligned,
            "rmse": float(np.sqrt(np.mean(errors**2))),
            "scale": scale,
        }
    return output


def read_telemetry(path: Path) -> dict[str, np.ndarray]:
    timestamps, temperatures, powers, memories, utilizations = [], [], [], [], []
    with path.open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            try:
                timestamps.append(datetime.strptime(row["timestamp"].strip(), "%Y/%m/%d %H:%M:%S.%f"))
                temperatures.append(float(row[" temperature.gpu"].strip()))
                powers.append(float(row[" power.draw [W]"].split()[0]))
                memories.append(float(row[" memory.used [MiB]"].split()[0]))
                utilizations.append(float(row[" utilization.gpu [%]"].split()[0]))
            except (KeyError, TypeError, ValueError):
                continue
    if not timestamps:
        raise RuntimeError(f"no telemetry samples in {path}")
    elapsed = np.asarray([(value - timestamps[0]).total_seconds() for value in timestamps])
    return {
        "elapsed": elapsed,
        "temperature": np.asarray(temperatures),
        "power": np.asarray(powers),
        "memory": np.asarray(memories),
        "utilization": np.asarray(utilizations),
    }


def main() -> None:
    args = parse_args()
    configure()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    groundtruth = labeled_paths(args.groundtruth, "--groundtruth")
    trajectories = aligned_trajectories(args.tracking_root, groundtruth)
    verification = json.loads(args.verification.read_text(encoding="utf-8"))
    events = verification["events"]
    scores = np.asarray([event["retrieval_score"] for event in events])
    matches = np.asarray([event.get("matches", 0) for event in events])
    telemetry = read_telemetry(args.telemetry)
    counters = verification["counters"]

    figure, axes = plt.subplots(2, 2, figsize=(9.2, 6.6), constrained_layout=True)
    trajectory_axis = axes[0, 0]
    origin = trajectories["robot0"]["truth"][0]
    for robot, values in trajectories.items():
        color = COLORS[robot]
        truth = values["truth"] - origin
        aligned = values["aligned"] - origin
        trajectory_axis.plot(
            truth[:, 0], truth[:, 1], color=color, linewidth=1.0,
            linestyle=(0, (2.5, 1.7)), alpha=0.48,
        )
        trajectory_axis.plot(
            aligned[:, 0], aligned[:, 1], color=color, linewidth=1.1,
            label=f"{robot} DPVO (ATE {values['rmse']:.1f} m)",
        )
    trajectory_axis.set_title("(a) Raw DPVO, independently Sim(3)-aligned")
    trajectory_axis.set_xlabel("UTM-relative x [m]")
    trajectory_axis.set_ylabel("UTM-relative y [m]")
    trajectory_axis.set_aspect("equal", adjustable="datalim")
    trajectory_axis.legend(frameon=False)

    retrieval_axis = axes[0, 1]
    scatter = retrieval_axis.scatter(
        np.arange(len(events)), scores, c=matches, cmap="viridis",
        vmin=0, vmax=max(15, int(matches.max())), s=5, alpha=0.75,
    )
    retrieval_axis.axhline(0.01, color="#CC3311", linewidth=0.8, linestyle="--")
    retrieval_axis.set_title("(b) DBoW2 top-1 candidates")
    retrieval_axis.set_xlabel("robot1 keyframe order")
    retrieval_axis.set_ylabel("BoW score")
    colorbar = figure.colorbar(scatter, ax=retrieval_axis, fraction=0.045, pad=0.02)
    colorbar.set_label("LightGlue 3D matches")

    rejection_axis = axes[1, 0]
    labels = ("Feature\nreject", "TEASER++\nreject", "Errors", "Accepted")
    values = (
        counters.get("rejected_feature_matches", 0),
        counters.get("rejected_sim3", 0),
        counters.get("errors", 0),
        verification.get("accepted_loop_count", 0),
    )
    rejection_axis.bar(labels, values, color=("#BBBBBB", "#EE7733", "#CC3311", "#009988"))
    rejection_axis.set_yscale("symlog", linthresh=1.0)
    rejection_axis.set_ylabel("Candidate count (symlog)")
    rejection_axis.set_title("(c) Verification outcome: disconnected graph")
    for index, value in enumerate(values):
        rejection_axis.text(index, max(value, 0.65), str(value), ha="center", va="bottom")

    telemetry_axis = axes[1, 1]
    minutes = telemetry["elapsed"] / 60.0
    telemetry_axis.plot(minutes, telemetry["power"], color="#CC3311", linewidth=0.85, label="Power [W]")
    telemetry_axis.plot(minutes, telemetry["temperature"], color="#EE7733", linewidth=0.85, label="Temperature [°C]")
    telemetry_axis.plot(minutes, telemetry["utilization"], color="#0077BB", linewidth=0.75, alpha=0.8, label="GPU utilization [%]")
    telemetry_axis.set_xlabel("MPS retry elapsed time [min]")
    telemetry_axis.set_ylabel("Value")
    telemetry_axis.set_title("(d) MPS retry GPU telemetry")
    telemetry_axis.legend(frameon=False, ncol=3, loc="upper center")

    figure.suptitle(
        "CU-Multi Main Campus — Stage 2 failure diagnostics\n"
        "0 verified cross-robot loops; Stage 3 correctly not executed",
        fontsize=10,
    )
    for suffix in ("png", "pdf"):
        figure.savefig(args.output_dir / f"cu_multi_stage2_failure_diagnostics.{suffix}")
    plt.close(figure)

    trace = {
        "format": "dpvo_cu_multi_stage2_failure_plot",
        "version": 1,
        "accepted_loop_count": verification.get("accepted_loop_count", 0),
        "counters": counters,
        "raw_tracking": {
            robot: {"ate_rmse_m": values["rmse"], "scale": values["scale"]}
            for robot, values in trajectories.items()
        },
        "telemetry": {
            "samples": int(len(telemetry["elapsed"])),
            "max_power_w": float(telemetry["power"].max()),
            "max_temperature_c": float(telemetry["temperature"].max()),
            "max_memory_mib": float(telemetry["memory"].max()),
            "max_gpu_utilization_percent": float(telemetry["utilization"].max()),
        },
        "note": "Ground truth is Base-frame UTM; camera extrinsic was unavailable, so translation uses a Base-position proxy.",
    }
    (args.output_dir / "cu_multi_stage2_failure_diagnostics.json").write_text(
        json.dumps(trace, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(trace, indent=2))


if __name__ == "__main__":
    main()
