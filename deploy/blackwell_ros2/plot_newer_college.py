#!/usr/bin/env python3
"""Plot Newer College raw/centralized/CBS trajectories and verified loops."""

from __future__ import annotations

import argparse
import csv
import io
import json
from pathlib import Path
import zipfile

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np


COLORS = {
    "robot0": "#0077BB",
    "robot1": "#EE7733",
    "robot2": "#009988",
    "robot3": "#CC3311",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--graph", type=Path, required=True)
    parser.add_argument("--solver-dir", type=Path, required=True)
    parser.add_argument("--evo-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dataset-label", default="Newer College")
    parser.add_argument("--output-prefix", default="newer_college")
    return parser.parse_args()


def configure() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
            "font.size": 7.5,
            "axes.titlesize": 8.2,
            "axes.labelsize": 7.5,
            "xtick.labelsize": 6.5,
            "ytick.labelsize": 6.5,
            "legend.fontsize": 6.8,
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def robot_sort_key(robot: str):
    suffix = robot.removeprefix("robot")
    return (0, int(suffix)) if suffix.isdigit() else (1, robot)


def read_graph(path: Path):
    document = json.loads(path.read_text(encoding="utf-8"))
    positions = {
        int(vertex["id"]): np.asarray(
            vertex["estimate"]["translation"], dtype=np.float64
        )
        for vertex in document["vertices"]
    }
    order = {}
    for vertex in document["vertices"]:
        order.setdefault(vertex["robot_id"], []).append(int(vertex["id"]))
    order = dict(sorted(order.items(), key=lambda item: robot_sort_key(item[0])))
    loops = [
        (int(edge["source"]), int(edge["target"]))
        for edge in document["edges"]
        if edge["type"] == "inter_robot_loop_closure"
    ]
    return positions, order, loops


def read_csv(path: Path) -> dict[int, np.ndarray]:
    positions = {}
    with path.open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            positions[int(row["vertex_id"])] = np.asarray(
                [row["tx"], row["ty"], row["tz"]], dtype=np.float64
            )
    return positions


def read_tum(path: Path) -> dict[str, np.ndarray]:
    data = np.atleast_2d(np.loadtxt(path, comments="#", dtype=np.float64))
    return {"timestamps": data[:, 0], "positions": data[:, 1:4]}


def load_evo(path: Path) -> tuple[float, np.ndarray]:
    with zipfile.ZipFile(path) as archive:
        rmse = float(json.loads(archive.read("stats.json"))["rmse"])
        transform = np.load(
            io.BytesIO(archive.read("alignment_transformation_sim3.npy"))
        )
    return rmse, transform


def transform_points(points: np.ndarray, transform: np.ndarray) -> np.ndarray:
    return points @ transform[:3, :3].T + transform[:3, 3]


def pca_projection(positions: dict[int, np.ndarray]):
    points = np.vstack(list(positions.values()))
    center = points.mean(axis=0)
    _, _, basis_rows = np.linalg.svd(points - center, full_matrices=False)
    basis = basis_rows[:2].T
    for column in range(2):
        dominant = int(np.argmax(np.abs(basis[:, column])))
        if basis[dominant, column] < 0.0:
            basis[:, column] *= -1.0
    return {
        vertex: (point - center) @ basis for vertex, point in positions.items()
    }, center, basis


def bounds(groups: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    pooled = np.vstack(groups)
    lower = np.quantile(pooled, 0.002, axis=0)
    upper = np.quantile(pooled, 0.998, axis=0)
    margin = np.maximum(0.04 * (upper - lower), 0.05)
    return lower - margin, upper + margin


def main() -> None:
    args = parse_args()
    configure()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    raw_3d, order, loops = read_graph(args.graph)
    robots = tuple(order)
    if set(robots) - set(COLORS):
        raise ValueError(f"no colors configured for {sorted(set(robots) - set(COLORS))}")
    raw, pca_center, pca_basis = pca_projection(raw_3d)

    cbs_method = "cbs"
    method_specs = (
        (
            "centralized_explicit_anchors",
            "Explicit-anchor centralized PGO",
            args.solver_dir / "centralized_explicit_anchors.csv",
        ),
        (
            cbs_method,
            "CBS (distributed robot-local readout)",
            args.solver_dir / f"{cbs_method}.csv",
        ),
    )
    aligned = {}
    metrics = {}
    groundtruth = {
        robot: read_tum(args.evo_dir / f"groundtruth_{robot}.tum")["positions"]
        for robot in robots
    }
    for method, _title, csv_path in method_specs:
        method_positions = read_csv(csv_path)
        rmse, transform = load_evo(args.evo_dir / "results" / f"{method}_joint.zip")
        aligned[method] = {
            vertex: transform_points(point[None], transform)[0]
            for vertex, point in method_positions.items()
        }
        metrics[method] = rmse

    figure, axes = plt.subplots(1, 3, figsize=(10.2, 3.0), constrained_layout=True)

    raw_groups = []
    for robot in robots:
        trajectory = np.vstack([raw[vertex] for vertex in order[robot]])
        raw_groups.append(trajectory)
        axes[0].plot(trajectory[:, 0], trajectory[:, 1], color=COLORS[robot], linewidth=1.15)
        axes[0].scatter(
            trajectory[0, 0], trajectory[0, 1], s=14,
            facecolor="white", edgecolor=COLORS[robot], linewidth=0.7,
        )
    for source, target in loops:
        segment = np.vstack((raw[source], raw[target]))
        axes[0].plot(segment[:, 0], segment[:, 1], color="#AA3377", linewidth=0.8, alpha=0.72)
    raw_lower, raw_upper = bounds(raw_groups)
    axes[0].set_title("(a) Raw per-robot DPVO")
    axes[0].set_xlabel("Map PCA axis 1 [local-map units]")
    axes[0].set_ylabel("Map PCA axis 2 [local-map units]")
    axes[0].set_xlim(raw_lower[0], raw_upper[0])
    axes[0].set_ylim(raw_lower[1], raw_upper[1])

    for panel, (method, title, _csv_path) in enumerate(method_specs, start=1):
        axis = axes[panel]
        method_positions = aligned[method]
        method_groups = []
        for robot in robots:
            truth = groundtruth[robot][:, :2]
            trajectory = np.vstack(
                [method_positions[vertex][:2] for vertex in order[robot]]
            )
            method_groups.extend((truth, trajectory))
            axis.plot(
                truth[:, 0], truth[:, 1], color=COLORS[robot], linewidth=1.0,
                linestyle=(0, (2.5, 1.7)), alpha=0.48,
            )
            axis.plot(trajectory[:, 0], trajectory[:, 1], color=COLORS[robot], linewidth=1.15)
            axis.scatter(
                trajectory[0, 0], trajectory[0, 1], s=14,
                facecolor="white", edgecolor=COLORS[robot], linewidth=0.7,
            )
        for source, target in loops:
            segment = np.vstack((method_positions[source], method_positions[target]))[:, :2]
            axis.plot(segment[:, 0], segment[:, 1], color="#AA3377", linewidth=0.8, alpha=0.72)
        lower, upper = bounds(method_groups)
        axis.set_title(
            f"({chr(ord('a') + panel)}) {title}\nJoint ATE {metrics[method]:.3f} m"
        )
        axis.set_xlabel(f"{args.dataset_label} world x [m]")
        if panel == 1:
            axis.set_ylabel(f"{args.dataset_label} world y [m]")
        axis.set_xlim(lower[0], upper[0])
        axis.set_ylim(lower[1], upper[1])

    for axis in axes:
        axis.set_aspect("equal", adjustable="box")
        axis.grid(True, color="#D9D9D9", linewidth=0.4, alpha=0.65)
    figure.legend(
        handles=[
            *[
                Line2D([0], [0], color=COLORS[robot], linewidth=1.6, label=robot.upper())
                for robot in robots
            ],
            Line2D([0], [0], color="#444444", linewidth=1.3, label="estimate"),
            Line2D(
                [0], [0], color="#444444", linewidth=1.0,
                linestyle=(0, (2.5, 1.7)), alpha=0.55,
                label="cam0 ground truth (optimized panels only)",
            ),
            Line2D([0], [0], color="#AA3377", linewidth=0.8, label="verified loop"),
        ],
        loc="upper center",
        bbox_to_anchor=(0.5, 1.08),
        ncol=len(robots) + 3,
        frameon=False,
    )
    outputs = {}
    for suffix in ("png", "pdf"):
        path = args.output_dir / f"{args.output_prefix}_trajectories_loops.{suffix}"
        figure.savefig(path)
        outputs[suffix] = str(path.resolve())
    plt.close(figure)
    trace = {
        "figure": f"{args.output_prefix}_trajectories_loops",
        "dataset_label": args.dataset_label,
        "graph": str(args.graph.resolve()),
        "robots": list(robots),
        "inter_robot_loops": len(loops),
        "raw_panel": {
            "groundtruth_overlay": False,
            "ate_reported": False,
            "projection_center": pca_center.tolist(),
            "projection_basis_columns": pca_basis.tolist(),
        },
        "optimized_joint_ate_rmse_m": metrics,
        "outputs": outputs,
    }
    (args.output_dir / f"{args.output_prefix}_trajectories_loops_trace.json").write_text(
        json.dumps(trace, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(trace, indent=2))


if __name__ == "__main__":
    main()
