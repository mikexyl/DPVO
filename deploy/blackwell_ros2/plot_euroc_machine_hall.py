#!/usr/bin/env python3
"""Plot Machine Hall trajectories, ground truth, and inter-robot loops."""

from __future__ import annotations

import argparse
import csv
import io
import json
import zipfile
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D


ROBOTS = ("robot0", "robot1", "robot2")
ROBOT_LABELS = {
    "robot0": "R0 (MH_01)",
    "robot1": "R1 (MH_02)",
    "robot2": "R2 (MH_03)",
}
ROBOT_COLORS = {
    "robot0": "#0077BB",
    "robot1": "#EE7733",
    "robot2": "#009988",
    "robot3": "#CC3311",
    "robot4": "#AA4499",
    "robot5": "#DDCC77",
    "robot6": "#44AA99",
    "robot7": "#88CCEE",
    "robot8": "#332288",
    "robot9": "#EE3377",
    "robot10": "#999933",
    "robot11": "#661100",
    "robot12": "#117733",
}
METHODS = (
    ("raw", "Raw per-robot DPVO"),
    ("full_graph_centralized", r"Centralized $\mathrm{Sim}(3)$ PGO"),
    ("cbs", "CBS (observer R0)"),
)


def robot_sort_key(robot_id: str):
    suffix = robot_id.removeprefix("robot")
    return (0, int(suffix)) if suffix.isdigit() else (1, robot_id)


def robot_label(robot_id: str) -> str:
    suffix = robot_id.removeprefix("robot")
    return f"R{suffix}" if suffix.isdigit() else robot_id


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("result_dir", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument(
        "--graph",
        type=Path,
        help="Direct graph path for a dataset without ground truth.",
    )
    parser.add_argument("--centralized-csv", type=Path)
    parser.add_argument("--cbs-csv", type=Path)
    parser.add_argument("--prefix", default="mh_trajectories_loops")
    return parser.parse_args()


def configure_matplotlib() -> None:
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


def read_tum(path: Path) -> np.ndarray:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line and not line.startswith("#"):
            fields = line.split()
            rows.append([float(value) for value in fields[1:4]])
    return np.asarray(rows, dtype=np.float64)


def read_graph(path: Path):
    graph = json.loads(path.read_text(encoding="utf-8"))
    positions = {
        int(vertex["id"]): np.asarray(
            vertex["estimate"]["translation"], dtype=np.float64
        )
        for vertex in graph["vertices"]
    }
    order = {}
    for vertex in graph["vertices"]:
        order.setdefault(vertex["robot_id"], []).append(int(vertex["id"]))
    order = dict(sorted(order.items(), key=lambda item: robot_sort_key(item[0])))
    loops = [
        (int(edge["source"]), int(edge["target"]))
        for edge in graph["edges"]
        if edge["type"] == "inter_robot_loop_closure"
    ]
    return positions, order, loops


def read_csv(path: Path) -> dict[int, np.ndarray]:
    positions = {}
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            positions[int(row["vertex_id"])] = np.asarray(
                [row["tx"], row["ty"], row["tz"]], dtype=np.float64
            )
    return positions


def load_evo(path: Path) -> tuple[float, np.ndarray]:
    with zipfile.ZipFile(path) as archive:
        rmse = float(json.loads(archive.read("stats.json"))["rmse"])
        transform = np.load(
            io.BytesIO(archive.read("alignment_transformation_sim3.npy"))
        )
    return rmse, transform


def transform_points(points: np.ndarray, transform: np.ndarray) -> np.ndarray:
    return points @ transform[:3, :3].T + transform[:3, 3]


def plot_without_groundtruth(args, output_dir: Path) -> None:
    required = (args.graph, args.centralized_csv, args.cbs_csv)
    if not all(path is not None for path in required):
        raise ValueError(
            "--graph, --centralized-csv, and --cbs-csv are required together"
        )
    for path in required:
        if not path.is_file():
            raise FileNotFoundError(path)

    raw, order, loops = read_graph(args.graph)
    robots = tuple(order)
    positions = {
        "raw": raw,
        "full_graph_centralized": read_csv(args.centralized_csv),
        "cbs": read_csv(args.cbs_csv),
    }
    central_points = np.vstack(
        list(positions["full_graph_centralized"].values())
    )
    center = central_points.mean(axis=0)
    _, _, basis_rows = np.linalg.svd(
        central_points - center, full_matrices=False
    )
    basis = basis_rows[:2].T
    for column in range(2):
        dominant = int(np.argmax(np.abs(basis[:, column])))
        if basis[dominant, column] < 0.0:
            basis[:, column] *= -1.0

    projected = {
        method: {
            vertex_id: (point - center) @ basis
            for vertex_id, point in method_positions.items()
        }
        for method, method_positions in positions.items()
    }
    pooled = np.vstack(
        [
            np.vstack(list(method_positions.values()))
            for method_positions in projected.values()
        ]
    )
    lower = np.quantile(pooled, 0.002, axis=0)
    upper = np.quantile(pooled, 0.998, axis=0)
    span = upper - lower
    lower -= np.maximum(0.04 * span, 0.05)
    upper += np.maximum(0.04 * span, 0.05)

    figure, axes = plt.subplots(
        1, 3, figsize=(10.1, 2.35), sharex=True, sharey=True,
        constrained_layout=True,
    )
    generic_labels = {robot: robot_label(robot) for robot in robots}
    for panel, (axis, (method, title)) in enumerate(zip(axes, METHODS)):
        method_positions = projected[method]
        for robot in robots:
            color = ROBOT_COLORS[robot]
            estimate = np.vstack(
                [method_positions[vertex_id] for vertex_id in order[robot]]
            )
            axis.plot(
                estimate[:, 0], estimate[:, 1], color=color,
                linewidth=1.15, zorder=3,
            )
            axis.scatter(
                estimate[0, 0], estimate[0, 1], s=14,
                facecolor="white", edgecolor=color, linewidth=0.7, zorder=5,
            )
        for source, target in loops:
            segment = np.vstack(
                [method_positions[source], method_positions[target]]
            )
            axis.plot(
                segment[:, 0], segment[:, 1], color="#AA3377",
                linewidth=0.8, alpha=0.72, zorder=4,
            )
            axis.scatter(
                segment[:, 0], segment[:, 1], s=5, color="#AA3377",
                edgecolor="white", linewidth=0.2, alpha=0.82, zorder=4,
            )
        axis.set_title(f"({chr(ord('a') + panel)}) {title}")
        axis.set_xlabel("Map axis 1 [R0 units]")
        axis.set_xlim(lower[0], upper[0])
        axis.set_ylim(lower[1], upper[1])
        axis.set_aspect("equal", adjustable="box")
        axis.grid(True, color="#D9D9D9", linewidth=0.4, alpha=0.65)
    axes[0].set_ylabel("Map axis 2 [R0 units]")

    figure.legend(
        handles=[
            *[
                Line2D(
                    [0], [0], color=ROBOT_COLORS[robot], linewidth=1.6,
                    label=generic_labels[robot],
                )
                for robot in robots
            ],
            Line2D(
                [0], [0], color="#444444", linewidth=1.3, label="estimate"
            ),
            Line2D(
                [0], [0], color="#AA3377", linewidth=0.8,
                label="verified loop",
            ),
        ],
        loc="upper center", bbox_to_anchor=(0.5, 1.14),
        ncol=len(robots) + 2,
        frameon=False, columnspacing=1.25,
    )
    outputs = {}
    for suffix in ("pdf", "png"):
        path = output_dir / f"{args.prefix}.{suffix}"
        figure.savefig(path)
        outputs[suffix] = str(path.resolve())
    plt.close(figure)
    trace = {
        "figure": args.prefix,
        "graph": str(args.graph.resolve()),
        "inter_robot_loops": len(loops),
        "robots": list(robots),
        "coordinate_frame": "R0 observer frame with a shared PCA projection",
        "projection_center": center.tolist(),
        "projection_basis_columns": basis.tolist(),
        "shared_bounds": {"lower": lower.tolist(), "upper": upper.tolist()},
        "outputs": outputs,
    }
    (output_dir / f"{args.prefix}_trace.json").write_text(
        json.dumps(trace, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(trace, indent=2))


def main() -> None:
    args = parse_args()
    configure_matplotlib()
    result_dir = args.result_dir.resolve()
    output_dir = (args.output_dir or result_dir / "plots").resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.graph is not None:
        plot_without_groundtruth(args, output_dir)
        return
    tag = result_dir.name
    graph_path = result_dir / f"{tag}_pose_graph_keyframes_unoptimized.json"
    raw, order, loops = read_graph(graph_path)
    cbs_dir = result_dir / f"{tag}_cbs"
    positions = {
        "raw": raw,
        "full_graph_centralized": read_csv(cbs_dir / "centralized.csv"),
        "cbs": read_csv(cbs_dir / "cbs.csv"),
    }
    groundtruth = {
        robot: read_tum(result_dir / "evo" / f"groundtruth_{robot}.tum")
        for robot in ROBOTS
    }

    aligned = {}
    metrics = {}
    for method, _ in METHODS:
        rmse, transform = load_evo(
            result_dir / "evo" / "results" / f"{method}_joint.zip"
        )
        metrics[method] = rmse
        aligned[method] = {
            vertex_id: transform_points(point[None], transform)[0]
            for vertex_id, point in positions[method].items()
        }

    pooled = np.vstack(
        [
            *groundtruth.values(),
            *[
                np.vstack(list(method_positions.values()))
                for method_positions in aligned.values()
            ],
        ]
    )
    lower = np.quantile(pooled[:, :2], 0.002, axis=0)
    upper = np.quantile(pooled[:, :2], 0.998, axis=0)
    span = upper - lower
    lower -= np.maximum(0.04 * span, 0.05)
    upper += np.maximum(0.04 * span, 0.05)

    figure, axes = plt.subplots(
        1, 3, figsize=(10.1, 3.3), sharex=True, sharey=True, constrained_layout=True
    )
    for panel, (axis, (method, title)) in enumerate(zip(axes, METHODS)):
        method_positions = aligned[method]
        for robot in ROBOTS:
            color = ROBOT_COLORS[robot]
            estimate = np.vstack(
                [method_positions[vertex_id] for vertex_id in order[robot]]
            )
            truth = groundtruth[robot]
            axis.plot(
                truth[:, 0],
                truth[:, 1],
                color=color,
                linewidth=1.25,
                linestyle=(0, (2.5, 1.7)),
                alpha=0.48,
                zorder=1,
            )
            axis.plot(
                estimate[:, 0], estimate[:, 1], color=color, linewidth=1.15, zorder=3
            )
            axis.scatter(
                estimate[0, 0],
                estimate[0, 1],
                s=14,
                facecolor="white",
                edgecolor=color,
                linewidth=0.7,
                zorder=5,
            )
        for source, target in loops:
            segment = np.vstack([method_positions[source], method_positions[target]])
            axis.plot(
                segment[:, 0],
                segment[:, 1],
                color="#AA3377",
                linewidth=0.8,
                alpha=0.72,
                zorder=4,
            )
            axis.scatter(
                segment[:, 0],
                segment[:, 1],
                s=5,
                color="#AA3377",
                edgecolor="white",
                linewidth=0.2,
                alpha=0.82,
                zorder=4,
            )
        axis.set_title(f"({chr(ord('a') + panel)}) {title}\nJoint ATE {metrics[method]:.3f} m")
        axis.set_xlabel(r"EuRoC world $x$ [m]")
        axis.set_xlim(lower[0], upper[0])
        axis.set_ylim(lower[1], upper[1])
        axis.set_aspect("equal", adjustable="box")
        axis.grid(True, color="#D9D9D9", linewidth=0.4, alpha=0.65)
    axes[0].set_ylabel(r"EuRoC world $y$ [m]")

    robot_handles = [
        Line2D([0], [0], color=ROBOT_COLORS[robot], linewidth=1.6, label=ROBOT_LABELS[robot])
        for robot in ROBOTS
    ]
    style_handles = [
        Line2D([0], [0], color="#444444", linewidth=1.3, label="estimate"),
        Line2D(
            [0],
            [0],
            color="#444444",
            linewidth=1.2,
            linestyle=(0, (2.5, 1.7)),
            alpha=0.55,
            label="cam0 ground truth",
        ),
        Line2D([0], [0], color="#AA3377", linewidth=0.8, label="verified loop"),
    ]
    figure.legend(
        handles=robot_handles + style_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.08),
        ncol=6,
        frameon=False,
        columnspacing=1.25,
    )

    for suffix in ("pdf", "png"):
        figure.savefig(output_dir / f"mh_trajectories_loops.{suffix}")
    plt.close(figure)
    trace = {
        "figure": "mh_trajectories_loops",
        "result_dir": str(result_dir),
        "inter_robot_loops": len(loops),
        "joint_ate_rmse_m": metrics,
        "shared_xy_bounds": {"lower": lower.tolist(), "upper": upper.tolist()},
    }
    (output_dir / "mh_trajectories_loops_trace.json").write_text(
        json.dumps(trace, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(trace, indent=2))


if __name__ == "__main__":
    main()
