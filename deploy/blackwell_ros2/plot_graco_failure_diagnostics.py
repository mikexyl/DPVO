#!/usr/bin/env python3
"""Plot diagnostic evidence for a disconnected GrAco verification graph."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path

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
    parser.add_argument("--verification", type=Path, required=True)
    parser.add_argument("--graph-gate", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--output-prefix", default="graco_failure_diagnostics")
    parser.add_argument("--dataset-label", default="GrAco Aerial 5–8")
    parser.add_argument(
        "--robot-label", action="append", default=[], metavar="ROBOT=LABEL"
    )
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


def pair_key(first: str, second: str) -> tuple[str, str]:
    return tuple(sorted((first, second), key=robot_sort_key))


def parse_robot_labels(values: list[str]) -> dict[str, str]:
    labels = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"--robot-label expects ROBOT=LABEL, got {value!r}")
        robot, label = value.split("=", 1)
        if not robot or not label or robot in labels:
            raise ValueError(f"invalid or duplicate --robot-label: {value!r}")
        labels[robot] = label
    return labels


def read_graph(path: Path):
    document = json.loads(path.read_text(encoding="utf-8"))
    positions = {
        int(vertex["id"]): np.asarray(
            vertex["estimate"]["translation"], dtype=np.float64
        )
        for vertex in document["vertices"]
    }
    vertex_robot = {
        int(vertex["id"]): vertex["robot_id"] for vertex in document["vertices"]
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
    return positions, vertex_robot, order, loops


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


def connected_components(robots, pair_counts):
    adjacency = {robot: set() for robot in robots}
    for (first, second), count in pair_counts.items():
        if count:
            adjacency[first].add(second)
            adjacency[second].add(first)
    components = []
    unseen = set(robots)
    while unseen:
        stack = [min(unseen, key=robot_sort_key)]
        component = []
        while stack:
            robot = stack.pop()
            if robot not in unseen:
                continue
            unseen.remove(robot)
            component.append(robot)
            stack.extend(adjacency[robot])
        components.append(sorted(component, key=robot_sort_key))
    return components


def main() -> None:
    args = parse_args()
    configure()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    positions_3d, vertex_robot, order, loops = read_graph(args.graph)
    positions, pca_center, pca_basis = pca_projection(positions_3d)
    robots = tuple(order)
    verification = json.loads(args.verification.read_text(encoding="utf-8"))
    graph_gate = json.loads(args.graph_gate.read_text(encoding="utf-8"))
    robot_labels = parse_robot_labels(args.robot_label)

    all_pairs = [
        (robots[first], robots[second])
        for first in range(len(robots))
        for second in range(first + 1, len(robots))
    ]
    candidate_counts = Counter()
    accepted_counts = Counter()
    rejection_stage_counts = Counter()
    for event in verification["events"]:
        pair = pair_key(event["owner_robot_id"], event["remote_robot_id"])
        candidate_counts[pair] += 1
        if event["status"] == "accepted":
            accepted_counts[pair] += 1
        else:
            rejection_stage_counts[(pair, event["stage"])] += 1

    pair_loop_counts = Counter(
        pair_key(vertex_robot[source], vertex_robot[target])
        for source, target in loops
    )
    components = connected_components(robots, pair_loop_counts)

    figure, axes = plt.subplots(
        1, 3, figsize=(10.4, 3.15), constrained_layout=True,
        gridspec_kw={"width_ratios": [1.3, 1.0, 0.85]},
    )

    axis = axes[0]
    groups = []
    for robot in robots:
        trajectory = np.vstack([positions[vertex] for vertex in order[robot]])
        groups.append(trajectory)
        axis.plot(
            trajectory[:, 0], trajectory[:, 1], color=COLORS[robot],
            linewidth=1.15,
        )
        axis.scatter(
            trajectory[0, 0], trajectory[0, 1], s=14, facecolor="white",
            edgecolor=COLORS[robot], linewidth=0.7, zorder=3,
        )
    for source, target in loops:
        segment = np.vstack((positions[source], positions[target]))
        axis.plot(
            segment[:, 0], segment[:, 1], color="#AA3377",
            linewidth=0.8, alpha=0.72,
        )
    pooled = np.vstack(groups)
    lower = np.quantile(pooled, 0.002, axis=0)
    upper = np.quantile(pooled, 0.998, axis=0)
    margin = np.maximum(0.04 * (upper - lower), 0.05)
    axis.set_xlim(lower[0] - margin[0], upper[0] + margin[0])
    axis.set_ylim(lower[1] - margin[1], upper[1] + margin[1])
    axis.set_aspect("equal", adjustable="box")
    axis.set_title("(a) Raw DPVO local gauges + accepted loops")
    axis.set_xlabel("Map PCA axis 1 [local-map units]")
    axis.set_ylabel("Map PCA axis 2 [local-map units]")
    axis.text(
        0.02, 0.02, "Diagnostic only: no GT overlay or raw ATE",
        transform=axis.transAxes, fontsize=6.2, color="#555555",
    )

    axis = axes[1]
    labels = [f"R{first[-1]}–R{second[-1]}" for first, second in all_pairs]
    totals = np.asarray([candidate_counts[pair] for pair in all_pairs])
    accepted = np.asarray([accepted_counts[pair] for pair in all_pairs])
    rejected_features = np.asarray(
        [rejection_stage_counts[(pair, "feature_matches")] for pair in all_pairs]
    )
    rejected_sim3 = totals - accepted - rejected_features
    y = np.arange(len(all_pairs))
    axis.barh(y, rejected_features, color="#BBBBBB", label="feature reject")
    axis.barh(
        y, rejected_sim3, left=rejected_features, color="#E6AB83",
        label="Sim(3) reject",
    )
    axis.barh(
        y, accepted, left=rejected_features + rejected_sim3,
        color="#AA3377", label="accepted",
    )
    for index, (kept, total) in enumerate(zip(accepted, totals)):
        axis.text(total + 2, index, f"{kept}/{total}", va="center", fontsize=6.4)
    axis.set_yticks(y, labels)
    axis.invert_yaxis()
    axis.set_xlim(0, max(totals) * 1.48)
    axis.set_xlabel("Top-1 candidates")
    axis.set_title("(b) Geometric-verification outcomes")
    axis.legend(frameon=False, loc="upper right")

    axis = axes[2]
    layout = {
        "robot0": np.array([0.0, 1.15]),
        "robot1": np.array([-0.85, -0.25]),
        "robot2": np.array([0.0, -0.95]),
        "robot3": np.array([0.85, -0.25]),
    }
    for pair, count in pair_loop_counts.items():
        first, second = pair
        segment = np.vstack((layout[first], layout[second]))
        axis.plot(
            segment[:, 0], segment[:, 1], color="#AA3377",
            linewidth=1.0 + 0.35 * count, alpha=0.8, zorder=1,
        )
        midpoint = segment.mean(axis=0)
        axis.text(
            midpoint[0], midpoint[1], str(count), ha="center", va="center",
            fontsize=7, color="#AA3377",
            bbox={"facecolor": "white", "edgecolor": "none", "pad": 0.8},
        )
    for robot in robots:
        point = layout[robot]
        axis.scatter(
            point[0], point[1], s=520, color=COLORS[robot],
            edgecolor="white", linewidth=1.4, zorder=2,
        )
        axis.text(
            point[0], point[1], f"R{robot[-1]}", ha="center", va="center",
            fontsize=8, color="white", weight="bold", zorder=3,
        )
    axis.set_xlim(-1.35, 1.35)
    axis.set_ylim(-1.35, 1.55)
    axis.set_aspect("equal", adjustable="box")
    axis.axis("off")
    axis.set_title("(c) Robot graph: FAILED (disconnected)")
    component_text = ", ".join(
        "+".join(f"R{robot[-1]}" for robot in component)
        for component in components
    )
    axis.text(
        0.0, -1.28, f"Components: {component_text}; Stage 3 was not run",
        ha="center", va="bottom", fontsize=6.8, color="#8B0000",
    )

    for axis in axes[:2]:
        axis.grid(True, color="#D9D9D9", linewidth=0.4, alpha=0.65)
    figure.legend(
        handles=[
            *[
                Line2D([0], [0], color=COLORS[robot], linewidth=1.6,
                       label=robot_labels.get(robot, f"R{robot[-1]}") )
                for robot in robots
            ],
            Line2D([0], [0], color="#AA3377", linewidth=1.2,
                   label="accepted inter-robot loop"),
        ],
        loc="upper center", bbox_to_anchor=(0.5, 1.075), ncol=5, frameon=False,
    )
    figure.suptitle(f"{args.dataset_label} — Stage 2 failure diagnostics", y=1.14)

    outputs = {}
    for suffix in ("png", "pdf"):
        path = args.output_dir / f"{args.output_prefix}.{suffix}"
        figure.savefig(path)
        outputs[suffix] = str(path.resolve())
    plt.close(figure)

    trace = {
        "figure": args.output_prefix,
        "dataset_label": args.dataset_label,
        "status": "failed_graph_disconnected",
        "graph": str(args.graph.resolve()),
        "verification": str(args.verification.resolve()),
        "graph_gate": str(args.graph_gate.resolve()),
        "graph_gate_pass": bool(graph_gate["pass"]),
        "graph_gate_error": graph_gate.get("error"),
        "robots": list(robots),
        "components": components,
        "top1_candidates": int(sum(candidate_counts.values())),
        "accepted_loops": int(sum(accepted_counts.values())),
        "pair_candidate_counts": {
            "--".join(pair): candidate_counts[pair] for pair in all_pairs
        },
        "pair_accepted_counts": {
            "--".join(pair): accepted_counts[pair] for pair in all_pairs
        },
        "raw_panel": {"groundtruth_overlay": False, "ate_reported": False},
        "projection_center": pca_center.tolist(),
        "projection_basis_columns": pca_basis.tolist(),
        "outputs": outputs,
    }
    trace_path = args.output_dir / f"{args.output_prefix}_trace.json"
    trace_path.write_text(json.dumps(trace, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(trace, indent=2))


if __name__ == "__main__":
    main()
