#!/usr/bin/env python3
"""Plot KITTI ground truth, multi-robot trajectories, and verified loops."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


COLORS = (
    "#0072B2",
    "#E69F00",
    "#009E73",
    "#CC3311",
    "#AA4499",
    "#56B4E9",
    "#D55E00",
    "#F0E442",
    "#332288",
    "#88CCEE",
)
SOLUTIONS = (
    "raw",
    "online_centralized",
    "full_graph_centralized",
    "cbs",
)
SOLUTION_LABELS = {
    "raw": "Raw per-robot DPVO",
    "online_centralized": "Online anchor PGO",
    "full_graph_centralized": "Centralized Sim(3) PGO",
    "cbs": "CBS (observer R0)",
}


def configure_matplotlib() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
            "font.size": 8,
            "axes.titlesize": 8.5,
            "axes.labelsize": 8,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 7,
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("result_dir", type=Path)
    parser.add_argument("--tag")
    parser.add_argument("--evo-dir", type=Path)
    parser.add_argument("--output-dir", type=Path)
    return parser.parse_args()


def read_tum(path: Path) -> tuple[np.ndarray, np.ndarray]:
    data = np.loadtxt(path, comments="#", dtype=np.float64)
    data = np.atleast_2d(data)
    return data[:, 0], data[:, 1:4]


def discover_robots(evo_dir: Path) -> tuple[str, ...]:
    robots = tuple(
        sorted(
            (
                path.stem.removeprefix("groundtruth_")
                for path in evo_dir.glob("groundtruth_robot*.tum")
            ),
            key=lambda robot: int(robot.removeprefix("robot")),
        )
    )
    if not robots:
        raise FileNotFoundError(f"no groundtruth_robot*.tum files in {evo_dir}")
    if len(robots) > len(COLORS):
        raise ValueError(f"plot supports at most {len(COLORS)} robots")
    return robots


def associate(
    reference_times: np.ndarray, estimate_times: np.ndarray, max_diff: float = 0.02
) -> tuple[np.ndarray, np.ndarray]:
    indices = np.searchsorted(reference_times, estimate_times)
    indices = np.clip(indices, 1, len(reference_times) - 1)
    before = indices - 1
    choose_before = (
        np.abs(reference_times[before] - estimate_times)
        <= np.abs(reference_times[indices] - estimate_times)
    )
    matched = np.where(choose_before, before, indices)
    valid = np.abs(reference_times[matched] - estimate_times) <= max_diff
    return matched[valid], np.flatnonzero(valid)


def umeyama(source: np.ndarray, target: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
    source_mean = source.mean(axis=0)
    target_mean = target.mean(axis=0)
    source_centered = source - source_mean
    target_centered = target - target_mean
    covariance = target_centered.T @ source_centered / len(source)
    left, singular, right_t = np.linalg.svd(covariance)
    correction = np.eye(3)
    if np.linalg.det(left @ right_t) < 0:
        correction[-1, -1] = -1.0
    rotation = left @ correction @ right_t
    variance = np.sum(source_centered * source_centered) / len(source)
    scale = float(np.sum(singular * np.diag(correction)) / variance)
    translation = target_mean - scale * (rotation @ source_mean)
    return scale, rotation, translation


def align_solution(
    reference_grouped: dict[str, tuple[np.ndarray, np.ndarray]],
    grouped: dict[str, tuple[np.ndarray, np.ndarray]],
) -> tuple[dict[str, tuple[np.ndarray, np.ndarray]], float]:
    source_parts = []
    target_parts = []
    for robot, (times, xyz) in grouped.items():
        reference_times, reference_xyz = reference_grouped[robot]
        reference_indices, estimate_indices = associate(reference_times, times)
        source_parts.append(xyz[estimate_indices])
        target_parts.append(reference_xyz[reference_indices])
    scale, rotation, translation = umeyama(
        np.concatenate(source_parts), np.concatenate(target_parts)
    )
    aligned = {
        robot: (times, scale * (rotation @ xyz.T).T + translation)
        for robot, (times, xyz) in grouped.items()
    }
    return aligned, scale


def read_tum_pose(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    data = np.loadtxt(path, comments="#", dtype=np.float64)
    data = np.atleast_2d(data)
    return data[:, 0], data[:, 1:4], data[:, 4:8]


def quaternion_xyzw_to_rotation(quaternion: np.ndarray) -> np.ndarray:
    x, y, z, w = quaternion / np.linalg.norm(quaternion)
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def common_origin_trajectories(
    evo_dir: Path, prefix: str, robots: tuple[str, ...]
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Express every trajectory relative to its own first camera pose."""
    relative = {}
    for robot in robots:
        times, xyz, quaternions = read_tum_pose(evo_dir / f"{prefix}_{robot}.tum")
        initial_rotation = quaternion_xyzw_to_rotation(quaternions[0])
        relative[robot] = (
            times,
            (initial_rotation.T @ (xyz - xyz[0]).T).T,
        )
    return relative


def align_solution_shared_origin(
    reference_grouped: dict[str, tuple[np.ndarray, np.ndarray]],
    grouped: dict[str, tuple[np.ndarray, np.ndarray]],
) -> tuple[dict[str, tuple[np.ndarray, np.ndarray]], float]:
    """Fit one rotation and scale while keeping every trajectory start at zero."""
    source_parts = []
    target_parts = []
    for robot, (times, xyz) in grouped.items():
        reference_times, reference_xyz = reference_grouped[robot]
        reference_indices, estimate_indices = associate(reference_times, times)
        source_parts.append(xyz[estimate_indices])
        target_parts.append(reference_xyz[reference_indices])
    source = np.concatenate(source_parts)
    target = np.concatenate(target_parts)
    covariance = target.T @ source
    left, singular, right_t = np.linalg.svd(covariance)
    correction = np.eye(3)
    if np.linalg.det(left @ right_t) < 0:
        correction[-1, -1] = -1.0
    rotation = left @ correction @ right_t
    scale = float(
        np.sum(singular * np.diag(correction)) / np.sum(source * source)
    )
    return (
        {
            robot: (times, scale * (rotation @ xyz.T).T)
            for robot, (times, xyz) in grouped.items()
        },
        scale,
    )


def joint_ate_rmse(
    reference_grouped: dict[str, tuple[np.ndarray, np.ndarray]],
    grouped: dict[str, tuple[np.ndarray, np.ndarray]],
) -> float:
    residuals = []
    for robot, (times, xyz) in grouped.items():
        reference_times, reference_xyz = reference_grouped[robot]
        reference_indices, estimate_indices = associate(reference_times, times)
        residuals.append(xyz[estimate_indices] - reference_xyz[reference_indices])
    errors = np.linalg.norm(np.concatenate(residuals), axis=1)
    return float(np.sqrt(np.mean(np.square(errors))))


def nearest_point(
    times: np.ndarray, points: np.ndarray, timestamp: float, max_diff: float = 0.02
) -> np.ndarray | None:
    index = int(np.searchsorted(times, timestamp))
    candidates = [candidate for candidate in (index - 1, index) if 0 <= candidate < len(times)]
    if not candidates:
        return None
    nearest = min(candidates, key=lambda candidate: abs(times[candidate] - timestamp))
    if abs(times[nearest] - timestamp) > max_diff:
        return None
    return points[nearest]


def main() -> None:
    configure_matplotlib()
    args = parse_args()
    result_dir = args.result_dir.resolve()
    tag = args.tag or result_dir.name
    evo_dir = (args.evo_dir or result_dir / "evo").resolve()
    output_dir = (args.output_dir or result_dir / "plots").resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    robots = discover_robots(evo_dir)
    solutions = tuple(
        solution
        for solution in SOLUTIONS
        if all(
            (evo_dir / f"{solution}_{robot}.tum").is_file()
            for robot in robots
        )
    )
    if not solutions:
        raise FileNotFoundError(f"no complete solution trajectories in {evo_dir}")

    reference_grouped = {
        robot: read_tum(evo_dir / f"groundtruth_{robot}.tum")
        for robot in robots
    }
    common_origin_reference = common_origin_trajectories(
        evo_dir, "groundtruth", robots
    )
    reference_times = np.concatenate(
        [reference_grouped[robot][0] for robot in robots]
    )
    reference_xyz = np.concatenate(
        [reference_grouped[robot][1] for robot in robots]
    )
    order = np.argsort(reference_times)
    reference_times = reference_times[order]
    reference_xyz = reference_xyz[order]
    _, unique_indices = np.unique(reference_times, return_index=True)
    reference_xyz = reference_xyz[unique_indices]
    graph_path = result_dir / f"{tag}_pose_graph_keyframes_unoptimized.json"
    if not graph_path.is_file():
        graph_path = (
            result_dir
            / "geometric_verification"
            / "unoptimized_verified_graph.json"
        )
    graph = json.loads(graph_path.read_text(encoding="utf-8"))
    vertices = {int(vertex["id"]): vertex for vertex in graph["vertices"]}
    loops = [
        edge
        for edge in graph["edges"]
        if edge["type"] == "inter_robot_loop_closure"
    ]

    rows, columns = (2, 2) if len(solutions) == 4 else (1, len(solutions))
    figure, axes = plt.subplots(
        rows,
        columns,
        figsize=(8.0, 6.5) if rows == 2 else (4.0 * columns, 4.0),
        constrained_layout=True,
        squeeze=False,
    )
    for axis, solution in zip(axes.flat, solutions):
        grouped = {
            robot: read_tum(evo_dir / f"{solution}_{robot}.tum")
            for robot in robots
        }
        if solution == "raw":
            # Keep independent odometry tracks at a common zero origin and use
            # only one shared rotation/scale so relative scale errors remain.
            grouped = common_origin_trajectories(evo_dir, solution, robots)
            aligned, _ = align_solution_shared_origin(
                common_origin_reference, grouped
            )
            title = SOLUTION_LABELS[solution]
        else:
            aligned, _ = align_solution(reference_grouped, grouped)
            ate_rmse = joint_ate_rmse(reference_grouped, aligned)
            title = f"{SOLUTION_LABELS[solution]}\nJoint ATE {ate_rmse:.1f} m"
            axis.plot(
                reference_xyz[:, 0],
                reference_xyz[:, 2],
                color="black",
                linewidth=1.2,
                linestyle="--",
                label="Ground truth",
                zorder=1,
            )
        for robot, color in zip(robots, COLORS, strict=False):
            _, points = aligned[robot]
            axis.plot(
                points[:, 0],
                points[:, 2],
                color=color,
                linewidth=1.5,
                label=robot,
                zorder=2,
            )
        loop_count = 0
        for edge in loops if solution != "raw" else ():
            source = vertices[int(edge["source"])]
            target = vertices[int(edge["target"])]
            source_point = nearest_point(
                *aligned[source["robot_id"]], float(source["timestamp"])
            )
            target_point = nearest_point(
                *aligned[target["robot_id"]], float(target["timestamp"])
            )
            if source_point is None or target_point is None:
                continue
            axis.plot(
                [source_point[0], target_point[0]],
                [source_point[2], target_point[2]],
                color="#D55E00",
                linewidth=0.9,
                alpha=0.8,
                zorder=3,
            )
            axis.scatter(
                [source_point[0], target_point[0]],
                [source_point[2], target_point[2]],
                color="#D55E00",
                s=10,
                zorder=4,
            )
            loop_count += 1
        axis.set_title(title)
        axis.set_xlabel("x [m]")
        axis.set_ylabel("z [m]")
        axis.axis("equal")
        axis.grid(alpha=0.25)

    for axis in axes.flat[len(solutions) :]:
        axis.set_visible(False)

    handles = []
    labels = []
    for axis in axes.flat[: len(solutions)]:
        for handle, label in zip(*axis.get_legend_handles_labels(), strict=False):
            if label not in labels:
                handles.append(handle)
                labels.append(label)
    legend_columns = min(len(handles), 6)
    legend_y = 1.08 if len(handles) > legend_columns else 1.025
    figure.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, legend_y),
        ncol=legend_columns,
        frameon=False,
    )
    for suffix in ("png", "pdf"):
        figure.savefig(output_dir / f"{tag}_trajectories_loops.{suffix}", dpi=220)
    plt.close(figure)

    print(
        json.dumps(
            {
                "tag": tag,
                "inter_robot_loops": len(loops),
                "outputs": [
                    str(output_dir / f"{tag}_trajectories_loops.png"),
                    str(output_dir / f"{tag}_trajectories_loops.pdf"),
                ],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
