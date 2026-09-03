#!/usr/bin/env python3
"""Plot centralized and CBS joint sparse maps from labeled PLY exports."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from plyfile import PlyData


METHODS = ("Centralized", "CBS")
SOLUTION_NAMES = {
    "Centralized": "full_graph_centralized",
    "CBS": "cbs",
}
COLORS = {
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
    "robot13": "#006666",
}


def robot_sort_key(robot_id: str):
    suffix = robot_id.removeprefix("robot")
    return (0, int(suffix)) if suffix.isdigit() else (1, robot_id)


def robot_label(robot_id: str) -> str:
    suffix = robot_id.removeprefix("robot")
    return f"Robot {suffix}" if suffix.isdigit() else robot_id


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--centralized-ply", type=Path, required=True)
    parser.add_argument("--cbs-ply", type=Path, required=True)
    parser.add_argument("--centralized-csv", type=Path, required=True)
    parser.add_argument("--cbs-csv", type=Path, required=True)
    parser.add_argument(
        "--evo-dir",
        type=Path,
        help="Optional evo directory. Omit for the CBS robot0 reference frame.",
    )
    parser.add_argument(
        "--centralized-solution-name",
        default=SOLUTION_NAMES["Centralized"],
        help="TUM filename prefix for the centralized trajectory in --evo-dir",
    )
    parser.add_argument(
        "--cbs-solution-name",
        default=SOLUTION_NAMES["CBS"],
        help="TUM filename prefix for the CBS trajectory in --evo-dir",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--prefix", default="joint_map")
    parser.add_argument(
        "--projection-axes",
        choices=("xy", "xz", "yz"),
        default="xz",
        help="World-coordinate plane used when --evo-dir is supplied.",
    )
    parser.add_argument("--crop-quantile", type=float, default=0.005)
    parser.add_argument(
        "--max-time-difference",
        type=float,
        default=0.02,
        help="Maximum ground-truth/estimate timestamp difference in seconds.",
    )
    return parser.parse_args()


def configure_matplotlib() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
            "font.size": 8,
            "axes.titlesize": 9,
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


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_ply(path: Path) -> dict[str, np.ndarray]:
    vertex = PlyData.read(path)["vertex"].data
    xyz = np.column_stack((vertex["x"], vertex["y"], vertex["z"])).astype(
        np.float64
    )
    robot_indices = np.asarray(vertex["robot_id"], dtype=np.int64)
    return {
        f"robot{index}": xyz[robot_indices == index]
        for index in sorted(np.unique(robot_indices).tolist())
    }


def read_tum(path: Path) -> tuple[np.ndarray, np.ndarray]:
    data = np.atleast_2d(np.loadtxt(path, comments="#", dtype=np.float64))
    return data[:, 0], data[:, 1:4]


def read_csv_trajectories(path: Path) -> dict[str, np.ndarray]:
    import csv

    grouped = {}
    with path.open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            grouped.setdefault(row["robot_id"], []).append(
                (
                    int(row["keyframe_id"]),
                    np.asarray(
                        [row["tx"], row["ty"], row["tz"]],
                        dtype=np.float64,
                    ),
                )
            )
    return {
        robot: np.vstack(
            [point for _, point in sorted(rows, key=lambda item: item[0])]
        )
        for robot, rows in grouped.items()
    }


def pca_projection(trajectories: dict[str, np.ndarray]):
    points = np.concatenate(list(trajectories.values()))
    center = points.mean(axis=0)
    _, _, basis_rows = np.linalg.svd(points - center, full_matrices=False)
    basis = basis_rows[:2].T
    for column in range(2):
        dominant = int(np.argmax(np.abs(basis[:, column])))
        if basis[dominant, column] < 0.0:
            basis[:, column] *= -1.0
    return center, basis


def associate(
    reference_times: np.ndarray,
    estimate_times: np.ndarray,
    max_difference: float = 0.02,
) -> tuple[np.ndarray, np.ndarray]:
    # Match the shorter trajectory into the longer one, as evo does.  S3E has
    # sparse ground truth and dense estimates; iterating over every estimate
    # would otherwise reuse each ground-truth sample many times and produce an
    # alignment/ATE inconsistent with the authoritative evo result.
    reference_is_shorter = len(reference_times) <= len(estimate_times)
    queries = reference_times if reference_is_shorter else estimate_times
    candidates = estimate_times if reference_is_shorter else reference_times
    indices = np.searchsorted(candidates, queries)
    indices = np.clip(indices, 1, len(candidates) - 1)
    before = indices - 1
    use_before = (
        np.abs(candidates[before] - queries)
        <= np.abs(candidates[indices] - queries)
    )
    matched = np.where(use_before, before, indices)
    valid = np.abs(candidates[matched] - queries) <= max_difference
    query_indices = np.flatnonzero(valid)
    if reference_is_shorter:
        return query_indices, matched[valid]
    return matched[valid], query_indices


def umeyama(source: np.ndarray, target: np.ndarray):
    source_mean = source.mean(axis=0)
    target_mean = target.mean(axis=0)
    source_centered = source - source_mean
    target_centered = target - target_mean
    covariance = target_centered.T @ source_centered / len(source)
    left, singular, right_t = np.linalg.svd(covariance)
    correction = np.eye(3)
    if np.linalg.det(left @ right_t) < 0.0:
        correction[-1, -1] = -1.0
    rotation = left @ correction @ right_t
    variance = np.sum(source_centered * source_centered) / len(source)
    scale = float(np.sum(singular * np.diag(correction)) / variance)
    translation = target_mean - scale * (rotation @ source_mean)
    return scale, rotation, translation


def apply_sim3(
    points: np.ndarray,
    scale: float,
    rotation: np.ndarray,
    translation: np.ndarray,
) -> np.ndarray:
    return scale * (points @ rotation.T) + translation


def align_method(
    method: str,
    maps: dict[str, np.ndarray],
    evo_dir: Path,
    solution_names: dict[str, str],
    robots: tuple[str, ...],
    max_time_difference: float,
):
    reference = {
        robot: read_tum(evo_dir / f"groundtruth_{robot}.tum")
        for robot in robots
    }
    estimate = {
        robot: read_tum(
            evo_dir / f"{solution_names[method]}_{robot}.tum"
        )
        for robot in robots
    }
    source_parts = []
    target_parts = []
    associations = {}
    for robot in robots:
        reference_indices, estimate_indices = associate(
            reference[robot][0], estimate[robot][0], max_time_difference
        )
        associations[robot] = (reference_indices, estimate_indices)
        source_parts.append(estimate[robot][1][estimate_indices])
        target_parts.append(reference[robot][1][reference_indices])
    scale, rotation, translation = umeyama(
        np.concatenate(source_parts), np.concatenate(target_parts)
    )
    aligned_trajectories = {
        robot: apply_sim3(estimate[robot][1], scale, rotation, translation)
        for robot in robots
    }
    aligned_maps = {
        robot: apply_sim3(maps[robot], scale, rotation, translation)
        for robot in robots
    }
    residuals = []
    for robot in robots:
        reference_indices, estimate_indices = associations[robot]
        residuals.append(
            aligned_trajectories[robot][estimate_indices]
            - reference[robot][1][reference_indices]
        )
    ate = float(
        np.sqrt(np.mean(np.sum(np.square(np.concatenate(residuals)), axis=1)))
    )
    return aligned_maps, aligned_trajectories, reference, {
        "scale": scale,
        "rotation": rotation.tolist(),
        "translation": translation.tolist(),
        "joint_ate_rmse_m": ate,
    }


def main() -> None:
    args = parse_args()
    if not 0.0 <= args.crop_quantile < 0.5:
        raise ValueError("--crop-quantile must be in [0, 0.5)")
    if args.max_time_difference <= 0.0:
        raise ValueError("--max-time-difference must be positive")
    configure_matplotlib()

    ply_paths = {"Centralized": args.centralized_ply, "CBS": args.cbs_ply}
    csv_paths = {"Centralized": args.centralized_csv, "CBS": args.cbs_csv}
    required_paths = [*ply_paths.values(), *csv_paths.values()]
    if args.evo_dir is not None:
        required_paths.append(args.evo_dir)
    for path in required_paths:
        if not path.is_file():
            if path != args.evo_dir or not path.is_dir():
                raise FileNotFoundError(path)

    source_maps = {method: read_ply(path) for method, path in ply_paths.items()}
    robot_sets = {method: set(grouped) for method, grouped in source_maps.items()}
    if len({frozenset(value) for value in robot_sets.values()}) != 1:
        raise ValueError(f"PLY robot sets differ: {robot_sets}")
    robots = tuple(sorted(next(iter(robot_sets.values())), key=robot_sort_key))
    if not robots:
        raise ValueError("No robot-labeled map points found in the PLY inputs")
    missing_colors = set(robots) - set(COLORS)
    if missing_colors:
        raise ValueError(f"No plot colors configured for {sorted(missing_colors)}")
    solution_names = {
        "Centralized": args.centralized_solution_name,
        "CBS": args.cbs_solution_name,
    }
    alignments = {}
    groundtruth_2d = None
    if args.evo_dir is not None:
        projection_indices = {
            "xy": (0, 1),
            "xz": (0, 2),
            "yz": (1, 2),
        }[args.projection_axes]
        maps = {}
        trajectories = {}
        groundtruth = None
        for method in METHODS:
            maps[method], trajectories[method], reference, alignments[method] = (
                align_method(
                    method, source_maps[method], args.evo_dir,
                    solution_names, robots, args.max_time_difference,
                )
            )
            if groundtruth is None:
                groundtruth = reference
        maps_2d = {
            method: {
                robot: points[:, projection_indices]
                for robot, points in grouped.items()
            }
            for method, grouped in maps.items()
        }
        trajectories_2d = {
            method: {
                robot: points[:, projection_indices]
                for robot, points in grouped.items()
            }
            for method, grouped in trajectories.items()
        }
        groundtruth_2d = {
            robot: points[:, projection_indices]
            for robot, (_times, points) in groundtruth.items()
        }
        projection_center = None
        projection_basis = None
    else:
        maps = source_maps
        trajectories = {
            method: read_csv_trajectories(csv_paths[method])
            for method in METHODS
        }
        for method, grouped in trajectories.items():
            if set(grouped) != set(robots):
                raise ValueError(
                    f"{method} CSV robots {sorted(grouped)} do not match "
                    f"PLY robots {list(robots)}"
                )
        projection_center, projection_basis = pca_projection(
            trajectories["Centralized"]
        )
        maps_2d = {
            method: {
                robot: (points - projection_center) @ projection_basis
                for robot, points in grouped.items()
            }
            for method, grouped in maps.items()
        }
        trajectories_2d = {
            method: {
                robot: (points - projection_center) @ projection_basis
                for robot, points in grouped.items()
            }
            for method, grouped in trajectories.items()
        }

    pooled_map = np.concatenate(
        [maps_2d[method][robot] for method in METHODS for robot in robots]
    )
    pooled_trajectory = np.concatenate(
        [trajectories_2d[method][robot] for method in METHODS for robot in robots]
    )
    lower = np.quantile(pooled_map, args.crop_quantile, axis=0)
    upper = np.quantile(pooled_map, 1.0 - args.crop_quantile, axis=0)
    lower = np.minimum(lower, pooled_trajectory.min(axis=0))
    upper = np.maximum(upper, pooled_trajectory.max(axis=0))
    if groundtruth_2d is not None:
        pooled_groundtruth = np.concatenate(
            [groundtruth_2d[robot] for robot in robots]
        )
        lower = np.minimum(lower, pooled_groundtruth.min(axis=0))
        upper = np.maximum(upper, pooled_groundtruth.max(axis=0))
    margin = np.maximum(0.035 * (upper - lower), 0.1)
    lower -= margin
    upper += margin

    figure, axes = plt.subplots(
        1, 2, figsize=(7.1, 3.15), sharex=True, sharey=True
    )
    figure.subplots_adjust(left=0.08, right=0.99, bottom=0.16, top=0.82, wspace=0.10)
    displayed = {method: {} for method in METHODS}
    cbs_title = "(b) CBS (distributed robot-local readout)"
    for axis, method, title in zip(
        axes,
        METHODS,
        (r"(a) Centralized $\mathrm{Sim}(3)$ PGO", cbs_title),
        strict=True,
    ):
        if groundtruth_2d is not None:
            for robot in robots:
                reference = groundtruth_2d[robot]
                axis.plot(
                    reference[:, 0],
                    reference[:, 1],
                    color="black",
                    linewidth=0.9,
                    linestyle="--",
                    alpha=0.8,
                    zorder=2,
                )
        for robot in robots:
            points = maps_2d[method][robot]
            visible = (
                (points[:, 0] >= lower[0])
                & (points[:, 0] <= upper[0])
                & (points[:, 1] >= lower[1])
                & (points[:, 1] <= upper[1])
            )
            displayed[method][robot] = int(np.count_nonzero(visible))
            axis.scatter(
                points[visible, 0],
                points[visible, 1],
                s=0.5,
                color=COLORS[robot],
                alpha=0.22,
                linewidths=0,
                rasterized=True,
            )
            trajectory = trajectories_2d[method][robot]
            axis.plot(
                trajectory[:, 0],
                trajectory[:, 1],
                color=COLORS[robot],
                linewidth=1.15,
            )
            axis.scatter(
                trajectory[0, 0],
                trajectory[0, 1],
                s=13,
                facecolor="white",
                edgecolor=COLORS[robot],
                linewidth=0.7,
            )
        if groundtruth_2d is not None:
            title = (
                f"{title} — joint ATE "
                f"{alignments[method]['joint_ate_rmse_m']:.1f} m"
            )
        axis.set_title(title)
        axis.set_xlim(lower[0], upper[0])
        axis.set_ylim(lower[1], upper[1])
        axis.set_aspect("equal", adjustable="box")
        axis.set_xlabel(
            f"{args.projection_axes[0]} [m]"
            if groundtruth_2d is not None
            else "Map axis 1 [R0 units]"
        )
        axis.grid(alpha=0.25, linewidth=0.45)
    axes[0].set_ylabel(
        f"{args.projection_axes[1]} [m]"
        if groundtruth_2d is not None
        else "Map axis 2 [R0 units]"
    )

    figure.legend(
        handles=[
            *(
                [
                    Line2D(
                        [0], [0], color="black", linestyle="--",
                        linewidth=1.0, label="Ground truth",
                    )
                ]
                if groundtruth_2d is not None
                else []
            ),
            *[
            Line2D(
                [0], [0], color=COLORS[robot], linewidth=1.4,
                label=robot_label(robot),
            )
            for robot in robots
            ],
        ],
        loc="upper center",
        bbox_to_anchor=(0.5, 0.995),
        ncol=len(robots) + (1 if groundtruth_2d is not None else 0),
        frameon=False,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    outputs = {}
    for suffix in ("png", "pdf"):
        path = args.output_dir / f"{args.prefix}_joint_map_alignment.{suffix}"
        figure.savefig(path, dpi=300)
        outputs[suffix] = str(path.resolve())
    plt.close(figure)

    trace = {
        "figure": f"{args.prefix}_joint_map_alignment",
        "inputs": {
            method: {
                "ply": str(ply_paths[method].resolve()),
                "ply_sha256": sha256(ply_paths[method]),
                "trajectory_csv": str(csv_paths[method].resolve()),
                "trajectory_csv_sha256": sha256(csv_paths[method]),
            }
            for method in METHODS
        },
        "coordinate_frame": (
            "ground-truth frame after joint Sim(3) alignment"
            if args.evo_dir is not None
            else "R0 observer frame with a shared PCA projection"
        ),
        "robots": list(robots),
        "evo_dir": str(args.evo_dir.resolve()) if args.evo_dir else None,
        "projection_center": (
            projection_center.tolist() if projection_center is not None else None
        ),
        "projection_basis_columns": (
            projection_basis.tolist() if projection_basis is not None else None
        ),
        "solution_names": solution_names,
        "alignments": alignments,
        "crop_quantile": args.crop_quantile,
        "max_time_difference_seconds": args.max_time_difference,
        "projection_axes": args.projection_axes,
        "shared_bounds": {"lower": lower.tolist(), "upper": upper.tolist()},
        "displayed_points": displayed,
        "outputs": outputs,
    }
    trace_path = args.output_dir / f"{args.prefix}_joint_map_alignment_trace.json"
    trace_path.write_text(json.dumps(trace, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"outputs": outputs, "trace": str(trace_path.resolve())}, indent=2))


if __name__ == "__main__":
    main()
