#!/usr/bin/env python3
"""Validate and visualize a controlled CBS covariance-transport suite."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


METHODS = ("none", "adjoint", "bernoulli")
METHOD_LABELS = {"none": "None", "adjoint": "Adjoint", "bernoulli": "Bernoulli"}
METHOD_COLORS = {"none": "#7f7f7f", "adjoint": "#1f77b4", "bernoulli": "#ff7f0e"}
DATASET_LABELS = {
    "iphone14": "iPhone (14 robots)",
    "kitti00_10_overlap50": "KITTI-00 (10 robots, overlap 50)",
    "tum": "TUM fr1/desk + desk2",
    "euroc": "EuRoC V1_01/02/03",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def _trajectory(path: Path) -> dict[int, dict[str, str]]:
    return {int(row["vertex_id"]): row for row in _rows(path)}


def _translation_distance(left: dict[str, str], right: dict[str, str]) -> float:
    return math.sqrt(
        sum((float(left[key]) - float(right[key])) ** 2 for key in ("tx", "ty", "tz"))
    )


def _rmse(values: list[float]) -> float:
    return math.sqrt(statistics.fmean(value * value for value in values))


def _command_flag(command: list[str], name: str) -> str:
    prefix = f"--{name}="
    values = [item[len(prefix) :] for item in command if item.startswith(prefix)]
    if len(values) != 1:
        raise ValueError(f"command must contain exactly one {prefix} flag")
    return values[0]


def _optional_command_flag(command: list[str], name: str, default: str) -> str:
    prefix = f"--{name}="
    values = [item[len(prefix) :] for item in command if item.startswith(prefix)]
    if len(values) > 1:
        raise ValueError(f"command contains more than one {prefix} flag")
    return values[0] if values else default


def _normalized_command(command: list[str]) -> list[str]:
    replacements = {
        "--input_graph=": "--input_graph=<copied-input>",
        "--output_dir=": "--output_dir=<output>",
        "--sim3_covariance_transport=": "--sim3_covariance_transport=<transport>",
    }
    output = []
    for item in command:
        replacement = next(
            (value for prefix, value in replacements.items() if item.startswith(prefix)),
            None,
        )
        output.append(replacement if replacement is not None else item)
    return output


def _projection(trajectory: dict[int, dict[str, str]]) -> tuple[int, int]:
    points = np.asarray(
        [
            [float(row["tx"]), float(row["ty"]), float(row["tz"])]
            for _, row in sorted(trajectory.items())
        ]
    )
    variances = np.var(points, axis=0)
    axes = np.argsort(variances)[-2:][::-1]
    return int(axes[0]), int(axes[1])


def _save_figure(figure: plt.Figure, base: Path) -> None:
    figure.tight_layout()
    for suffix in ("png", "pdf"):
        figure.savefig(base.with_suffix(f".{suffix}"), dpi=200)
    plt.close(figure)


def analyze_dataset(dataset_root: Path, dataset_name: str) -> dict:
    run_dirs = {method: dataset_root / method / "dpgo" for method in METHODS}
    summaries = {method: _rows(path / "summary.csv") for method, path in run_dirs.items()}
    by_solution = {
        method: {row["solution"]: row for row in rows}
        for method, rows in summaries.items()
    }
    for method in METHODS:
        if "cbs" not in by_solution[method]:
            raise ValueError(f"{dataset_name}/{method} is missing the single CBS trajectory")

    errors = {
        method: float(by_solution[method]["cbs"]["graph_error"])
        for method in METHODS
    }
    if not all(math.isfinite(value) and value >= 0.0 for value in errors.values()):
        raise ValueError(f"{dataset_name} contains a non-finite CBS graph error")
    centralized_error = float(
        by_solution["bernoulli"]["centralized_explicit_anchors"]["graph_error"]
    )

    convergence = {
        method: _rows(path / "cbs_convergence.csv") for method, path in run_dirs.items()
    }
    iterations = [int(row["iteration"]) for row in convergence["bernoulli"]]
    for method in METHODS:
        if iterations != [int(row["iteration"]) for row in convergence[method]]:
            raise ValueError(f"{dataset_name} convergence iterations differ")
    convergence_errors = {
        method: [float(row["graph_error"]) for row in convergence[method]]
        for method in METHODS
    }

    trajectories = {
        method: _trajectory(path / "cbs.csv") for method, path in run_dirs.items()
    }
    centralized = _trajectory(run_dirs["bernoulli"] / "centralized_explicit_anchors.csv")
    vertex_ids = sorted(centralized)
    if any(sorted(trajectories[method]) != vertex_ids for method in METHODS):
        raise ValueError(f"{dataset_name} trajectory vertex sets differ")
    robot_ids = sorted({row["robot_id"] for row in trajectories["bernoulli"].values()})
    trajectory_rmse = []
    for robot_id in robot_ids:
        robot_vertices = [
            vertex
            for vertex in vertex_ids
            if trajectories["bernoulli"][vertex]["robot_id"] == robot_id
        ]
        row = {"robot_id": robot_id, "pose_count": len(robot_vertices)}
        for method in METHODS:
            row[f"{method}_to_centralized_translation_rmse_m"] = _rmse(
                [
                    _translation_distance(trajectories[method][vertex], centralized[vertex])
                    for vertex in robot_vertices
                ]
            )
        trajectory_rmse.append(row)
    pairwise_trajectory = {}
    for left, right in (
        ("adjoint", "none"),
        ("bernoulli", "none"),
        ("bernoulli", "adjoint"),
    ):
        differences = [
            _translation_distance(trajectories[left][vertex], trajectories[right][vertex])
            for vertex in vertex_ids
        ]
        pairwise_trajectory[f"{left}_vs_{right}"] = {
            "translation_rmse_m": _rmse(differences),
            "max_translation_difference_m": max(differences),
        }

    provenance = {
        method: json.loads((path / "offline_dpgo_provenance.json").read_text())
        for method, path in run_dirs.items()
    }
    schedule = {
        "iterations": int(_command_flag(provenance["bernoulli"]["command"], "iterations")),
        "stage_mode": _command_flag(provenance["bernoulli"]["command"], "stage_mode"),
        "pose_warmup_iterations": int(
            _command_flag(provenance["bernoulli"]["command"], "pose_warmup_iterations")
        ),
        "pose_block_iterations": int(
            _command_flag(provenance["bernoulli"]["command"], "pose_block_iterations")
        ),
        "anchor_block_iterations": int(
            _command_flag(provenance["bernoulli"]["command"], "anchor_block_iterations")
        ),
        "target_hellinger": float(
            _command_flag(provenance["bernoulli"]["command"], "target_hellinger")
        ),
        "hellinger_quadratic_term": _optional_command_flag(
            provenance["bernoulli"]["command"],
            "hellinger_quadratic_term",
            "false",
        )
        == "true",
        "d_reset": float(
            _command_flag(provenance["bernoulli"]["command"], "d_reset")
        ),
        "bootstrap_robot_anchors": _command_flag(
            provenance["bernoulli"]["command"], "bootstrap_robot_anchors"
        ) == "true",
    }
    required_files = (
        "input_keyframes_unoptimized.json",
        "input_keyframes_unoptimized.g2o",
        "offline_dpgo_provenance.json",
        "summary.csv",
        "cbs_convergence.csv",
        "cbs_anchors.csv",
        "cbs.csv",
        "dpvo_sim3_cbs.rrd",
        "centralized.csv",
        "centralized_explicit_anchors.csv",
    )
    artifacts = {
        method: {
            filename: {
                "exists": (path / filename).is_file(),
                "bytes": (path / filename).stat().st_size if (path / filename).is_file() else 0,
            }
            for filename in required_files
        }
        for method, path in run_dirs.items()
    }
    source_hashes = {
        method: provenance[method]["source_sha256"] for method in METHODS
    }
    executable_hashes = {}
    for method in METHODS:
        executable = Path(provenance[method]["command"][0])
        if not executable.is_file():
            raise FileNotFoundError(executable)
        executable_hashes[method] = _sha256(executable)
    centralized_hashes = {
        filename: {
            method: _sha256(run_dirs[method] / filename) for method in METHODS
        }
        for filename in ("centralized.csv", "centralized_explicit_anchors.csv")
    }
    normalized_commands = {
        method: _normalized_command(provenance[method]["command"])
        for method in METHODS
    }
    controls_pass = bool(
        all(
            provenance[method].get("status") == "complete"
            and provenance[method].get("return_code") == 0
            and provenance[method].get("copied_input_is_byte_identical") is True
            and _command_flag(provenance[method]["command"], "sim3_covariance_transport")
            == method
            and all(item["exists"] and item["bytes"] > 0 for item in artifacts[method].values())
            for method in METHODS
        )
        and len(set(source_hashes.values())) == 1
        and len(set(executable_hashes.values())) == 1
        and len({tuple(command) for command in normalized_commands.values()}) == 1
        and all(len(set(hashes.values())) == 1 for hashes in centralized_hashes.values())
        and not schedule["bootstrap_robot_anchors"]
    )

    comparison_dir = dataset_root / "comparison"
    comparison_dir.mkdir(parents=True, exist_ok=True)
    improvements = {
        method: 100.0 * (errors["none"] - errors[method]) / errors["none"]
        for method in ("adjoint", "bernoulli")
    }
    minimum_error = min(errors.values())
    tied_winners = [
        method for method in METHODS if errors[method] == minimum_error
    ]
    metrics = {
        "format": "sim3_covariance_transport_abc_dataset",
        "version": 1,
        "dataset": dataset_name,
        "label": DATASET_LABELS.get(dataset_name, dataset_name),
        "criterion": "lower pose-graph error is better",
        "controlled_comparison_passed": controls_pass,
        "robot_count": len(robot_ids),
        "robot_ids": robot_ids,
        "schedule": schedule,
        "cbs_readout": (
            "each robot's own trajectory transformed to robot0 using that "
            "robot's relative anchor estimate"
        ),
        "graph_error": errors,
        "centralized_explicit_anchor_graph_error": centralized_error,
        "relative_improvement_vs_none_percent": improvements,
        "winner": tied_winners[0] if len(tied_winners) == 1 else "tie",
        "tied_winners": tied_winners,
        "convergence": {
            "logged_points": len(iterations),
            "iterations": iterations,
            "graph_error": convergence_errors,
            "minimum_logged_graph_error": {
                method: min(values) for method, values in convergence_errors.items()
            },
            "minimum_iteration": {
                method: iterations[min(range(len(values)), key=values.__getitem__)]
                for method, values in convergence_errors.items()
            },
        },
        "trajectory_vs_centralized": {"per_robot": trajectory_rmse},
        "pairwise_distributed_trajectory_difference": pairwise_trajectory,
        "provenance": {
            "source_sha256": source_hashes,
            "executable_sha256": executable_hashes,
            "commands_match_except_transport": len(
                {tuple(command) for command in normalized_commands.values()}
            ) == 1,
            "centralized_output_sha256": centralized_hashes,
            "artifacts": artifacts,
        },
    }
    (comparison_dir / "comparison_metrics.json").write_text(
        json.dumps(metrics, indent=2) + "\n"
    )

    figure, axis = plt.subplots(figsize=(7.6, 5.4))
    bars = axis.bar(
        [METHOD_LABELS[method] for method in METHODS],
        [errors[method] for method in METHODS],
        color=[METHOD_COLORS[method] for method in METHODS],
    )
    axis.axhline(
        centralized_error,
        color="black",
        linestyle="--",
        linewidth=1.0,
        label="Centralized explicit anchors",
    )
    axis.bar_label(bars, fmt="%.6g", padding=3)
    positive = [value for value in [*errors.values(), centralized_error] if value > 0.0]
    if positive and max(positive) / min(positive) > 50.0:
        axis.set_yscale("log")
    axis.set_ylabel("Pose-graph error (lower is better)")
    axis.set_title(f"{metrics['label']}: final CBS objective")
    axis.grid(axis="y", alpha=0.25)
    axis.legend()
    _save_figure(figure, comparison_dir / "final_graph_error")

    figure, (axis, delta_axis) = plt.subplots(
        2,
        1,
        figsize=(9.2, 7.4),
        sharex=True,
        gridspec_kw={"height_ratios": [2.0, 1.0]},
    )
    for method in METHODS:
        axis.plot(
            iterations,
            convergence_errors[method],
            color=METHOD_COLORS[method],
            label=METHOD_LABELS[method],
            linewidth=1.25,
        )
    axis.axhline(
        centralized_error,
        color="black",
        linestyle="--",
        linewidth=1.0,
        label="Centralized explicit anchors",
    )
    axis.set_yscale("log")
    axis.set_ylabel("Pose-graph error (log scale)")
    axis.set_title(f"{metrics['label']}: matched covariance-transport convergence")
    axis.grid(alpha=0.25)
    axis.legend(ncol=2)
    for method in ("adjoint", "bernoulli"):
        relative_delta = [
            100.0 * (none_error - method_error) / none_error
            for none_error, method_error in zip(
                convergence_errors["none"], convergence_errors[method]
            )
        ]
        delta_axis.plot(
            iterations,
            relative_delta,
            color=METHOD_COLORS[method],
            label=METHOD_LABELS[method],
            linewidth=1.15,
        )
    delta_axis.axhline(0.0, color="black", linewidth=0.8)
    delta_axis.set_xlabel("CBS iteration")
    delta_axis.set_ylabel("Improvement vs none (%)")
    delta_axis.grid(alpha=0.25)
    delta_axis.legend(ncol=2)
    _save_figure(figure, comparison_dir / "convergence")

    dimension_names = ("x", "y", "z")
    projection = _projection(centralized)
    colors = plt.get_cmap("tab20")
    figure, axes = plt.subplots(1, 4, figsize=(20, 5.1))
    for method_index, method in enumerate(METHODS):
        axis = axes[method_index]
        for robot_index, robot_id in enumerate(robot_ids):
            robot_vertices = [
                vertex
                for vertex in vertex_ids
                if trajectories[method][vertex]["robot_id"] == robot_id
            ]
            central_u = [float(centralized[v][("tx", "ty", "tz")[projection[0]]]) for v in robot_vertices]
            central_v = [float(centralized[v][("tx", "ty", "tz")[projection[1]]]) for v in robot_vertices]
            method_u = [float(trajectories[method][v][("tx", "ty", "tz")[projection[0]]]) for v in robot_vertices]
            method_v = [float(trajectories[method][v][("tx", "ty", "tz")[projection[1]]]) for v in robot_vertices]
            color = colors(robot_index % 20)
            axis.plot(central_u, central_v, color=color, linestyle=":", linewidth=0.8, alpha=0.6)
            axis.plot(method_u, method_v, color=color, linewidth=1.2)
        axis.set_title(f"{METHOD_LABELS[method]} (solid) vs central")
        axis.set_xlabel(f"{dimension_names[projection[0]]} (m)")
        axis.set_ylabel(f"{dimension_names[projection[1]]} (m)")
        axis.axis("equal")
        axis.grid(alpha=0.2)
    bar_axis = axes[3]
    robot_x = np.arange(len(robot_ids))
    width = 0.25
    for offset, method in zip((-1, 0, 1), METHODS):
        bar_axis.bar(
            robot_x + offset * width,
            [row[f"{method}_to_centralized_translation_rmse_m"] for row in trajectory_rmse],
            width,
            color=METHOD_COLORS[method],
            label=METHOD_LABELS[method],
        )
    bar_axis.set_title("Distance to centralized")
    bar_axis.set_xlabel("Robot")
    bar_axis.set_ylabel("Translation RMSE (m)")
    bar_axis.set_xticks(robot_x, [robot.removeprefix("robot") for robot in robot_ids])
    bar_axis.grid(axis="y", alpha=0.2)
    bar_axis.legend()
    figure.suptitle(f"{metrics['label']}: one distributed trajectory per method")
    _save_figure(figure, comparison_dir / "trajectory_comparison")
    return metrics


def analyze_suite(root: Path, dataset_names: list[str]) -> dict:
    datasets = {
        name: analyze_dataset(root / name, name) for name in dataset_names
    }
    comparison_dir = root / "comparison"
    comparison_dir.mkdir(parents=True, exist_ok=True)
    controls_pass = all(
        metrics["controlled_comparison_passed"] for metrics in datasets.values()
    )
    suite = {
        "format": "sim3_covariance_transport_abc_suite",
        "version": 1,
        "controlled_comparison_passed": controls_pass,
        "method_order": list(METHODS),
        "datasets": {
            name: {
                "label": metrics["label"],
                "robot_count": metrics["robot_count"],
                "graph_error": metrics["graph_error"],
                "centralized_explicit_anchor_graph_error": metrics[
                    "centralized_explicit_anchor_graph_error"
                ],
                "relative_improvement_vs_none_percent": metrics[
                    "relative_improvement_vs_none_percent"
                ],
                "winner": metrics["winner"],
                "tied_winners": metrics["tied_winners"],
                "source_sha256": metrics["provenance"]["source_sha256"]["bernoulli"],
            }
            for name, metrics in datasets.items()
        },
    }
    (comparison_dir / "suite_metrics.json").write_text(json.dumps(suite, indent=2) + "\n")
    with (comparison_dir / "suite_summary.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            [
                "dataset",
                "robot_count",
                "none_graph_error",
                "adjoint_graph_error",
                "bernoulli_graph_error",
                "adjoint_improvement_vs_none_percent",
                "bernoulli_improvement_vs_none_percent",
                "winner",
            ]
        )
        for name, metrics in datasets.items():
            writer.writerow(
                [
                    name,
                    metrics["robot_count"],
                    metrics["graph_error"]["none"],
                    metrics["graph_error"]["adjoint"],
                    metrics["graph_error"]["bernoulli"],
                    metrics["relative_improvement_vs_none_percent"]["adjoint"],
                    metrics["relative_improvement_vs_none_percent"]["bernoulli"],
                    metrics["winner"],
                ]
            )

    dataset_x = np.arange(len(dataset_names))
    figure, axis = plt.subplots(figsize=(10.5, 5.8))
    width = 0.25
    for offset, method in zip((-1, 0, 1), METHODS):
        axis.bar(
            dataset_x + offset * width,
            [datasets[name]["graph_error"][method] / datasets[name]["graph_error"]["none"] for name in dataset_names],
            width,
            color=METHOD_COLORS[method],
            label=METHOD_LABELS[method],
        )
    axis.axhline(1.0, color="black", linewidth=0.8)
    axis.set_xticks(dataset_x, [DATASET_LABELS.get(name, name) for name in dataset_names], rotation=12, ha="right")
    axis.set_ylabel("Final graph error / no-transport error")
    axis.set_title("CBS covariance transport across recent experiments")
    axis.grid(axis="y", alpha=0.25)
    axis.legend()
    _save_figure(figure, comparison_dir / "normalized_final_graph_error")

    figure, axis = plt.subplots(figsize=(10.5, 5.8))
    width = 0.34
    bars = []
    for offset, method in zip((-0.5, 0.5), ("adjoint", "bernoulli")):
        method_bars = axis.bar(
            dataset_x + offset * width,
            [
                datasets[name]["relative_improvement_vs_none_percent"][method]
                for name in dataset_names
            ],
            width,
            color=METHOD_COLORS[method],
            label=METHOD_LABELS[method],
        )
        bars.append(method_bars)
    axis.axhline(0.0, color="black", linewidth=0.8)
    for method_bars in bars:
        axis.bar_label(method_bars, fmt="%+.4f%%", padding=3, fontsize=8)
    axis.set_xticks(
        dataset_x,
        [DATASET_LABELS.get(name, name) for name in dataset_names],
        rotation=12,
        ha="right",
    )
    axis.set_ylabel("Final graph-error improvement vs no transport (%)")
    axis.set_title("Covariance transport effect (positive is better)")
    axis.grid(axis="y", alpha=0.25)
    axis.legend()
    _save_figure(figure, comparison_dir / "relative_improvement_vs_none")

    figure, axes = plt.subplots(2, 2, figsize=(13.5, 9.0))
    for axis, name in zip(axes.flat, dataset_names):
        metrics = datasets[name]
        iterations = metrics["convergence"]["iterations"]
        for method in METHODS:
            axis.plot(
                iterations,
                metrics["convergence"]["graph_error"][method],
                color=METHOD_COLORS[method],
                label=METHOD_LABELS[method],
                linewidth=1.1,
            )
        axis.axhline(
            metrics["centralized_explicit_anchor_graph_error"],
            color="black",
            linestyle="--",
            linewidth=0.9,
        )
        axis.set_yscale("log")
        axis.set_title(metrics["label"])
        axis.set_xlabel("Iteration")
        axis.set_ylabel("Graph error (log)")
        axis.grid(alpha=0.2)
    axes.flat[0].legend(ncol=2)
    figure.suptitle("Matched convergence: none vs adjoint vs Bernoulli")
    _save_figure(figure, comparison_dir / "cross_dataset_convergence")

    figure, axes = plt.subplots(2, 2, figsize=(13.5, 9.0))
    for axis, name in zip(axes.flat, dataset_names):
        metrics = datasets[name]
        iterations = metrics["convergence"]["iterations"]
        none_errors = metrics["convergence"]["graph_error"]["none"]
        for method in ("adjoint", "bernoulli"):
            method_errors = metrics["convergence"]["graph_error"][method]
            relative_delta = [
                100.0 * (none_error - method_error) / none_error
                for none_error, method_error in zip(none_errors, method_errors)
            ]
            axis.plot(
                iterations,
                relative_delta,
                color=METHOD_COLORS[method],
                label=METHOD_LABELS[method],
                linewidth=1.1,
            )
        axis.axhline(0.0, color="black", linewidth=0.8)
        axis.set_title(metrics["label"])
        axis.set_xlabel("Iteration")
        axis.set_ylabel("Improvement vs none (%)")
        axis.grid(alpha=0.2)
    axes.flat[0].legend(ncol=2)
    figure.suptitle("Covariance-transport improvement over no transport")
    _save_figure(figure, comparison_dir / "cross_dataset_delta_vs_none")
    return suite


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--datasets", nargs="+", required=True)
    args = parser.parse_args()
    suite = analyze_suite(args.root.expanduser().resolve(), args.datasets)
    print(json.dumps(suite["datasets"], indent=2))
    if not suite["controlled_comparison_passed"]:
        raise SystemExit("controlled-comparison provenance check failed")


if __name__ == "__main__":
    main()
