#!/usr/bin/env python3
"""Summarize and plot a controlled adjoint-versus-Bernoulli CBS experiment."""

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


def _distance(left: dict[str, str], right: dict[str, str]) -> float:
    return math.sqrt(
        sum((float(left[key]) - float(right[key])) ** 2 for key in ("tx", "ty", "tz"))
    )


def _rmse(values: list[float]) -> float:
    return math.sqrt(statistics.fmean(value * value for value in values))


def _normalized_command(command: list[str]) -> list[str]:
    normalized = []
    replacements = {
        "--input_graph=": "--input_graph=<copied-input>",
        "--output_dir=": "--output_dir=<output>",
        "--sim3_covariance_transport=": "--sim3_covariance_transport=<transport>",
    }
    for item in command:
        replacement = next(
            (value for prefix, value in replacements.items() if item.startswith(prefix)),
            None,
        )
        normalized.append(replacement if replacement is not None else item)
    return normalized


def _transport_from_command(command: list[str]) -> str:
    prefix = "--sim3_covariance_transport="
    matches = [item[len(prefix) :] for item in command if item.startswith(prefix)]
    if len(matches) != 1:
        raise ValueError("command must contain exactly one covariance transport flag")
    return matches[0]


def _command_flag(command: list[str], name: str) -> str:
    prefix = f"--{name}="
    matches = [item[len(prefix) :] for item in command if item.startswith(prefix)]
    if len(matches) != 1:
        raise ValueError(f"command must contain exactly one {prefix} flag")
    return matches[0]


def compare(root: Path) -> dict:
    run_dirs = {name: root / name / "dpgo" for name in ("adjoint", "bernoulli")}
    summaries = {name: _rows(path / "summary.csv") for name, path in run_dirs.items()}
    by_solution = {
        name: {row["solution"]: row for row in rows}
        for name, rows in summaries.items()
    }
    for name in ("adjoint", "bernoulli"):
        if "cbs" not in by_solution[name]:
            raise ValueError(f"{name} summary is missing the single CBS trajectory")

    errors = {
        name: float(by_solution[name]["cbs"]["graph_error"])
        for name in ("adjoint", "bernoulli")
    }
    absolute_change = errors["bernoulli"] - errors["adjoint"]
    relative_change = 100.0 * absolute_change / errors["adjoint"]

    comparison_dir = root / "comparison"
    comparison_dir.mkdir(parents=True, exist_ok=True)

    convergence = {
        name: _rows(path / "cbs_convergence.csv") for name, path in run_dirs.items()
    }
    iterations = [int(row["iteration"]) for row in convergence["adjoint"]]
    if iterations != [int(row["iteration"]) for row in convergence["bernoulli"]]:
        raise ValueError("convergence iterations differ")
    convergence_errors = {
        name: [float(row["graph_error"]) for row in convergence[name]]
        for name in ("adjoint", "bernoulli")
    }
    convergence_delta = [
        bernoulli - adjoint
        for adjoint, bernoulli in zip(
            convergence_errors["adjoint"], convergence_errors["bernoulli"]
        )
    ]
    first_crossing = {
        str(threshold): {
            name: next(
                (iteration for iteration, error in zip(iterations, values) if error < threshold),
                None,
            )
            for name, values in convergence_errors.items()
        }
        for threshold in (0.35, 0.32, 0.30, 0.29)
    }
    post_update_errors = {
        name: values[1:] for name, values in convergence_errors.items()
    }
    minimum_checkpoint = {
        name: {
            "iteration": iterations[min(range(len(values)), key=values.__getitem__)],
            "graph_error": min(values),
        }
        for name, values in convergence_errors.items()
    }

    trajectories = {
        name: _trajectory(path / "cbs.csv")
        for name, path in run_dirs.items()
    }
    centralized_trajectory = _trajectory(
        run_dirs["adjoint"] / "centralized_explicit_anchors.csv"
    )
    vertex_ids = sorted(trajectories["adjoint"])
    if vertex_ids != sorted(trajectories["bernoulli"]) or vertex_ids != sorted(
        centralized_trajectory
    ):
        raise ValueError("trajectory vertex sets differ")
    trajectory_comparison = []
    for observer in (f"robot{number}" for number in range(14)):
        robot_vertices = [
            vertex
            for vertex in vertex_ids
            if trajectories["adjoint"][vertex]["robot_id"] == observer
        ]
        adjoint_to_centralized = [
            _distance(trajectories["adjoint"][vertex], centralized_trajectory[vertex])
            for vertex in robot_vertices
        ]
        bernoulli_to_centralized = [
            _distance(trajectories["bernoulli"][vertex], centralized_trajectory[vertex])
            for vertex in robot_vertices
        ]
        between_transports = [
            _distance(trajectories["adjoint"][vertex], trajectories["bernoulli"][vertex])
            for vertex in robot_vertices
        ]
        adjoint_rmse = _rmse(adjoint_to_centralized)
        bernoulli_rmse = _rmse(bernoulli_to_centralized)
        trajectory_comparison.append(
            {
                "robot": observer,
                "pose_count": len(robot_vertices),
                "adjoint_to_centralized_translation_rmse_m": adjoint_rmse,
                "bernoulli_to_centralized_translation_rmse_m": bernoulli_rmse,
                "bernoulli_relative_change_percent": 100.0
                * (bernoulli_rmse - adjoint_rmse)
                / adjoint_rmse,
                "transport_translation_rmse_m": _rmse(between_transports),
                "transport_max_translation_difference_m": max(between_transports),
            }
        )

    provenance = {
        name: json.loads((path / "offline_dpgo_provenance.json").read_text())
        for name, path in run_dirs.items()
    }
    schedule = {
        "stage_mode": _command_flag(provenance["bernoulli"]["command"], "stage_mode"),
        "pose_warmup_iterations": int(
            _command_flag(
                provenance["bernoulli"]["command"], "pose_warmup_iterations"
            )
        ),
        "pose_block_iterations": int(
            _command_flag(
                provenance["bernoulli"]["command"], "pose_block_iterations"
            )
        ),
        "anchor_block_iterations": int(
            _command_flag(
                provenance["bernoulli"]["command"], "anchor_block_iterations"
            )
        ),
        "target_hellinger": float(
            _command_flag(provenance["bernoulli"]["command"], "target_hellinger")
        ),
    }
    centralized_files = ("centralized.csv", "centralized_explicit_anchors.csv")
    centralized_hashes = {
        filename: {name: _sha256(path / filename) for name, path in run_dirs.items()}
        for filename in centralized_files
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
        name: {
            filename: {
                "exists": (path / filename).is_file(),
                "bytes": (path / filename).stat().st_size if (path / filename).is_file() else 0,
            }
            for filename in required_files
        }
        for name, path in run_dirs.items()
    }
    commands_match_except_transport = (
        _normalized_command(provenance["adjoint"]["command"])
        == _normalized_command(provenance["bernoulli"]["command"])
    )
    source_hashes = {name: data["source_sha256"] for name, data in provenance.items()}
    existing_metrics_path = comparison_dir / "comparison_metrics.json"
    existing_metrics = (
        json.loads(existing_metrics_path.read_text())
        if existing_metrics_path.is_file()
        else {}
    )
    recorded_executable_hashes = (
        existing_metrics.get("provenance", {}).get("executable_sha256", {})
    )
    executable_hashes = {}
    for name, data in provenance.items():
        executable = Path(data["command"][0])
        if executable.is_file():
            executable_hashes[name] = _sha256(executable)
        elif name in recorded_executable_hashes:
            executable_hashes[name] = recorded_executable_hashes[name]
        else:
            raise FileNotFoundError(
                f"CBS executable is unavailable and no recorded hash exists: {executable}"
            )
    controls_pass = all(
        all(
            (
            provenance[name].get("status") == "complete",
            provenance[name].get("return_code") == 0,
            provenance[name].get("copied_input_is_byte_identical") is True,
            _transport_from_command(provenance[name]["command"]) == name,
            all(item["exists"] and item["bytes"] > 0 for item in artifacts[name].values()),
            )
        )
        for name in ("adjoint", "bernoulli")
    )
    controls_pass = bool(
        controls_pass
        and len(set(source_hashes.values())) == 1
        and len(set(executable_hashes.values())) == 1
        and commands_match_except_transport
        and all(len(set(value.values())) == 1 for value in centralized_hashes.values())
    )

    centralized_error = float(
        by_solution["adjoint"]["centralized_explicit_anchors"]["graph_error"]
    )
    metrics = {
        "format": "sim3_covariance_transport_ab",
        "version": 2,
        "criterion": "lower pose-graph error is better",
        "controlled_comparison_passed": controls_pass,
        "schedule": schedule,
        "cbs_readout": (
            "each robot's own trajectory transformed to robot0 using that "
            "robot's relative anchor estimate"
        ),
        "centralized_explicit_anchor_graph_error": centralized_error,
        "graph_error": {
            "adjoint": errors["adjoint"],
            "bernoulli": errors["bernoulli"],
            "bernoulli_minus_adjoint": absolute_change,
            "bernoulli_relative_change_percent": relative_change,
            "better": (
                "bernoulli"
                if absolute_change < 0.0
                else "adjoint"
                if absolute_change > 0.0
                else "tie"
            ),
        },
        "convergence": {
            "logged_points": len(iterations),
            "final_iteration": iterations[-1],
            "final_graph_error": {
                name: values[-1] for name, values in convergence_errors.items()
            },
            "minimum_logged_graph_error": {
                name: min(values) for name, values in convergence_errors.items()
            },
            "minimum_checkpoint": minimum_checkpoint,
            "first_iteration_below": first_crossing,
            "mean_graph_error_after_first_update": {
                name: statistics.fmean(values)
                for name, values in post_update_errors.items()
            },
            "bernoulli_lower_points": sum(value < 0.0 for value in convergence_delta),
            "adjoint_lower_points": sum(value > 0.0 for value in convergence_delta),
            "ties": sum(value == 0.0 for value in convergence_delta),
        },
        "distributed_trajectory_vs_centralized": {
            "bernoulli_closer_robot_count": sum(
                row["bernoulli_relative_change_percent"] < 0.0
                for row in trajectory_comparison
            ),
            "adjoint_closer_robot_count": sum(
                row["bernoulli_relative_change_percent"] > 0.0
                for row in trajectory_comparison
            ),
            "per_robot": trajectory_comparison,
        },
        "provenance": {
            "source_sha256": source_hashes,
            "executable_sha256": executable_hashes,
            "commands_match_except_transport": commands_match_except_transport,
            "centralized_output_sha256": centralized_hashes,
            "artifacts": artifacts,
        },
    }
    (comparison_dir / "comparison_metrics.json").write_text(
        json.dumps(metrics, indent=2) + "\n"
    )

    fig, axis = plt.subplots(figsize=(7.5, 5.5))
    bars = axis.bar(
        ["Adjoint", "Bernoulli"],
        [errors["adjoint"], errors["bernoulli"]],
        color=["#1f77b4", "#ff7f0e"],
    )
    axis.axhline(
        centralized_error,
        color="black",
        linestyle="--",
        linewidth=1.1,
        label="Centralized explicit anchors",
    )
    axis.bar_label(bars, fmt="%.6f", padding=3)
    axis.set_ylabel("Pose-graph error (lower is better)")
    axis.set_title("14-robot iPhone CBS: single distributed trajectory")
    axis.grid(axis="y", alpha=0.25)
    axis.legend()
    fig.tight_layout()
    for suffix in ("png", "pdf"):
        fig.savefig(comparison_dir / f"final_graph_error.{suffix}", dpi=200)
    plt.close(fig)

    colors = plt.get_cmap("tab20")
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.4))
    for method_index, method in enumerate(("adjoint", "bernoulli")):
        axis = axes[method_index]
        for robot_number in range(14):
            robot = f"robot{robot_number}"
            robot_vertices = [
                vertex
                for vertex in vertex_ids
                if trajectories[method][vertex]["robot_id"] == robot
            ]
            color = colors(robot_number)
            central_x = [float(centralized_trajectory[v]["tx"]) for v in robot_vertices]
            central_y = [float(centralized_trajectory[v]["ty"]) for v in robot_vertices]
            method_x = [float(trajectories[method][v]["tx"]) for v in robot_vertices]
            method_y = [float(trajectories[method][v]["ty"]) for v in robot_vertices]
            axis.plot(central_x, central_y, color=color, linestyle=":", linewidth=1.0,
                      alpha=0.65)
            axis.plot(method_x, method_y, color=color, linewidth=1.5)
        axis.set_title(f"{method.capitalize()} (solid) vs centralized (dotted)")
        axis.set_xlabel("x (m)")
        axis.set_ylabel("y (m)")
        axis.axis("equal")
        axis.grid(alpha=0.2)
    bar_axis = axes[2]
    robot_x = list(range(14))
    bar_width = 0.4
    bar_axis.bar(
        [value - bar_width / 2 for value in robot_x],
        [row["adjoint_to_centralized_translation_rmse_m"] for row in trajectory_comparison],
        bar_width,
        label="Adjoint",
    )
    bar_axis.bar(
        [value + bar_width / 2 for value in robot_x],
        [row["bernoulli_to_centralized_translation_rmse_m"] for row in trajectory_comparison],
        bar_width,
        label="Bernoulli",
    )
    bar_axis.set_title("Distributed trajectory distance to centralized")
    bar_axis.set_xlabel("Robot")
    bar_axis.set_ylabel("Translation RMSE (m)")
    bar_axis.set_xticks(robot_x, [str(value) for value in robot_x])
    bar_axis.grid(axis="y", alpha=0.2)
    bar_axis.legend()
    fig.suptitle("14-robot joint trajectories (same axes and robot colors)")
    fig.tight_layout()
    for suffix in ("png", "pdf"):
        fig.savefig(comparison_dir / f"trajectory_comparison.{suffix}", dpi=200)
    plt.close(fig)

    magnification = 250.0
    fig, axis = plt.subplots(figsize=(10, 8))
    for robot_number in range(14):
        robot = f"robot{robot_number}"
        robot_vertices = [
            vertex
            for vertex in vertex_ids
            if trajectories["adjoint"][vertex]["robot_id"] == robot
        ]
        color = colors(robot_number)
        adjoint_x = [float(trajectories["adjoint"][v]["tx"]) for v in robot_vertices]
        adjoint_y = [float(trajectories["adjoint"][v]["ty"]) for v in robot_vertices]
        displaced_x = [
            float(trajectories["adjoint"][v]["tx"])
            + magnification
            * (
                float(trajectories["bernoulli"][v]["tx"])
                - float(trajectories["adjoint"][v]["tx"])
            )
            for v in robot_vertices
        ]
        displaced_y = [
            float(trajectories["adjoint"][v]["ty"])
            + magnification
            * (
                float(trajectories["bernoulli"][v]["ty"])
                - float(trajectories["adjoint"][v]["ty"])
            )
            for v in robot_vertices
        ]
        axis.plot(adjoint_x, adjoint_y, color=color, linestyle=":", linewidth=0.9,
                  alpha=0.65)
        axis.plot(displaced_x, displaced_y, color=color, linewidth=1.5, label=robot)
    axis.set_title(
        f"Bernoulli − adjoint position displacement magnified {magnification:.0f}×\n"
        "dotted: adjoint; solid: exaggerated displacement (not true-scale geometry)"
    )
    axis.set_xlabel("x (m)")
    axis.set_ylabel("y (m)")
    axis.axis("equal")
    axis.grid(alpha=0.2)
    axis.legend(ncol=2, fontsize=8, loc="best")
    fig.tight_layout()
    for suffix in ("png", "pdf"):
        fig.savefig(comparison_dir / f"trajectory_difference_250x.{suffix}", dpi=200)
    plt.close(fig)

    fig, (overview, zoom, delta_axis) = plt.subplots(
        3,
        1,
        figsize=(11.5, 9.2),
        sharex=True,
        gridspec_kw={"height_ratios": [1.5, 2.2, 1]},
    )
    zoom_start_iteration = max(first_crossing["0.35"].values())
    zoom_start_index = iterations.index(zoom_start_iteration)
    for name, color in (("adjoint", "#1f77b4"), ("bernoulli", "#ff7f0e")):
        overview.plot(
            iterations,
            convergence_errors[name],
            label=name.capitalize(),
            color=color,
            linewidth=1.3,
        )
        zoom.plot(
            iterations[zoom_start_index:],
            convergence_errors[name][zoom_start_index:],
            label=name.capitalize(),
            color=color,
            linewidth=1.3,
        )
    for axis in (overview, zoom):
        axis.axhline(
            centralized_error,
            color="black",
            linestyle="--",
            linewidth=1.0,
            label="Centralized explicit anchors",
        )
        axis.grid(alpha=0.25)
    overview.set_yscale("log")
    overview.set_ylabel("Graph error (log)")
    if schedule["stage_mode"] == "alternating":
        schedule_label = (
            f'{schedule["pose_block_iterations"]} pose / '
            f'{schedule["anchor_block_iterations"]} anchor rounds'
        )
    else:
        schedule_label = f'{schedule["stage_mode"]} schedule'
    overview.set_title(
        "Matched CBS convergence "
        f'({schedule_label}, fixed Hellinger {schedule["target_hellinger"]:g})'
    )
    overview.legend(ncol=3)
    aligned_values = (
        convergence_errors["adjoint"][zoom_start_index:]
        + convergence_errors["bernoulli"][zoom_start_index:]
    )
    aligned_min = min(aligned_values + [centralized_error])
    aligned_max = max(aligned_values)
    zoom_margin = 0.05 * (aligned_max - aligned_min)
    zoom.set_ylim(aligned_min - zoom_margin, aligned_max + zoom_margin)
    zoom.set_ylabel(f"Graph error (iteration {zoom_start_iteration}+)")
    zoom.legend(ncol=3)
    delta_axis.plot(iterations, convergence_delta, color="#9467bd", linewidth=1.1)
    delta_axis.axhline(0.0, color="black", linewidth=0.8)
    delta_axis.set_ylabel("Bernoulli − adjoint")
    delta_axis.set_xlabel("CBS iteration")
    delta_axis.grid(alpha=0.25)
    fig.tight_layout()
    for suffix in ("png", "pdf"):
        fig.savefig(comparison_dir / f"convergence.{suffix}", dpi=200)
    plt.close(fig)

    if not all(math.isfinite(value) for value in errors.values()):
        raise ValueError("non-finite CBS graph error")
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    metrics = compare(args.root.expanduser().resolve())
    print(json.dumps(metrics["graph_error"], indent=2))
    if not metrics["controlled_comparison_passed"]:
        raise SystemExit("controlled-comparison provenance check failed")


if __name__ == "__main__":
    main()
