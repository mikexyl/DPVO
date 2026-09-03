#!/usr/bin/env python3
"""Analyze the controlled no-reset Hellinger quadratic-term suite."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from compare_sim3_covariance_transport_suite import (
    DATASET_LABELS,
    METHOD_COLORS,
    METHOD_LABELS,
    METHODS,
    _rows,
    analyze_suite,
)


DATASETS = ("iphone14", "kitti00_10_overlap50", "tum", "euroc")
QUADRATIC_MODES = ("quadratic_off", "quadratic_on")


def save_figure(figure: plt.Figure, base: Path) -> None:
    figure.tight_layout()
    for suffix in ("png", "pdf"):
        figure.savefig(base.with_suffix(f".{suffix}"), dpi=200)
    plt.close(figure)


def normalized_cross_quadratic_command(command: list[str]) -> list[str]:
    prefixes = (
        "--input_graph=",
        "--output_dir=",
        "--hellinger_quadratic_term=",
    )
    return [
        next(
            (
                f"{prefix}<controlled>"
                for prefix in prefixes
                if item.startswith(prefix)
            ),
            item,
        )
        for item in command
    ]


def command_flag(command: list[str], name: str) -> str:
    prefix = f"--{name}="
    values = [value[len(prefix) :] for value in command if value.startswith(prefix)]
    if len(values) != 1:
        raise ValueError(f"expected one {prefix} flag")
    return values[0]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--datasets", nargs="+", choices=DATASETS, default=list(DATASETS))
    args = parser.parse_args()
    root = args.root.resolve()
    dataset_names = tuple(args.datasets)

    mode_metrics = {
        mode: analyze_suite(root / mode, list(dataset_names))
        for mode in QUADRATIC_MODES
    }
    comparison = root / "comparison"
    comparison.mkdir(parents=True, exist_ok=True)

    final_errors: dict[str, dict[str, dict[str, float]]] = {}
    improvements: dict[str, dict[str, float]] = {}
    convergence: dict[str, dict[str, dict[str, list[float] | list[int]]]] = {}
    controls = []
    for dataset in dataset_names:
        final_errors[dataset] = {
            mode: mode_metrics[mode]["datasets"][dataset]["graph_error"]
            for mode in QUADRATIC_MODES
        }
        improvements[dataset] = {
            method: 100.0
            * (
                final_errors[dataset]["quadratic_off"][method]
                - final_errors[dataset]["quadratic_on"][method]
            )
            / final_errors[dataset]["quadratic_off"][method]
            for method in METHODS
        }
        convergence[dataset] = {}
        for method in METHODS:
            paths = {
                mode: root / mode / dataset / method / "dpgo"
                for mode in QUADRATIC_MODES
            }
            rows = {mode: _rows(path / "cbs_convergence.csv") for mode, path in paths.items()}
            off_iterations = [int(row["iteration"]) for row in rows["quadratic_off"]]
            on_iterations = [int(row["iteration"]) for row in rows["quadratic_on"]]
            if off_iterations != on_iterations:
                raise ValueError(f"iteration mismatch for {dataset}/{method}")
            convergence[dataset][method] = {
                "iterations": off_iterations,
                "quadratic_off": [
                    float(row["graph_error"]) for row in rows["quadratic_off"]
                ],
                "quadratic_on": [
                    float(row["graph_error"]) for row in rows["quadratic_on"]
                ],
            }

            provenance = {
                mode: json.loads(
                    (paths[mode] / "offline_dpgo_provenance.json").read_text()
                )
                for mode in QUADRATIC_MODES
            }
            off_command = provenance["quadratic_off"]["command"]
            on_command = provenance["quadratic_on"]["command"]
            controls.append(
                provenance["quadratic_off"]["source_sha256"]
                == provenance["quadratic_on"]["source_sha256"]
                and normalized_cross_quadratic_command(off_command)
                == normalized_cross_quadratic_command(on_command)
                and command_flag(off_command, "hellinger_quadratic_term") == "false"
                and command_flag(on_command, "hellinger_quadratic_term") == "true"
                and float(command_flag(off_command, "target_hellinger")) == 0.1
                and float(command_flag(on_command, "target_hellinger")) == 0.1
                and float(command_flag(off_command, "d_reset")) > 1.0
                and float(command_flag(on_command, "d_reset")) > 1.0
            )

    suite = {
        "format": "hellinger_no_reset_quadratic_comparison",
        "version": 1,
        "criterion": "lower pose-graph error is better",
        "controlled_comparison_passed": bool(
            all(controls)
            and all(
                mode_metrics[mode]["controlled_comparison_passed"]
                for mode in QUADRATIC_MODES
            )
        ),
        "configuration": {
            "target_hellinger": 0.1,
            "d_reset": 1.1,
            "reset_effectively_disabled": True,
        },
        "final_graph_error": final_errors,
        "quadratic_improvement_percent": improvements,
    }
    (comparison / "suite_metrics.json").write_text(
        json.dumps(suite, indent=2) + "\n"
    )
    with (comparison / "suite_summary.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            [
                "dataset",
                "transport",
                "quadratic_off_graph_error",
                "quadratic_on_graph_error",
                "quadratic_improvement_percent",
            ]
        )
        for dataset in dataset_names:
            for method in METHODS:
                writer.writerow(
                    [
                        dataset,
                        method,
                        final_errors[dataset]["quadratic_off"][method],
                        final_errors[dataset]["quadratic_on"][method],
                        improvements[dataset][method],
                    ]
                )

    x = np.arange(len(dataset_names))
    figure, axis = plt.subplots(figsize=(11.2, 6.0))
    width = 0.25
    bars = []
    for offset, method in zip((-1, 0, 1), METHODS):
        method_bars = axis.bar(
            x + offset * width,
            [improvements[dataset][method] for dataset in dataset_names],
            width,
            color=METHOD_COLORS[method],
            label=METHOD_LABELS[method],
        )
        bars.append(method_bars)
    axis.axhline(0.0, color="black", linewidth=0.8)
    for method_bars in bars:
        axis.bar_label(method_bars, fmt="%+.4f%%", padding=3, fontsize=7.5)
    axis.set_xticks(
        x,
        [DATASET_LABELS.get(dataset, dataset) for dataset in dataset_names],
        rotation=12,
        ha="right",
    )
    axis.set_ylabel("Improvement from quadratic term (%)")
    axis.set_title("Full Hellinger quadratic term with reset disabled")
    axis.grid(axis="y", alpha=0.25)
    axis.legend(ncol=3)
    save_figure(figure, comparison / "quadratic_improvement")

    figure, axes = plt.subplots(
        len(dataset_names),
        1,
        figsize=(11.5, max(4.8, 3.2 * len(dataset_names))),
        sharex=True,
        squeeze=False,
    )
    for axis, dataset in zip(axes.flat, dataset_names):
        for method in METHODS:
            series = convergence[dataset][method]
            deltas = [
                100.0 * (off - on) / off if off != 0.0 else math.nan
                for off, on in zip(series["quadratic_off"], series["quadratic_on"])
            ]
            axis.plot(
                series["iterations"],
                deltas,
                color=METHOD_COLORS[method],
                label=METHOD_LABELS[method],
                linewidth=1.05,
            )
        axis.axhline(0.0, color="black", linewidth=0.7)
        axis.set_title(DATASET_LABELS.get(dataset, dataset))
        axis.set_ylabel("Quadratic improvement (%)")
        axis.grid(alpha=0.2)
    axes[-1, 0].set_xlabel("CBS iteration")
    axes[0, 0].legend(ncol=3)
    figure.suptitle("Quadratic-term effect throughout CBS convergence")
    save_figure(figure, comparison / "quadratic_convergence_delta")


if __name__ == "__main__":
    main()
