#!/usr/bin/env python3
"""Manifest and completion gates for the staged ScaleMaster library run."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import sys

import numpy as np

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

try:
    from . import newer_college_staged as GATES
except ImportError:
    import newer_college_staged as GATES

from ros2.dpvo_multi_robot.dpvo_multi_robot.scalemaster_core import (
    read_camera_matrix,
    read_frames,
)


FORMAT = "dpvo_scalemaster_library_three_staged"
VERSION = 1


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_record(path_value: str | Path) -> dict:
    path = Path(path_value).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    return {
        "path": str(path),
        "size_bytes": path.stat().st_size,
        "sha256": sha256(path),
    }


def labeled(values: list[str], option: str) -> dict[str, str]:
    result = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"{option} expects ROBOT=VALUE, got {value!r}")
        robot, item = value.split("=", 1)
        if not robot or not item or robot in result:
            raise ValueError(f"invalid or duplicate {option} value: {value!r}")
        result[robot] = item
    return result


def decode_parameter(value: str):
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return value


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def robot_sort_key(robot: str):
    suffix = robot.removeprefix("robot")
    return (0, int(suffix)) if suffix.isdigit() else (1, robot)


def sequence_record(path_value: str | Path) -> dict:
    root = Path(path_value).expanduser().resolve()
    frames = read_frames(root)
    matrix = read_camera_matrix(root / "camera_matrix.csv")
    inventory = hashlib.sha256()
    total_bytes = 0
    for frame in frames:
        size = frame.image_path.stat().st_size
        total_bytes += size
        inventory.update(f"{frame.image_path.name}\0{size}\n".encode())
    duration = frames[-1].timestamp - frames[0].timestamp
    return {
        "path": str(root),
        "frame_count": len(frames),
        "first_frame_id": frames[0].frame_id,
        "last_frame_id": frames[-1].frame_id,
        "first_timestamp": frames[0].timestamp,
        "last_timestamp": frames[-1].timestamp,
        "nominal_rate_hz": (len(frames) - 1) / duration,
        "frames_total_bytes": total_bytes,
        "frame_inventory_sha256": inventory.hexdigest(),
        "camera_matrix": file_record(root / "camera_matrix.csv"),
        "camera_intrinsics": {
            "fx": float(matrix[0, 0]),
            "fy": float(matrix[1, 1]),
            "cx": float(matrix[0, 2]),
            "cy": float(matrix[1, 2]),
        },
        "timestamp_odometry": file_record(root / "odometry.csv"),
    }


def record_manifest(args) -> None:
    run_dir = args.run_dir.expanduser().resolve()
    path = run_dir / "experiment_manifest.json"
    sources = labeled(args.source, "--source")
    robots = sorted(sources, key=robot_sort_key)
    if robots != ["robot0", "robot1", "robot2"]:
        raise ValueError("ScaleMaster library experiment requires robot0..robot2")
    source_records = {
        robot: {"sequence": sequence_record(source)}
        for robot, source in sources.items()
    }
    immutable_inputs = {
        "orb_vocabulary": file_record(args.vocabulary),
        "dpvo_network": file_record(args.network),
        "dpvo_config": file_record(args.config),
    }
    parameters = {
        key: decode_parameter(value)
        for key, value in labeled(args.parameter, "--parameter").items()
    }

    if path.is_file():
        manifest = json.loads(path.read_text(encoding="utf-8"))
        if manifest.get("format") != FORMAT or manifest.get("version") != VERSION:
            raise ValueError(f"refusing to update incompatible manifest: {path}")
    else:
        manifest = {
            "format": FORMAT,
            "version": VERSION,
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "dataset": "ScaleMaster Library 01/02/03",
            "dataset_reference": {
                "project": "https://scalemaster-dataset.github.io/",
                "repository": "https://github.com/JooHyoSeok/ScaleMaster-Dataset",
                "paper": "https://arxiv.org/abs/2602.18174",
            },
            "evaluation": {
                "groundtruth_available": False,
                "scope": "no_ground_truth",
                "note": (
                    "ScaleMaster odometry supplies playback timestamps only and is "
                    "not treated as ground truth. Report solver diagnostics without "
                    "ATE or ground-truth overlays."
                ),
            },
            "tracking": {
                "directory": str(run_dir / "tracking"),
                "sources": {},
                "robot_ids": robots,
            },
            "outputs": {
                name: str(run_dir / name / "three")
                for name in (
                    "geometric_verification",
                    "dpgo",
                    "evaluation",
                    "plots",
                )
            },
        }

    tracking_exists = any((run_dir / "tracking").glob("robot*/manifest.json"))
    if tracking_exists:
        for name, current in immutable_inputs.items():
            previous = manifest.get(name)
            if previous and previous.get("sha256") != current["sha256"]:
                raise RuntimeError(f"immutable tracking artifacts use another {name}")
        for key, value in parameters.items():
            previous = manifest.get("parameters", {}).get(key)
            if key.startswith("tracking.") and previous is not None and previous != value:
                raise RuntimeError(f"immutable artifacts use another {key}")
        for robot, current in source_records.items():
            previous = manifest.get("tracking", {}).get("sources", {}).get(robot)
            if previous and (
                previous["sequence"]["frame_inventory_sha256"]
                != current["sequence"]["frame_inventory_sha256"]
            ):
                raise RuntimeError(f"immutable {robot} frame source changed")

    manifest.update(immutable_inputs)
    manifest["evaluation"] = {
        "groundtruth_available": False,
        "scope": "no_ground_truth",
        "note": (
            "ScaleMaster odometry supplies playback timestamps only and is not "
            "treated as ground truth."
        ),
    }
    manifest["outputs"] = {
        name: str(run_dir / name / "three")
        for name in ("geometric_verification", "dpgo", "evaluation", "plots")
    }
    manifest["parameters"] = parameters
    manifest["tracking"]["sources"] = source_records
    manifest["updated_utc"] = datetime.now(timezone.utc).isoformat()
    atomic_json(path, manifest)
    print(path)


def _solver_rows(path: Path) -> dict[int, tuple[np.ndarray, np.ndarray, float]]:
    rows = {}
    with path.open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            vertex = int(row["vertex_id"])
            translation = np.asarray([row["tx"], row["ty"], row["tz"]], dtype=float)
            quaternion = np.asarray(
                [row["qx"], row["qy"], row["qz"], row["qw"]], dtype=float
            )
            norm = float(np.linalg.norm(quaternion))
            scale = float(row["scale"])
            if (
                vertex in rows
                or not np.isfinite(translation).all()
                or not np.isfinite(norm)
                or norm < 1e-12
                or not math.isfinite(scale)
                or scale <= 0.0
            ):
                raise ValueError(f"invalid solver row for vertex {vertex} in {path}")
            rows[vertex] = (translation, quaternion / norm, scale)
    if not rows:
        raise RuntimeError(f"no solver rows in {path}")
    return rows


def solver_agreement(reference_path: Path, candidate_path: Path) -> dict:
    reference = _solver_rows(reference_path)
    candidate = _solver_rows(candidate_path)
    if set(reference) != set(candidate):
        raise ValueError("centralized and CBS vertex sets differ")
    translation_errors = []
    rotation_errors = []
    log_scale_errors = []
    for vertex in sorted(reference):
        ref_t, ref_q, ref_s = reference[vertex]
        cand_t, cand_q, cand_s = candidate[vertex]
        translation_errors.append(float(np.linalg.norm(ref_t - cand_t)))
        cosine = float(np.clip(abs(np.dot(ref_q, cand_q)), 0.0, 1.0))
        rotation_errors.append(2.0 * math.acos(cosine))
        log_scale_errors.append(abs(math.log(cand_s / ref_s)))
    return {
        "reference": str(reference_path.resolve()),
        "candidate": str(candidate_path.resolve()),
        "vertex_count": len(reference),
        "translation_rmse_local_units": float(
            np.sqrt(np.mean(np.square(translation_errors)))
        ),
        "translation_max_local_units": max(translation_errors),
        "rotation_rmse_rad": float(np.sqrt(np.mean(np.square(rotation_errors)))),
        "rotation_max_rad": max(rotation_errors),
        "log_scale_rmse": float(np.sqrt(np.mean(np.square(log_scale_errors)))),
        "log_scale_max": max(log_scale_errors),
    }


def solver_inventory(path: Path) -> dict:
    validated = _solver_rows(path)
    per_robot = {}
    with path.open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            robot = row["robot_id"]
            per_robot[robot] = per_robot.get(robot, 0) + 1
    return {
        "artifact": file_record(path),
        "vertex_count": len(validated),
        "per_robot_vertex_count": dict(sorted(per_robot.items())),
    }


def collect_metrics(args) -> None:
    optimized_robots = args.optimized_robot_ids or args.robot_ids
    if not set(optimized_robots) <= set(args.robot_ids):
        raise ValueError("optimized robot IDs must be a subset of tracked robot IDs")
    passthrough_robots = [
        robot for robot in args.robot_ids if robot not in set(optimized_robots)
    ]
    methods = {
        method: solver_inventory(args.dpgo_dir / f"{method}.csv")
        for method in ("centralized", "centralized_explicit_anchors", "cbs")
    }
    for method, inventory in methods.items():
        if list(inventory["per_robot_vertex_count"]) != optimized_robots:
            raise ValueError(f"{method} robot set does not match optimized component")
    metrics = {
        "format": "dpvo_scalemaster_no_ground_truth_metrics",
        "version": 1,
        "robot_ids": args.robot_ids,
        "optimized_robot_ids": optimized_robots,
        "passthrough_robot_ids": passthrough_robots,
        "component_policy": "optimize_connected_overlap_component",
        "evaluation_scope": "no_ground_truth",
        "absolute_accuracy_available": False,
        "ate_reported": False,
        "methods": methods,
        "solver_summary": file_record(args.dpgo_dir / "summary.csv"),
        "solver_agreement": solver_agreement(
            args.dpgo_dir / "centralized_explicit_anchors.csv",
            args.dpgo_dir / "cbs.csv",
        ),
    }
    atomic_json(args.output, metrics)
    print(json.dumps(metrics, indent=2))


def final_gate(args) -> None:
    output = args.output.expanduser().resolve()
    try:
        stage1 = json.loads(args.stage1_gate.read_text(encoding="utf-8"))
        graph = json.loads(args.graph_gate.read_text(encoding="utf-8"))
        metrics = json.loads(args.metrics.read_text(encoding="utf-8"))
        if not stage1.get("pass") or not graph.get("pass"):
            raise RuntimeError("stage-one or verified-graph gate did not pass")
        if stage1.get("robot_ids") != args.robot_ids:
            raise RuntimeError("stage-one robot set does not match")
        if graph.get("robots") != args.robot_ids:
            raise RuntimeError("verified-graph robot set does not match")
        if metrics.get("robot_ids") != args.robot_ids:
            raise RuntimeError("metrics robot set does not match")
        if metrics.get("optimized_robot_ids") != args.optimized_robot_ids:
            raise RuntimeError("metrics optimized robot set does not match")
        expected_passthrough = [
            robot for robot in args.robot_ids if robot not in args.optimized_robot_ids
        ]
        if metrics.get("passthrough_robot_ids") != expected_passthrough:
            raise RuntimeError("metrics passthrough robot set does not match")
        if metrics.get("evaluation_scope") != "no_ground_truth":
            raise RuntimeError("ScaleMaster metrics use an invalid evaluation scope")
        if metrics.get("absolute_accuracy_available") is not False:
            raise RuntimeError("ScaleMaster metrics incorrectly claim ground truth")
        if metrics.get("ate_reported") is not False:
            raise RuntimeError("ScaleMaster metrics incorrectly report ATE")

        if graph.get("components") != [args.optimized_robot_ids, expected_passthrough]:
            raise RuntimeError("verified graph does not have the expected place topology")
        component_gate = json.loads(
            args.component_graph_gate.read_text(encoding="utf-8")
        )
        if not component_gate.get("pass") or not component_gate.get("connected"):
            raise RuntimeError("optimized robot component gate did not pass")
        if component_gate.get("robots") != args.optimized_robot_ids:
            raise RuntimeError("optimized component robot set does not match")
        component = json.loads(
            args.component_provenance.read_text(encoding="utf-8")
        )
        if component.get("robot_ids") != args.optimized_robot_ids:
            raise RuntimeError("component provenance robot set does not match")

        provenance_path = args.dpgo_dir / "offline_dpgo_provenance.json"
        provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
        if provenance.get("status") != "complete" or provenance.get("return_code") != 0:
            raise RuntimeError("stage-three solver did not exit successfully")
        if not provenance.get("copied_input_is_byte_identical"):
            raise RuntimeError("stage-three input is not byte-identical to stage two")
        copied = args.dpgo_dir / "input_keyframes_unoptimized.json"
        if sha256(args.component_graph) != sha256(copied):
            raise RuntimeError("stage-two component and stage-three input differ")

        required = {
            "verified_graph_json": GATES.require_file(
                args.verified_graph, "verified JSON graph"
            ),
            "verified_graph_g2o": GATES.require_file(
                args.verified_graph.with_suffix(".g2o"), "verified G2O graph"
            ),
            "component_graph_json": GATES.require_file(
                args.component_graph, "optimized component JSON graph"
            ),
            "component_graph_g2o": GATES.require_file(
                args.component_graph.with_suffix(".g2o"),
                "optimized component G2O graph",
            ),
            "component_provenance": GATES.require_file(
                args.component_provenance, "component provenance"
            ),
            "centralized_csv": GATES.require_file(
                args.dpgo_dir / "centralized.csv", "centralized CSV"
            ),
            "explicit_centralized_csv": GATES.require_file(
                args.dpgo_dir / "centralized_explicit_anchors.csv",
                "explicit-anchor centralized CSV",
            ),
            "cbs_csv": GATES.require_file(args.dpgo_dir / "cbs.csv", "CBS CSV"),
            "dpgo_rrd": GATES.require_file(
                args.dpgo_dir / "dpvo_sim3_cbs.rrd", "DPGO RRD"
            ),
            "metrics": GATES.require_file(args.metrics, "metrics JSON"),
            "provenance": GATES.require_file(provenance_path, "DPGO provenance"),
        }
        for method in ("centralized", "centralized_explicit_anchors", "cbs"):
            actual = required[
                "explicit_centralized_csv"
                if method == "centralized_explicit_anchors"
                else f"{method}_csv"
            ]
            expected = metrics["methods"][method]["artifact"]
            if actual["sha256"] != expected["sha256"]:
                raise RuntimeError(f"solver artifact changed for {method}")
        plots = {}
        if args.plots_dir is not None:
            names = (
                "scalemaster_component_trajectories_loops.png",
                "scalemaster_component_trajectories_loops.pdf",
                "cbs_sparse_joint_map.rrd",
                "cbs_sparse_joint_map.ply",
                "cbs_sparse_joint_map.json",
                "centralized_sparse_joint_map.rrd",
                "centralized_sparse_joint_map.ply",
                "centralized_sparse_joint_map.json",
            )
            plots = {
                name: GATES.require_file(args.plots_dir / name, f"plot artifact {name}")
                for name in names
            }
        result = {
            "pass": True,
            "robot_ids": args.robot_ids,
            "optimized_robot_ids": args.optimized_robot_ids,
            "passthrough_robot_ids": expected_passthrough,
            "component_policy": "optimize_connected_overlap_component",
            "evaluation_scope": metrics["evaluation_scope"],
            "absolute_accuracy_available": False,
            "ate_reported": False,
            "solver_agreement": metrics["solver_agreement"],
            "required_artifacts": required,
            "plots": plots,
        }
    except Exception as error:
        atomic_json(
            output, {"pass": False, "robot_ids": args.robot_ids, "error": str(error)}
        )
        raise
    atomic_json(output, result)
    print(json.dumps(result, indent=2))


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description=__doc__)
    commands = root.add_subparsers(dest="command", required=True)

    manifest = commands.add_parser("record-manifest")
    manifest.add_argument("--run-dir", type=Path, required=True)
    manifest.add_argument("--source", action="append", required=True)
    manifest.add_argument("--vocabulary", type=Path, required=True)
    manifest.add_argument("--network", type=Path, required=True)
    manifest.add_argument("--config", type=Path, required=True)
    manifest.add_argument("--parameter", action="append", default=[])
    manifest.set_defaults(function=record_manifest)

    metrics = commands.add_parser("collect-metrics")
    metrics.add_argument("--dpgo-dir", type=Path, required=True)
    metrics.add_argument("--robot-ids", nargs="+", required=True)
    metrics.add_argument("--optimized-robot-ids", nargs="+")
    metrics.add_argument("--output", type=Path, required=True)
    metrics.set_defaults(function=collect_metrics)

    final = commands.add_parser("final-gate")
    final.add_argument("--stage1-gate", type=Path, required=True)
    final.add_argument("--graph-gate", type=Path, required=True)
    final.add_argument("--verified-graph", type=Path, required=True)
    final.add_argument("--component-graph", type=Path, required=True)
    final.add_argument("--component-provenance", type=Path, required=True)
    final.add_argument("--component-graph-gate", type=Path, required=True)
    final.add_argument("--dpgo-dir", type=Path, required=True)
    final.add_argument("--metrics", type=Path, required=True)
    final.add_argument("--plots-dir", type=Path)
    final.add_argument("--robot-ids", nargs="+", required=True)
    final.add_argument("--optimized-robot-ids", nargs="+", required=True)
    final.add_argument("--output", type=Path, required=True)
    final.set_defaults(function=final_gate)
    return root


def main(argv=None) -> None:
    args = parser().parse_args(argv)
    try:
        args.function(args)
    except (FileNotFoundError, KeyError, RuntimeError, TypeError, ValueError) as error:
        raise SystemExit(f"{args.command} failed: {error}") from error


if __name__ == "__main__":
    main()
