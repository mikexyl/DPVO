#!/usr/bin/env python3
"""Manifest and preflight gates for the staged GrAco Aerial 5--8 run."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path


FORMAT = "dpvo_graco_aerial_5_8_staged"
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


def bag_record(path_value: str | Path) -> dict:
    root = Path(path_value).expanduser().resolve()
    if not root.is_dir():
        raise NotADirectoryError(root)
    metadata = root / "metadata.yaml"
    databases = sorted(root.glob("*.db3"))
    if len(databases) != 1:
        raise RuntimeError(f"expected one db3 under {root}, found {len(databases)}")
    return {
        "path": str(root),
        "metadata": file_record(metadata),
        "database": file_record(databases[0]),
    }


def labeled(values: list[str], option: str) -> dict[str, str]:
    result = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"{option} expects ROBOT=VALUE, got {value!r}")
        robot, item = value.split("=", 1)
        if not robot or not item:
            raise ValueError(f"{option} expects non-empty ROBOT=VALUE")
        if robot in result:
            raise ValueError(f"duplicate {option} robot: {robot}")
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


def record_manifest(args) -> None:
    run_dir = args.run_dir.expanduser().resolve()
    path = run_dir / "experiment_manifest.json"
    source_bags = labeled(args.source_bag, "--source-bag")
    bags = labeled(args.bag, "--bag")
    groundtruth = labeled(args.groundtruth, "--groundtruth")
    if set(source_bags) != set(bags) or set(source_bags) != set(groundtruth):
        raise ValueError("source-bag, bag, and ground-truth robot sets must match")
    robots = sorted(source_bags, key=robot_sort_key)
    if robots != ["robot0", "robot1", "robot2", "robot3"]:
        raise ValueError(f"Aerial 5--8 requires robot0..robot3, got {robots}")
    source_records = {
        robot: {
            "authoritative_ros1_bag": file_record(source_bags[robot]),
            "converted_full_ros2_bag": bag_record(bags[robot]),
            "imu_groundtruth": file_record(groundtruth[robot]),
        }
        for robot in robots
    }
    immutable_inputs = {
        "orb_vocabulary": file_record(args.vocabulary),
        "dpvo_network": file_record(args.network),
        "dpvo_config": file_record(args.config),
        "stereo_calibration": file_record(args.stereo_calibration),
        "stereo_imu_calibration": file_record(args.stereo_imu_calibration),
        "groundtruth_calibration": file_record(args.groundtruth_calibration),
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
            "material_passport": {
                "origin_skill": "academic-research-suite/experiment-agent",
                "origin_mode": "run",
                "origin_date": "2026-09-02",
                "version_label": "graco_aerial_5_8_staged_v1",
                "verification_status": "execution_in_progress",
            },
            "dataset": "GrAco Aerial 5--8",
            "dataset_reference": {
                "repository": "https://github.com/SYSU-RoboticsLab/GrAco",
                "project": "https://sites.google.com/view/graco-dataset",
                "paper_doi": "10.1109/LRA.2023.3234790",
            },
            "camera": {
                "sensor": "FLIR BFS-U3-16S7C left camera (cam0)",
                "source_encoding": "mono8",
                "source_resolution": [1600, 1100],
                "source_rate_hz": 20.0,
                "source_distortion_model": "plumb_bob/radial-tangential",
                "published_encoding": "mono8",
                "published_distortion_model": "plumb_bob (zero residual distortion)",
                "rectification_intrinsics_source": "ROS2 CameraInfo",
            },
            "tracking": {
                "directory": str(run_dir / "tracking"),
                "sources": {},
            },
            "scenarios": {},
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
                previous["authoritative_ros1_bag"]["sha256"]
                != current["authoritative_ros1_bag"]["sha256"]
                or previous["converted_full_ros2_bag"]["database"]["sha256"]
                != current["converted_full_ros2_bag"]["database"]["sha256"]
            ):
                raise RuntimeError(f"immutable {robot} artifact source changed")

    manifest.update(immutable_inputs)
    manifest["parameters"] = parameters
    manifest["tracking"]["sources"].update(source_records)
    manifest["scenarios"]["four"] = {
        "robot_ids": robots,
        "outputs": {
            name: str(run_dir / name / "four")
            for name in ("geometric_verification", "dpgo", "evo", "plots")
        },
    }
    manifest["updated_utc"] = datetime.now(timezone.utc).isoformat()
    atomic_json(path, manifest)
    print(path)


def check_preflight_rate(args) -> None:
    artifact = args.artifact.expanduser().resolve()
    manifest = json.loads((artifact / "manifest.json").read_text(encoding="utf-8"))
    input_frames = int(manifest.get("input_frame_count", 0))
    keyframes = int(manifest.get("keyframe_count", 0))
    if not manifest.get("complete") or input_frames < 2 or keyframes <= 0:
        raise RuntimeError(f"preflight artifact is incomplete or empty: {artifact}")
    duration = (input_frames - 1) * args.stride / args.source_rate_hz
    rate = keyframes / duration
    passed = args.minimum <= rate <= args.maximum
    result = {
        "format": "dpvo_graco_preflight_gate",
        "version": 1,
        "passed": passed,
        "artifact": str(artifact),
        "input_frame_count": input_frames,
        "keyframe_count": keyframes,
        "estimated_duration_seconds": duration,
        "keyframe_rate_hz": rate,
        "accepted_keyframe_rate_hz": [args.minimum, args.maximum],
        "source_rate_hz": args.source_rate_hz,
        "stride": args.stride,
    }
    atomic_json(args.output.expanduser().resolve(), result)
    print(json.dumps(result, indent=2))
    if not passed:
        raise RuntimeError(
            f"preflight keyframe rate {rate:.3f} Hz is outside "
            f"[{args.minimum:.3f}, {args.maximum:.3f}] Hz"
        )


def check_preflight_input_rate(args) -> None:
    artifact = args.artifact.expanduser().resolve()
    manifest = json.loads((artifact / "manifest.json").read_text(encoding="utf-8"))
    input_frames = int(manifest.get("input_frame_count", 0))
    keyframes = int(manifest.get("keyframe_count", 0))
    if not manifest.get("complete") or input_frames < 2 or keyframes <= 0:
        raise RuntimeError(f"preflight artifact is incomplete or empty: {artifact}")
    duration = (input_frames - 1) * args.stride / args.source_rate_hz
    input_rate = args.source_rate_hz / args.stride
    retained_keyframe_rate = keyframes / duration
    passed = args.minimum <= input_rate <= args.maximum
    result = {
        "format": "dpvo_graco_preflight_input_gate",
        "version": 1,
        "passed": passed,
        "artifact": str(artifact),
        "input_frame_count": input_frames,
        "keyframe_count": keyframes,
        "estimated_duration_seconds": duration,
        "input_rate_hz": input_rate,
        "retained_keyframe_rate_hz": retained_keyframe_rate,
        "accepted_input_rate_hz": [args.minimum, args.maximum],
        "source_rate_hz": args.source_rate_hz,
        "stride": args.stride,
    }
    atomic_json(args.output.expanduser().resolve(), result)
    print(json.dumps(result, indent=2))
    if not passed:
        raise RuntimeError(
            f"preflight input rate {input_rate:.3f} Hz is outside "
            f"[{args.minimum:.3f}, {args.maximum:.3f}] Hz"
        )


def component_gate(args) -> None:
    output = args.output.expanduser().resolve()
    try:
        full_gate = json.loads(args.full_graph_gate.read_text(encoding="utf-8"))
        graph_gate = json.loads(args.component_graph_gate.read_text(encoding="utf-8"))
        subset = json.loads(args.component_provenance.read_text(encoding="utf-8"))
        metrics = json.loads(args.metrics.read_text(encoding="utf-8"))
        provenance_path = args.dpgo_dir / "offline_dpgo_provenance.json"
        provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
        if full_gate.get("pass") or full_gate.get("connected"):
            raise RuntimeError("full graph is not the recorded disconnected case")
        if not graph_gate.get("pass") or not graph_gate.get("connected"):
            raise RuntimeError("selected robot component is not connected")
        if graph_gate.get("robots") != args.robot_ids:
            raise RuntimeError("component graph robot set does not match")
        if subset.get("robot_ids") != args.robot_ids:
            raise RuntimeError("component provenance robot set does not match")
        anchor_robot = args.robot_ids[0]
        if subset.get("anchor_robot_id") != anchor_robot:
            raise RuntimeError("component anchor is not the first selected robot")
        if provenance.get("status") != "complete" or provenance.get("return_code") != 0:
            raise RuntimeError("component DPGO solver did not exit successfully")
        if not provenance.get("copied_input_is_byte_identical"):
            raise RuntimeError("component DPGO input copy is not byte-identical")
        graph_sha = sha256(args.verified_graph)
        copied_graph = args.dpgo_dir / "input_keyframes_unoptimized.json"
        if graph_sha != sha256(copied_graph):
            raise RuntimeError("component graph and DPGO input differ")
        if provenance.get("source_sha256") != graph_sha:
            raise RuntimeError("component graph checksum differs from DPGO provenance")
        if metrics.get("robot_ids") != args.robot_ids:
            raise RuntimeError("component metrics robot set does not match")

        cbs_method = "cbs"
        methods = (
            "centralized",
            "centralized_explicit_anchors",
            cbs_method,
        )
        for method in methods:
            for scope in (*args.robot_ids, "joint"):
                metric = metrics["methods"][method][scope]
                ate = float(metric["ate_rmse_m"])
                scale = float(metric["groundtruth_relative_scale"])
                if not math.isfinite(ate) or ate < 0.0:
                    raise RuntimeError(f"invalid {method}/{scope} ATE")
                if not math.isfinite(scale) or scale <= 0.0:
                    raise RuntimeError(f"invalid {method}/{scope} scale")
                archive = args.evo_dir / "results" / f"{method}_{scope}.zip"
                if file_record(archive)["sha256"] != metric["archive"]["sha256"]:
                    raise RuntimeError(f"evo archive checksum differs: {method}/{scope}")

        central_ate = float(
            metrics["methods"]["centralized_explicit_anchors"]["joint"][
                "ate_rmse_m"
            ]
        )
        cbs_ate = float(metrics["methods"][cbs_method]["joint"]["ate_rmse_m"])
        ratio = cbs_ate / central_ate if central_ate > 0.0 else math.inf
        if ratio > 1.25:
            raise RuntimeError("CBS joint ATE exceeds 125% of centralized ATE")

        required_paths = {
            "component_graph_json": args.verified_graph,
            "component_graph_g2o": args.verified_graph.with_suffix(".g2o"),
            "component_provenance": args.component_provenance,
            "dpgo_provenance": provenance_path,
            "dpgo_rrd": args.dpgo_dir / "dpvo_sim3_cbs.rrd",
            "centralized_csv": args.dpgo_dir / "centralized.csv",
            "explicit_anchor_csv": (
                args.dpgo_dir / "centralized_explicit_anchors.csv"
            ),
            "cbs_csv": args.dpgo_dir / f"{cbs_method}.csv",
            "metrics": args.metrics,
            "trajectory_png": (
                args.plots_dir / "graco_component_trajectories_loops.png"
            ),
            "trajectory_pdf": (
                args.plots_dir / "graco_component_trajectories_loops.pdf"
            ),
            "sparse_map_png": (
                args.plots_dir / "graco_component_sparse_joint_map_alignment.png"
            ),
            "sparse_map_pdf": (
                args.plots_dir / "graco_component_sparse_joint_map_alignment.pdf"
            ),
            "cbs_sparse_map_rrd": args.plots_dir / "cbs_sparse_joint_map.rrd",
            "centralized_sparse_map_rrd": (
                args.plots_dir / "centralized_sparse_joint_map.rrd"
            ),
        }
        required = {name: file_record(path) for name, path in required_paths.items()}
        result = {
            "pass": True,
            "policy": "optimize_each_connected_component",
            "full_graph_connected": False,
            "isolated_robots": sorted(
                set(full_gate.get("robots", [])) - set(args.robot_ids),
                key=robot_sort_key,
            ),
            "robot_ids": args.robot_ids,
            "anchor_robot_id": anchor_robot,
            "inter_robot_loop_count": int(graph_gate["inter_robot_loop_count"]),
            "centralized_explicit_anchor_joint_ate_rmse_m": central_ate,
            "cbs_joint_ate_rmse_m": cbs_ate,
            "cbs_to_centralized_ate_ratio": ratio,
            "accuracy_note": (
                "computational gate passed; high joint ATE indicates poor "
                "cross-flight registration"
            ),
            "required_artifacts": required,
        }
    except Exception as error:
        result = {"pass": False, "robot_ids": args.robot_ids, "error": str(error)}
        atomic_json(output, result)
        raise
    atomic_json(output, result)
    print(json.dumps(result, indent=2))


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description=__doc__)
    subparsers = root.add_subparsers(dest="command", required=True)
    manifest = subparsers.add_parser("manifest")
    manifest.add_argument("--run-dir", type=Path, required=True)
    manifest.add_argument("--source-bag", action="append", required=True)
    manifest.add_argument("--bag", action="append", required=True)
    manifest.add_argument("--groundtruth", action="append", required=True)
    manifest.add_argument("--vocabulary", type=Path, required=True)
    manifest.add_argument("--network", type=Path, required=True)
    manifest.add_argument("--config", type=Path, required=True)
    manifest.add_argument("--stereo-calibration", type=Path, required=True)
    manifest.add_argument("--stereo-imu-calibration", type=Path, required=True)
    manifest.add_argument("--groundtruth-calibration", type=Path, required=True)
    manifest.add_argument("--parameter", action="append", default=[])
    manifest.set_defaults(handler=record_manifest)

    preflight = subparsers.add_parser("preflight-rate")
    preflight.add_argument("--artifact", type=Path, required=True)
    preflight.add_argument("--source-rate-hz", type=float, required=True)
    preflight.add_argument("--stride", type=int, required=True)
    preflight.add_argument("--minimum", type=float, default=3.0)
    preflight.add_argument("--maximum", type=float, default=5.0)
    preflight.add_argument("--output", type=Path, required=True)
    preflight.set_defaults(handler=check_preflight_rate)

    input_rate = subparsers.add_parser("preflight-input-rate")
    input_rate.add_argument("--artifact", type=Path, required=True)
    input_rate.add_argument("--source-rate-hz", type=float, required=True)
    input_rate.add_argument("--stride", type=int, required=True)
    input_rate.add_argument("--minimum", type=float, default=3.0)
    input_rate.add_argument("--maximum", type=float, default=5.0)
    input_rate.add_argument("--output", type=Path, required=True)
    input_rate.set_defaults(handler=check_preflight_input_rate)

    component = subparsers.add_parser("component-gate")
    component.add_argument("--full-graph-gate", type=Path, required=True)
    component.add_argument("--component-graph-gate", type=Path, required=True)
    component.add_argument("--component-provenance", type=Path, required=True)
    component.add_argument("--verified-graph", type=Path, required=True)
    component.add_argument("--dpgo-dir", type=Path, required=True)
    component.add_argument("--evo-dir", type=Path, required=True)
    component.add_argument("--plots-dir", type=Path, required=True)
    component.add_argument("--metrics", type=Path, required=True)
    component.add_argument("--robot-ids", nargs="+", required=True)
    component.add_argument("--output", type=Path, required=True)
    component.set_defaults(handler=component_gate)
    return root


def main(argv=None) -> None:
    args = parser().parse_args(argv)
    try:
        args.handler(args)
    except (FileNotFoundError, NotADirectoryError, RuntimeError, ValueError) as error:
        raise SystemExit(f"{args.command} failed: {error}") from error


if __name__ == "__main__":
    main()
