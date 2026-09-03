#!/usr/bin/env python3
"""Validation, provenance, and acceptance gates for Newer College staging."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import io
import json
import math
from pathlib import Path
import sys
import zipfile

import numpy as np

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from dpvo.loop_closure.pose_graph import read_json
from dpvo.loop_closure.tracking_artifact import (
    load_artifact_root,
    load_tracking_artifact,
)
from ros2.dpvo_multi_robot.dpvo_multi_robot.newer_college_core import (
    build_fisheye_rectification,
    decode_compressed_mono8,
    read_kalibr_camera_calibration,
    stamp_components,
)


FORMAT = "dpvo_newer_college_quad_staged"
VERSION = 1


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def labeled_values(values: list[str] | None, option: str) -> dict[str, str]:
    parsed = {}
    for value in values or []:
        if "=" not in value:
            raise ValueError(f"{option} expects LABEL=VALUE, got {value!r}")
        label, item = value.split("=", 1)
        if not label or not item:
            raise ValueError(f"{option} expects non-empty LABEL=VALUE")
        if label in parsed:
            raise ValueError(f"duplicate {option} label: {label}")
        parsed[label] = item
    return parsed


def file_record(path_value: str) -> dict:
    path = Path(path_value).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    return {
        "path": str(path),
        "size_bytes": path.stat().st_size,
        "sha256": sha256(path),
    }


def decode_parameter(value: str):
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return value


def record_manifest(args) -> None:
    run_dir = args.run_dir.expanduser().resolve()
    path = run_dir / "experiment_manifest.json"
    sources = labeled_values(args.source, "--source")
    groundtruth = labeled_values(args.groundtruth, "--groundtruth")
    inventories = labeled_values(args.inventory, "--inventory")
    parameters = {
        key: decode_parameter(value)
        for key, value in labeled_values(args.parameter, "--parameter").items()
    }
    source_records = {}
    for robot, value in sources.items():
        if robot in inventories:
            inventory_path = Path(inventories[robot]).expanduser().resolve()
            inventory = json.loads(inventory_path.read_text(encoding="utf-8"))
            record = inventory.get("bag")
            if (
                not isinstance(record, dict)
                or Path(record.get("path", "")).resolve()
                != Path(value).expanduser().resolve()
            ):
                raise ValueError(f"inventory {inventory_path} does not describe {value}")
            source_records[robot] = record
        else:
            source_records[robot] = file_record(value)
    gt_records = {robot: file_record(value) for robot, value in groundtruth.items()}
    calibration = file_record(str(args.calibration))
    vocabulary = file_record(str(args.vocabulary))
    network = file_record(str(args.network))
    config = file_record(str(args.config))

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
                "origin_skill": "experiment-agent",
                "origin_mode": "plan",
                "origin_date": "2026-09-02",
                "version_label": "newer_college_staged_plan_v1",
            },
            "dataset": "Newer College multi-camera Collection 1",
            "camera": {
                "topic": "/alphasense_driver_ros/cam0/compressed",
                "frame": "AS C0 frontal-right",
                "projection_model": "pinhole",
                "source_distortion_model": "equidistant",
                "published_encoding": "mono8",
                "published_distortion_model": "plumb_bob",
            },
            "tracking": {"directory": str(run_dir / "tracking"), "sources": {}},
            "scenarios": {},
        }

    tracking_artifacts_exist = any(
        (run_dir / "tracking").glob("robot*/manifest.json")
    )
    immutable_inputs = {
        "calibration": calibration,
        "orb_vocabulary": vocabulary,
        "dpvo_network": network,
        "dpvo_config": config,
    }
    if tracking_artifacts_exist:
        for field, current in immutable_inputs.items():
            previous = manifest.get(field)
            if previous and previous.get("sha256") != current["sha256"]:
                raise RuntimeError(
                    f"immutable tracking artifacts use a different {field}"
                )
        previous_parameters = manifest.get("parameters", {})
        for key, value in parameters.items():
            if (
                key.startswith("tracking.")
                and key in previous_parameters
                and previous_parameters[key] != value
            ):
                raise RuntimeError(
                    f"immutable tracking artifacts use a different {key} parameter"
                )

    for robot, record in source_records.items():
        previous = manifest["tracking"]["sources"].get(robot, {}).get("bag")
        artifact_exists = (run_dir / "tracking" / robot / "manifest.json").is_file()
        if artifact_exists and previous and previous["sha256"] != record["sha256"]:
            raise RuntimeError(
                f"immutable {robot} artifact was created from a different bag checksum"
            )
        manifest["tracking"]["sources"][robot] = {
            "bag": record,
            "groundtruth": gt_records.get(robot),
        }
    manifest["calibration"] = calibration
    manifest["orb_vocabulary"] = vocabulary
    manifest["dpvo_network"] = network
    manifest["dpvo_config"] = config
    manifest["parameters"] = parameters
    robots = sorted(sources, key=robot_sort_key)
    scenario_root_names = (
        "geometric_verification",
        "dpgo",
        "evo",
        "plots",
    )
    manifest["scenarios"][args.scenario] = {
        "robot_ids": robots,
        "outputs": {
            name: str(run_dir / name / args.scenario) for name in scenario_root_names
        },
    }
    manifest["updated_utc"] = datetime.now(timezone.utc).isoformat()
    atomic_json(path, manifest)
    print(path)


def inspect_bag(args) -> None:
    from rosbags.highlevel import AnyReader

    bag = args.bag.expanduser().resolve()
    calibration_path = args.calibration.expanduser().resolve()
    if not bag.is_file():
        raise FileNotFoundError(bag)
    if args.stride < 1:
        raise ValueError("stride must be positive")
    calibration = read_kalibr_camera_calibration(calibration_path, args.camera_key)
    rectification = build_fisheye_rectification(calibration, args.balance)
    first = None
    last = None
    count = 0
    previous_stamp = None
    with AnyReader([bag]) as reader:
        connections = [
            connection
            for connection in reader.connections
            if connection.topic == args.topic
        ]
        if len(connections) != 1:
            raise RuntimeError(
                f"expected one {args.topic} connection in {bag}, found {len(connections)}"
            )
        connection = connections[0]
        if not connection.msgtype.endswith("/CompressedImage"):
            raise TypeError(f"{args.topic} has type {connection.msgtype}")
        for connection, _bag_stamp, rawdata in reader.messages(connections=connections):
            message = reader.deserialize(rawdata, connection.msgtype)
            stamp = stamp_components(message.header.stamp)
            stamp_ns = stamp[0] * 1_000_000_000 + stamp[1]
            if previous_stamp is not None and stamp_ns <= previous_stamp:
                raise ValueError(
                    f"non-increasing cam0 header timestamp at source frame {count}: "
                    f"{stamp_ns} <= {previous_stamp}"
                )
            previous_stamp = stamp_ns
            payload = bytes(message.data)
            if not payload:
                raise ValueError(f"empty compressed cam0 payload at source frame {count}")
            record = (stamp, str(message.format), payload)
            if first is None:
                first = record
            last = record
            count += 1
    if not count or first is None or last is None:
        raise RuntimeError(f"no cam0 images found in {bag}")
    decoded_shapes = []
    for stamp, source_format, payload in (first, last):
        if "jpeg" not in source_format.lower() and "jpg" not in source_format.lower():
            raise ValueError(f"cam0 compressed format is not JPEG: {source_format!r}")
        decoded_shapes.append(
            list(rectification.rectify(decode_compressed_mono8(payload)).shape)
        )
    inventory = {
        "format": "newer_college_ros1_cam0_inventory",
        "version": 1,
        "complete": True,
        "bag": file_record(str(bag)),
        "calibration": file_record(str(calibration_path)),
        "topic": args.topic,
        "message_type": connections[0].msgtype,
        "source_frame_count": count,
        "selected_frame_count": (count + args.stride - 1) // args.stride,
        "stride": args.stride,
        "first_header_stamp": {"sec": first[0][0], "nanosec": first[0][1]},
        "last_header_stamp": {"sec": last[0][0], "nanosec": last[0][1]},
        "decoded_first_last_shapes": decoded_shapes,
        "rectified_intrinsics": rectification.camera_matrix.tolist(),
        "resolution": list(calibration.resolution),
    }
    atomic_json(args.output, inventory)
    print(json.dumps(inventory, indent=2))


def artifact_ready(args) -> None:
    artifact = load_tracking_artifact(args.tracking_root / args.robot_id)
    if args.expected_frames is not None:
        actual = int(artifact.manifest["input_frame_count"])
        if abs(actual - args.expected_frames) > args.tolerance:
            raise RuntimeError(
                f"{args.robot_id} has {actual} input frames, expected approximately "
                f"{args.expected_frames} (+/- {args.tolerance})"
            )
    if args.rrd is not None and (
        not args.rrd.is_file() or args.rrd.stat().st_size == 0
    ):
        raise FileNotFoundError(f"missing or empty tracking RRD: {args.rrd}")
    print(
        f"{artifact.robot_id}: {artifact.manifest['input_frame_count']} frames, "
        f"{artifact.keyframe_count} keyframes"
    )


def validate_stage1(args) -> None:
    output = args.output.expanduser().resolve()
    try:
        artifacts = load_artifact_root(args.tracking_root, args.robot_ids)
        expected = {
            robot: int(value)
            for robot, value in labeled_values(args.expected, "--expected").items()
        }
        robots = {}
        for robot, artifact in artifacts.items():
            frame_count = int(artifact.manifest["input_frame_count"])
            if frame_count <= 0:
                raise ValueError(f"{robot} has no input frames")
            if (
                robot in expected
                and abs(frame_count - expected[robot]) > args.tolerance
            ):
                raise ValueError(
                    f"{robot} has {frame_count} frames, expected approximately "
                    f"{expected[robot]} (+/- {args.tolerance})"
                )
            rrd = args.rrd_root / f"{robot}.rrd"
            if not rrd.is_file() or rrd.stat().st_size == 0:
                raise FileNotFoundError(f"missing or empty tracking RRD: {rrd}")
            robots[robot] = {
                "input_frame_count": frame_count,
                "keyframe_count": artifact.keyframe_count,
                "state_sha256": artifact.manifest["state_sha256"],
                "map_sha256": artifact.manifest["map_sha256"],
                "keyframe_images_sha256": artifact.manifest[
                    "keyframe_images_sha256"
                ],
                "rrd": file_record(str(rrd)),
            }
        tracking_graphs = {
            "json": require_file(args.tracking_graph, "tracking JSON graph"),
            "g2o": require_file(
                args.tracking_graph.with_suffix(".g2o"), "tracking G2O graph"
            ),
        }
        gate = {
            "pass": True,
            "robot_ids": args.robot_ids,
            "robots": robots,
            "tracking_graphs": tracking_graphs,
        }
    except Exception as error:
        gate = {"pass": False, "robot_ids": args.robot_ids, "error": str(error)}
        atomic_json(output, gate)
        raise
    atomic_json(output, gate)
    print(json.dumps(gate, indent=2))


def robot_sort_key(robot_id: str):
    suffix = robot_id.removeprefix("robot")
    return (0, int(suffix)) if suffix.isdigit() else (1, robot_id)


def graph_diagnostics(path: Path, expected_robots: list[str]) -> dict:
    graph = read_json(path)
    robot_by_vertex = {vertex.vertex_id: vertex.robot_id for vertex in graph.vertices}
    robots = sorted(set(robot_by_vertex.values()), key=robot_sort_key)
    if set(robots) != set(expected_robots):
        raise ValueError(f"graph robots {robots} do not match expected {expected_robots}")
    adjacency = {robot: set() for robot in robots}
    pair_counts = Counter()
    loop_count = 0
    for edge in graph.edges:
        if edge.edge_type != "inter_robot_loop_closure":
            continue
        source_robot = robot_by_vertex[edge.source]
        target_robot = robot_by_vertex[edge.target]
        pair = tuple(sorted((source_robot, target_robot), key=robot_sort_key))
        pair_counts[pair] += 1
        adjacency[source_robot].add(target_robot)
        adjacency[target_robot].add(source_robot)
        loop_count += 1
    reached = set()
    frontier = [robots[0]] if robots else []
    while frontier:
        robot = frontier.pop()
        if robot in reached:
            continue
        reached.add(robot)
        frontier.extend(adjacency[robot] - reached)
    return {
        "pipeline_stage": graph.metadata.get("pipeline_stage"),
        "input_contains_global_optimization": graph.metadata.get(
            "input_contains_global_optimization"
        ),
        "robots": robots,
        "vertex_count": len(graph.vertices),
        "edge_count": len(graph.edges),
        "inter_robot_loop_count": loop_count,
        "pair_loop_counts": {
            "--".join(pair): count for pair, count in pair_counts.items()
        },
        "connected": reached == set(robots),
        "disconnected": sorted(set(robots) - reached, key=robot_sort_key),
    }


def check_graph(args) -> None:
    output = args.output.expanduser().resolve()
    try:
        diagnostics = graph_diagnostics(args.graph, args.robot_ids)
        if diagnostics["pipeline_stage"] != "geometric_verification":
            raise ValueError("graph is not a stage-two geometric-verification graph")
        if diagnostics["input_contains_global_optimization"] is not False:
            raise ValueError("graph contains or may contain global optimization")
        if not diagnostics["connected"]:
            raise ValueError(
                "verified robot graph is disconnected: "
                + ", ".join(diagnostics["disconnected"])
            )
        minimum_pairs = {
            key.replace(":", "--"): int(value)
            for key, value in labeled_values(args.min_pair, "--min-pair").items()
        }
        for pair, minimum in minimum_pairs.items():
            actual = diagnostics["pair_loop_counts"].get(pair, 0)
            if actual < minimum:
                raise ValueError(
                    f"verified pair {pair} has {actual} loops, requires {minimum}"
                )
        diagnostics.update(pass_=True, graph=file_record(str(args.graph)))
        diagnostics["pass"] = diagnostics.pop("pass_")
    except Exception as error:
        diagnostics = {"pass": False, "error": str(error)}
        if args.graph.is_file():
            try:
                diagnostics.update(graph_diagnostics(args.graph, args.robot_ids))
            except Exception as diagnostic_error:
                diagnostics["diagnostic_error"] = str(diagnostic_error)
        atomic_json(output, diagnostics)
        raise
    atomic_json(output, diagnostics)
    print(json.dumps(diagnostics, indent=2))


def evo_result(path: Path) -> dict:
    with zipfile.ZipFile(path) as archive:
        stats = json.loads(archive.read("stats.json"))
        transform = np.load(
            io.BytesIO(archive.read("alignment_transformation_sim3.npy"))
        )
    if transform.shape != (4, 4) or not np.isfinite(transform).all():
        raise ValueError(f"invalid evo Sim(3) alignment in {path}")
    scale = float(abs(np.linalg.det(transform[:3, :3])) ** (1.0 / 3.0))
    rmse = float(stats["rmse"])
    if not math.isfinite(rmse) or rmse < 0.0:
        raise ValueError(f"invalid evo RMSE in {path}: {rmse}")
    if not math.isfinite(scale) or scale <= 0.0:
        raise ValueError(f"invalid evo alignment scale in {path}: {scale}")
    return {
        "ate_rmse_m": rmse,
        "groundtruth_relative_scale": scale,
        "archive": file_record(str(path)),
    }


def collect_metrics(args) -> None:
    methods = {}
    expected_methods = args.methods or [
        "centralized",
        "centralized_explicit_anchors",
        "cbs",
    ]
    scopes = (*args.robot_ids, "joint")
    for method in expected_methods:
        methods[method] = {
            scope: evo_result(args.evo_dir / "results" / f"{method}_{scope}.zip")
            for scope in scopes
        }
    metrics = {
        "format": "dpvo_newer_college_evo_metrics",
        "version": 1,
        "robot_ids": args.robot_ids,
        "pose_relation": "translation part",
        "alignment": "Sim(3) (--align --correct_scale)",
        "t_max_diff_seconds": args.t_max_diff,
        "reported_centralized_baseline": "centralized_explicit_anchors",
        "methods": methods,
    }
    atomic_json(args.output, metrics)
    print(json.dumps(metrics, indent=2))


def require_file(path: Path, description: str) -> dict:
    if not path.is_file() or path.stat().st_size == 0:
        raise FileNotFoundError(f"missing or empty {description}: {path}")
    return file_record(str(path))


def final_gate(args) -> None:
    output = args.output.expanduser().resolve()
    try:
        stage1 = json.loads(args.stage1_gate.read_text(encoding="utf-8"))
        graph = json.loads(args.graph_gate.read_text(encoding="utf-8"))
        metrics = json.loads(args.metrics.read_text(encoding="utf-8"))
        if not stage1.get("pass"):
            raise RuntimeError("stage-one gate did not pass")
        if not graph.get("pass"):
            raise RuntimeError("verified-graph gate did not pass")
        if stage1.get("robot_ids") != args.robot_ids:
            raise RuntimeError("stage-one gate robot set does not match this scenario")
        if graph.get("robots") != args.robot_ids:
            raise RuntimeError("verified-graph gate robot set does not match this scenario")
        if metrics.get("robot_ids") != args.robot_ids:
            raise RuntimeError("metrics robot set does not match this scenario")

        provenance_path = args.dpgo_dir / "offline_dpgo_provenance.json"
        provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
        if provenance.get("status") != "complete" or provenance.get("return_code") != 0:
            raise RuntimeError("stage-three solver did not exit successfully")
        if not provenance.get("copied_input_is_byte_identical"):
            raise RuntimeError("stage-three input is not byte-identical to stage two")

        actual_source_sha256 = sha256(args.verified_graph)
        copied_input = args.dpgo_dir / "input_keyframes_unoptimized.json"
        actual_copied_sha256 = sha256(copied_input)
        if actual_source_sha256 != actual_copied_sha256:
            raise RuntimeError("current stage-two and stage-three JSON graphs differ")
        if provenance.get("source_sha256") != actual_source_sha256:
            raise RuntimeError("stage-two graph checksum no longer matches provenance")
        if provenance.get("copied_input_sha256") != actual_copied_sha256:
            raise RuntimeError("stage-three input checksum no longer matches provenance")

        required = {
            "verified_graph_json": require_file(
                args.verified_graph, "verified JSON graph"
            ),
            "verified_graph_g2o": require_file(
                args.verified_graph.with_suffix(".g2o"), "verified G2O graph"
            ),
            "dpgo_input_json": require_file(
                args.dpgo_dir / "input_keyframes_unoptimized.json", "DPGO input JSON"
            ),
            "dpgo_input_g2o": require_file(
                args.dpgo_dir / "input_keyframes_unoptimized.g2o", "DPGO input G2O"
            ),
            "centralized_csv": require_file(
                args.dpgo_dir / "centralized.csv", "centralized pose-variable CSV"
            ),
            "explicit_centralized_csv": require_file(
                args.dpgo_dir / "centralized_explicit_anchors.csv",
                "explicit-anchor centralized CSV",
            ),
            "cbs_csv": require_file(
                args.dpgo_dir / "cbs.csv", "distributed CBS trajectory CSV"
            ),
            "dpgo_rrd": require_file(
                args.dpgo_dir / "dpvo_sim3_cbs.rrd", "stage-three DPGO RRD"
            ),
            "metrics": require_file(args.metrics, "metrics JSON"),
            "provenance": require_file(provenance_path, "DPGO provenance"),
        }
        for method in (
            "centralized",
            "centralized_explicit_anchors",
            "cbs",
        ):
            for scope in (*args.robot_ids, "joint"):
                archive_path = (
                    args.evo_dir / "results" / f"{method}_{scope}.zip"
                )
                archive_record = require_file(
                    archive_path,
                    f"evo archive {method}/{scope}",
                )
                metric = metrics["methods"][method][scope]
                if metric.get("archive", {}).get("sha256") != archive_record["sha256"]:
                    raise RuntimeError(
                        f"evo archive checksum changed for {method}/{scope}"
                    )
                scale = float(
                    metric["groundtruth_relative_scale"]
                )
                if not math.isfinite(scale) or scale <= 0.0:
                    raise ValueError(f"invalid {method}/{scope} scale: {scale}")
                ate = float(metric["ate_rmse_m"])
                if not math.isfinite(ate) or ate < 0.0:
                    raise ValueError(f"invalid {method}/{scope} ATE: {ate}")
        central_ate = float(
            metrics["methods"]["centralized_explicit_anchors"]["joint"]["ate_rmse_m"]
        )
        cbs_ate = float(
            metrics["methods"]["cbs"]["joint"]["ate_rmse_m"]
        )
        if cbs_ate > 1.25 * central_ate:
            raise RuntimeError(
                f"CBS joint ATE {cbs_ate:.6g} exceeds 125% of explicit-anchor "
                f"centralized ATE {central_ate:.6g}"
            )
        ate_ratio = (
            cbs_ate / central_ate
            if central_ate > 0.0
            else (1.0 if cbs_ate == 0.0 else math.inf)
        )
        plots = {}
        if args.plots_dir is not None:
            prefix = args.plot_prefix
            for name in (
                f"{prefix}_trajectories_loops.png",
                f"{prefix}_trajectories_loops.pdf",
                f"{prefix}_sparse_joint_map_alignment.png",
                f"{prefix}_sparse_joint_map_alignment.pdf",
            ):
                plots[name] = require_file(args.plots_dir / name, f"plot {name}")
        gate = {
            "pass": True,
            "robot_ids": args.robot_ids,
            "centralized_explicit_anchor_joint_ate_rmse_m": central_ate,
            "cbs_joint_ate_rmse_m": cbs_ate,
            "cbs_to_centralized_ate_ratio": ate_ratio,
            "required_artifacts": required,
            "plots": plots,
        }
    except Exception as error:
        gate = {"pass": False, "robot_ids": args.robot_ids, "error": str(error)}
        atomic_json(output, gate)
        raise
    atomic_json(output, gate)
    print(json.dumps(gate, indent=2))


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description=__doc__)
    commands = root.add_subparsers(dest="command", required=True)

    manifest = commands.add_parser("record-manifest")
    manifest.add_argument("--run-dir", type=Path, required=True)
    manifest.add_argument("--scenario", choices=("two", "three"), required=True)
    manifest.add_argument("--calibration", type=Path, required=True)
    manifest.add_argument("--vocabulary", type=Path, required=True)
    manifest.add_argument("--network", type=Path, required=True)
    manifest.add_argument("--config", type=Path, required=True)
    manifest.add_argument("--source", action="append", required=True)
    manifest.add_argument("--groundtruth", action="append", required=True)
    manifest.add_argument("--inventory", action="append", default=[])
    manifest.add_argument("--parameter", action="append", default=[])
    manifest.set_defaults(function=record_manifest)

    inspect = commands.add_parser("inspect-bag")
    inspect.add_argument("--bag", type=Path, required=True)
    inspect.add_argument("--calibration", type=Path, required=True)
    inspect.add_argument("--camera-key", default="cam0")
    inspect.add_argument(
        "--topic", default="/alphasense_driver_ros/cam0/compressed"
    )
    inspect.add_argument("--stride", type=int, default=2)
    inspect.add_argument("--balance", type=float, default=0.0)
    inspect.add_argument("--output", type=Path, required=True)
    inspect.set_defaults(function=inspect_bag)

    ready = commands.add_parser("artifact-ready")
    ready.add_argument("--tracking-root", type=Path, required=True)
    ready.add_argument("--robot-id", required=True)
    ready.add_argument("--expected-frames", type=int)
    ready.add_argument("--tolerance", type=int, default=2)
    ready.add_argument("--rrd", type=Path)
    ready.set_defaults(function=artifact_ready)

    stage1 = commands.add_parser("validate-stage1")
    stage1.add_argument("--tracking-root", type=Path, required=True)
    stage1.add_argument("--rrd-root", type=Path, required=True)
    stage1.add_argument("--tracking-graph", type=Path, required=True)
    stage1.add_argument("--robot-ids", nargs="+", required=True)
    stage1.add_argument("--expected", action="append", default=[])
    stage1.add_argument("--tolerance", type=int, default=2)
    stage1.add_argument("--output", type=Path, required=True)
    stage1.set_defaults(function=validate_stage1)

    graph = commands.add_parser("check-graph")
    graph.add_argument("--graph", type=Path, required=True)
    graph.add_argument("--robot-ids", nargs="+", required=True)
    graph.add_argument("--min-pair", action="append", default=[])
    graph.add_argument("--output", type=Path, required=True)
    graph.set_defaults(function=check_graph)

    metrics = commands.add_parser("collect-metrics")
    metrics.add_argument("--evo-dir", type=Path, required=True)
    metrics.add_argument("--robot-ids", nargs="+", required=True)
    metrics.add_argument("--methods", nargs="+")
    metrics.add_argument("--t-max-diff", type=float, default=0.055)
    metrics.add_argument("--output", type=Path, required=True)
    metrics.set_defaults(function=collect_metrics)

    final = commands.add_parser("final-gate")
    final.add_argument("--stage1-gate", type=Path, required=True)
    final.add_argument("--graph-gate", type=Path, required=True)
    final.add_argument("--verified-graph", type=Path, required=True)
    final.add_argument("--dpgo-dir", type=Path, required=True)
    final.add_argument("--evo-dir", type=Path, required=True)
    final.add_argument("--metrics", type=Path, required=True)
    final.add_argument("--plots-dir", type=Path)
    final.add_argument("--plot-prefix", default="newer_college")
    final.add_argument("--robot-ids", nargs="+", required=True)
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
