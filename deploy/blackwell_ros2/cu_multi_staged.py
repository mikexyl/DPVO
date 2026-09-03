#!/usr/bin/env python3
"""Immutable-input manifest support for the staged CU-Multi experiment."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path


FORMAT = "dpvo_cu_multi_main_campus_staged"
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
    archives = labeled(args.source_archive, "--source-archive")
    bags = labeled(args.bag, "--bag")
    groundtruth = labeled(args.groundtruth, "--groundtruth")
    if set(archives) != set(bags) or set(archives) != set(groundtruth):
        raise ValueError("archive, bag, and ground-truth robot sets must match")
    robots = sorted(archives, key=robot_sort_key)
    source_records = {
        robot: {
            "rgb_archive": file_record(archives[robot]),
            "extracted_ros2_bag": bag_record(bags[robot]),
            "utm_groundtruth": file_record(groundtruth[robot]),
        }
        for robot in robots
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
            "material_passport": {
                "origin_skill": "academic-research-suite",
                "origin_mode": "experiment execution",
                "origin_date": "2026-09-02",
                "version_label": "cu_multi_main_campus_staged_v1",
                "verification_status": "execution_in_progress",
            },
            "dataset": "CU-Multi Main Campus",
            "dataset_reference": {
                "project": "https://arpg.colorado.edu/cumulti/",
                "repository": "https://github.com/arpg/CU-Multi",
                "paper": "https://arxiv.org/abs/2509.19463",
            },
            "camera": {
                "sensor": "Intel RealSense D455 RGB",
                "source_encoding": "rgb8",
                "source_resolution": [1280, 800],
                "source_rate_hz": 10.0,
                "source_distortion_model": "plumb_bob",
                "published_encoding": "bgr8",
                "published_distortion_model": "plumb_bob",
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
                previous["rgb_archive"]["sha256"]
                != current["rgb_archive"]["sha256"]
                or previous["extracted_ros2_bag"]["database"]["sha256"]
                != current["extracted_ros2_bag"]["database"]["sha256"]
            ):
                raise RuntimeError(f"immutable {robot} artifact source changed")

    manifest.update(immutable_inputs)
    manifest["parameters"] = parameters
    manifest["tracking"]["sources"].update(source_records)
    manifest["scenarios"][args.scenario] = {
        "robot_ids": robots,
        "outputs": {
            name: str(run_dir / name / args.scenario)
            for name in ("geometric_verification", "dpgo", "evo", "plots")
        },
    }
    manifest["updated_utc"] = datetime.now(timezone.utc).isoformat()
    atomic_json(path, manifest)
    print(path)


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description=__doc__)
    root.add_argument("--run-dir", type=Path, required=True)
    root.add_argument("--scenario", choices=("two", "four"), required=True)
    root.add_argument("--source-archive", action="append", required=True)
    root.add_argument("--bag", action="append", required=True)
    root.add_argument("--groundtruth", action="append", required=True)
    root.add_argument("--vocabulary", type=Path, required=True)
    root.add_argument("--network", type=Path, required=True)
    root.add_argument("--config", type=Path, required=True)
    root.add_argument("--parameter", action="append", default=[])
    return root


def main(argv=None) -> None:
    args = parser().parse_args(argv)
    try:
        record_manifest(args)
    except (FileNotFoundError, NotADirectoryError, RuntimeError, ValueError) as error:
        raise SystemExit(f"record manifest failed: {error}") from error


if __name__ == "__main__":
    main()
