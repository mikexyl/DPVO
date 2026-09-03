#!/usr/bin/env python3
"""Manifest helper for staged three-robot S3E experiments."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path


FORMAT = "dpvo_s3e_staged"
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


def record_manifest(args) -> None:
    run_dir = args.run_dir.expanduser().resolve()
    output = run_dir / "experiment_manifest.json"
    if output.exists():
        raise RuntimeError(f"refusing to overwrite existing manifest: {output}")
    calibrations = labeled(args.calibration, "--calibration")
    groundtruth = labeled(args.groundtruth, "--groundtruth")
    topics = labeled(args.topic, "--topic")
    expected_robots = ["robot0", "robot1", "robot2"]
    if sorted(calibrations) != expected_robots:
        raise ValueError(f"expected calibrations for {expected_robots}")
    if sorted(groundtruth) != expected_robots or sorted(topics) != expected_robots:
        raise ValueError(f"expected ground truth and topics for {expected_robots}")

    bag = args.bag.expanduser().resolve()
    metadata = bag / "metadata.yaml"
    databases = sorted(bag.glob("*.db3"))
    if len(databases) != 1:
        raise RuntimeError(f"expected one db3 under {bag}, found {len(databases)}")
    manifest = {
        "format": FORMAT,
        "version": VERSION,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "material_passport": {
            "origin_skill": "academic-research-suite/experiment-agent",
            "origin_mode": "run",
            "origin_date": datetime.now(timezone.utc).date().isoformat(),
            "version_label": args.version_label,
            "verification_status": "execution_in_progress",
        },
        "dataset": args.dataset_label,
        "bag": {
            "path": str(bag),
            "metadata": file_record(metadata),
            "database": file_record(databases[0]),
        },
        "robots": {
            robot: {
                "source_topic": topics[robot],
                "calibration": file_record(calibrations[robot]),
                "groundtruth": file_record(groundtruth[robot]),
            }
            for robot in expected_robots
        },
        "orb_vocabulary": file_record(args.vocabulary),
        "dpvo_network": file_record(args.network),
        "dpvo_config": file_record(args.config),
        "parameters": {
            key: decode_parameter(value)
            for key, value in labeled(args.parameter, "--parameter").items()
        },
        "tracking": {
            "directory": str(run_dir / "tracking"),
            "robot_ids": expected_robots,
        },
        "outputs": {
            name: str(run_dir / name / "three")
            for name in ("geometric_verification", "dpgo", "evo", "plots")
        },
    }
    atomic_json(output, manifest)
    print(output)


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description=__doc__)
    root.add_argument("--run-dir", type=Path, required=True)
    root.add_argument("--bag", type=Path, required=True)
    root.add_argument(
        "--dataset-label", default="S3Ev1 Teaching Building 1"
    )
    root.add_argument(
        "--version-label", default="s3e_teaching_building_1_staged_v1"
    )
    root.add_argument("--calibration", action="append", required=True)
    root.add_argument("--groundtruth", action="append", required=True)
    root.add_argument("--topic", action="append", required=True)
    root.add_argument("--vocabulary", type=Path, required=True)
    root.add_argument("--network", type=Path, required=True)
    root.add_argument("--config", type=Path, required=True)
    root.add_argument("--parameter", action="append", default=[])
    root.set_defaults(handler=record_manifest)
    return root


def main(argv=None) -> None:
    args = parser().parse_args(argv)
    try:
        args.handler(args)
    except (FileNotFoundError, RuntimeError, ValueError) as error:
        raise SystemExit(f"manifest failed: {error}") from error


if __name__ == "__main__":
    main()
