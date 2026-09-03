#!/usr/bin/env python3
"""Run the no-reset Hellinger quadratic-term covariance transport suite."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time


DATASETS = {
    "iphone14": {
        "graph": Path(
            "/data3/dpvo_cbs_ws/results/"
            "iphone_302_4_5_6_7_8_9_10_11_12_13_14_15_16_17_"
            "full_min18_fourteen_robot_20260901/geometric_verification/"
            "unoptimized_verified_graph.json"
        ),
        "legacy": False,
    },
    "kitti00_10_overlap50": {
        "graph": Path(
            "/data3/mikexyl/results/dpvo_multi_robot/"
            "kitti00_ten_robot_overlap50_staged_full_20260901/"
            "geometric_verification/unoptimized_verified_graph.json"
        ),
        "legacy": False,
    },
    "tum": {
        "graph": Path(
            "/data3/mikexyl/results/dpvo_multi_robot/"
            "tum_fr1_desk_desk2_megaloc_xfeat_full_20260830_cbs/"
            "input_keyframes_unoptimized.json"
        ),
        "legacy": True,
    },
    "euroc": {
        "graph": Path(
            "/data3/mikexyl/results/dpvo_multi_robot/"
            "v1_01_02_03_full_online_cbs_20260827_run1_cbs/"
            "input_keyframes_unoptimized.json"
        ),
        "legacy": True,
    },
}
TRANSPORTS = ("none", "adjoint", "bernoulli")
QUADRATIC_MODES = ("quadratic_off", "quadratic_on")
CONFIGURATION = {
    "iterations": 1000,
    "stage_mode": "alternating",
    "pose_block_iterations": 20,
    "anchor_block_iterations": 20,
    "pose_warmup_iterations": 0,
    "target_hellinger": 0.1,
    "contract_alpha": 0.95,
    "d_reset": 1.1,
    "reset_effectively_disabled": True,
    "random_seed": 42,
    "huber_k": -1.0,
    "bootstrap_robot_anchors": False,
    "rerun_iteration_stride": 1,
    "rerun_stream": False,
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, payload: dict) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    temporary.replace(path)


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser()
    result.add_argument("--output-root", type=Path, required=True)
    result.add_argument("--project-root", type=Path, required=True)
    result.add_argument("--python", type=Path, required=True)
    result.add_argument("--cbs-executable", type=Path, required=True)
    result.add_argument(
        "--resume",
        action="store_true",
        help=(
            "resume an interrupted suite after validating its immutable inputs; "
            "completed runs are skipped and partial attempts are archived"
        ),
    )
    return result


def validate_resume_manifest(
    manifest: dict,
    args: argparse.Namespace,
    executable: Path,
) -> None:
    if manifest.get("format") != "hellinger_no_reset_quadratic_suite":
        raise ValueError("output root is not a Hellinger quadratic suite")
    if manifest.get("version") != 1:
        raise ValueError(f"unsupported manifest version: {manifest.get('version')}")
    if manifest.get("configuration") != CONFIGURATION:
        raise ValueError("resume configuration differs from the original suite")

    immutable_paths = {
        "project_root": args.project_root.resolve(),
        "python": args.python.resolve(),
        "cbs_executable": executable,
    }
    for field, actual in immutable_paths.items():
        if Path(manifest.get(field, "")).resolve() != actual:
            raise ValueError(f"resume {field} differs from the original suite")
    if manifest.get("cbs_executable_sha256") != sha256(executable):
        raise ValueError("CBS executable checksum differs from the original suite")

    recorded_datasets = manifest.get("datasets", {})
    for name, dataset in DATASETS.items():
        recorded = recorded_datasets.get(name)
        if not isinstance(recorded, dict):
            raise ValueError(f"dataset {name} is missing from the manifest")
        if Path(recorded.get("source_graph", "")).resolve() != dataset["graph"]:
            raise ValueError(f"dataset {name} source path differs")
        if recorded.get("source_sha256") != sha256(dataset["graph"]):
            raise ValueError(f"dataset {name} source checksum differs")
        if recorded.get("legacy_unoptimized_opt_in") != dataset["legacy"]:
            raise ValueError(f"dataset {name} legacy-input setting differs")


def archive_partial_attempt(
    output_root: Path,
    logs: Path,
    run_id: str,
    run_output: Path,
    log_path: Path,
    attempt: int,
    prior_record: dict | None,
) -> None:
    if not run_output.exists() and not log_path.exists():
        return

    archive_root = output_root / "interrupted_artifacts"
    archived_output = archive_root / run_id / f"dpgo_attempt{attempt}"
    archived_log = archive_root / "logs" / f"{run_id.replace('/', '-')}-attempt{attempt}.log"
    if run_output.exists():
        if archived_output.exists():
            raise FileExistsError(f"refusing to overwrite {archived_output}")
        archived_output.parent.mkdir(parents=True, exist_ok=True)
        run_output.rename(archived_output)
        if prior_record is not None:
            prior_record["archived_output"] = str(archived_output)
    if log_path.exists():
        if archived_log.exists():
            raise FileExistsError(f"refusing to overwrite {archived_log}")
        archived_log.parent.mkdir(parents=True, exist_ok=True)
        log_path.rename(archived_log)
        if prior_record is not None:
            prior_record["archived_log"] = str(archived_log)


def main() -> None:
    args = parser().parse_args()
    output_root = args.output_root.resolve()
    if not args.resume and output_root.exists() and any(output_root.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty {output_root}")
    if args.resume and not output_root.is_dir():
        raise FileNotFoundError(f"resume output root does not exist: {output_root}")
    output_root.mkdir(parents=True, exist_ok=True)
    logs = output_root / "logs"
    logs.mkdir(exist_ok=True)

    executable = args.cbs_executable.resolve()
    if not executable.is_file():
        raise FileNotFoundError(executable)
    for dataset in DATASETS.values():
        if not dataset["graph"].is_file():
            raise FileNotFoundError(dataset["graph"])

    manifest_path = output_root / "experiment_manifest.json"
    prior_manifest_status = None
    if args.resume:
        manifest = json.loads(manifest_path.read_text())
        validate_resume_manifest(manifest, args, executable)
        prior_manifest_status = manifest.get("status")
        manifest["status"] = "running"
        manifest.setdefault("resumed_unix", []).append(time.time())
    else:
        manifest = {
            "format": "hellinger_no_reset_quadratic_suite",
            "version": 1,
            "status": "running",
            "started_unix": time.time(),
            "project_root": str(args.project_root.resolve()),
            "python": str(args.python.resolve()),
            "cbs_executable": str(executable),
            "cbs_executable_sha256": sha256(executable),
            "configuration": CONFIGURATION,
            "datasets": {
                name: {
                    "source_graph": str(config["graph"]),
                    "source_sha256": sha256(config["graph"]),
                    "legacy_unoptimized_opt_in": config["legacy"],
                }
                for name, config in DATASETS.items()
            },
            "runs": [],
        }
    atomic_json(manifest_path, manifest)

    completed_run_ids = {
        record["run_id"]
        for record in manifest["runs"]
        if record.get("status") == "complete"
    }
    expected_run_ids = {
        f"{quadratic_mode}/{dataset_name}/{transport}"
        for quadratic_mode in QUADRATIC_MODES
        for dataset_name in DATASETS
        for transport in TRANSPORTS
    }
    if (
        args.resume
        and prior_manifest_status == "running"
        and completed_run_ids == expected_run_ids
    ):
        anomaly = {
            "type": "MANIFEST_FINALIZER_RECOVERY",
            "detail": (
                "All 24 solver attempts completed, but the wrapper finalizer "
                "failed because the reboot-recovered manifest has no "
                "started_unix field. No solver was rerun; this invocation only "
                "finalized metadata after validating all completed artifacts."
            ),
        }
        if anomaly not in manifest.setdefault("anomalies", []):
            manifest["anomalies"].append(anomaly)
        atomic_json(manifest_path, manifest)
    for run_id in completed_run_ids:
        complete_output = output_root / run_id / "dpgo"
        required = (
            complete_output / "cbs.csv",
            complete_output / "summary.csv",
            complete_output / "offline_dpgo_provenance.json",
            complete_output / "dpvo_sim3_cbs.rrd",
        )
        missing = [str(path) for path in required if not path.is_file()]
        if missing:
            raise FileNotFoundError(
                f"completed run {run_id} is missing artifacts: {missing}"
            )

    try:
        for quadratic_mode in QUADRATIC_MODES:
            quadratic_enabled = quadratic_mode == "quadratic_on"
            for dataset_name, dataset in DATASETS.items():
                for transport in TRANSPORTS:
                    run_id = f"{quadratic_mode}/{dataset_name}/{transport}"
                    if run_id in completed_run_ids:
                        print(f"SKIP {run_id} (complete)", flush=True)
                        continue
                    run_output = output_root / quadratic_mode / dataset_name / transport / "dpgo"
                    base_log_path = logs / f"{quadratic_mode}-{dataset_name}-{transport}.log"
                    prior_records = [
                        record
                        for record in manifest["runs"]
                        if record.get("run_id") == run_id
                    ]
                    prior_attempt = max(
                        (int(record.get("attempt", 1)) for record in prior_records),
                        default=0,
                    )
                    prior_record = prior_records[-1] if prior_records else None
                    archive_partial_attempt(
                        output_root,
                        logs,
                        run_id,
                        run_output,
                        base_log_path,
                        prior_attempt,
                        prior_record,
                    )
                    attempt = prior_attempt + 1
                    log_path = (
                        logs / f"{quadratic_mode}-{dataset_name}-{transport}-attempt{attempt}.log"
                        if attempt > 1
                        else base_log_path
                    )
                    command = [
                        str(args.python.resolve()),
                        "-m",
                        "dpvo.loop_closure.offline_dpgo",
                        "--input-graph",
                        str(dataset["graph"]),
                        "--output-dir",
                        str(run_output),
                        "--cbs-executable",
                        str(executable),
                        "--iterations",
                        "1000",
                        "--stage-mode",
                        "alternating",
                        "--pose-warmup-iterations",
                        "0",
                        "--pose-block-iterations",
                        "20",
                        "--anchor-block-iterations",
                        "20",
                        "--target-hellinger",
                        "0.1",
                        "--contract-alpha",
                        "0.95",
                        "--d-reset",
                        "1.1",
                        "--sim3-covariance-transport",
                        transport,
                        "--no-bootstrap-robot-anchors",
                        "--random-seed",
                        "42",
                        "--huber-k",
                        "-1",
                        "--centralized-max-iterations",
                        "300",
                        "--write-rerun-rrd",
                        "--rerun-iteration-stride",
                        "1",
                    ]
                    command.append(
                        "--hellinger-quadratic-term"
                        if quadratic_enabled
                        else "--no-hellinger-quadratic-term"
                    )
                    if dataset["legacy"]:
                        command.append("--allow-legacy-unoptimized-input")

                    record = {
                        "run_id": run_id,
                        "attempt": attempt,
                        "quadratic_term": quadratic_enabled,
                        "transport": transport,
                        "command": command,
                        "log": str(log_path),
                        "status": "running",
                        "started_unix": time.time(),
                    }
                    manifest["runs"].append(record)
                    atomic_json(manifest_path, manifest)
                    print(f"START {run_id}", flush=True)
                    with log_path.open("wb") as stream:
                        completed = subprocess.run(
                            command,
                            cwd=args.project_root.resolve(),
                            stdout=stream,
                            stderr=subprocess.STDOUT,
                            check=False,
                        )
                    record["return_code"] = int(completed.returncode)
                    record["finished_unix"] = time.time()
                    record["duration_seconds"] = (
                        record["finished_unix"] - record["started_unix"]
                    )
                    record["status"] = (
                        "complete" if completed.returncode == 0 else "failed"
                    )
                    atomic_json(manifest_path, manifest)
                    print(
                        f"END {run_id} rc={completed.returncode} "
                        f"duration={record['duration_seconds']:.1f}s",
                        flush=True,
                    )
                    if completed.returncode != 0:
                        raise subprocess.CalledProcessError(completed.returncode, command)
                    completed_run_ids.add(run_id)
    except BaseException:
        manifest["status"] = "failed"
        manifest["finished_unix"] = time.time()
        atomic_json(manifest_path, manifest)
        raise

    manifest["status"] = "complete"
    manifest["finished_unix"] = time.time()
    if "started_unix" in manifest:
        manifest["duration_seconds"] = (
            manifest["finished_unix"] - manifest["started_unix"]
        )
    else:
        # The workstation-reboot recovery reconstructed the immutable run
        # records but could not recover the suite's original wall-clock start.
        manifest["duration_seconds"] = None
        manifest["timing_lost_in_metadata_recovery"] = True
    atomic_json(manifest_path, manifest)
    duration = manifest["duration_seconds"]
    duration_text = f"{duration:.1f}s" if duration is not None else "unavailable"
    print(f"SUITE COMPLETE duration={duration_text}", flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"SUITE FAILED: {error}", file=sys.stderr, flush=True)
        raise
