#!/usr/bin/env python3
"""Measure raw CU-Multi tracking quality without making it an acceptance gate."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from deploy.blackwell_ros2.export_cu_multi_evo_tum import read_utm_groundtruth
from dpvo.loop_closure.tracking_artifact import load_tracking_artifact


def labeled_paths(values: list[str], option: str) -> dict[str, Path]:
    result = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"{option} expects ROBOT=PATH, got {value!r}")
        robot, path_value = value.split("=", 1)
        if robot in result:
            raise ValueError(f"duplicate {option} robot: {robot}")
        result[robot] = Path(path_value).expanduser().resolve()
    return result


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def associate(
    estimate_timestamps: np.ndarray,
    reference_timestamps: np.ndarray,
    max_difference: float,
) -> tuple[np.ndarray, np.ndarray]:
    after = np.searchsorted(reference_timestamps, estimate_timestamps)
    after = np.clip(after, 1, len(reference_timestamps) - 1)
    before = after - 1
    choose_before = (
        np.abs(reference_timestamps[before] - estimate_timestamps)
        < np.abs(reference_timestamps[after] - estimate_timestamps)
    )
    reference_indices = np.where(choose_before, before, after)
    estimate_indices = np.flatnonzero(
        np.abs(reference_timestamps[reference_indices] - estimate_timestamps)
        <= max_difference
    )
    return estimate_indices, reference_indices[estimate_indices]


def align_similarity(source: np.ndarray, target: np.ndarray):
    source_mean = source.mean(axis=0)
    target_mean = target.mean(axis=0)
    centered_source = source - source_mean
    centered_target = target - target_mean
    variance = float(np.mean(np.sum(centered_source**2, axis=1)))
    if variance <= 1e-12:
        raise ValueError("estimate has zero translation variance")
    covariance = centered_target.T @ centered_source / len(source)
    left, singular_values, right = np.linalg.svd(covariance)
    signs = np.ones(3)
    signs[-1] = np.sign(np.linalg.det(left @ right))
    rotation = left @ np.diag(signs) @ right
    scale = float(np.sum(singular_values * signs) / variance)
    translation = target_mean - scale * rotation @ source_mean
    aligned = scale * source @ rotation.T + translation
    if not np.isfinite(aligned).all() or not np.isfinite(scale) or scale <= 0.0:
        raise ValueError("invalid Sim(3) alignment")
    return aligned, scale, rotation, translation


def path_length(points: np.ndarray) -> float:
    return float(np.linalg.norm(np.diff(points, axis=0), axis=1).sum())


def diagnose(artifact_path: Path, groundtruth_path: Path, max_difference: float):
    artifact = load_tracking_artifact(artifact_path)
    reference_rows = read_utm_groundtruth(groundtruth_path)
    reference_timestamps = np.asarray([row[0] for row in reference_rows])
    reference_positions = np.asarray([row[1] for row in reference_rows])
    estimate_indices, reference_indices = associate(
        artifact.keyframe_timestamps, reference_timestamps, max_difference
    )
    if len(estimate_indices) < 4:
        raise RuntimeError(
            f"only {len(estimate_indices)} associated poses for {artifact.robot_id}"
        )
    estimate = artifact.keyframe_poses_xyzw[estimate_indices, :3].astype(np.float64)
    reference = reference_positions[reference_indices]
    aligned, scale, rotation, translation = align_similarity(estimate, reference)
    errors = np.linalg.norm(aligned - reference, axis=1)
    elapsed = float(
        artifact.keyframe_timestamps[-1] - artifact.keyframe_timestamps[0]
    )
    return {
        "robot_id": artifact.robot_id,
        "keyframe_count": artifact.keyframe_count,
        "associated_keyframe_count": int(len(estimate_indices)),
        "association_fraction": float(len(estimate_indices) / artifact.keyframe_count),
        "keyframes_per_second": float(
            (artifact.keyframe_count - 1) / elapsed if elapsed > 0.0 else 0.0
        ),
        "sim3_scale_estimate_to_reference": scale,
        "sim3_rotation": rotation.tolist(),
        "sim3_translation": translation.tolist(),
        "ape_translation_m": {
            "rmse": float(np.sqrt(np.mean(errors**2))),
            "median": float(np.median(errors)),
            "p95": float(np.quantile(errors, 0.95)),
            "max": float(np.max(errors)),
        },
        "path_length_m": {
            "reference_at_keyframes": path_length(reference),
            "aligned_estimate": path_length(aligned),
        },
        "artifact_state_sha256": artifact.manifest["state_sha256"],
        "groundtruth": {
            "path": str(groundtruth_path),
            "sha256": sha256(groundtruth_path),
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", action="append", required=True)
    parser.add_argument("--groundtruth", action="append", required=True)
    parser.add_argument("--max-time-difference", type=float, default=0.055)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    artifacts = labeled_paths(args.artifact, "--artifact")
    groundtruth = labeled_paths(args.groundtruth, "--groundtruth")
    if set(artifacts) != set(groundtruth):
        raise SystemExit("artifact and ground-truth robot sets do not match")
    diagnostics = {
        robot: diagnose(
            artifacts[robot], groundtruth[robot], args.max_time_difference
        )
        for robot in sorted(artifacts)
    }
    output = {
        "format": "dpvo_cu_multi_raw_tracking_diagnostics",
        "version": 1,
        "acceptance_role": "diagnostic_only",
        "alignment": "Umeyama Sim(3), translation part",
        "max_time_difference_seconds": args.max_time_difference,
        "camera_extrinsic_applied": False,
        "robots": diagnostics,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(f".{args.output.name}.tmp")
    temporary.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    temporary.replace(args.output)
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
