#!/usr/bin/env python3
"""Prepare two extracted ICL-NUIM sequences for the ROS 2 TUM player."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


SEQUENCES = (
    ("living_room_traj0_frei_png", "livingRoom0.gt.freiburg", 0.0),
    ("living_room_traj1_frei_png", "livingRoom1.gt.freiburg", 10_000.0),
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--downloads-dir", type=Path)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def prepare_sequence(sequence_dir: Path, source_groundtruth: Path, offset: float):
    if not sequence_dir.is_dir():
        raise FileNotFoundError(f"missing extracted sequence: {sequence_dir}")
    if not source_groundtruth.is_file():
        raise FileNotFoundError(f"missing ground truth: {source_groundtruth}")

    groundtruth_rows = []
    rgb_rows = []
    original_rows = []
    for line_number, line in enumerate(
        source_groundtruth.read_text(encoding="utf-8").splitlines(), start=1
    ):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        fields = line.split()
        if len(fields) < 8:
            raise ValueError(
                f"malformed ground truth {source_groundtruth}:{line_number}: {line}"
            )
        original_timestamp = float(fields[0])
        frame_id = int(original_timestamp)
        if original_timestamp != frame_id:
            raise ValueError(f"non-integer ICL frame timestamp: {fields[0]}")
        image_path = sequence_dir / "rgb" / f"{frame_id}.png"
        if not image_path.is_file():
            raise FileNotFoundError(f"ground-truth frame has no RGB image: {image_path}")
        timestamp = original_timestamp + offset
        original_rows.append(line)
        groundtruth_rows.append(" ".join((f"{timestamp:g}", *fields[1:8])))
        rgb_rows.append(f"{timestamp:g} rgb/{frame_id}.png")

    if not groundtruth_rows:
        raise RuntimeError(f"no poses in {source_groundtruth}")
    (sequence_dir / "groundtruth_original.txt").write_text(
        "\n".join(original_rows) + "\n", encoding="utf-8"
    )
    (sequence_dir / "groundtruth.txt").write_text(
        "# timestamp tx ty tz qx qy qz qw\n"
        + "\n".join(groundtruth_rows)
        + "\n",
        encoding="utf-8",
    )
    (sequence_dir / "rgb.txt").write_text(
        "# timestamp filename\n" + "\n".join(rgb_rows) + "\n",
        encoding="utf-8",
    )
    return {
        "frames": len(rgb_rows),
        "timestamp_offset": offset,
        "timestamp_range": [
            float(groundtruth_rows[0].split()[0]),
            float(groundtruth_rows[-1].split()[0]),
        ],
        "source_groundtruth": str(source_groundtruth),
        "source_groundtruth_sha256": sha256(source_groundtruth),
    }


def main():
    args = parse_args()
    downloads_dir = args.downloads_dir or args.dataset_root / "downloads"
    manifest = {
        "format": "dpvo-icl-nuim-two-robot-v1",
        "camera": {
            "width": 640,
            "height": 480,
            "fx": 481.2,
            "fy": 480.0,
            "cx": 319.5,
            "cy": 239.5,
            "distortion": [0.0, 0.0, 0.0, 0.0, 0.0],
        },
        "timestamp_note": (
            "robot1 timestamps are offset by 10000 seconds only to make the two "
            "trajectory streams disjoint for joint EVO association"
        ),
        "sequences": {},
    }
    for sequence, groundtruth_name, offset in SEQUENCES:
        manifest["sequences"][sequence] = prepare_sequence(
            args.dataset_root / sequence,
            downloads_dir / groundtruth_name,
            offset,
        )

    output_path = args.dataset_root / "dpvo_icl_nuim_manifest.json"
    output_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
