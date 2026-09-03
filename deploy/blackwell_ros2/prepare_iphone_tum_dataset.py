#!/usr/bin/env python3
"""Convert iPhone ultra-wide captures to the TUM RGB layout used by DPVO."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import subprocess


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--sequence", action="append", required=True)
    parser.add_argument("--width", type=int, default=1920)
    parser.add_argument("--height", type=int, default=1080)
    parser.add_argument("--jpeg-quality", type=int, default=3)
    return parser.parse_args()


def source_video_frame_count(path: Path):
    command = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=nb_frames",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        str(path),
    ]
    value = subprocess.run(
        command,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    try:
        frame_count = int(value)
    except ValueError:
        return None
    return frame_count if frame_count > 0 else None


def selected_ultra_timestamps(path: Path, source_frame_count=None):
    selected = []
    skipped = 0
    with path.open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            if row["stream"] != "ultra":
                continue
            frame_index = int(row["frame_index"])
            if source_frame_count is not None and frame_index >= source_frame_count:
                skipped += 1
                continue
            if frame_index % 2 == 1:
                selected.append(float(row["relative_seconds"]))
    if not selected:
        raise RuntimeError(f"no odd ultra frames found in {path}")
    if any(current <= previous for previous, current in zip(selected, selected[1:])):
        raise ValueError(f"non-monotonic ultra timestamps in {path}")
    return selected, skipped


def convert_sequence(args, sequence: str):
    source = args.source_root / sequence
    video = source / "ultra.mov"
    timecodes = source / "frame_timecodes.csv"
    for path in (video, timecodes):
        if not path.is_file():
            raise FileNotFoundError(path)

    output = args.output_root / sequence
    if output.exists():
        raise FileExistsError(
            f"refusing to overwrite prepared sequence: {output}"
        )
    rgb_dir = output / "rgb"
    rgb_dir.mkdir(parents=True)
    source_frame_count = source_video_frame_count(video)
    timestamps, skipped_timecodes = selected_ultra_timestamps(
        timecodes,
        source_frame_count,
    )
    if skipped_timecodes:
        print(
            f"[{sequence}] ignoring {skipped_timecodes} timestamp rows beyond "
            f"the current {source_frame_count}-frame video",
            flush=True,
        )
    video_filter = (
        "select=eq(mod(n\\,2)\\,1),"
        f"scale={args.width}:{args.height}:flags=bicubic"
    )
    command = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "warning",
        "-i",
        str(video),
        "-vf",
        video_filter,
        "-fps_mode",
        "vfr",
        "-q:v",
        str(args.jpeg_quality),
        str(rgb_dir / "%06d.jpg"),
    ]
    print(f"[{sequence}] extracting {len(timestamps)} frames", flush=True)
    subprocess.run(command, check=True)
    images = sorted(rgb_dir.glob("*.jpg"))
    if len(images) != len(timestamps):
        raise RuntimeError(
            f"{sequence}: extracted {len(images)} images for "
            f"{len(timestamps)} timestamps"
        )
    lines = [
        f"{timestamp:.9f} rgb/{index:06d}.jpg"
        for index, timestamp in enumerate(timestamps, start=1)
    ]
    (output / "rgb.txt").write_text("\n".join(lines) + "\n")
    print(
        f"[{sequence}] {len(images)} frames, "
        f"{timestamps[0]:.6f}–{timestamps[-1]:.6f} s",
        flush=True,
    )


def main():
    args = parse_args()
    if args.width <= 0 or args.height <= 0:
        raise ValueError("image dimensions must be positive")
    if not 1 <= args.jpeg_quality <= 31:
        raise ValueError("--jpeg-quality must be in [1, 31]")
    args.output_root.mkdir(parents=True, exist_ok=True)
    for sequence in args.sequence:
        convert_sequence(args, sequence)


if __name__ == "__main__":
    main()
