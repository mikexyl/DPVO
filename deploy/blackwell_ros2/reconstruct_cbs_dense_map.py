#!/usr/bin/env python3
"""Reconstruct a dense CBS map from staged DPVO keyframes with DA3-Large."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

# Keep the standalone script usable from a source checkout; the Pixi task's
# working directory is deploy/blackwell_ros2 rather than the repository root.
REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from dpvo.loop_closure.dense_reconstruction import (
    DEFAULT_CODE_REVISION,
    DEFAULT_MODEL,
    DEFAULT_WEIGHTS_REVISION,
    FusionSettings,
    InferenceSettings,
    LazyDa3Backend,
    ReconstructionPaths,
    export_reconstruction,
    infer_jobs,
    load_reconstruction,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run resumable pose-estimated DA3 two-view inference, convert depth by "
            "the DA3/CBS baseline ratio, and fuse it in the CBS robot0 frame."
        )
    )
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--tracking-dir", type=Path)
    parser.add_argument("--graph", type=Path, dest="graph_path")
    parser.add_argument("--cbs-csv", type=Path)
    parser.add_argument("--sparse-ply", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--code-revision", default=DEFAULT_CODE_REVISION)
    parser.add_argument("--weights-revision", default=DEFAULT_WEIGHTS_REVISION)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--process-res", type=int, default=504)
    parser.add_argument(
        "--confidence-percentile",
        type=float,
        default=40.0,
        help="Reject confidence values below this per-view percentile.",
    )
    parser.add_argument("--far-depth-percentile", type=float, default=99.5)
    parser.add_argument(
        "--reprojection-relative-tolerance", type=float, default=0.10
    )
    parser.add_argument(
        "--require-reprojection-overlap",
        action="store_true",
        help=(
            "Keep a depth sample only if it projects inside the partner view and "
            "passes the relative-depth consistency check."
        ),
    )
    parser.add_argument(
        "--depth-scale-refinement",
        choices=("none", "dpvo_keypoints"),
        default="none",
        help="Optionally refine each cached pair scale using sparse DPVO patch depths.",
    )
    parser.add_argument("--dpvo-keypoint-confidence-percentile", type=float, default=50.0)
    parser.add_argument("--dpvo-keypoint-min-depth", type=float, default=0.05)
    parser.add_argument("--dpvo-keypoint-max-depth", type=float, default=20.0)
    parser.add_argument("--dpvo-keypoint-min-matches-per-view", type=int, default=8)
    parser.add_argument("--pixel-stride", type=int, default=4)
    parser.add_argument(
        "--voxel-size",
        type=float,
        default=0.02,
        help="Voxel edge length in CBS robot0-frame units.",
    )
    parser.add_argument(
        "--minimum-baseline",
        type=float,
        default=1e-7,
        help="Pairs at or below this CBS-unit baseline use temporal fallback.",
    )
    parser.add_argument(
        "--intra-pair-gap",
        type=int,
        default=1,
        help="Keyframe-index gap within each temporal DA3 pair.",
    )
    parser.add_argument(
        "--intra-pair-step",
        type=int,
        default=2,
        help="Keyframe-index step between consecutive temporal pair starts.",
    )
    parser.add_argument(
        "--minimum-da3-baseline",
        type=float,
        default=1e-8,
        help="Reject DA3 predicted baselines at or below this arbitrary-gauge value.",
    )
    parser.add_argument("--stage", choices=("infer", "fuse", "all"), default="all")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Replace existing pair caches during inference, even if valid.",
    )
    parser.add_argument(
        "--max-pairs",
        type=int,
        help="Restrict the deterministic job prefix for smoke tests.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.process_res < 14:
        raise ValueError("--process-res must be at least 14")
    if args.minimum_baseline < 0.0:
        raise ValueError("--minimum-baseline must be non-negative")
    if args.intra_pair_gap < 1 or args.intra_pair_step < 1:
        raise ValueError("--intra-pair-gap and --intra-pair-step must be positive")
    if args.minimum_da3_baseline <= 0.0:
        raise ValueError("--minimum-da3-baseline must be positive")
    if args.max_pairs is not None and args.max_pairs < 1:
        raise ValueError("--max-pairs must be positive")
    if args.stage == "fuse" and args.force:
        raise ValueError("--force only applies to stages that perform inference")
    if args.sparse_ply is not None and not args.sparse_ply.expanduser().is_file():
        raise FileNotFoundError(f"--sparse-ply does not exist: {args.sparse_ply}")

    paths = ReconstructionPaths.from_run_dir(
        args.run_dir,
        tracking_dir=args.tracking_dir,
        graph_path=args.graph_path,
        cbs_csv=args.cbs_csv,
        sparse_ply=args.sparse_ply,
        output_dir=args.output_dir,
    )
    loaded = load_reconstruction(
        paths,
        minimum_baseline=args.minimum_baseline,
        intra_pair_gap=args.intra_pair_gap,
        intra_pair_step=args.intra_pair_step,
    )
    selected_jobs = loaded.jobs[: args.max_pairs] if args.max_pairs else loaded.jobs
    inference_settings = InferenceSettings(
        model=args.model,
        code_revision=args.code_revision,
        weights_revision=args.weights_revision,
        process_res=args.process_res,
        minimum_da3_baseline=args.minimum_da3_baseline,
    ).validate()
    fusion_settings = FusionSettings(
        confidence_percentile=args.confidence_percentile,
        far_depth_percentile=args.far_depth_percentile,
        reprojection_relative_tolerance=args.reprojection_relative_tolerance,
        require_reprojection_overlap=args.require_reprojection_overlap,
        depth_scale_refinement=args.depth_scale_refinement,
        dpvo_keypoint_confidence_percentile=args.dpvo_keypoint_confidence_percentile,
        dpvo_keypoint_min_depth=args.dpvo_keypoint_min_depth,
        dpvo_keypoint_max_depth=args.dpvo_keypoint_max_depth,
        dpvo_keypoint_min_matches_per_view=args.dpvo_keypoint_min_matches_per_view,
        pixel_stride=args.pixel_stride,
        voxel_size=args.voxel_size,
    ).validate()

    base_count = sum(job.kind.startswith("intra_robot") for job in loaded.jobs)
    loop_count = sum(
        job.kind == "inter_robot_loop_closure" for job in loaded.jobs
    )
    keyframe_count = sum(
        artifact.keyframe_count for artifact in loaded.artifacts.values()
    )
    print(
        f"Validated {len(loaded.artifacts)} robots, {keyframe_count} keyframes; "
        f"scheduled {base_count} base + {loop_count} loop = {len(loaded.jobs)} pairs"
    )
    if len(selected_jobs) != len(loaded.jobs):
        print(f"Smoke-test prefix: processing {len(selected_jobs)} pairs")

    cache_status = None
    model_revisions = {}
    if args.stage in ("infer", "all"):
        backend = LazyDa3Backend(inference_settings, args.device)
        cache_status = infer_jobs(
            selected_jobs,
            loaded.artifacts,
            loaded.cbs_poses,
            paths.output_dir / "cache" / "pairs",
            inference_settings,
            backend,
            force=args.force,
        )
        model_revisions = backend.revisions
        print(
            "Pair cache complete: "
            f"{cache_status['reused']} reused, {cache_status['written']} written, "
            f"{cache_status['invalidated']} invalidated, "
            f"{cache_status['rejected']} rejected"
        )

    if args.stage in ("fuse", "all"):
        manifest = export_reconstruction(
            loaded,
            selected_jobs,
            inference_settings,
            fusion_settings,
            cache_status,
            model_revisions,
        )
        print(
            f"Wrote {manifest['point_counts']['dense_global']} dense points to "
            f"{paths.output_dir}"
        )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileNotFoundError, RuntimeError, ValueError) as error:
        print(f"dense reconstruction failed: {error}", file=sys.stderr)
        raise SystemExit(2) from error
