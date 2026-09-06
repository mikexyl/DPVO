"""Run CBS and centralized Sim(3) PGO from a verified raw pose graph."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

from .pose_graph import read_json, write_g2o


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_copy(source: Path, destination: Path) -> None:
    temporary = destination.with_name(f".{destination.name}.tmp")
    shutil.copyfile(source, temporary)
    temporary.replace(destination)


def validate_dpgo_input(path: Path, *, allow_legacy_unoptimized_input=False):
    graph = read_json(path)
    legacy_unoptimized = (
        allow_legacy_unoptimized_input
        and graph.metadata.get("initialization")
        == "per_robot_local_map_original_scale"
    )
    if (
        graph.metadata.get("pipeline_stage") != "geometric_verification"
        and not legacy_unoptimized
    ):
        raise ValueError(
            "DPGO input must be the stage-two geometric-verification graph; "
            "legacy raw exports require --allow-legacy-unoptimized-input"
        )
    if (
        not legacy_unoptimized
        and graph.metadata.get("input_contains_global_optimization", True)
    ):
        raise ValueError("DPGO input is contaminated by global optimization")
    if any(vertex.optimized_estimate is not None for vertex in graph.vertices):
        raise ValueError("DPGO input contains optimized vertex estimates")
    loop_count = sum(
        edge.edge_type == "inter_robot_loop_closure" for edge in graph.edges
    )
    if loop_count == 0:
        raise ValueError("DPGO input contains no verified inter-robot loops")
    robot_by_vertex = {
        vertex.vertex_id: vertex.robot_id for vertex in graph.vertices
    }
    robot_ids = set(robot_by_vertex.values())
    adjacency = {robot_id: set() for robot_id in robot_ids}
    for edge in graph.edges:
        if edge.edge_type != "inter_robot_loop_closure":
            continue
        source_robot = robot_by_vertex[edge.source]
        target_robot = robot_by_vertex[edge.target]
        adjacency[source_robot].add(target_robot)
        adjacency[target_robot].add(source_robot)
    reached = set()
    frontier = [min(robot_ids)]
    while frontier:
        robot_id = frontier.pop()
        if robot_id in reached:
            continue
        reached.add(robot_id)
        frontier.extend(adjacency[robot_id] - reached)
    if reached != robot_ids:
        missing = sorted(robot_ids - reached)
        raise ValueError(
            "verified loop graph does not connect every robot; disconnected: "
            + ", ".join(missing)
        )
    return graph, loop_count


def build_command(args, input_graph: Path, output_dir: Path):
    executable = Path(args.cbs_executable).expanduser().resolve()
    if not executable.is_file():
        resolved = shutil.which(args.cbs_executable)
        if resolved is None:
            raise FileNotFoundError(f"CBS executable not found: {args.cbs_executable}")
        executable = Path(resolved)
    values = {
        "input_graph": input_graph,
        "output_dir": output_dir,
        "iterations": args.iterations,
        "stage_mode": args.stage_mode,
        "anchor_start_iteration": args.anchor_start_iteration,
        "anchor_stage_probability": args.anchor_stage_probability,
        "pose_warmup_iterations": args.pose_warmup_iterations,
        "pose_block_iterations": args.pose_block_iterations,
        "anchor_block_iterations": args.anchor_block_iterations,
        "target_hellinger": args.target_hellinger,
        "hellinger_quadratic_term": str(args.hellinger_quadratic_term).lower(),
        "contract_alpha": args.contract_alpha,
        "d_reset": args.d_reset,
        "sim3_covariance_transport": args.sim3_covariance_transport,
        "bootstrap_robot_anchors": str(args.bootstrap_robot_anchors).lower(),
        "odom_scale_sigma": args.odom_scale_sigma,
        "inter_loop_scale_sigma": args.inter_loop_scale_sigma,
        "random_seed": args.random_seed,
        "huber_k": args.huber_k,
        "run_cbs": str(getattr(args, "run_cbs", True)).lower(),
        "run_centralized": "true",
        "run_explicit_anchor_centralized": "true",
        "centralized_max_iterations": args.centralized_max_iterations,
        "write_rerun_rrd": str(args.write_rerun_rrd).lower(),
        "rerun_iteration_stride": args.rerun_iteration_stride,
        "trajectory_snapshot_iterations": args.trajectory_snapshot_iterations,
        "rerun_stream": str(args.rerun_stream).lower(),
        "rerun_url": args.rerun_url,
    }
    # Preserve replay compatibility with archived solvers when this optional
    # recording feature is not requested.
    if getattr(args, "rerun_factor_graphs", False):
        values["rerun_factor_graphs"] = "true"
    if getattr(args, "pose_beliefs_only", False):
        values["pose_beliefs_only"] = "true"
    return [str(executable)] + [f"--{key}={value}" for key, value in values.items()]


def run(args):
    source_path = Path(args.input_graph).expanduser().resolve()
    graph, loop_count = validate_dpgo_input(
        source_path,
        allow_legacy_unoptimized_input=args.allow_legacy_unoptimized_input,
    )
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    input_path = output_dir / "input_keyframes_unoptimized.json"
    _atomic_copy(source_path, input_path)
    write_g2o(graph, output_dir / "input_keyframes_unoptimized.g2o")
    command = build_command(args, input_path, output_dir)
    provenance = {
        "format": "dpvo_offline_dpgo_run",
        "version": 1,
        "source_graph": str(source_path),
        "source_sha256": _sha256(source_path),
        "copied_input_graph": str(input_path),
        "copied_input_sha256": _sha256(input_path),
        "copied_input_is_byte_identical": _sha256(source_path)
        == _sha256(input_path),
        "verified_inter_robot_loops": loop_count,
        "input_contains_global_optimization": False,
        "legacy_unoptimized_input_accepted": bool(
            args.allow_legacy_unoptimized_input
            and graph.metadata.get("pipeline_stage") != "geometric_verification"
        ),
        "command": command,
        "status": "prepared" if args.dry_run else "running",
    }
    provenance_path = output_dir / "offline_dpgo_provenance.json"
    _atomic_json(provenance_path, provenance)
    if args.dry_run:
        print(" ".join(command))
        return
    completed = subprocess.run(command, check=False)
    provenance["return_code"] = int(completed.returncode)
    provenance["status"] = "complete" if completed.returncode == 0 else "failed"
    _atomic_json(provenance_path, provenance)
    if completed.returncode != 0:
        raise subprocess.CalledProcessError(completed.returncode, command)


def _parser():
    parser = argparse.ArgumentParser(
        description="Offline CBS and centralized Sim(3) pose-graph optimization"
    )
    parser.add_argument("--input-graph", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--cbs-executable", required=True)
    parser.add_argument("--iterations", type=int, default=1000)
    parser.add_argument(
        "--stage-mode",
        choices=("random", "fixed", "alternating"),
        default="alternating",
    )
    parser.add_argument("--anchor-start-iteration", type=int, default=30)
    parser.add_argument(
        "--pose-beliefs-only",
        action="store_true",
        help=(
            "fixed-stage ablation: initialize and exchange pose beliefs without "
            "optimization; start ordinary CBS only at the anchor stage"
        ),
    )
    parser.add_argument("--anchor-stage-probability", type=float, default=0.5)
    parser.add_argument("--pose-warmup-iterations", type=int, default=0)
    parser.add_argument("--pose-block-iterations", type=int, default=20)
    parser.add_argument("--anchor-block-iterations", type=int, default=20)
    parser.add_argument("--target-hellinger", type=float, default=0.1)
    parser.add_argument(
        "--hellinger-quadratic-term",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="include the mean-covariance quadratic term in the CBS Hellinger step",
    )
    parser.add_argument("--contract-alpha", type=float, default=0.95)
    parser.add_argument("--d-reset", type=float, default=0.6)
    parser.add_argument(
        "--sim3-covariance-transport",
        choices=("bernoulli", "adjoint", "none"),
        default="bernoulli",
        help="Sim(3) covariance transport used by CBS; none keeps covariance unchanged",
    )
    parser.add_argument(
        "--bootstrap-robot-anchors",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "initialize centralized solvers with a global inter-robot spanning "
            "tree (default); never changes CBS's local-chart initialization"
        ),
    )
    parser.add_argument(
        "--run-cbs", action=argparse.BooleanOptionalAction, default=True,
        help="run CBS as well as the centralized baselines; --no-run-cbs reruns only centralized",
    )
    parser.add_argument(
        "--allow-legacy-unoptimized-input",
        action="store_true",
        help=(
            "accept a pre-stage-metadata raw graph only when it declares "
            "per_robot_local_map_original_scale; optimized vertices and "
            "disconnected robot graphs remain rejected"
        ),
    )
    parser.add_argument("--odom-scale-sigma", type=float, default=-1.0)
    parser.add_argument("--inter-loop-scale-sigma", type=float, default=-1.0)
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--huber-k", type=float, default=-1.0)
    parser.add_argument("--centralized-max-iterations", type=int, default=300)
    parser.add_argument("--write-rerun-rrd", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--rerun-iteration-stride", type=int, default=10)
    parser.add_argument(
        "--rerun-factor-graphs",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="record each robot's live optimizer factors at every recorded iteration",
    )
    parser.add_argument(
        "--trajectory-snapshot-iterations",
        default="",
        help="comma-separated CBS iterations whose trajectories are saved",
    )
    parser.add_argument("--rerun-stream", action="store_true")
    parser.add_argument(
        "--rerun-url", default="rerun+http://127.0.0.1:9876/proxy"
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser


def _atomic_json(path: Path, data: dict) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(data, indent=2) + "\n")
    temporary.replace(path)


def main(argv=None):
    run(_parser().parse_args(argv))


if __name__ == "__main__":
    main()
