"""Lossless JSON and g2o I/O for DPVO Sim(3) pose graphs."""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
from scipy.optimize import least_squares
from scipy.sparse import lil_matrix
from scipy.spatial.transform import Rotation

from .centralized import (
    CentralizedPgoResult,
    CentralizedRobotMapPGO,
    RobotMapConstraint,
    Sim3,
)


FORMAT_NAME = "dpvo_sim3_pose_graph"
FORMAT_VERSION = 1
RESIDUAL_ORDER = ("tx", "ty", "tz", "rx", "ry", "rz", "log_scale")
G2O_TANGENT_ORDER = ("rx", "ry", "rz", "tx", "ty", "tz", "log_scale")


def sim3_to_dict(transform: Sim3) -> dict:
    return {
        "translation": transform.translation.tolist(),
        "quaternion_xyzw": Rotation.from_matrix(transform.rotation).as_quat().tolist(),
        "scale": transform.scale,
    }


def sim3_from_dict(data: dict) -> Sim3:
    return Sim3(
        data["translation"],
        Rotation.from_quat(data["quaternion_xyzw"]).as_matrix(),
        data["scale"],
    )


@dataclass
class PoseGraphVertex:
    vertex_id: int
    robot_id: str
    session_id: str
    estimate: Sim3
    fixed: bool = False
    keyframe_id: Optional[int] = None
    timestamp: Optional[float] = None
    optimized_estimate: Optional[Sim3] = None


@dataclass
class PoseGraphEdge:
    edge_id: int
    source: int
    target: int
    measurement: Sim3
    information_diagonal: np.ndarray
    edge_type: str
    metadata: dict = field(default_factory=dict)

    def __post_init__(self):
        self.information_diagonal = np.asarray(
            self.information_diagonal,
            dtype=np.float64,
        ).reshape(7)


@dataclass
class Sim3PoseGraph:
    name: str
    vertices: list[PoseGraphVertex]
    edges: list[PoseGraphEdge]
    metadata: dict = field(default_factory=dict)

    def validate(self):
        vertex_ids = [vertex.vertex_id for vertex in self.vertices]
        if len(vertex_ids) != len(set(vertex_ids)):
            raise ValueError("pose graph contains duplicate vertex IDs")
        known = set(vertex_ids)
        for vertex in self.vertices:
            if vertex.estimate.scale <= 0:
                raise ValueError(f"vertex {vertex.vertex_id} has non-positive scale")
        for edge in self.edges:
            if edge.source not in known or edge.target not in known:
                raise ValueError(f"edge {edge.edge_id} references an unknown vertex")
            if edge.measurement.scale <= 0:
                raise ValueError(f"edge {edge.edge_id} has non-positive scale")
            if np.any(edge.information_diagonal <= 0):
                raise ValueError(f"edge {edge.edge_id} has non-positive information")
        return self

    def to_dict(self) -> dict:
        self.validate()
        return {
            "format": FORMAT_NAME,
            "version": FORMAT_VERSION,
            "name": self.name,
            "convention": {
                "vertex": "T_world_frame; p_world = s * R * p_frame + t",
                "edge": (
                    "source_to_target; p_target = s * R * p_source + t; "
                    "prediction = inverse(T_world_target) * T_world_source"
                ),
                "quaternion_order": "xyzw",
                "residual_order": list(RESIDUAL_ORDER),
            },
            "metadata": self.metadata,
            "vertices": [
                {
                    "id": vertex.vertex_id,
                    "robot_id": vertex.robot_id,
                    "session_id": vertex.session_id,
                    "keyframe_id": vertex.keyframe_id,
                    "timestamp": vertex.timestamp,
                    "fixed": vertex.fixed,
                    "estimate": sim3_to_dict(vertex.estimate),
                    "optimized_estimate": (
                        sim3_to_dict(vertex.optimized_estimate)
                        if vertex.optimized_estimate is not None
                        else None
                    ),
                }
                for vertex in self.vertices
            ],
            "edges": [
                {
                    "id": edge.edge_id,
                    "type": edge.edge_type,
                    "source": edge.source,
                    "target": edge.target,
                    "measurement": sim3_to_dict(edge.measurement),
                    "information_diagonal": edge.information_diagonal.tolist(),
                    "metadata": edge.metadata,
                }
                for edge in self.edges
            ],
        }

    @classmethod
    def from_dict(cls, data: dict):
        if data.get("format") != FORMAT_NAME:
            raise ValueError("not a DPVO Sim(3) pose graph")
        if int(data.get("version", -1)) != FORMAT_VERSION:
            raise ValueError(f"unsupported pose graph version {data.get('version')}")
        graph = cls(
            name=data["name"],
            metadata=data.get("metadata", {}),
            vertices=[
                PoseGraphVertex(
                    vertex_id=int(vertex["id"]),
                    robot_id=vertex["robot_id"],
                    session_id=vertex["session_id"],
                    keyframe_id=(
                        int(vertex["keyframe_id"])
                        if vertex.get("keyframe_id") is not None
                        else None
                    ),
                    timestamp=(
                        float(vertex["timestamp"])
                        if vertex.get("timestamp") is not None
                        else None
                    ),
                    fixed=bool(vertex.get("fixed", False)),
                    estimate=sim3_from_dict(vertex["estimate"]),
                    optimized_estimate=(
                        sim3_from_dict(vertex["optimized_estimate"])
                        if vertex.get("optimized_estimate") is not None
                        else None
                    ),
                )
                for vertex in data["vertices"]
            ],
            edges=[
                PoseGraphEdge(
                    edge_id=int(edge["id"]),
                    source=int(edge["source"]),
                    target=int(edge["target"]),
                    measurement=sim3_from_dict(edge["measurement"]),
                    information_diagonal=edge["information_diagonal"],
                    edge_type=edge["type"],
                    metadata=edge.get("metadata", {}),
                )
                for edge in data["edges"]
            ],
        )
        return graph.validate()


def _atomic_write(path: Path, contents: str):
    path = Path(path).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(contents)
    temporary.replace(path)


def write_json(graph: Sim3PoseGraph, path: Path):
    _atomic_write(Path(path), json.dumps(graph.to_dict(), indent=2) + "\n")


def read_json(path: Path) -> Sim3PoseGraph:
    return Sim3PoseGraph.from_dict(json.loads(Path(path).expanduser().read_text()))


def sim3_log(transform: Sim3):
    """Return g2o's [omega, upsilon, sigma] Sim(3) logarithm."""

    omega = Rotation.from_matrix(transform.rotation).as_rotvec()
    theta = float(np.linalg.norm(omega))
    sigma = float(np.log(transform.scale))
    omega_matrix = np.array(
        [
            [0.0, -omega[2], omega[1]],
            [omega[2], 0.0, -omega[0]],
            [-omega[1], omega[0], 0.0],
        ]
    )
    omega_squared = omega_matrix @ omega_matrix
    epsilon = 1e-5
    if abs(sigma) < epsilon:
        c = 1.0
        if theta < epsilon:
            a = 0.5
            b = 1.0 / 6.0
        else:
            theta_squared = theta * theta
            a = (1.0 - np.cos(theta)) / theta_squared
            b = (theta - np.sin(theta)) / (theta_squared * theta)
    else:
        scale = transform.scale
        c = (scale - 1.0) / sigma
        sigma_squared = sigma * sigma
        if theta < epsilon:
            a = ((sigma - 1.0) * scale + 1.0) / sigma_squared
            b = (
                (0.5 * sigma_squared - sigma + 1.0) * scale - 1.0
            ) / (sigma_squared * sigma)
        else:
            theta_squared = theta * theta
            denominator = theta_squared + sigma_squared
            scaled_sine = scale * np.sin(theta)
            scaled_cosine = scale * np.cos(theta)
            a = (
                scaled_sine * sigma + (1.0 - scaled_cosine) * theta
            ) / (theta * denominator)
            b = (
                c
                - (
                    (scaled_cosine - 1.0) * sigma
                    + scaled_sine * theta
                )
                / denominator
            ) / theta_squared
    w = a * omega_matrix + b * omega_squared + c * np.eye(3)
    upsilon = np.linalg.solve(w, transform.translation)
    return np.concatenate([omega, upsilon, [sigma]])


def _format_vector(values):
    return " ".join(f"{float(value):.17g}" for value in values)


def _upper_triangular(diagonal):
    information = np.diag(diagonal)
    return [information[row, col] for row in range(7) for col in range(row, 7)]


def write_g2o(graph: Sim3PoseGraph, path: Path, use_optimized: bool = False):
    """Write g2o's built-in Sim3 Expmap format.

    g2o serializes the inverse estimate/measurement as a seven-dimensional Lie
    logarithm, followed by four camera-calibration placeholders on vertices.
    """

    graph.validate()
    lines = [
        "# DPVO Sim(3) pose graph",
        "# tangent order: rx ry rz tx ty tz log_scale",
        "# edge direction: source frame -> target frame",
    ]
    for vertex in graph.vertices:
        lines.append(
            "# VERTEX_META %d %s %s %s"
            % (
                vertex.vertex_id,
                vertex.robot_id,
                vertex.session_id,
                vertex.keyframe_id if vertex.keyframe_id is not None else "map",
            )
        )
        estimate = (
            vertex.optimized_estimate
            if use_optimized and vertex.optimized_estimate is not None
            else vertex.estimate
        )
        # DPVO stores T_world_frame (camera-to-world). g2o's Sim3 vertex stores
        # its inverse, T_frame_world, and the file reader itself inverts the
        # serialized value. Therefore the line contains T_world_frame.
        serialized = sim3_log(estimate)
        lines.append(
            f"VERTEX_SIM3:EXPMAP {vertex.vertex_id} "
            f"{_format_vector(serialized)} 1 1 0 0"
        )
        if vertex.fixed:
            lines.append(f"FIX {vertex.vertex_id}")

    # JSON residual order is translation, rotation, log-scale. g2o orders the
    # Sim(3) tangent as rotation, translation, log-scale.
    reorder = [3, 4, 5, 0, 1, 2, 6]
    for edge in graph.edges:
        information = edge.information_diagonal[reorder]
        serialized = sim3_log(edge.measurement.inverse())
        lines.append(f"# EDGE_META {edge.edge_id} {edge.edge_type}")
        lines.append(
            f"EDGE_SIM3:EXPMAP {edge.source} {edge.target} "
            f"{_format_vector(serialized)} "
            f"{_format_vector(_upper_triangular(information))}"
        )
    _atomic_write(Path(path), "\n".join(lines) + "\n")


def _constraint_metadata(constraint: RobotMapConstraint):
    return {
        "query_robot": constraint.query_robot[0],
        "query_session": constraint.query_robot[1],
        "query_keyframe_id": constraint.query_keyframe_id,
        "match_robot": constraint.match_robot[0],
        "match_session": constraint.match_robot[1],
        "match_keyframe_id": constraint.match_keyframe_id,
        "bow_score": constraint.bow_score,
        "inliers": constraint.inliers,
        "inlier_ratio": constraint.inlier_ratio,
        "verification_method": constraint.verification_method,
        "weight": constraint.weight,
        "query_pose": sim3_to_dict(constraint.query_pose),
        "match_pose": sim3_to_dict(constraint.match_pose),
        "query_to_match": sim3_to_dict(constraint.query_to_match),
    }


def build_map_graph(
    constraints: Iterable[RobotMapConstraint],
    result: CentralizedPgoResult,
    anchor=None,
):
    constraints = list(constraints)
    solver = CentralizedRobotMapPGO(anchor=anchor)
    initial, anchors = solver.initialize(constraints)
    robots = sorted(initial)
    vertex_ids = {robot: index for index, robot in enumerate(robots)}
    vertices = [
        PoseGraphVertex(
            vertex_id=vertex_ids[robot],
            robot_id=robot[0],
            session_id=robot[1],
            estimate=initial[robot],
            optimized_estimate=result.transforms.get(robot),
            fixed=robot in anchors,
        )
        for robot in robots
    ]
    edges = [
        PoseGraphEdge(
            edge_id=index,
            source=vertex_ids[constraint.query_robot],
            target=vertex_ids[constraint.match_robot],
            measurement=constraint.map_measurement(),
            information_diagonal=np.full(7, max(constraint.weight, 1e-3)),
            edge_type="inter_robot_map_alignment",
            metadata=_constraint_metadata(constraint),
        )
        for index, constraint in enumerate(constraints)
    ]
    return Sim3PoseGraph(
        name="DPVO robot-map alignment graph",
        vertices=vertices,
        edges=edges,
        metadata={
            "cost": result.cost,
            "residual_norm": result.residual_norm,
            "solver_success": result.success,
        },
    )


def _robot_components(robot_ids, constraints):
    adjacency = {robot_id: set() for robot_id in robot_ids}
    for constraint in constraints:
        query = constraint.query_robot[0]
        match = constraint.match_robot[0]
        if query in adjacency and match in adjacency:
            adjacency[query].add(match)
            adjacency[match].add(query)
    components = []
    unseen = set(robot_ids)
    while unseen:
        root = min(unseen)
        stack = [root]
        component = set()
        while stack:
            robot = stack.pop()
            if robot in component:
                continue
            component.add(robot)
            stack.extend(adjacency[robot] - component)
        unseen -= component
        components.append(component)
    return components


def build_keyframe_graph(
    constraints: Iterable[RobotMapConstraint],
    result: CentralizedPgoResult,
    paths: dict[str, list[Sim3]],
    sessions: dict[str, str],
    anchor_robot_id: str,
    odometry_weight: float = 100.0,
    align_to_global: bool = True,
    timestamps: Optional[dict[str, list[float]]] = None,
):
    constraints = list(constraints)
    robot_ids = sorted(robot for robot, poses in paths.items() if poses)
    component_anchors = set()
    for component in _robot_components(robot_ids, constraints):
        component_anchors.add(
            anchor_robot_id if anchor_robot_id in component else min(component)
        )

    vertices = []
    lookup = {}
    for robot_id in robot_ids:
        session_id = sessions.get(robot_id, "unknown")
        map_transform = (
            result.transforms.get((robot_id, session_id), Sim3.identity())
            if align_to_global
            else Sim3.identity()
        )
        for keyframe_id, local_pose in enumerate(paths[robot_id]):
            vertex_id = len(vertices)
            lookup[(robot_id, session_id, keyframe_id)] = vertex_id
            vertices.append(
                PoseGraphVertex(
                    vertex_id=vertex_id,
                    robot_id=robot_id,
                    session_id=session_id,
                    keyframe_id=keyframe_id,
                    timestamp=(
                        float(timestamps[robot_id][keyframe_id])
                        if timestamps is not None
                        and robot_id in timestamps
                        and keyframe_id < len(timestamps[robot_id])
                        else None
                    ),
                    estimate=map_transform.compose(local_pose),
                    fixed=(keyframe_id == 0 and robot_id in component_anchors),
                )
            )

    edges = []
    for robot_id in robot_ids:
        session_id = sessions.get(robot_id, "unknown")
        poses = paths[robot_id]
        for source_keyframe in range(len(poses) - 1):
            target_keyframe = source_keyframe + 1
            edges.append(
                PoseGraphEdge(
                    edge_id=len(edges),
                    source=lookup[(robot_id, session_id, source_keyframe)],
                    target=lookup[(robot_id, session_id, target_keyframe)],
                    measurement=(
                        poses[target_keyframe].inverse().compose(poses[source_keyframe])
                    ),
                    information_diagonal=np.full(7, odometry_weight),
                    edge_type="visual_odometry",
                    metadata={
                        "robot_id": robot_id,
                        "session_id": session_id,
                        "source_keyframe_id": source_keyframe,
                        "target_keyframe_id": target_keyframe,
                    },
                )
            )

    skipped_loops = 0
    for constraint in constraints:
        query_id = lookup.get(
            (
                constraint.query_robot[0],
                constraint.query_robot[1],
                constraint.query_keyframe_id,
            )
        )
        match_id = lookup.get(
            (
                constraint.match_robot[0],
                constraint.match_robot[1],
                constraint.match_keyframe_id,
            )
        )
        if query_id is None or match_id is None:
            skipped_loops += 1
            continue
        edges.append(
            PoseGraphEdge(
                edge_id=len(edges),
                source=query_id,
                target=match_id,
                measurement=constraint.query_to_match,
                information_diagonal=np.full(7, max(constraint.weight, 1e-3)),
                edge_type="inter_robot_loop_closure",
                metadata=_constraint_metadata(constraint),
            )
        )

    return Sim3PoseGraph(
        name="DPVO keyframe Sim(3) pose graph",
        vertices=vertices,
        edges=edges,
        metadata={
            "odometry_weight": odometry_weight,
            "inter_robot_loop_count": len(constraints) - skipped_loops,
            "skipped_inter_robot_loops": skipped_loops,
            "initialization": (
                "centralized_map_alignment"
                if align_to_global
                else "per_robot_local_map_original_scale"
            ),
        },
    )


def split_keyframe_graph_by_robot(
    graph: Sim3PoseGraph,
) -> dict[str, Sim3PoseGraph]:
    """Extract robot-local graphs while preserving their original scale."""

    graph.validate()
    output = {}
    for robot_id in sorted({vertex.robot_id for vertex in graph.vertices}):
        selected = [
            vertex for vertex in graph.vertices if vertex.robot_id == robot_id
        ]
        remap = {
            vertex.vertex_id: new_id for new_id, vertex in enumerate(selected)
        }
        vertices = [
            PoseGraphVertex(
                vertex_id=remap[vertex.vertex_id],
                robot_id=vertex.robot_id,
                session_id=vertex.session_id,
                keyframe_id=vertex.keyframe_id,
                timestamp=vertex.timestamp,
                estimate=vertex.estimate,
                fixed=vertex.keyframe_id == 0,
                optimized_estimate=None,
            )
            for vertex in selected
        ]
        edges = []
        for edge in graph.edges:
            if edge.source not in remap or edge.target not in remap:
                continue
            edges.append(
                PoseGraphEdge(
                    edge_id=len(edges),
                    source=remap[edge.source],
                    target=remap[edge.target],
                    measurement=edge.measurement,
                    information_diagonal=edge.information_diagonal.copy(),
                    edge_type=edge.edge_type,
                    metadata=dict(edge.metadata),
                )
            )
        output[robot_id] = Sim3PoseGraph(
            name=f"DPVO {robot_id} original-scale keyframe graph",
            vertices=vertices,
            edges=edges,
            metadata={
                "robot_id": robot_id,
                "session_ids": sorted({vertex.session_id for vertex in selected}),
                "coordinate_frame": "robot_local_map",
                "initialization": "original_dpvo_scale_unoptimized",
                "source_graph": graph.name,
                "odometry_weight": graph.metadata.get("odometry_weight"),
            },
        ).validate()
    return output


def restore_original_robot_scales(
    graph: Sim3PoseGraph,
    map_transforms: dict[tuple[str, str], Sim3],
) -> Sim3PoseGraph:
    """Undo centralized map alignment on an already exported keyframe graph."""

    graph.validate()
    vertices = []
    missing = set()
    for vertex in graph.vertices:
        key = (vertex.robot_id, vertex.session_id)
        transform = map_transforms.get(key)
        if transform is None:
            missing.add(key)
            transform = Sim3.identity()
        vertices.append(
            PoseGraphVertex(
                vertex_id=vertex.vertex_id,
                robot_id=vertex.robot_id,
                session_id=vertex.session_id,
                keyframe_id=vertex.keyframe_id,
                timestamp=vertex.timestamp,
                estimate=transform.inverse().compose(vertex.estimate),
                fixed=vertex.fixed,
                optimized_estimate=None,
            )
        )
    metadata = dict(graph.metadata)
    metadata.update(
        {
            "initialization": "per_robot_local_map_original_scale",
            "coordinate_frame": "independent_robot_local_maps",
            "missing_map_transforms": [list(key) for key in sorted(missing)],
        }
    )
    return Sim3PoseGraph(
        name="DPVO unoptimized keyframe graph at original robot scales",
        vertices=vertices,
        edges=[
            PoseGraphEdge(
                edge_id=edge.edge_id,
                source=edge.source,
                target=edge.target,
                measurement=edge.measurement,
                information_diagonal=edge.information_diagonal.copy(),
                edge_type=edge.edge_type,
                metadata=dict(edge.metadata),
            )
            for edge in graph.edges
        ],
        metadata=metadata,
    ).validate()


@dataclass
class PoseGraphOptimizationResult:
    success: bool
    cost: float
    residual_norm: float
    iterations: int


def optimize_pose_graph(
    graph: Sim3PoseGraph,
    loss: str = "huber",
    f_scale: float = 1.0,
    max_nfev: int = 200,
    min_information: float = 0.0,
    edge_types: Optional[set[str]] = None,
):
    """Optimize a JSON pose graph with the same residual used online."""

    graph.validate()
    vertices = {vertex.vertex_id: vertex for vertex in graph.vertices}
    edges = [
        edge
        for edge in graph.edges
        if np.min(edge.information_diagonal) >= min_information
        and (edge_types is None or edge.edge_type in edge_types)
    ]
    fixed = {vertex.vertex_id for vertex in graph.vertices if vertex.fixed}

    adjacency = {vertex_id: set() for vertex_id in vertices}
    for edge in edges:
        adjacency[edge.source].add(edge.target)
        adjacency[edge.target].add(edge.source)
    unseen = set(vertices)
    while unseen:
        root = min(unseen)
        stack = [root]
        component = set()
        while stack:
            vertex_id = stack.pop()
            if vertex_id in component:
                continue
            component.add(vertex_id)
            stack.extend(adjacency[vertex_id] - component)
        unseen -= component
        if not fixed.intersection(component):
            fixed.add(min(component))

    variables = sorted(set(vertices) - fixed)
    offsets = {vertex_id: 7 * index for index, vertex_id in enumerate(variables)}
    initial = np.concatenate(
        [vertices[vertex_id].estimate.to_vector() for vertex_id in variables]
    ) if variables else np.empty(0)

    def unpack(vector):
        estimates = {
            vertex_id: vertex.estimate
            for vertex_id, vertex in vertices.items()
            if vertex_id in fixed
        }
        for vertex_id in variables:
            offset = offsets[vertex_id]
            estimates[vertex_id] = Sim3.from_vector(vector[offset : offset + 7])
        return estimates

    def residual(vector):
        estimates = unpack(vector)
        output = []
        for edge in edges:
            predicted = estimates[edge.target].inverse().compose(
                estimates[edge.source]
            )
            error = edge.measurement.inverse().compose(predicted)
            raw = np.concatenate(
                [
                    error.translation,
                    Rotation.from_matrix(error.rotation).as_rotvec(),
                    [np.log(error.scale)],
                ]
            )
            output.extend(np.sqrt(edge.information_diagonal) * raw)
        return np.asarray(output, dtype=np.float64)

    if not variables or not edges:
        estimates = unpack(initial)
        for vertex_id, estimate in estimates.items():
            vertices[vertex_id].optimized_estimate = estimate
        norm = float(np.linalg.norm(residual(initial))) if edges else 0.0
        return PoseGraphOptimizationResult(True, 0.0, norm, 0)

    sparsity = lil_matrix((7 * len(edges), 7 * len(variables)), dtype=int)
    for edge_index, edge in enumerate(edges):
        rows = slice(7 * edge_index, 7 * edge_index + 7)
        for vertex_id in (edge.source, edge.target):
            if vertex_id in offsets:
                offset = offsets[vertex_id]
                sparsity[rows, offset : offset + 7] = 1

    result = least_squares(
        residual,
        initial,
        loss=loss,
        f_scale=f_scale,
        max_nfev=max_nfev,
        jac_sparsity=sparsity.tocsr(),
    )
    estimates = unpack(result.x)
    for vertex_id, estimate in estimates.items():
        vertices[vertex_id].optimized_estimate = estimate
    final_residual = residual(result.x)
    return PoseGraphOptimizationResult(
        bool(result.success),
        float(result.cost),
        float(np.linalg.norm(final_residual)),
        int(result.nfev),
    )


def _default_optimized_path(input_path: Path):
    if input_path.suffix == ".json":
        return input_path.with_name(f"{input_path.stem}.optimized.json")
    return input_path.with_name(f"{input_path.name}.optimized.json")


def main(argv=None):
    parser = argparse.ArgumentParser(description="Optimize a DPVO Sim(3) graph")
    parser.add_argument("input", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--loss", default="huber")
    parser.add_argument("--f-scale", type=float, default=1.0)
    parser.add_argument("--max-nfev", type=int, default=200)
    parser.add_argument("--min-information", type=float, default=0.0)
    parser.add_argument("--edge-type", action="append", dest="edge_types")
    args = parser.parse_args(argv)

    graph = read_json(args.input)
    result = optimize_pose_graph(
        graph,
        loss=args.loss,
        f_scale=args.f_scale,
        max_nfev=args.max_nfev,
        min_information=args.min_information,
        edge_types=set(args.edge_types) if args.edge_types else None,
    )
    graph.metadata["offline_optimization"] = {
        "success": result.success,
        "cost": result.cost,
        "residual_norm": result.residual_norm,
        "iterations": result.iterations,
        "loss": args.loss,
        "f_scale": args.f_scale,
        "min_information": args.min_information,
        "edge_types": args.edge_types,
    }
    output = args.output or _default_optimized_path(args.input)
    write_json(graph, output)
    write_g2o(graph, output.with_suffix(".g2o"), use_optimized=True)
    print(
        "optimized %d vertices / %d edges: residual %.6f -> %s"
        % (len(graph.vertices), len(graph.edges), result.residual_norm, output)
    )


if __name__ == "__main__":
    main()
