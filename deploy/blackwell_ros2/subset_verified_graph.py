#!/usr/bin/env python3
"""Extract a connected robot component from a verified DPVO pose graph."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from dpvo.loop_closure.pose_graph import (
    PoseGraphEdge,
    PoseGraphVertex,
    Sim3PoseGraph,
    read_json,
    write_g2o,
    write_json,
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def subset_graph(
    graph: Sim3PoseGraph, robot_ids: list[str], anchor_robot: str
) -> Sim3PoseGraph:
    selected_robots = set(robot_ids)
    available = {vertex.robot_id for vertex in graph.vertices}
    unknown = selected_robots - available
    if unknown:
        raise ValueError(f"unknown robot IDs: {sorted(unknown)}")
    if len(selected_robots) != len(robot_ids):
        raise ValueError("duplicate robot IDs")
    if anchor_robot not in selected_robots:
        raise ValueError("anchor robot must be in the selected component")
    if graph.metadata.get("pipeline_stage") != "geometric_verification":
        raise ValueError("source must be a geometric-verification graph")
    if graph.metadata.get("input_contains_global_optimization", True):
        raise ValueError("source graph contains global optimization")

    selected = [
        vertex for vertex in graph.vertices if vertex.robot_id in selected_robots
    ]
    remap = {
        vertex.vertex_id: new_id for new_id, vertex in enumerate(selected)
    }
    anchor_candidates = [
        vertex for vertex in selected if vertex.robot_id == anchor_robot
    ]
    anchor_vertex = min(
        anchor_candidates,
        key=lambda vertex: (
            vertex.keyframe_id if vertex.keyframe_id is not None else 10**18,
            vertex.vertex_id,
        ),
    )
    vertices = [
        PoseGraphVertex(
            vertex_id=remap[vertex.vertex_id],
            robot_id=vertex.robot_id,
            session_id=vertex.session_id,
            keyframe_id=vertex.keyframe_id,
            timestamp=vertex.timestamp,
            estimate=vertex.estimate,
            fixed=vertex.vertex_id == anchor_vertex.vertex_id,
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
    loop_count = sum(
        edge.edge_type == "inter_robot_loop_closure" for edge in edges
    )
    if loop_count == 0:
        raise ValueError("selected component contains no inter-robot loops")
    metadata = dict(graph.metadata)
    metadata.update(
        {
            "pipeline_stage": "geometric_verification",
            "input_contains_global_optimization": False,
            "component_policy": "optimize_each_connected_component",
            "selected_robot_ids": robot_ids,
            "excluded_robot_ids": sorted(available - selected_robots),
            "anchor_robot_id": anchor_robot,
            "inter_robot_loop_count": loop_count,
        }
    )
    return Sim3PoseGraph(
        name=f"{graph.name} component {'-'.join(robot_ids)}",
        vertices=vertices,
        edges=edges,
        metadata=metadata,
    ).validate()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-base", type=Path, required=True)
    parser.add_argument("--robot-ids", nargs="+", required=True)
    parser.add_argument("--anchor-robot", required=True)
    args = parser.parse_args()

    source = args.input.expanduser().resolve()
    output_base = args.output_base.expanduser().resolve()
    graph = subset_graph(read_json(source), args.robot_ids, args.anchor_robot)
    json_path = output_base.with_suffix(".json")
    g2o_path = output_base.with_suffix(".g2o")
    write_json(graph, json_path)
    write_g2o(graph, g2o_path)
    provenance = {
        "format": "dpvo_verified_component",
        "version": 1,
        "source": {"path": str(source), "sha256": sha256(source)},
        "robot_ids": args.robot_ids,
        "anchor_robot_id": args.anchor_robot,
        "vertex_count": len(graph.vertices),
        "edge_count": len(graph.edges),
        "inter_robot_loop_count": graph.metadata["inter_robot_loop_count"],
        "outputs": {
            "json": {"path": str(json_path), "sha256": sha256(json_path)},
            "g2o": {"path": str(g2o_path), "sha256": sha256(g2o_path)},
        },
    }
    provenance_path = output_base.with_name(
        f"{output_base.name}_provenance.json"
    )
    provenance_path.write_text(json.dumps(provenance, indent=2) + "\n")
    print(json.dumps(provenance, indent=2))


if __name__ == "__main__":
    main()
