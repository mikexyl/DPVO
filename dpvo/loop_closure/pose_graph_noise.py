"""Apply deterministic synthetic noise to DPVO pose-graph loop measurements."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path

import numpy as np

from .centralized import Sim3
from .pose_graph import Sim3PoseGraph, read_json, sim3_to_dict, write_json


NOISE_METADATA_KEY = "synthetic_loop_noise"
TANGENT_ORDER = ("tx", "ty", "tz", "rx", "ry", "rz", "log_scale")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def perturb_loop_measurements(
    graph: Sim3PoseGraph,
    *,
    translation_sigma: float,
    rotation_sigma_degrees: float,
    log_scale_sigma: float,
    seed: int,
) -> tuple[Sim3PoseGraph, list[dict]]:
    """Right-compose Gaussian Sim(3) noise onto inter-robot loop edges."""

    for name, value in (
        ("translation_sigma", translation_sigma),
        ("rotation_sigma_degrees", rotation_sigma_degrees),
        ("log_scale_sigma", log_scale_sigma),
    ):
        if value < 0.0:
            raise ValueError(f"{name} must be non-negative")
    if NOISE_METADATA_KEY in graph.metadata:
        raise ValueError("input graph already declares synthetic loop noise")

    output = copy.deepcopy(graph)
    generator = np.random.default_rng(seed)
    rotation_sigma_radians = np.deg2rad(rotation_sigma_degrees)
    records = []
    for edge in output.edges:
        if edge.edge_type != "inter_robot_loop_closure":
            continue
        if NOISE_METADATA_KEY in edge.metadata:
            raise ValueError(f"edge {edge.edge_id} already has synthetic noise")

        tangent = np.concatenate(
            (
                generator.normal(0.0, translation_sigma, size=3),
                generator.normal(0.0, rotation_sigma_radians, size=3),
                [generator.normal(0.0, log_scale_sigma)],
            )
        )
        original = edge.measurement
        noise = Sim3.from_vector(tangent)
        edge.measurement = original.compose(noise)
        record = {
            "edge_id": edge.edge_id,
            "sampled_tangent": tangent.tolist(),
            "original_measurement": sim3_to_dict(original),
            "perturbed_measurement": sim3_to_dict(edge.measurement),
        }
        edge.metadata[NOISE_METADATA_KEY] = record
        if "query_to_match" in edge.metadata:
            edge.metadata["query_to_match"] = sim3_to_dict(edge.measurement)
        records.append(record)

    if not records:
        raise ValueError("input graph contains no inter-robot loop closures")
    output.name = f"{graph.name} with deterministic synthetic loop noise"
    output.metadata[NOISE_METADATA_KEY] = {
        "format": "dpvo_synthetic_loop_noise",
        "version": 1,
        "composition": "measurement.compose(noise)",
        "tangent_order": list(TANGENT_ORDER),
        "translation_frame": "measurement source frame",
        "translation_sigma_local_units_per_axis": translation_sigma,
        "rotation_sigma_degrees_per_axis": rotation_sigma_degrees,
        "log_scale_sigma": log_scale_sigma,
        "seed": seed,
        "perturbed_edge_type": "inter_robot_loop_closure",
        "perturbed_edge_count": len(records),
        "information_diagonal_modified": False,
    }
    return output.validate(), records


def _atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--translation-sigma", type=float, required=True)
    parser.add_argument("--rotation-sigma-degrees", type=float, required=True)
    parser.add_argument("--log-scale-sigma", type=float, required=True)
    parser.add_argument("--seed", type=int, required=True)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    input_path = args.input.expanduser().resolve()
    output_path = args.output.expanduser().resolve()
    manifest_path = (
        args.manifest.expanduser().resolve()
        if args.manifest is not None
        else output_path.with_name(f"{output_path.stem}_noise_manifest.json")
    )
    if input_path == output_path:
        raise ValueError("input and output graph paths must differ")
    for path in (output_path, manifest_path):
        if path.exists():
            raise FileExistsError(f"refusing to overwrite {path}")

    source_sha256 = sha256(input_path)
    graph = read_json(input_path)
    perturbed, records = perturb_loop_measurements(
        graph,
        translation_sigma=args.translation_sigma,
        rotation_sigma_degrees=args.rotation_sigma_degrees,
        log_scale_sigma=args.log_scale_sigma,
        seed=args.seed,
    )
    write_json(perturbed, output_path)
    read_json(output_path).validate()
    manifest = {
        "format": "dpvo_synthetic_loop_noise_manifest",
        "version": 1,
        "source_graph": str(input_path),
        "source_sha256": source_sha256,
        "output_graph": str(output_path),
        "output_sha256": sha256(output_path),
        "configuration": perturbed.metadata[NOISE_METADATA_KEY],
        "edges": records,
    }
    _atomic_json(manifest_path, manifest)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
