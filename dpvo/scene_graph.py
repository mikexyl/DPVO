import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np


def _quaternion_to_matrix(quaternion):
    x, y, z, w = quaternion / np.linalg.norm(quaternion)
    return np.asarray(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float32,
    )


@dataclass
class SceneNode:
    node_id: str
    class_id: int
    label: str
    position: np.ndarray
    color: np.ndarray
    confidence: float
    observations: int
    last_seen: int


class SceneGraphBuilder:
    """Fuse YOLO regions with DPVO patches into persistent object landmarks."""

    def __init__(
        self,
        match_distance=0.75,
        relation_distance=2.0,
        min_patch_points=2,
        min_observations=3,
    ):
        self.match_distance = match_distance
        self.relation_distance = relation_distance
        self.min_patch_points = min_patch_points
        self.min_observations = min_observations
        self.nodes = {}
        self.next_node_id = 1
        self.last_keyframe_timestamp = None

    @staticmethod
    def _frame_patch_geometry(patch_graph, frame_index):
        patch_size = patch_graph.patches_.shape[-1]
        center = patch_size // 2
        patches = (
            patch_graph.patches_[frame_index, :, :, center, center]
            .detach()
            .cpu()
            .numpy()
        )
        intrinsics = patch_graph.intrinsics_[frame_index].detach().cpu().numpy()
        pose = patch_graph.poses_[frame_index].detach().cpu().numpy()

        xy = 4.0 * (patches[:, :2] + 0.5)
        disparity = patches[:, 2]
        valid = np.isfinite(patches).all(axis=1) & (disparity > 1e-4)

        camera_points = np.zeros((len(patches), 3), dtype=np.float32)
        depth = np.zeros_like(disparity)
        depth[valid] = 1.0 / disparity[valid]
        camera_points[:, 0] = (patches[:, 0] - intrinsics[2]) / intrinsics[0] * depth
        camera_points[:, 1] = (patches[:, 1] - intrinsics[3]) / intrinsics[1] * depth
        camera_points[:, 2] = depth

        # DPVO poses are world-to-camera with xyzw quaternions.
        rotation_camera_from_world = _quaternion_to_matrix(pose[3:])
        rotation_world_from_camera = rotation_camera_from_world.T
        translation_world_from_camera = -rotation_world_from_camera @ pose[:3]
        world_points = (
            camera_points @ rotation_world_from_camera.T
            + translation_world_from_camera
        )
        valid &= np.isfinite(world_points).all(axis=1)
        return xy, world_points, valid

    @staticmethod
    def _robust_centroid(points):
        center = np.median(points, axis=0)
        if len(points) < 4:
            return center

        distances = np.linalg.norm(points - center, axis=1)
        cutoff = np.quantile(distances, 0.8)
        inliers = points[distances <= cutoff]
        return np.median(inliers, axis=0)

    def _add_observation(
        self,
        class_id,
        label,
        color,
        confidence,
        position,
        frame_index,
        used_nodes,
    ):
        candidates = [
            node
            for node in self.nodes.values()
            if node.class_id == class_id and node.node_id not in used_nodes
        ]
        node = None
        if candidates:
            distances = np.asarray(
                [np.linalg.norm(candidate.position - position) for candidate in candidates]
            )
            best = int(np.argmin(distances))
            if distances[best] <= self.match_distance:
                node = candidates[best]

        if node is None:
            node_id = f"{label.replace(' ', '_')}_{self.next_node_id:03d}"
            self.next_node_id += 1
            node = SceneNode(
                node_id=node_id,
                class_id=class_id,
                label=label,
                position=position,
                color=color,
                confidence=confidence,
                observations=1,
                last_seen=frame_index,
            )
            self.nodes[node_id] = node
        else:
            alpha = max(0.2, 1.0 / (node.observations + 1))
            node.position = (1.0 - alpha) * node.position + alpha * position
            node.color = color
            node.confidence = (
                node.confidence * node.observations + confidence
            ) / (node.observations + 1)
            node.observations += 1
            node.last_seen = frame_index

        used_nodes.add(node.node_id)

    def update(self, annotations, patch_graph, num_frames, frame_index):
        if annotations is None or num_frames <= 0:
            return

        keyframe_index = num_frames - 1
        keyframe_timestamp = int(patch_graph.tstamps_[keyframe_index])
        if keyframe_timestamp == self.last_keyframe_timestamp:
            return
        self.last_keyframe_timestamp = keyframe_timestamp

        xy, world_points, valid = self._frame_patch_geometry(
            patch_graph,
            keyframe_index,
        )
        pixel_x = np.rint(xy[:, 0]).astype(np.int64)
        pixel_y = np.rint(xy[:, 1]).astype(np.int64)
        height, width = annotations.image_shape
        in_image = (
            (pixel_x >= 0)
            & (pixel_x < width)
            & (pixel_y >= 0)
            & (pixel_y < height)
        )
        valid &= in_image

        used_nodes = set()
        for index, (box, class_id, color, confidence) in enumerate(
            zip(
                annotations.boxes,
                annotations.class_ids,
                annotations.colors,
                annotations.scores,
            )
        ):
            if annotations.instance_masks is not None:
                inside = np.zeros(len(xy), dtype=bool)
                sample = valid.copy()
                inside[sample] = annotations.instance_masks[index][
                    pixel_y[sample], pixel_x[sample]
                ]
            else:
                x1, y1, x2, y2 = box
                inside = (
                    valid
                    & (xy[:, 0] >= x1)
                    & (xy[:, 0] <= x2)
                    & (xy[:, 1] >= y1)
                    & (xy[:, 1] <= y2)
                )

            points = world_points[inside]
            if len(points) < self.min_patch_points:
                continue

            self._add_observation(
                class_id=int(class_id),
                label=annotations.class_names[index],
                color=np.asarray(color, dtype=np.uint8),
                confidence=float(confidence),
                position=self._robust_centroid(points),
                frame_index=frame_index,
                used_nodes=used_nodes,
            )

    def visible_nodes(self):
        return [
            node
            for node in self.nodes.values()
            if node.observations >= self.min_observations
        ]

    def edges(self):
        nodes = self.visible_nodes()
        if len(nodes) < 2:
            return []

        edge_pairs = {}
        for index, source in enumerate(nodes):
            distances = []
            for other_index, target in enumerate(nodes):
                if index == other_index:
                    continue
                distance = float(np.linalg.norm(source.position - target.position))
                if distance <= self.relation_distance:
                    distances.append((distance, other_index))
            for distance, other_index in sorted(distances)[:2]:
                pair = tuple(sorted((index, other_index)))
                edge_pairs[pair] = min(distance, edge_pairs.get(pair, np.inf))

        return [
            {
                "source": nodes[source].node_id,
                "target": nodes[target].node_id,
                "relation": "near",
                "distance": distance,
            }
            for (source, target), distance in sorted(edge_pairs.items())
        ]

    def save(self, output_path):
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        nodes = self.visible_nodes()
        document = {
            "coordinate_frame": "DPVO world (monocular scale)",
            "method": {
                "association": "class-aware nearest 3D centroid",
                "match_distance": self.match_distance,
                "relation_distance": self.relation_distance,
                "min_patch_points": self.min_patch_points,
                "min_observations": self.min_observations,
            },
            "nodes": [
                {
                    "id": node.node_id,
                    "class_id": node.class_id,
                    "label": node.label,
                    "position": node.position.tolist(),
                    "mean_confidence": node.confidence,
                    "observations": node.observations,
                    "last_seen": node.last_seen,
                }
                for node in nodes
            ],
            "edges": self.edges(),
        }
        output_path.write_text(json.dumps(document, indent=2) + "\n")
