import io
from pathlib import Path, PurePosixPath
from zipfile import ZipFile

import cv2
import numpy as np
import torch
import torch.utils.data as data

from .augmentation import RGBDAugmentor


class TartanAirV2(data.Dataset):
    """Read DPVO training clips directly from TartanAir V2 zip archives."""

    DEPTH_SCALE = 5.0
    CAMERA = "lcam_front"
    PERMUTATION = [1, 2, 0, 4, 5, 3, 6]  # NED -> DPVO xyz

    def __init__(self, datapath, n_frames=15, crop_size=(480, 640),
                 max_stride=3, **kwargs):
        del kwargs
        self.root = Path(datapath)
        self.n_frames = n_frames
        self.max_stride = max_stride
        self.augmentor = RGBDAugmentor(crop_size=crop_size)
        self.sequences = []
        self.dataset_index = []
        self._archives = {}
        self._build_index()

    @staticmethod
    def _members_by_trajectory(archive, modality):
        grouped = {}
        marker = f"{modality}_{TartanAirV2.CAMERA}"
        for member in archive.namelist():
            path = PurePosixPath(member)
            if path.suffix != ".png" or len(path.parts) < 2 or path.parts[-2] != marker:
                continue
            try:
                frame_id = int(path.name.split("_", 1)[0])
            except ValueError:
                continue
            trajectory = "/".join(path.parts[:-2])
            grouped.setdefault(trajectory, {})[frame_id] = member
        return grouped

    def _build_index(self):
        required = (f"image_{self.CAMERA}.zip", f"depth_{self.CAMERA}.zip")
        for difficulty in sorted(self.root.glob("*/Data_easy")):
            image_path, depth_path = [difficulty / name for name in required]
            if not image_path.is_file() or not depth_path.is_file():
                continue

            with ZipFile(image_path) as image_zip, ZipFile(depth_path) as depth_zip:
                images = self._members_by_trajectory(image_zip, "image")
                depths = self._members_by_trajectory(depth_zip, "depth")
                pose_members = {
                    "/".join(PurePosixPath(name).parts[:-1]): name
                    for name in image_zip.namelist()
                    if name.endswith(f"pose_{self.CAMERA}.txt")
                }

                for trajectory in sorted(images.keys() & depths.keys() & pose_members.keys()):
                    poses = np.loadtxt(io.BytesIO(image_zip.read(pose_members[trajectory])))
                    frame_ids = sorted(images[trajectory].keys() & depths[trajectory].keys())
                    frame_ids = [frame_id for frame_id in frame_ids if frame_id < len(poses)]
                    if len(frame_ids) < self.n_frames:
                        continue

                    sequence = {
                        "name": trajectory,
                        "image_archive": str(image_path),
                        "depth_archive": str(depth_path),
                        "images": [images[trajectory][i] for i in frame_ids],
                        "depths": [depths[trajectory][i] for i in frame_ids],
                        "poses": poses[frame_ids][:, self.PERMUTATION].astype(np.float32),
                    }
                    sequence["poses"][:, :3] /= self.DEPTH_SCALE
                    sequence_id = len(self.sequences)
                    self.sequences.append(sequence)
                    self.dataset_index.extend(
                        (sequence_id, start)
                        for start in range(len(frame_ids) - self.n_frames + 1)
                    )

        if not self.dataset_index:
            raise RuntimeError(f"No complete TartanAir V2 RGB/depth sequences found in {self.root}")

        print(
            f"TartanAirV2: indexed {len(self.sequences)} sequences and "
            f"{len(self.dataset_index)} clips from {self.root}"
        )

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_archives"] = {}
        return state

    def _archive(self, path):
        if path not in self._archives:
            self._archives[path] = ZipFile(path)
        return self._archives[path]

    def _read_image(self, archive, member, flags):
        encoded = np.frombuffer(self._archive(archive).read(member), dtype=np.uint8)
        image = cv2.imdecode(encoded, flags)
        if image is None:
            raise RuntimeError(f"Could not decode {member} from {archive}")
        return image

    def __getitem__(self, index):
        sequence_id, start = self.dataset_index[index % len(self.dataset_index)]
        sequence = self.sequences[sequence_id]
        available = len(sequence["images"]) - 1 - start
        stride = np.random.randint(1, min(self.max_stride, available // (self.n_frames - 1)) + 1)
        indices = start + stride * np.arange(self.n_frames)

        images, depths = [], []
        for i in indices:
            images.append(self._read_image(
                sequence["image_archive"], sequence["images"][i], cv2.IMREAD_COLOR))
            encoded_depth = self._read_image(
                sequence["depth_archive"], sequence["depths"][i], cv2.IMREAD_UNCHANGED)
            depth = encoded_depth.view("<f4").reshape(encoded_depth.shape[:2]).copy()
            depth[~np.isfinite(depth) | (depth <= 0)] = 1.0
            depths.append(depth / self.DEPTH_SCALE)

        images = torch.from_numpy(np.stack(images).astype(np.float32)).permute(0, 3, 1, 2)
        depths = torch.from_numpy(np.stack(depths).astype(np.float32))
        poses = torch.from_numpy(sequence["poses"][indices].copy())
        intrinsics = torch.tensor([320.0, 320.0, 320.0, 240.0]).repeat(self.n_frames, 1)

        disps = 1.0 / depths
        images, poses, disps, intrinsics = self.augmentor(images, poses, disps, intrinsics)

        scale = 0.7 * torch.quantile(disps, 0.98)
        disps = disps / scale
        poses[..., :3] *= scale
        return images, poses, disps, intrinsics

    def __len__(self):
        return len(self.dataset_index)
