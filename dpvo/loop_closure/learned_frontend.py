"""Learned retrieval and local-feature front ends for loop closure.

The wrappers deliberately follow the public PyTorch Hub APIs from the official
MegaLoc and XFeat repositories.  Keeping the optional models behind this small
interface means importing DPVO does not download weights or require the model
repositories unless the learned pipeline is selected.
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn.functional as F


def _load_hub_model(repo_or_dir: str, entrypoint: str, **kwargs):
    repository = str(Path(repo_or_dir).expanduser())
    if Path(repository).is_dir():
        return torch.hub.load(
            repository,
            entrypoint,
            source="local",
            **kwargs,
        )
    return torch.hub.load(
        repo_or_dir,
        entrypoint,
        source="github",
        trust_repo=True,
        **kwargs,
    )


class MegaLocDescriptorExtractor:
    """Extract official MegaLoc global image descriptors."""

    def __init__(self, repo_or_dir: str, device: str = "cuda"):
        self.device = torch.device(device)
        self.model = _load_hub_model(repo_or_dir, "get_trained_model")
        self.model = self.model.to(self.device).eval()
        self.mean = torch.tensor(
            [0.485, 0.456, 0.406],
            device=self.device,
        ).view(1, 3, 1, 1)
        self.std = torch.tensor(
            [0.229, 0.224, 0.225],
            device=self.device,
        ).view(1, 3, 1, 1)

    @torch.inference_mode()
    def __call__(self, images_bgr: torch.Tensor) -> torch.Tensor:
        if images_bgr.ndim != 4 or images_bgr.shape[1] != 3:
            raise ValueError("MegaLoc expects Bx3xHxW images")
        images = images_bgr.to(self.device, dtype=torch.float32)
        if images.numel() and float(images.max()) > 1.0:
            images = images / 255.0
        # DPVO/OpenCV images are BGR, while MegaLoc was trained with RGB.
        images = images[:, [2, 1, 0]]
        images = (images - self.mean) / self.std
        images = F.interpolate(
            images,
            size=(322, 322),
            mode="bilinear",
            align_corners=False,
            antialias=True,
        )
        descriptors = self.model(images)
        if descriptors.ndim != 2 or descriptors.shape[0] != images.shape[0]:
            raise RuntimeError(
                f"Unexpected MegaLoc descriptor shape: {tuple(descriptors.shape)}"
            )
        return F.normalize(descriptors.float(), dim=-1)


class XFeatLighterGlueMatcher(torch.nn.Module):
    """Adapt XFeat's trained LighterGlue matcher to Kornia's match interface."""

    def __init__(self, xfeat, min_confidence: float):
        super().__init__()
        self.xfeat = xfeat
        self.min_confidence = float(min_confidence)

    @staticmethod
    def _unbatch(features: dict) -> dict:
        image_size = features["image_size"][0].detach().cpu().tolist()
        return {
            "keypoints": features["keypoints"][0],
            "descriptors": features["descriptors"][0],
            "image_size": image_size,
        }

    @torch.inference_mode()
    def forward(self, data: dict) -> dict:
        first = self._unbatch(data["image0"])
        second = self._unbatch(data["image1"])
        _first_points, _second_points, indices = self.xfeat.match_lighterglue(
            first,
            second,
            min_conf=self.min_confidence,
        )
        matches = torch.as_tensor(
            indices,
            dtype=torch.long,
            device=first["keypoints"].device,
        )
        return {"matches": matches[None]}


class XFeatFrontend:
    """Official XFeat sparse detector paired with its trained LighterGlue."""

    def __init__(
        self,
        repo_or_dir: str,
        top_k: int = 2048,
        detection_threshold: float = 0.05,
        min_confidence: float = 0.1,
        device: str = "cuda",
    ):
        self.device = torch.device(device)
        self.top_k = int(top_k)
        self.model = _load_hub_model(
            repo_or_dir,
            "XFeat",
            pretrained=True,
            top_k=self.top_k,
            detection_threshold=float(detection_threshold),
        )
        # The official wrapper chooses its device at construction time. Keep
        # that public attribute consistent if an explicit device was supplied.
        self.model.dev = self.device
        self.model = self.model.to(self.device).eval()
        self.matcher = XFeatLighterGlueMatcher(self.model, min_confidence).eval()

    @torch.inference_mode()
    def detect(self, images: torch.Tensor) -> list[dict]:
        images = images.to(self.device, dtype=torch.float32)
        if images.numel() and float(images.max()) > 1.0:
            images = images / 255.0
        _, _, height, width = images.shape
        image_size = torch.tensor(
            [width, height],
            dtype=torch.float32,
            device=self.device,
        ).view(1, 2)
        raw_features = self.model.detectAndCompute(images, top_k=self.top_k)
        return [
            {
                "keypoints": features["keypoints"][None],
                "descriptors": features["descriptors"][None],
                "image_size": image_size,
            }
            for features in raw_features
        ]
