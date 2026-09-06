"""Image-space annotations shared by the optional segmentation backends."""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class FrameAnnotations:
    boxes: np.ndarray
    class_ids: np.ndarray
    class_names: list
    scores: np.ndarray
    labels: list
    colors: np.ndarray
    image_shape: tuple
    instance_masks: np.ndarray | None
    segmentation: np.ndarray | None
    segmentation_context: list | None
    instance_ids: np.ndarray | None = None
