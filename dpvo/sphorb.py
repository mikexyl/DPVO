"""Persistent optional SPHORB extractor. Native GPL-derived code lives in native/sphorb.

Angles are degrees in the local geodesic tangent grid, sizes are level patch
diameters scaled to original panorama width. Input colors are BGR with alpha
255 meaning observed; black is a valid color. Depth never affects appearance.
"""
from pathlib import Path
from time import perf_counter

import cv2
import numpy as np

from .sphere_bow import SphereFeatures


class SphorbExtractor:
    def __init__(self, features=3000, levels=7, threshold=20, workers=4, tables=None):
        start = perf_counter()
        try:
            from . import _sphorb
        except ImportError as exc:
            raise ImportError('Optional SPHORB backend unavailable; run pixi run build-sphorb') from exc
        self.tables = Path(tables).expanduser().resolve() if tables else Path(__file__).resolve().parents[1] / 'models/sphorb'
        self.native = _sphorb.Extractor(str(self.tables), features, levels, threshold, workers)
        self.setup_timing = dict(table_load_and_native_setup_seconds=self.native.setup_seconds,
                                 total_setup_seconds=perf_counter() - start)
        self.last_timing = {}
        self.family = _sphorb.descriptor_family
        self.version = _sphorb.descriptor_version

    def __call__(self, bgra, radial_depth):
        start = perf_counter()
        if bgra.dtype != np.uint8 or bgra.ndim != 3 or bgra.shape[2] != 4 or min(bgra.shape[:2]) < 2:
            raise ValueError('expected uint8 BGRA panorama of at least 2x2')
        if radial_depth.shape != bgra.shape[:2]:
            raise ValueError('radial depth must match panorama dimensions')
        valid = np.ascontiguousarray(bgra[..., 3] == 255, dtype=np.uint8)
        gray = cv2.cvtColor(bgra, cv2.COLOR_BGRA2GRAY)
        result = self.native.extract(gray, valid)
        native_seconds = result.pop('native_seconds')
        uv, bearings = result['uv'], result['bearings']
        x = np.floor(uv[:, 0] + .5).astype(int) % bgra.shape[1]
        y = np.clip(np.floor(uv[:, 1] + .5).astype(int), 0, bgra.shape[0] - 1)
        depth = radial_depth[y, x]
        depth = np.where(np.isfinite(depth) & (depth > 0), depth, np.nan)
        features = SphereFeatures(**result, points=(bearings * depth[:, None]).astype(np.float32),
                                  face_ids=None, face_xy=None, descriptor_family=self.family,
                                  descriptor_version=self.version)
        total = perf_counter() - start
        self.last_timing = dict(native_seconds=native_seconds, wrapper_seconds=total-native_seconds,
                                extraction_seconds=total)
        return features
