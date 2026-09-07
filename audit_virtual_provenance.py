"""Replay accepted virtual descriptors directly from recorded source-grid samples."""
import argparse
import json
from pathlib import Path

import numpy as np

from dpvo.sphorb import SphorbExtractor
from dpvo.virtual_sphere import FAMILY, VERSION
from inspect_sphere_matches import load_features


def audit(experiment, anchors):
    extractor = SphorbExtractor(features=1000000).native
    total = 0
    for anchor in anchors:
        features = load_features(experiment / 'features' / f'{anchor:06d}.npz', FAMILY, VERSION)
        with np.load(experiment / 'provenance' / f'{anchor:06d}.npz') as p:
            prov = {k: p[k].copy() for k in p.files}
        for timestamp in np.unique(prov['source_timestamps']):
            gray, valid, sources, saved = [], [], [], []
            for level in range(7):
                with np.load(experiment / 'grids' / f'{anchor:06d}' / f'source_{timestamp:06d}_level_{level}.npz') as data:
                    d = {k: data[k].copy() for k in data.files}
                g = np.zeros(tuple(d['shape']), np.uint8); g.ravel()[d['indices']] = d['gray']
                mask = np.zeros_like(g); mask.ravel()[d['indices']] = 255
                gray.append(g); valid.append(mask); sources.append((mask > 0).astype(np.uint8)); saved.append(d)
            replay = extractor.extract_grids(gray, valid, sources)
            lookup = {(int(l), int(s), int(x), int(y)): i for i, (l, s, (x, y)) in enumerate(zip(replay['octaves'], replay['sections'], replay['grid']))}
            for index in np.flatnonzero(prov['source_timestamps'] == timestamp):
                level, section = int(features.octaves[index]), int(prov['grid_sections'][index])
                x, y = prov['grid_xy'][index].astype(int)
                match = lookup[level, section, x, y]
                for name in ('descriptors', 'uv', 'bearings', 'responses', 'orientations', 'sizes', 'octaves'):
                    np.testing.assert_array_equal(getattr(features, name)[index], replay[name][match])
                d = saved[level]
                address = (section * d['shape'][1] + y - 18) * d['shape'][2] + x - 17
                sampled = np.searchsorted(d['indices'], address)
                assert d['indices'][sampled] == address
                np.testing.assert_array_equal(prov['source_uv'][index], d['source_uv'][sampled])
                assert prov['source_lod'][index] == d['source_lod'][sampled]
                np.testing.assert_allclose(features.points[index], features.bearings[index] * d['depth'][sampled], atol=1e-7)
                total += 1
        print(f'Replayed anchor {anchor}: all {len(features.uv)} retained descriptors and source coordinates identical', flush=True)
    result = dict(anchors=anchors, identical_descriptors_and_provenance=total,
                  input='sparse source-grid intensity/provenance archives only; no panorama or original image reads')
    (experiment / 'provenance_validation.json').write_text(json.dumps(result, indent=2) + '\n')
    return result


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--experiment', type=Path, required=True)
    p.add_argument('--anchors', type=int, nargs='+', default=[243, 336])
    args = p.parse_args()
    audit(args.experiment.resolve(), args.anchors)
