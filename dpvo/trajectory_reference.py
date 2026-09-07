"""Frame-exact reference association and evaluation-only global Sim3 alignment."""
from pathlib import Path

import numpy as np
from evo.core.geometry import umeyama_alignment


def video_reference(csv_path, input_timestamps, *, stride, skip):
    """Associate DPVO video_stream indices with the CSV's explicit frame column.

    The stream consumes stride frames before producing its first image:
    raw_frame = skip + (input_timestamp + 1) * stride - 1.
    No nearest-timestamp fit, interpolation, or row-number assumption is used.
    """
    timestamps = np.asarray(input_timestamps)
    if (timestamps.ndim != 1 or timestamps.dtype.kind not in 'iu' or len(timestamps) < 3
            or (timestamps < 0).any() or (np.diff(timestamps) <= 0).any()
            or not isinstance(stride, (int, np.integer)) or stride < 1
            or not isinstance(skip, (int, np.integer)) or skip < 0):
        raise ValueError('invalid video stream indices or stride/skip')
    data = np.genfromtxt(Path(csv_path), delimiter=',', names=True, ndmin=1)
    expected = ('timestamp','frame','x','y','z','qx','qy','qz','qw')
    if data.dtype.names != expected or len(data) < 3:
        raise ValueError('unexpected reference CSV format')
    matrix = np.column_stack([data[k] for k in expected])
    if not np.isfinite(matrix).all():
        raise ValueError('reference contains nonfinite values')
    ids = data['frame'].astype(np.int64)
    if (not np.array_equal(ids, data['frame']) or (ids < 0).any()
            or len(np.unique(ids)) != len(ids)):
        raise ValueError('reference frame IDs must be unique nonnegative integers')
    order = np.argsort(ids)
    if (np.diff(data['timestamp'][order]) <= 0).any():
        raise ValueError('reference timestamps must increase with frame IDs')
    norms = np.linalg.norm(matrix[:,5:9],axis=1)
    if not np.allclose(norms,1.,atol=1e-5):
        raise ValueError('reference quaternion is not normalized')
    raw_ids = skip + (timestamps + 1) * stride - 1
    lookup = {frame:row for row,frame in enumerate(ids)}
    missing = [int(frame) for frame in raw_ids if frame not in lookup]
    if missing:
        raise ValueError(f'reference missing required video frame IDs: {missing[:8]}')
    rows = np.asarray([lookup[frame] for frame in raw_ids])
    return dict(raw_frame_ids=raw_ids, csv_rows=rows, seconds=data['timestamp'][rows].copy(),
                positions=matrix[rows,2:5].copy(), quaternions=matrix[rows,5:9].copy(),
                reference_frame_count=len(data))


def apply_alignment(points, alignment):
    return (alignment['scale'] * np.asarray(points) @ np.asarray(alignment['rotation']).T
            + np.asarray(alignment['translation']))


def align_positions(estimate, reference):
    estimate, reference = np.asarray(estimate,float), np.asarray(reference,float)
    if (estimate.ndim != 2 or estimate.shape[1] != 3 or reference.shape != estimate.shape
            or len(estimate)<3 or not np.isfinite(estimate).all() or not np.isfinite(reference).all()):
        raise ValueError('expected equal finite Nx3 point arrays')
    rotation, translation, scale = umeyama_alignment(estimate.T,reference.T,with_scale=True)
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError('invalid fitted global scale')
    alignment = dict(rotation=rotation.tolist(),translation=translation.tolist(),scale=float(scale))
    return apply_alignment(estimate,alignment),alignment


def position_errors(estimate, reference):
    errors = np.linalg.norm(np.asarray(estimate)-np.asarray(reference),axis=1)
    return errors,dict(rmse_m=float(np.sqrt(np.mean(errors**2))),median_m=float(np.median(errors)),
                       p95_m=float(np.percentile(errors,95)),max_m=float(errors.max()),count=len(errors))
