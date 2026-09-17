"""Detect catastrophic solver failures that still contain finite poses."""
import torch


def tracking_health(poses, inverse_depths, initialized):
    finite = bool(torch.isfinite(poses).all() and torch.isfinite(inverse_depths).all())
    median = float(inverse_depths.median())
    # fastba/ba_cuda.cu clamps inverse depth to 1e-4. A median at that
    # bound means most recent patches have collapsed to the far-depth limit.
    collapsed = initialized and median <= 1.01e-4
    return dict(finite=finite, median_inverse_depth=median,
                lost=not finite or collapsed)
