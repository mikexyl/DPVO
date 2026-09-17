"""Bounded DPVO patch reprojection trails on the exact processed image."""
from collections import deque
import numpy as np


class PatchPreview:
    def __init__(self, max_tracks=128, trail_length=12):
        self.max_tracks = max_tracks
        self.trail_length = trail_length
        self.history = {}

    def observe(self, ids, coordinates, width, height, tracked):
        visible = {}
        for identity, xy in zip(ids[:self.max_tracks], coordinates[:self.max_tracks]):
            if not np.isfinite(xy).all() or not (0 <= xy[0] < width and 0 <= xy[1] < height):
                continue
            identity = tuple(identity)
            history = self.history.get(identity, deque(maxlen=self.trail_length)) if tracked else deque(maxlen=self.trail_length)
            history.append(tuple(np.rint(xy).astype(int)))
            visible[identity] = history
        self.history = visible if tracked else {}
        return visible

    def draw(self, bgr, ids, coordinates, tracked=True):
        import cv2
        canvas = bgr.copy()
        h, w = canvas.shape[:2]
        visible = self.observe(ids, coordinates, w, h, tracked)
        count = 0
        trails = 0
        for identity, xy in zip(ids[:self.max_tracks], coordinates[:self.max_tracks]):
            if not np.isfinite(xy).all() or not (0 <= xy[0] < w and 0 <= xy[1] < h):
                continue
            identity = tuple(identity)
            point = tuple(np.rint(xy).astype(int))
            hue = (identity[0] * 37 + identity[1] * 17) % 180
            color = tuple(int(v) for v in cv2.cvtColor(np.uint8([[[hue, 220, 255]]]), cv2.COLOR_HSV2BGR)[0, 0])
            history = visible[identity]
            if len(history) > 1 and len(set(history)) > 1:
                trail = [np.asarray(history, np.int32)]
                cv2.polylines(canvas, trail, False, (0, 0, 0), 4, cv2.LINE_AA)
                cv2.polylines(canvas, trail, False, color, 2, cv2.LINE_AA)
                trails += 1
            x, y = point
            cv2.rectangle(canvas, (x-5, y-5), (x+5, y+5), color, 1, cv2.LINE_AA)
            cv2.circle(canvas, point, 2, color, -1, cv2.LINE_AA)
            count += 1
        label = f'{count} patches | {trails} trails' if tracked else f'{count} candidate patches - initializing'
        cv2.rectangle(canvas, (0, 0), (w, 21), (25, 25, 25), -1)
        cv2.putText(canvas, label, (5, 15), cv2.FONT_HERSHEY_SIMPLEX, .4, (255, 255, 255), 1, cv2.LINE_AA)
        return canvas, count

    def snapshot(self, slam, bgr, input_index, render=True):
        import torch
        pg = slam.pg
        current = slam.n - 1
        accepted = current >= 0 and int(pg.tstamps_[current]) == input_index
        tracked = accepted and slam.is_initialized
        if tracked:
            edges = torch.where((pg.jj == current) & (pg.ii != current))[0]
            sources = pg.ii[edges].detach().cpu().numpy()
            patches = pg.kk[edges].detach().cpu().numpy() % slam.M
            identities = [(int(pg.tstamps_[source]), int(patch)) for source, patch in zip(sources, patches)]
            # Follow one source keyframe while it remains in the active graph.
            # Switching to the newest source every frame would erase its trails.
            available = {identity[0] for identity in identities}
            retained = [identity[0] for identity in self.history if identity[0] in available]
            source_stamp = retained[0] if retained else max(available, default=None)
            selected = [i for i, identity in enumerate(identities)
                        if identity[0] == source_stamp][:self.max_tracks]
            edges = edges[torch.as_tensor(selected, device=edges.device, dtype=torch.long)]
            if not len(edges):
                self.history.clear()
                return self.draw(bgr, [], [], tracked=True) if render else (None, 0)
            ii, jj, kk = pg.ii[edges], pg.jj[edges], pg.kk[edges]
            with torch.no_grad():
                xy = slam.reproject((ii, jj, kk))[0, :, :, slam.P//2, slam.P//2]
                xy = xy.detach().float().cpu().numpy() * slam.RES
            origins = ii.detach().cpu().numpy()
            patch_ids = kk.detach().cpu().numpy() % slam.M
            ids = [(int(pg.tstamps_[source]), int(patch)) for source, patch in zip(origins, patch_ids)]
        else:
            # A motion-probe rejection leaves the just-extracted patches in slot n.
            slot = current if accepted else slam.n
            patches = pg.patches_[slot].detach().float().cpu().numpy()
            xy = patches[:, :2, slam.P//2, slam.P//2] * slam.RES
            ids = [(input_index, index) for index in range(len(xy))]
        if not render:
            visible = self.observe(ids, xy, bgr.shape[1], bgr.shape[0], tracked)
            return None, len(visible)
        return self.draw(bgr, ids, xy, tracked)
