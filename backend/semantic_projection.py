"""Visibility evidence for region association on an identity-bound full PLY.

Point-center z-buffer support is a conservative association heuristic. It is
not the alpha rasterizer, a depth ground truth, or an instance-quality metric.
"""
from __future__ import annotations
import numpy as np
from .semantics import _camera_matrices


def visible_point_grid(points, camera, width, height, *, opacity=None, chunk_size=100_000):
    if any(isinstance(value,(bool,np.bool_)) or not isinstance(value,(int,np.integer)) or value<1
           for value in (width,height,camera.get('width'),camera.get('height'),chunk_size)):
        raise ValueError('Projection dimensions and chunk_size must be positive integers')
    points = np.asarray(points)
    if points.ndim != 2 or points.shape[1] != 3 or not np.isfinite(points).all():
        raise ValueError('Association needs finite Nx3 points')
    if opacity is not None:
        opacity = np.asarray(opacity).reshape(-1)
        if len(opacity) != len(points) or not np.isfinite(opacity).all():
            raise ValueError('Opacity must match full point order')
    transform, k = _camera_matrices(camera)
    scale = np.array([width / camera['width'], height / camera['height']])
    depth = np.full(width * height, np.inf)
    ids = np.full(width * height, -1, np.int64)
    for start in range(0, len(points), chunk_size):
        xyz = points[start:start + chunk_size] @ transform[:3, :3].T + transform[:3, 3]
        valid = xyz[:, 2] > 1e-6
        if opacity is not None:
            valid &= opacity[start:start + len(xyz)] >= .05
        local = np.flatnonzero(valid)
        uvh = xyz[local] @ k.T
        uv = uvh[:, :2] / uvh[:, 2:] * scale
        inside = (uv[:, 0] >= 0) & (uv[:, 0] < width) & (uv[:, 1] >= 0) & (uv[:, 1] < height)
        local, uv = local[inside], uv[inside].astype(np.int64)
        pixels = uv[:, 1] * width + uv[:, 0]
        order = np.argsort(xyz[local, 2], kind='stable')
        _, first = np.unique(pixels[order], return_index=True)
        selected = order[first]
        pix, loc = pixels[selected], local[selected]
        closer = xyz[loc, 2] < depth[pix]
        depth[pix[closer]] = xyz[loc[closer], 2]
        ids[pix[closer]] = start + loc[closer]
    return ids.reshape(height, width)


def projection_support_loader(points, camera_map, *, opacity=None, cancelled=None):
    """One-frame cache bounds image memory; callers should group records by frame."""
    cache = {}
    stride=max(1,int(np.ceil(len(points)/250_000)))
    def load(record, mask):
        if cancelled is not None:
            cancelled()
        frame = record['frame_id']
        key = (frame, mask.shape)
        if key not in cache:
            cache.clear()
            cache[key] = visible_point_grid(points, camera_map[frame], mask.shape[1], mask.shape[0], opacity=opacity)
        supported = cache[key][np.asarray(mask, bool)]
        supported=supported[(supported >= 0) & (supported % stride == 0)]
        return np.unique(supported).tolist()
    return load
