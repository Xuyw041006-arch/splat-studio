"""Input-supported inverse-depth anchors and guarded pinned-trainer hooks.

The pinned rasterizer returns alpha-composited inverse depth, not metric depth.
This is a small initialization prior, not ground-truth depth supervision.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from .colmap_export import file_sha256


def build_sparse_depth_anchors(exported, output_dir, *, inferred_weight=.25, max_reprojection_pixels=4.):
    """Project only per-view observed initialization points; z-buffer occlusions.

    Predicted DUSt3R points can have real input-pixel support, but remain inferred
    and receive the explicitly recorded weaker weight. Unobserved pixels never
    acquire a depth target. No camera/image outside the exported inputs is read.
    """
    inferred_weight = float(inferred_weight)
    if not 0 <= inferred_weight <= 1 or not np.isfinite(max_reprojection_pixels) or max_reprojection_pixels <= 0:
        raise ValueError('Invalid sparse anchor confidence/reprojection configuration')
    directory = Path(output_dir).resolve(); directory.mkdir(parents=True, exist_ok=True)
    points = np.asarray(exported['points'], np.float64)
    confidence = np.asarray(exported['confidence'], np.float64)
    sources = np.asarray(exported['sources'])
    observations = exported['observations']
    rows = {}; total = 0
    for c in exported['cameras']:
        key = str(c['image_index']); width, height = c['width'], c['height']
        transform = np.asarray(c['world_to_camera']); k = c['intrinsics']
        xyz = points @ transform[:3, :3].T + transform[:3, 3]
        z = xyz[:, 2]; front = np.isfinite(xyz).all(1) & (z > 1e-6)
        uv = np.full((len(points), 2), np.nan)
        uv[front] = xyz[front, :2] / z[front, None] * [k['fx'], k['fy']] + [k['cx'], k['cy']]
        inside = front & (uv[:, 0] >= 0) & (uv[:, 0] < width) & (uv[:, 1] >= 0) & (uv[:, 1] < height)
        candidates = np.flatnonzero(inside)
        # All projected initialization points can occlude an eligible anchor.
        pixel = np.zeros((len(points), 2), np.int64)
        pixel[candidates] = np.floor(uv[candidates]).astype(np.int64)
        nearest = {}
        for index in candidates[np.argsort(z[candidates], kind='stable')]:
            nearest.setdefault(int(pixel[index, 1] * width + pixel[index, 0]), int(index))
        scale = np.asarray(c.get('initialization_to_original_scale_xy', [1., 1.]))
        selected = []; residuals = []; supported = 0
        for index in nearest.values():
            observation = observations[index].get(key)
            if observation is None: continue
            supported += 1
            residual = float(np.linalg.norm((uv[index] - np.asarray(observation)) / scale))
            if residual > max_reprojection_pixels or confidence[index] <= 0 or sources[index] == 'unknown': continue
            if sources[index] == 'inferred' and inferred_weight == 0: continue
            selected.append(index); residuals.append(residual)
        selected = np.asarray(selected, np.int64)
        inverse = 1. / z[selected]
        weights = confidence[selected] * np.where(sources[selected] == 'inferred', inferred_weight, 1.)
        filename = f"view-{c['image_index']:04d}.npz"
        np.savez_compressed(directory / filename, x=pixel[selected, 0], y=pixel[selected, 1],
                            inverse_depth=inverse.astype(np.float32), weight=weights.astype(np.float32),
                            width=np.int64(width), height=np.int64(height))
        count = len(selected); total += count
        rows[c['image_name']] = {'path': filename, 'sha256': file_sha256(directory / filename),
            'count': count, 'observed': int(np.sum(sources[selected] == 'observed')),
            'inferred': int(np.sum(sources[selected] == 'inferred')),
            'visible_with_observation': supported,
            'median_inverse_depth': float(np.median(inverse)) if count else None,
            'max_accepted_reprojection_feature_pixels': max(residuals) if residuals else None}
    metadata = {'format': 'splat-sparse-inverse-depth/1', 'total_anchors': total, 'views': rows,
        'inferred_confidence_multiplier': inferred_weight, 'max_reprojection_feature_pixels': max_reprojection_pixels,
        'target': 'Initialization inverse depth at input-supported projected pixels; nearest-point z-buffer.',
        'renderer_quantity': 'Pinned rasterizer alpha-composited inverse depth; not opacity-normalized metric depth.',
        'normalization': 'Per-view median target inverse depth after training-resolution z-buffer.',
        'not_ground_truth': True, 'unobserved_pixels_supervised': False}
    path = directory / 'manifest.json'
    path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + '\n')
    return {'manifest_path': str(path), 'metadata': metadata}


# Embedded into a copied trainer so the CUDA environment need not import the
# application backend or inherit process-global per-job environment variables.
_DEPTH_HOOK = '''
import json as splat_json
import numpy as splat_np
from pathlib import Path as SplatPath
splat_anchor_manifest_path = SplatPath(__ANCHOR_MANIFEST__)
splat_anchor_manifest = splat_json.loads(splat_anchor_manifest_path.read_text())
splat_anchor_cache = {}
def splat_sparse_depth_loss(inverse_depth, name):
    depth = inverse_depth.squeeze(0) if inverse_depth.ndim == 3 else inverse_depth
    if depth.ndim != 2:
        raise RuntimeError("Sparse depth hook requires an H x W inverse-depth render")
    key = (name, tuple(depth.shape), str(depth.device), str(depth.dtype))
    if key not in splat_anchor_cache:
        row = splat_anchor_manifest['views'].get(name)
        if row is None:
            raise RuntimeError("Training camera is missing from input-only sparse anchor manifest: " + name)
        with splat_np.load(splat_anchor_manifest_path.parent / row['path'], allow_pickle=False) as data:
            x = splat_np.minimum((data['x'] * depth.shape[1] / int(data['width'])).astype('int64'), depth.shape[1]-1)
            y = splat_np.minimum((data['y'] * depth.shape[0] / int(data['height'])).astype('int64'), depth.shape[0]-1)
            target = data['inverse_depth']; weights = data['weight']
            # Downsampling may collapse anchors. Keep the closest surface.
            order = splat_np.argsort(-target, kind='stable')
            _, unique = splat_np.unique((y * depth.shape[1] + x)[order], return_index=True)
            keep = order[unique]
            values = (y[keep], x[keep], target[keep], weights[keep])
            splat_anchor_cache[key] = tuple(torch.as_tensor(v.copy(), device=depth.device,
                dtype=torch.long if i < 2 else depth.dtype) for i, v in enumerate(values))
    y, x, target, weights = splat_anchor_cache[key]
    if target.numel() == 0:
        return depth.sum() * 0
    scale = target.median().clamp_min(1e-8)
    loss = torch.nn.functional.smooth_l1_loss(depth[y, x] / scale, target / scale, reduction='none', beta=0.1)
    # Divide by count, preserving lower confidence for inferred targets.
    return (loss * weights).sum() / target.numel()
'''

_PRIORITY_HOOK = '''
import numpy as splat_np
from PIL import Image as SplatImage
from pathlib import Path as SplatPath
splat_mask_dir = SplatPath(__PRIORITY_DIRECTORY__)
splat_mask_cache = {}
def splat_weighted_l1(image, target, name):
    key = (name, tuple(image.shape), str(image.device), str(image.dtype))
    if key not in splat_mask_cache:
        if len(splat_mask_cache)>=16:splat_mask_cache.pop(next(iter(splat_mask_cache)))
        path = splat_mask_dir / (name + ".png")
        if path.is_file():
            with SplatImage.open(path) as source_mask:
                mask = source_mask.convert("L").resize((image.shape[2], image.shape[1]))
                weight = 1 + 2 * torch.tensor(splat_np.array(mask).copy(), device=image.device, dtype=image.dtype) / 255
        else:
            weight = torch.ones_like(image[0])
        splat_mask_cache[key] = weight
    weight = splat_mask_cache[key]
    return (torch.abs(image-target)*weight).sum()/(image.shape[0]*weight.sum())
'''


def patch_training_source(source_path, destination, *, anchor_manifest=None, depth_weight=.02, priority_dir=None, lineage_hook=None, priority_detail=None):
    """Patch exact single markers or fail; original upstream file stays intact."""
    weight = float(depth_weight)
    if not np.isfinite(weight) or not 0 <= weight <= 1:
        raise ValueError('sparse_depth_weight must be finite in [0, 1]')
    original = Path(source_path).read_text(); source = original; hooks = []
    if source.count('def training(') != 1:
        raise RuntimeError('上游 train.py training marker 已改变，无法安全适配')
    if lineage_hook:
        marker='    scene = Scene(dataset, gaussians)'
        if source.count(marker)!=1: raise RuntimeError('上游 Scene 初始化 marker 已改变，无法安全跟踪补全来源')
        source=source.replace(marker,'\n'.join('    '+line for line in lineage_hook.splitlines())+'\n'+marker)
    if priority_dir:
        marker = 'Ll1 = l1_loss(image, gt_image)'
        if source.count(marker) != 1: raise RuntimeError('上游 train.py 物品损失 marker 已改变')
        source = source.replace(marker, 'Ll1 = splat_weighted_l1(image, gt_image, viewpoint_cam.image_name)')
        hooks.append(_PRIORITY_HOOK.replace('__PRIORITY_DIRECTORY__', repr(str(Path(priority_dir).resolve()))))
    if anchor_manifest and weight > 0:
        marker = '        loss.backward()'
        if source.count(marker) != 1: raise RuntimeError('上游 train.py backward marker 已改变')
        source = source.replace(marker, f'        loss = loss + {weight!r} * splat_sparse_depth_loss(render_pkg["depth"], viewpoint_cam.image_name)\n' + marker)
        hooks.append(_DEPTH_HOOK.replace('__ANCHOR_MANIFEST__', repr(str(Path(anchor_manifest).resolve()))))
    position = source.index('def training(')
    source = source[:position] + '\n'.join(hooks) + '\n' + source[position:]
    if priority_detail:
        if not priority_dir:raise ValueError('重点细化需要确认的像素掩码')
        from .priority_detail import patch_detail
        source=patch_detail(source,priority_detail,priority_dir)
    compile(source, str(destination), 'exec')
    Path(destination).write_text(source)
    return {'original_train_sha256': hashlib.sha256(original.encode()).hexdigest(),
            'patched_train_sha256': file_sha256(destination), 'priority_loss': bool(priority_dir),
            'sparse_depth_weight': weight if anchor_manifest else 0.,
            'initialization_lineage': bool(lineage_hook),
            'priority_detail':priority_detail,
            'view_sampling': 'Unchanged original random-without-replacement viewpoint stack'}


def sparse_densification_settings(iterations):
    return {'densify_until_iter': min(6000, max(500, int(iterations) // 2)),
            'densification_interval': 200, 'densify_grad_threshold': .0004}
