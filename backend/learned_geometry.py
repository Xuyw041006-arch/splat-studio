"""Pinned DUSt3R visible-image geometry, with explicit inferred provenance.

No author poses, hidden surfaces, or synthetic photographs are loaded. Padded
preprocessing is an application adaptation, not the official crop evaluation.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

from .reconstruction import ReconstructionCancelled, ReconstructionError

CODE_COMMIT = '4c24a6ebf04809f2cfe59915e51779c8984aaa40'
CROCO_COMMIT = 'd7de0705845239092414480bd829228723bf20de'
MODEL_REPO = 'naver/DUSt3R_ViTLarge_BaseDecoder_512_dpt'
MODEL_REVISION = '61c57447d7b0adc8a1a30b2b0adec7a8935aa2a3'
MODEL_BYTES = 2284790056
MODEL_SHA256 = '7c300a89534113436bde52732d3151212bcbd90f0aa3c8d1496f86d84bfe4b42'
CONFIG_SHA256 = '3a95c4ea45381e13e998ec059e91f641e3d43b3230e85afb29e46b47e76c1ba5'
SOURCE_MANIFEST_SHA256 = '3d2896019061b0f556c2de01284611966e7fd7a2f62c767e1edc3d083337c4ba'
DEPENDENCIES = ('torch', 'torchvision', 'cv2', 'scipy', 'roma', 'einops', 'trimesh', 'matplotlib', 'huggingface_hub', 'safetensors', 'PIL', 'tqdm')


class LearnedGeometryError(ReconstructionError):
    def __init__(self, message, code='learned_geometry_failed', diagnostics=None):
        super().__init__(message)
        self.code = code
        self.diagnostics = diagnostics or {}


def _cancel(cancelled):
    if cancelled and cancelled():
        raise ReconstructionCancelled('Learning-based initialization cancelled')


def _progress(callback, value, message):
    if callback:
        callback(float(value), message)


def _paths(config):
    root = Path(config.get('geometry_cache_dir') or os.environ.get('SPLAT_GEOMETRY_HOME') or
                Path(__file__).resolve().parents[1] / 'work' / 'geometry').expanduser().resolve()
    repo = Path(config.get('geometry_repo_path') or os.environ.get('SPLAT_GEOMETRY_REPO') or root / 'dust3r').expanduser().resolve()
    model = Path(config.get('dust3r_model_path') or root / 'models' / 'dust3r-512-dpt').expanduser().resolve()
    if model.name == 'model.safetensors':
        model = model.parent
    return root, repo, model


def geometry_availability(config=None):
    """Read-only capability check: no torch import, model load, or download."""
    config = config or {}
    root, repo, model = _paths(config)
    missing = [name for name in DEPENDENCIES if importlib.util.find_spec(name) is None]
    source_ok = (repo / 'dust3r/model.py').is_file() and (repo / 'croco/models/croco.py').is_file()
    model_ok = (model / 'model.safetensors').is_file() and (model / 'config.json').is_file()
    allowed_repo = config.get('geometry_model_repo', MODEL_REPO) == MODEL_REPO
    allowed_download = config.get('allow_geometry_download', False) is True
    reasons = []
    if not allowed_repo:
        reasons.append('Only the pinned official DUSt3R 512 dpt checkpoint is supported')
    if not source_ok:
        reasons.append('Pinned DUSt3R/CroCo source is missing; run scripts/setup_geometry.py')
    if missing:
        reasons.append('Missing dependencies: ' + ', '.join(missing))
    if not model_ok:
        reasons.append('Checkpoint download is required' + ('' if allowed_download else ' and not enabled'))
    return {'available': bool(source_ok and not missing and allowed_repo and (model_ok or allowed_download)),
            'requires_download': not model_ok, 'reason': '; '.join(reasons) or 'Local assets available; runtime/hash validation occurs before inference',
            'source_available': source_ok, 'missing_dependencies': missing, 'repo_path': str(repo), 'model_path': str(model),
            'model_repo': MODEL_REPO, 'model_revision': MODEL_REVISION, 'model_sha256': MODEL_SHA256,
            'model_bytes': MODEL_BYTES, 'code_commit': CODE_COMMIT, 'license': 'CC-BY-NC-SA-4.0 plus upstream checkpoint dataset conditions',
            'geometry_source': 'learned_visible_pointmaps', 'backside_completion': False}


def file_sha256(path, cancelled=None):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            _cancel(cancelled)
            digest.update(block)
    return digest.hexdigest()


def download_checkpoint(model_dir, cancelled=None, progress=None):
    """Only the pinned public safetensors/config; never arbitrary pickled models."""
    import urllib.request
    model_dir = Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    for filename, expected_sha, expected_size in (('config.json', CONFIG_SHA256, 450), ('model.safetensors', MODEL_SHA256, MODEL_BYTES)):
        _cancel(cancelled)
        destination = model_dir / filename
        if destination.is_file():
            if file_sha256(destination, cancelled) != expected_sha:
                raise LearnedGeometryError('Existing checkpoint does not match pinned weights; select a clean model directory', 'learned_checkpoint_invalid')
            continue
        temporary = model_dir / (filename + '.download')
        request = urllib.request.Request(f'https://huggingface.co/{MODEL_REPO}/resolve/{MODEL_REVISION}/{filename}',
                                         headers={'User-Agent': 'SplatStudio-pinned-geometry-installer'})
        count, last_update = 0, time.monotonic()
        digest = hashlib.sha256()
        with urllib.request.urlopen(request, timeout=20) as response, temporary.open('wb') as stream:
            while True:
                _cancel(cancelled)
                block = response.read(1024 * 1024)
                if not block:
                    break
                count += len(block)
                if count > expected_size:
                    raise LearnedGeometryError('Checkpoint response exceeded pinned file size', 'learned_checkpoint_invalid')
                stream.write(block); digest.update(block)
                if progress and time.monotonic() - last_update >= .5:
                    _progress(progress, .01 + .04 * count / expected_size, f'Downloading DUSt3R {count}/{expected_size} bytes')
                    last_update = time.monotonic()
        if count != expected_size or digest.hexdigest() != expected_sha:
            raise LearnedGeometryError('Downloaded checkpoint failed exact size/SHA256 verification', 'learned_checkpoint_invalid')
        _cancel(cancelled)
        temporary.replace(destination)
    _cancel(cancelled)
    return model_dir


def verify_source(repo, cancelled=None):
    """Verify a clean development checkout or a manifest-pinned portable bundle."""
    repo = Path(repo).resolve()
    if (repo / '.git').exists():
        for directory, expected in ((repo, CODE_COMMIT), (repo / 'croco', CROCO_COMMIT)):
            try:
                revision = subprocess.check_output(['git', '-C', str(directory), 'rev-parse', 'HEAD'], text=True, stderr=subprocess.DEVNULL).strip()
                dirty = subprocess.check_output(['git', '-C', str(directory), 'status', '--porcelain', '--untracked-files=no'], text=True).strip()
            except (OSError, subprocess.CalledProcessError) as exc:
                raise LearnedGeometryError('Cannot verify pinned geometry source checkout', 'learned_source_invalid') from exc
            if revision != expected or dirty:
                raise LearnedGeometryError('Geometry source differs from the pinned clean checkout', 'learned_source_invalid', {'path': str(directory), 'revision': revision})
    manifest_path = repo / 'PINNED_SOURCE_MANIFEST.json'
    if not manifest_path.is_file() or file_sha256(manifest_path, cancelled) != SOURCE_MANIFEST_SHA256:
        raise LearnedGeometryError('Pinned geometry source manifest missing or modified; rerun setup_geometry.py', 'learned_source_invalid')
    manifest = json.loads(manifest_path.read_text())
    if manifest.get('code_commit') != CODE_COMMIT or manifest.get('croco_commit') != CROCO_COMMIT:
        raise LearnedGeometryError('Geometry manifest revision mismatch', 'learned_source_invalid')
    for relative, digest in manifest['files'].items():
        path = (repo / relative).resolve()
        if not path.is_relative_to(repo) or not path.is_file() or file_sha256(path, cancelled) != digest:
            raise LearnedGeometryError('Bundled geometry source mismatch: ' + relative, 'learned_source_invalid')
    for path in repo.rglob('*'):
        if '.git' in path.parts:
            continue
        if path.is_symlink():
            raise LearnedGeometryError('Symlinked geometry source is not permitted: ' + str(path.relative_to(repo)), 'learned_source_invalid')
        if not path.is_file():
            continue
        executable = path.suffix.lower() in ('.py', '.so', '.pyd', '.dll', '.dylib') or (path.suffix == '.pyc' and '__pycache__' not in path.parts)
        if executable and path.relative_to(repo).as_posix() not in manifest['files']:
            raise LearnedGeometryError('Unregistered executable geometry source: ' + str(path.relative_to(repo)), 'learned_source_invalid')
    return manifest


def verify_assets(config=None, cancelled=None, progress=None):
    config = config or {}
    status = geometry_availability(config)
    if not status['available']:
        raise LearnedGeometryError(status['reason'], 'learned_unavailable', status)
    _, repo, model = _paths(config)
    verify_source(repo, cancelled)
    if status['requires_download']:
        download_checkpoint(model, cancelled, progress)
    for name, digest in (('config.json', CONFIG_SHA256), ('model.safetensors', MODEL_SHA256)):
        path = model / name
        if name.endswith('safetensors') and path.stat().st_size != MODEL_BYTES:
            raise LearnedGeometryError('DUSt3R checkpoint size mismatch', 'learned_checkpoint_invalid')
        if file_sha256(path, cancelled) != digest:
            raise LearnedGeometryError('DUSt3R checkpoint SHA256 mismatch: ' + name, 'learned_checkpoint_invalid')
    return repo, model


def preprocess_image(image, long_edge=512):
    """Resize + symmetric padding, with exact pixel-center mapping, no crop."""
    from PIL import Image
    image = np.asarray(image, dtype=np.uint8)
    if image.ndim != 3 or image.shape[2] != 3 or min(image.shape[:2]) < 16:
        raise LearnedGeometryError('Expected an RGB photograph at least 16 pixels on each side', 'invalid_geometry_image')
    height, width = image.shape[:2]
    factor = long_edge / max(height, width)
    resized_w, resized_h = max(1, round(width * factor)), max(1, round(height * factor))
    tensor_w, tensor_h = math.ceil(resized_w / 16) * 16, math.ceil(resized_h / 16) * 16
    left, top = (tensor_w - resized_w) // 2, (tensor_h - resized_h) // 2
    resized = np.asarray(Image.fromarray(image).resize((resized_w, resized_h), Image.Resampling.LANCZOS))
    padded = np.full((tensor_h, tensor_w, 3), 127, np.uint8)
    padded[top:top + resized_h, left:left + resized_w] = resized
    valid = np.zeros((tensor_h, tensor_w), bool)
    valid[top:top + resized_h, left:left + resized_w] = True
    sx, sy = resized_w / width, resized_h / height
    affine = np.array([[sx, 0, left + (sx - 1) / 2], [0, sy, top + (sy - 1) / 2], [0, 0, 1]], np.float64)
    return {'rgb': padded, 'valid': valid, 'affine': affine, 'inverse_affine': np.linalg.inv(affine),
            'metadata': {'original_size': [width, height], 'resized_size': [resized_w, resized_h],
                         'tensor_size': [tensor_w, tensor_h], 'padding_ltrb': [left, top, tensor_w - resized_w - left, tensor_h - resized_h - top],
                         'valid_rect_xyxy': [left, top, left + resized_w, top + resized_h],
                         'original_to_tensor': affine.tolist(), 'pixel_convention': 'integer centers; resize uses (x+0.5)*scale-0.5',
                         'crop_applied': False, 'exif_orientation_applied': True}}


def _model_principal_point(prep):
    width, height = prep['metadata']['original_size']
    return transform_pixels(np.array([width / 2, height / 2]), prep['affine'])


def transform_pixels(pixels, affine):
    pixels = np.asarray(pixels, np.float64)
    shape = pixels.shape
    xy = pixels.reshape(-1, 2)
    projected = np.c_[xy, np.ones(len(xy))] @ np.asarray(affine).T
    return (projected[:, :2] / projected[:, 2:]).reshape(shape)


def _grid(shape):
    y, x = np.indices(shape)
    return np.stack((x, y), -1).astype(np.float32)


def _backproject(depth, intrinsic, camera_to_world):
    pixels = _grid(depth.shape)
    rays = np.concatenate((pixels, np.ones((*depth.shape, 1), np.float32)), -1) @ np.linalg.inv(intrinsic).T
    local = rays * depth[..., None]
    return (local @ camera_to_world[:3, :3].T + camera_to_world[:3, 3]).astype(np.float32)


def checked_pnp(points, pixels, intrinsic, min_inliers=24, reprojection_px=4., cv2_module=None):
    """Independent checked PnP: failure can never become an identity pose."""
    if cv2_module is None:
        import cv2 as cv2_module
    xyz, xy = np.asarray(points, np.float32), np.asarray(pixels, np.float32)
    good = np.isfinite(xyz).all(1) & np.isfinite(xy).all(1)
    xyz, xy = xyz[good], xy[good]
    if len(xyz) < min_inliers:
        raise LearnedGeometryError('Too few reliable points for PnP', 'learned_pnp_failed', {'candidates': len(xyz)})
    if len(xyz) > 6000:
        choose = np.linspace(0, len(xyz) - 1, 6000, dtype=int)
        xyz, xy = xyz[choose], xy[choose]
    try:
        success, rvec, tvec, inliers = cv2_module.solvePnPRansac(xyz, xy, np.asarray(intrinsic, np.float64), None,
            iterationsCount=200, reprojectionError=reprojection_px, confidence=.999, flags=cv2_module.SOLVEPNP_SQPNP)
    except Exception as exc:
        raise LearnedGeometryError('DUSt3R PnP failed; no substitute pose was created', 'learned_pnp_failed') from exc
    count = 0 if inliers is None else len(inliers)
    if not success or count < min_inliers or count / len(xyz) < .15:
        raise LearnedGeometryError('DUSt3R PnP lacks reliable inliers', 'learned_pnp_failed', {'inliers': count, 'candidates': len(xyz)})
    rotation = cv2_module.Rodrigues(rvec)[0]
    transform = np.eye(4)
    transform[:3, :3], transform[:3, 3] = rotation, np.asarray(tvec).ravel()
    selected = inliers.ravel()
    camera_points = xyz[selected] @ rotation.T + transform[:3, 3]
    projection = camera_points @ np.asarray(intrinsic).T
    projected = projection[:, :2] / projection[:, 2:]
    errors = np.linalg.norm(projected - xy[selected], axis=1)
    positive = float(np.mean(camera_points[:, 2] > 0))
    if not np.isfinite(transform).all() or positive < .9 or not np.isfinite(errors).all():
        raise LearnedGeometryError('Invalid or behind-camera DUSt3R pose', 'learned_pnp_failed')
    median = float(np.median(errors))
    if median > reprojection_px:
        raise LearnedGeometryError('DUSt3R PnP reprojection error is too high', 'learned_pnp_failed')
    return transform, {'inliers': count, 'candidates': len(xyz), 'inlier_ratio': count / len(xyz),
                       'median_reprojection_px': median, 'positive_depth_fraction': positive}


def _connectivity(count, edges):
    seen = {0}
    while True:
        updated = seen | {b for a, b in edges if a in seen} | {a for a, b in edges if b in seen}
        if updated == seen:
            return sorted(seen), len(seen) == count
        seen = updated


def _best_checked_pair(accepted):
    """Prefer measured PnP support; equal ratios alone can favor tiny regions."""
    return max(accepted, key=lambda record: (record['diagnostic']['inlier_ratio'] * record['diagnostic']['inliers'],
                                             -record['diagnostic']['median_reprojection_px']))


def _runtime(repo):
    for name, expected in (('dust3r', repo / 'dust3r'), ('models', repo / 'croco/models')):
        module = sys.modules.get(name)
        if module is not None:
            filename = getattr(module, '__file__', None)
            paths = [Path(filename).resolve()] if filename else [Path(p).resolve() for p in getattr(module, '__path__', [])]
            if not paths or any(not p.is_relative_to(expected) for p in paths):
                raise LearnedGeometryError('A conflicting upstream Python module is loaded: ' + name, 'learned_namespace_conflict')
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))
    from dust3r.model import AsymmetricCroCo3DStereo
    from dust3r.post_process import estimate_focal_knowing_depth
    return AsymmetricCroCo3DStereo, estimate_focal_knowing_depth


def _choose_device(requested):
    import torch
    available = {'cpu': True, 'cuda': torch.cuda.is_available(), 'mps': torch.backends.mps.is_available()}
    if requested == 'auto':
        return 'cuda' if available['cuda'] else 'mps' if available['mps'] else 'cpu'
    if requested not in available or not available[requested]:
        raise LearnedGeometryError('Requested geometry device is unavailable: ' + str(requested), 'learned_device_unavailable')
    return requested


def _sync(device):
    import torch
    if device == 'cuda':
        torch.cuda.synchronize()
    elif device == 'mps':
        torch.mps.synchronize()


def _candidate_pair(record, focal_estimator, preprocessed, threshold):
    import torch
    i, j = record['i'], record['j']
    intrinsics = []
    for index, own_map in ((i, record['a']), (j, record['reverse_a'])):
        prep = preprocessed[index]
        h, w = prep['valid'].shape
        left, top, right, bottom = prep['metadata']['valid_rect_xyxy']
        principal = _model_principal_point(prep)
        pp = torch.tensor(principal - [left, top], dtype=torch.float32)
        focal = float(focal_estimator(torch.from_numpy(own_map[None, top:bottom, left:right]), pp, focal_mode='weiszfeld'))
        if not np.isfinite(focal) or focal <= 0 or not .2 * max(h, w) < focal < 5 * max(h, w):
            raise LearnedGeometryError('Implausible DUSt3R focal estimate', 'learned_focal_failed')
        intrinsics.append(np.array([[focal, 0, principal[0]], [0, focal, principal[1]], [0, 0, 1]], np.float64))
    mask = preprocessed[j]['valid'] & (record['cb'] > threshold) & np.isfinite(record['b']).all(-1)
    pose, diagnostics = checked_pnp(record['b'][mask], _grid(mask.shape)[mask], intrinsics[1])
    centers_baseline = np.linalg.norm(pose[:3, 3])
    depth = record['a'][..., 2][preprocessed[i]['valid'] & (record['ca'] > threshold)]
    median_depth = float(np.median(depth[depth > 0])) if np.any(depth > 0) else 0
    ratio = centers_baseline / max(median_depth, 1e-8)
    if ratio < .005 or not np.isfinite(ratio):
        raise LearnedGeometryError('Insufficient estimated parallax; duplicate/pure-rotation views cannot certify stereo geometry', 'learned_degenerate_pair')
    diagnostics.update({'i': i, 'j': j, 'baseline_to_median_depth': ratio})
    return intrinsics, pose, diagnostics


def _checked_global_initialization(scene, accepted, preprocessed, threshold, cancelled,
                                   tree_initializer=None, apply_initializer=None):
    """Use official pointmap MST, but never its failed-PnP identity fallback."""
    import torch
    if tree_initializer is None or apply_initializer is None:
        from dust3r.cloud_opt.init_im_poses import minimum_spanning_tree, init_from_pts3d
        tree_initializer = tree_initializer or minimum_spanning_tree
        apply_initializer = apply_initializer or init_from_pts3d
    _cancel(cancelled)
    points, _, _, _ = tree_initializer(scene.imshapes, scene.edges, scene.pred_i, scene.pred_j,
                                       scene.conf_i, scene.conf_j, scene.im_conf, scene.min_conf_thr,
                                       scene.device, has_im_poses=False, verbose=False)
    poses, focals, diagnostics = [], [], []
    for index, (pointmap, prep) in enumerate(zip(points, preprocessed)):
        _cancel(cancelled)
        candidates = [record['intrinsics'][slot][0, 0] for record in accepted
                      for slot, frame in enumerate((record['i'], record['j'])) if frame == index]
        if not candidates:
            raise LearnedGeometryError('No checked focal candidate for global camera', 'learned_alignment_failed', {'image_index': index})
        focal = float(np.median(candidates))
        principal = _model_principal_point(prep)
        intrinsic = np.array([[focal, 0, principal[0]], [0, focal, principal[1]], [0, 0, 1]], np.float64)
        array = pointmap.detach().cpu().numpy()
        confidence = scene.im_conf[index].detach().cpu().numpy()
        valid = prep['valid'] & (confidence > threshold) & np.isfinite(array).all(-1)
        world_to_camera, checked = checked_pnp(array[valid], _grid(valid.shape)[valid], intrinsic)
        poses.append(torch.as_tensor(np.linalg.inv(world_to_camera), dtype=pointmap.dtype, device=scene.device))
        focals.append(focal)
        diagnostics.append({'image_index': index, 'focal_network_px': focal, **checked})
    _cancel(cancelled)
    apply_initializer(scene, points, focals, torch.stack(poses))
    return {'method': 'official pointmap MST followed by independent checked per-camera PnP',
            'identity_pose_fallback_allowed': False, 'cameras': diagnostics}


def _assemble_points(pointmaps, depthmaps, intrinsics, poses, confidences, preprocessed, images, config, cancelled):
    count = len(images)
    max_points = max(32, min(500000, int(config.get('geometry_max_points', 30000))))
    threshold = float(config.get('geometry_conf_threshold', 3.))
    relative_tol = float(config.get('geometry_depth_consistency', .1))
    per_view_budget = max_points * 2
    points, colors, scores, observations, support_counts = [], [], [], [], []
    cross_view_support = np.zeros((count, count), np.int64)
    w2cs = [np.linalg.inv(pose) for pose in poses]
    for i in range(count):
        _cancel(cancelled)
        mask = preprocessed[i]['valid'] & (confidences[i] > threshold) & np.isfinite(pointmaps[i]).all(-1) & (depthmaps[i] > 0)
        xy = _grid(mask.shape)[mask]
        xyz = pointmaps[i][mask]
        raw_conf = confidences[i][mask]
        if len(xyz) > per_view_budget:
            choose = np.linspace(0, len(xyz) - 1, per_view_budget, dtype=int)
            xy, xyz, raw_conf = xy[choose], xyz[choose], raw_conf[choose]
        own_pixels = transform_pixels(xy, preprocessed[i]['inverse_affine'])
        own_h, own_w = images[i].shape[:2]
        own_inside = (own_pixels[:, 0] >= 0) & (own_pixels[:, 0] < own_w) & (own_pixels[:, 1] >= 0) & (own_pixels[:, 1] < own_h)
        # Resized border pixels can inverse-map outside original pixel centers
        # when a small image was enlarged. Do not invent out-of-frame tracks.
        xy, xyz, raw_conf, own_pixels = xy[own_inside], xyz[own_inside], raw_conf[own_inside], own_pixels[own_inside]
        tracks = [{str(i): pixel.tolist()} for pixel in own_pixels]
        support = np.ones(len(xyz), np.int32)
        for j in range(count):
            if i == j:
                continue
            cam = xyz @ w2cs[j][:3, :3].T + w2cs[j][:3, 3]
            uvz = cam @ intrinsics[j].T
            uv = uvz[:, :2] / np.maximum(uvz[:, 2:], 1e-12)
            finite = np.isfinite(uv).all(1)
            integer = np.zeros_like(uv, dtype=np.int64)
            integer[finite] = np.rint(uv[finite]).astype(np.int64)
            h, w = depthmaps[j].shape
            inside = finite & (cam[:, 2] > 0) & (integer[:, 0] >= 0) & (integer[:, 0] < w) & (integer[:, 1] >= 0) & (integer[:, 1] < h)
            selected = np.flatnonzero(inside)
            x, y = integer[selected].T
            valid = preprocessed[j]['valid'][y, x] & (confidences[j][y, x] > threshold)
            predicted_depth = depthmaps[j][y, x]
            valid &= np.isfinite(predicted_depth) & (predicted_depth > 0)
            valid &= np.abs(cam[selected, 2] - predicted_depth) <= relative_tol * np.maximum(predicted_depth, 1e-6)
            selected = selected[valid]
            other_pixels = transform_pixels(uv[selected], preprocessed[j]['inverse_affine'])
            other_h, other_w = images[j].shape[:2]
            in_original = (other_pixels[:, 0] >= 0) & (other_pixels[:, 0] < other_w) & (other_pixels[:, 1] >= 0) & (other_pixels[:, 1] < other_h)
            selected, other_pixels = selected[in_original], other_pixels[in_original]
            support[selected] += 1
            cross_view_support[i, j] = len(selected)
            for index, pixel in zip(selected, other_pixels):
                tracks[index][str(j)] = pixel.tolist()
        keep = support >= 2
        pixels = np.rint(own_pixels[keep]).astype(int)
        pixels[:, 0] = np.clip(pixels[:, 0], 0, images[i].shape[1] - 1)
        pixels[:, 1] = np.clip(pixels[:, 1], 0, images[i].shape[0] - 1)
        points.extend(xyz[keep]); colors.extend(images[i][pixels[:, 1], pixels[:, 0]] / 255.)
        scores.extend(np.minimum(.49, .25 + .08 * np.log(np.maximum(raw_conf[keep], 1))))
        observations.extend(track for track, keep_one in zip(tracks, keep) if keep_one)
        support_counts.extend(support[keep].tolist())
    if len(points) < 32:
        raise LearnedGeometryError('Too few points have two-view depth support; inferred geometry was not accepted', 'learned_inconsistent_depth', {'supported_points': len(points)})
    reached, connected = _connectivity(count, list(zip(*np.where(cross_view_support >= 24))))
    if not connected:
        raise LearnedGeometryError('Final aligned depths do not support a connected camera graph', 'learned_inconsistent_depth',
                                   {'supported_points': len(points), 'cross_view_depth_support_counts': cross_view_support.tolist(),
                                    'registered_component': reached, 'minimum_supported_points_per_edge': 24})
    choose = np.linspace(0, len(points) - 1, min(max_points, len(points)), dtype=int)
    return (np.asarray(points, np.float32)[choose], np.asarray(colors, np.float32)[choose], np.asarray(scores, np.float32)[choose],
            [observations[k] for k in choose], {'candidate_supported_points': len(points), 'exported_points': len(choose),
             'min_view_support': 2, 'mean_view_support': float(np.mean(np.asarray(support_counts)[choose])),
             'cross_view_depth_support_counts': cross_view_support.tolist(), 'minimum_supported_points_per_edge': 24,
             'final_depth_support_graph_connected': True,
             'depth_relative_tolerance': relative_tol, 'support_is_predicted_depth_consistency_not_ground_truth': True})


def alignment_samples(tracks, pointmaps, confidences, preprocessed, threshold=3.):
    samples = []
    for track_index, track in enumerate(tracks or []):
        for frame, pixel in track.items():
            frame_index = int(frame)
            if frame_index < 0 or frame_index >= len(pointmaps):
                continue
            original = np.asarray(pixel, np.float64)
            if original.shape != (2,) or not np.isfinite(original).all():
                continue
            prep = preprocessed[frame_index]
            width, height = prep['metadata']['original_size']
            if not (0 <= original[0] < width and 0 <= original[1] < height):
                continue
            x, y = np.rint(transform_pixels(original, prep['affine'])).astype(int)
            h, w = prep['valid'].shape
            if not (0 <= x < w and 0 <= y < h and prep['valid'][y, x]):
                continue
            raw_conf = float(confidences[frame_index][y, x])
            point = pointmaps[frame_index][y, x]
            if not np.isfinite(point).all() or not np.isfinite(raw_conf) or raw_conf <= threshold:
                continue
            samples.append({'track_index': track_index, 'frame_index': frame_index, 'point': point.tolist(), 'pixel': original.tolist(),
                            'confidence': min(.49, .25 + .08 * math.log(max(raw_conf, 1))), 'raw_confidence': raw_conf})
    return samples


def initialize_learned_geometry(image_paths, config=None, progress=None, cancelled=None):
    """Infer cameras and only multi-view-supported, visible-image dense priors."""
    import torch
    from PIL import Image, ImageOps
    config = dict(config or {})
    _cancel(cancelled)
    if not 2 <= len(image_paths) <= 12:
        raise LearnedGeometryError('DUSt3R first-stage initialization accepts 2–12 input photographs', 'learned_view_count')
    provided_intrinsics = config.get('intrinsics_original')
    intrinsic_sources = config.get('intrinsics_sources')
    has_supplied_intrinsics = provided_intrinsics is not None and len(provided_intrinsics) > 0
    approximate_only = (isinstance(intrinsic_sources, list) and len(intrinsic_sources) == len(image_paths)
                        and all(source in ('heuristic', 'exif_35mm_approximate') for source in intrinsic_sources))
    if has_supplied_intrinsics and not approximate_only:
        raise LearnedGeometryError('This DUSt3R adapter estimates intrinsics and cannot yet constrain supplied calibration; select SfM to use calibrated cameras.',
                                   'learned_calibration_unsupported')
    hashes = [file_sha256(path, cancelled) for path in image_paths]
    if len(set(hashes)) != len(hashes):
        raise LearnedGeometryError('Duplicate photographs do not add independent views', 'learned_duplicate_views')
    started = time.perf_counter()
    repo, model_dir = verify_assets(config, cancelled, progress)
    device = _choose_device(config.get('geometry_device', 'auto'))
    cls, focal_estimator = _runtime(repo)
    images, preprocessed = [], []
    for path in image_paths:
        _cancel(cancelled)
        with Image.open(path) as opened:
            if opened.width * opened.height > 80_000_000:
                raise LearnedGeometryError('Photograph exceeds 80 megapixels', 'invalid_geometry_image')
            rgb = np.asarray(ImageOps.exif_transpose(opened).convert('RGB')).copy()
        images.append(rgb); preprocessed.append(preprocess_image(rgb))
    _progress(progress, .06, 'Loading pinned DUSt3R visible-geometry model')
    load_started = time.perf_counter()
    model = cls.from_pretrained(str(model_dir), local_files_only=True).to(device=device, dtype=torch.float32).eval()
    _sync(device)
    load_seconds = time.perf_counter() - load_started
    views = []
    for index, prep in enumerate(preprocessed):
        normalized = torch.from_numpy(prep['rgb'].copy()).permute(2, 0, 1).float() / 127.5 - 1
        views.append({'img': normalized[None], 'true_shape': torch.tensor([prep['valid'].shape], dtype=torch.int32), 'idx': [index], 'instance': [str(index)]})
    records = []
    pairs = [(i, j) for i in range(len(views)) for j in range(len(views)) if i != j]
    inference_started = time.perf_counter()
    try:
        with torch.no_grad():
            for n, (i, j) in enumerate(pairs):
                _cancel(cancelled)
                first = {**views[i], 'img': views[i]['img'].to(device)}
                second = {**views[j], 'img': views[j]['img'].to(device)}
                a, b = model(first, second)
                record = {'i': i, 'j': j, 'a': a['pts3d'][0].cpu().numpy(), 'b': b['pts3d_in_other_view'][0].cpu().numpy(),
                          'ca': a['conf'][0].cpu().numpy(), 'cb': b['conf'][0].cpu().numpy()}
                if any(not np.isfinite(record[key]).all() for key in ('a', 'b', 'ca', 'cb')):
                    raise LearnedGeometryError('DUSt3R returned non-finite predictions', 'learned_nonfinite')
                records.append(record)
                _progress(progress, .08 + .15 * (n + 1) / len(pairs), f'DUSt3R predicted pair {n + 1}/{len(pairs)}')
        _sync(device)
    finally:
        del model
        if device == 'cuda':
            torch.cuda.empty_cache()
        elif device == 'mps':
            torch.mps.empty_cache()
    inference_seconds = time.perf_counter() - inference_started
    by_pair = {(record['i'], record['j']): record for record in records}
    threshold = float(config.get('geometry_conf_threshold', 3.))
    accepted, pair_diagnostics = [], []
    for record in records:
        _cancel(cancelled)
        record['reverse_a'] = by_pair[(record['j'], record['i'])]['a']
        try:
            intrinsic, pose, diagnostic = _candidate_pair(record, focal_estimator, preprocessed, threshold)
            record.update({'intrinsics': intrinsic, 'w2c_j': pose, 'diagnostic': diagnostic})
            accepted.append(record);pair_diagnostics.append({**diagnostic, 'accepted': True})
        except LearnedGeometryError as exc:
            pair_diagnostics.append({'i': record['i'], 'j': record['j'], 'accepted': False, 'code': exc.code, 'message': str(exc), **exc.diagnostics})
    reached, connected = _connectivity(len(views), [(record['i'], record['j']) for record in accepted])
    if not connected:
        raise LearnedGeometryError('Input views lack a connected, checked camera graph; add overlapping views', 'learned_disconnected', {'registered_component': reached, 'pairs': pair_diagnostics})
    alignment_started = time.perf_counter()
    global_initialization = None
    if len(views) == 2:
        best = _best_checked_pair(accepted)
        i, j = best['i'], best['j']
        intrinsics, poses, depths, confidences = [None] * 2, [None] * 2, [None] * 2, [None] * 2
        intrinsics[i], intrinsics[j] = best['intrinsics']
        poses[i], poses[j] = np.eye(4), np.linalg.inv(best['w2c_j'])
        depths[i] = best['a'][..., 2]
        depths[j] = (best['b'] @ best['w2c_j'][:3, :3].T + best['w2c_j'][:3, 3])[..., 2]
        confidences[i], confidences[j] = best['ca'], best['cb']
        alignment_device, actual_steps = 'cpu', 0
    else:
        from dust3r.cloud_opt import global_aligner, GlobalAlignerMode
        from dust3r.cloud_opt.base_opt import global_alignment_iter
        good_pairs = {(record['i'], record['j']) for record in accepted}
        selected = [record for record in records if (record['i'], record['j']) in good_pairs or (record['j'], record['i']) in good_pairs]
        output = {'view1': {'idx': [r['i'] for r in selected]}, 'view2': {'idx': [r['j'] for r in selected]},
                  'pred1': {'pts3d': [], 'conf': []}, 'pred2': {'pts3d_in_other_view': [], 'conf': []}}
        for record in selected:
            for branch, point_key, conf_key, target_key, index in (('pred1', 'a', 'ca', 'pts3d', record['i']), ('pred2', 'b', 'cb', 'pts3d_in_other_view', record['j'])):
                output[branch][target_key].append(torch.from_numpy(record[point_key]))
                conf = np.where(preprocessed[index]['valid'], record[conf_key], 1.).astype(np.float32)
                output[branch]['conf'].append(torch.from_numpy(conf))
        alignment_device = 'cuda' if device == 'cuda' else 'cpu'
        scene = global_aligner(output, device=alignment_device, mode=GlobalAlignerMode.PointCloudOptimizer, verbose=False, min_conf_thr=threshold)
        with torch.no_grad():
            for index, prep in enumerate(preprocessed):
                desired = torch.tensor(_model_principal_point(prep), dtype=scene.im_pp.dtype, device=scene.device)
                scene.im_pp[index].copy_((desired - scene._pp[index]) / 10)
        scene.im_pp.requires_grad_(False)
        _cancel(cancelled)
        global_initialization = _checked_global_initialization(scene, accepted, preprocessed, threshold, cancelled)
        actual_steps = max(1, min(1000, int(config.get('geometry_alignment_steps', 300))))
        optimizer = torch.optim.Adam([p for p in scene.parameters() if p.requires_grad], lr=.01, betas=(.9, .9))
        for step in range(actual_steps):
            _cancel(cancelled)
            loss, _ = global_alignment_iter(scene, step, actual_steps, .01, 1e-6, optimizer, 'cosine')
            if not math.isfinite(loss):
                raise LearnedGeometryError('DUSt3R global alignment became non-finite', 'learned_alignment_failed')
            if step % 10 == 0 or step + 1 == actual_steps:
                _progress(progress, .23 + .12 * (step + 1) / actual_steps, f'Aligning learned cameras {step + 1}/{actual_steps}')
        intrinsics = [x.detach().cpu().numpy() for x in scene.get_intrinsics()]
        poses = [x.detach().cpu().numpy() for x in scene.get_im_poses()]
        depths = [x.detach().cpu().numpy() for x in scene.get_depthmaps()]
        confidences = [x.detach().cpu().numpy() for x in scene.im_conf]
        del scene, optimizer
    if any(not np.isfinite(x).all() for x in intrinsics + poses + depths):
        raise LearnedGeometryError('Learned camera/depth result is non-finite', 'learned_alignment_failed')
    # Restore the exact constrained pixel center after float32 optimization.
    intrinsics = [np.asarray(intrinsic, np.float64).copy() for intrinsic in intrinsics]
    for intrinsic, prep in zip(intrinsics, preprocessed):
        intrinsic[:2, 2] = _model_principal_point(prep)
    pointmaps = [_backproject(depth, intrinsic, pose) for depth, intrinsic, pose in zip(depths, intrinsics, poses)]
    alignment_seconds = time.perf_counter() - alignment_started
    _cancel(cancelled)
    points, colors, confidence, observations, support_diagnostics = _assemble_points(pointmaps, depths, intrinsics, poses, confidences, preprocessed, images, config, cancelled)
    cameras = []
    for index, (image, intrinsic, pose, prep) in enumerate(zip(images, intrinsics, poses, preprocessed)):
        raw_intrinsic = prep['inverse_affine'] @ intrinsic
        cameras.append({'image_index': index, 'image_path': str(image_paths[index]), 'width': image.shape[1], 'height': image.shape[0],
                        'intrinsics': raw_intrinsic.tolist(), 'world_to_camera': np.linalg.inv(pose).tolist(), 'position': pose[:3, 3].tolist(),
                        'camera_source': 'dust3r_inferred', 'pixel_grid_verified': True})
    samples = alignment_samples(config.get('geometry_alignment_observations'), pointmaps, confidences, preprocessed, threshold)
    diagnostics = {'pairs': pair_diagnostics, **support_diagnostics, 'network_device': device, 'alignment_device': alignment_device,
                   'two_view_chosen_direction': [best['i'], best['j']] if len(views) == 2 else None,
                   'two_view_direction_rule': 'maximize checked PnP inlier support times ratio; reprojection median breaks ties',
                   'global_initialization': global_initialization,
                   'alignment_steps': actual_steps, 'model_load_seconds': load_seconds, 'network_seconds': inference_seconds,
                   'alignment_seconds': alignment_seconds, 'total_seconds': time.perf_counter() - started,
                   'supplied_intrinsics_sources': intrinsic_sources,
                   'approximate_intrinsics_supplied_but_not_constrained': bool(has_supplied_intrinsics and approximate_only),
                   'supplied_approximate_intrinsics': provided_intrinsics if approximate_only else None,
                   'preprocessing': [prep['metadata'] for prep in preprocessed], 'input_sha256': hashes,
                   'model_repo': MODEL_REPO, 'model_revision': MODEL_REVISION, 'model_sha256': MODEL_SHA256,
                   'code_commit': CODE_COMMIT, 'croco_commit': CROCO_COMMIT, 'full_scene_calibration_used': False}
    _progress(progress, .38, f'Validated {len(points)} inferred visible-geometry points from {len(images)} views')
    return {'points': points, 'colors': colors, 'confidence': confidence, 'sources': ['inferred'] * len(points),
            'observations': observations, 'cameras': cameras, 'images': images, 'alignment_samples': samples,
            'metadata': {'registered_views': len(cameras), 'input_views': len(images), 'rejected_view_indices': [],
                         'camera_source': 'dust3r_inferred', 'geometry_source': 'learned_visible_pointmaps', 'inferred_points': len(points),
                         'triangulated_points': 0, 'scale': 'arbitrary learned reconstruction scale; not metrically verified',
                         'geometry_diagnostics': diagnostics, 'backside_completion': False,
                         'warnings': ['Geometry and cameras are learned priors; all points remain inferred, including multi-view-consistent points.',
                                      'Only pixels from supplied images are reconstructed; unseen backsides are not generated.',
                                      'Aspect-preserving padding differs from official center-crop evaluation; confidence is an uncalibrated ranking score.',
                                      'DUSt3R code/checkpoint uses non-commercial and dataset-specific license conditions.']}}
