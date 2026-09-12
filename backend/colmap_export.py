"""Export only shared-initializer input geometry to a standalone COLMAP model.

No COLMAP executable, author camera model, or held-out RGB is consulted. Input
names are retained for mask identity; canonical RGB is losslessly PNG encoded
at those internal dataset paths, irrespective of the original filename suffix.
"""
from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path
import struct
import unicodedata

import numpy as np
from PIL import Image, ImageOps


def file_sha256(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def _json_safe(value):
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)):
        return [_json_safe(v) for v in value]
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if isinstance(value, Path):
        return str(value)
    return value


def rotation_to_qvec(rotation):
    """World-to-camera rotation to COLMAP's scalar-first unit quaternion."""
    r = np.asarray(rotation, np.float64)
    if r.shape != (3, 3) or not np.isfinite(r).all() or not np.allclose(r.T @ r, np.eye(3), atol=1e-5) or not np.isclose(np.linalg.det(r), 1., atol=1e-5):
        raise ValueError('相机旋转不是有效的 SO(3) 矩阵')
    # Symmetric eigenproblem is stable at rotations close to 180 degrees.
    k = np.array([[r[0, 0]-r[1, 1]-r[2, 2], r[1, 0]+r[0, 1], r[2, 0]+r[0, 2], r[2, 1]-r[1, 2]],
                  [r[1, 0]+r[0, 1], r[1, 1]-r[0, 0]-r[2, 2], r[2, 1]+r[1, 2], r[0, 2]-r[2, 0]],
                  [r[2, 0]+r[0, 2], r[2, 1]+r[1, 2], r[2, 2]-r[0, 0]-r[1, 1], r[1, 0]-r[0, 1]],
                  [r[2, 1]-r[1, 2], r[0, 2]-r[2, 0], r[1, 0]-r[0, 1], r.trace()]]) / 3
    _, vectors = np.linalg.eigh(k)
    q = vectors[:, -1][[3, 0, 1, 2]]
    return q if q[0] >= 0 else -q


def _canonical_rgb(path, feature_image):
    with Image.open(path) as opened:
        if opened.width * opened.height > 80_000_000:
            raise ValueError('输入照片超过 80MP，无法安全导出')
        canonical = ImageOps.exif_transpose(opened).convert('RGB')
        original = np.asarray(canonical).copy()
    feature = np.asarray(feature_image)
    if feature.ndim != 3 or feature.shape[2] != 3 or feature.dtype != np.uint8:
        raise ValueError('共享初始化 images 必须是 RGB uint8 数组')
    if feature.shape == original.shape and np.array_equal(feature, original):
        return original, [1., 1.], 'exact_exif_transposed_original_rgb'
    # This is exactly the full-frame PIL thumbnail operation used by SfM. An
    # arbitrary same-aspect crop/color edit is deliberately not accepted.
    candidate = Image.fromarray(original)
    candidate.thumbnail((feature.shape[1], feature.shape[0]))
    if candidate.size != (feature.shape[1], feature.shape[0]) or not np.array_equal(np.asarray(candidate), feature):
        raise ValueError('初始化图像不是可验证的原图或完整画幅 thumbnail，不能猜测裁剪/掩码坐标')
    return original, [original.shape[1] / feature.shape[1], original.shape[0] / feature.shape[0]], 'verified_full_frame_thumbnail_to_original_rgb'


def export_initialization_dataset(initialization, image_paths, dataset_dir, cancelled=None):
    """Return dataset/model paths, canonical cameras/observations and provenance.

    Cameras and observations returned here use EXIF-transposed original pixels.
    The initializer's feature grid is validated before any scaling is applied.
    Registered inputs only are written as training images. Source-point arrays
    are retained separately; their identity is not claimed after densification.
    """
    def check_cancel():
        if cancelled and cancelled():
            raise RuntimeError('任务已取消')
    paths = [Path(path).resolve() for path in image_paths]
    if len(paths) < 2 or len(paths) > 80 or len(set(paths)) != len(paths):
        raise ValueError('共享 CUDA 初始化需要 2–80 张不同的实际输入照片')
    names = [p.name for p in paths]
    normalized = [unicodedata.normalize('NFC', name).casefold() for name in names]
    if len(set(normalized)) != len(names) or any('\x00' in name or '\\' in name or ':' in name for name in names):
        raise ValueError('照片文件名不唯一或不可安全用于 COLMAP 身份')
    points = np.asarray(initialization['points'], np.float64)
    colors = np.asarray(initialization['colors'], np.float64)
    confidence = np.asarray(initialization['confidence'], np.float64)
    sources = np.asarray(initialization['sources'])
    observations = initialization['observations']
    images = initialization['images']
    cameras = initialization['cameras']
    count = len(points)
    if points.shape != (count, 3) or count < 4 or colors.shape != points.shape or confidence.shape != (count,) or sources.shape != (count,) or len(observations) != count:
        raise ValueError('共享初始化点/颜色/置信度/来源/观测的数量或形状不一致')
    if not np.isfinite(points).all() or not np.isfinite(colors).all() or not np.isfinite(confidence).all() or np.any((colors < 0) | (colors > 1)) or np.any((confidence < 0) | (confidence > 1)):
        raise ValueError('初始化点、颜色或置信度包含无效值')
    if not set(sources.tolist()) <= {'observed', 'inferred', 'unknown'} or len(images) != len(paths):
        raise ValueError('初始化来源或图像列表不符合共享接口')
    if len(cameras) < 2:
        raise ValueError('至少需要两个实际注册相机')
    dataset = Path(dataset_dir).resolve()
    if dataset.exists() and any(dataset.iterdir()):
        raise ValueError('独立 CUDA 数据集目录必须为空，不能混入其他场景或旧相机')
    dataset.mkdir(parents=True, exist_ok=True)
    rgb_dir = dataset / 'images'; rgb_dir.mkdir()
    model = dataset / 'sparse/0'; model.mkdir(parents=True)
    input_rows = [{'image_index': i, 'image_name': p.name, 'source_sha256': file_sha256(p), 'registered': False} for i, p in enumerate(paths)]
    canonical_cameras = []
    scale_by_view = {}
    for camera in cameras:
        check_cancel()
        i = camera.get('image_index')
        if type(i) is not int or not 0 <= i < len(paths) or i in scale_by_view:
            raise ValueError('初始化相机必须唯一映射到实际 input image_index')
        if camera.get('image_path') and Path(camera['image_path']).resolve() != paths[i]:
            raise ValueError('初始化相机引用了非当前输入照片')
        feature = np.asarray(images[i])
        if feature.shape[:2] != (camera['height'], camera['width']):
            raise ValueError('相机尺寸与初始化图像像素网格不一致')
        original, scale, verification = _canonical_rgb(paths[i], feature)
        k = np.asarray(camera['intrinsics'], np.float64)
        w2c = np.asarray(camera['world_to_camera'], np.float64)
        if k.shape != (3, 3) or not np.isfinite(k).all() or k[0, 0] <= 0 or k[1, 1] <= 0 or not np.allclose(k[2], [0, 0, 1]) or abs(k[0, 1]) > 1e-8 or abs(k[1, 0]) > 1e-8:
            raise ValueError('初始化 K 必须是无 skew 的有效 PINHOLE 内参')
        if w2c.shape != (4, 4) or not np.isfinite(w2c).all() or not np.allclose(w2c[3], [0, 0, 0, 1]):
            raise ValueError('初始化 world_to_camera 必须是有效的 4×4 变换')
        qvec = rotation_to_qvec(w2c[:3, :3])
        k = np.diag([*scale, 1.]) @ k
        height, width = original.shape[:2]
        if not np.allclose([k[0, 2], k[1, 2]], [width / 2, height / 2], atol=1e-5, rtol=0):
            raise ValueError('原始 3DGS 采用中心对称投影；当前偏心主点尚未支持，不能静默忽略 cx/cy')
        encoded_path = rgb_dir / paths[i].name
        Image.fromarray(original).save(encoded_path, format='PNG')
        if file_sha256(paths[i]) != input_rows[i]['source_sha256']:
            raise ValueError('输入照片在初始化导出期间发生变化')
        pixel_sha = hashlib.sha256(str(original.shape).encode() + original.tobytes()).hexdigest()
        input_rows[i].update(registered=True, exported_image_sha256=file_sha256(encoded_path), canonical_rgb_sha256=pixel_sha,
                             width=width, height=height, stored_encoding='PNG', pixel_grid_verification=verification,
                             initialization_to_original_scale_xy=scale)
        scale_by_view[i] = np.asarray(scale)
        canonical_cameras.append({'image_index': i, 'image_name': paths[i].name, 'image_path': str(paths[i]),
            'width': width, 'height': height, 'intrinsics': {'fx': float(k[0, 0]), 'fy': float(k[1, 1]), 'cx': float(k[0, 2]), 'cy': float(k[1, 2])},
            'world_to_camera': w2c.tolist(), 'qvec': qvec.tolist(), 'colmap_camera_id': len(canonical_cameras)+1,
            'mask_pixel_coordinates_verified': True, 'original_registered_pixel_grid_equal': True,
            'rgb_symmetric_projection_matches': True, 'mask_coordinate_source': verification,
            'initialization_to_original_scale_xy': scale})
    centers = np.array([-np.array(c['world_to_camera'])[:3, :3].T @ np.array(c['world_to_camera'])[:3, 3] for c in canonical_cameras])
    if np.max(np.linalg.norm(centers - centers.mean(0), axis=1)) <= 1e-8:
        raise ValueError('初始化相机没有有效位移基线')
    camera_by_index = {c['image_index']: c for c in canonical_cameras}
    canonical_observations = []
    image_tracks = {i: [] for i in camera_by_index}
    point_tracks = [[] for _ in range(count)]
    point_errors = [[] for _ in range(count)]
    skipped_unregistered_observations = 0
    for point_index, obs in enumerate(observations):
        if not isinstance(obs, dict):
            raise ValueError('点观测须为 image_index 到像素坐标的映射')
        converted = {}
        for key, value in obs.items():
            if not str(key).isdigit() or not 0 <= int(key) < len(paths):
                raise ValueError('点观测引用了非输入视角')
            i = int(key)
            if str(i) in converted:
                raise ValueError('同一视角的点观测不能使用重复的数字索引别名')
            if i not in camera_by_index:
                skipped_unregistered_observations += 1
                continue
            xy = np.asarray(value, np.float64)
            if xy.shape != (2,) or not np.isfinite(xy).all():
                raise ValueError('点观测像素无效')
            xy = xy * scale_by_view[i]
            c = camera_by_index[i]
            if not (0 <= xy[0] < c['width'] and 0 <= xy[1] < c['height']):
                raise ValueError('点观测超出已验证图像网格')
            converted[str(i)] = xy.tolist()
            slot = len(image_tracks[i])
            image_tracks[i].append((xy, point_index + 1))
            point_tracks[point_index].append((c['colmap_camera_id'], slot))
            transform = np.asarray(c['world_to_camera'])
            camera_point = transform[:3, :3] @ points[point_index] + transform[:3, 3]
            if camera_point[2] > 1e-6:
                k = c['intrinsics']
                projection = camera_point[:2] / camera_point[2] * [k['fx'], k['fy']] + [k['cx'], k['cy']]
                point_errors[point_index].append(float(np.linalg.norm(projection - xy)))
        canonical_observations.append(converted)
    with (model / 'cameras.bin').open('wb') as stream:
        stream.write(struct.pack('<Q', len(canonical_cameras)))
        for c in canonical_cameras:
            k = c['intrinsics']
            stream.write(struct.pack('<iiQQdddd', c['colmap_camera_id'], 1, c['width'], c['height'], k['fx'], k['fy'], k['cx'], k['cy']))
    with (model / 'images.bin').open('wb') as stream:
        stream.write(struct.pack('<Q', len(canonical_cameras)))
        for c in canonical_cameras:
            w2c = np.asarray(c['world_to_camera']); cid = c['colmap_camera_id']
            stream.write(struct.pack('<idddddddi', cid, *c['qvec'], *w2c[:3, 3], cid))
            stream.write(c['image_name'].encode('utf-8') + b'\0')
            tracks = image_tracks[c['image_index']]
            stream.write(struct.pack('<Q', len(tracks)))
            for xy, point_id in tracks:
                stream.write(struct.pack('<ddq', *xy, point_id))
    rgb8 = np.clip(np.rint(colors * 255), 0, 255).astype(np.uint8)
    with (model / 'points3D.bin').open('wb') as stream:
        stream.write(struct.pack('<Q', count))
        for i, xyz in enumerate(points):
            error = float(np.mean(point_errors[i])) if point_errors[i] else -1.
            stream.write(struct.pack('<QdddBBBdQ', i+1, *xyz, *rgb8[i], error, len(point_tracks[i])))
            for image_id, slot in point_tracks[i]:
                stream.write(struct.pack('<ii', image_id, slot))
    from plyfile import PlyData, PlyElement
    vertices = np.empty(count, dtype=[(name, 'f4') for name in ('x', 'y', 'z', 'nx', 'ny', 'nz')] + [(name, 'u1') for name in ('red', 'green', 'blue')])
    for axis, name in enumerate(('x', 'y', 'z')): vertices[name] = points[:, axis]
    for name in ('nx', 'ny', 'nz'): vertices[name] = 0
    for axis, name in enumerate(('red', 'green', 'blue')): vertices[name] = rgb8[:, axis]
    PlyData([PlyElement.describe(vertices, 'vertex')], text=False).write(model / 'points3D.ply')
    retained = dataset / 'initialization_points.npz'
    np.savez_compressed(retained, points=points, colors=colors.astype(np.float32), confidence=confidence.astype(np.float32), sources=sources.astype('U8'))
    metadata = {'format': 'splat-input-only-colmap/1', 'external_full_scene_camera_or_pointcloud_used': False,
        'input_manifest': input_rows, 'registered_input_indices': [c['image_index'] for c in canonical_cameras],
        'unregistered_input_indices': [i for i in range(len(paths)) if i not in camera_by_index],
        'initialization': _json_safe(initialization.get('metadata', {})), 'point_count': count,
        'initialization_source_counts': dict(Counter(sources.tolist())), 'point_provenance_path': str(retained),
        'point_provenance_sha256': file_sha256(retained), 'pointcloud_sha256': file_sha256(model / 'points3D.ply'),
        'colmap_model_sha256': {name: file_sha256(model / name) for name in ('cameras.bin', 'images.bin', 'points3D.bin')},
        'registered_cameras': canonical_cameras, 'skipped_unregistered_observations': skipped_unregistered_observations,
        'pixel_grid': 'Verified EXIF-transposed original RGB; K and observations rescaled from exact full-frame thumbnail when necessary.',
        'encoding': 'Lossless PNG bytes at original input basenames for mask identity; no EXIF rotation remains.',
        'colmap_point_error': 'Mean projected observation residual in original pixels; -1 indicates no positive-depth observation for a residual.',
        'output_gaussian_provenance': 'Initialization sources are retained per point; densified output Gaussians have no asserted 1:1 source-point identity.'}
    (dataset / 'initialization.json').write_text(json.dumps(_json_safe(metadata), ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    check_cancel()
    return {'dataset_path': str(dataset), 'model_path': str(model), 'metadata': metadata,
            'points': points, 'confidence': confidence, 'sources': sources, 'cameras': canonical_cameras,
            'observations': canonical_observations}
