"""Capture metadata in the EXIF-upright image grid; no GPS retention."""
from __future__ import annotations
import hashlib
import json
import math
import re
import numpy as np


def photo_metadata(image, upright, source_bytes):
    exif = image.getexif()
    # FocalLengthIn35mmFilm is an approximation, not a calibration certificate.
    try:
        f35 = float(exif.get(41989, 0))
    except (TypeError, ValueError, ZeroDivisionError):
        f35 = 0.
    w, h = upright.size
    try:
        orientation=int(exif.get(274,1) or 1)
        if orientation not in range(1,9):orientation=1
    except (TypeError,ValueError,OverflowError):
        orientation=1
    valid = math.isfinite(f35) and 1 <= f35 <= 2000
    focal = f35 * math.hypot(w, h) / math.hypot(36, 24) if valid else .9 * max(w, h)
    return {'source_sha256': hashlib.sha256(source_bytes).hexdigest(),
            'source_size': list(image.size), 'exif_orientation': orientation,
            'coordinate_space': 'uploaded_pixels', 'focal_35mm': f35 if valid else None,
            'intrinsics_source': 'exif_35mm_approximate' if valid else 'heuristic',
            'K': [[focal, 0., w/2], [0., focal, h/2], [0., 0., 1.]]}


def validate_calibration(payload, images):
    """Require exact image binding; do not silently interpret raw rotated pixels."""
    if not isinstance(payload, dict) or set(payload) - {'coordinate_space', 'cameras'}:
        raise ValueError('相机文件仅支持 coordinate_space 和 cameras 字段')
    if payload.get('coordinate_space') != 'uploaded_pixels':
        raise ValueError('相机参数必须使用 EXIF 转正后的 uploaded_pixels 坐标')
    records = payload.get('cameras')
    if not isinstance(records, list) or len(records) != len(images):
        raise ValueError('相机参数数量必须与上传照片一致')
    by_name = {item['name']: item for item in images}
    calibrated = {}
    for record in records:
        if not isinstance(record, dict) or set(record) - {'image_name', 'width', 'height', 'K', 'source_sha256'}:
            raise ValueError('仅支持 PINHOLE 内参；不能导入尚未支持的畸变或外参字段')
        name = record.get('image_name')
        if not isinstance(name,str) or name not in by_name or name in calibrated:
            raise ValueError('image_name 必须唯一匹配上传后的照片编号，如 0000.jpg')
        item = by_name[name]
        if [record.get('width'), record.get('height')] != [item['width'], item['height']]:
            raise ValueError('相机尺寸与 EXIF 转正后的照片不符；禁止混用裁切或缩放坐标')
        expected_sha=item.get('capture', {}).get('source_sha256')
        if not isinstance(expected_sha,str) or not re.fullmatch(r'[a-f0-9]{64}',expected_sha):
            raise ValueError('照片缺少有效 source_sha256；请先由项目读取实际照片建立身份绑定')
        if record.get('source_sha256') != expected_sha:
            raise ValueError('相机 source_sha256 未与原始上传照片绑定')
        try:
            k = np.asarray(record['K'], dtype=float)
        except (KeyError, TypeError, ValueError):
            raise ValueError('K 必须为有限的 3×3 PINHOLE 矩阵') from None
        if (k.shape != (3, 3) or not np.isfinite(k).all() or
                not np.allclose(k[2], [0, 0, 1], atol=1e-8) or
                abs(k[0, 1]) > 1e-8 or abs(k[1, 0]) > 1e-8 or
                min(k[0, 0], k[1, 1]) <= 0 or
                not 0 <= k[0, 2] <= item['width'] or not 0 <= k[1, 2] <= item['height']):
            raise ValueError('K 不是有效的无畸变 PINHOLE 内参')
        calibrated[name] = k.tolist()
    return calibrated


def geometry_config(manifest, config):
    cfg = dict(config)
    cfg['geometry_device']=cfg.get('device','auto')
    images = manifest['images']
    cfg['original_sizes'] = [[i['width'], i['height']] for i in images]
    cfg['intrinsics_original'] = [i.get('calibrated_K') or i.get('capture', {}).get('K') or
        [[.9*max(i['width'], i['height']), 0, i['width']/2],
         [0, .9*max(i['width'], i['height']), i['height']/2], [0, 0, 1]] for i in images]
    cfg['intrinsics_sources'] = ['user_calibrated' if i.get('calibrated_K') else
        i.get('capture', {}).get('intrinsics_source', 'heuristic') for i in images]
    return cfg


def capture_fingerprint(manifest):
    fields = [{k: i.get(k) for k in ('name', 'width', 'height', 'capture', 'calibrated_K')}
              for i in manifest['images']]
    return hashlib.sha256(json.dumps(fields, sort_keys=True, allow_nan=False).encode()).hexdigest()
