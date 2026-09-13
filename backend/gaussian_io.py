"""Portable scene JSON and the original INRIA 3DGS vertex PLY convention."""
from __future__ import annotations

import json
import hashlib
import math
import os
from pathlib import Path
import tempfile
from typing import Any

from .scene_limits import MAX_SCENE_GAUSSIANS, validate_gaussian_count

SH_C0 = 0.28209479177387814


def write_scene(scene: dict[str, Any], path: str | Path) -> str:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # Reject NaN/Infinity instead of handing corrupt geometry to the viewer.
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', dir=path.parent, delete=False) as stream:
            temporary = Path(stream.name)
            # Encode bounded Gaussian batches in the C JSON encoder. json.dump
            # iterates over every float in Python; millions of full-SH records
            # otherwise spend minutes writing a canonical scene. Keep ordering
            # and JSON values identical without constructing a scene-sized str.
            encode=lambda value:json.dumps(value,ensure_ascii=False,allow_nan=False,separators=(',', ':'))
            stream.write('{')
            for ordinal,(key,value) in enumerate(scene.items()):
                if not isinstance(key,str):raise TypeError('Scene field names must be strings')
                if ordinal:stream.write(',')
                stream.write(encode(key)+':')
                if key=='gaussians' and isinstance(value,list):
                    stream.write('[')
                    for start in range(0,len(value),25000):
                        if start:stream.write(',')
                        stream.write(encode(value[start:start+25000])[1:-1])
                    stream.write(']')
                else:stream.write(encode(value))
            stream.write('}')
        os.replace(temporary, path)
    finally:
        if temporary is not None:temporary.unlink(missing_ok=True)
    return str(path.resolve())


def write_ply(scene: dict[str, Any], path: str | Path, sh_degree: int = 3) -> str:
    """Write splats compatible with the original default GaussianModel(sh_degree=3).

    Original 3DGS stores *unactivated* scale/opacity and wxyz rotation.  RGB
    properties alone would silently produce an incompatible point-cloud PLY.
    Untrained higher-order SH terms are explicitly zero. Instance IDs and
    provenance remain in the companion scene.json.
    """
    if sh_degree not in (0, 1, 2, 3):
        raise ValueError("sh_degree must be 0, 1, 2 or 3")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    gaussians = scene.get("gaussians", [])
    rest_count = 3 * ((sh_degree + 1) ** 2 - 1)
    properties = ["x", "y", "z", "nx", "ny", "nz", "f_dc_0", "f_dc_1", "f_dc_2", *[f"f_rest_{i}" for i in range(rest_count)], "opacity", "scale_0", "scale_1", "scale_2", "rot_0", "rot_1", "rot_2", "rot_3"]
    with path.open("w", encoding="ascii") as f:
        has_sh=any('sh' in g for g in gaussians)
        note='preserved SH; provenance in scene.json' if has_sh else 'only DC trained; provenance in scene.json'
        f.write(f"ply\nformat ascii 1.0\ncomment splat-studio SH degree {sh_degree}; {note}\n")
        f.write(f"element vertex {len(gaussians)}\n")
        f.writelines(f"property float {name}\n" for name in properties)
        f.write("end_header\n")
        for g in gaussians:
            opacity = min(1 - 1e-6, max(1e-6, float(g["opacity"])))
            rotation = [float(v) for v in g.get("rotation", [1, 0, 0, 0])]
            norm = math.sqrt(sum(v * v for v in rotation)) or 1.0
            dc=[(float(v)-.5)/SH_C0 for v in g['color']];higher=[0.]*rest_count
            if 'sh' in g:
                degree=g.get('sh_degree');source=g['sh']
                if type(degree) is not int or degree not in range(4) or len(source)!=3*(degree+1)**2:
                    raise ValueError('Invalid source SH coefficients')
                source_n=(degree+1)**2;target_n=(sh_degree+1)**2
                dc=[float(source[channel*source_n]) for channel in range(3)]
                higher=[float(source[channel*source_n+j]) if j<source_n else 0.
                        for channel in range(3) for j in range(1,target_n)]
            values = [*g["position"], 0, 0, 0, *dc, *higher, math.log(opacity / (1 - opacity)), *[math.log(max(1e-8, float(v))) for v in g["scale"]], *[v / norm for v in rotation]]
            if not all(math.isfinite(float(v)) for v in values):
                raise ValueError("Cannot export non-finite Gaussian parameters")
            f.write(" ".join(f"{v:.9g}" for v in values) + "\n")
    return str(path.resolve())


def read_ply(path: str | Path, max_points: int | str | None = None, include_sh: bool = True) -> dict[str, Any]:
    """Read original PLY, preserving source row identity and degree 0..3 SH.

    Per-Gaussian sh is channel-major [R coefficients, G coefficients, B coefficients],
    including DC. It is evaluated from camera to point, as in original 3DGS.
    None or 'full' preserves every Gaussian, subject to the desktop resource
    limit. Only an explicit integer opts into preview sampling. Preserving SH
    cannot restore omitted geometry.
    """
    import numpy as np
    from plyfile import PlyData

    if max_points == 'full':
        max_points = None
    if max_points is not None and (type(max_points) is not int or not 1 <= max_points <= MAX_SCENE_GAUSSIANS):
        raise ValueError(f'max_points must be None/full or an integer from 1 to {MAX_SCENE_GAUSSIANS}')
    # Reject excessive full imports from their bounded header before an ASCII
    # parser or a Python object per Gaussian can consume unbounded RAM.
    declared = None
    with Path(path).open('rb') as stream:
        if stream.readline(256).strip() != b'ply':
            raise ValueError('Input is not a PLY file')
        for _ in range(4096):
            line = stream.readline(65537)
            if not line or len(line) > 65536 or stream.tell() > 262144:
                raise ValueError('PLY header is incomplete or exceeds the resource limit')
            fields = line.decode('ascii').strip().split()
            if fields[:2] == ['element', 'vertex']:
                if declared is not None or len(fields) != 3:
                    raise ValueError('PLY vertex count is invalid')
                declared = int(fields[2])
            if fields == ['end_header']:
                break
        else:
            raise ValueError('PLY header is incomplete')
    if declared is None or declared < 1:
        raise ValueError('PLY has no Gaussian vertices')
    validate_gaussian_count(declared if max_points is None else min(declared, max_points))
    data = PlyData.read(str(path))["vertex"]
    source_count = len(data)
    if source_count != declared:
        raise ValueError('PLY vertex count changed while reading')
    indices = range(source_count) if max_points is None or source_count <= max_points else np.linspace(0, source_count - 1, max_points, dtype=np.int64)
    names = data.data.dtype.names or ()
    rest = [name for name in names if name.startswith('f_rest_')]
    degree = next((d for d in range(4) if len(rest) == 3*((d+1)**2-1)), None)
    if degree is None or set(rest) != {f'f_rest_{i}' for i in range(len(rest))}:
        raise ValueError('PLY has incomplete or unsupported spherical harmonics')
    preview_degree = degree if include_sh else 0
    for required in ["x", "y", "z", "f_dc_0", "f_dc_1", "f_dc_2", "opacity", "scale_0", "scale_1", "scale_2", "rot_0", "rot_1", "rot_2", "rot_3"]:
        if required not in names:
            raise ValueError(f"PLY is missing original 3DGS property {required}")
    gaussians = []
    for i in indices:
        op = float(np.clip(data["opacity"][i], -30, 30))
        gaussians.append({
            "position": [float(data[key][i]) for key in ["x", "y", "z"]],
            "color": [float(np.clip(0.5 + SH_C0 * data[key][i], 0, 1)) for key in ["f_dc_0", "f_dc_1", "f_dc_2"]],
            "scale": [float(np.exp(np.clip(data[key][i], -20, 10))) for key in ["scale_0", "scale_1", "scale_2"]],
            "rotation": [float(data[key][i]) for key in ["rot_0", "rot_1", "rot_2", "rot_3"]],
            "opacity": 1 / (1 + math.exp(-op)), "source": "unknown", "source_index": int(i),
        })
        if include_sh:
            n = (preview_degree+1)**2-1
            coefficients = [[float(data[f'f_dc_{channel}'][i]),
                             *[float(data[f'f_rest_{channel*n+j}'][i]) for j in range(n)]]
                            for channel in range(3)]
            if not np.isfinite(coefficients).all():
                raise ValueError('PLY has non-finite spherical harmonics')
            gaussians[-1].update(sh_degree=preview_degree, sh=[value for channel in coefficients for value in channel])
    with Path(path).open("rb") as stream:
        source_hash = hashlib.file_digest(stream, "sha256").hexdigest()
    sampled=len(gaussians)<source_count
    status=(f'按明确预览预算显示 {len(gaussians)} / {source_count} 个高斯，保留 SH{preview_degree} 颜色。'
            if sampled else f'显示全部 {source_count} 个高斯，保留 SH{preview_degree} 颜色。')
    return {"gaussians": gaussians, "cameras": [], "metadata": {"imported_from": str(path), "source_ply_sha256": source_hash, "preview_sh_degree": preview_degree, "source_sh_degree":degree, "sh_layout":"channel_major_including_dc", "source_gaussian_count": source_count, "preview_gaussian_count": len(gaussians), "preview_sampled": sampled, "display_mode":"preview" if sampled else "full", "source_note": ("Original SH coefficients retained; sampling is recorded explicitly in preview_sampled. PLY alone has no observation provenance." if include_sh else "DC-only preview explicitly requested; higher-order SH omitted. PLY alone has no observation provenance."), "warnings": [status, "原始 PLY 未记录观测来源和语义置信度，均按未知处理。"]}}
