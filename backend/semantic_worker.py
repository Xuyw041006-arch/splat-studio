"""Production fixed-PLY semantic refinement, with no benchmark/GT dependencies.

Run ``python -m backend.semantic_worker --help`` for the standalone CUDA worker.
The input must be the complete original 3DGS PLY, never a browser's sampled SH0
scene. This module writes semantic sidecars only; it never writes RGB geometry.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import sys
import tempfile
import time
from collections import Counter
from types import SimpleNamespace

import numpy as np

from .semantic_targets import _read_mask, build_semantic_targets, normalize_label, semantic_targets_metadata
from .semantics import _camera_matrices


_FROZEN_TENSORS = ("_xyz", "_features_dc", "_features_rest", "_scaling", "_rotation", "_opacity")


def semantic_parameter_budget(config, free_device_bytes=None):
    """Bound dense state against currently free VRAM, keeping workspace headroom."""
    gib = 1024 ** 3
    if free_device_bytes is not None and (type(free_device_bytes) is not int or free_device_bytes < 0):
        raise ValueError('Free device memory must be a nonnegative byte count')
    automatic = min(32 * gib, free_device_bytes * 3 // 4) if free_device_bytes is not None else 8 * gib
    requested = config.get('max_semantic_parameter_bytes', automatic)
    if type(requested) is not int or requested < 0 or ('max_semantic_parameter_bytes' in config and requested < 1):
        raise ValueError('max_semantic_parameter_bytes must be a positive integer')
    # An explicit smaller limit remains useful on shared GPUs; it cannot bypass
    # the measured hardware bound or consume the reserved working memory.
    limit = min(requested, automatic)
    return limit, {'free_device_bytes': free_device_bytes, 'free_memory_fraction': .75,
                   'automatic_limit_bytes': automatic, 'applied_limit_bytes': limit,
                   'explicit_limit': 'max_semantic_parameter_bytes' in config,
                   'scope': 'Dense parameters, gradients, Adam state and working copies; remaining memory reserved for rendering and temporary operations.'}


class SemanticRefinementUnsupported(RuntimeError):
    """The original 3DGS CUDA backend is unavailable on this machine."""


class SemanticRefinementCancelled(RuntimeError):
    def __init__(self, output_dir):
        self.output_dir = str(output_dir)
        super().__init__("语义优化已取消；已完成的检查点保留在 " + self.output_dir)


def _sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _write_json(path, value):
    path = Path(path)
    content = json.dumps(value, ensure_ascii=False, allow_nan=False, indent=2)
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent, delete=False) as stream:
        temporary = Path(stream.name)
        stream.write(content)
    try:
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _write_npz(path, **arrays):
    path = Path(path)
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as stream:
        temporary = Path(stream.name)
        np.savez_compressed(stream, **arrays)
    try:
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _ply_description(path, config):
    """Read only the header; the upstream loader reads every vertex afterward."""
    count, properties, comments, in_vertices = None, [], [], False
    with path.open("rb") as stream:
        if stream.readline().strip() != b"ply":
            raise ValueError("输入不是 PLY 文件")
        for _ in range(4096):
            if stream.tell() > 262144:
                raise ValueError("PLY header 过大")
            line = stream.readline()
            if not line:
                raise ValueError("PLY header 不完整")
            text = line.decode("ascii", errors="strict").strip()
            if text == "end_header":
                break
            if text.startswith("comment "):
                comments.append(text[8:])
            elif text.startswith("element "):
                parts = text.split()
                in_vertices = parts[1] == "vertex"
                if in_vertices:
                    if count is not None:
                        raise ValueError("PLY 包含重复 vertex element")
                    count = int(parts[2])
            elif text.startswith("property ") and in_vertices:
                if text.split()[1] == "list":
                    raise ValueError("原始高斯顶点不支持 list 属性")
                properties.append(text.split()[-1])
        else:
            raise ValueError("PLY header 不完整")
    if count is None or count <= 0:
        raise ValueError("PLY 不包含高斯顶点")
    required = {"x", "y", "z", "f_dc_0", "f_dc_1", "f_dc_2", "opacity",
                "scale_0", "scale_1", "scale_2", "rot_0", "rot_1", "rot_2", "rot_3"}
    if required - set(properties):
        raise ValueError("需要原始 3DGS PLY 属性，不能使用普通点云或浏览器预览")
    if len(properties) != len(set(properties)):
        raise ValueError("PLY 属性重复")
    rest = [name for name in properties if name.startswith("f_rest_")]
    degree = next((degree for degree in range(4) if len(rest) == 3 * ((degree + 1) ** 2 - 1)), None)
    if degree is None or set(rest) != {f"f_rest_{i}" for i in range(len(rest))}:
        raise ValueError("PLY 的球谐系数不完整")
    description = " ".join(comments).casefold()
    if config.get("input_is_sampled") or config.get("input_kind", "full_ply") != "full_ply" or any(
            marker in description for marker in ("preview", "sampled", "only dc trained")):
        raise ValueError("拒绝采样/SH0 本地预览模型；请使用完整原始 3DGS PLY")
    expected = config.get("expected_gaussian_count")
    if expected is not None and (not isinstance(expected, int) or isinstance(expected, bool) or expected <= 0 or expected != count):
        raise ValueError("PLY 顶点数与完整源模型身份不一致；可能误用了采样预览")
    return {"gaussian_count": count, "sh_degree": degree,
            "full_model_count_verified": expected is not None,
            "sampling_limit": "Unmarked PLY sampling cannot be inferred; provide expected_gaussian_count from source metadata."}


def _frame_name(record):
    if record.get("frame_id") is not None:
        result = str(record["frame_id"]).strip()
    else:
        value = record.get("image_name") or record.get("image_path") or record.get("name") or ""
        result = str(value).replace("\\", "/").rstrip("/").rsplit("/", 1)[-1]
    if not result:
        raise ValueError("相机需要 frame_id 或 image_name")
    return result


def _training_dimensions(record, size):
    native_width, native_height = record.get("width"), record.get("height")
    if any(not isinstance(value, (int, np.integer)) or isinstance(value, bool) or value < 1
           for value in (native_width, native_height)):
        raise ValueError("相机需要有效的整数 width/height")
    scale = min(1.0, size / max(native_width, native_height))
    return max(1, round(native_width * scale)), max(1, round(native_height * scale))


def _normalize_training_masks(records, camera_map, size):
    """Resize full-frame masks to one registered camera grid before unioning.

    Legacy app masks are full-image grids by the existing API contract. Explicit
    crop/ROI declarations are rejected. Aspect ratio alone cannot prove that an
    unmarked, same-aspect crop is full-frame; the provenance states this limit.
    """
    from PIL import Image
    normalized, provenance = [], []
    full_frame_spaces = {"full_frame", "full_image", "full_frame_pixels", "full_image_pixels", "full_frame_train_pixels"}
    for index, original in enumerate(records):
        record = dict(original)
        frame = record["frame_id"]
        if frame not in camera_map:
            raise ValueError("掩码必须对应已注册相机")
        explicit = False
        for key in ("coordinate_space", "mask_coordinate_space"):
            if key in record:
                if not isinstance(record[key], str) or record[key] not in full_frame_spaces:
                    raise ValueError("只接受 full-frame 掩码，不接受裁剪/ROI 坐标")
                explicit = True
        for key in ("full_frame", "is_full_frame", "mask_full_frame"):
            if key in record:
                if record[key] is not True:
                    raise ValueError("只接受覆盖完整图片坐标范围的 full-frame 掩码")
                explicit = True
        for key in ("crop", "crop_box", "crop_rect", "crop_origin", "crop_offset", "crop_bounds",
                    "mask_roi", "roi", "roi_origin", "mask_offset", "mask_origin", "is_cropped"):
            if key in record and record[key] is not None and record[key] is not False:
                raise ValueError("不接受裁剪/ROI 掩码；请先在完整图片坐标中标注")
        camera = camera_map[frame]
        width, height = _training_dimensions(camera, size)
        mask = _read_mask(record, None)
        source_height, source_width = mask.shape
        # Full-image thumbnail dimensions may round each axis by half a pixel.
        # Require that a single isotropic resize scale explains both dimensions.
        lower = max((source_width - .5) / camera["width"], (source_height - .5) / camera["height"])
        upper = min((source_width + .5) / camera["width"], (source_height + .5) / camera["height"])
        if lower > upper + 1e-12:
            raise ValueError("掩码与注册相机照片的宽高比不一致，不能直接拉伸用于训练")
        resized = np.asarray(Image.fromarray(mask.astype(np.uint8)).resize((width, height), Image.Resampling.NEAREST), dtype=bool)
        provenance.append({"frame_id": frame, "mask_id": str(record.get("mask_id", f"mask-{index}")),
            "annotation_source": str(record.get("annotation_source", "unspecified")),
            "coordinate_source": "explicit_full_frame_declaration" if explicit else "legacy_mask_record_full_image_contract",
            "source_mask_shape": [source_height, source_width],
            "camera_image_shape": [int(camera["height"]), int(camera["width"])],
            "training_mask_shape": [height, width], "full_frame_camera_extent": [0, 0, int(camera["width"]), int(camera["height"])],
            "source_to_training_scale_xy": [width / source_width, height / source_height],
            "resampling": "nearest", "crop_applied": False,
            "source_binary_sha256": hashlib.sha256(mask.astype(np.uint8).tobytes()).hexdigest(),
            "training_binary_sha256": hashlib.sha256(resized.astype(np.uint8).tobytes()).hexdigest(),
            "source_mask_path": str(Path(record["mask_path"]).resolve()) if record.get("mask_path") else None,
            "limit": "A same-aspect undeclared crop cannot be identified from mask dimensions alone."})
        # Threshold/mask-value interpretation has already happened exactly once.
        # The target builder must receive these binary pixels without repeating it.
        record["mask"] = resized
        for key in ("mask_path", "mask_value", "mask_threshold"):
            record.pop(key, None)
        normalized.append(record)
    return normalized, provenance


def _camera_for_record(record, size, torch, device):
    """Use camera calibration directly; do not open RGB photographs."""
    width, height = _training_dimensions(record, size)
    native_width, native_height = record["width"], record["height"]
    transform, k = _camera_matrices(record)
    if not np.allclose(transform[3], [0, 0, 0, 1], atol=1e-5):
        raise ValueError("world_to_camera 不是有效的齐次变换")
    if not np.allclose(transform[:3, :3].T @ transform[:3, :3], np.eye(3), atol=1e-3) or np.linalg.det(transform[:3, :3]) <= 0:
        raise ValueError("相机旋转需要右手系正交矩阵")
    if not np.allclose(k[2], [0, 0, 1], atol=1e-6) or abs(k[0, 1]) > 1e-6 or abs(k[1, 0]) > 1e-6:
        raise ValueError("原始光栅器需要无 skew 的针孔相机")
    if "distortion" in record and np.any(np.asarray(record["distortion"], dtype=float) != 0):
        raise ValueError("请传入去畸变后的相机和掩码")
    k = k.copy()
    k[0] *= width / native_width
    k[1] *= height / native_height
    world_view = torch.tensor(transform, dtype=torch.float32, device=device).T.contiguous()
    near, far = .01, 100.
    projection = torch.zeros((4, 4), dtype=torch.float32, device=device)
    projection[0, 0], projection[1, 1] = 2 * k[0, 0] / width, 2 * k[1, 1] / height
    projection[0, 2], projection[1, 2] = 2 * k[0, 2] / width - 1, 2 * k[1, 2] / height - 1
    projection[2, 2], projection[2, 3], projection[3, 2] = far / (far - near), -far * near / (far - near), 1
    return SimpleNamespace(image_width=width, image_height=height,
        FoVx=2 * math.atan(width / (2 * k[0, 0])), FoVy=2 * math.atan(height / (2 * k[1, 1])),
        world_view_transform=world_view, full_proj_transform=world_view @ projection.T.contiguous(),
        camera_center=torch.linalg.inv(world_view)[3, :3], image_name=_frame_name(record))


def _module_belongs_to_root(module, root):
    """Accept regular modules or genuine namespaces exclusively under one root."""
    root = Path(root).resolve()
    location = getattr(module, "__file__", None)
    search_path = getattr(module, "__path__", None)
    try:
        if location is not None and not Path(location).resolve().is_relative_to(root):
            return False
        if search_path is not None:
            paths = list(search_path)
            if not paths or any(not Path(path).resolve().is_relative_to(root) for path in paths):
                return False
            return True
        return location is not None
    except (TypeError, ValueError, OSError):
        return False


def _load_upstream_model(ply_path, root, degree):
    import torch
    if not torch.cuda.is_available():
        raise SemanticRefinementUnsupported("完整高斯语义优化需要 NVIDIA CUDA；macOS/MPS 暂不支持，请使用 A100/Colab。")
    if not (root / "scene/gaussian_model.py").is_file() or not (root / "gaussian_renderer/__init__.py").is_file():
        raise SemanticRefinementUnsupported("原始 3DGS 仓库缺失；请运行 scripts/setup_upstream.py 或配置 SPLAT_3DGS_REPO。")
    for name in ("scene", "gaussian_renderer", "utils"):
        module = sys.modules.get(name)
        if module is not None and not _module_belongs_to_root(module, root):
            raise SemanticRefinementUnsupported("检测到同名 Python 模块冲突；请通过独立 semantic_worker 进程运行。")
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    try:
        from scene.gaussian_model import GaussianModel
        from gaussian_renderer import render
        model = GaussianModel(degree)
        model.load_ply(str(ply_path))
    except ImportError as exc:
        raise SemanticRefinementUnsupported("原始 3DGS CUDA 扩展不可用；请完成上游环境安装。") from exc
    pipe = SimpleNamespace(compute_cov3D_python=False, convert_SHs_python=False, debug=False, antialiasing=False)
    black = torch.zeros(3, device="cuda")

    def renderer(camera, colors):
        return render(camera, model, pipe, black, override_color=colors.contiguous(), use_trained_exp=False)["render"]

    return model, renderer, torch.device("cuda")


def _tensor_hashes(model, freeze=False):
    result = {}
    for name in _FROZEN_TENSORS:
        tensor = getattr(model, name)
        if freeze:
            tensor.requires_grad_(False)
            tensor.grad = None
        if tensor.requires_grad:
            raise RuntimeError("语义优化不得解冻原始高斯参数: " + name)
        array = tensor.detach().cpu().contiguous().numpy()
        digest = hashlib.sha256()
        digest.update(str((tuple(array.shape), str(array.dtype))).encode())
        if array.size:
            digest.update(memoryview(array).cast("B"))
        result[name] = digest.hexdigest()
    return result


def _initial_probabilities(config, labels, count, ply_sha, device, torch):
    initial = torch.full((count, len(labels)), .05, dtype=torch.float32, device=device)
    prior = config.get("prior")
    if prior is None:
        return initial, {"kind": "uniform_unknown", "probability": .05}
    if not isinstance(prior, dict) or prior.get("source_ply_sha256") != ply_sha or not prior.get("path"):
        raise ValueError("完整语义先验必须提供 path 和匹配的 source_ply_sha256；不能使用预览先验")
    if prior.get("input_is_sampled"):
        raise ValueError("不允许用采样预览作为完整模型语义先验")
    declared_classes = prior.get("classes", [])
    if not isinstance(declared_classes, (list, tuple)) or any(not isinstance(label, str) for label in declared_classes):
        raise ValueError("语义先验需要有序的字符串 classes 列表")
    prior_labels = [normalize_label(label) for label in declared_classes]
    if not prior_labels or len(set(prior_labels)) != len(prior_labels):
        raise ValueError("语义先验需要唯一、有序的 classes")
    path = Path(prior["path"]).expanduser().resolve(strict=True)
    with np.load(path, allow_pickle=False) as data:
        if "probabilities" not in data:
            raise ValueError("语义先验 NPZ 缺少 probabilities")
        if "source_ply_sha256" not in data or "classes" not in data:
            raise ValueError("语义先验 NPZ 必须内嵌 source_ply_sha256 和 classes；外部声明不能替代文件身份")
        embedded_sha, embedded_classes = data["source_ply_sha256"], data["classes"]
        if embedded_sha.ndim != 0 or embedded_sha.dtype.kind != "U" or str(embedded_sha) != ply_sha:
            raise ValueError("语义先验文件内部的源 PLY 身份不匹配")
        if embedded_classes.ndim != 1 or embedded_classes.dtype.kind != "U" or [normalize_label(label) for label in embedded_classes.tolist()] != prior_labels:
            raise ValueError("语义先验文件内部的类别顺序不匹配")
        values = np.asarray(data["probabilities"], dtype=np.float32)
    if values.shape != (count, len(prior_labels)) or not np.isfinite(values).all() or ((values < 0) | (values > 1)).any():
        raise ValueError("语义先验尺寸/概率不合法，或不是源 PLY 的完整顺序")
    for index, label in enumerate(prior_labels):
        if label in labels:
            initial[:, labels.index(label)] = torch.from_numpy(values[:, index].copy()).to(device)
    return initial, {"kind": "verified_full_ply_prior", "path": str(path), "sha256": _sha256(path),
                     "source_ply_sha256": ply_sha, "classes": prior_labels}


def _approved_edges(built, approved):
    if not isinstance(approved, list):
        raise ValueError("approved_hierarchy 必须是显式 {parent, child} 列表")
    requested = {}
    for edge in approved:
        key = (normalize_label(edge["parent"]), normalize_label(edge["child"]))
        relation, source = edge.get("relation", "unspecified"), edge.get("source", "user_explicit")
        if relation not in {"part_of", "contents_of", "member_of", "unspecified"}:
            raise ValueError("批准的层级 relation 必须是 part_of、contents_of 或 member_of")
        if not isinstance(source, str) or not source.strip():
            raise ValueError("批准的层级 source 必须明确记录来源")
        declaration = {"relation": relation, "source": source.strip()}
        if key in requested and requested[key] != declaration:
            raise ValueError("同一父子层级的 relation/source 声明互相冲突")
        requested[key] = declaration
    labels = built["class_labels"]
    accepted, rejected = [], []
    candidates = {(edge["parent"], edge["child"]): edge for edge in built["hierarchy"]["edges"]}
    for parent, child in sorted(requested):
        if (parent, child) in candidates:
            candidate = candidates[(parent, child)]
            accepted.append({**candidate, **requested[(parent, child)],
                             "observed_relation": candidate["relation"], "observation_source": candidate["source"],
                             "approved_by_caller": True})
        else:
            rejected.append({"parent": parent, "child": child, **requested[(parent, child)],
                             "reason": "missing_or_ambiguous_training_mask_support"})
    indices = [(labels.index(edge["parent"]), labels.index(edge["child"])) for edge in accepted]
    return indices, accepted, rejected


def _hierarchy_closure(probabilities, edges):
    result = probabilities.copy()
    # The target builder rejects cycles. Repeated upward unions support a DAG
    # without making the raw field or its diagnostic values appear consistent.
    for _ in range(max(0, result.shape[1] - 1)):
        changed = False
        for parent, child in edges:
            values = np.maximum(result[:, parent], result[:, child])
            changed |= bool(np.any(values != result[:, parent]))
            result[:, parent] = values
        if not changed:
            break
    return result


def _hierarchy_geometry_support(built, records, support_loader):
    """Per-edge child/parent intersection seen in repeated declared views.

    Visit accepted edge regions in frame order through the one-frame cache.
    No semantic probabilities establish this geometric support, and no support
    from a different hierarchy edge can regularize an unrelated child.
    """
    rows = {(row['frame_id'], row['region_id']): row for row in records}
    tracks = {track['id']: {obs['frame']: obs['region_id'] for obs in track['observations']}
              for track in built['tracks']}
    labels = built['class_labels']
    minimum = max(2, built['diagnostics']['matching_config']['min_parent_views'])
    edges = built['hierarchy']['edges']
    counts_by_edge = [Counter() for _ in edges]
    frames_by_edge = [[] for _ in edges]
    requests = {}
    for index, edge in enumerate(edges):
        for view in edge['per_view']:
            if not view['supports'] or not view['explicit_declaration']:
                continue
            requests.setdefault(view['frame'], []).append(index)
    # The point grid projects the complete cloud. Revisiting all frames for
    # each edge used to discard its one-frame cache thousands of times.
    for frame, indices in sorted(requests.items()):
        region_support = {}
        for index in indices:
            edge = edges[index]
            parent_id = tracks[edge['parent']][frame]
            child_id = tracks[edge['child']][frame]
            for region_id in (parent_id, child_id):
                if region_id not in region_support:
                    row = rows[(frame, region_id)]
                    region_support[region_id] = set(support_loader(row, row['mask']))
            counts_by_edge[index].update(region_support[parent_id] & region_support[child_id])
            frames_by_edge[index].append(frame)
    mapping, audit = {}, []
    for index, edge in enumerate(edges):
        counts, valid_frames = counts_by_edge[index], frames_by_edge[index]
        ids = np.asarray(sorted(point for point, views in counts.items() if views >= minimum), dtype=np.int64)
        pair = (labels.index(edge['parent']), labels.index(edge['child']))
        mapping[pair] = ids
        audit.append({'parent': edge['parent'], 'child': edge['child'], 'point_count': len(ids),
            'minimum_distinct_views': minimum, 'supporting_declared_frames': valid_frames,
            'sha256': hashlib.sha256(ids.tobytes()).hexdigest(),
            'rule': 'child_and_parent_visible_point_intersection_in_repeated_explicit_supporting_views'})
    return mapping, audit


def refine_upstream_semantics(ply_path, cameras, masks, output_dir, config=None, progress=None, cancelled=None):
    """Write complete, source-identified semantic sidecars beside a frozen PLY.

    config: semantic_steps (400/1200/2400, default 1200), preset (improved/simple),
    train_size (long edge, default 256), seed, upstream_repo, expected_gaussian_count,
    optional source_ply_sha256, approved_hierarchy=[{parent,child}], and prior
    {path,source_ply_sha256,classes}; resume_path may name a trusted worker resume.pt
    and continues into a new output directory. Pass the original source count to reject a
    sampled PLY even when its header has no sampling comment. Raw probabilities
    preserve optimization output; closed probabilities add only approved parents.

    progress(fraction,message) uses .95-.99. cancelled() is checked at every
    render and around checkpoint I/O. A cancellation raises
    SemanticRefinementCancelled, preserving the most recent complete checkpoint.
    output_dir must be new, so another run's sidecars are never overwritten.
    """
    from .semantic_refinement import SemanticRefinementConfig, refine_semantics
    import torch

    worker_started = time.perf_counter()
    config = dict(config or {})
    granularity = config.get('semantic_granularity', 'flat')
    if granularity not in {'flat', 'multilevel'}:
        raise ValueError('semantic_granularity must be flat or multilevel')
    steps = config.get("semantic_steps", 1200)
    if steps not in (400, 1200, 2400) or isinstance(steps, bool):
        raise ValueError("semantic_steps 必须为 400、1200 或 2400")
    size = config.get("train_size", 256)
    if not isinstance(size, int) or isinstance(size, bool) or not 1 <= size <= 1024:
        raise ValueError("train_size 必须是 [1,1024] 的整数")
    preset = config.get("preset", "improved")
    checkpoints = tuple(sorted({0, steps, *[step for step in (400, 1200, 2400) if step <= steps]}))
    optimizer_config = SemanticRefinementConfig.preset(preset, steps=steps, seed=config.get("seed", 0), checkpoint_steps=checkpoints,
        sampling_schedule=config.get('sampling_schedule', 'legacy'), minimum_sweeps=config.get('minimum_sweeps', 0))
    optimizer_config.validate()
    source = Path(ply_path).expanduser().resolve(strict=True)
    if not source.is_file() or source.suffix.casefold() != ".ply":
        raise ValueError("需要完整的原始 3DGS .ply 文件")
    description = _ply_description(source, config)
    ply_sha = _sha256(source)
    if config.get("source_ply_sha256") is not None and config["source_ply_sha256"] != ply_sha:
        raise ValueError("源 PLY SHA256 不匹配")
    output = Path(output_dir).expanduser().resolve()
    if output.exists():
        raise FileExistsError("输出目录已存在，请为语义优化选择新的目录")
    camera_map = {}
    train_frames = config.get('training_frames')
    held_out_values=config.get('held_out_frames', [])
    for name,values in (('training_frames',train_frames),('held_out_frames',held_out_values)):
        if values is not None and (not isinstance(values,list) or any(not isinstance(x,str) or not x for x in values) or len(set(values))!=len(values)):
            raise ValueError(name+' must be a list of unique exact frame IDs')
    held_out_frames = set(held_out_values)
    cameras=list(cameras)
    available_names=[_frame_name(camera) for camera in cameras]
    if len(set(available_names))!=len(available_names):raise ValueError('相机帧身份重复')
    if train_frames is not None and (set(train_frames)&held_out_frames or set(train_frames)-set(available_names)):
        raise ValueError('Training split overlaps held-out frames or references missing cameras')
    for camera in cameras:
        frame = _frame_name(camera)
        if frame in held_out_frames or (train_frames is not None and frame not in train_frames):
            continue
        if frame in camera_map:
            raise ValueError("相机帧身份重复；请提供唯一 frame_id")
        camera_map[frame] = camera
    if not camera_map:
        raise ValueError("语义优化需要相机列表")
    calibration = []
    for frame, record in sorted(camera_map.items()):
        # Validate projection conventions before reading any mask pixels or
        # loading the CUDA model; this uses only a tiny CPU camera transform.
        _camera_for_record(record, size, torch, "cpu")
        transform, intrinsic = _camera_matrices(record)
        calibration.append({"frame_id": frame, "width": int(record["width"]), "height": int(record["height"]),
                            "world_to_camera": transform.tolist(), "intrinsics": intrinsic.tolist()})
    camera_sha = hashlib.sha256(json.dumps(calibration, sort_keys=True, allow_nan=False).encode()).hexdigest()
    # Resolve a mask to exactly one camera before loading its pixels. An explicit
    # frame_id can distinguish equal basenames from different input directories.
    aliases = {}
    for frame in camera_map:
        basename = frame.replace("\\", "/").rsplit("/", 1)[-1]
        aliases.setdefault(basename, []).append(frame)
    mask_records = []
    for original in masks:
        record = dict(original)
        if record.get("frame_id") is not None:
            frame = str(record["frame_id"]).strip()
        else:
            frame = str(record.get("image_name") or record.get("image_path") or record.get("frame") or "")
            frame = frame.replace("\\", "/").rstrip("/").rsplit("/", 1)[-1]
        if frame not in camera_map:
            matches = aliases.get(frame, [])
            if len(matches) != 1:
                raise ValueError("Mask frame is outside the camera list or ambiguous; provide its exact frame_id")
            frame = matches[0]
        record["frame_id"] = frame
        mask_records.append(record)
    mask_records, mask_coordinate_provenance = _normalize_training_masks(mask_records, camera_map, size)
    if granularity == 'multilevel' and config.get('automatic_hierarchy', False):
        from .semantic_hierarchy import prepare_hierarchy_records
        mask_records = prepare_hierarchy_records(mask_records)
    approved = config.get("approved_hierarchy", [])
    # Explicit approval narrows containment candidate selection; it cannot
    # manufacture support. Every other candidate remains diagnostic only.
    if granularity == 'flat':
        built = build_semantic_targets(mask_records, {"explicit_hierarchy": approved}, allowed_frames=camera_map)
        labels = built["class_labels"]
        if not labels or not any(built["per_frame"].values()):
            raise ValueError("没有可用的语义训练掩码")
        edges, accepted, rejected = _approved_edges(built, approved)
    root = Path(config.get("upstream_repo") or os.environ.get("SPLAT_3DGS_REPO") or
                Path(__file__).resolve().parents[1] / "vendor/gaussian-splatting").expanduser().resolve()

    def check_cancel():
        if cancelled is not None and cancelled():
            raise SemanticRefinementCancelled(output)

    check_cancel()
    model, native_renderer, device = _load_upstream_model(source, root, description["sh_degree"])
    count = int(model.get_xyz.shape[0])
    if count != description["gaussian_count"]:
        raise ValueError("上游模型未加载完整 PLY，禁止优化采样模型")
    frozen_hashes = _tensor_hashes(model, freeze=True)
    matching_started = time.perf_counter()
    hierarchy_support_indices, hierarchy_support_audit = None, None
    if granularity == 'multilevel':
        from .semantic_granularity import build_granularity_targets
        from .semantic_projection import projection_support_loader
        region_records = []
        for i, row in enumerate(mask_records):
            row = dict(row)
            row.setdefault('region_id', row.get('mask_id', f'region-{i}'))
            region_records.append(row)
        # User parent IDs are mapped within a frame, never by a shared name.
        by_object = {}
        for row in region_records:
            if row.get('annotation_source') == 'manual' and row.get('object_id'):
                by_object.setdefault((row['frame_id'], row['object_id']), []).append(row['region_id'])
        for row in region_records:
            if row.get('annotation_source') == 'manual' and row.get('parent_id'):
                parents = by_object.get((row['frame_id'], row['parent_id']), [])
                if len(parents) == 1:
                    row.update(parent_region_id=parents[0], relation=row.get('parent_relation', 'part_of'),
                               relation_source='user_manual_parent_id')
        points = model.get_xyz.detach().cpu().numpy()
        opacity = torch.sigmoid(model._opacity.detach()).cpu().numpy()
        loader = projection_support_loader(points, camera_map, opacity=opacity, cancelled=check_cancel)
        built = build_granularity_targets(sorted(region_records, key=lambda r:r['frame_id']),
            {**config.get('granularity_config', {}), 'geometry_binding': {'source_ply_sha256':ply_sha, 'gaussian_count':count}},
            support_loader=loader, allowed_frames=camera_map)
        built['diagnostics']['visibility_support']={'method':'nearest_opaque_point_center_zbuffer',
            'global_point_id_stride':max(1,math.ceil(count/250_000)),
            'limit':'conservative point-center association heuristic, not full alpha compositing or geometry GT'}
        labels = built['class_labels']
        accepted = built['hierarchy']['edges']
        rejected = built['hierarchy'].get('unknown', [])
        edges = [(labels.index(e['parent']), labels.index(e['child'])) for e in accepted]
        if not labels:
            raise ValueError('No nonempty explicit-granularity regions to optimize')
    # Parameters, gradients, Adam state and working copies are still dense NxC.
    # Refuse excessive allocation instead of silently dropping region tracks.
    estimated_parameter_bytes = count * len(labels) * 24
    free_device_bytes = int(torch.cuda.mem_get_info(device)[0]) if device.type == 'cuda' else None
    max_parameter_bytes, parameter_budget = semantic_parameter_budget(config, free_device_bytes)
    if estimated_parameter_bytes > max_parameter_bytes:
        raise ValueError(f'Semantic field needs about {estimated_parameter_bytes/1024**3:.2f} GiB for {len(labels)} tracks; '
                         f'available dense-state budget is {max_parameter_bytes/1024**3:.2f} GiB; '
                         'use more GPU memory or a smaller explicitly selected budget; no tracks were silently dropped')
    if granularity == 'multilevel':
        hierarchy_support_indices, hierarchy_support_audit = _hierarchy_geometry_support(built, region_records, loader)
    matching_seconds = time.perf_counter() - matching_started
    initial, prior_metadata = _initial_probabilities(config, labels, count, ply_sha, device, torch)
    observations = []
    for frame, items in sorted(built["per_frame"].items()):
        if not items:
            continue
        camera = _camera_for_record(camera_map[frame], size, torch, device)
        height, width = camera.image_height, camera.image_width
        local_labels = sorted(items)
        class_indices = np.asarray([labels.index(label) for label in local_labels], np.int64)
        targets = np.zeros((len(local_labels), height, width), np.float32)
        observed, confidence = np.zeros(len(local_labels), bool), np.zeros(len(local_labels), np.float32)
        for index, label in enumerate(local_labels):
            item = items[label]
            if item["mask"].shape != (height, width):
                raise RuntimeError("内部错误：训练掩码与注册相机的统一网格不一致")
            targets[index] = item["mask"].astype(np.float32)
            observed[index] = bool(targets[index].any())
            confidence[index] = item["confidence"]
        observations.append({"frame_id": frame, "camera": camera, "targets": targets,
                             "observed": observed, "confidence": confidence, 'class_indices':class_indices})
        if optimizer_config.sampling_schedule == 'legacy':
            # Preserve v6 worker supervision digests for trusted old resume files.
            dense=np.zeros((len(labels),height,width),np.float32)
            dense[class_indices]=targets
            full_observed=np.zeros(len(labels),bool);full_observed[class_indices]=observed
            full_confidence=np.zeros(len(labels),np.float32);full_confidence[class_indices]=confidence
            observations[-1].update(targets=dense,observed=full_observed,confidence=full_confidence)
            observations[-1].pop('class_indices')
    if not any(obs["observed"].any() for obs in observations):
        raise ValueError("训练分辨率下所有掩码均为空；请提高 train_size")
    identity = {"source_ply_sha256": ply_sha, "classes": labels,
                "source_tensor_hashes": frozen_hashes, "camera_calibration_sha256": camera_sha}
    if hierarchy_support_audit is not None:
        identity['hierarchy_geometry_support'] = hierarchy_support_audit
    resume_state, resumed_from = None, None
    if config.get("resume_path"):
        resume_path = Path(config["resume_path"]).expanduser().resolve(strict=True)
        bundle = torch.load(resume_path, map_location="cpu", weights_only=True)
        if not isinstance(bundle, dict) or bundle.get("worker_identity") != identity:
            raise ValueError("断点与当前完整 PLY、相机标定或类别顺序不一致")
        resume_state = bundle["optimizer_state"]
        resumed_from = {"path": str(resume_path), "sha256": _sha256(resume_path), "step": resume_state["step"]}
    check_cancel()
    output.mkdir(parents=True, exist_ok=False)
    metadata = {"format": "splat-studio-semantic-sidecar/1", "status": "running", "source_ply": str(source),
                "source_ply_sha256": ply_sha, **description, "classes": labels,
                "class_metadata": built["classes"], "order": "unaltered_source_ply_vertex_order",
                "semantic_steps": steps, "preset": preset, "train_size": size, "seed": optimizer_config.seed,
                'semantic_profile':config.get('semantic_profile'), 'semantic_granularity':granularity,
                'hierarchy_executed':granularity=='multilevel',
                'cross_view_confidence':built.get('diagnostics',{}).get('cross_view_confidence'),
                'training_coverage':{'registered_training_views':len(camera_map), 'views_with_masks':len(observations),
                    'missing_mask_views':sorted(set(camera_map)-{o['frame_id'] for o in observations}),
                    'missing_masks_are_unknown':True},
                'estimated_parameter_bytes':estimated_parameter_bytes,
                'parameter_memory_budget':parameter_budget,
                "training_frames": [obs["frame_id"] for obs in observations], "initialization": prior_metadata,
                "mask_coordinate_provenance": mask_coordinate_provenance,
                "hierarchy_edges": accepted, "hierarchy_rejected": rejected,
                "unconfirmed_containment_is_not_part_hierarchy": True,
                "rgb_geometry_frozen": True, "source_tensor_hashes": frozen_hashes,
                "camera_calibration_sha256": camera_sha, "resumed_from": resumed_from,
                "device": str(device), "ground_truth_dependency": False,
                "class_semantics": "geometrically associated region hypotheses; not verified instances" if granularity=='multilevel' else "multi-label category unions; not separate physical instance identities",
                "raw_and_closed": "closed adds approved ancestors; raw remains available for honest evaluation"}
    if hierarchy_support_audit is not None:
        metadata['hierarchy_geometry_support'] = hierarchy_support_audit
        metadata['hierarchy_regularization_source'] = 'per_edge_repeated_visible_geometry_independent_of_semantic_prior'
    _write_json(output / "catalog.json", metadata)
    _write_json(output / "targets.json", semantic_targets_metadata(built, include_pair_diagnostics=True))
    if granularity == 'multilevel':
        _write_json(output/'granularity.json', {key:built[key] for key in ('tracks','matches','rejected_matches','diagnostics','warnings')})
    last_step, last_checkpoint = (resume_state["step"] if resume_state is not None else 0), None
    started = time.perf_counter()
    sweep_steps=sum(math.ceil(int(obs['observed'].sum())/optimizer_config.channels_per_step) for obs in observations)
    planned_steps = max(steps,sweep_steps*optimizer_config.minimum_sweeps)

    def report(step, message, status="running", **extra):
        fraction = .95 + .04 * min(1., step / planned_steps)
        elapsed = time.perf_counter() - started
        completed_here = step - (resume_state['step'] if resume_state is not None else 0)
        remaining = elapsed * (planned_steps-step)/completed_here if completed_here >= 20 else None
        state = {"status": status, "step": step, "total_steps": planned_steps, "progress": fraction,
                 'remaining_seconds':remaining, 'eta_scope':'current semantic stage; excludes later export and matching',
                 "message": message, "last_checkpoint": last_checkpoint,
                 "wall_seconds": time.perf_counter() - started, **extra}
        _write_json(output / "progress.json", state)
        if progress:
            progress(fraction, message)

    def renderer(camera, colors):
        nonlocal last_step
        check_cancel()
        result = native_renderer(camera, colors)
        last_step += 1
        return result

    def checkpoint(step, probabilities, stats):
        nonlocal last_checkpoint, last_step, planned_steps
        planned_steps = stats.get('planned_steps', planned_steps)
        check_cancel()
        last_step = step
        folder = output / "checkpoints" / f"step_{step:05d}"
        folder.mkdir(parents=True, exist_ok=True)
        _write_npz(folder / "probabilities.npz", probabilities=probabilities.numpy(),
                   source_ply_sha256=np.asarray(ply_sha), classes=np.asarray(labels))
        _write_json(folder / "checkpoint.json", {"step": step, "source_ply_sha256": ply_sha,
                    "gaussian_count": count, "classes": labels, "stats": stats,
                    "geometry_hash_verification": "final verification pending"})
        last_checkpoint = str(folder)
        report(step, f"保存语义检查点 {step}/{planned_steps}")

    def save_state(step, state):
        # torch checkpoint is for trusted local resumption only, not an upload API.
        temporary = output / "resume.tmp.pt"
        torch.save({"worker_identity": identity, "optimizer_state": state}, temporary)
        os.replace(temporary, output / "resume.pt")
        _write_json(output / "resume.json", {"step": step, "source_ply_sha256": ply_sha,
                                            "classes": labels, "path": "resume.pt"})

    def on_progress(step, row):
        nonlocal last_step, planned_steps
        planned_steps = row.get('planned_steps', planned_steps)
        last_step = step
        report(step, f"语义优化 {step}/{planned_steps} · loss {row['loss']:.4f}", loss=row)

    report(last_step, "固定完整高斯模型，开始多视角语义优化")
    try:
        probabilities, stats = refine_semantics(initial, observations, renderer, edges, optimizer_config,
            checkpoint_callback=checkpoint, state_callback=save_state, progress_callback=on_progress,
            resume_state=resume_state, **({'hierarchy_support_indices': hierarchy_support_indices}
                                        if hierarchy_support_indices is not None else {}))
        check_cancel()
        after_hashes = _tensor_hashes(model)
        if after_hashes != frozen_hashes or _sha256(source) != ply_sha:
            raise RuntimeError("原始高斯参数或源 PLY 发生变化，拒绝发布语义结果")
        raw = probabilities.numpy()
        closed = _hierarchy_closure(raw, edges)
        common = {"source_ply_sha256": np.asarray(ply_sha), "classes": np.asarray(labels)}
        _write_npz(output / "probabilities.npz", probabilities=raw, **common)
        _write_npz(output / "closed_probabilities.npz", probabilities=closed, **common)
        raw_membership, closed_membership = raw >= .5, closed >= .5
        _write_npz(output / "semantic_membership.npz", **{f"class_{i}": raw_membership[:, i] for i in range(len(labels))}, **common)
        _write_npz(output / "closed_semantic_membership.npz", **{f"class_{i}": closed_membership[:, i] for i in range(len(labels))}, **common)
        artifacts = {"probabilities": "probabilities.npz", "closed_probabilities": "closed_probabilities.npz",
                     "membership": "semantic_membership.npz", "closed_membership": "closed_semantic_membership.npz"}
        metadata.update(status="completed", step=stats["completed_steps"], geometry_unchanged=True,
                        timing={'worker_seconds':time.perf_counter()-worker_started,
                                'matching_seconds':matching_seconds, 'optimizer_seconds':stats['training_seconds'],
                                'setup_seconds':stats['setup_seconds'], 'checkpoint_seconds':stats['callback_seconds'],
                                'scope':'worker through sidecar serialization; excludes teacher inference and RGB training; components overlap worker total'},
                        coverage=stats.get('coverage'),
                        final_tensor_hashes=after_hashes, threshold=.5, probability_dtype=str(raw.dtype),
                        hierarchy_closure_added_memberships=int((closed_membership & ~raw_membership).sum()),
                        closure_is_not_an_accuracy_measure=True,
                        artifacts={key: {"path": name, "sha256": _sha256(output / name)} for key, name in artifacts.items()})
        _write_json(output / "catalog.json", metadata)
        _write_json(output / "stats.json", stats)
        planned_steps = stats.get('planned_steps', stats['completed_steps'])
        report(stats['completed_steps'], "完整模型语义侧文件已保存；原始 RGB 高斯参数保持一致", status="completed")
        return {"status": "completed", "sidecar_path": str(output / "catalog.json"), "catalog_path": str(output / "catalog.json"),
                "probabilities_path": str(output / "probabilities.npz"), "raw_probabilities_path": str(output / "probabilities.npz"),
                "closed_probabilities_path": str(output / "closed_probabilities.npz"),
                "membership_path": str(output / "semantic_membership.npz"), "raw_membership_path": str(output / "semantic_membership.npz"),
                "closed_membership_path": str(output / "closed_semantic_membership.npz"),
                "stats_path": str(output / "stats.json"), "resume_path": str(output / "resume.pt"),
                "metadata": metadata, "stats": stats}
    except BaseException as exc:
        status = "cancelled" if isinstance(exc, SemanticRefinementCancelled) else "failed"
        try:
            unchanged = _tensor_hashes(model) == frozen_hashes and _sha256(source) == ply_sha
        except Exception:
            unchanged = False
        metadata.update(status=status, last_step=last_step, geometry_unchanged=unchanged,
                        error_type=type(exc).__name__, error=str(exc))
        _write_json(output / "catalog.json", metadata)
        report(last_step, str(exc), status=status, geometry_unchanged=unchanged)
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ply", required=True)
    parser.add_argument("--cameras", required=True, help="JSON list or scene.json with cameras; RGB images are not read")
    parser.add_argument("--masks", required=True, help="JSON list or {masks:[...]}; mask paths are relative to this file")
    parser.add_argument("--output", required=True)
    parser.add_argument("--config", help="JSON worker configuration")
    parser.add_argument("--steps", type=int, choices=(400, 1200, 2400))
    parser.add_argument("--expected-gaussian-count", type=int)
    parser.add_argument("--source-ply-sha256")
    args = parser.parse_args(argv)
    config = json.loads(Path(args.config).read_text()) if args.config else {}
    for option, key in ((args.steps, "semantic_steps"), (args.expected_gaussian_count, "expected_gaussian_count"),
                        (args.source_ply_sha256, "source_ply_sha256")):
        if option is not None:
            config[key] = option
    cameras = json.loads(Path(args.cameras).read_text())
    if isinstance(cameras, dict):
        scene_metadata = cameras.get("metadata", {})
        if "expected_gaussian_count" not in config and scene_metadata.get("source_gaussian_count"):
            config["expected_gaussian_count"] = scene_metadata["source_gaussian_count"]
        cameras = cameras["cameras"]
    mask_file = Path(args.masks).resolve()
    masks = json.loads(mask_file.read_text())
    if isinstance(masks, dict):
        masks = masks["masks"]
    for record in masks:
        if record.get("mask_path") and not Path(record["mask_path"]).is_absolute():
            record["mask_path"] = str(mask_file.parent / record["mask_path"])
    cancelled = False

    def stop(_signal, _frame):
        nonlocal cancelled
        cancelled = True

    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)
    try:
        result = refine_upstream_semantics(args.ply, cameras, masks, args.output, config,
            progress=lambda fraction, message: print(json.dumps({"progress": fraction, "message": message}, ensure_ascii=False), flush=True),
            cancelled=lambda: cancelled)
        print(json.dumps({key: value for key, value in result.items() if key not in {"metadata", "stats"}}, ensure_ascii=False), flush=True)
    except (RuntimeError, ValueError, OSError) as exc:
        print(json.dumps({"status": "cancelled" if isinstance(exc, SemanticRefinementCancelled) else "failed",
                          "error_type": type(exc).__name__, "message": str(exc)}, ensure_ascii=False), file=sys.stderr, flush=True)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
