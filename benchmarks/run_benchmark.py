#!/usr/bin/env python3
"""Small real-data benchmark. These are NOT original CUDA 3DGS paper metrics.

Example:
 python benchmarks/run_benchmark.py --dataset mipnerf360 --scene-dir /data/bonsai \
   --output benchmarks/results/bonsai --device mps --modes fast,balanced,fine

COLMAP cameras/points are precomputed on the complete dataset (transductive SfM).
Only training-view-visible points are retained, and their colors are re-sampled
from training RGB. Test RGB and annotations never enter optimization or fusion.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import platform
import sys
import time
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from backend.reconstruction import PRESETS, choose_device, render_gaussians, train_gaussians
from backend.semantics import build_priority_masks, discover_objects, fuse_semantics


def write_json(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False))


def evenly_spaced(values, limit):
    if limit <= 0 or len(values) <= limit:
        return list(values)
    return [values[i] for i in np.linspace(0, len(values) - 1, limit).round().astype(int)]


def split_views(records, train_views=16, eval_views=8, holdout=8, annotated_names=()):
    """Resolve disjointness before subsampling; annotated evaluation frames are held out."""
    if holdout < 2:
        raise ValueError("holdout must be at least 2")
    annotated = {Path(x).name for x in annotated_names}
    ordered = sorted(records, key=lambda x: x["name"])
    train, test = [], []
    for i, record in enumerate(ordered):
        if record.get("split") == "test" or Path(record["name"]).name in annotated:
            test.append(record)
        elif record.get("split") == "train":
            train.append(record)
        elif record.get("full_scene_ordinal", i) % holdout == 0:
            test.append(record)
        else:
            train.append(record)
    # Keep labeled test views preferentially, then fill the remaining RGB quota.
    labeled = [r for r in test if Path(r["name"]).name in annotated]
    unlabeled = [r for r in test if Path(r["name"]).name not in annotated]
    if labeled and eval_views > 0:
        test = evenly_spaced(labeled, eval_views) + evenly_spaced(unlabeled, max(0, eval_views - len(labeled))) if len(labeled) < eval_views else evenly_spaced(labeled, eval_views)
    else:
        test = evenly_spaced(test, eval_views)
    train = evenly_spaced(train, train_views)
    # A dataset can contain equivalent names through symlinks: check actual paths.
    if {Path(r["file"]).resolve() for r in train} & {Path(r["file"]).resolve() for r in test}:
        raise ValueError("Training and evaluation views overlap")
    if len(train) < 2 or not test:
        raise ValueError("Need at least two training views and one disjoint evaluation view")
    return train, test


def load_annotations(path, scene_dir):
    """JSON: {images: {image.jpg: [{label, mask_path}]}}; only evaluation GT."""
    if not path:
        return {}
    path = Path(path).resolve()
    data = json.loads(path.read_text())
    records = data.get("images", data)
    if not isinstance(records, dict):
        raise ValueError("annotations must map image names to mask records")
    output = {}
    for name, masks in records.items():
        output[Path(name).name] = []
        for record in masks:
            mask_path = Path(record["mask_path"])
            if not mask_path.is_absolute():
                mask_path = path.parent / mask_path
            output[Path(name).name].append({**record, "mask_path": str(mask_path.resolve())})
    return output


def _colmap_module(filename=None):
    filename = filename or ROOT / "vendor/gaussian-splatting/utils/read_write_model.py"
    spec = importlib.util.spec_from_file_location("benchmark_colmap_model", filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_colmap(scene_dir, model_reader_path=None):
    root = Path(scene_dir).resolve()
    model = next((p for p in (root / "sparse/0", root / "sparse", root / "colmap/sparse/0") if (p / "cameras.bin").exists()), None)
    if model is None:
        raise ValueError("Missing COLMAP sparse/0/cameras.bin, images.bin and points3D.bin")
    colmap = _colmap_module(model_reader_path)
    cameras, images, points = colmap.read_model(str(model), ext=".bin")
    records = []
    missing = []
    for ordinal, image in enumerate(sorted(images.values(), key=lambda im: im.name)):
        path = next((p for p in (root / "images_4" / image.name, root / "images_2" / image.name, root / "images" / image.name, root / image.name) if p.exists()), None)
        if path is None:
            missing.append(image.name)
            continue
        camera = cameras[image.camera_id]
        if camera.model == "PINHOLE":
            fx, fy, cx, cy = camera.params
        elif camera.model == "SIMPLE_PINHOLE":
            fx, cx, cy = camera.params
            fy = fx
        else:
            raise ValueError(f"Camera model {camera.model} is distorted; first use undistorted PINHOLE COLMAP export")
        with Image.open(path) as opened:
            width, height = opened.size
        k = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], float)
        k[0] *= width / camera.width
        k[1] *= height / camera.height
        transform = np.eye(4)
        transform[:3, :3] = colmap.qvec2rotmat(image.qvec)
        transform[:3, 3] = image.tvec
        records.append({"name": image.name, "file": str(path), "id": image.id, "width": width, "height": height,
                        "intrinsics": k.tolist(), "world_to_camera": transform.tolist(),
                        "track_ids": image.point3D_ids, "track_xy": image.xys * [width / camera.width, height / camera.height],
                        # Full-pose-list split is stable even when a download only contains a subset.
                        "full_scene_ordinal": ordinal})
    if not records:
        raise ValueError("No COLMAP images found on disk")
    return records, points, {"source": str(model), "total_registered": len(images), "available_images": len(records), "missing_images": missing,
                             "calibration": "provided full-scene COLMAP poses and sparse points; not an end-to-end SfM benchmark"}


def load_manifest(scene_dir):
    root = Path(scene_dir).resolve()
    data = json.loads((root / "scene.json").read_text())
    records = []
    for i, item in enumerate(data["images"]):
        file = (root / item["file"]).resolve()
        with Image.open(file) as opened:
            width, height = opened.size
        records.append({**item, "file": str(file), "name": item.get("name", file.name), "id": item.get("id", i), "width": width, "height": height})
    npz = np.load(root / data["points_file"], allow_pickle=False)
    return records, {"points": np.asarray(npz["points"]), "colors": np.asarray(npz["colors"])}, {"calibration": data.get("provenance", "user supplied; geometry provenance unverified")}


def initialize_from_train(records, raw_points, kind):
    """COLMAP track observation coordinates ensure no test RGB initializes splats."""
    images = []
    cameras = []
    for i, record in enumerate(records):
        image = np.asarray(Image.open(record["file"]).convert("RGB"))
        images.append(image)
        cameras.append({"image_index": i, "image_path": record["file"], "width": record["width"], "height": record["height"],
                        "intrinsics": record["intrinsics"], "world_to_camera": record["world_to_camera"]})
    if kind == "manifest":
        points = raw_points["points"].copy()
        # Project only into training views; exclude points never visible there.
        colors = np.zeros_like(points, dtype=float)
        counts = np.zeros(len(points), int)
        for image, camera in zip(images, cameras):
            transform = np.asarray(camera["world_to_camera"])
            xyz = points @ transform[:3, :3].T + transform[:3, 3]
            uv = xyz @ np.asarray(camera["intrinsics"]).T
            uv = np.rint(uv[:, :2] / np.maximum(xyz[:, 2:3], 1e-8)).astype(int)
            valid = (xyz[:, 2] > .05) & (uv[:, 0] >= 0) & (uv[:, 0] < image.shape[1]) & (uv[:, 1] >= 0) & (uv[:, 1] < image.shape[0])
            colors[valid] += image[uv[valid, 1], uv[valid, 0]] / 255.
            counts[valid] += 1
        kept = counts > 0
        points, colors = points[kept], colors[kept] / counts[kept, None]
    else:
        accum = {}
        for record, image in zip(records, images):
            xy = np.rint(record["track_xy"]).astype(int)
            for point_id, (x, y) in zip(record["track_ids"], xy):
                if point_id < 0 or point_id not in raw_points or not (0 <= x < image.shape[1] and 0 <= y < image.shape[0]):
                    continue
                color, count = accum.get(int(point_id), (np.zeros(3), 0))
                accum[int(point_id)] = (color + image[y, x] / 255., count + 1)
        ids = sorted(accum)
        points = np.asarray([raw_points[i].xyz for i in ids], float)
        colors = np.asarray([accum[i][0] / accum[i][1] for i in ids], float)
    if len(points) == 0:
        raise ValueError("No sparse points supported by selected training views")
    finite = np.isfinite(points).all(1) & np.isfinite(colors).all(1)
    return points[finite], colors[finite], cameras, images


def rgb_metrics(prediction, target, mask=None):
    """PSNR in [0,1]; SSIM uses an 11x11 Gaussian window, sigma 1.5, valid crop.

    ROI SSIM averages only windows whose center lies in the mask, unlike ROI
    PSNR which uses only masked RGB pixels. Empty ROIs are omitted explicitly.
    """
    x, y = np.asarray(prediction, np.float64), np.asarray(target, np.float64)
    if x.shape != y.shape or x.ndim != 3 or x.shape[2] != 3:
        raise ValueError("RGB inputs must have equal HxWx3 shape")
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("Metrics require finite RGB")
    selected = np.ones(x.shape[:2], bool) if mask is None else np.asarray(mask, bool)
    if selected.shape != x.shape[:2]:
        raise ValueError("ROI shape differs from RGB")
    if not selected.any():
        return {"pixels": 0, "psnr_db": None, "ssim": None}
    mse = float(np.square(x - y)[selected].mean())
    psnr = float(-10 * math.log10(max(mse, 1e-12)))
    size = min(11, min(x.shape[:2]) if min(x.shape[:2]) % 2 else min(x.shape[:2]) - 1)
    if size < 3:
        return {"pixels": int(selected.sum()), "mse": mse, "psnr_db": psnr, "ssim": None}
    blur = lambda value: cv2.GaussianBlur(value, (size, size), 1.5)
    mx, my = blur(x), blur(y)
    vx, vy, covariance = blur(x*x) - mx*mx, blur(y*y) - my*my, blur(x*y) - mx*my
    score = ((2*mx*my + .01**2) * (2*covariance + .03**2)) / ((mx*mx + my*my + .01**2) * (vx + vy + .03**2))
    crop = size // 2
    roi = selected[crop:-crop, crop:-crop]
    ssim = float(score[crop:-crop, crop:-crop][roi].mean()) if roi.any() else None
    return {"pixels": int(selected.sum()), "mse": mse, "psnr_db": psnr, "ssim": ssim}


def inner_boundary(mask, width=2):
    """Inner morphological boundary; zero padding counts image-edge truncation."""
    mask = np.asarray(mask, bool)
    padded = np.pad(mask.astype(np.uint8), width)
    eroded = cv2.erode(padded, np.ones((3, 3), np.uint8), iterations=width)
    return mask & ~eroded[width:-width, width:-width].astype(bool)


def mask_metrics(prediction, truth, boundary_width=2):
    prediction, truth = np.asarray(prediction, bool), np.asarray(truth, bool)
    if prediction.shape != truth.shape:
        raise ValueError("Mask shapes differ")
    intersection = int((prediction & truth).sum())
    union = int((prediction | truth).sum())
    pb, tb = inner_boundary(prediction, boundary_width), inner_boundary(truth, boundary_width)
    boundary_intersection = int((pb & tb).sum())
    boundary_union = int((pb | tb).sum())
    return {"iou": intersection / union if union else None, "intersection": intersection, "union": union,
            "gt_pixels": int(truth.sum()), "predicted_pixels": int(prediction.sum()),
            "boundary_iou": boundary_intersection / boundary_union if boundary_union else None,
            "boundary_width_pixels": boundary_width,
            "gt_touches_image_border": bool(truth[0].any() or truth[-1].any() or truth[:, 0].any() or truth[:, -1].any())}


def resized_view(record, size):
    original = np.asarray(Image.open(record["file"]).convert("RGB"))
    ratio = size / max(original.shape[:2])
    width, height = max(8, round(original.shape[1] * ratio)), max(8, round(original.shape[0] * ratio))
    target = cv2.resize(original, (width, height), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.
    k = np.asarray(record["intrinsics"], np.float32).copy()
    k[0] *= width / record["width"]
    k[1] *= height / record["height"]
    return target, k


def render_scene(scene, record, size=128, device="cpu", labels=None):
    """Render all splats including occluders; semantic alpha is not isolated-object alpha."""
    import torch
    target, k = resized_view(record, size)
    gs = scene["gaussians"]
    tensor = lambda v: torch.as_tensor(v, device=device, dtype=torch.float32)
    logit = lambda v: np.log(np.clip(v, 1e-6, 1-1e-6) / (1-np.clip(v, 1e-6, 1-1e-6)))
    values = [tensor([g["position"] for g in gs]), tensor(np.log([g["scale"] for g in gs])), tensor([g["rotation"] for g in gs]),
              tensor(logit(np.asarray([g["color"] for g in gs]))), tensor(logit(np.asarray([g["opacity"] for g in gs]))),
              tensor(k), tensor(record["world_to_camera"]), target.shape[0], target.shape[1]]
    with torch.no_grad():
        if labels is None:
            prediction = render_gaussians(*values, background=tensor(scene.get("metadata", {}).get("background", [.05]*3)))
        else:
            by_id = {o["id"]: o["label"].casefold().strip() for o in scene.get("objects", [])}
            member = np.array([any(by_id.get(i, "") in labels for i in g.get("semantic_ids", [])) for g in gs], np.float32)
            values[3] = tensor(logit(np.repeat(member[:, None], 3, axis=1)))
            prediction = render_gaussians(*values, background=tensor([0, 0, 0]))
        return prediction.cpu().numpy(), target


def canonical_label(label, vocabulary):
    lower = label.casefold().strip()
    exact = next((v for v in vocabulary if lower == v.casefold()), None)
    if exact:
        return exact
    matches = [v for v in vocabulary if v.casefold() in lower or lower in v.casefold()]
    return max(matches, key=len) if matches else label


def prepare_train_masks(records, output, device, vocabulary, cache_dir, max_views=8):
    """GroundingDINO+SAM sees training views only; save masks for repeatability."""
    selected = evenly_spaced(records, max_views)
    cache = output / "training_masks.json"
    signature = {"images": [r["file"] for r in selected], "vocabulary": vocabulary, "device_requested": device}
    if cache.exists():
        saved = json.loads(cache.read_text())
        if saved.get("signature") == signature and all(Path(r["mask_path"]).exists() for r in saved["masks"]):
            return saved
    started = time.perf_counter()
    masks = []
    devices = []
    warnings = []
    for i, record in enumerate(selected):
        print(f"semantic train view {i+1}/{len(selected)}: {record['name']}", flush=True)
        result = discover_objects([record["file"]], {"provider": "local", "device": device, "candidate_labels": vocabulary,
            "cache_dir": cache_dir, "semantic_max_image_size": 640, "detection_threshold": .25, "max_detections_per_image": 24})
        if result.get("available") is False:
            raise RuntimeError("Semantic models failed: " + str(result.get("warnings")))
        devices.append(result.get("device", "unreported"))
        warnings.extend(result.get("warnings", []))
        for j, mask in enumerate(result.get("masks", [])):
            path = output / "train-masks" / f"{i:03d}-{j:03d}.png"
            path.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(np.asarray(mask["mask"], np.uint8) * 255).save(path)
            masks.append({k: v for k, v in mask.items() if k != "mask"} | {"label": canonical_label(mask["label"], vocabulary), "mask_path": str(path.resolve())})
    saved = {"signature": signature, "masks": masks, "seconds": time.perf_counter()-started, "inference_devices": sorted(set(devices)), "warnings": sorted(set(warnings)),
             "model": "IDEA-Research/grounding-dino-tiny + facebook/sam-vit-base", "ground_truth_used": False}
    write_json(cache, saved)
    return saved


def mean_metric(rows, key):
    values = [r[key] for r in rows if r.get(key) is not None]
    return float(np.mean(values)) if values else None


def evaluate(scene, records, output, annotations, priority_labels, eval_size, device):
    rgb, semantic, roi = [], [], []
    for i, record in enumerate(records):
        prediction, target = render_scene(scene, record, eval_size, device)
        rgb.append({"image": record["name"], **rgb_metrics(prediction, target)})
        # Mask records can include several instances of the same label; category IoU unions them.
        gt_by_label = {}
        for item in annotations.get(Path(record["name"]).name, []):
            mask = np.asarray(Image.open(item["mask_path"]).convert("L")) > 0
            mask = cv2.resize(mask.astype(np.uint8), (target.shape[1], target.shape[0]), interpolation=cv2.INTER_NEAREST).astype(bool)
            label = item["label"].casefold().strip()
            gt_by_label[label] = mask | gt_by_label.get(label, np.zeros(mask.shape, bool))
        union_roi = np.zeros(target.shape[:2], bool)
        for label, mask in gt_by_label.items():
            if label in priority_labels:
                union_roi |= mask
            projected, _ = render_scene(scene, record, eval_size, device, labels={label})
            sem = mask_metrics(projected[:, :, 0] >= .5, mask)
            semantic.append({"image": record["name"], "label": label, **sem})
            if label in priority_labels:
                triptych = np.concatenate([target, np.repeat(mask[:, :, None], 3, axis=2).astype(float), np.repeat((projected[:, :, :1] >= .5), 3, axis=2).astype(float)], axis=1)
                output.mkdir(parents=True, exist_ok=True)
                Image.fromarray((triptych * 255).clip(0,255).astype(np.uint8)).save(output / f"{i:02d}-{Path(record['name']).stem}-priority-mask.png")
        if union_roi.any():
            roi.append({"image": record["name"], **rgb_metrics(prediction, target, union_roi)})
        if i < 8:
            tiles = [(target*255).clip(0,255).astype(np.uint8), (prediction*255).clip(0,255).astype(np.uint8), (np.abs(prediction-target)*255).clip(0,255).astype(np.uint8)]
            strip = Image.new("RGB", (target.shape[1]*3, target.shape[0]+24), "white")
            draw = ImageDraw.Draw(strip)
            for j, (tile, label) in enumerate(zip(tiles, ["Held-out GT", "Render", "Absolute error"])):
                strip.paste(Image.fromarray(tile), (j*target.shape[1], 24))
                draw.text((j*target.shape[1]+3, 4), label, fill="black")
            output.mkdir(parents=True, exist_ok=True)
            strip.save(output / f"{i:02d}-{Path(record['name']).stem}.png")
    labels = sorted({r["label"] for r in semantic})
    by_label = {label: {"mean_iou": mean_metric([r for r in semantic if r["label"] == label], "iou"),
                        "mean_boundary_iou": mean_metric([r for r in semantic if r["label"] == label], "boundary_iou"),
                        "views": sum(r["label"] == label for r in semantic)} for label in labels}
    border_groups = {"touching_image_border" if touches else "inside_image": {
        "object_view_pairs": sum(r["gt_touches_image_border"] == touches for r in semantic),
        "mean_iou": mean_metric([r for r in semantic if r["gt_touches_image_border"] == touches], "iou"),
        "mean_boundary_iou": mean_metric([r for r in semantic if r["gt_touches_image_border"] == touches], "boundary_iou")}
        for touches in [False, True]}
    return {"rgb": {"mean_psnr_db": mean_metric(rgb, "psnr_db"), "mean_ssim": mean_metric(rgb, "ssim"), "views": len(rgb), "per_view": rgb},
            "priority_roi": {"mean_psnr_db": mean_metric(roi, "psnr_db"), "mean_ssim": mean_metric(roi, "ssim"), "views": len(roi), "per_view": roi},
            "semantics": {"mean_iou": mean_metric(semantic, "iou"), "class_mean_iou": float(np.mean([v["mean_iou"] for v in by_label.values() if v["mean_iou"] is not None])) if by_label else None,
                          "annotated_object_view_pairs": len(semantic), "per_label": by_label, "per_object_view": semantic,
                          "mean_boundary_iou": mean_metric(semantic, "boundary_iou"), "image_border_groups": border_groups,
                          "boundary_definition": f"Intersection/union of inner morphological boundary bands, width 2 pixels at long-edge {eval_size} evaluation resolution, zero padding includes image edges",
                          "definition": "Held-out 2D projected category-mask IoU (NOT 3D IoU): masks rendered from training-view-fused 3D splats, full-scene alpha compositing, fixed threshold 0.5; unknown splats occlude; absent predictions score zero against nonempty GT"}}


def run(args):
    import torch
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    code_hashes = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(__file__), ROOT / "backend/reconstruction.py", ROOT / "backend/semantics.py"]}
    device = choose_device(args.device)
    annotations = load_annotations(args.annotations, args.scene_dir)
    records, raw_points, provenance = load_manifest(args.scene_dir) if args.dataset == "manifest" else load_colmap(args.scene_dir)
    train, test = split_views(records, args.train_views, args.eval_views, args.holdout, annotations)
    points, colors, cameras, images = initialize_from_train(train, raw_points, args.dataset)
    vocabulary = [s.strip() for s in args.candidate_labels.split(",") if s.strip()]
    priorities = {s.strip().casefold() for s in args.priority_labels.split(",") if s.strip()}
    semantic_run = None
    if args.semantic:
        if not vocabulary:
            vocabulary = sorted({m["label"] for masks in annotations.values() for m in masks})
        if not vocabulary:
            raise ValueError("Semantic benchmark needs --candidate-labels or annotation label vocabulary")
        semantic_run = prepare_train_masks(train, output, device, vocabulary, args.cache_dir, args.semantic_views)
    masks = semantic_run["masks"] if semantic_run else []
    priority_masks = build_priority_masks([r["file"] for r in train], masks, list(priorities)) if priorities else {}
    split = {"train": [r["name"] for r in train], "test": [r["name"] for r in test]}
    summary = {"dataset": args.dataset, "scene": Path(args.scene_dir).name, "device": device, "platform": platform.platform(), "torch": torch.__version__,
               "backend": "torch_reference_preview", "evaluation_long_edge": args.eval_size, "seed": 7, "split": split,
               "code_sha256": code_hashes,
               "initialization": {"train_visible_sparse_points": len(points), "color_source": "selected training RGB only", **provenance},
               "semantics_training": semantic_run, "priority_labels": sorted(priorities), "priority_mask_views": len(priority_masks), "runs": [],
               "limitations": ["Bounded SH0 reference backend: <=256 splats, no densification/pruning. Not original CUDA 3DGS quality or a paper reproduction.",
                 "Provided COLMAP poses and geometry use full-scene SfM; RGB fitting and semantic fusion exclude test views, but geometry is transductive.",
                 "Low-resolution held-out rendering only; PSNR/SSIM do not measure metric geometry accuracy.",
                 "Single scene/run seed; mode or ROI improvements are observations, not statistical generalization.",
                 "ROI and semantic metrics require external withheld manual masks. Detection confidence is never treated as segmentation accuracy.",
                 "Semantic classes are fixed by candidate vocabulary; benchmark does not establish part hierarchy quality or instance-level association accuracy."],
               "metric_definitions": {"PSNR": "Mean per-view 10 log10(1/MSE), RGB [0,1]; all modes use identical held-out images and evaluation size",
                    "SSIM": "11x11 Gaussian window sigma1.5, constants0.01^2/0.03^2, valid spatial crop, mean RGB",
                    "timing": "wall time on the recorded device; training total includes backend initial/final training-view loss rendering; semantic inference measured separately",
                    "priority_ablation": "balanced baseline vs balanced with 4x ROI RGB weighting and changed sampling probabilities at identical steps, resolution and splat budget; same RNG seed; initialization identities may differ"}}
    write_json(output / "summary.json", summary)
    modes = [x.strip() for x in args.modes.split(",")]
    if any(m not in PRESETS for m in modes):
        raise ValueError("Unknown mode")
    experiments = [(mode, False) for mode in modes]
    if priorities and priority_masks:
        if "balanced" not in modes:
            experiments.append(("balanced", False))
        experiments.append(("balanced", True))
    for mode, priority in experiments:
        name = mode + ("-priority" if priority else "")
        print(f"RUN {name}: train={len(train)} test={len(test)} points={len(points)} device={device}", flush=True)
        cfg = {"device": device, "mode": mode}
        if priority:
            cfg.update(priority_masks=priority_masks, priority_strength=3)
        started = time.perf_counter()
        trained = train_gaussians(points, colors, cameras, images, cfg,
            progress=lambda _, msg: print(f"[{name}] {msg}", flush=True))
        training_seconds = time.perf_counter() - started
        scene = {"gaussians": trained["gaussians"], "cameras": cameras, "metadata": trained["metrics"]}
        fusion_start = time.perf_counter()
        if masks:
            scene = fuse_semantics(scene, config={"masks": masks, "priority_labels": list(priorities)})
        fusion_seconds = time.perf_counter()-fusion_start
        write_json(output / name / "scene.json", scene)
        evaluation_start = time.perf_counter()
        evaluated = evaluate(scene, test, output / name / "renders", annotations, priorities, args.eval_size, device)
        row = {"name": name, "mode": mode, "priority": priority, "settings": PRESETS[mode], "training_wall_seconds": training_seconds,
               "fusion_seconds": fusion_seconds, "evaluation_seconds": time.perf_counter()-evaluation_start,
               "training_metrics": trained["metrics"], "semantic_fusion": scene["metadata"].get("semantics"), **evaluated}
        summary["runs"].append(row)
        write_json(output / name / "metrics.json", row)
        write_json(output / "summary.json", summary)
        print(f"RESULT {name} held-out PSNR={row['rgb']['mean_psnr_db']:.3f} SSIM={row['rgb']['mean_ssim']:.4f} seconds={training_seconds:.2f}", flush=True)
    base = next((r for r in summary["runs"] if r["name"] == "balanced"), None)
    prior = next((r for r in summary["runs"] if r["name"] == "balanced-priority"), None)
    if base and prior:
        summary["priority_ablation"] = {"equal_budget": base["settings"] == prior["settings"],
            "rgb_psnr_delta_db": prior["rgb"]["mean_psnr_db"]-base["rgb"]["mean_psnr_db"],
            "roi_psnr_delta_db": prior["priority_roi"]["mean_psnr_db"]-base["priority_roi"]["mean_psnr_db"] if base["priority_roi"]["views"] else None,
            "roi_ssim_delta": prior["priority_roi"]["mean_ssim"]-base["priority_roi"]["mean_ssim"] if base["priority_roi"]["mean_ssim"] is not None else None,
            "note": "Positive delta supports improvement only for this scene/split/budget. Negative values are retained and reported."}
    write_json(output / "summary.json", summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=["mipnerf360", "lerf-ovs", "manifest"], required=True)
    parser.add_argument("--scene-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", choices=["auto", "cpu", "mps", "cuda"], default="mps")
    parser.add_argument("--modes", default="fast,balanced,fine")
    parser.add_argument("--train-views", type=int, default=16)
    parser.add_argument("--eval-views", type=int, default=8)
    parser.add_argument("--holdout", type=int, default=8)
    parser.add_argument("--eval-size", type=int, default=128)
    parser.add_argument("--annotations")
    parser.add_argument("--semantic", action="store_true")
    parser.add_argument("--semantic-views", type=int, default=8)
    parser.add_argument("--candidate-labels", default="")
    parser.add_argument("--priority-labels", default="")
    parser.add_argument("--cache-dir")
    args = parser.parse_args()
    if not 16 <= args.eval_size <= 256:
        parser.error("eval-size must be between 16 and 256")
    if args.train_views < 2 or args.eval_views < 1:
        parser.error("train-views >=2 and eval-views >=1 required")
    run(args)


if __name__ == "__main__":
    main()
