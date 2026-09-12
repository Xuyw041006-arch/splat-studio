#!/usr/bin/env python3
"""Evaluate a FULL upstream CUDA PLY with cached train masks and held-out GT.

Run from an environment with the upstream CUDA extensions installed. No sampling,
no predicted masks on test views, no large semantic JSON export. CPU RAM use of
the application fusion layer grows with the complete point count.

--split must identify the actual RGB TRAINING split, not an invented eval split.
It accepts {train:[image names],test:[image names]} or a benchmark summary's split.
The script validates membership but cannot verify a training process retrospectively.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from benchmarks.run_benchmark import load_annotations, load_colmap, mask_metrics, mean_metric, resized_view, rgb_metrics, write_json
from backend.semantics import fuse_semantics


def camera_for_record(record, size, torch, device="cuda"):
    target, k = resized_view(record, size)
    height, width = target.shape[:2]
    transform = torch.tensor(record["world_to_camera"], dtype=torch.float32, device=device)
    world_view = transform.T.contiguous()
    near, far = .01, 100.
    projection = torch.zeros((4,4), dtype=torch.float32, device=device)
    projection[0,0], projection[1,1] = float(2*k[0,0]/width), float(2*k[1,1]/height)
    projection[0,2], projection[1,2] = float(2*k[0,2]/width-1), float(2*k[1,2]/height-1)
    projection[2,2], projection[2,3], projection[3,2] = far/(far-near), -far*near/(far-near), 1
    projection = projection.T.contiguous()
    camera = SimpleNamespace(image_width=width, image_height=height, FoVx=2*math.atan(width/(2*k[0,0])),
        FoVy=2*math.atan(height/(2*k[1,1])), world_view_transform=world_view,
        full_proj_transform=(world_view @ projection), camera_center=torch.linalg.inv(world_view)[3,:3],
        image_name=Path(record["name"]).stem)
    return camera, target


def read_training_masks(filename, records_by_name, actual_train_names):
    path = Path(filename).resolve()
    raw = json.loads(path.read_text())
    masks = raw.get("masks", raw) if isinstance(raw, dict) else raw
    output = []
    for item in masks:
        image_name = Path(item.get("image_path", item.get("image_name", ""))).name
        if image_name not in actual_train_names:
            raise ValueError(f"Mask {image_name} is not in the actual training split")
        record = records_by_name[image_name]
        mask_path = Path(item["mask_path"])
        if not mask_path.is_absolute():
            mask_path = path.parent / mask_path
        if not mask_path.exists():
            raise FileNotFoundError(mask_path)
        output.append({**item, "image_path": record["file"], "mask_path": str(mask_path)})
    return output


def run(args):
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("Original CUDA evaluation requires an available CUDA GPU and built rasterizer")
    upstream = Path(args.upstream or ROOT / "vendor/gaussian-splatting").resolve()
    sys.path.insert(0, str(upstream))
    from scene.gaussian_model import GaussianModel
    from gaussian_renderer import render
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    records, raw_points, provenance = load_colmap(args.scene_dir, upstream / "utils/read_write_model.py")
    del raw_points
    by_name = {Path(r["name"]).name: r for r in records}
    raw_split = json.loads(Path(args.split).read_text())
    split = raw_split.get("split", raw_split)
    train_names, test_names = {Path(n).name for n in split["train"]}, {Path(n).name for n in split["test"]}
    if train_names & test_names:
        raise ValueError("Actual RGB training and evaluation splits overlap")
    if (train_names | test_names) - set(by_name):
        raise ValueError("Split names missing from the dataset")
    annotations = load_annotations(args.annotations, args.scene_dir)
    if set(annotations) & train_names:
        raise ValueError("Annotated evaluation views were included in RGB training; retrain with these views held out")
    if set(annotations) - test_names:
        raise ValueError("All annotation views must be part of the declared held-out test split")
    masks = read_training_masks(args.training_masks, by_name, train_names)
    if getattr(args, "normalize_labels", False):
        from backend.semantic_targets import normalize_label
        masks = [{**m, "label": normalize_label(m["label"])} for m in masks]
    mask_names = {Path(m["image_path"]).name for m in masks}
    cameras = [{"image_index": i, "image_path": by_name[name]["file"], "width": by_name[name]["width"], "height": by_name[name]["height"],
                "intrinsics": by_name[name]["intrinsics"], "world_to_camera": by_name[name]["world_to_camera"]}
               for i, name in enumerate(sorted(mask_names))]
    start = time.perf_counter()
    model = GaussianModel(args.sh_degree)
    model.load_ply(args.ply)
    positions = model.get_xyz.detach().cpu().numpy()
    opacities = model.get_opacity.detach().cpu().numpy().reshape(-1)
    count = len(positions)
    print(f"Loaded all {count:,} CUDA Gaussians; no point sampling", flush=True)
    # Only geometry needed by app fusion; avoiding color/SH/scale dictionaries
    # reduces host RAM significantly. Full model remains in GaussianModel.
    semantic_scene = {"gaussians": [{"position": p.tolist(), "opacity": float(a)} for p, a in zip(positions, opacities)],
                      "cameras": cameras, "metadata": {"source": "complete upstream PLY"}}
    del positions, opacities
    fusion_start = time.perf_counter()
    semantic_scene = fuse_semantics(semantic_scene, config={"masks": masks})
    fusion_seconds = time.perf_counter()-fusion_start
    fusion_metadata = semantic_scene["metadata"]["semantics"]
    labels = sorted({m["label"].casefold().strip() for items in annotations.values() for m in items})
    labels_by_id = {o["id"]: o["label"].casefold().strip() for o in semantic_scene.get("objects", [])}
    # A part can belong both to itself and its parent object. Preserve that path.
    memberships = {label: np.array([any(labels_by_id.get(i) == label for i in g.get("semantic_ids", []))
                                    for g in semantic_scene["gaussians"]], dtype=bool) for label in labels}
    np.savez_compressed(output / "semantic_membership.npz", **{f"class_{i}": memberships[label] for i,label in enumerate(labels)})
    write_json(output / "semantic_classes.json", {"classes": labels, "gaussian_count": count, "order": "unaltered input PLY order"})
    del semantic_scene
    gc.collect()
    pipe = SimpleNamespace(compute_cov3D_python=False, convert_SHs_python=False, debug=False, antialiasing=args.antialiasing)
    background = torch.tensor([args.background]*3, dtype=torch.float32, device="cuda")
    black = torch.zeros(3, dtype=torch.float32, device="cuda")
    rgb, semantic, object_roi = [], [], []
    priority_roi, outside_priority_roi, priority_view_full_rgb = [], [], []
    priority_labels = {label.strip().casefold() for label in args.priority_labels.split(",") if label.strip()}
    for view_index, name in enumerate(sorted(test_names)):
        record = by_name[name]
        camera, target = camera_for_record(record, args.eval_size, torch)
        with torch.no_grad():
            image = render(camera, model, pipe, background, use_trained_exp=False)["render"].permute(1,2,0).clamp(0,1).cpu().numpy()
        rgb.append({"image": name, **rgb_metrics(image, target)})
        triptych = np.concatenate([target,image,np.abs(target-image)], axis=1)
        Image.fromarray((triptych*255).clip(0,255).astype(np.uint8)).save(output / f"{view_index:02d}-{Path(name).stem}-rgb.png")
        gt_labels = {}
        for item in annotations.get(name, []):
            label = item["label"].casefold().strip()
            truth = np.asarray(Image.open(item["mask_path"]).convert("L")) > 0
            truth = cv2.resize(truth.astype(np.uint8), (target.shape[1],target.shape[0]), interpolation=cv2.INTER_NEAREST).astype(bool)
            gt_labels[label] = truth | gt_labels.get(label, np.zeros(truth.shape, bool))
        for label, truth in gt_labels.items():
            object_roi.append({"image":name,"label":label, **rgb_metrics(image,target,truth)})
            # Nonmember splats are black but retain opacity, so they still occlude.
            colors = torch.from_numpy(np.repeat(memberships[label][:,None], 3, axis=1).astype(np.float32)).cuda()
            with torch.no_grad():
                projected = render(camera, model, pipe, black, override_color=colors, use_trained_exp=False)["render"][0].cpu().numpy()
            del colors
            prediction = projected >= .5
            semantic.append({"image":name,"label":label, **mask_metrics(prediction,truth)})
            safe_label = "".join(c if c.isalnum() else "_" for c in label)
            Image.fromarray(np.concatenate([truth,prediction],axis=1).astype(np.uint8)*255).save(output / f"{view_index:02d}-{safe_label}-masks.png")
        priority_truth = np.zeros(target.shape[:2], dtype=bool)
        for label, truth in gt_labels.items():
            if label in priority_labels:
                priority_truth |= truth
        if priority_truth.any():
            priority_roi.append({"image":name, **rgb_metrics(image,target,priority_truth)})
            outside_priority_roi.append({"image":name, **rgb_metrics(image,target,~priority_truth)})
            priority_view_full_rgb.append({"image":name, **rgb_metrics(image,target)})
        print(f"Evaluated {view_index+1}/{len(test_names)} held-out views", flush=True)
    by_label = {label:{"views":sum(r["label"]==label for r in semantic),
                      "mean_iou":mean_metric([r for r in semantic if r["label"]==label],"iou"),
                      "mean_boundary_iou":mean_metric([r for r in semantic if r["label"]==label],"boundary_iou")} for label in labels}
    groups = {"touching_image_border" if t else "inside_image":{
        "pairs":sum(r["gt_touches_image_border"]==t for r in semantic),
        "mean_iou":mean_metric([r for r in semantic if r["gt_touches_image_border"]==t],"iou"),
        "mean_boundary_iou":mean_metric([r for r in semantic if r["gt_touches_image_border"]==t],"boundary_iou")} for t in [False,True]}
    def region_summary(rows):
        return {"views":len(rows),"mean_psnr_db":mean_metric(rows,"psnr_db"),"mean_ssim":mean_metric(rows,"ssim"),"per_view":rows}
    result = {"backend":"upstream_cuda_3dgs", "ply":str(Path(args.ply).resolve()), "gaussian_count":count,"sampling":"none; all original PLY Gaussians",
        "device":torch.cuda.get_device_name(),"evaluation_long_edge":args.eval_size,"split":split,"initialization":provenance,
        "rgb":{"mean_psnr_db":mean_metric(rgb,"psnr_db"),"mean_ssim":mean_metric(rgb,"ssim"),"per_view":rgb},
        "rgb_regions":{"priority_labels":sorted(priority_labels),"priority_roi":region_summary(priority_roi),
                       "outside_priority_roi":region_summary(outside_priority_roi),"full_image_on_priority_views":region_summary(priority_view_full_rgb),
                       "per_label":{label:region_summary([r for r in object_roi if r["label"]==label]) for label in labels},
                       "per_object_view":object_roi,
                       "definition":"All ROIs use withheld manual GT only, RGB PSNR inside masked pixels; SSIM windows selected by center; outside_priority_roi is the complement on the same annotated views, and may contain other objects"},
        "semantics":{"mean_iou":mean_metric(semantic,"iou"),"class_mean_iou":mean_metric(list(by_label.values()),"mean_iou"),
                     "mean_boundary_iou":mean_metric(semantic,"boundary_iou"),"per_label":by_label,"image_border_groups":groups,"per_object_view":semantic,
                     "metric":"held-out 2D projected masks; not 3D IoU; full-scene alpha compositing; fixed threshold0.5; inner2pxboundary"},
        "fusion":fusion_metadata,"fusion_seconds":fusion_seconds,"total_evaluation_seconds":time.perf_counter()-start,
        "training_mask_count":len(masks),"training_mask_views":len(mask_names),
        "label_normalization_only":bool(getattr(args,"normalize_labels",False)),
        "code_sha256":{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(__file__),ROOT/"backend/semantics.py"]},
        "limitations":["Actual training split is supplied by the caller; this script validates lists but cannot prove training history.",
                       "Full-scene COLMAP geometry is transductive. All annotated views must still be excluded from RGB optimization.",
                       "Semantic masks are inherited from cached train-view GroundingDINO/SAM, no held-out image inference.",
                       "This does not test joint semantic-loss training or priority-weighted CUDA reconstruction.",
                       "App CPU semantic fusion can consume substantial RAM with millions of points; there is no automatic silent subsampling."]}
    write_json(output / "summary.json", result)
    print(json.dumps({"rgb_psnr":result["rgb"]["mean_psnr_db"],"semantic_miou":result["semantics"]["class_mean_iou"]}),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ply",required=True)
    parser.add_argument("--scene-dir",required=True)
    parser.add_argument("--split",required=True)
    parser.add_argument("--training-masks",required=True)
    parser.add_argument("--annotations",required=True)
    parser.add_argument("--output",required=True)
    parser.add_argument("--upstream")
    parser.add_argument("--sh-degree",type=int,default=3)
    parser.add_argument("--eval-size",type=int,default=256)
    parser.add_argument("--background",type=float,default=0)
    parser.add_argument("--priority-labels",default="stuffed bear")
    parser.add_argument("--antialiasing",action="store_true")
    parser.add_argument("--normalize-labels",action="store_true",help="Text-normalization-only ablation; original fusion remains unchanged")
    run(parser.parse_args())


if __name__=="__main__":
    main()
