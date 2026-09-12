"""Run the full unposed-image pipeline on an explicit synthetic stereo fixture."""
from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from backend.reconstruction import reconstruct


def make_photogrammetry_fixture(directory: Path):
    directory.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(21)
    tiles = []
    for x in np.linspace(-1.25, 1.25, 6):
        for y in np.linspace(-.8, .8, 4):
            texture = cv2.GaussianBlur(rng.integers(0, 255, (50, 50, 3), dtype=np.uint8), (3, 3), .5)
            tiles.append((x, y, rng.uniform(3.4, 5.5), texture))
    paths = []
    for image_index, tx in enumerate([0, -.45]):
        image = np.zeros((320, 420, 3), dtype=np.uint8)
        for x, y, z, texture in sorted(tiles, key=lambda tile: -tile[2]):
            left, right = round(378*(x+tx-.18)/z+210), round(378*(x+tx+.18)/z+210)
            top, bottom = round(378*(y-.18)/z+160), round(378*(y+.18)/z+160)
            if 0 <= left < right < 420 and 0 <= top < bottom < 320:
                image[top:bottom, left:right] = cv2.resize(texture, (right-left, bottom-top))
        path = directory / f"synthetic-stereo-{image_index}.png"
        Image.fromarray(image).save(path); paths.append(str(path))
    return paths


def main():
    import torch
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="mps", choices=["mps", "cpu", "cuda", "auto"])
    parser.add_argument("--work-dir", type=Path, default=REPO.parents[1] / "work" / "smoke")
    parser.add_argument("--report", type=Path, default=REPO / "docs" / "reconstruction-e2e.json")
    args = parser.parse_args()
    paths = make_photogrammetry_fixture(args.work_dir / "images")
    configuration = {"device": args.device, "mode": "fast", "focal_ratio": .9, "background": [0, 0, 0], "completion": "none"}
    report = {"fixture": "24 random textured front-facing tiles at varying depths, two 420×320 images", "is_real_dataset": False, "camera_poses_supplied": False, "ground_truth_geometry_supplied": False, "focal_prior_matches_generator": True, "purpose": "SIFT→essential matrix→pose recovery→triangulation→Gaussian training→JSON/PLY export", "platform": platform.platform(), "torch_version": torch.__version__, "mps_available": torch.backends.mps.is_available(), "config": configuration}
    started = time.perf_counter()
    try:
        result = reconstruct(paths, str(args.work_dir / "result"), configuration, progress=lambda fraction, message: print(f"{fraction:.2f} {message}", flush=True))
        report.update({"status": "passed", "wall_seconds": time.perf_counter()-started, "metadata": result["metadata"], "outputs": {"scene_path": result["scene_path"], "ply_path": result["ply_path"]}, "checks": {"scene_exists": Path(result["scene_path"]).is_file(), "ply_exists": Path(result["ply_path"]).is_file(), "finite_geometry": all(np.isfinite(g["position"]).all() for g in result["scene"]["gaussians"]), "registered_two_views": result["metadata"]["registered_views"] == 2, "loss_decreased": result["metadata"]["final_l1"] < result["metadata"]["initial_l1"]}, "limitations": ["Synthetic textured input; not evidence of quality on real photographs.", "Unseen surfaces remain unknown. No learned completion was attempted.", "This is the small PyTorch reference pipeline, not the upstream CUDA rasterizer."]})
        if not all(report["checks"].values()):
            report["status"] = "failed"
    except Exception as exc:
        report.update({"status": "failed", "wall_seconds": time.perf_counter()-started, "error": str(exc)})
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps({k: report[k] for k in ("status", "wall_seconds")}, ensure_ascii=False))
    if report["status"] != "passed":
        raise SystemExit(report.get("error", "End-to-end checks failed"))


if __name__ == "__main__":
    main()
