#!/usr/bin/env python3
"""Run the pinned original Graphdeco CUDA trainer, with independent held-out metrics.

Standalone uploadable script; dependencies: upstream CUDA extensions, torch,
torchvision, Pillow, numpy, plyfile, tqdm. It does not import Splat Studio.
The upstream source tree stays unchanged. An instrumentation-only worker imports
its original training() function and records scene loading, iteration and save time.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import re
import subprocess
import struct
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

UPSTREAM_COMMIT = "54c035f7834b564019656c3e3fcc3646292f727d"
PRESETS = {
    "fast": {"iterations": 7000, "resolution": 4},
    "balanced": {"iterations": 15000, "resolution": 2},
    "fine": {"iterations": 22000, "resolution": 1},
}

WORKER = r'''
import argparse, json, os, sys, time
from pathlib import Path

job = json.loads(Path(sys.argv[1]).read_text())
sys.path.insert(0, job["upstream"])
os.chdir(job["upstream"])
import torch
import train
from scene.dataset_readers import sceneLoadTypeCallbacks, getNerfppNorm

# A data-loader callback controls only the recorded train/test split. The
# original optimization and renderer remain untouched. Both subprocess phases
# use this callback, including all six teatime annotation views in the test set.
original_colmap = sceneLoadTypeCallbacks["Colmap"]
test_names = set(job["split"]["test"])
train_names = set(job["split"]["train"])
assert not test_names & train_names
def split_colmap(path, images, depths, eval, train_test_exp):
    info = original_colmap(path, images, depths, False, False)
    cameras = info.train_cameras + info.test_cameras
    found = {Path(c.image_path).name for c in cameras}
    if found != train_names | test_names:
        raise RuntimeError("Recorded split does not cover exactly the registered scene cameras")
    cameras = [c._replace(is_test=Path(c.image_path).name in test_names) for c in cameras]
    training = [c for c in cameras if not c.is_test]
    testing = [c for c in cameras if c.is_test]
    return info._replace(train_cameras=training, test_cameras=testing, nerf_normalization=getNerfppNorm(training))
sceneLoadTypeCallbacks["Colmap"] = split_colmap

if not torch.cuda.is_available():
    raise RuntimeError("This benchmark requires a connected NVIDIA CUDA GPU")
torch.cuda.set_device(0)
torch.set_num_threads(4)
torch.cuda.reset_peak_memory_stats()
output = Path(job["model_path"])
output.mkdir(parents=True, exist_ok=True)
timing = {"scene_initialization_wall_seconds": 0., "checkpoint_save_wall_seconds": 0.}
start = time.perf_counter()
last_wall, last_iter = None, 0
event_ms = 0.
OriginalScene = train.Scene
original_report = train.training_report

class MeasuredScene(OriginalScene):
    def __init__(self, *args, **kwargs):
        stamp = time.perf_counter()
        super().__init__(*args, **kwargs)
        torch.cuda.synchronize()
        timing["scene_initialization_wall_seconds"] += time.perf_counter() - stamp
        timing["train_view_count"] = len(self.getTrainCameras())
        timing["test_view_count"] = len(self.getTestCameras())
        timing["train_image_sizes"] = sorted({(c.image_width,c.image_height) for c in self.getTrainCameras()})
        timing["train_image_names"] = sorted(c.image_name for c in self.getTrainCameras())
        timing["test_image_names"] = sorted(c.image_name for c in self.getTestCameras())
        assert not set(timing["train_image_names"]) & set(timing["test_image_names"])
    def save(self, iteration):
        stamp = time.perf_counter()
        super().save(iteration)
        torch.cuda.synchronize()
        timing["checkpoint_save_wall_seconds"] += time.perf_counter() - stamp
        timing["gaussian_count"] = int(self.gaussians.get_xyz.shape[0])

def measured_report(tb, iteration, Ll1, loss, l1_loss, elapsed, tests, scene, render_func, render_args, train_test_exp):
    global event_ms, last_wall, last_iter
    event_ms += float(elapsed)
    original_report(tb, iteration, Ll1, loss, l1_loss, elapsed, [], scene, render_func, render_args, train_test_exp)
    if iteration == 1 or iteration % 250 == 0 or iteration == job["iterations"]:
        now = time.perf_counter()
        segment_seconds = None if last_wall is None else now - last_wall
        speed = None if segment_seconds is None else (iteration-last_iter) / max(segment_seconds, 1e-12)
        row = {"iteration": iteration, "training_function_elapsed_seconds": now-start,
            "segment_wall_seconds": segment_seconds, "segment_iterations_per_second": speed,
            "gaussians": int(scene.gaussians.get_xyz.shape[0]), "loss": float(loss.detach()),
            "peak_allocated_vram_gib": torch.cuda.max_memory_allocated()/1024**3,
            "remaining_seconds_estimate_from_last_segment": None if speed is None else (job["iterations"]-iteration)/speed}
        with (output/"progress.jsonl").open("a") as stream:
            stream.write(json.dumps(row)+"\n")
        print("BENCHMARK_PROGRESS "+json.dumps(row), flush=True)
        last_wall, last_iter = now, iteration

train.Scene = MeasuredScene
train.training_report = measured_report
parser = argparse.ArgumentParser()
lp, op, pp = train.ModelParams(parser), train.OptimizationParams(parser), train.PipelineParams(parser)
cli = ["--source_path",job["scene_dir"],"--model_path",job["model_path"],
    "--images",job["images"],"--resolution",str(job["resolution"]),"--iterations",str(job["iterations"]),
    "--data_device",job["data_device"],"--eval"]
args = parser.parse_args(cli)
train.safe_state(False)
if job.get("action") == "render":
    import render as render_module
    args.resolution = job["eval_width"]
    render_module.args = argparse.Namespace(train_test_exp=args.train_test_exp)
    render_module.render_sets(lp.extract(args),job["iterations"],pp.extract(args),True,False,render_module.SPARSE_ADAM_AVAILABLE)
    sys.exit(0)
torch.cuda.synchronize()
start = time.perf_counter()
train.training(lp.extract(args), op.extract(args), pp.extract(args), [], [job["iterations"]], [], None, -1)
torch.cuda.synchronize()
timing["training_function_wall_seconds"] = time.perf_counter()-start
timing["training_loop_wall_seconds"] = timing["training_function_wall_seconds"]-timing["scene_initialization_wall_seconds"]-timing["checkpoint_save_wall_seconds"]
timing["gpu_render_loss_backward_event_seconds"] = event_ms/1000
timing["gpu_event_scope"] = "Original upstream CUDA events bracket rendering/loss/backward only; excludes optimizer, densification, camera loading, checkpoint saving. Use wall-clock fields for user-facing time."
timing["peak_allocated_vram_gib"] = torch.cuda.max_memory_allocated()/1024**3
timing["peak_reserved_vram_gib"] = torch.cuda.max_memory_reserved()/1024**3
props = torch.cuda.get_device_properties(0)
timing["hardware"] = {"gpu": torch.cuda.get_device_name(0),"total_vram_gib":props.total_memory/1024**3,
    "compute_capability":list(torch.cuda.get_device_capability(0)),"torch":torch.__version__,"cuda":torch.version.cuda}
timing["optimizer_type"] = args.optimizer_type
timing["fused_ssim_available"] = train.FUSED_SSIM_AVAILABLE
timing["sparse_adam_extension_available"] = train.SPARSE_ADAM_AVAILABLE
timing["optimization_parameters"] = vars(op.extract(args))
(output/"training_timing.json").write_text(json.dumps(timing,indent=2))
print("BENCHMARK_TRAINING_COMPLETE "+json.dumps(timing),flush=True)
'''


def write_json(path, record):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(record, indent=2, ensure_ascii=False, allow_nan=False))
    temporary.replace(path)


def run_logged(command, cwd, log_path):
    """Stream original diagnostics and preserve them, including failed builds/runs."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    print("COMMAND:", " ".join(str(x) for x in command), flush=True)
    stamp = time.perf_counter()
    with log_path.open("w") as log:
        process = subprocess.Popen([str(x) for x in command], cwd=cwd,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
            env={**os.environ, "PYTHONUNBUFFERED":"1", "OMP_NUM_THREADS":"4"}, bufsize=1)
        for line in process.stdout:
            log.write(line)
            log.flush()
            print(line, end="", flush=True)
        code = process.wait()
    if code:
        raise RuntimeError(f"Command exited {code}; see {log_path}")
    return time.perf_counter() - stamp


def check_upstream(path):
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=path, text=True).strip()
    if commit != UPSTREAM_COMMIT:
        raise ValueError(f"Expected upstream {UPSTREAM_COMMIT}, got {commit}")
    for command in (["git", "diff", "--ignore-submodules=untracked", "--exit-code"], ["git", "diff", "--cached", "--ignore-submodules=untracked", "--exit-code"]):
        if subprocess.run(command, cwd=path, stdout=subprocess.DEVNULL).returncode:
            raise ValueError("The upstream tracked source has edits; use a clean pinned checkout")
    for relative in ("submodules/simple-knn","submodules/diff-gaussian-rasterization","submodules/fused-ssim"):
        pinned=subprocess.check_output(["git","ls-tree","HEAD",relative],cwd=path,text=True).split()[2]
        actual=subprocess.check_output(["git","rev-parse","HEAD"],cwd=path/relative,text=True).strip()
        if actual!=pinned:
            raise ValueError(f"Submodule {relative} is at {actual}, expected {pinned}")
        for extra in ([],["--cached"]):
            command=["git","diff",*extra,"--ignore-submodules=untracked","--exit-code"]
            if subprocess.run(command,cwd=path/relative,stdout=subprocess.DEVNULL).returncode:
                raise ValueError(f"Tracked submodule source has edits: {relative}")
    submodules = subprocess.check_output(["git","submodule","status"],cwd=path,text=True)
    return {"repository":"https://github.com/graphdeco-inria/gaussian-splatting", "commit":commit, "submodules":submodules}


def scene_path(root, scene):
    candidates = [root/scene, root/"mipnerf360"/scene, root/"lerf-ovs"/"lerf_ovs"/scene,
        root/"lerf_ovs"/scene]
    for candidate in candidates:
        if (candidate/"sparse"/"0"/"images.bin").exists():
            return candidate.resolve()
    raise FileNotFoundError(f"No COLMAP scene {scene} in {root}")


def scene_metadata(path):
    # Prefer the downloaded quarter-resolution bonsai source consistently across
    # all modes, even when a later download adds higher-resolution images.
    images = "images_4" if (path/"images_4").exists() else "images"
    photos = sorted(p for p in (path/images).iterdir() if p.suffix.lower() in {".jpg",".jpeg",".png"})
    if not photos:
        raise ValueError(f"No RGB photographs in {path/images}")
    dimensions = sorted({Image.open(p).size for p in photos})
    signature = hashlib.sha256()
    for p in photos + [path/"sparse"/"0"/n for n in ("cameras.bin","images.bin","points3D.bin")]:
        signature.update(p.relative_to(path).as_posix().encode())
        signature.update(hashlib.sha256(p.read_bytes()).digest())
    return {"source_path":str(path),"images":images,"image_count":len(photos),
        "base_image_sizes":dimensions,"source_signature_sha256":signature.hexdigest(),
        "photo_test_policy":"Original upstream --eval: every 8th COLMAP registered image after sorting names; remaining images train. Same split for all modes.",
        "initial_geometry":"Author-provided full-scene COLMAP poses and original sparse-point RGB. These may use held-out photos: transductive SfM benchmark, not end-to-end sparse reconstruction.",
        "base_resolution_note":"For bonsai images_4, fast/balanced/fine use approximately 1/16, 1/8, 1/4 of original image width and height. This is not original-photo full-resolution paper evaluation." if images == "images_4" else "Mode resolution divisor is applied to downloaded RGB base dimensions."}


def registered_names(path):
    names=[]
    with (path/"sparse"/"0"/"images.bin").open("rb") as stream:
        count=struct.unpack("<Q",stream.read(8))[0]
        for _ in range(count):
            stream.read(64)  # image id, quaternion, translation, camera id
            name=bytearray()
            while True:
                char=stream.read(1)
                if not char:
                    raise ValueError("Truncated COLMAP image name")
                if char==b"\0":
                    break
                name.extend(char)
            names.append(name.decode("utf-8"))
            points=struct.unpack("<Q",stream.read(8))[0]
            stream.seek(24*points,1)
    return sorted(names)


def scene_split(path,scene):
    names=registered_names(path)
    test={name for i,name in enumerate(names) if i%8==0}
    if scene=="teatime":
        annotation_dir=path.parent/"label"/scene
        annotations=sorted(annotation_dir.glob("*.json"))
        if not annotations:
            raise FileNotFoundError(f"Teatime requires author GT JSON to prevent held-out leakage: {annotation_dir}")
        annotated={json.loads(p.read_text())["info"]["name"] for p in annotations}
        if not annotated <= set(names):
            raise ValueError("Some annotated views are not registered in COLMAP")
        test |= annotated
    return {"train":[n for n in names if n not in test],"test":[n for n in names if n in test]}


def rgb_metrics(prediction, reference):
    """11x11, sigma=1.5, valid-window SSIM; mean per-view PSNR in RGB [0,1]."""
    x, y = np.asarray(prediction, dtype=np.float64), np.asarray(reference, dtype=np.float64)
    if x.shape != y.shape or x.ndim != 3 or x.shape[-1] != 3:
        raise ValueError("Metrics require matching HxWx3 images")
    mse = float(((x-y)**2).mean())
    psnr = -10*math.log10(max(mse,1e-12))
    size = min(11, min(x.shape[:2]))
    size -= (size % 2 == 0)
    if size < 3:
        raise ValueError("Evaluation images are too small for meaningful SSIM")
    coords = np.arange(size)-size//2
    kernel = np.exp(-(coords**2)/(2*1.5**2)); kernel /= kernel.sum()
    def blur(a):
        vertical = sum(kernel[i]*a[i:a.shape[0]-size+1+i,:,:] for i in range(size))
        return sum(kernel[i]*vertical[:,i:vertical.shape[1]-size+1+i,:] for i in range(size))
    mx,my = blur(x),blur(y)
    vx,vy,cov = blur(x*x)-mx*mx,blur(y*y)-my*my,blur(x*y)-mx*my
    c1,c2 = .01**2,.03**2
    ssim = float((((2*mx*my+c1)*(2*cov+c2))/((mx*mx+my*my+c1)*(vx+vy+c2))).mean())
    return {"psnr_db":psnr,"ssim":ssim,"mse":mse}


def evaluate_pngs(model, iterations, names):
    root = model/"test"/f"ours_{iterations}"
    renders = sorted((root/"renders").glob("*.png"))
    references = sorted((root/"gt").glob("*.png"))
    if len(renders) != len(names) or len(references) != len(names):
        raise RuntimeError(f"Render count {len(renders)} != expected held-out views {len(names)}")
    results = []
    for i, (prediction_path, reference_path) in enumerate(zip(renders, references)):
        pred=np.asarray(Image.open(prediction_path).convert("RGB"))/255.
        gt=np.asarray(Image.open(reference_path).convert("RGB"))/255.
        results.append({"image_name":names[i],"render":str(prediction_path.resolve()),
            "ground_truth":str(reference_path.resolve()),"width":int(gt.shape[1]),"height":int(gt.shape[0]),**rgb_metrics(pred,gt)})
    return {"mean_psnr_db":float(np.mean([x["psnr_db"] for x in results])),
        "mean_ssim":float(np.mean([x["ssim"] for x in results])),"test_view_count":len(results),
        "metric_definition":"Mean of per-view RGB PSNR and Gaussian-window(11,sigma1.5,valid) SSIM computed on saved 8-bit PNGs. No LPIPS or geometric ground-truth metric.",
        "per_view":results}


def refresh_summary(output):
    rows=[]
    for result_path in sorted(output.glob("*/*/result.json")):
        result=json.loads(result_path.read_text())
        rows.append(result)
    write_json(output/"summary.json", {"results":rows,"upstream_commit":UPSTREAM_COMMIT,
        "time_scope":"Actual training loop, scene initialization and final save reported separately; no SfM, download or extension compilation is included in training timings."})
    fields=["scene","mode","gpu","iterations","training_resolution","eval_width","train_views","test_views","gaussians","training_loop_seconds","scene_initialization_seconds","checkpoint_save_seconds","subprocess_wall_seconds","psnr_db","ssim"]
    with (output/"summary.csv").open("w",newline="") as stream:
        writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader()
        for r in rows:
            t,e=r["timing"],r["evaluation"]
            writer.writerow({"scene":r["scene"],"mode":r["mode"],"gpu":t["hardware"]["gpu"],
                "iterations":r["iterations"],"training_resolution":str(t["train_image_sizes"]),"eval_width":r["eval_width"],
                "train_views":t["train_view_count"],"test_views":t["test_view_count"],"gaussians":t["gaussian_count"],
                "training_loop_seconds":t["training_loop_wall_seconds"],"scene_initialization_seconds":t["scene_initialization_wall_seconds"],
                "checkpoint_save_seconds":t["checkpoint_save_wall_seconds"],"subprocess_wall_seconds":t.get("training_subprocess_wall_seconds"),
                "psnr_db":e["mean_psnr_db"],"ssim":e["mean_ssim"]})
    for scene in sorted({r["scene"] for r in rows}):
        records=sorted((r for r in rows if r["scene"]==scene),key=lambda r:list(PRESETS).index(r["mode"]))
        base=records[0]["evaluation"]["per_view"]
        ids=np.linspace(0,len(base)-1,min(3,len(base))).round().astype(int).tolist()
        thumb_w=256;thumb_h=max(round(Image.open(base[i]["ground_truth"]).height*thumb_w/Image.open(base[i]["ground_truth"]).width) for i in ids)
        header=58;row_h=thumb_h+26
        canvas=Image.new("RGB",(thumb_w*(1+len(records)),header+row_h*len(ids)),"#151923")
        draw=ImageDraw.Draw(canvas);draw.text((8,8),f"{scene} | held-out GT",fill="white")
        for col, record in enumerate(records,1):
            draw.text((col*thumb_w+8,8),f"{record['mode']} {record['iterations']} iter",fill="white")
            draw.text((col*thumb_w+8,26),f"PSNR {record['evaluation']['mean_psnr_db']:.2f} SSIM {record['evaluation']['mean_ssim']:.3f}",fill="#d2daee")
            draw.text((col*thumb_w+8,42),f"Loop {record['timing']['training_loop_wall_seconds']/60:.2f} min",fill="#d2daee")
        for row,i in enumerate(ids):
            for col,record in enumerate([None]+records):
                item=base[i] if record is None else record["evaluation"]["per_view"][i]
                if item["image_name"] != base[i]["image_name"]:
                    raise ValueError("Mode comparisons use different test view ordering")
                image=Image.open(item["ground_truth"] if record is None else item["render"]).convert("RGB")
                image.thumbnail((thumb_w,thumb_h),Image.Resampling.LANCZOS)
                canvas.paste(image,(col*thumb_w,header+row*row_h))
                draw.text((col*thumb_w+6,header+row*row_h+thumb_h+5),item["image_name"] if record is None else f"PSNR {item['psnr_db']:.2f} / SSIM {item['ssim']:.3f}",fill="white")
        canvas.save(output/f"{scene}-comparison.png")


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upstream",type=Path,required=True)
    parser.add_argument("--data-root",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--scenes",nargs="+",default=["bonsai"])
    parser.add_argument("--modes",nargs="+",default=["fast","balanced","fine"])
    parser.add_argument("--eval-width",type=int,choices=[128,256],default=256)
    parser.add_argument("--data-device",choices=["cpu","cuda"],default="cpu")
    parser.add_argument("--dry-run",action="store_true",help="Validate data and pinned source, write plans, but do not import CUDA or start training")
    args=parser.parse_args()
    args.upstream=args.upstream.resolve();args.data_root=args.data_root.resolve();args.output=args.output.resolve()
    args.output.mkdir(parents=True,exist_ok=True)
    modes=[x for token in args.modes for x in re.split(r"[,/]",token) if x]
    scenes=[x for token in args.scenes for x in re.split(r"[,/]",token) if x]
    if not modes or any(m not in PRESETS for m in modes) or len(set(modes))!=len(modes):
        parser.error("--modes must be distinct values from fast, balanced, fine")
    if not scenes or any(s not in {"bonsai","teatime"} for s in scenes):
        parser.error("--scenes accepts bonsai and/or teatime")
    modes=sorted(modes,key=lambda m:list(PRESETS).index(m))
    upstream=check_upstream(args.upstream)
    worker_path=args.output/"_upstream_training_worker.py";worker_path.write_text(WORKER)
    for scene in scenes:
        path=scene_path(args.data_root,scene);metadata=scene_metadata(path)
        split=scene_split(path,scene)
        write_json(args.output/scene/"split.json",split)
        if scene=="teatime":
            metadata["photo_test_policy"]="Every 8th sorted COLMAP registered image UNION all downloaded author-annotated teatime frames. These views are excluded from both RGB and semantic training."
        for mode in modes:
            preset=PRESETS[mode];model=args.output/scene/mode;model.mkdir(parents=True,exist_ok=True)
            job={"upstream":str(args.upstream),"model_path":str(model),"scene_dir":str(path),
                "scene":scene,"mode":mode,"images":metadata["images"],"data_device":args.data_device,
                "eval_width":args.eval_width,"source_signature_sha256":metadata["source_signature_sha256"],"split":split,**preset}
            job_path=model/"job.json"
            if job_path.exists() and json.loads(job_path.read_text()) != job:
                raise ValueError(f"Existing job configuration differs: {model}; use a new output directory")
            write_json(job_path,job);write_json(model/"scene_metadata.json",metadata)
            print("BENCHMARK_PLAN "+json.dumps({k:v for k,v in job.items() if k!="split"})+f" train={len(split['train'])} test={len(split['test'])}",flush=True)
            if args.dry_run:
                continue
            if (model/"result.json").exists():
                print(f"Already completed: {scene}/{mode}",flush=True);continue
            try:
                timing_path=model/"training_timing.json"
                if timing_path.exists():
                    timing=json.loads(timing_path.read_text())
                else:
                    # Starting an interrupted job from scratch must not mix old
                    # iteration records into this run's speed measurements.
                    (model/"progress.jsonl").unlink(missing_ok=True)
                    duration=run_logged([sys.executable,worker_path,job_path],args.upstream,model/"train.log")
                    timing=json.loads(timing_path.read_text());timing["training_subprocess_wall_seconds"]=duration
                    write_json(timing_path,timing)
                render_job_path=model/"render_job.json"
                write_json(render_job_path,{**job,"action":"render"})
                render_command=[sys.executable,worker_path,render_job_path]
                render_seconds=run_logged(render_command,args.upstream,model/"render.log")
                metrics=evaluate_pngs(model,preset["iterations"],timing["test_image_names"])
                result={"scene":scene,"mode":mode,"iterations":preset["iterations"],"resolution_divisor":preset["resolution"],
                    "eval_width":args.eval_width,"upstream":upstream,"dataset":metadata,"timing":timing,
                    "evaluation":metrics,"render_subprocess_wall_seconds":render_seconds,"split":split,
                    "scope":"Original pinned Graphdeco 3DGS CUDA trainer and default optimizer; preset changes iteration count and input resolution only. No semantic or priority loss. Full-scene precomputed SfM initialization.",
                    "completed_utc":time.strftime("%Y-%m-%dT%H:%M:%SZ",time.gmtime())}
                write_json(model/"result.json",result);refresh_summary(args.output)
                print("BENCHMARK_RESULT "+json.dumps({"scene":scene,"mode":mode,"psnr_db":metrics["mean_psnr_db"],"ssim":metrics["mean_ssim"],"training_loop_seconds":timing["training_loop_wall_seconds"]}),flush=True)
            except Exception as exc:
                write_json(model/"failure.json",{"error":str(exc),"utc":time.strftime("%Y-%m-%dT%H:%M:%SZ",time.gmtime()),"job":job})
                raise
    if not args.dry_run:
        refresh_summary(args.output)
        print(f"Completed results: {args.output/'summary.csv'}",flush=True)


if __name__=="__main__":
    main()
