#!/usr/bin/env python3
"""Re-render seven existing CUDA checkpoints at downloaded-base full resolution.

No training or checkpoint edits. Uses the completed baseline's frozen renderer
worker, separate output model directories and read-through point_cloud symlinks.
Bonsai images_4 is the downloaded base (780x520), not original sensor resolution.

python evaluate_cuda_full_resolution.py --baseline-root /content/benchmark-results \
    --priority-root /content/teatime-priority --output /content/full-resolution-results
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import sys
import time
from pathlib import Path

from PIL import Image


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024*1024), b""):
            value.update(block)
    return value.hexdigest()


def load_benchmark(path):
    spec = importlib.util.spec_from_file_location("full_resolution_benchmark", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def disjoint_output(output, roots):
    for source in roots:
        if output == source or source in output.parents or output in source.parents:
            raise ValueError("Full-resolution output must be separate from both input result trees")


def build_plan(baseline_root, priority_root, output, worker):
    """Validate completed jobs and exact downloaded base dimensions before GPU work."""
    cases = [(scene, mode, baseline_root/scene/mode) for scene in ["bonsai", "teatime"] for mode in ["fast", "balanced", "fine"]]
    cases.append(("teatime", "balanced-priority", priority_root/"balanced"))
    if not worker.is_file():
        raise FileNotFoundError(f"Frozen baseline worker missing: {worker}")
    worker_hash = digest(worker)
    plan = []
    expected_splits = {}
    for scene, mode, source in cases:
        job_file, result_file = source/"job.json", source/"result.json"
        if not job_file.is_file() or not result_file.is_file():
            raise FileNotFoundError(f"All seven models must have completed first: {source}")
        job = json.loads(job_file.read_text())
        completed = json.loads(result_file.read_text())
        if job["scene"] != scene:
            raise ValueError(f"Scene mismatch in {job_file}")
        split = job["split"]
        if set(split["train"]) & set(split["test"]):
            raise ValueError(f"Overlapping split in {job_file}")
        if split != completed["split"]:
            raise ValueError(f"Completed result and job split differ: {source}")
        if scene in expected_splits and split != expected_splits[scene]:
            raise ValueError(f"Mode test splits differ within {scene}")
        expected_splits[scene] = split
        names = completed["timing"]["test_image_names"]
        # The benchmark worker retains full dataset filenames, including suffixes.
        # Removing suffixes can also merge distinct images with the same stem.
        if len(names) != len(set(names)) or set(names) != set(split["test"]):
            raise ValueError(f"Saved test names do not match the declared split: {source}")
        image_dir = Path(job["scene_dir"]) / job["images"]
        sizes = set()
        for name in split["test"]:
            with Image.open(image_dir/name) as image:
                sizes.add(image.size)
        if len(sizes) != 1:
            raise ValueError(f"Downloaded test images have different native sizes: {sizes}; a single --eval_width cannot preserve every native size")
        width, height = next(iter(sizes))
        expected_size = (780,520) if scene == "bonsai" else (988,730)
        if (width,height) != expected_size:
            raise ValueError(f"Expected the recorded downloaded base {expected_size}, found {(width,height)}")
        cloud = source / "point_cloud" / f"iteration_{job['iterations']}" / "point_cloud.ply"
        if not cloud.is_file():
            raise FileNotFoundError(cloud)
        destination = output/scene/mode
        render_job = {**job,"model_path":str(destination),"eval_width":width,"action":"render"}
        plan.append({"scene":scene,"mode":mode,"source_model":str(source),"destination":str(destination),
                     "source_checkpoint":str(cloud),"source_checkpoint_sha256":digest(cloud),
                     "source_result_sha256":digest(result_file),"frozen_worker_sha256":worker_hash,
                     "source_job_sha256":digest(job_file),"downloaded_base_size":[width,height],
                     "test_image_names":names,"test_views":len(names),"render_job":render_job,
                     "previous_256_evaluation":completed["evaluation"]})
    return plan


def summarize(output, results, write_json):
    summary = {"experiment":"Same seven trained checkpoints rendered at full downloaded base resolution; no retraining",
               "interpretation":"Supplementary scale evaluation. It does not establish the cause of any non-monotonic mode ranking.",
               "base_resolution_note":"Bonsai uses images_4 at780x520, not the original3118x2078 camera image size. Teatime uses988x730 downloaded RGB.",
               "results":results}
    write_json(output/"summary.json",summary)
    fields = ["scene","mode","iterations","width","height","test_views","psnr_db","ssim","previous_256_psnr_db","previous_256_ssim","render_seconds"]
    with (output/"summary.csv").open("w",newline="") as stream:
        writer=csv.DictWriter(stream,fieldnames=fields)
        writer.writeheader()
        for row in results:
            writer.writerow({"scene":row["scene"],"mode":row["mode"],"iterations":row["iterations"],
                "width":row["downloaded_base_size"][0],"height":row["downloaded_base_size"][1],"test_views":row["evaluation"]["test_view_count"],
                "psnr_db":row["evaluation"]["mean_psnr_db"],"ssim":row["evaluation"]["mean_ssim"],
                "previous_256_psnr_db":row["previous_256_evaluation"]["mean_psnr_db"],"previous_256_ssim":row["previous_256_evaluation"]["mean_ssim"],
                "render_seconds":row["render_subprocess_wall_seconds"]})


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-root",type=Path,required=True)
    parser.add_argument("--priority-root",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--benchmark-runner",type=Path,default=Path(__file__).with_name("run_cuda_benchmark.py"))
    parser.add_argument("--dry-run",action="store_true")
    args=parser.parse_args()
    baseline,priority,output=args.baseline_root.resolve(),args.priority_root.resolve(),args.output.resolve()
    disjoint_output(output,[baseline,priority])
    benchmark=load_benchmark(args.benchmark_runner.resolve())
    worker=baseline/"_upstream_training_worker.py"
    plan=build_plan(baseline,priority,output,worker)
    output.mkdir(parents=True,exist_ok=True)
    benchmark.write_json(output/"plan.json",plan)
    if args.dry_run:
        print(json.dumps({"status":"validated_plan_only","models":len(plan),"output":str(output)}),flush=True)
        return
    frozen_worker=output/"_frozen_original_render_worker.py"
    if frozen_worker.exists() and digest(frozen_worker)!=digest(worker):
        raise ValueError("Output contains a different frozen worker; use a fresh output directory")
    frozen_worker.write_bytes(worker.read_bytes())
    results=[]
    gt_hashes_by_scene={}
    for item in plan:
        destination=Path(item["destination"])
        destination.mkdir(parents=True,exist_ok=True)
        completed_path=destination/"result.json"
        if completed_path.exists():
            result=json.loads(completed_path.read_text())
            if any(result.get(key)!=item[key] for key in ["source_checkpoint_sha256","source_job_sha256","frozen_worker_sha256","downloaded_base_size"]):
                raise ValueError(f"Existing full-resolution result came from different inputs: {destination}")
        else:
            source_model=Path(item["source_model"])
            target=destination/"point_cloud"
            if target.exists() or target.is_symlink():
                if not target.is_symlink() or target.resolve()!=(source_model/"point_cloud").resolve():
                    raise ValueError(f"Unexpected existing point-cloud destination: {target}")
            else:
                target.symlink_to(source_model/"point_cloud",target_is_directory=True)
            job_path=destination/"render_job.json"
            benchmark.write_json(job_path,item["render_job"])
            source_checkpoint=Path(item["source_checkpoint"])
            before=source_checkpoint.stat()
            seconds=benchmark.run_logged([sys.executable,frozen_worker,job_path],Path(item["render_job"]["upstream"]),destination/"render.log")
            after=source_checkpoint.stat()
            if (before.st_size,before.st_mtime_ns)!=(after.st_size,after.st_mtime_ns) or digest(source_checkpoint)!=item["source_checkpoint_sha256"]:
                raise RuntimeError("Checkpoint changed during read-only rendering")
            evaluation=benchmark.evaluate_pngs(destination,item["render_job"]["iterations"],item["test_image_names"])
            if any((r["width"],r["height"])!=tuple(item["downloaded_base_size"]) for r in evaluation["per_view"]):
                raise ValueError("Actual renderer output differs from downloaded-base full resolution")
            hashes={r["image_name"]:digest(Path(r["ground_truth"])) for r in evaluation["per_view"]}
            result={k:v for k,v in item.items() if k not in {"render_job","test_image_names"}}
            result.update(iterations=item["render_job"]["iterations"],split=item["render_job"]["split"],evaluation=evaluation,
                          render_subprocess_wall_seconds=seconds,ground_truth_png_sha256=hashes,
                          checkpoint_unchanged=True,training_performed=False,completed_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ",time.gmtime()))
            benchmark.write_json(completed_path,result)
        previous=gt_hashes_by_scene.setdefault(item["scene"],result["ground_truth_png_sha256"])
        if previous!=result["ground_truth_png_sha256"]:
            raise ValueError("Full-resolution ground truths differ across modes within the same scene")
        results.append(result)
        summarize(output,results,benchmark.write_json)
        print(json.dumps({"scene":item["scene"],"mode":item["mode"],"width":result["downloaded_base_size"][0],
                          "psnr_db":result["evaluation"]["mean_psnr_db"],"ssim":result["evaluation"]["mean_ssim"]}),flush=True)
    print(f"Finished seven models without retraining: {output/'summary.csv'}",flush=True)


if __name__=="__main__":
    main()
