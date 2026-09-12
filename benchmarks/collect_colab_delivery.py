#!/usr/bin/env python3
"""Collect completed Colab research results into a bounded, verifiable delivery.

CPU/file processing only. Explicit output-tree allowlist; never walks datasets,
HF caches, credentials, or full point-cloud directories. Original archives and
all source results are preserved. No metric, semantic array, or preview is sampled.
An explicit --include-bonsai-full-model adds exactly the untouched fine PLY.

python collect_colab_delivery.py --root /content
python collect_colab_delivery.py --root /path/to/local/results --validate-only
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import os
import re
import subprocess
import sys
import tempfile
import time
import zipfile
from pathlib import Path

import numpy as np

MODES = {"fast":7000,"balanced":15000,"fine":22000}
SEMANTIC_MODES = ["fast","balanced","fine","balanced-priority"]
TEXT_SUFFIXES = {".json",".jsonl",".csv",".py",".patch",".log",".md",".txt"}
EXCLUDED_DIR_NAMES = {"point_cloud","__pycache__",".cache",".git","data","datasets","model-cache","hf-cache","huggingface","credentials"}
EXCLUDED_FILE_NAMES = {".env","credentials.json","token.json","tokens.json","application_default_credentials.json","cookies.json","cookies.txt"}
TREES = ["benchmark-results","teatime-priority","semantic-results","full-resolution-results","render-scale-analysis","previews"]
TOP_FILES = ["benchmark-finalization-status.json","extended-evaluation-status.json","cuda-discovery-latency.json",
    "run_a100_experiment.py","run_cuda_benchmark.py","finish_colab_benchmark.py","finish_colab_extended.py",
    "evaluate_cuda_full_resolution.py","analyze_render_scale.py","export_benchmark_viewer.py","benchmark_cuda_discovery.py",
    "collect_colab_delivery.py","a100-experiment.log","priority-experiment.log","cuda-build-verbose.log",
    "data-download.log","teatime-download.log","cuda-discovery.log","full-resolution-evaluation.log",
    "render-scale-analysis.log","discovery-dependencies.log","bonsai-preview-export.log","teatime-preview-export.log"]
TASK_WORKERS = {"_upstream_training_worker.py","_priority_training_worker.py","_frozen_original_render_worker.py",
    "evaluate_cuda_semantics.py","benchmark_cuda_discovery.py","run_cuda_priority_ablation.py",
    "run_cuda_benchmark.py","run_a100_experiment.py","evaluate_cuda_full_resolution.py"}
SECRET_PATTERNS = [re.compile(rb"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
    re.compile(rb"(?<![A-Za-z0-9])hf_[A-Za-z0-9]{25,}"),
    re.compile(rb"(?<![A-Za-z0-9])sk-(?:proj-)?[A-Za-z0-9_-]{25,}"),
    re.compile(rb"ya29\.[A-Za-z0-9_-]{30,}")]


def sha256(path):
    digest=hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda:stream.read(1024*1024),b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path):
    if not path.is_file():
        raise FileNotFoundError(f"Required completed artifact missing: {path}")
    value=json.loads(path.read_text())
    if not isinstance(value,dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def write_json(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+"\n")


def finite_metric(value,name):
    if not isinstance(value,(int,float)) or not math.isfinite(value):
        raise ValueError(f"Missing/nonfinite metric: {name}")


def check_rgb_evaluation(record,expected_views,label):
    evaluation=record["evaluation"]
    if evaluation["test_view_count"]!=expected_views or len(evaluation["per_view"])!=expected_views:
        raise ValueError(f"Incomplete view metrics: {label}")
    finite_metric(evaluation["mean_psnr_db"],label+" PSNR")
    finite_metric(evaluation["mean_ssim"],label+" SSIM")
    return evaluation


def validate_completed(root):
    """Validate result evidence, never substitute dry-run plans for completed work."""
    for filename in ["benchmark-finalization-status.json","extended-evaluation-status.json"]:
        if read_json(root/filename).get("phase")!="complete":
            raise RuntimeError(f"GPU/extended pipeline has not completed: {filename}")
    baseline_summary=read_json(root/"benchmark-results/summary.json")
    if {(r["scene"],r["mode"]) for r in baseline_summary.get("results",[])}!={(s,m) for s in ["bonsai","teatime"] for m in MODES}:
        raise ValueError("Baseline summary does not contain exactly the six completed scene/mode combinations")
    if not (root/"benchmark-results/summary.csv").is_file():
        raise FileNotFoundError("Baseline summary.csv missing")
    checked=[]
    checkpoint_evidence=[]
    gaussian_counts={}
    required_visuals=[root/"benchmark-results/bonsai-comparison.png",root/"benchmark-results/teatime-comparison.png",root/"teatime-priority/baseline-vs-priority.png"]
    for path in required_visuals:
        if not path.is_file():
            raise FileNotFoundError(f"Required mode comparison missing: {path}")
    for scene in ["bonsai","teatime"]:
        expected_views=37 if scene=="bonsai" else 27
        for mode,iterations in MODES.items():
            directory=root/"benchmark-results"/scene/mode
            result=read_json(directory/"result.json")
            timing=read_json(directory/"training_timing.json")
            if result.get("scene")!=scene or result.get("mode")!=mode or result.get("iterations")!=iterations:
                raise ValueError(f"Wrong completed model: {directory}")
            if "A100" not in timing["hardware"].get("gpu","") or not timing["hardware"].get("cuda"):
                raise ValueError(f"Missing actual A100/CUDA timing evidence: {directory}")
            if timing.get("training_loop_wall_seconds",0)<=0 or timing.get("gaussian_count",0)<=0:
                raise ValueError(f"Training timing/count invalid: {directory}")
            check_rgb_evaluation(result,expected_views,str(directory))
            if timing["test_view_count"]!=expected_views or len(result["split"]["test"])!=expected_views:
                raise ValueError(f"Incomplete test split: {directory}")
            if set(result["split"]["train"]) & set(result["split"]["test"]):
                raise ValueError(f"Train/test overlap: {directory}")
            gaussian_counts[(scene,mode)]=timing["gaussian_count"]
            checked.append(str((directory/"result.json").relative_to(root)))
    priority=root/"teatime-priority/balanced"
    result=read_json(priority/"result.json")
    timing=read_json(priority/"training_timing.json")
    if result.get("iterations")!=15000 or "A100" not in timing["hardware"].get("gpu","") or timing.get("training_loop_wall_seconds",0)<=0:
        raise ValueError("Priority CUDA training has not completed")
    if result["ablation"].get("ground_truth_used_for_training") is not False:
        raise ValueError("Priority experiment lacks no-GT-training declaration")
    check_rgb_evaluation(result,27,"priority")
    read_json(priority/"priority-patch-evidence.json")
    read_json(root/"teatime-priority/comparison.json")
    gaussian_counts[("teatime","balanced-priority")]=timing["gaussian_count"]
    checked.append("teatime-priority/balanced/result.json")
    for mode in SEMANTIC_MODES:
        directory=root/"semantic-results"/mode
        result=read_json(directory/"summary.json")
        classes=read_json(directory/"semantic_classes.json")
        count=gaussian_counts[("teatime",mode)]
        if result.get("backend")!="upstream_cuda_3dgs" or result.get("gaussian_count")!=count or classes.get("gaussian_count")!=count:
            raise ValueError(f"Full-model semantic Gaussian count mismatch: {mode}")
        if len(result["semantics"]["per_object_view"])!=59 or len(classes["classes"])!=14:
            raise ValueError(f"Incomplete semantic GT/class evaluation: {mode}")
        for name in ["mean_iou","class_mean_iou","mean_boundary_iou"]:
            finite_metric(result["semantics"][name],mode+" "+name)
        if len(list(directory.glob("*-masks.png")))!=59:
            raise ValueError(f"Expected every one of 59 semantic mask PNGs: {mode}")
        membership=directory/"semantic_membership.npz"
        with np.load(membership,allow_pickle=False) as arrays:
            if set(arrays.files)!={f"class_{i}" for i in range(14)}:
                raise ValueError(f"Incomplete semantic membership keys: {mode}")
            for key in arrays.files:
                values=arrays[key]
                if values.shape!=(count,) or values.dtype.kind not in "biu" or not np.isin(values,[0,1]).all():
                    raise ValueError(f"Membership array is not full-length binary: {mode}/{key}")
        checked.append(str((directory/"summary.json").relative_to(root)))
    full=read_json(root/"full-resolution-results/summary.json")
    scale=read_json(root/"render-scale-analysis/summary.json")
    expected_pairs={(s,m) for s in ["bonsai","teatime"] for m in MODES}|{("teatime","balanced-priority")}
    if {(r["scene"],r["mode"]) for r in full.get("results",[])}!=expected_pairs or {(r["scene"],r["mode"]) for r in scale.get("results",[])}!=expected_pairs:
        raise ValueError("Full-resolution and scale summaries must each contain all seven completed models")
    for scene,mode in sorted(expected_pairs):
        item=read_json(root/"full-resolution-results"/scene/mode/"result.json")
        if item.get("checkpoint_unchanged") is not True or item.get("training_performed") is not False:
            raise ValueError(f"Full-resolution render lacks checkpoint/read-only evidence: {scene}/{mode}")
        check_rgb_evaluation(item,37 if scene=="bonsai" else 27,f"full {scene}/{mode}")
        width,height=item["downloaded_base_size"]
        if (width,height)!=((780,520) if scene=="bonsai" else (988,730)):
            raise ValueError(f"Wrong downloaded-base evaluation resolution: {scene}/{mode}")
        scale_item=read_json(root/"render-scale-analysis"/scene/mode/"result.json")
        if scale_item.get("all_native_resized_gt_equal_original_256") is not True or len(scale_item["per_view"])!=(37 if scene=="bonsai" else 27):
            raise ValueError(f"Incomplete or GT-mismatched scale evaluation: {scene}/{mode}")
        checkpoint_evidence.append({"scene":scene,"mode":mode,"source_path":item["source_checkpoint"],
            "sha256_from_readonly_evaluation":item["source_checkpoint_sha256"],"included":False,"reason":"Full PLY retained on Colab; excluded from compact delivery"})
    for folder in ["full-resolution-results","render-scale-analysis"]:
        if not (root/folder/"summary.csv").is_file():
            raise FileNotFoundError(f"{folder}/summary.csv missing")
    discovery=read_json(root/"cuda-discovery-latency.json")
    if discovery.get("status")!="completed_cuda_nonempty_masks_verified" or discovery.get("runtime",{}).get("device")!="cuda" or "A100" not in discovery.get("runtime",{}).get("gpu",""):
        raise ValueError("CUDA discovery did not actually complete on A100")
    if not discovery.get("warm_passes") or discovery["cold_first_pass"].get("nonempty_mask_count",0)<=0:
        raise ValueError("CUDA discovery lacks nonempty cold/warm measurements")
    for model_device in discovery["model_parameter_devices"].values():
        if not model_device.startswith("cuda"):
            raise ValueError("CUDA discovery contains a CPU model fallback")
    for scene in ["bonsai","teatime"]:
        directory=root/"previews"/scene
        preview=read_json(directory/"scene.json")
        exported=read_json(directory/"export_manifest.json")
        count=gaussian_counts[(scene,"fine")]
        expected_count=min(100000,count)
        indices=np.load(directory/"selected_indices.npy",allow_pickle=False)
        if indices.shape!=(expected_count,) or indices.dtype.kind not in "iu" or (np.diff(indices)<=0).any() or indices.min()<0 or indices.max()>=count:
            raise ValueError(f"Preview selected indices invalid: {scene}")
        if exported["source_gaussian_count"]!=count or exported["preview_gaussian_count"]!=expected_count or len(preview["gaussians"])!=expected_count:
            raise ValueError(f"Preview count differs from full-model source: {scene}")
        if [g["source_index"] for g in preview["gaussians"]]!=indices.tolist():
            raise ValueError(f"Preview scene and selected indices do not align: {scene}")
        source_hash=next(p["sha256_from_readonly_evaluation"] for p in checkpoint_evidence if p["scene"]==scene and p["mode"]=="fine")
        if exported["source_ply_sha256"]!=source_hash or preview["metadata"]["source_ply_sha256"]!=source_hash:
            raise ValueError(f"Preview source PLY differs from evaluated fine model: {scene}")
        del preview
    return {"completion_status":"All required completed result artifacts validated; live GPU check still required before archive",
            "baseline_models":6,"priority_models":1,"semantic_full_models":4,"full_resolution_models":7,"scale_comparisons":7,
            "preview_scenes":2,"checked_primary_results":checked,"excluded_full_model_evidence":checkpoint_evidence}


def live_gpu_completion_check():
    """Read-only check: no known training/evaluation workers and GPU utilization0."""
    command=["nvidia-smi","--query-gpu=name,utilization.gpu","--format=csv,noheader,nounits"]
    try:
        output=subprocess.check_output(command,text=True,timeout=15)
        processes=subprocess.check_output(["ps","-eo","pid,args"],text=True,timeout=15)
    except (OSError,subprocess.SubprocessError) as exc:
        raise RuntimeError("Cannot verify live GPU completion; use --validate-only for offline checks, not a final archive") from exc
    devices=[]
    for values in csv.reader(io.StringIO(output)):
        if not values:
            continue
        name,utilization=values[0].strip(),values[1].strip()
        if "A100" not in name or not utilization.isdigit() or int(utilization)!=0:
            raise RuntimeError(f"GPU is not verified idle A100: {name}, utilization={utilization}; retry after the experiment finishes")
        devices.append({"gpu":name,"utilization_percent":int(utilization)})
    if not devices:
        raise RuntimeError("No GPU reported by nvidia-smi")
    active=[]
    for line in processes.splitlines()[1:]:
        parts=line.strip().split(None,1)
        if len(parts)!=2 or not parts[0].isdigit() or int(parts[0])==os.getpid():
            continue
        # Compare command argument basenames, not substrings in an ipykernel's
        # source text or this archiver's output filename.
        tokens=parts[1].split()
        if any(Path(token).name in TASK_WORKERS for token in tokens):
            active.append({"pid":int(parts[0]),"command":parts[1]})
    if active:
        raise RuntimeError("Training/evaluation worker still running: "+json.dumps(active))
    return {"checked_utc":time.strftime("%Y-%m-%dT%H:%M:%SZ",time.gmtime()),"devices":devices,
            "known_training_evaluation_workers":[],"interpretation":"Recorded completion plus no active known workers and zero current GPU utilization; no CUDA computation was started"}


def representative_files(root):
    selected=set()
    models=[root/"benchmark-results"/s/m for s in ["bonsai","teatime"] for m in MODES]+[root/"teatime-priority/balanced"]
    for model in models:
        result=read_json(model/"result.json")
        views=result["evaluation"]["per_view"]
        for index in np.linspace(0,len(views)-1,min(3,len(views))).round().astype(int):
            for key in ["render","ground_truth"]:
                path=Path(views[index][key]).resolve()
                if root not in path.parents or model.resolve() not in path.parents:
                    raise ValueError("Representative image points outside its verified model directory")
                selected.add(path)
    for mode in SEMANTIC_MODES:
        files=sorted((root/"semantic-results"/mode).glob("*-rgb.png"))
        for index in np.linspace(0,len(files)-1,min(3,len(files))).round().astype(int):
            selected.add(files[index].resolve())
    return selected


def choose_files(root,include_bonsai_full_model=False,expected_bonsai_sha256=None):
    """Return explicit include/exclude inventory; never recurse point-cloud links."""
    representative=representative_files(root)
    included={}
    excluded=[]
    def record(path,reason):
        relative=path.relative_to(root).as_posix()
        included[relative]={"path":path,"reason":reason}
    for tree in TREES:
        directory=root/tree
        for parent,dirs,files in os.walk(directory,followlinks=False):
            parent=Path(parent)
            retained=[]
            for name in dirs:
                child=parent/name
                if name.lower() in EXCLUDED_DIR_NAMES or child.is_symlink():
                    excluded.append({"path":child.relative_to(root).as_posix()+"/","reason":"Subtree not traversed; only the explicitly requested original Bonsai fine PLY can be selected separately" if include_bonsai_full_model and child==root/"benchmark-results/bonsai/fine/point_cloud" else "Excluded point-cloud/cache/symlink subtree; not traversed"})
                else:
                    retained.append(name)
            dirs[:]=retained
            for name in files:
                path=parent/name
                suffix=path.suffix.lower()
                relative=path.relative_to(root).as_posix()
                if name.lower() in EXCLUDED_FILE_NAMES or name.lower().startswith("credentials."):
                    excluded.append({"path":relative,"reason":"Credential/account file excluded without reading content"})
                    continue
                if path.is_symlink() or path.resolve()!=path:
                    excluded.append({"path":relative,"reason":"Symlink excluded"})
                    continue
                reason=None
                if suffix in TEXT_SUFFIXES or name=="cfg_args":
                    reason="Complete metrics, timing, parameters, provenance, code or experiment log"
                elif tree=="semantic-results" and name in {"semantic_membership.npz"}:
                    reason="Complete full-model semantic membership arrays; no sampling"
                elif tree=="previews" and name=="selected_indices.npy":
                    reason="Complete declared100k preview-to-full-model index mapping"
                elif suffix==".png":
                    if tree=="full-resolution-results":
                        reason=None
                    elif tree=="render-scale-analysis":
                        reason="All CPU render-scale comparison panels"
                    elif tree=="semantic-results" and name.endswith("-masks.png"):
                        reason="All evaluated semantic class GT/prediction mask pairs"
                    elif path.resolve() in representative:
                        reason="Three deterministic representative RGB views per model; metrics retain every view"
                    elif name in {"bonsai-comparison.png","teatime-comparison.png","baseline-vs-priority.png"}:
                        reason="Complete three-mode or priority comparison panel"
                if reason:
                    record(path,reason)
                else:
                    item={"path":relative,"size_bytes":path.stat().st_size,"reason":"Full-resolution RGB PNG excluded; all per-view metrics retained" if tree=="full-resolution-results" and suffix==".png" else
                          "Full PLY retained remotely" if suffix==".ply" else "Nonrepresentative RGB or non-delivery binary; no metric values removed"}
                    if suffix!=".ply":
                        item["sha256"]=sha256(path)
                    excluded.append(item)
    for name in TOP_FILES:
        path=root/name
        if path.is_file() and not path.is_symlink():
            record(path,"Explicit known experiment script, status, or key log")
    for code_root in [root/"priority-tools",root/"splat-benchmark-tools/benchmarks",root/"splat-benchmark-tools/backend"]:
        if code_root.is_dir():
            for path in code_root.glob("*.py"):
                if path.is_file() and not path.is_symlink():
                    record(path,"Exact uploaded benchmark/application dependency code")
    allowed_ply="benchmark-results/bonsai/fine/point_cloud/iteration_22000/point_cloud.ply"
    if include_bonsai_full_model:
        full_path=root/allowed_ply
        if not expected_bonsai_sha256 or not full_path.is_file() or full_path.is_symlink() or full_path.resolve()!=full_path:
            raise ValueError("Explicit full Bonsai model requires the original regular PLY and its completed-evaluation hash")
        if sha256(full_path)!=expected_bonsai_sha256:
            raise ValueError("Original Bonsai fine PLY hash differs from the completed full-resolution evaluation")
        record(full_path,"Explicitly requested full original Bonsai fine PLY; unchanged bytes, no sampling, quantization, or re-export")
    if any(Path(path).suffix.lower() in {".ply",".pt",".pth",".safetensors"} and not (include_bonsai_full_model and path==allowed_ply) for path in included):
        raise AssertionError("A forbidden full-model/weight binary entered the archive selection")
    return included,excluded


def build_manifest(root,included,excluded,completion,live):
    entries=[]
    for relative,item in sorted(included.items()):
        path=item["path"]
        if path.suffix.lower() in TEXT_SUFFIXES or path.name=="cfg_args":
            content=path.read_bytes()
            if any(pattern.search(content) for pattern in SECRET_PATTERNS):
                raise RuntimeError(f"Possible credential content detected; archive refused without exposing content: {relative}")
        entries.append({"path":relative,"source_path":str(path),"size_bytes":path.stat().st_size,"sha256":sha256(path),"reason":item["reason"]})
    full_bonsai_included="benchmark-results/bonsai/fine/point_cloud/iteration_22000/point_cloud.ply" in included
    if full_bonsai_included:
        for item in completion.get("excluded_full_model_evidence",[]):
            if item["scene"]=="bonsai" and item["mode"]=="fine":
                item["included"]=True
                item["reason"]="Explicit option: full original PLY included and actual source SHA256 rechecked"
    return {"archive_purpose":"Compact user delivery of completed A100 research benchmarks",
            "created_utc":time.strftime("%Y-%m-%dT%H:%M:%SZ",time.gmtime()),"source_root":str(root),
            "completion_validation":completion,"live_gpu_check":live,"included_files":entries,"excluded_files":excluded,
            "uncompressed_included_bytes":sum(e["size_bytes"] for e in entries),
            "full_original_bonsai_fine_ply_included":full_bonsai_included,
            "excluded_without_traversal":["Original data/ datasets and RGB photographs","Hugging Face/model caches","Credentials and account files","Full point_cloud directories and full_model.ply copies (except the single explicitly requested original Bonsai fine PLY, if enabled)","Existing complete benchmark/extended ZIP archives"],
            "data_policy":"Every original numeric per-view metric and full semantic membership array is included unchanged. Only representative rendered RGB images are selected. The existing100k SH0 previews retain all their exported points and their complete mapping, and are not re-sampled here.",
            "source_integrity":"SHA256 refers to exact source file bytes copied into the archive. Full PLY hashes come from the completed read-only full-resolution evaluation and are listed separately; original datasets/caches are not traversed.",
            "archive_hash_location":"External .sha256 and .manifest.json sidecars avoid a self-referential ZIP hash"}


def verify_archive(path,manifest):
    with zipfile.ZipFile(path) as archive:
        if archive.testzip() is not None:
            raise RuntimeError("Archive CRC verification failed")
        for item in manifest["included_files"]:
            digest=hashlib.sha256()
            with archive.open(item["path"]) as stream:
                for chunk in iter(lambda:stream.read(1024*1024),b""):
                    digest.update(chunk)
            if digest.hexdigest()!=item["sha256"]:
                raise RuntimeError("Archive differs from original source hash: "+item["path"])


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root",type=Path,default=Path("/content"))
    parser.add_argument("--output",type=Path,help="Default ROOT/Splat-Studio-A100-results.zip; never overwrites an existing archive")
    cap=parser.add_mutually_exclusive_group()
    cap.add_argument("--max-mb",type=float,help="Decimal compressed ZIP MB ceiling; defaults to100MB")
    cap.add_argument("--max-mib",type=float,help="Binary MiB ceiling, e.g.350 for an explicitly requested full Bonsai PLY")
    parser.add_argument("--include-bonsai-full-model",action="store_true",help="Add only the unchanged full original Bonsai fine iteration22000 PLY and verify its SHA256")
    parser.add_argument("--validate-only",action="store_true",help="Validate completed artifact evidence/selection locally; no live GPU certification and no ZIP")
    args=parser.parse_args()
    root=args.root.resolve()
    output=(args.output or root/"Splat-Studio-A100-results.zip").resolve()
    max_bytes=int(args.max_mib*1024*1024) if args.max_mib is not None else int((args.max_mb if args.max_mb is not None else 100)*1_000_000)
    if max_bytes<=0:
        raise ValueError("Archive size ceiling must be positive")
    completion=validate_completed(root)
    live={"status":"not_checked_offline_validation_only"} if args.validate_only else live_gpu_completion_check()
    full_bonsai_sha=next(item["sha256_from_readonly_evaluation"] for item in completion["excluded_full_model_evidence"] if item["scene"]=="bonsai" and item["mode"]=="fine")
    included,excluded=choose_files(root,args.include_bonsai_full_model,full_bonsai_sha)
    manifest=build_manifest(root,included,excluded,completion,live)
    if args.validate_only:
        print(json.dumps({"status":"completed_artifacts_validated_offline_only","files":len(included),
            "uncompressed_bytes":manifest["uncompressed_included_bytes"],"excluded_inventory_entries":len(excluded),
            "live_gpu_completion_certified":False,"zip_written":False},indent=2))
        return
    if output.exists():
        raise FileExistsError(f"Existing archive preserved; choose a different --output: {output}")
    if any(output==root/tree or (root/tree) in output.parents for tree in TREES):
        raise ValueError("Archive output must be outside all source result trees")
    output.parent.mkdir(parents=True,exist_ok=True)
    descriptor,temporary_name=tempfile.mkstemp(prefix="splat-delivery-",suffix=".zip.partial",dir=output.parent)
    os.close(descriptor)
    temporary=Path(temporary_name)
    try:
        with zipfile.ZipFile(temporary,"w",zipfile.ZIP_DEFLATED,compresslevel=6) as archive:
            for entry in manifest["included_files"]:
                archive.write(included[entry["path"]]["path"],entry["path"])
            archive.writestr("DELIVERY_MANIFEST.json",json.dumps(manifest,ensure_ascii=False,indent=2,allow_nan=False))
        archive_bytes=temporary.stat().st_size
        if archive_bytes>max_bytes:
            oversized=output.with_suffix(".oversize.zip")
            if oversized.exists():
                raise FileExistsError(f"Oversized prior attempt preserved: {oversized}")
            temporary.replace(oversized)
            raise RuntimeError(f"ZIP is {archive_bytes:,} bytes, above requested{max_bytes:,} bytes. No metrics/points were sampled. Unpublished oversized artifact retained at {oversized}; choose an explicit higher ceiling or review image/log inclusion.")
        verify_archive(temporary,manifest)
        zip_hash=sha256(temporary)
        temporary.replace(output)
        (output.with_suffix(output.suffix+".sha256")).write_text(f"{zip_hash}  {output.name}\n")
        write_json(output.with_suffix(output.suffix+".manifest.json"),{**manifest,"archive":{"path":str(output),"bytes":archive_bytes,"sha256":zip_hash,"verified_each_member_against_source_sha256":True}})
        print(json.dumps({"status":"complete","archive":str(output),"bytes":archive_bytes,"sha256":zip_hash,
            "included_files":len(included),"original_archives_preserved":True},indent=2))
    finally:
        if temporary.exists():
            temporary.unlink()


if __name__=="__main__":
    main()
