#!/usr/bin/env python3
"""Controlled CUDA importance-loss experiment against a completed teatime baseline.

Uses the sibling run_cuda_benchmark.py's exact instrumentation, data split,
optimizer defaults, renderer and evaluation. Only the app's normalized weighted
L1 replaces original L1: foreground=3, background=1; default SSIM stays unchanged.
No upstream tracked source file is edited. No held-out GT enters optimization.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


# This hook matches backend/upstream.py, including PIL's default resize and the
# per-image normalization by total weight. Cached masks are predicted train masks.
PRIORITY_HOOK = r'''
import numpy as splat_np
from PIL import Image as SplatImage
splat_mask_dir = SPLAT_BENCHMARK_PRIORITY_MASK_DIR
splat_mask_cache = {}
splat_mask_foreground = {}
splat_priority_expected = set(SPLAT_BENCHMARK_PRIORITY_IMAGE_NAMES)
splat_priority_calls = {"total": 0, "weighted": 0, "background_only": 0}
splat_priority_loaded = set()
def splat_weighted_l1(image, target, name):
    key = (name, tuple(image.shape))
    if key not in splat_mask_cache:
        path = os.path.join(splat_mask_dir, name + ".png")
        if name in splat_priority_expected and not os.path.isfile(path):
            raise RuntimeError("Declared priority mask is missing for exact camera image_name: " + name)
        if os.path.isfile(path):
            mask = SplatImage.open(path).convert("L").resize((image.shape[2], image.shape[1]))
            weight = 1 + 2 * torch.tensor(splat_np.array(mask).copy(), device=image.device, dtype=image.dtype) / 255
            foreground = bool((weight > 1).any().item())
            if name in splat_priority_expected and not foreground:
                raise RuntimeError("Declared priority mask became empty at training resolution: " + name)
            if foreground: splat_priority_loaded.add(name)
        else:
            weight = torch.ones_like(image[0])
            foreground = False
        splat_mask_cache[key] = weight
        splat_mask_foreground[key] = foreground
    splat_priority_calls["total"] += 1
    splat_priority_calls["weighted" if splat_mask_foreground[key] else "background_only"] += 1
    weight = splat_mask_cache[key]
    return (torch.abs(image-target)*weight).sum()/(3*weight.sum())

def splat_priority_report():
    missing = sorted(splat_priority_expected - splat_priority_loaded)
    if not splat_priority_calls["weighted"] or missing:
        raise RuntimeError("Priority weights were not applied to every declared view; missing=" + repr(missing))
    return {"status": "verified_applied", "calls": dict(splat_priority_calls),
        "expected_image_names": sorted(splat_priority_expected),
        "loaded_nonempty_image_names": sorted(splat_priority_loaded),
        "mask_filename_rule": "exact_camera_image_name_plus_dot_png",
        "formula": "normalized_pixel_weight_1_plus_2M"}
'''


def load_benchmark(path):
    spec=importlib.util.spec_from_file_location("priority_original_benchmark",path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


def build_worker(original_worker):
    insertion="train.Scene = MeasuredScene"
    if original_worker.count(insertion)!=1:
        raise ValueError("Benchmark worker changed; cannot instrument priority experiment safely")
    hook=r'''
if job.get("action") != "render":
    import inspect, hashlib, difflib
    original_training = inspect.getsource(train.training)
    marker = "Ll1 = l1_loss(image, gt_image)"
    replacement = "Ll1 = splat_weighted_l1(image, gt_image, viewpoint_cam.image_name)"
    if original_training.count(marker) != 1:
        raise RuntimeError("Pinned training loss marker must occur exactly once")
    priority_training = original_training.replace(marker,replacement)
    original_file=output/"training_original_function.py"
    patched_file=output/"training_priority_function.py"
    hook_file=output/"priority_loss_hook.py"
    original_file.write_text(original_training)
    patched_file.write_text(priority_training)
    hook_file.write_text(PRIORITY_HOOK_TEXT)
    delta="".join(difflib.unified_diff(original_training.splitlines(True),priority_training.splitlines(True),
        fromfile="original/training",tofile="importance/training"))
    (output/"priority-loss.patch").write_text(delta)
    evidence={"marker_count":1,"only_training_function_change":delta,
        "original_training_sha256":hashlib.sha256(original_training.encode()).hexdigest(),
        "priority_training_sha256":hashlib.sha256(priority_training.encode()).hexdigest(),
        "priority_hook_sha256":hashlib.sha256(PRIORITY_HOOK_TEXT.encode()).hexdigest(),
        "upstream_train_py_sha256":hashlib.sha256((Path(job["upstream"])/"train.py").read_bytes()).hexdigest(),
        "formula":"sum(abs(render-target)*(1+2*mask))/(3*sum(1+2*mask)); no-mask image uses all-one weight",
        "ssim":"unchanged original default; lambda_dssim=0.2",
        "upstream_files_modified":False,"training_masks":job["priority_mask_provenance"]}
    (output/"priority-patch-evidence.json").write_text(json.dumps(evidence,indent=2))
    train.__dict__["SPLAT_BENCHMARK_PRIORITY_MASK_DIR"]=job["priority_masks_dir"]
    train.__dict__["SPLAT_BENCHMARK_PRIORITY_IMAGE_NAMES"]=[row["image_name"] for row in job["priority_mask_provenance"]["priority_masks"]]
    exec(compile(PRIORITY_HOOK_TEXT,str(hook_file),"exec"),train.__dict__)
    exec(compile(priority_training,str(patched_file),"exec"),train.__dict__)
'''
    hook=hook.replace("PRIORITY_HOOK_TEXT",repr(PRIORITY_HOOK))
    result=original_worker.replace(insertion,hook+"\n"+insertion)
    timing_marker='(output/"training_timing.json").write_text(json.dumps(timing,indent=2))'
    if result.count(timing_marker)!=1:raise ValueError('Benchmark timing publication marker changed')
    audit='timing["priority_loss_application"] = train.splat_priority_report()\n(output/"priority-loss-application.json").write_text(json.dumps(timing["priority_loss_application"],indent=2))\n'
    return result.replace(timing_marker,audit+timing_marker)


def prepare_masks(filename,scene_dir,images_dir,split,destination,priority_labels=('stuffed bear',)):
    raw=json.loads(filename.read_text())
    if not isinstance(raw,dict) or raw.get("ground_truth_used") is not False:
        raise ValueError("Training mask package must explicitly declare ground_truth_used=false")
    selected={label.strip().casefold() for label in priority_labels}
    if not selected:raise ValueError('At least one priority label is required')
    records=raw.get("masks",raw) if isinstance(raw,dict) else raw
    train_names,test_names=set(split["train"]),set(split["test"])
    if train_names & test_names:
        raise ValueError("Baseline train/test split overlaps")
    all_mask_views=set();grouped={}
    source_records=[]
    for record in records:
        name=Path(record.get("image_name",record.get("image_path",""))).name
        if name not in train_names:
            raise ValueError(f"Predicted mask {name} is not a baseline training view")
        all_mask_views.add(name)
        if record["label"].strip().casefold() not in selected:
            continue
        mask_path=Path(record["mask_path"])
        if not mask_path.is_absolute():
            mask_path=filename.parent/mask_path
        with Image.open(scene_dir/images_dir/name) as opened:
            expected_size=opened.size
        mask_image=Image.open(mask_path).convert("L")
        native_mask_size=mask_image.size
        if mask_image.size!=expected_size:
            ratio_error=abs((mask_image.width/mask_image.height)/(expected_size[0]/expected_size[1])-1)
            if ratio_error>.01:
                raise ValueError(f"Predicted mask {mask_path} aspect ratio differs from RGB; alignment must be verified")
            # App build_priority_masks likewise maps thumbnail SAM masks back to
            # original RGB dimensions with nearest-neighbor before the loss hook.
            mask_image=mask_image.resize(expected_size,Image.Resampling.NEAREST)
        foreground=np.asarray(mask_image)>0
        if not foreground.any():
            continue
        grouped[name]=foreground|grouped.get(name,np.zeros(foreground.shape,dtype=bool))
        source_records.append({"image_name":name,"label":record["label"],"mask_sha256":hashlib.sha256(mask_path.read_bytes()).hexdigest(),
            "native_mask_size":list(native_mask_size),"mapped_to_rgb_size":list(expected_size),"mapping":"nearest-neighbor, matching app build_priority_masks"})
    if not grouped:
        raise ValueError("No nonempty predicted priority masks available in training views")
    destination.mkdir(parents=True,exist_ok=True)
    stale={p.name for p in destination.glob("*.png")}-{n+".png" for n in grouped}
    if stale:
        raise ValueError("Priority mask output contains masks from a different run; choose a new output directory")
    output_records=[]
    for name,mask in sorted(grouped.items()):
        path=destination/(name+".png")
        Image.fromarray(mask.astype(np.uint8)*255).save(path)
        output_records.append({"image_name":name,"mask_path":str(path.resolve()),
            "foreground_fraction":float(mask.mean()),"sha256":hashlib.sha256(path.read_bytes()).hexdigest()})
    return {"source_json_sha256":hashlib.sha256(filename.read_bytes()).hexdigest(),
        "source_model":raw.get("source",raw.get("model")) if isinstance(raw,dict) else None,
        "label":next(iter(selected)) if len(selected)==1 else None,"labels":sorted(selected),"union_of_source_masks":len(source_records),
        "input_prediction_view_count":len(all_mask_views),"priority_view_count":len(grouped),
        "training_view_count":len(train_names),"priority_training_view_fraction":len(grouped)/len(train_names),
        "source_masks":source_records,"priority_masks":output_records,"ground_truth_used":False,
        "unmasked_training_views":"All other training images receive weight=1 everywhere.",
        "mask_filename_rule":"exact_camera_image_name_plus_dot_png",
        "mask_provenance_limitation":"Membership and hashes are verified; original prediction provenance is declared by the cached mask package."}


def comparison(baseline,priority,output):
    old=baseline["evaluation"];new=priority["evaluation"]
    a=old["per_view"];b=new["per_view"]
    if [r["image_name"] for r in a]!=[r["image_name"] for r in b]:
        raise ValueError("Cannot compare different held-out views")
    delta={"baseline_mean_psnr_db":old["mean_psnr_db"],"priority_mean_psnr_db":new["mean_psnr_db"],
        "delta_psnr_db":new["mean_psnr_db"]-old["mean_psnr_db"],
        "baseline_mean_ssim":old["mean_ssim"],"priority_mean_ssim":new["mean_ssim"],
        "delta_ssim":new["mean_ssim"]-old["mean_ssim"],
        "baseline_training_loop_seconds":baseline["timing"]["training_loop_wall_seconds"],
        "priority_training_loop_seconds":priority["timing"]["training_loop_wall_seconds"],
        "interpretation":"These are whole-image held-out deltas. Evaluate separate stuffed-bear GT ROI and background ROI before claiming priority-object quality improved. Negative deltas are retained."}
    ids=np.linspace(0,len(a)-1,min(3,len(a))).round().astype(int).tolist()
    width=256;height=round(a[0]["height"]*width/a[0]["width"]);header=52;row_h=height+24
    canvas=Image.new("RGB",(width*3,header+row_h*len(ids)),"#151923");draw=ImageDraw.Draw(canvas)
    for col,title in enumerate(["Held-out GT","Balanced baseline","Balanced + bear priority"]):
        draw.text((col*width+6,8),title,fill="white")
    draw.text((width+6,26),f"PSNR {old['mean_psnr_db']:.2f}, SSIM {old['mean_ssim']:.3f}",fill="#d2daee")
    draw.text((2*width+6,26),f"PSNR {new['mean_psnr_db']:.2f}, SSIM {new['mean_ssim']:.3f}",fill="#d2daee")
    for row,i in enumerate(ids):
        for col,image_path in enumerate([a[i]["ground_truth"],a[i]["render"],b[i]["render"]]):
            image=Image.open(image_path).convert("RGB");image.thumbnail((width,height),Image.Resampling.LANCZOS)
            canvas.paste(image,(col*width,header+row*row_h))
            label=a[i]["image_name"] if col==0 else f"PSNR {(a if col==1 else b)[i]['psnr_db']:.2f}"
            draw.text((col*width+6,header+row*row_h+height+5),label,fill="white")
    canvas.save(output/"baseline-vs-priority.png")
    return delta


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-model",type=Path,required=True,help="Completed baseline directory containing job.json and result.json")
    parser.add_argument("--training-masks",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True,help="Separate experiment root, for example /content/teatime-priority")
    parser.add_argument("--benchmark-runner",type=Path,default=Path(__file__).with_name("run_cuda_benchmark.py"))
    parser.add_argument("--dry-run",action="store_true",help="Validate job/splits/masks and write worker without requiring a completed CUDA baseline")
    args=parser.parse_args();args.baseline_model=args.baseline_model.resolve();args.output=args.output.resolve()
    benchmark=load_benchmark(args.benchmark_runner.resolve())
    original_job=json.loads((args.baseline_model/"job.json").read_text())
    if original_job["scene"]!="teatime" or original_job["iterations"]!=15000 or original_job["resolution"]!=2 or original_job["mode"]!="balanced":
        raise ValueError("This controlled experiment requires teatime balanced, exactly 15000 iterations and resolution divisor 2")
    split=original_job["split"]
    scene_dir=Path(original_job["scene_dir"]);upstream=Path(original_job["upstream"])
    upstream_info=benchmark.check_upstream(upstream)
    fresh_split=benchmark.scene_split(scene_dir,"teatime")
    if split!=fresh_split:
        raise ValueError("Baseline split differs from standard holdout plus all six author GT views")
    metadata=benchmark.scene_metadata(scene_dir)
    if metadata["source_signature_sha256"]!=original_job["source_signature_sha256"]:
        raise ValueError("Baseline dataset files changed")
    model=args.output/"balanced";model.mkdir(parents=True,exist_ok=True)
    masks=prepare_masks(args.training_masks.resolve(),scene_dir,original_job["images"],split,args.output/"priority_masks")
    benchmark.write_json(args.output/"split.json",split)
    benchmark.write_json(args.output/"priority-mask-provenance.json",masks)
    job={**original_job,"model_path":str(model),"priority_masks_dir":str((args.output/"priority_masks").resolve()),
        "priority_mask_provenance":masks}
    job_path=model/"job.json"
    if job_path.exists() and json.loads(job_path.read_text())!=job:
        raise ValueError("Existing priority job differs; use a new experiment directory")
    benchmark.write_json(job_path,job)
    worker=args.output/"_priority_training_worker.py";worker.write_text(build_worker(benchmark.WORKER))
    print(json.dumps({"mode":"balanced","iterations":15000,"resolution":2,"priority_views":masks["priority_view_count"],
        "train_views":len(split["train"]),"test_views":len(split["test"]),"foreground_weight":3,"background_weight":1,
        "seed":0,"same_as_baseline":True,"no_mask_weight":1,"ground_truth_used_for_training":False}),flush=True)
    if args.dry_run:
        return
    baseline_path=args.baseline_model/"result.json"
    baseline=json.loads(baseline_path.read_text())
    if baseline["split"]!=split or baseline["eval_width"]!=job["eval_width"]:
        raise ValueError("Completed baseline metadata differs from its recorded job")
    if (model/"result.json").exists():
        completed=json.loads((model/"result.json").read_text())
        if completed.get('timing',{}).get('priority_loss_application',{}).get('status')!='verified_applied':
            raise ValueError('Existing result has no verified priority mask application; use a new output directory')
        benchmark.write_json(args.output/"comparison.json",comparison(baseline,completed,args.output))
        print(f"Completed priority result already exists: {model/'result.json'}",flush=True)
        return
    timing_path=model/"training_timing.json"
    if not timing_path.exists():
        (model/"progress.jsonl").unlink(missing_ok=True)
        import sys
        duration=benchmark.run_logged([sys.executable,worker,job_path],upstream,model/"train.log")
        timing=json.loads(timing_path.read_text());timing["training_subprocess_wall_seconds"]=duration
        benchmark.write_json(timing_path,timing)
    timing=json.loads(timing_path.read_text())
    if timing.get('priority_loss_application',{}).get('status')!='verified_applied':
        raise ValueError('Priority training has no verified mask-hit evidence; refusing to claim an importance ablation')
    if timing["hardware"]!=baseline["timing"]["hardware"]:
        raise ValueError("Baseline and priority training hardware/PyTorch differ; do not label this a controlled same-device ablation")
    render_job=model/"render_job.json";benchmark.write_json(render_job,{**job,"action":"render"})
    import sys
    render_seconds=benchmark.run_logged([sys.executable,worker,render_job],upstream,model/"render.log")
    evaluation=benchmark.evaluate_pngs(model,15000,timing["test_image_names"])
    result={"scene":"teatime-priority","mode":"balanced","iterations":15000,"resolution_divisor":2,
        "eval_width":job["eval_width"],"split":split,"upstream":upstream_info,"dataset":metadata,"timing":timing,
        "evaluation":evaluation,"render_subprocess_wall_seconds":render_seconds,"ablation":{
            "baseline_model":str(args.baseline_model),"baseline_result_sha256":hashlib.sha256(baseline_path.read_bytes()).hexdigest(),
            "benchmark_runner_sha256":hashlib.sha256(args.benchmark_runner.read_bytes()).hexdigest(),
            "importance_script_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "loss":"App's foreground3/background1 normalized weighted L1; lambda_dssim=0.2 and SSIM unchanged",
            "only_changed_factor":"predicted stuffed-bear priority weights in normalized L1",
            "seed":0,"predicted_training_masks":masks,"ground_truth_used_for_training":False},
        "scope":"Full upstream CUDA model, same baseline iteration/resolution/view/seed budget. Eight predicted training-mask views, no GT training. No joint semantic loss, no explicit extra-Gaussian quota.",
        "completed_utc":time.strftime("%Y-%m-%dT%H:%M:%SZ",time.gmtime())}
    benchmark.write_json(model/"result.json",result)
    deltas=comparison(baseline,result,args.output)
    benchmark.write_json(args.output/"comparison.json",deltas)
    print("PRIORITY_ABLATION_RESULT "+json.dumps(deltas),flush=True)


if __name__=="__main__":
    main()
