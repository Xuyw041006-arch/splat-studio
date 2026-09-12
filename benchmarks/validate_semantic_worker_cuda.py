#!/usr/bin/env python3
"""Independent A100 smoke test of the production semantic worker.

This is NOT a new row of the frozen R1/R2 research ablations. It uses the full
balanced 15k PLY, the original eight training-mask views, an identity-verified
step-zero prior, and 400 improved semantic steps. Only coffee mug -> coffee is
approved (contents_of). GT annotations are opened only after worker completion.

Default paths target the existing Colab workspace; this script never connects to
Colab, downloads data, trains RGB, or changes an earlier experiment. Run it there
with ``python -u benchmarks/validate_semantic_worker_cuda.py``. A new output
directory is required. Exit 0 means complete; exit 2 means failed.

``--fresh-default`` runs a separate 1200-step product-condition validation with
uniform .05 initialization, no prior, and no approved hierarchy. It requires the
user-selected cuda_iterative semantic mode; the overall app default remains
projection. The fresh run defaults to /content/semantic-worker-fresh-validation.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import subprocess
import sys
import time
import traceback

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

ORIGINAL_TRAINING_VIEWS = {
    "frame_00003.jpg", "frame_00026.jpg", "frame_00050.jpg", "frame_00072.jpg",
    "frame_00106.jpg", "frame_00130.jpg", "frame_00157.jpg", "frame_00179.jpg",
}
APPROVED_HIERARCHY = [{"parent": "coffee mug", "child": "coffee", "relation": "contents_of", "source": "user_explicit"}]
EXPERIMENT_KIND = "independent_production_worker_cuda_smoke_test_not_frozen_R1_R2"
FRESH_EXPERIMENT_KIND = "independent_fresh_cuda_iterative_default_validation_not_frozen_R1_R2"


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, allow_nan=False, indent=2) + "\n")
    temporary.replace(path)


def step_zero_checkpoint(experiment_dir):
    """The source trainer writes step_{step:05d}, i.e. step_00000 at step zero."""
    root = Path(experiment_dir)
    candidates = []
    for file in root.glob("step_*/probabilities.npz"):
        suffix = file.parent.name.removeprefix("step_")
        if suffix.isdigit() and int(suffix) == 0:
            candidates.append(file)
    if len(candidates) != 1:
        raise ValueError("Require exactly one step-zero probabilities.npz; refuse an ambiguous/missing initializer")
    return candidates[0]


def prepare_verified_prior(experiment_dir, ply_path, split_path, mask_path, training_views, output_path):
    """Add identity only after validating the source experiment's provenance."""
    root, output = Path(experiment_dir), Path(output_path)
    if output.exists():
        raise FileExistsError(output)
    source = step_zero_checkpoint(root)
    experiment_path = root / "experiment.json"
    metadata = json.loads(experiment_path.read_text())
    expected_sha = sha(ply_path)
    checks = {"ply_sha256": expected_sha, "split_sha256": sha(split_path), "training_masks_sha256": sha(mask_path)}
    if any(metadata.get(key) != value for key, value in checks.items()):
        raise ValueError("Step-zero prior source PLY/split/training-mask hashes do not match this validation")
    if metadata.get("gt_read_for_training") is not False or metadata.get("geometry_frozen") is not True or metadata.get("opacity_frozen") is not True:
        raise ValueError("Step-zero prior lacks explicit train-only/frozen-geometry provenance")
    if set(metadata.get("training_views", [])) != set(training_views):
        raise ValueError("Step-zero prior was not initialized from the same original eight training views")
    labels, count = metadata.get("classes"), metadata.get("gaussian_count")
    if not isinstance(labels, list) or not labels or any(not isinstance(label, str) for label in labels) or len(set(labels)) != len(labels):
        raise ValueError("Step-zero experiment classes are invalid")
    if not isinstance(count, int) or isinstance(count, bool) or count <= 0:
        raise ValueError("Step-zero experiment Gaussian count is invalid")
    classes_path = source.parent / "semantic_classes.json"
    class_metadata = json.loads(classes_path.read_text())
    if class_metadata.get("classes") != labels or class_metadata.get("gaussian_count") != count or class_metadata.get("order") != "unaltered input PLY order":
        raise ValueError("Step-zero checkpoint class order or full PLY vertex order is unverified")
    checkpoint_path = source.parent / "checkpoint.json"
    checkpoint = json.loads(checkpoint_path.read_text())
    if checkpoint.get("step") != 0 or checkpoint.get("geometry_unchanged") is not True:
        raise ValueError("Initializer must be a verified, unchanged step-zero checkpoint")
    with np.load(source, allow_pickle=False) as data:
        values = data["probabilities"]
        if values.shape != (count, len(labels)) or values.dtype.kind != "f" or not np.isfinite(values).all() or ((values < 0) | (values > 1)).any():
            raise ValueError("Step-zero probabilities do not match the complete PLY/classes")
        source_dtype = str(values.dtype)
        values = values.astype(np.float32)
    with output.open("xb") as stream:
        np.savez_compressed(stream, probabilities=values, source_ply_sha256=np.asarray(expected_sha), classes=np.asarray(labels))
    return {"path": str(output.resolve()), "source_ply_sha256": expected_sha, "classes": labels,
            "gaussian_count": count, "source_checkpoint": str(source.resolve()), "source_checkpoint_sha256": sha(source),
            "source_experiment_sha256": sha(experiment_path), "source_classes_sha256": sha(classes_path),
            "source_checkpoint_metadata_sha256": sha(checkpoint_path), "prior_sha256": sha(output),
            "source_storage_dtype": source_dtype, "worker_prior_dtype": "float32",
            "values_changed": "dtype promotion only; no re-fusion, learning, or GT-dependent selection"}


def environment(torch):
    if not torch.cuda.is_available():
        raise RuntimeError("This smoke test requires the existing A100 CUDA runtime")
    device = torch.cuda.current_device()
    name = torch.cuda.get_device_name(device)
    if "A100" not in name.upper():
        raise RuntimeError("This validation is declared for A100; current GPU is " + name)
    result = {"python": sys.version, "platform": platform.platform(), "torch": torch.__version__,
              "torch_cuda": torch.version.cuda, "cuda_device": device, "gpu": name,
              "gpu_total_memory_bytes": torch.cuda.get_device_properties(device).total_memory,
              "torch_cpu_threads": torch.get_num_threads()}
    for name in ("numpy", "Pillow", "opencv-python", "plyfile", "transformers"):
        try:
            result[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            result[name] = None
    for name, command in (("nvidia_smi", ["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"]),
                          ("nvcc", ["nvcc", "--version"])):
        try:
            result[name] = subprocess.check_output(command, text=True, stderr=subprocess.STDOUT, timeout=15).strip()
        except (OSError, subprocess.SubprocessError) as exc:
            result[name] = {"unavailable": type(exc).__name__}
    return result


def prepare_training_inputs(args):
    """Only train masks and supplied scene calibration enter this function."""
    from benchmarks.run_benchmark import load_colmap
    from benchmarks.evaluate_cuda_semantics import read_training_masks
    records, points, provenance = load_colmap(args.scene_dir, Path(args.upstream) / "utils/read_write_model.py")
    del points
    by_name = {}
    for record in records:
        name = Path(record["name"]).name
        if name in by_name:
            raise ValueError("Duplicate image basenames in COLMAP scene")
        by_name[name] = record
    raw_split = json.loads(Path(args.split).read_text())
    split = raw_split.get("split", raw_split)
    train, test = ({Path(name).name for name in split[key]} for key in ("train", "test"))
    if train & test or (train | test) - set(by_name):
        raise ValueError("Invalid supplied RGB train/test split")
    mask_metadata = json.loads(Path(args.training_masks).read_text())
    if mask_metadata.get("ground_truth_used") is not False:
        raise ValueError("Original cached training masks must declare ground_truth_used=false")
    masks = read_training_masks(args.training_masks, by_name, train)
    views = {Path(record["image_path"]).name for record in masks}
    if views != ORIGINAL_TRAINING_VIEWS or views & test:
        raise ValueError("Production smoke test must use exactly the original eight semantic training views")
    cameras = [{"image_name": name, "width": by_name[name]["width"], "height": by_name[name]["height"],
                "intrinsics": by_name[name]["intrinsics"], "world_to_camera": by_name[name]["world_to_camera"]}
               for name in sorted(views)]
    return {"by_name": by_name, "split": split, "train": train, "test": test, "masks": masks,
            "views": sorted(views), "cameras": cameras, "colmap_provenance": provenance,
            "training_mask_files": [{"path": str(Path(mask["mask_path"]).resolve()), "sha256": sha(mask["mask_path"])} for mask in masks]}


def load_worker_representation(path, representation, count, labels, ply_sha):
    """Load exported product artifacts, verifying their embedded source identity."""
    with np.load(path, allow_pickle=False) as data:
        if str(data["source_ply_sha256"]) != ply_sha or data["classes"].tolist() != labels:
            raise ValueError("Worker output identity or class order changed")
        if representation == "raw_probability":
            values = np.asarray(data["probabilities"], dtype=np.float32)
        else:
            values = np.stack([np.asarray(data[f"class_{index}"]) for index in range(len(labels))], axis=1)
            if values.dtype.kind != "b":
                raise ValueError("Editable membership must be boolean")
            values = values.astype(np.float32)
    if values.shape != (count, len(labels)) or not np.isfinite(values).all() or ((values < 0) | (values > 1)).any():
        raise ValueError("Worker output shape/probability range is invalid")
    return values


def evaluate_worker_outputs(args, prepared, result, destination, *, verify_loader_reentry=False):
    """Evaluation-only entry point; called strictly after the worker completes."""
    import cv2
    import torch
    from PIL import Image
    from backend.semantic_targets import normalize_label
    from backend.semantic_worker import _load_upstream_model, _tensor_hashes
    from benchmarks.evaluate_cuda_semantics import camera_for_record
    from benchmarks.run_benchmark import load_annotations, mask_metrics, mean_metric
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    annotations = load_annotations(args.annotations, args.scene_dir)
    if set(annotations) & prepared["train"] or set(annotations) - prepared["test"]:
        raise ValueError("GT views are not strictly held out from RGB/semantic training")
    if len(annotations) != 6:
        raise ValueError("Expected the same six held-out evaluation views")
    metadata = result["metadata"]
    reentry = {"requested": verify_loader_reentry}
    if verify_loader_reentry:
        load_start = time.perf_counter()
        first_model, first_renderer, _ = _load_upstream_model(Path(args.ply).resolve(), Path(args.upstream).resolve(), metadata["sh_degree"])
        first_hashes = _tensor_hashes(first_model, freeze=True)
        reentry.update(first_load_and_hash_seconds=time.perf_counter() - load_start,
                       first_tensor_hashes=first_hashes,
                       namespaces_after_first_load={name: {"file": getattr(sys.modules.get(name), "__file__", None),
                           "path": list(getattr(sys.modules.get(name), "__path__", []))}
                           for name in ("scene", "gaussian_renderer", "utils")})
        del first_model, first_renderer
        gc.collect()
        torch.cuda.empty_cache()
    load_start = time.perf_counter()
    model, renderer, device = _load_upstream_model(Path(args.ply).resolve(), Path(args.upstream).resolve(), metadata["sh_degree"])
    before = _tensor_hashes(model, freeze=True)
    if verify_loader_reentry:
        if before != first_hashes:
            raise RuntimeError("Consecutive native model loads produced different frozen tensor hashes")
        reentry.update(second_load_and_hash_seconds=time.perf_counter() - load_start,
                       consecutive_same_process_loads=2, first_model_released_before_second=True,
                       tensor_hashes_identical=True, second_tensor_hashes=before,
                       namespaces_after_second_load={name: {"file": getattr(sys.modules.get(name), "__file__", None),
                           "path": list(getattr(sys.modules.get(name), "__path__", []))}
                           for name in ("scene", "gaussian_renderer", "utils")})
    if before != metadata["final_tensor_hashes"]:
        raise ValueError("Evaluation model tensors differ from the production worker's fixed PLY")
    ground_truth, cameras, gt_file_hashes = {}, {}, []
    for name in sorted(annotations):
        cameras[name], _ = camera_for_record(prepared["by_name"][name], 256, torch)
        truth = {}
        for item in annotations[name]:
            label = normalize_label(item["label"])
            path = Path(item["mask_path"])
            with Image.open(path) as image:
                mask = np.asarray(image.convert("L")) > 0
            mask = cv2.resize(mask.astype(np.uint8), (cameras[name].image_width, cameras[name].image_height), interpolation=cv2.INTER_NEAREST).astype(bool)
            truth[label] = mask | truth.get(label, np.zeros_like(mask))
            gt_file_hashes.append({"image": name, "label": label, "path": str(path), "sha256": sha(path)})
        ground_truth[name] = truth
    all_labels = sorted({label for frame in ground_truth.values() for label in frame})
    pairs = sum(len(frame) for frame in ground_truth.values())
    if pairs != 59 or len(all_labels) != 14:
        raise ValueError(f"Expected the same 59 mask/view pairs and 14 classes; got {pairs}/{len(all_labels)}")
    labels = metadata["classes"]
    lookup = {normalize_label(label): index for index, label in enumerate(labels)}
    paths = {"raw_probability": result["raw_probabilities_path"], "raw_binary": result["raw_membership_path"],
             "closed_binary": result["closed_membership_path"]}
    summaries = []
    for representation, path in paths.items():
        folder = destination / representation
        folder.mkdir()
        values = load_worker_representation(path, representation, metadata["gaussian_count"], labels, metadata["source_ply_sha256"])
        tensor = torch.from_numpy(values).to(device)
        rows = []
        with torch.no_grad():
            for view, name in enumerate(sorted(ground_truth)):
                for label, truth in ground_truth[name].items():
                    if label in lookup:
                        colors = tensor[:, lookup[label], None].expand(-1, 3).contiguous()
                        score = renderer(cameras[name], colors)[0].detach().cpu().numpy()
                        if not np.isfinite(score).all():
                            raise FloatingPointError("Non-finite semantic render during held-out evaluation")
                    else:
                        score = np.zeros(truth.shape, np.float32)
                    prediction = score >= .5
                    rows.append({"image": name, "label": label, **mask_metrics(prediction, truth)})
                    safe = "".join(character if character.isalnum() else "_" for character in label)
                    Image.fromarray(np.concatenate([truth, prediction], axis=1).astype(np.uint8) * 255).save(folder / f"{view:02d}-{safe}.png")
        per_label = {label: {"views": sum(row["label"] == label for row in rows),
                            "mean_iou": mean_metric([row for row in rows if row["label"] == label], "iou"),
                            "mean_boundary_iou": mean_metric([row for row in rows if row["label"] == label], "boundary_iou")}
                     for label in all_labels}
        groups = {("touching_image_border" if flag else "inside_image"): {
            "pairs": sum(row["gt_touches_image_border"] == flag for row in rows),
            "mean_iou": mean_metric([row for row in rows if row["gt_touches_image_border"] == flag], "iou"),
            "mean_boundary_iou": mean_metric([row for row in rows if row["gt_touches_image_border"] == flag], "boundary_iou")}
            for flag in (False, True)}
        summary = {"experiment_kind": result.get("experiment_kind", FRESH_EXPERIMENT_KIND if getattr(args, "fresh_default", False) else EXPERIMENT_KIND),
                   "representation": representation, "semantic_steps": metadata["semantic_steps"],
                   "class_mean_iou": mean_metric(list(per_label.values()), "mean_iou"), "mean_iou": mean_metric(rows, "iou"),
                   "mean_boundary_iou": mean_metric(rows, "boundary_iou"), "views": 6, "pairs": pairs, "classes": 14,
                   "eval_long_edge": 256, "prediction_threshold": .5, "boundary_width_pixels": 2,
                   "per_label": per_label, "image_border_groups": groups, "per_object_view": rows,
                   "source_artifact": str(path), "source_artifact_sha256": sha(path),
                   "metric": "held-out 2D alpha-composited category mask IoU, not 3D/instance IoU",
                   "closed_warning": "Parent closure is a derived edit field; zero closure violations are not an accuracy metric."}
        write_json(folder / "summary.json", summary)
        summaries.append(summary)
        print(json.dumps({"stage": "evaluation", "representation": representation,
                          "class_mean_iou": summary["class_mean_iou"], "mean_boundary_iou": summary["mean_boundary_iou"]}), flush=True)
        del tensor, values
        gc.collect()
        torch.cuda.empty_cache()
    after = _tensor_hashes(model)
    if after != before or sha(args.ply) != metadata["source_ply_sha256"]:
        raise RuntimeError("Source PLY or frozen tensors changed during evaluation")
    result = {"summaries": summaries, "evaluation_seconds": time.perf_counter() - started,
              "annotations_sha256": sha(args.annotations), "gt_mask_files": gt_file_hashes,
              "evaluation_geometry_unchanged": True, "evaluation_tensor_hashes": after,
              "gt_opened_only_after_worker_returned_completed": True, "loader_reentry_validation": reentry}
    write_json(destination / "summary.json", result)
    return result


def recover_evaluation(args):
    """Evaluate an already completed worker, preserving its original failure log."""
    import torch
    output = Path(args.output).resolve()
    if not output.is_dir() or (output / "complete.json").exists():
        raise ValueError("Evaluation recovery requires an existing incomplete validation directory")
    immutable = [output / name for name in ("run-plan.json", "worker-invocation.json", "worker/catalog.json", "worker/stats.json")]
    immutable += [output / name for name in ("failed.json", "status.json") if (output / name).is_file()]
    original_hashes = {str(path.relative_to(output)): sha(path) for path in immutable}
    manifest = json.loads((output / "run-plan.json").read_text())
    invocation = json.loads((output / "worker-invocation.json").read_text())
    metadata = json.loads((output / "worker/catalog.json").read_text())
    stats = json.loads((output / "worker/stats.json").read_text())
    if (metadata.get("status") != "completed" or not metadata.get("geometry_unchanged") or
            invocation.get("worker_complete") is not True or stats.get("completed_steps") != metadata.get("semantic_steps") or
            metadata.get("source_tensor_hashes") != metadata.get("final_tensor_hashes") or
            invocation.get("final_tensor_hashes") != metadata.get("final_tensor_hashes")):
        raise ValueError("Recovery refuses an incomplete/unverified production worker")
    if sha(args.ply) != metadata["source_ply_sha256"] or sha(args.split) != manifest["split_sha256"] or sha(args.training_masks) != manifest["training_masks_sha256"]:
        raise ValueError("Recovery inputs do not match the completed worker")
    old_code_hashes = manifest["code_sha256"]
    current_hashes = {name: sha(ROOT / name) for name in old_code_hashes}
    deviations = {name: {"during_worker": old_code_hashes[name], "during_recovery": current_hashes[name]}
                  for name in old_code_hashes if old_code_hashes[name] != current_hashes[name]}
    allowed_changes = {"backend/semantic_worker.py", "benchmarks/validate_semantic_worker_cuda.py"}
    if set(deviations) - allowed_changes:
        raise ValueError("Recovery refuses changes outside the worker loader guard and independent validation script")
    recovery_path = output / "recovery.json"
    recovery = json.loads(recovery_path.read_text()) if recovery_path.exists() else {"attempts": []}
    if not isinstance(recovery.get("attempts"), list):
        raise ValueError("Existing recovery audit is malformed")
    ordinal = len(recovery["attempts"]) + 1
    evaluation_dir = output / ("evaluation-recovery" if ordinal == 1 else f"evaluation-recovery-{ordinal}")
    if evaluation_dir.exists():
        raise FileExistsError(evaluation_dir)
    started = time.perf_counter()
    attempt = {"attempt": ordinal, "status": "running", "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
               "evaluation_directory": str(evaluation_dir), "training_repeated": False,
               "original_audit_hashes": original_hashes, "code_deviations": deviations,
               "reason": "Correct legal namespace-package reentry validation; preserve completed worker outputs.",
               "original_semantic_steps": metadata["semantic_steps"],
               "reentry_test": "two native model loads in this process, release first, compare frozen tensors and record namespace paths"}
    recovery["attempts"].append(attempt)
    write_json(recovery_path, recovery)
    try:
        attempt["environment"] = environment(torch)
        prepared = prepare_training_inputs(args)
        if set(prepared["views"]) != set(metadata["training_frames"]):
            raise ValueError("Recovery camera/mask views differ from completed worker")
        if any(sha(item["path"]) != item["sha256"] for item in manifest["training_mask_files"]):
            raise ValueError("Original training masks changed before recovery")
        artifact_keys = {"raw_probabilities_path": "probabilities", "raw_membership_path": "membership", "closed_membership_path": "closed_membership"}
        result = {"status": "completed", "metadata": metadata, "stats": stats, "experiment_kind": manifest["experiment_kind"],
                  "catalog_path": str(output / "worker/catalog.json"), "stats_path": str(output / "worker/stats.json")}
        for result_key, artifact_key in artifact_keys.items():
            artifact = metadata["artifacts"][artifact_key]
            path = (output / "worker" / artifact["path"]).resolve()
            if not path.is_relative_to(output / "worker") or sha(path) != artifact["sha256"]:
                raise ValueError("Completed worker artifact path/hash changed")
            result[result_key] = str(path)
        evaluation = evaluate_worker_outputs(args, prepared, result, evaluation_dir, verify_loader_reentry=True)
        if {name: sha(ROOT / name) for name in current_hashes} != current_hashes:
            raise RuntimeError("Recovery source code changed while evaluation was running")
        if {str(path.relative_to(output)): sha(path) for path in immutable} != original_hashes:
            raise RuntimeError("An original failure/status/worker audit file changed during recovery")
        summary = {"status": "complete", "experiment_kind": manifest["experiment_kind"], "run_plan": manifest,
                   "worker_total_seconds": invocation["worker_total_seconds"], "worker_optimizer_seconds": stats["training_seconds"],
                   "worker_stats": stats, "evaluation_seconds": evaluation["evaluation_seconds"],
                   "recovery_seconds": time.perf_counter() - started, "training_repeated": False,
                   "source_ply_sha256_after": sha(args.ply), "original_failed_status_worker_audits_preserved": True,
                   "code_deviations": deviations, "evaluation": evaluation,
                   "reporting_scope": "Independent production smoke test; original loader failure preserved; recovered evaluation is not an R1/R2 ablation."}
        summary_path = output / ("summary.json" if not (output / "summary.json").exists() else f"summary-recovery-{ordinal}.json")
        if summary_path.exists():
            raise FileExistsError(summary_path)
        write_json(summary_path, summary)
        attempt.update(status="complete", recovery_seconds=time.perf_counter() - started,
                       summary=str(summary_path), loader_reentry_validation=evaluation["loader_reentry_validation"],
                       original_files_preserved=True)
        write_json(recovery_path, recovery)
        write_json(output / "complete.json", {"status": "complete", "complete": True, "recovered_evaluation": True,
            "summary": str(summary_path), "summary_sha256": sha(summary_path), "recovery_audit": str(recovery_path)})
        print(json.dumps({"status": "complete", "recovered_evaluation": True, "training_repeated": False, "summary": str(summary_path)}), flush=True)
        return summary
    except BaseException as exc:
        attempt.update(status="failed", error_type=type(exc).__name__, message=str(exc), traceback=traceback.format_exc(),
                       recovery_seconds=time.perf_counter() - started)
        write_json(recovery_path, recovery)
        raise


def run(args):
    fresh = bool(getattr(args, "fresh_default", False))
    if args.output is None:
        args.output = "/content/semantic-worker-fresh-validation" if fresh else "/content/semantic-worker-cuda-validation"
    if getattr(args, "evaluate_existing", False):
        return recover_evaluation(args)
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()

    def status(phase, **detail):
        value = {"phase": phase, "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                 "wall_seconds": time.perf_counter() - started, **detail}
        write_json(output / "status.json", value)
        print(json.dumps(value, ensure_ascii=False), flush=True)

    try:
        import torch
        from backend.semantic_worker import _ply_description, refine_upstream_semantics
        status("preparing_train_only_inputs")
        runtime = environment(torch)
        code_files = [Path(__file__), ROOT / "backend/semantic_worker.py", ROOT / "backend/semantic_refinement.py",
                      ROOT / "backend/semantic_targets.py", ROOT / "backend/semantics.py",
                      ROOT / "benchmarks/evaluate_cuda_semantics.py", ROOT / "benchmarks/run_benchmark.py"]
        code_hashes = {str(file.relative_to(ROOT)): sha(file) for file in code_files}
        prepared = prepare_training_inputs(args)
        if fresh:
            prior = None
            source_description = _ply_description(Path(args.ply).resolve(), {"input_kind": "full_ply"})
            source_count, source_sha = source_description["gaussian_count"], sha(args.ply)
            semantic_steps, hierarchy, experiment_kind = 1200, [], FRESH_EXPERIMENT_KIND
        else:
            prior = prepare_verified_prior(args.prior_experiment, args.ply, args.split, args.training_masks,
                                           prepared["views"], output / "verified_prior.npz")
            source_count, source_sha = prior["gaussian_count"], prior["source_ply_sha256"]
            semantic_steps, hierarchy, experiment_kind = 400, APPROVED_HIERARCHY, EXPERIMENT_KIND
        worker_config = {"semantic_steps": semantic_steps, "preset": "improved", "train_size": 256, "seed": 0,
                         "upstream_repo": str(Path(args.upstream).resolve()), "expected_gaussian_count": source_count,
                         "source_ply_sha256": source_sha, "approved_hierarchy": hierarchy,
                         "prior": None if prior is None else {key: prior[key] for key in ("path", "source_ply_sha256", "classes")}}
        manifest = {"experiment_kind": experiment_kind, "frozen_R1_R2_results_modified": False,
                    "rgb_training_performed": False, "ground_truth_read_for_optimization": False,
                    "configuration": worker_config, "environment": runtime, "code_sha256": code_hashes,
                    "prior": prior, "split_sha256": sha(args.split), "training_masks_sha256": sha(args.training_masks),
                    "training_views": prepared["views"], "training_mask_count": len(prepared["masks"]),
                    "training_mask_files": prepared["training_mask_files"], "colmap_provenance": prepared["colmap_provenance"],
                    "app_usage_context": {"overall_app_semantic_refinement_default": "projection",
                        "mode_that_requires_user_selection": "cuda_iterative", "fresh_default_conditions": fresh,
                        "initialization": "uniform .05; no full prior" if fresh else "verified full-Ply step-zero prior",
                        "approved_hierarchy": hierarchy, "semantic_steps": semantic_steps},
                    "source_ply_sha256_before": sha(args.ply), "annotations_path_for_later_evaluation": str(args.annotations),
                    "planned_evaluation": {"views": 6, "pairs": 59, "classes": 14, "long_edge": 256,
                                           "representations": ["raw_probability", "raw_binary", "closed_binary"], "threshold": .5},
                    "limitation": "Full-scene COLMAP geometry is transductive; this smoke test does not reproduce end-to-end SfM or R1/R2 timing."}
        write_json(output / "run-plan.json", manifest)
        write_json(output / "training_cameras.json", {"cameras": prepared["cameras"]})
        status("running_production_worker", semantic_steps=semantic_steps, fresh_default=fresh)
        torch.cuda.synchronize()
        worker_start = time.perf_counter()
        result = refine_upstream_semantics(args.ply, prepared["cameras"], prepared["masks"], output / "worker", worker_config,
            progress=lambda fraction, message: status("running_production_worker", progress=fraction, message=message))
        result["experiment_kind"] = experiment_kind
        torch.cuda.synchronize()
        worker_seconds = time.perf_counter() - worker_start
        if result.get("status") != "completed" or not result["metadata"].get("geometry_unchanged"):
            raise RuntimeError("Production worker did not complete with fixed geometry")
        if result["metadata"]["source_tensor_hashes"] != result["metadata"]["final_tensor_hashes"]:
            raise RuntimeError("Worker initial/final tensor hashes differ")
        accepted_hierarchy = {(edge["parent"], edge["child"]) for edge in result["metadata"].get("hierarchy_edges", [])}
        if not accepted_hierarchy.issubset({(edge["parent"], edge["child"]) for edge in hierarchy}):
            raise RuntimeError("Production worker activated an unapproved hierarchy relation")
        write_json(output / "worker-invocation.json", {"worker_complete": True, "worker_total_seconds": worker_seconds,
            "timing_scope": "entire public worker call including model load, validation, normalization, training, checkpoints and export",
            "worker_optimizer_seconds": result["stats"]["training_seconds"], "source_ply_sha256": sha(args.ply),
            "worker_catalog": result["catalog_path"], "worker_stats": result["stats_path"],
            "geometry_unchanged": True, "initial_tensor_hashes": result["metadata"]["source_tensor_hashes"],
            "final_tensor_hashes": result["metadata"]["final_tensor_hashes"], "gt_annotations_opened": False})
        status("worker_complete_starting_heldout_evaluation", worker_total_seconds=worker_seconds)
        gc.collect()
        torch.cuda.empty_cache()
        evaluation = evaluate_worker_outputs(args, prepared, result, output / "evaluation")
        final_hashes = {str(file.relative_to(ROOT)): sha(file) for file in code_files}
        if final_hashes != code_hashes:
            raise RuntimeError("Source code changed during the production validation")
        if sha(args.ply) != manifest["source_ply_sha256_before"] or sha(args.split) != manifest["split_sha256"] or sha(args.training_masks) != manifest["training_masks_sha256"]:
            raise RuntimeError("Source PLY, split, or original mask manifest changed")
        if any(sha(item["path"]) != item["sha256"] for item in prepared["training_mask_files"]):
            raise RuntimeError("Original training mask pixels changed")
        old_prior_files = {} if prior is None else {Path(prior["source_checkpoint"]): prior["source_checkpoint_sha256"],
                           Path(args.prior_experiment) / "experiment.json": prior["source_experiment_sha256"],
                           Path(prior["source_checkpoint"]).parent / "semantic_classes.json": prior["source_classes_sha256"],
                           Path(prior["source_checkpoint"]).parent / "checkpoint.json": prior["source_checkpoint_metadata_sha256"]}
        if any(sha(path) != expected for path, expected in old_prior_files.items()):
            raise RuntimeError("Original frozen step-zero prior/provenance changed")
        summary = {"status": "complete", "experiment_kind": experiment_kind, "run_plan": manifest,
                   "worker_total_seconds": worker_seconds, "worker_optimizer_seconds": result["stats"]["training_seconds"],
                   "evaluation_seconds": evaluation["evaluation_seconds"], "full_run_seconds": time.perf_counter() - started,
                   "full_run_timing_scope": "run() through validation/evaluation/export; excludes earlier Python module import",
                   "worker_stats": result["stats"], "worker_catalog_path": result["catalog_path"],
                   "source_ply_sha256_after": sha(args.ply), "all_original_inputs_unchanged": True,
                   "code_hashes_unchanged": True, "evaluation": evaluation,
                   "reporting_scope": "Independent production validation; distinguish seeded 400-step and uniform fresh 1200-step conditions. Do not combine into the frozen R1/R2 ablation tables."}
        write_json(output / "summary.json", summary)
        write_json(output / "complete.json", {"status": "complete", "complete": True, "experiment_kind": experiment_kind,
                                              "summary": str(output / "summary.json"), "summary_sha256": sha(output / "summary.json")})
        status("complete", summary=str(output / "summary.json"))
        return summary
    except BaseException as exc:
        write_json(output / "failed.json", {"status": "failed", "error_type": type(exc).__name__, "message": str(exc),
                                            "traceback": traceback.format_exc(), "wall_seconds": time.perf_counter() - started})
        status("failed", error_type=type(exc).__name__, message=str(exc))
        raise


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--upstream", default="/content/splat-benchmark/gaussian-splatting")
    result.add_argument("--scene-dir", default="/content/data/lerf-ovs/lerf_ovs/teatime")
    result.add_argument("--split", default="/content/benchmark-results/teatime/split.json")
    result.add_argument("--training-masks", default="/content/splat-benchmark-tools/training_masks.json")
    result.add_argument("--annotations", default="/content/splat-benchmark-tools/annotations.json")
    result.add_argument("--ply", default="/content/benchmark-results/teatime/balanced/point_cloud/iteration_15000/point_cloud.ply")
    result.add_argument("--prior-experiment", default="/content/semantic-v2-results/simple-8")
    result.add_argument("--output", help="New output directory; defaults to the separate seeded or fresh validation directory")
    result.add_argument("--evaluate-existing", action="store_true", help="Recover evaluation from an already completed worker; never repeat training or replace old failed/status audit")
    result.add_argument("--fresh-default", action="store_true", help="Separate actual cuda_iterative defaults: uniform .05, no prior/hierarchy, 1200 steps; overall app default remains projection")
    return result


def main(argv=None):
    args = parser().parse_args(argv)
    try:
        run(args)
    except BaseException as exc:
        print(json.dumps({"status": "failed", "error_type": type(exc).__name__, "message": str(exc)}), file=sys.stderr, flush=True)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
