#!/usr/bin/env python3
"""Import official LaGa region arrays into the multilevel semantic worker.

Masks stay in their original full-frame pixel grid. No detector/model is run.
An explicit training split is checked before any NumPy source is opened.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys
import tempfile

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from backend.semantic_granularity import LEVELS


MAX_SOURCE_BYTES = 512 * 1024**2
MAX_PIXELS = 16_777_216
MAX_REGIONS = 10_000
MAX_FEATURE_DIM = 4096
MAX_FRAMES = 1000


def sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _text(value, name):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a nonempty string")
    return value.strip()


def _frame_list(value, name):
    if not isinstance(value, list) or any(not isinstance(v, str) or not v.strip() for v in value):
        raise ValueError(f"{name} must be a list of explicit frame IDs")
    values = [_text(v, name) for v in value]
    if len(values) != len(set(values)):
        raise ValueError(f"{name} contains duplicate frame IDs")
    return values


def _array(path):
    path = Path(path).resolve(strict=True)
    if not path.is_file() or path.suffix.lower() != ".npy":
        raise ValueError("Source must be a NumPy .npy file")
    if path.stat().st_size > MAX_SOURCE_BYTES:
        raise ValueError("A LaGa source array exceeds the 512 MiB limit")
    # mmap examines shape/dtype before allocating a dense array. allow_pickle is
    # always false; an object array cannot execute code or allocate object graphs.
    value = np.load(path, allow_pickle=False, mmap_mode="r")
    if value.dtype.kind not in "iuf" or value.nbytes > MAX_SOURCE_BYTES:
        raise ValueError("Source must be a bounded numeric NumPy array")
    return path, value


def _dimensions(row, shape):
    width, height = row.get("image_width"), row.get("image_height")
    if any(type(v) is not int or not 1 <= v <= 65536 for v in (width, height)):
        raise ValueError("Each frame requires integer image_width and image_height")
    if row.get("coordinate_space") != "full_frame":
        raise ValueError("Each source must explicitly declare coordinate_space=full_frame")
    for key in ("crop", "crop_box", "roi", "is_cropped"):
        if row.get(key) is not None and row.get(key) is not False:
            raise ValueError("Cropped/ROI sources are unsupported")
    sh, sw = shape
    lower = max((sw-.5)/width, (sh-.5)/height)
    upper = min((sw+.5)/width, (sh+.5)/height)
    if lower > upper+1e-12:
        raise ValueError("LaGa grid does not share the declared full image aspect ratio")
    return width, height


def _mapping(raw, channels):
    raw = {1: "s", 2: "m", 3: "l"} if raw is None else raw
    if not isinstance(raw, dict) or not raw:
        raise ValueError("level_mapping must map channel indices to explicit granularities")
    result = {}
    for channel, level in raw.items():
        if isinstance(channel, str) and channel.isdecimal():
            channel = int(channel)
        if type(channel) is not int or not 0 <= channel < channels or level not in LEVELS:
            raise ValueError("Invalid level_mapping channel or granularity")
        if channel in result:
            raise ValueError("Duplicate level_mapping channel")
        result[channel] = level
    if len(result.values()) != len(set(result.values())):
        raise ValueError("Separate channels must retain separate granularities")
    return result


def import_regions(manifest, output_dir, *, base_dir="."):
    """Import each selected training frame atomically; returns the output summary.

    manifest: {training_frames, held_out_frames, feature_space, frames:[{
      frame_id, segmentation_path, features_path, image_width, image_height,
      coordinate_space:'full_frame', optional region_metadata, level_mapping}]}.
    region_metadata maps generated local IDs ('laga-1-42') to {label,
    parent_region_id, relation, relation_source}; labels never establish identity.
    Held-out rows are recorded as skipped BEFORE opening their array files.
    """
    if not isinstance(manifest, dict):
        raise ValueError("Source manifest must be a JSON object")
    train = _frame_list(manifest.get("training_frames"), "training_frames")
    held_out = _frame_list(manifest.get("held_out_frames", []), "held_out_frames")
    if not train or set(train) & set(held_out):
        raise ValueError("Training frames must be nonempty and disjoint from held-out frames")
    frames = manifest.get("frames")
    if not isinstance(frames, list) or not 1 <= len(frames) <= MAX_FRAMES:
        raise ValueError("Source manifest needs 1–1000 frame entries")
    seen, selected, skipped = set(), [], []
    # Validate ALL frame identities before the first source array is touched.
    for row in frames:
        if not isinstance(row, dict):
            raise ValueError("Each source frame must be an object")
        frame = _text(row.get("frame_id"), "frame_id")
        if frame in seen:
            raise ValueError("Duplicate frame entry in source manifest")
        seen.add(frame)
        if frame in held_out:
            skipped.append({"frame_id": frame, "reason": "held_out_not_loaded"})
        elif frame not in train:
            raise ValueError(f"Frame {frame!r} is outside the declared training split")
        else:
            selected.append((frame, row))
    missing = set(train) - seen
    if missing:
        raise ValueError(f"Missing source entries for training frames: {sorted(missing)}")
    base, output = Path(base_dir).resolve(), Path(output_dir).resolve()
    if output.exists():
        raise FileExistsError("Choose a new output directory; existing results are preserved")
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}-", dir=output.parent))
    records, audits = [], []
    try:
        (staging/"masks").mkdir()
        for frame_index, (frame, row) in enumerate(selected):
            feature_space = _text(row.get("feature_space", manifest.get("feature_space")), "feature_space")
            seg_path, seg = _array(base/_text(row.get("segmentation_path"), "segmentation_path"))
            feature_path, features = _array(base/_text(row.get("features_path"), "features_path"))
            if seg.ndim != 3 or min(seg.shape) < 1 or seg.shape[0] > 16 or seg.shape[1]*seg.shape[2] > MAX_PIXELS:
                raise ValueError("Segmentation must be bounded [levels,H,W]")
            if features.ndim != 2 or not 1 <= features.shape[0] <= MAX_REGIONS or not 1 <= features.shape[1] <= MAX_FEATURE_DIM:
                raise ValueError("Features must be bounded [regions,dimensions]")
            if not np.isfinite(features).all():
                raise ValueError("Feature array contains nonfinite descriptors")
            width, height = _dimensions(row, seg.shape[1:])
            mapping = _mapping(row.get("level_mapping", manifest.get("level_mapping")), len(seg))
            seg_sha, features_sha = sha256(seg_path), sha256(feature_path)
            metadata = row.get("region_metadata", {})
            if not isinstance(metadata, dict):
                raise ValueError("region_metadata must map local region IDs to declarations")
            source_records, source_ids = [], set()
            for channel, level in sorted(mapping.items()):
                plane = seg[channel]
                if not np.isfinite(plane).all() or not np.equal(plane, np.floor(plane)).all() or (plane < -1).any() or (plane >= len(features)).any():
                    raise ValueError("Segmentation contains invalid feature indices")
                indices = np.unique(plane).astype(np.int64)
                for feature_id in indices[indices >= 0]:
                    descriptor = np.asarray(features[feature_id], np.float32)
                    if np.linalg.norm(descriptor) < 1e-12:
                        raise ValueError("Referenced descriptor must be nonzero")
                    region_id = f"laga-{channel}-{feature_id}"
                    info = metadata.get(region_id, {})
                    if isinstance(info, str):
                        info = {"label": info}
                    if not isinstance(info, dict):
                        raise ValueError("Region declaration must be a label or object")
                    label = _text(info.get("label", region_id), "label")
                    filename = f"masks/{frame_index:04d}-{channel:02d}-{feature_id:06d}.png"
                    # Stream one region at a time rather than materializing all
                    # N masks in RAM. Original pixels and aspect are unchanged.
                    pixels = np.asarray(plane == feature_id, np.uint8)*255
                    Image.fromarray(pixels).save(staging/filename)
                    record = {"frame_id": frame, "region_id": region_id, "mask_id": region_id,
                        "label": label, "granularity": level, "mask_path": filename,
                        "coordinate_space": "full_frame", "full_frame": True,
                        "image_width": width, "image_height": height,
                        "descriptor": descriptor.tolist(), "feature_space": feature_space,
                        "annotation_source": "imported_laga_sam_clip_arrays",
                        "semantic_name_known": "label" in info,
                        "confidence": 1.0,
                        "confidence_source": "unknown; LaGa arrays do not contain detector confidence",
                        "source_segmentation_sha256": seg_sha, "source_features_sha256": features_sha,
                        "source_feature_index": int(feature_id), "source_channel": channel,
                        "mask_sha256": sha256(staging/filename)}
                    if info.get("parent_region_id") is not None:
                        record["parent_region_id"] = _text(info["parent_region_id"], "parent_region_id")
                        if info.get("relation") not in {"part_of", "member_of", "contents_of"}:
                            raise ValueError("Parent declarations require a semantic relation")
                        record["relation"] = info["relation"]
                        record["relation_source"] = _text(info.get("relation_source"), "relation_source")
                    source_records.append(record)
                    source_ids.add(region_id)
            if set(metadata) - source_ids:
                raise ValueError("Region metadata references absent regions or omitted channels")
            if any(r.get("parent_region_id") and r["parent_region_id"] not in source_ids for r in source_records):
                raise ValueError("Declared parent region is absent from its source frame")
            if not source_records:
                raise ValueError(f"Training frame {frame!r} has no nonempty selected regions")
            if sha256(seg_path) != seg_sha or sha256(feature_path) != features_sha:
                raise ValueError("Source arrays changed during import")
            records.extend(source_records)
            audits.append({"frame_id": frame, "segmentation_path": str(seg_path),
                           "features_path": str(feature_path), "segmentation_sha256": seg_sha,
                           "features_sha256": features_sha, "feature_space": feature_space,
                           "array_grid": list(seg.shape), "original_image_grid": [height, width],
                           "feature_shape": list(features.shape), "region_count": len(source_records),
                           "level_mapping": mapping, "crop_applied": False, "resize_applied": False})
        audit = {"format": "laga_region_import_v1", "training_frames": train, "held_out_frames": held_out,
                 "frames": audits, "skipped_frames": skipped, "region_count": len(records),
                 "geometry_support": "Computed from the bound frozen Gaussian model by semantic_worker; never trusted from imported arrays.",
                 "limits": ["s/m/l are segmentation proposal scales, not certified semantic part roles.",
                            "Full-frame extent is caller-declared; an undeclared same-aspect crop cannot be detected from arrays alone.",
                            "No learned LaGa affinity field or CLIP encoder is trained by this importer."]}
        worker_config = {"semantic_granularity": "multilevel", "sampling_schedule": "view_cycle",
                         "minimum_sweeps": 1, "training_frames": train, "held_out_frames": held_out}
        outputs = {"masks.json": {"format": "semantic_multilevel_regions_v1", "masks": records,
                                  "training_frames": train, "held_out_frames": held_out},
                   "import-audit.json": audit, "worker-config.json": worker_config}
        for name, data in outputs.items():
            (staging/name).write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False)+"\n", encoding="utf-8")
        staging.rename(output)
        return {"output": str(output), "mask_manifest": str(output/"masks.json"),
                "worker_config": str(output/"worker-config.json"), "training_frame_count": len(selected),
                "region_count": len(records), "held_out_frame_count_skipped": len(skipped)}
    except BaseException:
        shutil.rmtree(staging)
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", help="Multi-frame JSON source manifest with explicit training/held-out splits")
    parser.add_argument("--segmentation", help="Single frame official *_s.npy")
    parser.add_argument("--features", help="Single frame official *_f.npy")
    parser.add_argument("--frame", help="Exact registered camera frame ID")
    parser.add_argument("--feature-space", help="Encoder/checkpoint identity shared by these feature vectors")
    parser.add_argument("--image-width", type=int, help="Width of the corresponding full photograph")
    parser.add_argument("--image-height", type=int, help="Height of the corresponding full photograph")
    parser.add_argument("--output", required=True, help="New output directory")
    args = parser.parse_args(argv)
    single_values = (args.segmentation, args.features, args.frame, args.feature_space, args.image_width, args.image_height)
    if args.manifest:
        if any(v is not None for v in single_values):
            parser.error("--manifest cannot be combined with single-frame options")
        path = Path(args.manifest).resolve(strict=True)
        if path.stat().st_size > 20*1024**2:
            parser.error("Source JSON manifest exceeds 20 MiB")
        manifest, base = json.loads(path.read_text(encoding="utf-8")), path.parent
    else:
        if any(v is None for v in single_values):
            parser.error("Single-frame mode needs --segmentation --features --frame --feature-space --image-width --image-height")
        manifest = {"training_frames": [args.frame], "held_out_frames": [], "feature_space": args.feature_space,
                    "frames": [{"frame_id": args.frame, "segmentation_path": args.segmentation,
                                "features_path": args.features, "image_width": args.image_width,
                                "image_height": args.image_height, "coordinate_space": "full_frame"}]}
        base = Path.cwd()
    result = import_regions(manifest, args.output, base_dir=base)
    print(json.dumps(result, ensure_ascii=False))
    return result


if __name__ == "__main__":
    main()
