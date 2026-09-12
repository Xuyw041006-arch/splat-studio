"""Training-mask targets and conservative, observed containment candidates.

The computation uses NumPy only. Arrays can be supplied directly or by a
mask_loader(record) callback. The optional default PNG loader lazily uses Pillow.
No photographs, GT annotations, detector or language model are read here.

build_semantic_targets(records, config=None, *, mask_loader=None, allowed_frames=None)
returns class metadata, sparse per_frame[frame][label] targets and hierarchy
diagnostics. Missing/empty detections are UNKNOWN, never implicit negative masks.
Relations describe observed mask containment, not verified semantic parts or
separate physical instances. Same-class instances are deliberately unioned.
"""
from __future__ import annotations

from collections import defaultdict
import copy
import hashlib
from pathlib import Path
import re
import unicodedata

import numpy as np


def normalize_label(value):
    """Normalize spelling only; do not turn fragments or synonyms into classes."""
    if not isinstance(value, str):
        raise ValueError("A class label must be a nonempty string")
    text = unicodedata.normalize("NFKC", value).casefold()
    text = text.translate(str.maketrans({c: "-" for c in "‐‑‒–—−"}))
    text = re.sub(r"\s+", " ", text).strip()
    text = re.sub(r"\s*-\s*", "-", text)
    if not text:
        raise ValueError("A class label must be a nonempty string")
    return text


def _basename(value):
    return str(value).replace("\\", "/").rstrip("/").rsplit("/", 1)[-1]


def _frame(record):
    if record.get("image_name") and record.get("image_path") and _basename(record["image_name"]) != _basename(record["image_path"]):
        raise ValueError("image_name and image_path identify different frames")
    if record.get("frame_id") is not None:
        value = str(record["frame_id"]).strip()
    else:
        value = _basename(record.get("image_name") or record.get("image_path") or record.get("frame") or "")
    if not value:
        raise ValueError("Every mask requires frame_id, image_name or image_path")
    return value


def _read_mask(record, loader):
    if loader is not None:
        value = loader(record)
    elif "mask" in record:
        value = record["mask"]
    elif record.get("mask_path"):
        path = Path(record["mask_path"])
        if path.suffix.lower() == ".npy":
            value = np.load(path, allow_pickle=False)
        else:
            from PIL import Image
            with Image.open(path) as image:
                value = np.asarray(image.convert("L"))
    else:
        raise ValueError("A target requires mask pixels, mask_path or a mask_loader")
    array = np.asarray(value)
    if array.ndim != 2 or not array.size or array.dtype.kind not in "buif" or not np.isfinite(array).all():
        raise ValueError("A mask must be a finite, nonempty H×W numeric or boolean array")
    if "mask_value" in record:
        return np.asarray(array == record["mask_value"], dtype=bool)
    threshold = float(record.get("mask_threshold", 0))
    if not np.isfinite(threshold):
        raise ValueError("mask_threshold must be finite")
    return np.asarray(array > threshold, dtype=bool)


def _explicit_relations(value, labels_by_id):
    if value is None:
        return {}
    if isinstance(value, dict):
        pairs = [(child, parent) for child, parents in value.items()
                 for parent in (parents if isinstance(parents, (list, tuple)) else [parents])]
    elif isinstance(value, (list, tuple)):
        pairs = [(edge["child"], edge["parent"]) for edge in value]
    else:
        raise ValueError("explicit_hierarchy must be child:parent mappings or {child,parent} edges")
    result = defaultdict(set)
    for child, parent in pairs:
        child = labels_by_id.get(str(child), normalize_label(child))
        parent = labels_by_id.get(str(parent), normalize_label(parent))
        result[child].add(parent)
    return dict(result)


def _hierarchy(per_frame, labels, explicit, config):
    containment = float(config.get("min_containment", .9))
    area_ratio = float(config.get("min_parent_area_ratio", 1.5))
    minimum_views = int(config.get("min_parent_views", 2))
    if not .9 <= containment <= 1 or not np.isfinite(area_ratio) or area_ratio < 1.5 or minimum_views < 2:
        raise ValueError("Hierarchy requires containment >= .9, parent area ratio >= 1.5 and at least two views")
    observed = {label: {frame for frame, targets in per_frame.items() if label in targets} for label in labels}
    diagnostics, reliable = [], defaultdict(list)
    for child in labels:
        for parent in labels:
            if child == parent:
                continue
            shared = sorted(observed[child] & observed[parent])
            views = []
            for frame in shared:
                small, large = per_frame[frame][child], per_frame[frame][parent]
                inside = int(np.count_nonzero(small["mask"] & large["mask"]))
                fraction = inside / small["area"]
                ratio = large["area"] / small["area"]
                views.append({"frame": frame, "child_area": small["area"], "parent_area": large["area"],
                    "intersection": inside, "containment": fraction, "parent_area_ratio": ratio,
                    "supports": fraction >= containment and ratio >= area_ratio})
            support = sum(v["supports"] for v in views)
            eligible = child not in explicit or parent in explicit[child]
            sufficient = support >= minimum_views and support * 2 > len(views)
            item = {"child": child, "parent": parent, "valid_view_count": len(views),
                "support_view_count": support, "support_fraction": support / len(views) if views else None,
                "mean_containment": float(np.mean([v["containment"] for v in views])) if views else None,
                "min_containment": min((v["containment"] for v in views), default=None),
                "child_observed_view_count": len(observed[child]),
                "parent_unobserved_views": sorted(observed[child] - observed[parent]),
                "per_view": views, "observation_criteria_met": sufficient,
                "eligible_by_explicit_hierarchy": eligible,
                "source": "explicit_and_observed" if child in explicit and parent in explicit[child] else "training_masks"}
            diagnostics.append(item)
            if eligible and sufficient:
                reliable[child].append(item)
    edges, unknown = [], {}
    for child in labels:
        candidates = reliable[child]
        if len(candidates) == 1:
            edges.append({**candidates[0], "relation": "observed_mask_containment",
                          "semantic_part_verified": False, "instance_relation_verified": False})
        else:
            unknown[child] = {"child": child,
                "reason": "ambiguous_parents" if len(candidates) > 1 else "explicit_parent_unsupported" if child in explicit else "insufficient_observed_containment",
                "candidate_parents": [v["parent"] for v in candidates],
                "requested_parents": sorted(explicit.get(child, []))}
    # Remove every edge participating in a cycle; never choose one by iteration order.
    parents = {edge["child"]: edge["parent"] for edge in edges}
    cyclic = set()
    for start in parents:
        order, seen, current = [], {}, start
        while current in parents and current not in seen:
            seen[current] = len(order); order.append(current); current = parents[current]
        if current in seen:
            cyclic.update(order[seen[current]:])
    if cyclic:
        for child in cyclic:
            unknown[child] = {"child": child, "reason": "cycle", "candidate_parents": [parents[child]],
                              "requested_parents": sorted(explicit.get(child, []))}
        edges = [edge for edge in edges if edge["child"] not in cyclic]
    absent_explicit = [{"child": child, "parents": sorted(parents)} for child, parents in explicit.items() if child not in observed]
    return {"edges": edges, "unknown": [unknown[k] for k in sorted(unknown)], "diagnostics": diagnostics,
        "unobserved_explicit_children": absent_explicit,
        "criteria": {"min_containment": containment, "min_parent_area_ratio": area_ratio,
                     "min_supporting_views": minimum_views, "strict_majority_of_coobserved_views": True},
        "meaning": "Training-mask containment candidates only; missing detections are unknown, not counter-evidence. No semantic parts or instances are certified."}


def build_semantic_targets(records, config=None, *, mask_loader=None, allowed_frames=None):
    """Union training detections per independent frame/class without negative filling.

    Set allowed_frames to the actual RGB training frame names/IDs to reject a
    held-out record BEFORE its pixels are loaded. Basenames normalize path versus
    filename references; conflicting concrete image paths require explicit IDs.
    Within a frame every mask must have the same pixel grid; no implicit resize.
    explicit_hierarchy restricts candidates for the specified child, but cannot
    bypass observed containment, majority, view-count or acyclicity requirements.
    """
    config = config or {}
    records = list(records)
    allowed_values = None if allowed_frames is None else list(allowed_frames)
    allowed = None if allowed_values is None else {str(v) for v in allowed_values} | {_basename(v) for v in allowed_values}
    per_frame, shapes, image_refs = {}, {}, defaultdict(set)
    source_labels, empty = defaultdict(set), []
    nonempty = 0
    for index, record in enumerate(records):
        frame = _frame(record)
        if allowed is not None and frame not in allowed:
            raise ValueError(f"Mask frame {frame!r} is outside the declared training split")
        if record.get("frame_id") is None and record.get("image_path"):
            ref = str(record["image_path"]).replace("\\", "/")
            if "/" in ref:
                image_refs[frame].add(ref)
                if len(image_refs[frame]) > 1:
                    raise ValueError(f"Conflicting image paths for frame {frame!r}; supply explicit frame_id")
        label = normalize_label(record.get("label"))
        source_labels[label].add(record["label"])
        mask = _read_mask(record, mask_loader)
        if frame in shapes and shapes[frame] != mask.shape:
            raise ValueError(f"Mask dimensions disagree within frame {frame!r}")
        shapes[frame] = mask.shape
        per_frame.setdefault(frame, {})
        confidence = float(record.get("confidence", 1.0))
        if not np.isfinite(confidence) or not 0 <= confidence <= 1:
            raise ValueError("Mask confidence must be finite and in [0,1]")
        mask_id = str(record.get("mask_id", f"mask-{index}"))
        if not mask.any():
            empty.append({"frame": frame, "label": label, "mask_id": mask_id})
            continue
        nonempty += 1
        target = per_frame[frame].setdefault(label, {"mask": np.zeros(mask.shape, dtype=bool),
            "confidence_map": np.zeros(mask.shape, dtype=np.float32), "mask_ids": [], "observed": True})
        target["mask"] |= mask
        target["confidence_map"][mask] = np.maximum(target["confidence_map"][mask], confidence)
        if mask_id not in target["mask_ids"]:
            target["mask_ids"].append(mask_id)
    for targets in per_frame.values():
        for target in targets.values():
            target["area"] = int(np.count_nonzero(target["mask"]))
            target["confidence"] = float(target["confidence_map"][target["mask"]].mean())
            target["confidence_kind"] = "mean_of_max_detection_confidence_on_union; not calibrated probability"
    labels = sorted(source_labels)
    classes = [{"id": "class-" + hashlib.sha256(label.encode()).hexdigest()[:12], "label": label,
        "source_labels": sorted(source_labels[label]),
        "view_count": sum(label in targets for targets in per_frame.values()),
        "granularity": "category_union"} for label in labels]
    explicit = _explicit_relations(config.get("explicit_hierarchy"), {c["id"]: c["label"] for c in classes})
    hierarchy = _hierarchy(per_frame, labels, explicit, config)
    return {"classes": classes, "class_labels": labels, "per_frame": per_frame, "hierarchy": hierarchy,
        "diagnostics": {"input_record_count": len(records), "nonempty_record_count": nonempty,
            "independent_frame_count": len(per_frame), "frame_class_target_count": sum(map(len, per_frame.values())),
            "empty_records": empty, "training_frame_whitelist_checked": allowed is not None},
        "warnings": ["Absent classes and empty detections are unknown; do not expand this sparse mapping into all-zero class supervision.",
                     "Containment candidates do not prove semantic part relations or distinct instances."]}


def semantic_targets_metadata(targets, *, include_pair_diagnostics=False):
    """Small JSON-safe audit record; never serialize dense mask/confidence arrays.

    Full pair diagnostics are optional. Accepted edges retain their view-level
    evidence even in the compact form. Missing class/frame targets stay absent.
    """
    hierarchy = {k: copy.deepcopy(v) for k, v in targets["hierarchy"].items()
                 if k != "diagnostics" or include_pair_diagnostics}
    return {"format": "semantic_training_targets_v1", "classes": copy.deepcopy(targets["classes"]),
        "class_labels": list(targets["class_labels"]),
        "per_frame": {frame: {label: {"shape": list(value["mask"].shape), "area": value["area"],
            "confidence": value["confidence"], "confidence_kind": value["confidence_kind"],
            "mask_ids": list(value["mask_ids"]), "observed": value["observed"]}
            for label, value in classes.items()} for frame, classes in targets["per_frame"].items()},
        "hierarchy": hierarchy, "diagnostics": copy.deepcopy(targets["diagnostics"]),
        "warnings": list(targets["warnings"]),
        "dense_arrays": "omitted; keep original training masks and target construction configuration"}
