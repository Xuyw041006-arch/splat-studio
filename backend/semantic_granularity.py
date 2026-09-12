"""Explicit multi-granularity regions and conservative cross-view association.

This NumPy implementation borrows LaGa's separation of 3D association from
multi-view descriptor aggregation. It does NOT reproduce its learned affinity
field or claim that SAM s/m/l proposal scales are verified semantic part levels.
Geometry support is a set of visible IDs in ONE hash-bound Gaussian cloud. Names
never establish identity. Missing observations never become negative targets.
"""
from __future__ import annotations

from collections import defaultdict
import hashlib
import json
from pathlib import Path
import re

import numpy as np

from .semantic_targets import _frame, _read_mask, normalize_label


LEVELS = {"coarse": 0, "object": 1, "part": 2, "l": 0, "m": 1, "s": 2}
_DEFAULTS = {"min_shared_points": 8, "min_support_overlap": .35,
             "min_support_iou": .15, "ambiguity_margin": .08,
             "feature_weight": .15, "min_descriptor_cosine": -.25,
             "min_parent_views": 2, "min_containment": .9,
             "min_parent_area_ratio": 1.2, "max_descriptors": 8,
             "descriptor_split_cosine": .9}


def _identifier(value, name):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a nonempty string")
    return value.strip()


def _unit(value):
    value = np.asarray(value, dtype=np.float64)
    norm = np.linalg.norm(value)
    return value / norm if norm > 1e-12 else np.zeros_like(value)


def _config(config):
    config = dict(config or {})
    unknown = set(config) - set(_DEFAULTS) - {"geometry_binding"}
    if unknown:
        raise ValueError(f"Unknown granularity configuration: {sorted(unknown)}")
    result = {**_DEFAULTS, **config}
    for key in ("min_shared_points", "min_parent_views", "max_descriptors"):
        if type(result[key]) is not int or result[key] < (2 if key == "min_parent_views" else 1):
            raise ValueError(f"Invalid {key}")
    for key in ("min_support_overlap", "min_support_iou", "ambiguity_margin",
                "feature_weight", "min_containment", "descriptor_split_cosine"):
        if not np.isfinite(result[key]) or not 0 <= result[key] <= 1:
            raise ValueError(f"Invalid {key}")
    if result["min_containment"] < .9 or not np.isfinite(result["min_parent_area_ratio"]) or result["min_parent_area_ratio"] < 1:
        raise ValueError("Hierarchy requires containment >= .9 and parent area ratio >= 1")
    if not np.isfinite(result["min_descriptor_cosine"]) or not -1 <= result["min_descriptor_cosine"] <= 1:
        raise ValueError("Invalid min_descriptor_cosine")
    binding = result.get("geometry_binding")
    if binding is not None:
        if not isinstance(binding, dict) or not re.fullmatch(r"[0-9a-f]{64}", str(binding.get("source_ply_sha256", ""))):
            raise ValueError("geometry_binding requires source_ply_sha256")
        if type(binding.get("gaussian_count")) is not int or binding["gaussian_count"] < 1:
            raise ValueError("geometry_binding requires a positive gaussian_count")
    return result


def _support(value, binding):
    raw = np.asarray(value)
    if raw.size == 0:
        return frozenset()
    if raw.ndim != 1 or raw.dtype.kind not in "iu" or raw.dtype.kind == "b" or (raw < 0).any():
        raise ValueError("support_ids must contain nonnegative integer global Gaussian IDs")
    if binding is None:
        raise ValueError("Nonempty support_ids require geometry_binding")
    if (raw >= binding["gaussian_count"]).any():
        raise ValueError("support_ids exceed the bound Gaussian cloud")
    return frozenset(map(int, raw))


def _validated_regions(records, config, mask_loader, support_loader, allowed_frames):
    allowed = None if allowed_frames is None else {str(frame) for frame in allowed_frames}
    regions, keys, grids, spaces = [], set(), {}, {}
    empty = []
    for raw in records:
        frame = _frame(raw)
        # Check the split before touching mask/descriptor assets.
        if allowed is not None and frame not in allowed:
            raise ValueError(f"Region frame {frame!r} is outside the declared training split")
        region_id = _identifier(raw.get("region_id", raw.get("mask_id")), "region_id")
        key = (frame, region_id)
        if key in keys:
            raise ValueError(f"Duplicate region identity: {key}")
        keys.add(key)
        level = raw.get("granularity", raw.get("level"))
        if level not in LEVELS:
            raise ValueError("Explicit granularity must be coarse/object/part or native SAM s/m/l")
        if "granularity" in raw and "level" in raw and raw["level"] != level:
            raise ValueError("granularity and level declarations disagree")
        mask = _read_mask(raw, mask_loader).copy()
        if frame in grids and grids[frame] != mask.shape:
            raise ValueError(f"Region grids disagree in frame {frame!r}")
        grids[frame] = mask.shape
        confidence = float(raw.get("confidence", 1.0))
        if not np.isfinite(confidence) or not 0 <= confidence <= 1:
            raise ValueError("Region confidence must be in [0,1]")
        support = _support(support_loader(raw, mask) if support_loader else raw.get("support_ids", []), config.get("geometry_binding"))
        descriptor, feature_space = None, None
        if raw.get("descriptor") is not None:
            descriptor = np.asarray(raw["descriptor"], dtype=np.float64)
            if descriptor.ndim != 1 or not descriptor.size or not np.isfinite(descriptor).all() or np.linalg.norm(descriptor) < 1e-12:
                raise ValueError("descriptor must be a finite nonzero vector")
            feature_space = _identifier(raw.get("feature_space"), "feature_space")
            if feature_space in spaces and spaces[feature_space] != descriptor.size:
                raise ValueError("Descriptor dimensions disagree within feature_space")
            spaces[feature_space] = descriptor.size
            descriptor = _unit(descriptor)
        parent_id = raw.get("parent_region_id")
        relation = raw.get("relation")
        relation_source = raw.get("relation_source")
        if parent_id is not None:
            parent_id = _identifier(parent_id, "parent_region_id")
            if relation not in {"part_of", "member_of", "contents_of"}:
                raise ValueError("A parent declaration needs an explicit semantic relation")
            relation_source = _identifier(relation_source, "relation_source")
        if not mask.any():
            empty.append({"frame": frame, "region_id": region_id, "reason": "empty_unknown"})
            continue
        regions.append({"key": key, "frame": frame, "region_id": region_id,
                        "label": normalize_label(raw["label"]), "granularity": level,
                        "mask": mask, "area": int(mask.sum()), "confidence": confidence,
                        "support": support, "descriptor": descriptor, "feature_space": feature_space,
                        "parent_region_id": parent_id, "relation": relation, "relation_source": relation_source,
                        "semantic_name_known": raw.get('semantic_name_known',not region_id.startswith('laga-'))})
    return sorted(regions, key=lambda r: r["key"]), empty


def _candidate(a, b, config):
    if a["frame"] == b["frame"] or a["granularity"] != b["granularity"]:
        return None
    shared = len(a["support"] & b["support"])
    if shared < config["min_shared_points"]:
        return None
    overlap = shared / min(len(a["support"]), len(b["support"]))
    iou = shared / len(a["support"] | b["support"])
    if overlap < config["min_support_overlap"] or iou < config["min_support_iou"]:
        return None
    cosine = None
    if a["descriptor"] is not None and b["descriptor"] is not None and a["feature_space"] == b["feature_space"]:
        cosine = float(np.clip(a["descriptor"] @ b["descriptor"], -1, 1))
        if cosine < config["min_descriptor_cosine"]:
            return None
    geometry = .5 * (overlap + iou)
    score = geometry if cosine is None else (1-config["feature_weight"])*geometry + config["feature_weight"]*(cosine+1)/2
    return {"shared_points": shared, "support_overlap": overlap, "support_iou": iou,
            "descriptor_cosine": cosine, "score": score}


def _match(regions, config):
    # Sparse inverted geometry index avoids all region × region comparisons.
    index = defaultdict(list)
    candidate_pairs = set()
    for i, region in enumerate(regions):
        for point in region["support"]:
            key = (region["granularity"], point)
            for j in index[key]:
                if regions[j]["frame"] != region["frame"]:
                    candidate_pairs.add((j, i))
            index[key].append(i)
    candidates, alternatives = {}, defaultdict(list)
    for i, j in sorted(candidate_pairs):
        score = _candidate(regions[i], regions[j], config)
        if score is None:
            continue
        candidates[(i, j)] = score
        alternatives[(i, regions[j]["frame"])].append((score["score"], j))
        alternatives[(j, regions[i]["frame"])].append((score["score"], i))
    best = {}
    for key, choices in alternatives.items():
        choices.sort(key=lambda x: (-x[0], regions[x[1]]["key"]))
        if len(choices) == 1 or choices[0][0] - choices[1][0] > config["ambiguity_margin"]:
            best[key] = choices[0][1]
    parents, frames = list(range(len(regions))), [{r["frame"]} for r in regions]
    def find(i):
        while parents[i] != i:
            parents[i] = parents[parents[i]]
            i = parents[i]
        return i
    accepted, rejected = [], []
    for (i, j), score in sorted(candidates.items(), key=lambda pair: (-pair[1]["score"], pair[0])):
        item = {"a": list(regions[i]["key"]), "b": list(regions[j]["key"]), **score}
        if best.get((i, regions[j]["frame"])) != j or best.get((j, regions[i]["frame"])) != i:
            rejected.append({**item, "reason": "nonmutual_or_ambiguous"})
            continue
        ri, rj = find(i), find(j)
        if ri != rj and frames[ri] & frames[rj]:
            rejected.append({**item, "reason": "transitive_same_view_instance_conflict"})
            continue
        if ri != rj:
            parents[rj] = ri
            frames[ri] |= frames[rj]
        accepted.append(item)
    groups = defaultdict(list)
    for i in range(len(regions)):
        groups[find(i)].append(regions[i])
    return list(groups.values()), accepted, rejected


def aggregate_descriptors(features, *, max_descriptors=8, split_cosine=.9):
    """Deterministic adaptive spherical clusters, not LaGa's silhouette K-means.

    Preserve view-dependent modes. Weights use cluster population, compactness,
    and alignment to the global feature; all modes remain explicitly available.
    """
    vectors = np.asarray(features, dtype=np.float64)
    if vectors.ndim != 2 or not vectors.size or not np.isfinite(vectors).all() or (np.linalg.norm(vectors, axis=1) < 1e-12).any():
        raise ValueError("features must be finite nonzero N×D vectors")
    if type(max_descriptors) is not int or max_descriptors < 1 or not 0 <= split_cosine <= 1:
        raise ValueError("Invalid descriptor aggregation settings")
    vectors = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)
    centers = [vectors[0]]
    while len(centers) < min(max_descriptors, len(vectors)):
        similarities = vectors @ np.stack(centers).T
        furthest = int(np.argmin(similarities.max(axis=1)))
        if similarities[furthest].max() >= split_cosine:
            break
        centers.append(vectors[furthest])
    centers = np.stack(centers)
    for _ in range(20):
        assignment = (vectors @ centers.T).argmax(axis=1)
        updated = []
        for i in range(len(centers)):
            members = vectors[assignment == i]
            mean = members.mean(axis=0) if len(members) else centers[i]
            # Opposite view descriptors can cancel. Preserve an actual direction
            # instead of exporting a zero descriptor that cannot be queried.
            updated.append(_unit(mean) if np.linalg.norm(mean) > 1e-12 else centers[i])
        updated = np.stack(updated)
        if np.allclose(updated, centers, atol=1e-7):
            centers = updated
            break
        centers = updated
    assignment = (vectors @ centers.T).argmax(axis=1)
    global_feature = _unit(vectors.mean(axis=0))
    items = []
    for i, center in enumerate(centers):
        selected = vectors[assignment == i]
        if not len(selected):
            continue
        compactness = float(np.clip((selected @ center).mean(), 0, 1))
        alignment = float(np.clip((center @ global_feature + 1)/2, 0, 1))
        weight = len(selected) * compactness * max(.05, alignment)
        items.append({"descriptor": center.tolist(), "view_count": len(selected),
                      "compactness": compactness, "global_alignment": alignment, "weight": weight})
    total = sum(item["weight"] for item in items)
    for item in items:
        item["weight"] = item["weight"] / total if total else 1 / len(items)
    return {"method": "adaptive_spherical_clusters_global_alignment_compactness",
            "official_laga_reproduction": False, "observation_count": len(vectors), "descriptors": items}


def _hierarchy(regions, identity, config):
    lookup = {r["key"]: r for r in regions}
    requested, invalid = defaultdict(list), []
    for child in regions:
        if child["parent_region_id"] is None:
            continue
        parent = lookup.get((child["frame"], child["parent_region_id"]))
        if parent is None:
            invalid.append({"child_region": list(child["key"]), "reason": "parent_unobserved"})
            continue
        child_scale, parent_scale = child["granularity"], parent["granularity"]
        same_family = (child_scale in {"s", "m", "l"}) == (parent_scale in {"s", "m", "l"})
        if not same_family or LEVELS[parent_scale] >= LEVELS[child_scale]:
            invalid.append({"child_region": list(child["key"]), "reason": "parent_level_not_coarser"})
            continue
        requested[(identity[child["key"]], identity[parent["key"]])].append(child)
    edges, diagnostics = [], []
    by_track = defaultdict(dict)
    for region in regions:
        by_track[identity[region["key"]]][region["frame"]] = region
    qualified = defaultdict(list)
    for (child_id, parent_id), declarations in sorted(requested.items()):
        relations = {row["relation"] for row in declarations}
        declared_frames = {r["frame"] for r in declarations}
        per_view = []
        for frame in sorted(set(by_track[child_id]) & set(by_track[parent_id])):
            child, parent = by_track[child_id][frame], by_track[parent_id][frame]
            containment = float((child["mask"] & parent["mask"]).sum() / child["area"])
            ratio = parent["area"] / child["area"]
            minimum_ratio=1. if child['relation_source']=='observed_scene_collection' and child['relation']=='member_of' else config['min_parent_area_ratio']
            supported = frame in declared_frames and containment >= config["min_containment"] and ratio >= minimum_ratio
            per_view.append({"frame": frame, "containment": containment, "parent_area_ratio": ratio,
                             "explicit_declaration": frame in declared_frames, "supports": supported})
        support = sum(row["supports"] for row in per_view)
        valid = len(relations) == 1 and support >= config["min_parent_views"] and support*2 > len(per_view)
        item = {"child": child_id, "parent": parent_id, "relation": next(iter(relations)) if len(relations) == 1 else "conflicting",
                "source": "explicit_region_relation_and_multiview_containment",
                "relation_sources": sorted({r["relation_source"] for r in declarations}),
                "support_view_count": support, "valid_view_count": len(per_view), "per_view": per_view,
                "semantic_part_verified": False, "instance_relation_verified": False,
                "observation_criteria_met": valid}
        diagnostics.append(item)
        if valid:
            qualified[child_id].append(item)
    for child, candidates in sorted(qualified.items()):
        if len(candidates) == 1:
            edges.append(candidates[0])
        else:
            invalid.append({"child": child, "reason": "ambiguous_parent_tracks"})
    # Strictly decreasing explicit tiers make cycles impossible.
    return {"edges": edges, "diagnostics": diagnostics, "unknown": invalid,
            "meaning": "Explicit region-level relations with repeated training-view containment; proposal size alone never proves a semantic part relation."}


def cross_view_confidence(group, visibility):
    """Leave-one-view-out weighted agreement on co-visible global Gaussian IDs.

    Only frames observing this track are compared. Absent detections are unknown,
    occluded points do not vote, and a single-view region receives uncertainty
    discount rather than being described as cross-view verified.
    """
    result={}
    for region in group:
        positive=eligible=0.;views=0
        for other in group:
            if other['frame']==region['frame']:continue
            co_visible=region['support'] & visibility.get(other['frame'],frozenset())
            if not co_visible:continue
            weight=other['confidence'];eligible+=len(co_visible)*weight
            positive+=len(co_visible & other['support'])*weight;views+=1
        agreement=positive/eligible if eligible>0 else None
        reliability=.35 if agreement is None else .35+.65*agreement
        result[region['key']]={'detector_confidence':region['confidence'],
            'agreement':agreement,'comparison_views':views,
            'effective_confidence':float(region['confidence']*reliability),
            'status':'cross_view_checked' if agreement is not None else 'single_view_or_occluded_unknown',
            'kind':'uncalibrated_detector_times_leave_one_view_out_visible_agreement'}
    return result


def build_granularity_targets(records, config=None, *, mask_loader=None, support_loader=None, allowed_frames=None):
    """Build independently identifiable region tracks compatible with GS targets.

    region_id is local to frame; granularity (or level) is mandatory. support_ids
    must use global, visibility-filtered IDs from config.geometry_binding. An
    optional support_loader(record, mask) can compute these from frozen geometry.
    Parent candidates are per-record parent_region_id/relation/relation_source;
    equal labels and box sizes never create identity or hierarchy automatically.
    """
    config = _config(config)
    records = list(records)
    regions, empty = _validated_regions(records, config, mask_loader, support_loader, allowed_frames)
    groups, matches, rejected = _match(regions, config)
    visibility={}
    for region in regions:
        if region['frame'] not in visibility:
            visibility[region['frame']]=(_support(support_loader({'frame_id':region['frame']},np.ones_like(region['mask'])),config.get('geometry_binding'))
                if support_loader else frozenset().union(*(r['support'] for r in regions if r['frame']==region['frame'])))
    confidence_audit=[]
    identity, classes, per_frame, tracks = {}, [], {}, []
    for group in groups:
        confidence=cross_view_confidence(group,visibility)
        key_bytes = json.dumps([r["key"] for r in group], ensure_ascii=False, separators=(",", ":")).encode()
        track_id = "region-" + hashlib.sha256(key_bytes).hexdigest()[:20]
        labels = sorted({r["label"] for r in group})
        votes = defaultdict(float)
        for region in group:
            votes[region["label"]] += region["confidence"]
        display = sorted(votes, key=lambda label: (-votes[label], label))[0]
        metadata = {"id": track_id, "label": track_id, "display_label": display,
                    "original_labels": labels, "source_labels": labels,
                    "granularity": group[0]["granularity"], "view_count": len(group),
                    "granularity_kind": "proposal_scale" if group[0]["granularity"] in {"s", "m", "l"} else "declared_semantic_role",
                    "identity_source": "mutual_visible_geometry_association" if len(group) > 1 else "unmatched_single_view_region"}
        metadata['cross_view_confidence']=float(np.mean([confidence[r['key']]['effective_confidence'] for r in group]))
        classes.append(metadata)
        banks = defaultdict(list)
        for region in group:
            checked=confidence[region['key']]
            confidence_audit.append({'track_id':track_id,'frame':region['frame'],'region_id':region['region_id'],**checked})
            identity[region["key"]] = track_id
            if region["descriptor"] is not None:
                banks[region["feature_space"]].append(region["descriptor"])
            per_frame.setdefault(region["frame"], {})[track_id] = {
                "mask": region["mask"].copy(), "confidence_map": region["mask"].astype(np.float32)*checked['effective_confidence'],
                "mask_ids": [region["region_id"]], "observed": True, "area": region["area"],
                "confidence": checked['effective_confidence'], "confidence_kind": checked['kind']}
        tracks.append({**metadata, "observations": [{"frame": r["frame"], "region_id": r["region_id"],
                    "support_count": len(r["support"])} for r in group],
                    "descriptor_banks": {space: aggregate_descriptors(values, max_descriptors=config["max_descriptors"],
                        split_cosine=config["descriptor_split_cosine"]) for space, values in sorted(banks.items())}})
    classes.sort(key=lambda value: value["id"])
    hierarchy = _hierarchy(regions, identity, config)
    return {"classes": classes, "class_labels": [c["label"] for c in classes], "per_frame": per_frame,
            "hierarchy": hierarchy, "tracks": sorted(tracks, key=lambda t: t["id"]),
            "matches": matches, "rejected_matches": rejected,
            "diagnostics": {"input_record_count": len(records), "nonempty_record_count": len(regions),
                'cross_view_confidence':{'executed':True,'method':'leave_one_view_out_visible_agreement','observations':confidence_audit},
                "independent_frame_count": len(per_frame), "frame_class_target_count": len(regions),
                "empty_records": empty, "training_frame_whitelist_checked": allowed_frames is not None,
                "geometry_binding": config.get("geometry_binding"), "track_count": len(classes),
                "matching_config": {k: config[k] for k in _DEFAULTS}},
            "warnings": ["Missing/empty detections are unknown; labels never merge separate instances.",
                         "Cross-view IDs are conservative geometric hypotheses, not instance ground truth.",
                         "SAM s/m/l are proposal scales; explicit parent relations still require repeated evidence.",
                         "LaGa-inspired association and descriptor bank; no learned LaGa affinity field is trained."]}


def read_laga_feature_files(segmentation_path, features_path, *, frame_id, feature_space,
                           level_mapping=None):
    """Read official LaGa *_s.npy + *_f.npy arrays without executing repo code.

    Default maps channels 1/2/3 to native s/m/l, omitting channel 0 (default).
    The mapping must be explicit for any alternative layout. Semantic names are
    unknown until assigned by a caller; generated region names carry no meaning.
    """
    frame_id, feature_space = _identifier(frame_id, "frame_id"), _identifier(feature_space, "feature_space")
    seg = np.load(Path(segmentation_path), allow_pickle=False)
    features = np.load(Path(features_path), allow_pickle=False)
    if seg.ndim != 3 or min(seg.shape) < 1 or seg.dtype.kind not in "iuf" or not np.isfinite(seg).all() or not np.equal(seg, np.floor(seg)).all():
        raise ValueError("LaGa segmentation must be finite integer-valued [levels,H,W]")
    if features.ndim != 2 or min(features.shape) < 1 or features.dtype.kind not in "iuf" or not np.isfinite(features).all():
        raise ValueError("LaGa features must be finite [regions,dimensions]")
    if (seg < -1).any() or (seg >= len(features)).any():
        raise ValueError("LaGa segmentation contains out-of-range feature IDs")
    mapping = {1: "s", 2: "m", 3: "l"} if level_mapping is None else dict(level_mapping)
    if not mapping or any(type(channel) is not int or channel < 0 or channel >= len(seg) or level not in LEVELS for channel, level in mapping.items()):
        raise ValueError("Invalid explicit LaGa level mapping")
    if len(set(mapping.values())) != len(mapping):
        raise ValueError("LaGa channels must map to distinct granularities")
    rows = []
    for channel, level in sorted(mapping.items()):
        for index in np.unique(seg[channel]).astype(np.int64):
            if index < 0:
                continue
            if np.linalg.norm(features[index]) < 1e-12:
                raise ValueError("LaGa referenced descriptor is zero")
            region_id = f"laga-{channel}-{index}"
            rows.append({"frame_id": frame_id, "region_id": region_id, "label": region_id,
                         "granularity": level, "mask": seg[channel] == index,
                         "descriptor": features[index].astype(np.float32), "feature_space": feature_space,
                         "source": "official_laga_array_format", "semantic_name_known": False})
    return rows
