"""Attach full-resolution semantic predictions to an indexed preview on CPU.

No files, models, ground truth, or geometry are read or changed. The caller must
load the membership columns and separately declare their order. A SHA256 is an
identity assertion from the exporter; this adapter cannot hash a PLY it is not
given. Candidate containment is not proof of semantic parts or instances.
"""
from __future__ import annotations

import copy
import hashlib
import re
from collections import deque

import numpy as np


def _integer(value, name, minimum=0):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return int(value)


def _sha(value, name):
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-fA-F]{64}", value) is None:
        raise ValueError(f"{name} must contain an actual 64-character PLY SHA256")
    return value.lower()


def _classes(catalog, declared):
    labels = catalog.get("classes")
    if not isinstance(labels, list) or not labels or any(not isinstance(x, str) or not x.strip() for x in labels):
        raise ValueError("catalog.classes must be a nonempty ordered list of class strings")
    if len(labels) != len(set(labels)):
        raise ValueError("catalog.classes must not contain duplicate labels")
    if declared is None or isinstance(declared, str) or list(declared) != labels:
        raise ValueError("membership_classes column order does not exactly match catalog.classes")
    # Do not normalize, sort, alias, or silently reorder matrix columns here.
    return list(labels)


def _edge(value, count):
    if not isinstance(value, dict):
        raise ValueError("Relation edges must be {parent_idx, child_idx, relation, source} mappings")
    parent = _integer(value.get("parent_idx"), "parent_idx")
    child = _integer(value.get("child_idx"), "child_idx")
    if max(parent, child) >= count or parent == child:
        raise ValueError("Relation indices must be distinct valid class indices")
    relation = value.get("relation")
    source = value.get("source", "unknown")
    if not isinstance(relation, str) or not relation.strip() or not isinstance(source, str) or not source.strip():
        raise ValueError("Every relation needs nonempty relation and source strings")
    return {"parent_idx": parent, "child_idx": child, "relation": relation, "source": source,
            "evidence_kind": "observed_containment_candidate", "semantic_verified": False,
            "instance_verified": False, "containment_verified_by_adapter": False}


def _signature(edge):
    return edge["parent_idx"], edge["child_idx"], edge["relation"], edge["source"]


def _relations(candidates, approved, count):
    # If only approved_edges is supplied, those records are both candidate and
    # explicit approval. Merely supplying relation_edges never activates them.
    approved = list(approved or [])
    candidates = list(candidates if candidates is not None else approved)
    records = {}
    for value in candidates:
        edge = _edge(value, count)
        records[_signature(edge)] = edge
    approved_keys = set()
    for value in approved:
        edge = _edge(value, count)
        key = _signature(edge)
        if key not in records:
            raise ValueError("An approved edge must exactly match a candidate relation and source")
        approved_keys.add(key)
    active = []
    for key, edge in records.items():
        edge["approved_by_caller"] = key in approved_keys
        if edge["approved_by_caller"]:
            active.append(edge)

    parents = [set() for _ in range(count)]
    children = [set() for _ in range(count)]
    for edge in active:
        parent, child = edge["parent_idx"], edge["child_idx"]
        parents[child].add(parent)
        children[parent].add(child)
    indegree = [len(x) for x in parents]
    queue = deque(i for i, degree in enumerate(indegree) if degree == 0)
    order = []
    while queue:
        parent = queue.popleft()
        order.append(parent)
        for child in sorted(children[parent]):
            indegree[child] -= 1
            if indegree[child] == 0:
                queue.append(child)
    if len(order) != count:
        raise ValueError("Approved hierarchy contains a cycle; refusing semantic closure")
    return list(records.values()), active, parents, order


_SEMANTIC_FIELDS = ("object_id", "semantic_ids", "semantic_path", "semantic_confidence",
                    "semantic_view_count", "semantic_candidates", "semantic_ambiguous",
                    "semantic_probabilities", "semantic_confidence_kind", "semantic_raw_ids",
                    "semantic_inherited_ids", "semantic_binding")


def bind_semantic_sidecar(scene, catalog, membership, *, membership_classes,
                          probabilities=None, relation_edges=None, approved_edges=None):
    """Return ``{scene, membership_closed, diagnostics}`` without input mutation.

    membership is binary ndarray[N,C] in membership_classes order, which must
    match catalog.classes exactly. probabilities optionally supplies actual
    ndarray[N,C] values in that SAME declared order, in [0,1]; NaN means unknown.
    It is not thresholded, normalized, calibrated, or propagated to ancestors.

    Both metadata.source_ply_sha256 and source_gaussian_count must match the
    catalog. Every present source_index must be a unique valid integer. Missing
    indices keep their geometry and lose stale semantic assignments; an entirely
    unindexed preview is rejected. Indexed all-zero rows also remain unclassified.

    Only approved_edges participate in an acyclic ancestor closure over the FULL
    matrix, before applying the exact preview indices. The tree uses a parent_id
    only for unique approved parents; multi-parent nodes remain rooted at scene
    with parent_ids plus full DAG metadata. All closed memberships are explicit,
    so renderer ancestor closure and full-model selection have the same meaning.
    Scene-root membership is structural, not a semantic class or observation.
    Old object-based isolation/history is invalidated on this new semantic
    snapshot. Existing per-Gaussian hidden/source/geometry fields are preserved.
    """
    if not isinstance(scene, dict) or not isinstance(catalog, dict):
        raise ValueError("scene and catalog must be mappings")
    meta = scene.get("metadata", {})
    if not isinstance(meta, dict):
        raise ValueError("scene.metadata must be a mapping")
    count = _integer(catalog.get("gaussian_count"), "catalog.gaussian_count", 1)
    if _integer(meta.get("source_gaussian_count"), "metadata.source_gaussian_count", 1) != count:
        raise ValueError("Source Gaussian count differs from the full semantic sidecar")
    ply_hash = _sha(catalog.get("source_ply_sha256"), "catalog.source_ply_sha256")
    if _sha(meta.get("source_ply_sha256"), "metadata.source_ply_sha256") != ply_hash:
        raise ValueError("Source PLY SHA256 differs; equal point counts do not prove identity")
    labels = _classes(catalog, membership_classes)
    shape = (count, len(labels))
    if not isinstance(membership, np.ndarray) or membership.shape != shape or membership.dtype.kind not in "biu":
        raise ValueError(f"membership must be a binary boolean/integer ndarray of shape {shape}")
    if np.any((membership != 0) & (membership != 1)):
        raise ValueError("membership contains nonbinary values")
    if probabilities is not None:
        if not isinstance(probabilities, np.ndarray) or probabilities.shape != shape or probabilities.dtype.kind not in "fiu":
            raise ValueError(f"probabilities must be a numeric ndarray of shape {shape}")
        if np.isinf(probabilities).any() or np.any(probabilities < 0) or np.any(probabilities > 1):
            raise ValueError("probabilities must be in [0,1] or NaN for unknown")
    gaussians = scene.get("gaussians")
    if not isinstance(gaussians, list) or not gaussians:
        raise ValueError("A nonempty indexed Gaussian preview is required")
    indices, rows, seen = [], [], set()
    for row, gaussian in enumerate(gaussians):
        if not isinstance(gaussian, dict):
            raise ValueError("Every preview Gaussian must be a mapping")
        if gaussian.get("source_index") is None:
            continue
        index = _integer(gaussian["source_index"], "source_index")
        if index >= count:
            raise ValueError("source_index is outside the full PLY range")
        if index in seen:
            raise ValueError("Duplicate source_index would bind a full Gaussian more than once")
        seen.add(index)
        indices.append(index)
        rows.append(row)
    if not indices:
        raise ValueError("Preview has no source_index mapping; re-export it from the full PLY")

    candidates, active, parents, order = _relations(relation_edges, approved_edges, len(labels))
    raw = membership.astype(bool, copy=False)
    closed = raw.copy()
    for child in reversed(order):
        for parent in parents[child]:
            closed[:, parent] |= closed[:, child]
    indices = np.asarray(indices, dtype=np.int64)
    raw_counts = raw.sum(axis=0, dtype=np.int64)
    closed_counts = closed.sum(axis=0, dtype=np.int64)
    preview_raw = raw[indices]
    preview_closed = closed[indices]
    preview_counts = preview_closed.sum(axis=0, dtype=np.int64)
    ids = ["semantic-class-" + hashlib.sha256(label.encode("utf-8")).hexdigest()[:16] for label in labels]
    if len(ids) != len(set(ids)):
        raise ValueError("Class ID collision")

    result = copy.deepcopy(scene)
    old_roots = [o for o in scene.get("objects", []) if o.get("level") == "scene" and o.get("parent_id") is None]
    root = copy.deepcopy(old_roots[0]) if len(old_roots) == 1 else {"id": "scene", "label": "场景 · 类别分组", "visible": True}
    root_id = str(root["id"])
    if root_id in ids:
        raise ValueError("Scene root collides with a semantic class ID")
    root.update(id=root_id, level="scene", parent_id=None, children=[], gaussian_count=len(gaussians))
    objects = [root]
    class_metadata={entry.get('label'):entry for entry in catalog.get('class_metadata',[]) if isinstance(entry,dict)}
    # A zero-support query is kept in the catalog, not presented as a found object.
    for index, label in enumerate(labels):
        if not closed_counts[index]:
            continue
        parent_id = ids[next(iter(parents[index]))] if len(parents[index]) == 1 else root_id
        detail=class_metadata.get(label,{})
        granularity=detail.get('granularity','category_union')
        objects.append({"id": ids[index], "label": detail.get('display_label',label), "level": granularity if granularity in {'object','part'} else "object", "class_index": index,
                        'semantic_key':label, 'granularity_kind':detail.get('granularity_kind'),
                        'identity_source':detail.get('identity_source'),
                        'cross_view_confidence':detail.get('cross_view_confidence'),
                        'confidence':detail.get('cross_view_confidence'),
                        'confidence_kind':'cross_view_consistency_weight; not calibrated probability',
                        "semantic_granularity": granularity, "parent_id": parent_id,
                        "parent_ids": [ids[p] for p in sorted(parents[index])], "children": [],
                        "visible": True, "priority": False, "source": "semantic_sidecar_prediction",
                        "semantic_verified": False, "instance_verified": False,
                        "hierarchy_kind": "caller_approved_observed_containment" if parents[index] else "flat_category",
                        "gaussian_count": int(preview_counts[index]),
                        "source_gaussian_count": int(closed_counts[index]),
                        "raw_source_gaussian_count": int(raw_counts[index])})
    object_map = {o["id"]: o for o in objects}
    for obj in objects[1:]:
        object_map[obj["parent_id"]]["children"].append(obj["id"])
    result["objects"] = objects
    for gaussian in result["gaussians"]:
        for key in _SEMANTIC_FIELDS:
            gaussian.pop(key, None)
        gaussian["semantic_ids"] = [root_id]
        gaussian["semantic_binding"] = "unindexed_unknown"
    for row, source_index, raw_row, closed_row in zip(rows, indices, preview_raw, preview_closed):
        gaussian = result["gaussians"][row]
        raw_columns = np.flatnonzero(raw_row)
        closed_columns = np.flatnonzero(closed_row)
        gaussian["semantic_ids"] += [ids[i] for i in closed_columns]
        gaussian["semantic_raw_ids"] = [ids[i] for i in raw_columns]
        gaussian["semantic_inherited_ids"] = [ids[i] for i in closed_columns if not raw_row[i]]
        gaussian["semantic_binding"] = "indexed_prediction" if raw_columns.size else "indexed_unclassified"
        if probabilities is not None:
            # Ancestor membership is structural; its supplied probability remains
            # exactly the supplied value, even when smaller than its child's.
            values = probabilities[source_index]
            gaussian["semantic_probabilities"] = {ids[i]: float(values[i]) if np.isfinite(values[i]) else None
                                                   for i in closed_columns}
            known = [float(values[i]) for i in raw_columns if np.isfinite(values[i])]
            if known:
                gaussian["semantic_confidence"] = max(known)
                gaussian["semantic_confidence_kind"] = "max_supplied_probability_of_raw_memberships; calibration_unknown"

    diagnostics = {"source_ply_sha256": ply_hash, "source_gaussian_count": count,
                   "class_count": len(labels), "class_order_verified": True,
                   "preview_gaussian_count": len(gaussians), "indexed_preview_count": len(indices),
                   "unindexed_preview_count": len(gaussians) - len(indices),
                   "unclassified_indexed_preview_count": int(np.count_nonzero(~preview_raw.any(axis=1))),
                   "raw_membership_count": int(raw_counts.sum()), "closed_membership_count": int(closed_counts.sum()),
                   "added_membership_count": int(closed_counts.sum() - raw_counts.sum()),
                   "changed_gaussian_count": int(np.count_nonzero(np.any(closed != raw, axis=1))),
                   "preview_added_membership_count": int(preview_closed.sum() - preview_raw.sum()),
                   "preview_changed_gaussian_count": int(np.count_nonzero(np.any(preview_closed != preview_raw, axis=1))),
                   "raw_class_counts": raw_counts.tolist(), "closed_class_counts": closed_counts.tolist(),
                   "candidate_edge_count": len(candidates), "approved_edge_count": len(active),
                   "multi_parent_class_indices": [i for i, p in enumerate(parents) if len(p) > 1],
                   "probabilities_supplied": probabilities is not None,
                   "probability_rule": "Supplied values only; NaN/missing stays unknown; no ancestor probability propagation",
                   "closure_rule": "Approved DAG ancestors are added to full membership before applying preview source_index",
                   "geometry_modified": False, "observation_provenance_modified": False}
    out_meta = result.setdefault("metadata", {})
    old_visibility = out_meta.get("editor_visibility")
    diagnostics["old_isolation_invalidated"] = isinstance(old_visibility, dict) and old_visibility.get("isolated_ids") is not None
    if isinstance(old_visibility, dict):
        old_visibility["isolated_ids"] = None
    diagnostics["old_command_history_invalidated"] = bool(out_meta.get("command_history"))
    if "command_history" in out_meta:
        out_meta["command_history"] = []
    out_meta["semantic_sidecar"] = {"catalog": copy.deepcopy(catalog), "diagnostics": diagnostics,
                                    "relation_candidates": candidates, "approved_relations": active,
                                    "warning": "Category unions, not instances. Approved containment is not verified part-of; multi-parent DAG nodes have no arbitrary display-tree parent."}
    out_meta["semantics"] = {"status": "sidecar_prediction_bound", "granularity": "category_union",
                              "confidence": "supplied_probabilities" if probabilities is not None else "unknown",
                              "hierarchy": "caller_approved_containment_dag" if active else "flat_categories",
                              "ground_truth_used_by_adapter": False}
    out_meta["class_catalog"] = [{"index": i, "id": ids[i], "label": label,
                                   "source_count": int(closed_counts[i]), "raw_source_count": int(raw_counts[i]),
                                   "preview_count": int(preview_counts[i])} for i, label in enumerate(labels)]
    return {"scene": result, "membership_closed": closed, "diagnostics": diagnostics}
