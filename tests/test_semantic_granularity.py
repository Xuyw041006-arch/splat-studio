"""Geometry/identity/hierarchy counterexamples, without models or evaluation GT."""
import json
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from backend.semantic_granularity import (aggregate_descriptors,
    build_granularity_targets, read_laga_feature_files)
from backend.semantic_targets import semantic_targets_metadata


BINDING = {"source_ply_sha256": "a"*64, "gaussian_count": 1000}
CONFIG = {"geometry_binding": BINDING, "min_shared_points": 3}


def mask(small=False):
    value = np.zeros((10, 10), bool)
    value[3:5, 3:5] = True
    if not small:
        value[1:8, 1:8] = True
    return value


def row(frame, rid="cup", ids=(1, 2, 3, 4), label="cup", level="object", **extra):
    return {"frame_id": frame, "region_id": rid, "label": label,
            "granularity": level, "mask": mask(level == "part"), "support_ids": list(ids), **extra}


def nested():
    return [r for frame in ("a", "b") for r in (
        row(frame, "whole", range(1, 11)),
        row(frame, "handle", (1, 2, 3), "handle", "part", parent_region_id="whole",
            relation="part_of", relation_source="user_region_manifest"))]


def test_cross_view_same_object_gets_one_track_and_compatible_targets():
    rows = [row("a"), row("b")]
    original = rows[0]["mask"].copy()
    result = build_granularity_targets(rows, CONFIG, allowed_frames=["a", "b"])
    assert len(result["classes"]) == len(result["matches"]) == 1
    assert result["classes"][0]["view_count"] == 2
    assert result["classes"][0]["display_label"] == "cup"
    assert result["tracks"][0]["identity_source"] == "mutual_visible_geometry_association"
    payload = semantic_targets_metadata(result, include_pair_diagnostics=True)
    json.dumps(payload)
    result["per_frame"]["a"][result["class_labels"][0]]["mask"][:] = False
    np.testing.assert_array_equal(rows[0]["mask"], original)


def test_same_name_and_identical_features_do_not_merge_disjoint_objects():
    result = build_granularity_targets([
        row(frame, rid, ids, descriptor=[1, 0], feature_space="clip-test")
        for frame in ("a", "b") for rid, ids in (("left", [1, 2, 3]), ("right", [7, 8, 9]))], CONFIG)
    assert len(result["classes"]) == 2
    assert [item["view_count"] for item in result["classes"]] == [2, 2]
    assert {item["display_label"] for item in result["classes"]} == {"cup"}


def test_label_changes_do_not_break_visible_geometry_identity():
    result = build_granularity_targets([row("a", label="cup"), row("b", label="mug")], CONFIG)
    assert len(result["classes"]) == 1
    assert result["classes"][0]["original_labels"] == ["cup", "mug"]


def test_distinct_levels_never_merge_even_if_geometry_and_name_equal():
    result = build_granularity_targets([row("a"), row("b", level="part")], CONFIG)
    assert len(result["classes"]) == 2
    assert result["matches"] == []


def test_no_geometry_means_unknown_identity_even_with_equal_label_and_features():
    result = build_granularity_targets([row("a", ids=[], descriptor=[1, 0], feature_space="x"),
                                        row("b", ids=[], descriptor=[1, 0], feature_space="x")])
    assert len(result["classes"]) == 2
    assert result["matches"] == []


def test_ambiguous_geometry_is_rejected_instead_of_iteration_order_matching():
    rows = [row("a"), row("b", "left"), row("b", "right")]
    result = build_granularity_targets(rows, CONFIG)
    assert len(result["classes"]) == 3
    assert len(result["rejected_matches"]) == 2
    assert all(m["reason"] == "nonmutual_or_ambiguous" for m in result["rejected_matches"])


def test_transitive_chain_cannot_merge_two_instances_in_same_view():
    rows = [row("a", "left", range(1, 5)), row("a", "right", range(9, 13)),
            row("b", ids=range(1, 9)), row("c", ids=range(5, 13))]
    result = build_granularity_targets(rows, CONFIG)
    assert len(result["classes"]) == 2
    assert len(result["rejected_matches"]) == 1
    assert result["rejected_matches"][0]["reason"] == "transitive_same_view_instance_conflict"
    assert all(len({o["frame"] for o in track["observations"]}) == len(track["observations"])
               for track in result["tracks"])


def test_descriptor_breaks_geometry_tie_without_name_matching():
    rows = [row("a", descriptor=[1, 0], feature_space="clip"),
            row("b", "correct", descriptor=[1, 0], feature_space="clip"),
            row("b", "wrong", descriptor=[0, 1], feature_space="clip")]
    result = build_granularity_targets(rows, {**CONFIG, "ambiguity_margin": .05})
    assert len(result["classes"]) == 2
    assert result["matches"][0]["b"] == ["b", "correct"]


def test_incompatible_feature_spaces_never_compare_dot_products():
    result = build_granularity_targets([row("a", descriptor=[1, 0], feature_space="a"),
                                        row("b", descriptor=[-1, 0, 0], feature_space="b")], CONFIG)
    assert len(result["classes"]) == 1
    assert result["matches"][0]["descriptor_cosine"] is None
    assert len(result["tracks"][0]["descriptor_banks"]) == 2


def test_explicit_two_view_hierarchy_passes_with_stable_instance_tracks():
    result = build_granularity_targets(nested(), CONFIG)
    assert len(result["classes"]) == 2
    edge = result["hierarchy"]["edges"][0]
    assert edge["relation"] == "part_of"
    assert edge["support_view_count"] == 2
    assert edge["child"] != edge["parent"]
    assert not edge["semantic_part_verified"]


def test_containment_alone_never_invents_hierarchy():
    rows = nested()
    for item in rows:
        item.pop("parent_region_id", None)
    result = build_granularity_targets(rows, CONFIG)
    assert result["hierarchy"]["edges"] == []


def test_single_view_parent_declaration_cannot_establish_hierarchy():
    rows = nested()
    rows[-1].pop("parent_region_id")
    result = build_granularity_targets(rows, CONFIG)
    assert result["hierarchy"]["edges"] == []


def test_same_name_distinct_parent_never_confused():
    rows = nested()
    rows += [row(f, "other", range(100, 110)) for f in ("a", "b")]
    result = build_granularity_targets(rows, CONFIG)
    assert len(result["classes"]) == 3
    edge = result["hierarchy"]["edges"][0]
    parent = next(t for t in result["tracks"] if t["id"] == edge["parent"])
    assert {o["region_id"] for o in parent["observations"]} == {"whole"}


def test_relation_conflict_and_bad_containment_do_not_activate_loss():
    rows = nested()
    rows[-1]["relation"] = "contents_of"
    assert build_granularity_targets(rows, CONFIG)["hierarchy"]["edges"] == []
    rows = nested()
    rows[-1]["mask"] = np.roll(rows[-1]["mask"], 5, axis=0)
    assert build_granularity_targets(rows, CONFIG)["hierarchy"]["edges"] == []


def test_sam_proposal_levels_are_not_claimed_semantic_parts():
    rows = [row("a", level="s"), row("b", level="s")]
    result = build_granularity_targets(rows, CONFIG)
    assert result["classes"][0]["granularity_kind"] == "proposal_scale"
    assert not result["hierarchy"]["edges"]


def test_missing_and_empty_are_unknown_no_dense_negatives():
    empty = row("b")
    empty["mask"][:] = False
    result = build_granularity_targets([row("a"), empty], CONFIG)
    assert "b" not in result["per_frame"]
    assert result["diagnostics"]["empty_records"][0]["reason"] == "empty_unknown"


def test_holdout_rejected_before_mask_or_support_loader_touched():
    calls = []
    def forbidden(*args):
        calls.append(args)
        raise AssertionError("test leakage")
    with pytest.raises(ValueError, match="training split"):
        build_granularity_targets([row("test")], CONFIG, allowed_frames=["train"],
                                   mask_loader=forbidden, support_loader=forbidden)
    assert calls == []


def test_support_loader_uses_actual_mask_and_binding():
    captured = []
    def loader(record, value):
        captured.append((record["frame_id"], value.copy()))
        return [1, 2, 3]
    result = build_granularity_targets([row("a", ids=[]), row("b", ids=[])], CONFIG, support_loader=loader)
    assert [frame for frame, _ in captured] == ["a", "b", "a", "b"]
    assert all(mask.shape == (10, 10) for _, mask in captured)
    assert all(mask.all() for _, mask in captured[2:])
    assert all(not mask.all() for _, mask in captured[:2])
    assert len(result["classes"]) == 1


@pytest.mark.parametrize("patch,pattern", [
    ({"granularity": "unknown"}, "Explicit granularity"),
    ({"granularity": None}, "Explicit granularity"),
    ({"level": "part"}, "disagree"),
    ({"region_id": ""}, "region_id"),
    ({"support_ids": [-1, 2, 3]}, "nonnegative"),
    ({"support_ids": [1.2, 2, 3]}, "integer"),
    ({"support_ids": [1000]}, "exceed"),
    ({"confidence": float("nan")}, "confidence"),
    ({"descriptor": [0, 0], "feature_space": "clip"}, "nonzero"),
    ({"descriptor": [1, 0]}, "feature_space"),
    ({"parent_region_id": "parent"}, "semantic relation"),
])
def test_malformed_regions_fail_closed(patch, pattern):
    with pytest.raises(ValueError, match=pattern):
        build_granularity_targets([{**row("a"), **patch}], CONFIG)


def test_geometry_binding_duplicate_identity_and_grid_checks():
    with pytest.raises(ValueError, match="geometry_binding"):
        build_granularity_targets([row("a")])
    with pytest.raises(ValueError, match="Duplicate region"):
        build_granularity_targets([row("a"), row("a")], CONFIG)
    with pytest.raises(ValueError, match="grids disagree"):
        build_granularity_targets([row("a"), row("a", "other", mask=np.ones((11, 10), bool))], CONFIG)
    with pytest.raises(ValueError, match="dimensions disagree"):
        build_granularity_targets([row("a", descriptor=[1, 0], feature_space="clip"),
                                   row("b", descriptor=[1, 0, 0], feature_space="clip")], CONFIG)


def test_track_ids_and_descriptor_banks_reproducible_under_input_permutation():
    rows = [row("a", descriptor=[1, 0], feature_space="clip"),
            row("b", descriptor=[.9, .1], feature_space="clip"),
            row("c", descriptor=[.5, .5], feature_space="clip")]
    a, b = [build_granularity_targets(items, CONFIG) for items in (rows, list(reversed(rows)))]
    assert a["classes"] == b["classes"]
    assert a["tracks"] == b["tracks"]


def test_descriptor_modes_preserved_instead_of_collapsed_mean():
    bank = aggregate_descriptors([[1, 0], [1, .01], [0, 1], [.01, 1]])
    assert len(bank["descriptors"]) == 2
    assert sum(r["weight"] for r in bank["descriptors"]) == pytest.approx(1)
    assert sorted(r["view_count"] for r in bank["descriptors"]) == [2, 2]
    assert not bank["official_laga_reproduction"]
    assert len(aggregate_descriptors([[1, 0]]*5)["descriptors"]) == 1


def test_descriptor_opposite_directions_are_finite():
    bank = aggregate_descriptors([[1, 0], [-1, 0]])
    assert len(bank["descriptors"]) == 2
    assert all(np.isfinite(row["weight"]) for row in bank["descriptors"])
    limited = aggregate_descriptors([[1, 0], [-1, 0]], max_descriptors=1)
    assert np.linalg.norm(limited["descriptors"][0]["descriptor"]) == pytest.approx(1)


def test_laga_native_arrays_preserve_feature_ids_and_proposal_levels(tmp_path):
    seg = np.full((4, 3, 3), -1, np.float32)
    seg[1, 0, 0], seg[2, :2, :2], seg[3] = 2, 4, 6
    feat = np.eye(7, dtype=np.float32)
    np.save(tmp_path/"x_s.npy", seg)
    np.save(tmp_path/"x_f.npy", feat)
    rows = read_laga_feature_files(tmp_path/"x_s.npy", tmp_path/"x_f.npy", frame_id="x", feature_space="clip-checkpoint")
    assert [r["granularity"] for r in rows] == ["s", "m", "l"]
    assert [r["mask"].sum() for r in rows] == [1, 4, 9]
    np.testing.assert_array_equal(rows[0]["descriptor"], feat[2])
    assert all(not r["semantic_name_known"] for r in rows)


@pytest.mark.parametrize("seg,feat,pattern", [
    (np.full((4, 2, 2), .5), np.ones((2, 3)), "integer-valued"),
    (np.full((4, 2, 2), 5), np.ones((2, 3)), "out-of-range"),
    (np.zeros((4, 2, 2)), np.zeros((2, 3)), "zero"),
    (np.zeros((3, 2, 2)), np.ones((2, 3)), "mapping"),
])
def test_laga_array_validation(tmp_path, seg, feat, pattern):
    np.save(tmp_path/"s.npy", seg)
    np.save(tmp_path/"f.npy", feat)
    with pytest.raises(ValueError, match=pattern):
        read_laga_feature_files(tmp_path/"s.npy", tmp_path/"f.npy", frame_id="a", feature_space="clip")
