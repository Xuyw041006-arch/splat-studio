"""Train-only target construction; synthetic masks, no models or GT assets."""
from pathlib import Path
import json
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from backend.semantic_targets import build_semantic_targets, normalize_label, semantic_targets_metadata


def rectangle(y0=2, y1=4, x0=2, x1=4):
    mask = np.zeros((10, 10), dtype=bool)
    mask[y0:y1, x0:x1] = True
    return mask


def row(frame, label, mask=None, confidence=.8, **kwargs):
    return {"image_name": frame, "label": label, "mask": rectangle() if mask is None else mask,
            "confidence": confidence, **kwargs}


def nested(frames=("f1", "f2")):
    return [record for frame in frames for record in
            [row(frame, "small"), row(frame, "large", rectangle(1, 6, 1, 6))]]


def edge(result, child):
    return next((e for e in result["hierarchy"]["edges"] if e["child"] == child), None)


def unknown(result, child):
    return next(e for e in result["hierarchy"]["unknown"] if e["child"] == child)


@pytest.mark.parametrize("label", ["  DALL - E   BRAND ", "dall‐e brand", "Dall – E\tbrand", "dall-e brand"])
def test_text_normalization_without_semantic_aliases(label):
    assert normalize_label(label) == "dall-e brand"
    assert normalize_label("coffee mug") != normalize_label("coffee")
    assert normalize_label("##l - e") == "##l-e"
    assert normalize_label("bag dall - e brand") == "bag dall-e brand"


def test_union_and_confidence_max_do_not_count_duplicate_views_or_mutate_input():
    first, second = rectangle(), rectangle(3, 5, 3, 5)
    before = first.copy()
    records = [row("f1", "Dall - e brand", first, .4, mask_id="a"),
               row("f1", "dall-e brand", second, .9, mask_id="b"),
               row("f1", "dall-e brand", second, .9, mask_id="b")]
    result = build_semantic_targets(records)
    target = result["per_frame"]["f1"]["dall-e brand"]
    np.testing.assert_array_equal(target["mask"], first | second)
    assert target["confidence_map"][3, 3] == pytest.approx(.9)
    assert target["confidence_map"][2, 2] == pytest.approx(.4)
    assert target["confidence_map"][0, 0] == 0
    assert target["mask_ids"] == ["a", "b"]
    assert result["classes"][0]["view_count"] == 1
    assert result["diagnostics"]["frame_class_target_count"] == 1
    assert target["confidence"] == pytest.approx((3*.4 + 4*.9)/7)
    target["mask"][:] = False
    np.testing.assert_array_equal(first, before)
    assert records[0]["label"] == "Dall - e brand"


def test_missing_and_empty_detections_never_become_negative_targets():
    result = build_semantic_targets([row("f1", "apple"), row("f2", "cup"), row("f2", "apple", np.zeros((10, 10)))])
    assert "apple" not in result["per_frame"]["f2"]
    assert "cup" not in result["per_frame"]["f1"]
    assert result["diagnostics"]["empty_records"][0]["label"] == "apple"
    assert next(c for c in result["classes"] if c["label"] == "apple")["view_count"] == 1


def test_two_independent_views_support_only_a_containment_candidate():
    result = build_semantic_targets(nested())
    candidate = edge(result, "small")
    assert candidate["parent"] == "large"
    assert candidate["valid_view_count"] == candidate["support_view_count"] == 2
    assert candidate["min_containment"] == candidate["support_fraction"] == 1
    assert candidate["relation"] == "observed_mask_containment"
    assert candidate["semantic_part_verified"] is False
    assert candidate["instance_relation_verified"] is False
    assert all(v["parent_area_ratio"] >= 1.5 for v in candidate["per_view"])


def test_repeating_one_frame_cannot_establish_multiview_hierarchy():
    records = nested(("f1",)) * 8
    result = build_semantic_targets(records)
    assert edge(result, "small") is None
    diagnostic = next(d for d in result["hierarchy"]["diagnostics"] if d["child"] == "small" and d["parent"] == "large")
    assert diagnostic["support_view_count"] == diagnostic["valid_view_count"] == 1


def test_absent_parent_is_unknown_but_observed_disagreement_counts():
    result = build_semantic_targets(nested() + [row("f3", "small")])
    candidate = edge(result, "small")
    assert candidate["valid_view_count"] == 2
    assert candidate["parent_unobserved_views"] == ["f3"]
    bad = [row("f3", "small"), row("f3", "large", rectangle(6, 10, 6, 10))]
    result = build_semantic_targets(nested() + bad)
    assert edge(result, "small")["support_fraction"] == pytest.approx(2/3)
    result = build_semantic_targets(nested() + bad + [row("f4", "small"), row("f4", "large", rectangle(6, 10, 6, 10))])
    assert edge(result, "small") is None  # 2/4 is not a strict majority.


def test_area_and_containment_both_required():
    equal = [r for f in ("f1", "f2") for r in [row(f, "small"), row(f, "large")]]
    assert edge(build_semantic_targets(equal), "small") is None
    shifted = [r for f in ("f1", "f2") for r in [row(f, "small"), row(f, "large", rectangle(3, 8, 3, 8))]]
    assert edge(build_semantic_targets(shifted), "small") is None


def test_ambiguous_parents_unknown_unless_user_explicitly_restricts_candidate():
    records = nested() + [row(f, "also large", rectangle(0, 8, 0, 8)) for f in ("f1", "f2")]
    result = build_semantic_targets(records)
    assert edge(result, "small") is None
    assert unknown(result, "small")["reason"] == "ambiguous_parents"
    result = build_semantic_targets(records, {"explicit_hierarchy": {"small": "large"}})
    assert edge(result, "small")["parent"] == "large"
    assert edge(result, "small")["source"] == "explicit_and_observed"


def test_explicit_parent_still_requires_observed_support():
    records = [r for f in ("f1", "f2") for r in [row(f, "bear nose"), row(f, "stuffed bear", rectangle(6, 10, 6, 10))]]
    result = build_semantic_targets(records, {"explicit_hierarchy": [{"child": "bear nose", "parent": "stuffed bear"}]})
    assert edge(result, "bear nose") is None
    assert unknown(result, "bear nose")["reason"] == "explicit_parent_unsupported"
    result = build_semantic_targets(nested(), {"explicit_hierarchy": {"small": "unobserved parent"}})
    assert edge(result, "small") is None
    assert "unobserved parent" not in result["class_labels"]


def test_cross_view_cycles_are_all_removed():
    records = []
    for child, parent, prefix in [("a", "b", "ab"), ("b", "c", "bc"), ("c", "a", "ca")]:
        for i in range(2):
            records += [row(prefix+str(i), child), row(prefix+str(i), parent, rectangle(1, 6, 1, 6))]
    result = build_semantic_targets(records)
    assert result["hierarchy"]["edges"] == []
    assert {x["reason"] for x in result["hierarchy"]["unknown"]} == {"cycle"}


def test_reject_heldout_frame_before_loading_any_pixels():
    calls = []
    def loader(record):
        calls.append(record)
        raise AssertionError("Held-out pixels must not be read")
    with pytest.raises(ValueError, match="training split"):
        build_semantic_targets([{"image_name": "held-out.jpg", "label": "bear"}], mask_loader=loader, allowed_frames=["train.jpg"])
    assert calls == []


def test_frame_aliases_merge_but_conflicting_concrete_paths_are_rejected():
    records = [row("f1.jpg", "apple", image_path="/images/f1.jpg"), {"image_path": "/images/f1.jpg", "label": "apple", "mask": rectangle()}]
    assert build_semantic_targets(records)["classes"][0]["view_count"] == 1
    with pytest.raises(ValueError, match="Conflicting image paths"):
        build_semantic_targets(records + [row("f1.jpg", "apple", image_path="/other/f1.jpg")])
    explicit = [{**record, "frame_id": str(i)} for i, record in enumerate(records)]
    assert build_semantic_targets(explicit)["classes"][0]["view_count"] == 2


def test_shape_mismatch_and_invalid_confidence_fail_without_implicit_resize():
    with pytest.raises(ValueError, match="dimensions"):
        build_semantic_targets([row("f1", "a"), row("f1", "b", np.ones((4, 4)))])
    with pytest.raises(ValueError, match="confidence"):
        build_semantic_targets([row("f1", "a", confidence=float("nan"))])


def test_frame_whitelist_generator_and_conflicting_name_cannot_hide_heldout_path():
    result = build_semantic_targets([row("train.jpg", "a")], allowed_frames=(x for x in ["/images/train.jpg"]))
    assert result["diagnostics"]["training_frame_whitelist_checked"] is True
    with pytest.raises(ValueError, match="different frames"):
        build_semantic_targets([row("train.jpg", "a", image_path="/images/held-out.jpg")], allowed_frames=["train.jpg"])


def test_thresholds_are_inclusive_but_may_not_be_weakened_below_required_evidence():
    child = np.zeros((1, 20), bool); child[0, :10] = True
    parent = np.zeros((1, 20), bool); parent[0, :9] = True; parent[0, 10:16] = True
    records = [r for f in ("a", "b") for r in [row(f, "small", child), row(f, "large", parent)]]
    assert edge(build_semantic_targets(records), "small")["min_containment"] == .9
    for config in [{"min_containment": .89}, {"min_parent_area_ratio": 1.49}, {"min_parent_views": 1}]:
        with pytest.raises(ValueError, match="Hierarchy requires"):
            build_semantic_targets(records, config)


def test_numpy_loader_and_stable_class_ids(tmp_path):
    path = tmp_path/"mask.npy"; np.save(path, rectangle())
    result = build_semantic_targets([{"image_name": "f", "label": "large", "mask_path": str(path)}])
    assert result["per_frame"]["f"]["large"]["area"] == 4
    first = build_semantic_targets(nested())["classes"]
    reverse = build_semantic_targets(list(reversed(nested())))["classes"]
    assert first == reverse


def test_compact_json_audit_is_serializable_without_dense_masks_or_negative_filling():
    targets = build_semantic_targets(nested() + [row("f3", "small")])
    metadata = semantic_targets_metadata(targets)
    restored = json.loads(json.dumps(metadata, allow_nan=False))
    assert "diagnostics" not in restored["hierarchy"]
    assert restored["per_frame"]["f1"]["small"]["shape"] == [10, 10]
    assert "mask" not in restored["per_frame"]["f1"]["small"]
    assert "confidence_map" not in restored["per_frame"]["f1"]["small"]
    assert "large" not in restored["per_frame"]["f3"]
    restored["classes"][0]["label"] = "changed"
    assert targets["classes"][0]["label"] != "changed"
    complete = semantic_targets_metadata(targets, include_pair_diagnostics=True)
    assert complete["hierarchy"]["diagnostics"]
    json.dumps(complete, allow_nan=False)
