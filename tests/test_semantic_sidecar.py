import copy
import json

import numpy as np
import pytest

from backend.commands import apply_command
from backend.semantic_sidecar import bind_semantic_sidecar


def fixture():
    scene = {"gaussians": [
        {"position": [i, 2., 3.], "scale": [.1, .2, .3], "rotation": [1., 0., 0., 0.],
         "color": [.2, .3, .4], "sh": [1., -2., 3.], "opacity": .7, "source_index": i,
         "source": "unknown", "confidence": .24, "hidden": False}
        for i in [3, 0, 2]],
        "objects": [{"id": "scene", "label": "CUDA preview", "level": "scene", "parent_id": None, "visible": True}],
        "cameras": [{"matrix": [1., 2., 3.]}],
        "metadata": {"source_gaussian_count": 4, "source_ply_sha256": "a" * 64,
                     "backend": "CUDA", "preview_sh_degree": 0, "training_seconds": 123.4}}
    catalog = {"classes": ["body", "hand", "finger", "other"], "gaussian_count": 4,
               "source_ply_sha256": "a" * 64}
    membership = np.array([[0, 0, 1, 1], [0, 1, 0, 0], [0, 0, 0, 0], [1, 0, 0, 0]], dtype=bool)
    return scene, catalog, membership


def bind(scene, catalog, membership, **kwargs):
    return bind_semantic_sidecar(scene, catalog, membership, membership_classes=catalog["classes"], **kwargs)


def edge(parent, child, relation="part_of"):
    return {"parent_idx": parent, "child_idx": child, "relation": relation, "source": "train_mask_containment"}


def ids(result):
    return {c["label"]: c["id"] for c in result["scene"]["metadata"]["class_catalog"]}


@pytest.mark.parametrize("change", ["hash", "missing_hash", "invalid_hash", "count"])
def test_rejects_ambiguous_source_even_with_equal_point_count(change):
    scene, catalog, membership = fixture()
    if change == "hash":
        catalog["source_ply_sha256"] = "b" * 64
    elif change == "missing_hash":
        del catalog["source_ply_sha256"]
    elif change == "invalid_hash":
        catalog["source_ply_sha256"] = "not-a-hash"
    else:
        scene["metadata"]["source_gaussian_count"] = 5
    with pytest.raises(ValueError):
        bind(scene, catalog, membership)


@pytest.mark.parametrize("index", [3, -1, 4, True, 1.0, "1"])
def test_rejects_duplicate_out_of_range_or_noninteger_indices(index):
    scene, catalog, membership = fixture()
    scene["gaussians"][1]["source_index"] = index
    with pytest.raises(ValueError):
        bind(scene, catalog, membership)


def test_rejects_all_unindexed_but_preserves_partial_unknown_without_stale_classes():
    scene, catalog, membership = fixture()
    scene["gaussians"][1].pop("source_index")
    scene["gaussians"][1].update(object_id="old", semantic_ids=["old"], semantic_confidence=1., semantic_path=["old"])
    result = bind(scene, catalog, membership)
    unknown = result["scene"]["gaussians"][1]
    assert unknown["position"] == scene["gaussians"][1]["position"]
    assert unknown["semantic_ids"] == ["scene"]
    assert unknown["semantic_binding"] == "unindexed_unknown"
    assert "object_id" not in unknown and "semantic_confidence" not in unknown and "semantic_path" not in unknown
    assert result["diagnostics"]["unindexed_preview_count"] == 1
    for gaussian in scene["gaussians"]:
        gaussian.pop("source_index", None)
    with pytest.raises(ValueError, match="re-export"):
        bind(scene, catalog, membership)


def test_class_order_is_independently_declared_and_never_silently_reordered():
    scene, catalog, membership = fixture()
    with pytest.raises(ValueError, match="order"):
        bind_semantic_sidecar(scene, catalog, membership, membership_classes=list(reversed(catalog["classes"])))
    catalog["classes"][1] = "body"
    with pytest.raises(ValueError, match="duplicate"):
        bind(scene, catalog, membership)


@pytest.mark.parametrize("bad", [np.ones((4, 3), bool), np.full((4, 4), .5), np.full((4, 4), 2)])
def test_membership_shape_and_binary_validation(bad):
    scene, catalog, _ = fixture()
    with pytest.raises(ValueError):
        bind(scene, catalog, bad)


def test_candidates_do_not_activate_hierarchy_without_explicit_approval():
    scene, catalog, membership = fixture()
    result = bind(scene, catalog, membership, relation_edges=[edge(0, 1), edge(1, 2)])
    np.testing.assert_array_equal(result["membership_closed"], membership)
    assert result["diagnostics"]["approved_edge_count"] == 0
    assert all(obj["parent_id"] == "scene" for obj in result["scene"]["objects"][1:])
    assert result["scene"]["metadata"]["semantic_sidecar"]["relation_candidates"][0]["approved_by_caller"] is False


def test_multilabel_full_ancestor_closure_then_same_preview_indices():
    scene, catalog, membership = fixture()
    result = bind(scene, catalog, membership, approved_edges=[edge(0, 1), edge(1, 2)])
    expected = membership.copy()
    expected[0, :3] = True
    expected[1, 0] = True  # Full-model row 1 is not in this preview.
    np.testing.assert_array_equal(result["membership_closed"], expected)
    class_ids = ids(result)
    for gaussian in result["scene"]["gaussians"]:
        source = gaussian["source_index"]
        assert set(gaussian["semantic_ids"]) == {"scene"} | {class_ids[label] for i, label in enumerate(catalog["classes"]) if expected[source, i]}
        assert "object_id" not in gaussian and "semantic_path" not in gaussian
    assert set(result["scene"]["gaussians"][1]["semantic_inherited_ids"]) == {class_ids["body"], class_ids["hand"]}
    assert result["diagnostics"]["added_membership_count"] == 3
    assert result["diagnostics"]["changed_gaussian_count"] == 2
    assert result["diagnostics"]["preview_added_membership_count"] == 2
    assert result["diagnostics"]["preview_changed_gaussian_count"] == 1
    objects = {obj["id"]: obj for obj in result["scene"]["objects"]}
    assert objects[class_ids["finger"]]["parent_id"] == class_ids["hand"]
    assert objects[class_ids["hand"]]["parent_id"] == class_ids["body"]
    assert objects[class_ids["finger"]]["semantic_granularity"] == "category_union"
    assert objects[class_ids["finger"]]["instance_verified"] is False


def test_multi_parent_dag_does_not_arbitrarily_claim_one_owner():
    scene, catalog, membership = fixture()
    result = bind(scene, catalog, membership, approved_edges=[edge(0, 2), edge(1, 2)])
    class_ids = ids(result)
    finger = next(o for o in result["scene"]["objects"] if o["label"] == "finger")
    assert finger["parent_id"] == "scene"
    assert set(finger["parent_ids"]) == {class_ids["body"], class_ids["hand"]}
    assert result["membership_closed"][0].all()
    assert result["diagnostics"]["multi_parent_class_indices"] == [2]


def test_cycle_or_unmatched_approval_is_rejected_before_scene_mutation():
    scene, catalog, membership = fixture()
    original = copy.deepcopy(scene)
    with pytest.raises(ValueError, match="cycle"):
        bind(scene, catalog, membership, approved_edges=[edge(0, 1), edge(1, 2), edge(2, 0)])
    with pytest.raises(ValueError, match="exactly match"):
        bind(scene, catalog, membership, relation_edges=[edge(0, 1)], approved_edges=[edge(0, 2)])
    assert scene == original


def test_no_geometry_sh_source_or_input_changes_and_json_scene_serializes():
    scene, catalog, membership = fixture()
    original, original_catalog, original_membership = copy.deepcopy(scene), copy.deepcopy(catalog), membership.copy()
    result = bind(scene, catalog, membership, approved_edges=[edge(0, 1)])
    assert scene == original and catalog == original_catalog
    np.testing.assert_array_equal(membership, original_membership)
    assert not np.shares_memory(membership, result["membership_closed"])
    for before, after in zip(scene["gaussians"], result["scene"]["gaussians"]):
        assert all(after[key] == value for key, value in before.items())
    assert result["scene"]["cameras"] == scene["cameras"]
    assert all(result["scene"]["metadata"][key] == value for key, value in scene["metadata"].items())
    json.dumps(result["scene"], allow_nan=False)
    result["scene"]["gaussians"][0]["position"][0] = 999
    assert scene == original


def test_actual_probabilities_do_not_propagate_or_fabricate_certainty():
    scene, catalog, membership = fixture()
    probabilities = np.full(membership.shape, np.nan)
    probabilities[0] = [.1, .2, .73, np.nan]
    probabilities[3, 0] = .0
    saved = probabilities.copy()
    result = bind(scene, catalog, membership, probabilities=probabilities, approved_edges=[edge(0, 1), edge(1, 2)])
    class_ids = ids(result)
    gaussian = result["scene"]["gaussians"][1]
    assert gaussian["semantic_confidence"] == .73
    assert gaussian["semantic_probabilities"][class_ids["body"]] == .1
    assert gaussian["semantic_probabilities"][class_ids["other"]] is None
    assert result["scene"]["gaussians"][0]["semantic_confidence"] == 0
    assert "semantic_confidence" not in result["scene"]["gaussians"][2]
    assert gaussian["source"] == "unknown" and gaussian["confidence"] == .24
    np.testing.assert_array_equal(probabilities, saved)
    json.dumps(result["scene"], allow_nan=False)
    unknown = bind(scene, catalog, membership)
    assert all("semantic_confidence" not in g for g in unknown["scene"]["gaussians"])


@pytest.mark.parametrize("bad", [np.full((4, 4), np.inf), np.full((4, 4), -.1), np.full((4, 4), 1.01), np.zeros((4, 3))])
def test_invalid_probabilities_are_rejected(bad):
    scene, catalog, membership = fixture()
    with pytest.raises(ValueError, match="probabilities"):
        bind(scene, catalog, membership, probabilities=bad)


def test_root_hide_is_reversible_and_retains_even_unindexed_unclassified_geometry():
    scene, catalog, membership = fixture()
    scene["gaussians"].append({"position": [8., 9., 10.], "source": "inferred", "semantic_ids": ["stale"]})
    bound = bind(scene, catalog, membership)["scene"]
    hidden = apply_command(bound, {"action": "hide", "object_ids": ["scene"]})["scene"]
    assert len(hidden["gaussians"]) == 4
    assert all(g["hidden"] for g in hidden["gaussians"])
    assert [g["position"] for g in hidden["gaussians"]] == [g["position"] for g in scene["gaussians"]]
    restored = apply_command(hidden, {"action": "undo"})["scene"]
    assert not any(g["hidden"] for g in restored["gaussians"])


def test_parent_delete_uses_closed_multilabel_membership_and_preserves_unrelated_points():
    scene, catalog, membership = fixture()
    result = bind(scene, catalog, membership, approved_edges=[edge(0, 1), edge(1, 2)])
    class_ids = ids(result)
    hidden = apply_command(result["scene"], {"action": "delete", "object_ids": [class_ids["body"]]})["scene"]
    assert [g.get("hidden", False) for g in hidden["gaussians"]] == [True, True, False]
    assert len(hidden["gaussians"]) == len(scene["gaussians"])


def test_old_catalog_isolation_and_history_do_not_silently_hide_new_catalog():
    scene, catalog, membership = fixture()
    scene["metadata"]["editor_visibility"] = {"isolated_ids": ["class-000"], "unrelated": "keep"}
    scene["metadata"]["command_history"] = [{"objects": [{"id": "class-000"}]}]
    scene["gaussians"][0]["hidden"] = True
    before = copy.deepcopy(scene)
    result = bind(scene, catalog, membership)
    assert result["scene"]["metadata"]["editor_visibility"] == {"isolated_ids": None, "unrelated": "keep"}
    assert result["scene"]["metadata"]["command_history"] == []
    assert result["diagnostics"]["old_isolation_invalidated"] is True
    assert result["diagnostics"]["old_command_history_invalidated"] is True
    assert result["scene"]["gaussians"][0]["hidden"] is True
    assert scene == before
