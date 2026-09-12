import unittest
from unittest.mock import patch
import numpy as np
from backend.commands import apply_command, parse_command, validate_command
from backend.semantics import associate_masks, build_priority_masks, discover_objects, fuse_semantics, project_visible


def camera(name):
    return {"image_name": name, "width": 20, "height": 20, "fx": 10, "fy": 10, "cx": 10, "cy": 10,
            "world_to_camera": np.eye(4).tolist()}


def sample_scene():
    return {"gaussians": [{"position": p, "opacity": 1.0, "source": "observed"}
                          for p in [[-.4, 0, 2], [-.2, 0, 2], [0, 0, 2], [.8, 0, 2], [1, 0, 2], [0, 0, 4]]],
            "cameras": [camera("a.png"), camera("b.png")], "metadata": {}}


def masks():
    chair = np.zeros((20, 20), dtype=bool)
    chair[9:12, 7:12] = True
    table = np.zeros_like(chair)
    table[9:12, 13:17] = True
    return [{"image_name": view, "mask_id": f"local-{index}", "mask": mask, "label": label}
            for view in ["a.png", "b.png"] for index, (mask, label) in enumerate([(chair, "椅子"), (table, "桌子")])]


class FusionTests(unittest.TestCase):
    def test_explicit_coordinate_mismatch_never_binds_default_projection(self):
        scene = sample_scene()
        for cam in scene['cameras']:
            cam['mask_pixel_coordinates_verified'] = False
        result = fuse_semantics(scene, config={'masks': masks()})
        self.assertEqual(result['metadata']['semantics']['status'], 'awaiting_valid_masks')
        self.assertTrue(all('object_id' not in g for g in result['gaussians']))
        warnings = result['metadata']['semantics']['warnings']
        self.assertEqual(len(warnings), len(masks()))
        self.assertTrue(all('像素坐标' in warning and '已跳过' in warning for warning in warnings))

    def test_invalid_view_is_skipped_while_verified_view_can_ground(self):
        scene = sample_scene()
        scene['cameras'][0]['mask_pixel_coordinates_verified'] = False
        scene['cameras'][1]['mask_pixel_coordinates_verified'] = True
        records = masks()
        for record in records:
            record['label'] = 'invalid-view-label' if record['image_name'] == 'a.png' else 'valid-view-label'
        result = fuse_semantics(scene, config={'masks': records})
        self.assertEqual(result['metadata']['semantics']['status'], 'grounded')
        labels = {obj['label'] for obj in result['objects'] if obj['level'] != 'scene'}
        self.assertEqual(labels, {'valid-view-label'})
        self.assertEqual(len(result['metadata']['semantics']['warnings']), 2)

    def test_continuous_frame_bounds_keep_visible_border_not_outside_centers(self):
        # u=-0.2 used to round onto foreground pixel zero; u=19.8 was lost.
        points = [[(u-10)/5, 0, 2] for u in [-.2, 0, 19.8, 20.2]]
        x, _, visible, _ = project_visible(points, camera("a"), (20, 20))
        self.assertEqual(visible.tolist(), [False, True, True, False])
        self.assertEqual(x[2], 19)

    def test_touching_image_border_does_not_discard_object_or_treat_outside_as_background(self):
        scene = sample_scene()
        scene["gaussians"] = [{"position": [-2, 0, 2]}, {"position": [-1.6, 0, 2]}]
        mask = np.zeros((20, 20), bool)
        mask[:, :3] = True
        records = [{"image_name": name, "object_id": "border-object", "label": "边缘物品", "mask": mask}
                   for name in ["a.png", "b.png"]]
        result = fuse_semantics(scene, [], {"masks": records})
        self.assertTrue(all(g.get("object_id") == "border-object" for g in result["gaussians"]))
        self.assertEqual(result["gaussians"][0]["semantic_confidence"], 1)
        self.assertLess(result["gaussians"][1]["semantic_confidence"], 1)

    def test_overlapping_object_votes_preserve_uncertainty_and_do_not_delete_neighbors(self):
        scene = sample_scene()
        mask = np.zeros((20, 20), bool)
        mask[7:14, 6:12] = True
        records = [{"image_name": view, "object_id": oid, "label": oid, "mask": mask}
                   for view in ["a.png", "b.png"] for oid in ["object-a", "object-b"]]
        result = fuse_semantics(scene, [], {"masks": records})
        edge = result["gaussians"][0]
        self.assertNotIn("object_id", edge)
        self.assertTrue(edge["semantic_ambiguous"])
        self.assertEqual({c["object_id"] for c in edge["semantic_candidates"]}, {"object-a", "object-b"})
        deleted = apply_command(result, {"action": "delete", "object_ids": ["object-a"]})["scene"]
        self.assertFalse(deleted["gaussians"][0].get("hidden", False))
        self.assertEqual(deleted["gaussians"][0]["semantic_candidates"], edge["semantic_candidates"])

    def test_depth_tolerance_does_not_imply_full_confidence(self):
        scene = sample_scene()
        scene["gaussians"] = [{"position": [0, 0, z]} for z in [2, 2.04, 4]]
        mask = np.ones((20, 20), bool)
        records = [{"image_name": name, "object_id": "front", "label": "front", "mask": mask}
                   for name in ["a.png", "b.png"]]
        result = fuse_semantics(scene, [], {"masks": records})
        self.assertEqual(result["gaussians"][0]["semantic_confidence"], 1)
        self.assertGreater(result["gaussians"][1]["semantic_confidence"], 0)
        self.assertLess(result["gaussians"][1]["semantic_confidence"], .5)
        self.assertNotIn("semantic_candidates", result["gaussians"][2])

    def test_part_cannot_override_different_supported_parent(self):
        records = masks()
        for record in records:
            record["object_id"] = "chair" if record["label"] == "椅子" else "table"
        records.extend({"image_name": name, "object_id": "table-part", "parent_id": "table",
                        "level": "part", "label": "不一致部件", "mask": records[0]["mask"]}
                       for name in ["a.png", "b.png"])
        result = fuse_semantics(sample_scene(), [], {"masks": records})
        self.assertEqual(result["gaussians"][0]["object_id"], "chair")
        self.assertNotIn("table-part", result["gaussians"][0]["semantic_ids"])

    def test_occluded_centers_never_receive_front_mask(self):
        scene = sample_scene()
        _, _, visible, _ = project_visible([g["position"] for g in scene["gaussians"]], scene["cameras"][0], (20, 20))
        self.assertTrue(visible[2])
        self.assertFalse(visible[5])
        fused = fuse_semantics(scene, [], {"masks": masks(), "association_min_overlap": 1})
        self.assertNotIn("object_id", fused["gaussians"][5])
        self.assertNotIn("object_id", scene["gaussians"][0])

    def test_cross_view_ids_and_priority_share_geometric_support(self):
        result = fuse_semantics(sample_scene(), [], {"masks": masks(), "association_min_overlap": 1, "priority_objects": ["椅子"]})
        items = [o for o in result["objects"] if o["level"] == "object"]
        self.assertEqual(len(items), 2)
        self.assertEqual({o["view_count"] for o in items}, {2})
        chair = next(o for o in items if o["label"] == "椅子")
        self.assertEqual(result["gaussians"][0]["object_id"], chair["id"])
        self.assertEqual(result["gaussians"][0]["semantic_confidence"], 1)
        self.assertEqual(result["gaussians"][0]["reconstruction_weight"], 2)
        self.assertEqual(result["gaussians"][3]["reconstruction_weight"], 1)

    def test_candidate_id_priority_survives_global_id_association(self):
        result = fuse_semantics(sample_scene(), [], {"masks": masks(), "association_min_overlap": 1, "priority_objects": ["local-0"]})
        self.assertTrue(next(o for o in result["objects"] if o["label"] == "椅子")["priority"])
        self.assertFalse(next(o for o in result["objects"] if o["label"] == "桌子")["priority"])

    def test_names_without_masks_do_not_fabricate_geometry(self):
        result = fuse_semantics(sample_scene(), [], {"manual_labels": ["椅子"]})
        self.assertEqual(result["metadata"]["semantics"]["status"], "awaiting_masks")
        self.assertTrue(all("object_id" not in g for g in result["gaussians"]))
        found = discover_objects([], {"manual_labels": "椅子，桌子"})
        self.assertEqual(len(found["objects"]), 2)
        self.assertFalse(found["objects"][0]["grounded"])

    def test_hierarchy_follows_geometry_containment(self):
        records = masks()
        part = np.zeros((20, 20), dtype=bool)
        part[10, 8] = True
        records += [{"image_name": view, "mask": part, "label": "椅背", "level": "part"} for view in ["a.png", "b.png"]]
        result = fuse_semantics(sample_scene(), [], {"masks": records, "association_min_overlap": 1})
        back = next(o for o in result["objects"] if o["label"] == "椅背")
        chair = next(o for o in result["objects"] if o["label"] == "椅子")
        self.assertEqual(back["parent_id"], chair["id"])
        self.assertIn(back["id"], chair["children"])
        self.assertEqual(result["gaussians"][0]["semantic_path"], ["scene", chair["id"], back["id"]])

    def test_same_label_does_not_merge_disjoint_instances(self):
        observations = [{"frame": frame, "indices": points, "label": "chair", "mask_id": mid}
                        for frame in ["a", "b"] for mid, points in [("left", [0, 1, 2]), ("right", [3, 4, 5])]]
        groups = associate_masks(observations)
        self.assertEqual(len(groups), 2)
        self.assertTrue(all(len(g["observations"]) == 2 for g in groups))

    def test_explicit_global_identity_combines_fragments_without_double_votes(self):
        records = masks()
        for record in records:
            if record["label"] == "椅子":
                record["object_id"] = "confirmed-chair"
        records.append(dict(records[0]))
        result = fuse_semantics(sample_scene(), [], {"masks": records, "association_min_overlap": 1})
        self.assertEqual(len([o for o in result["objects"] if o["id"] == "confirmed-chair"]), 1)
        self.assertEqual(result["gaussians"][0]["semantic_view_count"], 2)
        self.assertLessEqual(result["gaussians"][0]["semantic_confidence"], 1)

    def test_transitive_overlap_cannot_merge_same_frame_instances(self):
        obs = [{"frame": "a", "indices": [0, 1, 2], "label": "chair"},
               {"frame": "a", "indices": [3, 4, 5], "label": "chair"},
               {"frame": "b", "indices": [0, 1, 2, 3, 4, 5], "label": "chair"}]
        self.assertEqual(len(associate_masks(obs)), 2)

    def test_depth_and_resized_mask_projection(self):
        cam = camera("a")
        cam["depth"] = np.ones((20, 20), dtype=np.float32) * 2
        original = cam["depth"].copy()
        x, y, visible, _ = project_visible([[0, 0, 2], [0, 0, 4]], cam, (10, 10))
        self.assertEqual((x[0], y[0]), (5, 5))
        self.assertEqual(visible.tolist(), [True, False])
        np.testing.assert_equal(cam["depth"], original)

    def test_invalid_camera_reports_error_and_keeps_unlabeled(self):
        scene = sample_scene()
        scene["cameras"] = []
        result = fuse_semantics(scene, [], {"masks": masks()})
        self.assertEqual(result["metadata"]["semantics"]["status"], "awaiting_valid_masks")
        self.assertTrue(result["metadata"]["semantics"]["warnings"])

    def test_priority_masks_only_cover_selected_grounded_pixels(self):
        import tempfile
        from pathlib import Path
        from PIL import Image
        with tempfile.TemporaryDirectory() as tmp:
            image = Path(tmp) / "view.png"
            Image.new("RGB", (20, 20)).save(image)
            mask = np.zeros((10, 10), dtype=bool)
            mask[1:3, 2:4] = True
            records = [{"image_name": image.name, "mask": mask, "label": "椅子", "mask_id": "candidate-1"}]
            result = build_priority_masks([str(image)], records, ["candidate-1"])
            self.assertEqual(result[str(image)].shape, (20, 20))
            self.assertEqual(int(result[str(image)].sum()), 16)
            self.assertEqual(build_priority_masks([str(image)], records, ["桌子"]), {})

    def test_remote_inventory_requires_explicit_image_upload(self):
        with patch("backend.semantics.llm_json") as network:
            result = discover_objects([], {"provider": "vision_api"})
            self.assertFalse(result["available"])
            network.assert_not_called()


class CommandTests(unittest.TestCase):
    def setUp(self):
        self.scene = fuse_semantics(sample_scene(), [], {"masks": masks(), "association_min_overlap": 1})

    def test_chinese_and_english_commands_are_targeted(self):
        for text, action, query in [("隐藏所有椅子", "hide", "椅子"), ("请把椅子删除", "delete", "椅子"),
                                    ("show all chairs", "show", "chairs"), ("显示全部", "show", "all"),
                                    ("提升椅子的精度", "priority", "椅子"), ("只显示椅子", "isolate", "椅子"),
                                    ("focus on chair", "focus", "chair")]:
            with self.subTest(text=text):
                command = parse_command(text)
                self.assertEqual((command["action"], command["query"]), (action, query))
        self.assertFalse(parse_command("取消加强椅子")["priority"])

    def test_delete_is_reversible_and_retains_geometry(self):
        deleted = apply_command(self.scene, parse_command("删除椅子"))["scene"]
        self.assertEqual(len(deleted["gaussians"]), len(self.scene["gaussians"]))
        self.assertTrue(deleted["gaussians"][0]["hidden"])
        restored = apply_command(deleted, parse_command("撤销"))["scene"]
        self.assertFalse(restored["gaussians"][0]["hidden"])
        self.assertEqual(restored["gaussians"][0]["position"], self.scene["gaussians"][0]["position"])

    def test_unknown_ids_and_code_actions_rejected(self):
        for command in [{"action": "exec", "query": "rm -rf /"}, {"action": "hide", "object_ids": ["invented"]},
                        {"action": "hide", "query": "椅子", "script": "alert(1)"}]:
            with self.subTest(command=command), self.assertRaises(ValueError):
                validate_command(command, self.scene["objects"])

    def test_llm_output_is_validated(self):
        with patch("backend.commands.llm_json", return_value={"action": "shell", "query": "anything"}):
            with self.assertRaises(ValueError):
                parse_command("do something", self.scene["objects"], {"provider": "llm"})

    def test_isolate_unknown_geometry_and_undo(self):
        result = apply_command(self.scene, parse_command("只显示椅子"))["scene"]
        self.assertFalse(result["gaussians"][0]["hidden"])
        self.assertTrue(result["gaussians"][3]["hidden"])
        self.assertTrue(result["gaussians"][5]["hidden"])
        result = apply_command(result, parse_command("undo"))["scene"]
        self.assertFalse(result["gaussians"][5]["hidden"])


if __name__ == "__main__":
    unittest.main()
