"""CPU integration checks for full-view budgets and region hierarchy training."""
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from backend import semantic_refinement, semantic_worker as worker
from backend.semantic_profiles import resolve_semantic_profile, worker_semantic_config
from tests.test_semantic_worker import FakeGaussians, camera, make_ply, masks


class SemanticFullViewsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.old_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.old_threads)

    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.ply = self.root / "original.ply"
        self.sha = make_ply(self.ply)
        self.model = FakeGaussians()
        self.renders = 0

    def loader(self, *_):
        def render(cam, colors):
            self.renders += 1
            return colors.T.reshape(3, cam.image_height, cam.image_width)
        return self.model, render, torch.device("cpu")

    def run_worker(self, output="run", config=None, records=None, progress=None):
        with patch.object(worker, "_load_upstream_model", self.loader):
            return worker.refine_upstream_semantics(
                self.ply, [camera("a.png"), camera("b.png")], masks() if records is None else records,
                self.root / output,
                {"semantic_steps": 400, "expected_gaussian_count": 6,
                 "sampling_schedule": "view_cycle", **(config or {})}, progress=progress)

    def test_worker_expands_budget_preserves_geometry_and_resumes_full_view_sweeps(self):
        before = {name: getattr(self.model, name).detach().clone() for name in worker._FROZEN_TENSORS}
        snapshots = []
        first = self.run_worker(config={"minimum_sweeps": 201}, progress=lambda *_:
                                snapshots.append(json.loads((self.root / "run/progress.json").read_text())))
        self.assertEqual(first["stats"]["requested_steps"], 400)
        self.assertEqual(first["stats"]["completed_steps"], 402)
        self.assertEqual(first["stats"]["coverage"]["minimum_pair_visits"], 201)
        self.assertEqual(self.renders, 402)
        self.assertTrue((self.root / "run/checkpoints/step_00402/probabilities.npz").is_file())
        self.assertTrue(all(row["total_steps"] == 402 for row in snapshots))
        self.assertEqual(snapshots[-1]["progress"], .99)
        self.assertEqual(snapshots[-1]["remaining_seconds"], 0)
        resumed_snapshots = []
        resumed = self.run_worker(output="resume", config={"minimum_sweeps": 260,
                                 "resume_path": first["resume_path"]}, progress=lambda *_:
                                 resumed_snapshots.append(json.loads((self.root / "resume/progress.json").read_text())))
        self.assertEqual(resumed["stats"]["resumed_from_step"], 402)
        self.assertEqual(resumed["stats"]["completed_steps"], 520)
        self.assertEqual(resumed["stats"]["coverage"]["completed_sweeps"], 260)
        self.assertEqual(resumed["stats"]["coverage"]["uncovered_observed_frame_class_pairs"], [])
        self.assertTrue(all(row["total_steps"] == 520 for row in resumed_snapshots))
        self.assertTrue(all(row["remaining_seconds"] is None or row["remaining_seconds"] >= 0
                            for row in resumed_snapshots))
        self.assertEqual(self.renders, 520)
        self.assertEqual(hashlib.sha256(self.ply.read_bytes()).hexdigest(), self.sha)
        for name, value in before.items():
            torch.testing.assert_close(getattr(self.model, name), value, rtol=0, atol=0)
            self.assertFalse(getattr(self.model, name).requires_grad)
        self.assertTrue(resumed["metadata"]["geometry_unchanged"])

    def test_training_split_rejects_held_out_masks_before_reading_any_pixels(self):
        for split in ({"held_out_frames": ["b.png"]}, {"training_frames": ["a.png"]}):
            with self.subTest(split=split):
                records = [{"image_name": frame, "label": "object", "mask_path": "DO_NOT_READ.png"}
                           for frame in ("a.png", "b.png")]
                with patch.object(worker, "_read_mask", side_effect=AssertionError("pixel read before split rejection")), \
                     patch.object(worker, "_load_upstream_model", side_effect=AssertionError("CUDA load before split rejection")):
                    with self.assertRaisesRegex(ValueError, "outside"):
                        worker.refine_upstream_semantics(self.ply, [camera("a.png"), camera("b.png")], records,
                                                        self.root / "rejected", split)
                self.assertFalse((self.root / "rejected").exists())

    def test_worker_keeps_per_frame_masks_compact_and_reports_missing_views(self):
        records = [record for record in masks() if record["label"] == "object" or record["image_name"] == "a.png"]
        captured = []
        optimize = semantic_refinement.refine_semantics

        def capture(initial, observations, *args, **kwargs):
            for observation in observations:
                captured.append((observation["frame_id"], observation["targets"].shape,
                                 observation["class_indices"].tolist()))
            return optimize(initial, observations, *args, **kwargs)

        with patch.object(semantic_refinement, "refine_semantics", side_effect=capture):
            result = self.run_worker(records=records)
        self.assertEqual([shape[0] for _, shape, _ in captured], [2, 1])
        self.assertEqual(captured[1][2], [0])
        self.assertEqual(result["stats"]["view_class_visits"]["b.png"][1], 0)
        self.assertEqual(result["stats"]["coverage"]["observed_frame_class_pairs"], 3)
        self.assertEqual(result["metadata"]["training_coverage"]["missing_mask_views"], [])
        one_view = self.run_worker(output="one-view", records=[record for record in records if record["image_name"] == "a.png"])
        self.assertEqual(one_view["metadata"]["training_coverage"]["registered_training_views"], 2)
        self.assertEqual(one_view["metadata"]["training_coverage"]["views_with_masks"], 1)
        self.assertEqual(one_view["metadata"]["training_coverage"]["missing_mask_views"], ["b.png"])

    def test_manual_multilevel_regions_match_on_projected_geometry_and_activate_hierarchy(self):
        # Six opaque point centers project to the six synthetic image pixels.
        with torch.no_grad():
            self.model._xyz.copy_(torch.tensor([[(index + .5 - 3) / 3, 0., 1.] for index in range(6)]))
        rows = []
        for record in masks():
            is_part = record["label"] == "part"
            rows.append({**record, "annotation_source": "manual", "object_id": record["label"],
                         "granularity": "part" if is_part else "object",
                         **({"parent_id": "object", "parent_relation": "part_of"} if is_part else {})})
        before = worker._tensor_hashes(self.model, freeze=True)
        result = self.run_worker(records=rows, config={"semantic_granularity": "multilevel",
                                  "granularity_config": {"min_shared_points": 2}})
        self.assertEqual(len(result["metadata"]["classes"]), 2)
        edges = result["metadata"]["hierarchy_edges"]
        self.assertEqual(len(edges), 1)
        self.assertEqual(edges[0]["relation"], "part_of")
        self.assertEqual(edges[0]["relation_sources"], ["user_manual_parent_id"])
        self.assertEqual(edges[0]["support_view_count"], 2)
        self.assertFalse(edges[0]["semantic_part_verified"])
        self.assertEqual(len(result["stats"]["hierarchy_edges_used"]), 1)
        self.assertEqual(worker._tensor_hashes(self.model), before)
        self.assertTrue(result["metadata"]["geometry_unchanged"])
        granularity = json.loads((self.root / "run/granularity.json").read_text())
        self.assertEqual(len(granularity["matches"]), 2)
        self.assertEqual(sorted(track["granularity"] for track in granularity["tracks"]), ["object", "part"])
        self.assertTrue(all(track["view_count"] == 2 for track in granularity["tracks"]))
        self.assertTrue(all(track["identity_source"] == "mutual_visible_geometry_association"
                            for track in granularity["tracks"]))
        self.assertEqual(granularity["diagnostics"]["geometry_binding"],
                         {"source_ply_sha256": self.sha, "gaussian_count": 6})

    def test_parameter_budget_refuses_tracks_without_dropping_them(self):
        with self.assertRaisesRegex(ValueError, "no tracks were silently dropped"):
            self.run_worker(config={"max_semantic_parameter_bytes": 1})
        self.assertEqual(self.renders, 0)
        self.assertFalse((self.root / "run").exists())

    def test_semantic_profiles_preserve_explicit_choices_and_rgb_steps(self):
        for mode, steps, sweeps, size in (("fast", 400, 1, 192), ("balanced", 1200, 2, 256), ("fine", 2400, 3, 384)):
            config = {"mode": mode, "semantic_budget": "mode", "iterations": 22000}
            before = copy.deepcopy(config)
            resolved = resolve_semantic_profile(config)
            self.assertEqual(config, before)
            self.assertEqual((resolved["semantic_steps"], resolved["minimum_sweeps"], resolved["train_size"]),
                             (steps, sweeps, size))
            self.assertEqual(resolved["iterations"], 22000)
            self.assertEqual(resolved["sampling_schedule"], "view_cycle")
            self.assertFalse(resolved["semantic_profile"]["accuracy_validated"])
        explicit = {"mode": "fine", "semantic_budget": "manual", "semantic_steps": 400,
                    "minimum_sweeps": 1, "train_size": 64, "semantic_granularity": "flat",
                    "semantic_view_mode": "sampled", "semantic_max_views": 7}
        resolved = resolve_semantic_profile(explicit)
        self.assertEqual({key: resolved[key] for key in explicit}, explicit)
        worker_config = worker_semantic_config({"mode": "balanced", "semantic_budget": "mode",
                                               "held_out_frames": ["eval.png"]})
        self.assertEqual(worker_config["held_out_frames"], ["eval.png"])
        self.assertEqual(worker_config["minimum_sweeps"], 2)
        self.assertNotIn("iterations", worker_config)


if __name__ == "__main__":
    unittest.main()
