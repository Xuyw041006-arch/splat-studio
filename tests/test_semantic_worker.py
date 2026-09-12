import hashlib
import inspect
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

from backend import semantic_worker as worker


class FakeGaussians:
    def __init__(self, count=6):
        self._xyz = torch.nn.Parameter(torch.arange(count * 3, dtype=torch.float32).reshape(count, 3))
        self._features_dc = torch.nn.Parameter(torch.zeros(count, 1, 3))
        self._features_rest = torch.nn.Parameter(torch.zeros(count, 0, 3))
        self._scaling = torch.nn.Parameter(torch.zeros(count, 3))
        self._rotation = torch.nn.Parameter(torch.ones(count, 4))
        self._opacity = torch.nn.Parameter(torch.ones(count, 1))

    @property
    def get_xyz(self):
        return self._xyz


def make_ply(path, count=6, comment=None):
    properties = ["x", "y", "z", "f_dc_0", "f_dc_1", "f_dc_2", "opacity",
                  "scale_0", "scale_1", "scale_2", "rot_0", "rot_1", "rot_2", "rot_3"]
    text = "ply\nformat ascii 1.0\n" + (f"comment {comment}\n" if comment else "")
    text += f"element vertex {count}\n" + "".join(f"property float {key}\n" for key in properties)
    text += "end_header\n" + (" ".join(["0"] * len(properties)) + "\n") * count
    path.write_text(text)
    return hashlib.sha256(path.read_bytes()).hexdigest()


def camera(name):
    return {"image_name": name, "width": 6, "height": 1, "fx": 3., "fy": 3., "cx": 3., "cy": .5,
            "world_to_camera": np.eye(4).tolist()}


def masks():
    return [{"image_name": frame, "label": label, "mask_id": f"{frame}-{label}",
             "mask": np.asarray(mask, dtype=bool).reshape(1, 6)}
            for frame in ("a.png", "b.png")
            for label, mask in (("object", [1, 1, 1, 1, 0, 0]), ("part", [1, 1, 0, 0, 0, 0]))]


class SemanticWorkerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.old_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.old_threads)

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name).resolve()
        self.ply = self.root / "original.ply"
        self.sha = make_ply(self.ply)
        self.model = FakeGaussians()
        self.renders = 0

    def loader(self, path, root, degree):
        self.assertEqual(path, self.ply)
        self.assertEqual(degree, 0)

        def render(cam, colors):
            self.renders += 1
            return colors.T.reshape(3, cam.image_height, cam.image_width)

        return self.model, render, torch.device("cpu")

    def run_worker(self, *, config=None, output="run", **kwargs):
        with patch.object(worker, "_load_upstream_model", self.loader):
            return worker.refine_upstream_semantics(self.ply, [camera("a.png"), camera("b.png")],
                masks(), self.root / output, {"semantic_steps": 400, "expected_gaussian_count": 6,
                                             **(config or {})}, **kwargs)

    def test_complete_original_identity_sidecars_and_frozen_geometry(self):
        before = {key: getattr(self.model, key).detach().clone() for key in worker._FROZEN_TENSORS}
        messages = []
        result = self.run_worker(config={"semantic_steps": 1200}, progress=lambda value, text: messages.append((value, text)))
        self.assertEqual(self.renders, 1200)
        self.assertEqual(result["status"], "completed")
        self.assertEqual(result["metadata"]["source_ply_sha256"], self.sha)
        self.assertEqual(result["metadata"]["classes"], ["object", "part"])
        self.assertEqual(result["metadata"]["gaussian_count"], 6)
        self.assertTrue(result["metadata"]["geometry_unchanged"])
        self.assertEqual(hashlib.sha256(self.ply.read_bytes()).hexdigest(), self.sha)
        for key, value in before.items():
            torch.testing.assert_close(getattr(self.model, key), value, rtol=0, atol=0)
            self.assertFalse(getattr(self.model, key).requires_grad)
        with np.load(result["raw_probabilities_path"], allow_pickle=False) as raw, np.load(result["closed_probabilities_path"], allow_pickle=False) as closed:
            self.assertEqual(raw["probabilities"].shape, (6, 2))
            self.assertEqual(str(raw["source_ply_sha256"]), self.sha)
            np.testing.assert_array_equal(raw["probabilities"], closed["probabilities"])
            self.assertGreater(raw["probabilities"][0, 0], .98)
            self.assertGreater(raw["probabilities"][0, 1], .98)
        self.assertEqual(result["metadata"]["hierarchy_edges"], [])
        target_metadata = json.loads((self.root / "run/targets.json").read_text())
        self.assertTrue(target_metadata["hierarchy"]["edges"])
        self.assertTrue(all(.95 <= fraction <= .99 for fraction, _ in messages))
        self.assertEqual(messages[-1][0], .99)
        self.assertTrue(Path(result["resume_path"]).is_file())
        progress = json.loads((self.root / "run/progress.json").read_text())
        self.assertEqual(progress["status"], "completed")
        self.assertFalse(result["metadata"]["ground_truth_dependency"])

    def test_only_approved_and_supported_hierarchy_is_activated(self):
        result = self.run_worker(config={"approved_hierarchy": [{"parent": "object", "child": "part",
                                                                 "relation": "part_of", "source": "user_manual_parent_id"},
                                                               {"parent": "unseen", "child": "object"}]})
        edges = result["metadata"]["hierarchy_edges"]
        self.assertEqual([(edge["parent"], edge["child"]) for edge in edges], [("object", "part")])
        self.assertEqual(result["stats"]["hierarchy_edges_used"], [(0, 1)])
        self.assertEqual(edges[0]["relation"], "part_of")
        self.assertEqual(edges[0]["source"], "user_manual_parent_id")
        self.assertEqual(edges[0]["observed_relation"], "observed_mask_containment")
        self.assertFalse(edges[0]["semantic_part_verified"])
        self.assertEqual(len(result["metadata"]["hierarchy_rejected"]), 1)
        with np.load(result["closed_probabilities_path"]) as data:
            self.assertTrue(np.all(data["probabilities"][:, 0] >= data["probabilities"][:, 1]))
        self.assertTrue(result["metadata"]["closure_is_not_an_accuracy_measure"])

    def test_cancel_is_checked_per_render_and_retains_only_completed_checkpoint(self):
        with self.assertRaises(worker.SemanticRefinementCancelled):
            self.run_worker(cancelled=lambda: self.renders >= 3)
        self.assertEqual(self.renders, 3)
        output = self.root / "run"
        self.assertTrue((output / "checkpoints/step_00000/probabilities.npz").is_file())
        self.assertTrue((output / "resume.pt").is_file())
        self.assertFalse((output / "probabilities.npz").exists())
        metadata = json.loads((output / "catalog.json").read_text())
        self.assertEqual(metadata["status"], "cancelled")
        self.assertEqual(metadata["last_step"], 3)
        self.assertTrue(metadata["geometry_unchanged"])

    def test_rejects_sampled_count_and_sampling_comment_before_loading_cuda(self):
        with self.assertRaisesRegex(ValueError, "采样"):
            self.run_worker(config={"expected_gaussian_count": 1000})
        make_ply(self.ply, comment="sampled SH0 preview")
        with self.assertRaisesRegex(ValueError, "采样"):
            self.run_worker()
        self.assertFalse((self.root / "run").exists())

    def test_prior_requires_full_source_identity_and_vertex_order(self):
        prior = self.root / "prior.npz"
        np.savez(prior, probabilities=np.ones((2, 2), np.float32))
        with self.assertRaisesRegex(ValueError, "source_ply_sha256"):
            self.run_worker(config={"prior": {"path": str(prior), "classes": ["object", "part"]}})
        with self.assertRaisesRegex(ValueError, "内嵌"):
            self.run_worker(config={"prior": {"path": str(prior), "classes": ["object", "part"],
                                             "source_ply_sha256": self.sha}})
        np.savez(prior, probabilities=np.ones((2, 2), np.float32), source_ply_sha256=np.asarray(self.sha),
                 classes=np.asarray(["object", "part"]))
        with self.assertRaisesRegex(ValueError, "完整顺序"):
            self.run_worker(config={"prior": {"path": str(prior), "classes": ["object", "part"],
                                             "source_ply_sha256": self.sha}})
        np.savez(prior, probabilities=np.full((6, 2), .4, np.float32), source_ply_sha256=np.asarray(self.sha),
                 classes=np.asarray(["object", "part"]))
        result = self.run_worker(config={"prior": {"path": str(prior), "classes": ["object", "part"],
                                                  "source_ply_sha256": self.sha}})
        self.assertEqual(result["metadata"]["initialization"]["kind"], "verified_full_ply_prior")

    def test_prior_internal_identity_and_order_cannot_be_overridden_by_descriptor(self):
        prior = self.root / "prior.npz"
        config = {"prior": {"path": str(prior), "classes": ["object", "part"], "source_ply_sha256": self.sha}}
        for embedded_sha, embedded_classes, expected_error in (
            (np.asarray("0" * 64), np.asarray(["object", "part"]), "源 PLY 身份"),
            (np.asarray([self.sha]), np.asarray(["object", "part"]), "源 PLY 身份"),
            (np.asarray(self.sha), np.asarray(["part", "object"]), "类别顺序"),
            (np.asarray(self.sha), np.asarray([["object", "part"]]), "类别顺序"),
        ):
            np.savez(prior, probabilities=np.full((6, 2), .4, np.float32),
                     source_ply_sha256=embedded_sha, classes=embedded_classes)
            with self.assertRaisesRegex(ValueError, expected_error):
                self.run_worker(config=config)
        self.assertFalse((self.root / "run").exists())

    def test_mixed_full_image_grids_are_normalized_before_builder_without_mutation(self):
        cameras = [{**camera(name), "width": 600, "height": 100, "fx": 300., "fy": 300., "cx": 300., "cy": 50.}
                   for name in ("a.png", "b.png")]
        records = []
        for name in ("a.png", "b.png"):
            automatic = np.zeros((100, 600), dtype=np.uint8)
            automatic[:, :400] = 255
            manual = np.zeros((85, 512), dtype=np.uint8)
            manual[:, :171] = 7
            records += [{"image_name": name, "label": "object", "mask": automatic, "mask_threshold": 127,
                         "coordinate_space": "full_frame", "annotation_source": "automatic"},
                        {"image_name": name, "label": "part", "mask": manual, "mask_value": 7,
                         "annotation_source": "manual"}]
        originals = [record["mask"].copy() for record in records]
        with patch.object(worker, "_load_upstream_model", self.loader):
            result = worker.refine_upstream_semantics(self.ply, cameras, records, self.root / "mixed",
                {"semantic_steps": 400, "train_size": 6, "expected_gaussian_count": 6,
                 "approved_hierarchy": [{"parent": "object", "child": "part", "relation": "part_of"}]})
        for record, original in zip(records, originals):
            np.testing.assert_array_equal(record["mask"], original)
        targets = json.loads((self.root / "mixed/targets.json").read_text())
        for frame in targets["per_frame"].values():
            self.assertEqual(frame["object"]["shape"], [1, 6])
            self.assertEqual(frame["part"]["shape"], [1, 6])
            self.assertEqual(frame["object"]["area"], 4)
            self.assertEqual(frame["part"]["area"], 2)
        audit = result["metadata"]["mask_coordinate_provenance"]
        self.assertEqual([entry["source_mask_shape"] for entry in audit], [[100, 600], [85, 512]] * 2)
        self.assertTrue(all(entry["training_mask_shape"] == [1, 6] and entry["resampling"] == "nearest" for entry in audit))
        self.assertEqual(audit[0]["coordinate_source"], "explicit_full_frame_declaration")
        self.assertEqual(audit[1]["coordinate_source"], "legacy_mask_record_full_image_contract")
        self.assertEqual(result["stats"]["hierarchy_edges_used"], [(0, 1)])

    def test_crop_and_aspect_mismatch_fail_before_loading_model(self):
        for changes, expected_error in (({"crop_box": [0, 0, 3, 1]}, "裁剪"),
                                        ({"coordinate_space": "roi_pixels"}, "裁剪"),
                                        ({"full_frame": False}, "full-frame"),
                                        ({"mask": np.ones((2, 2), bool)}, "宽高比")):
            records = masks()
            records[0] = {**records[0], **changes}
            with patch.object(worker, "_load_upstream_model", side_effect=AssertionError("must reject before CUDA load")):
                with self.assertRaisesRegex(ValueError, expected_error):
                    worker.refine_upstream_semantics(self.ply, [camera("a.png"), camera("b.png")], records,
                                                    self.root / "invalid")
        self.assertFalse((self.root / "invalid").exists())

    def test_existing_destination_and_unknown_camera_do_not_overwrite_or_read_masks(self):
        destination = self.root / "run"
        destination.mkdir()
        marker = destination / "user.txt"
        marker.write_text("keep")
        with self.assertRaises(FileExistsError):
            self.run_worker()
        self.assertEqual(marker.read_text(), "keep")
        invalid = [{"image_name": "not-training.png", "label": "x", "mask_path": "DO_NOT_READ.png"}]
        with self.assertRaisesRegex(ValueError, "outside"):
            worker.refine_upstream_semantics(self.ply, [camera("a.png")], invalid, self.root / "new")
        self.assertFalse((self.root / "new").exists())

    def test_ambiguous_frame_basename_is_rejected_before_loading_pixels(self):
        cameras = [{**camera("a.png"), "frame_id": "left/a.png"},
                   {**camera("a.png"), "frame_id": "right/a.png"}]
        invalid = [{"image_name": "a.png", "label": "object", "mask_path": "DO_NOT_READ.png"}]
        with self.assertRaisesRegex(ValueError, "ambiguous"):
            worker.refine_upstream_semantics(self.ply, cameras, invalid, self.root / "ambiguous")
        self.assertFalse((self.root / "ambiguous").exists())

    def test_resume_preserves_source_camera_identity_and_extends_schedule(self):
        initial_run = self.run_worker(output="first")
        resumed = self.run_worker(output="continued", config={"semantic_steps": 1200,
                                  "resume_path": initial_run["resume_path"]})
        self.assertEqual(resumed["stats"]["resumed_from_step"], 400)
        self.assertEqual(resumed["stats"]["completed_steps"], 1200)
        self.assertEqual(resumed["metadata"]["resumed_from"]["step"], 400)
        changed = camera("a.png")
        changed["world_to_camera"][0][3] = .1
        with patch.object(worker, "_load_upstream_model", self.loader):
            with self.assertRaisesRegex(ValueError, "相机标定"):
                worker.refine_upstream_semantics(self.ply, [changed, camera("b.png")], masks(), self.root / "bad-resume",
                    {"semantic_steps": 1200, "resume_path": initial_run["resume_path"]})
        self.assertFalse((self.root / "bad-resume").exists())

    def test_changed_geometry_is_detected_and_not_published(self):
        original_loader = self.loader

        def mutating_loader(*args):
            model, render, device = original_loader(*args)

            def changed_render(camera, colors):
                with torch.no_grad():
                    model._xyz[0, 0] += .01
                return render(camera, colors)

            return model, changed_render, device

        with patch.object(worker, "_load_upstream_model", mutating_loader):
            with self.assertRaisesRegex(RuntimeError, "原始高斯参数"):
                worker.refine_upstream_semantics(self.ply, [camera("a.png"), camera("b.png")], masks(),
                                                self.root / "changed", {"semantic_steps": 400})
        metadata = json.loads((self.root / "changed/catalog.json").read_text())
        self.assertEqual(metadata["status"], "failed")
        self.assertFalse(metadata["geometry_unchanged"])
        self.assertFalse((self.root / "changed/probabilities.npz").exists())

    def test_cuda_unavailable_is_explicit_and_never_falls_back_to_mps(self):
        with patch.object(torch.cuda, "is_available", return_value=False):
            with self.assertRaisesRegex(worker.SemanticRefinementUnsupported, "macOS/MPS"):
                worker.refine_upstream_semantics(self.ply, [camera("a.png"), camera("b.png")],
                                                masks(), self.root / "cuda-run")
        self.assertFalse((self.root / "cuda-run").exists())

    def test_reentrant_module_guard_accepts_only_same_root_namespace_paths(self):
        upstream = self.root / "upstream"
        valid = SimpleNamespace(__file__=None, __path__=[str(upstream / "utils")])
        self.assertTrue(worker._module_belongs_to_root(valid, upstream))
        self.assertTrue(worker._module_belongs_to_root(
            SimpleNamespace(__file__=str(upstream / "scene/__init__.py"), __path__=[str(upstream / "scene")]), upstream))
        self.assertTrue(worker._module_belongs_to_root(SimpleNamespace(__file__=str(upstream / "module.py")), upstream))
        for invalid in (
            SimpleNamespace(__file__=None, __path__=[]),
            SimpleNamespace(__file__=None),
            SimpleNamespace(__file__=None, __path__=[str(self.root / "foreign/utils")]),
            SimpleNamespace(__file__=None, __path__=[str(upstream / "utils"), str(self.root / "foreign/utils")]),
            SimpleNamespace(__file__=str(upstream / "utils/__init__.py"), __path__=[str(self.root / "foreign/utils")]),
            SimpleNamespace(__file__=str(self.root / "foreign.py"), __path__=[str(upstream / "utils")]),
        ):
            self.assertFalse(worker._module_belongs_to_root(invalid, upstream))

    def test_camera_projection_uses_calibration_without_opening_rgb(self):
        record = {"image_name": "/missing/image.png", "width": 100, "height": 50,
                  "fx": 80., "fy": 70., "cx": 40., "cy": 20., "world_to_camera": np.eye(4)}
        cam = worker._camera_for_record(record, 50, torch, "cpu")
        self.assertEqual((cam.image_width, cam.image_height), (50, 25))
        point = torch.tensor([0., 0., 2., 1.])
        clip = point @ cam.full_proj_transform
        ndc = clip[:2] / clip[3]
        pixels = (ndc + 1) * torch.tensor([50., 25.]) / 2
        torch.testing.assert_close(pixels, torch.tensor([20., 10.]))
        source = inspect.getsource(worker)
        self.assertNotIn("from benchmarks", source)
        self.assertNotIn("import benchmarks", source)
        self.assertNotIn("load_annotations", source)


if __name__ == "__main__":
    unittest.main()
