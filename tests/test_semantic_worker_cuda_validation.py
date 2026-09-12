import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from benchmarks import validate_semantic_worker_cuda as validation


class SemanticWorkerCUDAValidationTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name).resolve()
        self.ply = self.root / "source.ply"
        self.split = self.root / "split.json"
        self.masks = self.root / "masks.json"
        self.ply.write_bytes(b"immutable full source fixture")
        self.split.write_text("{}")
        self.masks.write_text("{}")
        self.experiment = self.root / "simple-8"
        self.zero = self.experiment / "step_00000"
        self.zero.mkdir(parents=True)
        self.values = np.asarray([[.05, .95], [.95, .05], [.05, .05]], dtype=np.float16)
        np.savez(self.zero / "probabilities.npz", probabilities=self.values)
        (self.zero / "semantic_classes.json").write_text(json.dumps({"classes": ["coffee", "coffee mug"],
            "gaussian_count": 3, "order": "unaltered input PLY order"}))
        (self.zero / "checkpoint.json").write_text(json.dumps({"step": 0, "geometry_unchanged": True}))
        self.metadata = {"classes": ["coffee", "coffee mug"], "gaussian_count": 3,
            "ply_sha256": validation.sha(self.ply), "split_sha256": validation.sha(self.split),
            "training_masks_sha256": validation.sha(self.masks), "gt_read_for_training": False,
            "geometry_frozen": True, "opacity_frozen": True,
            "training_views": sorted(validation.ORIGINAL_TRAINING_VIEWS)}
        (self.experiment / "experiment.json").write_text(json.dumps(self.metadata))

    def prepare(self, output="prior.npz"):
        return validation.prepare_verified_prior(self.experiment, self.ply, self.split, self.masks,
            validation.ORIGINAL_TRAINING_VIEWS, self.root / output)

    def arguments(self, output="validation"):
        return validation.parser().parse_args(["--ply", str(self.ply), "--split", str(self.split),
            "--training-masks", str(self.masks), "--prior-experiment", str(self.experiment),
            "--output", str(self.root / output), "--annotations", str(self.root / "MUST_NOT_OPEN_GT.json")])

    def prepared_inputs(self):
        return {"views": sorted(validation.ORIGINAL_TRAINING_VIEWS), "cameras": [], "masks": [],
                "training_mask_files": [], "colmap_provenance": {"source": "test fixture"}}

    def completed_worker_fixture(self):
        args = self.arguments()
        output = Path(args.output)
        product = output / "worker"
        product.mkdir(parents=True)
        tensor_hashes = {"_xyz": "unchanged"}
        artifacts = {}
        for key, filename in (("probabilities", "probabilities.npz"), ("membership", "semantic_membership.npz"),
                              ("closed_membership", "closed_semantic_membership.npz")):
            path = product / filename
            np.savez(path, probabilities=self.values.astype(np.float32), source_ply_sha256=np.asarray(validation.sha(self.ply)),
                     classes=np.asarray(["coffee", "coffee mug"]))
            artifacts[key] = {"path": filename, "sha256": validation.sha(path)}
        metadata = {"status": "completed", "geometry_unchanged": True, "semantic_steps": 400,
                    "source_ply_sha256": validation.sha(self.ply), "source_tensor_hashes": tensor_hashes,
                    "final_tensor_hashes": tensor_hashes, "training_frames": sorted(validation.ORIGINAL_TRAINING_VIEWS),
                    "artifacts": artifacts}
        validation.write_json(product / "catalog.json", metadata)
        validation.write_json(product / "stats.json", {"completed_steps": 400, "training_seconds": .1})
        validation.write_json(output / "worker-invocation.json", {"worker_complete": True, "final_tensor_hashes": tensor_hashes,
                                                                  "worker_total_seconds": 30.96244})
        code = {name: validation.sha(validation.ROOT / name) for name in
                ("backend/semantic_worker.py", "benchmarks/validate_semantic_worker_cuda.py", "backend/semantic_refinement.py")}
        validation.write_json(output / "run-plan.json", {"code_sha256": code, "split_sha256": validation.sha(self.split),
            "training_masks_sha256": validation.sha(self.masks), "training_mask_files": [], "experiment_kind": validation.EXPERIMENT_KIND})
        validation.write_json(output / "failed.json", {"status": "failed", "original_error": "namespace conflict"})
        validation.write_json(output / "status.json", {"phase": "failed", "original_error": "namespace conflict"})
        (output / "evaluation").mkdir()
        args.evaluate_existing = True
        return args

    def test_zero_folder_and_identity_promotion_preserve_original_values_and_files(self):
        source_before = {path: validation.sha(path) for path in self.experiment.rglob("*") if path.is_file()}
        self.assertEqual(validation.step_zero_checkpoint(self.experiment), self.zero / "probabilities.npz")
        prior = self.prepare()
        with np.load(prior["path"], allow_pickle=False) as data:
            np.testing.assert_array_equal(data["probabilities"], self.values.astype(np.float32))
            self.assertEqual(str(data["source_ply_sha256"]), validation.sha(self.ply))
            self.assertEqual(data["classes"].tolist(), ["coffee", "coffee mug"])
        self.assertEqual(prior["source_storage_dtype"], "float16")
        self.assertEqual(prior["worker_prior_dtype"], "float32")
        self.assertEqual(source_before, {path: validation.sha(path) for path in source_before})
        with self.assertRaises(FileExistsError):
            self.prepare()

    def test_ambiguous_step_zero_and_metadata_mismatch_are_rejected(self):
        duplicate = self.experiment / "step_000000"
        duplicate.mkdir()
        np.savez(duplicate / "probabilities.npz", probabilities=self.values)
        with self.assertRaisesRegex(ValueError, "exactly one"):
            self.prepare()
        (duplicate / "probabilities.npz").unlink()
        changed = {**self.metadata, "ply_sha256": "0" * 64}
        (self.experiment / "experiment.json").write_text(json.dumps(changed))
        with self.assertRaisesRegex(ValueError, "hashes"):
            self.prepare()
        self.assertFalse((self.root / "prior.npz").exists())

    def test_prior_requires_train_only_provenance_and_checkpoint_order(self):
        (self.experiment / "experiment.json").write_text(json.dumps({**self.metadata, "gt_read_for_training": True}))
        with self.assertRaisesRegex(ValueError, "train-only"):
            self.prepare()
        (self.experiment / "experiment.json").write_text(json.dumps(self.metadata))
        (self.zero / "semantic_classes.json").write_text(json.dumps({"classes": ["coffee mug", "coffee"],
            "gaussian_count": 3, "order": "unaltered input PLY order"}))
        with self.assertRaisesRegex(ValueError, "class order"):
            self.prepare()

    def test_exported_binary_and_soft_artifacts_verify_source_identity(self):
        prior = self.prepare()
        values = validation.load_worker_representation(prior["path"], "raw_probability", 3,
            ["coffee", "coffee mug"], validation.sha(self.ply))
        np.testing.assert_array_equal(values, self.values.astype(np.float32))
        binary = self.root / "binary.npz"
        np.savez(binary, class_0=self.values[:, 0] >= .5, class_1=self.values[:, 1] >= .5,
                 classes=np.asarray(["coffee", "coffee mug"]), source_ply_sha256=np.asarray(validation.sha(self.ply)))
        values = validation.load_worker_representation(binary, "raw_binary", 3,
            ["coffee", "coffee mug"], validation.sha(self.ply))
        np.testing.assert_array_equal(values, (self.values >= .5).astype(np.float32))
        with self.assertRaisesRegex(ValueError, "identity"):
            validation.load_worker_representation(binary, "closed_binary", 3,
                ["coffee mug", "coffee"], validation.sha(self.ply))

    def test_worker_failure_never_enters_gt_evaluation(self):
        args = self.arguments()
        with patch.object(validation, "environment", return_value={"gpu": "A100 unit-test fixture"}), \
             patch.object(validation, "prepare_training_inputs", return_value=self.prepared_inputs()), \
             patch("torch.cuda.synchronize"), \
             patch("backend.semantic_worker.refine_upstream_semantics", side_effect=RuntimeError("worker failure")), \
             patch.object(validation, "evaluate_worker_outputs") as evaluate, \
             patch("builtins.print"):
            with self.assertRaisesRegex(RuntimeError, "worker failure"):
                validation.run(args)
        evaluate.assert_not_called()
        self.assertEqual(json.loads((Path(args.output) / "failed.json").read_text())["status"], "failed")
        self.assertFalse((Path(args.output) / "complete.json").exists())
        self.assertFalse(Path(args.annotations).exists())

    def test_orchestration_marks_smoke_complete_only_after_worker_and_evaluation(self):
        args = self.arguments()
        sequence = []
        tensor_hashes = {"_xyz": "frozen"}

        def fake_worker(*_args, **_kwargs):
            sequence.append("worker_completed")
            return {"status": "completed", "metadata": {"geometry_unchanged": True,
                    "source_tensor_hashes": tensor_hashes, "final_tensor_hashes": tensor_hashes},
                    "stats": {"training_seconds": .1}, "catalog_path": "fixture-catalog.json", "stats_path": "fixture-stats.json"}

        def fake_evaluation(*_args):
            self.assertEqual(sequence, ["worker_completed"])
            sequence.append("evaluation")
            invocation = json.loads((Path(args.output) / "worker-invocation.json").read_text())
            self.assertFalse(invocation["gt_annotations_opened"])
            return {"evaluation_seconds": .2, "summaries": []}

        with patch.object(validation, "environment", return_value={"gpu": "A100 unit-test fixture"}), \
             patch.object(validation, "prepare_training_inputs", return_value=self.prepared_inputs()), \
             patch("torch.cuda.synchronize"), patch("torch.cuda.empty_cache"), \
             patch("backend.semantic_worker.refine_upstream_semantics", side_effect=fake_worker), \
             patch.object(validation, "evaluate_worker_outputs", side_effect=fake_evaluation), \
             patch("builtins.print"):
            result = validation.run(args)
        self.assertEqual(sequence, ["worker_completed", "evaluation"])
        self.assertEqual(result["status"], "complete")
        self.assertEqual(result["experiment_kind"], validation.EXPERIMENT_KIND)
        self.assertTrue(result["all_original_inputs_unchanged"])
        self.assertTrue((Path(args.output) / "complete.json").is_file())
        before = (Path(args.output) / "summary.json").read_bytes()
        with self.assertRaises(FileExistsError):
            validation.run(args)
        self.assertEqual((Path(args.output) / "summary.json").read_bytes(), before)

    def test_eval_only_recovery_preserves_failure_and_never_calls_worker(self):
        args = self.completed_worker_fixture()
        output = Path(args.output)
        protected = {name: (output / name).read_bytes() for name in
                     ("failed.json", "status.json", "worker/catalog.json", "worker/stats.json", "worker-invocation.json")}

        def recovered_evaluation(*_args, **kwargs):
            self.assertTrue(kwargs["verify_loader_reentry"])
            self.assertEqual(Path(_args[3]).name, "evaluation-recovery")
            return {"evaluation_seconds": .2, "summaries": [],
                    "loader_reentry_validation": {"requested": True, "consecutive_same_process_loads": 2, "tensor_hashes_identical": True}}

        with patch.object(validation, "environment", return_value={"gpu": "A100 fixture"}), \
             patch.object(validation, "prepare_training_inputs", return_value=self.prepared_inputs()), \
             patch.object(validation, "evaluate_worker_outputs", side_effect=recovered_evaluation), \
             patch("backend.semantic_worker.refine_upstream_semantics") as worker, patch("builtins.print"):
            result = validation.run(args)
        worker.assert_not_called()
        self.assertFalse(result["training_repeated"])
        self.assertEqual(result["worker_total_seconds"], 30.96244)
        self.assertEqual(protected, {name: (output / name).read_bytes() for name in protected})
        recovery = json.loads((output / "recovery.json").read_text())
        self.assertEqual(recovery["attempts"][0]["status"], "complete")
        self.assertTrue(recovery["attempts"][0]["loader_reentry_validation"]["tensor_hashes_identical"])
        self.assertTrue((output / "complete.json").is_file())

    def test_recovery_failure_appends_audit_without_replacing_original_status(self):
        args = self.completed_worker_fixture()
        output = Path(args.output)
        before = (output / "failed.json").read_bytes(), (output / "status.json").read_bytes()
        with patch.object(validation, "environment", return_value={"gpu": "A100 fixture"}), \
             patch.object(validation, "prepare_training_inputs", return_value=self.prepared_inputs()), \
             patch.object(validation, "evaluate_worker_outputs", side_effect=RuntimeError("evaluation stopped")):
            with self.assertRaisesRegex(RuntimeError, "evaluation stopped"):
                validation.run(args)
        self.assertEqual(before, ((output / "failed.json").read_bytes(), (output / "status.json").read_bytes()))
        recovery = json.loads((output / "recovery.json").read_text())
        self.assertEqual(recovery["attempts"][0]["status"], "failed")
        self.assertFalse((output / "complete.json").exists())

    def test_fresh_mode_uses_uniform_default_without_opening_prior_or_approving_hierarchy(self):
        args = self.arguments(output="fresh")
        args.fresh_default = True
        args.prior_experiment = str(self.root / "PRIOR_MUST_NOT_BE_OPENED")
        tensor_hashes = {"_xyz": "unchanged"}

        def fake_worker(*worker_args, **_kwargs):
            config = worker_args[4]
            self.assertEqual(config["semantic_steps"], 1200)
            self.assertEqual(config["preset"], "improved")
            self.assertIsNone(config["prior"])
            self.assertEqual(config["approved_hierarchy"], [])
            return {"status": "completed", "metadata": {"geometry_unchanged": True,
                    "source_tensor_hashes": tensor_hashes, "final_tensor_hashes": tensor_hashes, "hierarchy_edges": []},
                    "stats": {"training_seconds": .1}, "catalog_path": "fixture-catalog.json", "stats_path": "fixture-stats.json"}

        with patch.object(validation, "environment", return_value={"gpu": "A100 unit-test fixture"}), \
             patch.object(validation, "prepare_training_inputs", return_value=self.prepared_inputs()), \
             patch.object(validation, "prepare_verified_prior", side_effect=AssertionError("fresh must not load a prior")) as prior, \
             patch("backend.semantic_worker._ply_description", return_value={"gaussian_count": 3}), \
             patch("torch.cuda.synchronize"), patch("torch.cuda.empty_cache"), \
             patch("backend.semantic_worker.refine_upstream_semantics", side_effect=fake_worker), \
             patch.object(validation, "evaluate_worker_outputs", return_value={"evaluation_seconds": .2, "summaries": []}), \
             patch("builtins.print"):
            result = validation.run(args)
        prior.assert_not_called()
        self.assertEqual(result["experiment_kind"], validation.FRESH_EXPERIMENT_KIND)
        self.assertIsNone(result["run_plan"]["prior"])
        context = result["run_plan"]["app_usage_context"]
        self.assertEqual(context["overall_app_semantic_refinement_default"], "projection")
        self.assertEqual(context["mode_that_requires_user_selection"], "cuda_iterative")
        self.assertTrue(context["fresh_default_conditions"])
        self.assertFalse((Path(args.output) / "verified_prior.npz").exists())


if __name__ == "__main__":
    unittest.main()
