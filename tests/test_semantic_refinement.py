import copy
import json
import unittest
from unittest.mock import patch

import numpy as np
import torch

from backend.semantic_refinement import (
    SemanticRefinementConfig,
    _stage_observation,
    boundary_uncertainty_weights,
    refine_semantics,
)


def identity_render(camera, colors):
    """Each opaque Gaussian covers one pixel; camera permutes the same geometry."""
    order = camera["order"]
    return colors[order].T.reshape(3, camera["height"], camera["width"])


def observations_for(probabilities, permutations=None, observed=None):
    points, classes = probabilities.shape
    permutations = permutations or [list(range(points))]
    observed = observed if observed is not None else [True] * classes
    return [{"frame_id": f"view-{index}",
             "camera": {"order": permutation, "height": 1, "width": points},
             "targets": probabilities[permutation].T.reshape(classes, 1, points),
             "observed": observed, "confidence": [1.0] * classes}
            for index, permutation in enumerate(permutations)]


class SemanticRefinementTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Tiny convolution/reduction tests should not start a large CPU thread pool.
        cls.old_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.old_threads)

    def test_multiview_shared_field_learns_and_visits_every_positive_pair(self):
        truth = torch.tensor([[1., 0.], [1., 0.], [0., 1.], [0., 1.]])
        observations = observations_for(truth, [[0, 1, 2, 3], [3, 2, 0, 1]])
        initial = torch.full_like(truth, 0.3)
        config = SemanticRefinementConfig.preset("simple", steps=80, learning_rate=.12, log_every=20)
        result, stats = refine_semantics(initial, observations, identity_render, config=config)
        self.assertLess(float((result - truth).square().mean()), .002)
        self.assertLess(stats["history"][-1]["loss"], stats["history"][0]["loss"] * .15)
        self.assertTrue(all(n > 0 for visits in stats["view_class_visits"].values() for n in visits))
        self.assertEqual(stats["completed_steps"], 80)
        json.dumps(stats)

    def test_missing_detection_is_not_negative_and_unobserved_class_is_exact(self):
        truth = torch.tensor([[1., 1., 1.], [0., 1., 0.]])
        obs = observations_for(truth, [[0, 1], [1, 0]], [True, True, False])
        obs[1]["observed"] = [False, True, False]
        obs[1]["targets"][0] = 0  # A missed detection must not contradict view 0.
        initial = torch.tensor([[.2, .2, 0.], [.2, .2, .8317]])
        result, stats = refine_semantics(initial, obs, identity_render,
                                       config={"variant": "simple", "steps": 200, "learning_rate": .15})
        torch.testing.assert_close(result[:, 2], initial[:, 2], rtol=0, atol=0)
        self.assertGreater(float(result[0, 0]), .98)
        self.assertEqual(stats["view_class_visits"]["view-1"][0], 0)
        self.assertEqual(stats["class_visits"][2], 0)
        self.assertEqual(stats["unknown_classes"], [2])

    def test_parent_and_part_coexist_without_softmax_competition(self):
        truth = torch.tensor([[1., 1.], [1., 0.], [0., 0.], [0., 0.]])
        initial = torch.tensor([[.55, .9], [.9, .1], [.1, .1], [.1, .1]])
        result, stats = refine_semantics(initial, observations_for(truth), identity_render,
                                       hierarchy_edges=[(0, 1)],
                                       config={"variant": "improved", "steps": 80, "learning_rate": .12})
        self.assertGreater(float(result[0, 0]), .98)
        self.assertGreater(float(result[0, 1]), .98)
        self.assertGreater(float(result[0].sum()), 1.96)
        self.assertLess(float(result[1, 1]), .03)
        self.assertEqual(stats["hierarchy_edges_used"], [(0, 1)])

    def test_hierarchy_does_not_invent_an_unobserved_parent(self):
        truth = torch.tensor([[1., 1.], [1., 0.]])
        initial = torch.tensor([[0., .5], [0., .1]])
        result, stats = refine_semantics(initial, observations_for(truth, observed=[False, True]),
                                       identity_render, [(0, 1)], {"steps": 10})
        torch.testing.assert_close(result[:, 0], initial[:, 0], rtol=0, atol=0)
        self.assertEqual(stats["hierarchy_edges_used"], [])

    def test_boundary_band_weights_both_sides_and_preserves_border_objects(self):
        target = torch.zeros(1, 1, 12)
        target[:, :, :6] = 1
        before = target.clone()
        weights = boundary_uncertainty_weights(target, radius=2, boundary_weight=.2)
        self.assertEqual(tuple(weights.shape), tuple(target.shape))
        torch.testing.assert_close(weights[0, 0, :4], torch.ones(4))
        torch.testing.assert_close(weights[0, 0, 4:8], torch.full((4,), .2))
        torch.testing.assert_close(weights[0, 0, 8:], torch.ones(4))
        torch.testing.assert_close(target, before, rtol=0, atol=0)
        torch.testing.assert_close(boundary_uncertainty_weights(torch.ones(2, 1, 1), 5), torch.ones(2, 1, 1))

    def test_exact_resume_matches_uninterrupted_sampling_and_optimizer(self):
        truth = torch.tensor([[1., 1.], [1., 0.], [0., 0.], [0., 0.]])
        initial = torch.full_like(truth, .6)
        obs = observations_for(truth, [[0, 1, 2, 3], [2, 3, 1, 0]])
        common = dict(variant="improved", learning_rate=.1, channels_per_step=1,
                      seed=12, log_every=1, checkpoint_steps=(0, 8, 20), regularization_samples=2)
        whole, whole_stats = refine_semantics(initial, obs, identity_render, [(0, 1)], {**common, "steps": 20})
        saved = {}
        checkpoints = []
        progress = []

        def state_callback(step, state):
            saved[step] = state

        def probability_callback(step, probs, metadata):
            checkpoints.append((step, tuple(probs.shape), metadata["shape"]))

        refine_semantics(initial, obs, identity_render, [(0, 1)], {**common, "steps": 8},
                         probability_callback, state_callback=state_callback,
                         progress_callback=lambda step, row: progress.append((step, row)))
        resumed, resumed_stats = refine_semantics(initial, obs, identity_render, [(0, 1)],
                                                  {**common, "steps": 20}, resume_state=saved[8])
        torch.testing.assert_close(whole, resumed, rtol=0, atol=0)
        self.assertEqual(whole_stats["view_class_visits"], resumed_stats["view_class_visits"])
        self.assertEqual(whole_stats["history"], resumed_stats["history"])
        self.assertEqual(checkpoints, [(0, (4, 2), [4, 2]), (8, (4, 2), [4, 2])])
        self.assertEqual([step for step, _ in progress], list(range(1, 9)))
        self.assertTrue(all(row["training_seconds"] >= 0 for _, row in progress))
        incompatible = copy.deepcopy(obs)
        incompatible[0]["targets"][0, 0, 0] = 0
        with self.assertRaisesRegex(ValueError, "incompatible"):
            refine_semantics(initial, incompatible, identity_render, [(0, 1)],
                             {**common, "steps": 20}, resume_state=saved[8])

    def test_no_positive_evidence_returns_exact_input_without_render(self):
        initial = np.array([[0., .99], [.1, .2]], dtype=np.float32)
        truth = torch.zeros(2, 2)
        called = []
        result, stats = refine_semantics(initial, observations_for(truth),
                                       lambda *_: self.fail("renderer must not run"),
                                       checkpoint_callback=lambda step, *_: called.append(step))
        np.testing.assert_array_equal(result.numpy(), initial)
        self.assertEqual(stats["status"], "no_positive_observations")
        self.assertEqual(stats["completed_steps"], 0)
        self.assertEqual(called, [0])

    def test_rejects_nonfinite_render_and_invalid_hierarchy(self):
        initial = torch.full((2, 1), .5)
        obs = observations_for(torch.ones(2, 1))
        with self.assertRaisesRegex(FloatingPointError, "non-finite render"):
            refine_semantics(initial, obs, lambda camera, colors: identity_render(camera, colors) * float("nan"),
                             config={"steps": 1})
        with self.assertRaisesRegex(ValueError, "cycles"):
            refine_semantics(torch.full((2, 2), .5), observations_for(torch.ones(2, 2)),
                             identity_render, [(0, 1), (1, 0)], {"steps": 1})

    def test_valid_pixels_do_not_train_unknown_image_regions(self):
        initial = torch.full((4, 1), .4)
        obs = observations_for(torch.tensor([[1.], [0.], [0.], [0.]]))
        obs[0]["valid_pixels"] = torch.tensor([[1., 1., 0., 0.]])
        result, _ = refine_semantics(initial, obs, identity_render,
                                    config={"variant": "simple", "steps": 40, "learning_rate": .1})
        torch.testing.assert_close(result[2:], initial[2:], atol=1e-7, rtol=0)
        self.assertGreater(float(result[0]), .9)
        self.assertLess(float(result[1]), .1)

    def test_view_cycle_covers_all_observed_pairs_and_expands_budget(self):
        truth = torch.ones(2, 7)
        initial = torch.full_like(truth, .4)
        observations = observations_for(truth, [[0, 1]] * 4)
        observations[0]["observed"] = [True] * 6 + [False]
        observations[1]["observed"] = [True, False, True, False, True, False, False]
        observations[2]["observed"] = [False, True, False, False, False, False, False]
        observations[2]["targets"] = torch.zeros_like(observations[2]["targets"])
        observations[3]["observed"] = [False] * 7
        stages, progress, checkpoints = [], [], []

        def stage(observation, selected, device):
            self.assertTrue(all(observation[key].device.type == "cpu"
                                for key in ("targets", "weights", "confidence")))
            result = _stage_observation(observation, selected, device)
            self.assertLessEqual(result[0].shape[0], 3)
            stages.append((observation["frame_id"], list(selected)))
            return result

        with patch("backend.semantic_refinement._stage_observation", side_effect=stage):
            result, stats = refine_semantics(
                initial, observations, identity_render,
                config={"variant": "simple", "steps": 1, "sampling_schedule": "view_cycle",
                        "minimum_sweeps": 2, "log_every": 1},
                checkpoint_callback=lambda step, _, metadata: checkpoints.append((step, metadata)),
                progress_callback=lambda step, metadata: progress.append((step, metadata)))
        self.assertEqual(stats["requested_steps"], 1)
        self.assertEqual(stats["planned_steps"], 8)
        self.assertEqual(stats["completed_steps"], 8)
        self.assertEqual(stats["coverage"]["sweep_steps"], 4)
        self.assertEqual(stats["coverage"]["min_required_steps"], 8)
        self.assertEqual(stats["coverage"]["completed_sweeps"], 2)
        self.assertEqual(stats["coverage"]["observed_frame_class_pairs"], 10)
        self.assertEqual(stats["coverage"]["uncovered_observed_frame_class_pairs"], [])
        self.assertEqual(stats["coverage"]["uncovered_observed_frames"], [])
        self.assertEqual(stats["view_visits"], {"view-0": 4, "view-1": 2, "view-2": 2, "view-3": 0})
        self.assertEqual(stats["view_class_visits"]["view-2"][1], 2)  # Explicit negative evidence.
        self.assertEqual(stats["view_class_visits"]["view-1"][1], 0)  # Missing detection stays unknown.
        torch.testing.assert_close(result[:, 6], initial[:, 6], rtol=0, atol=0)
        self.assertEqual({frame for frame, _ in stages[:3]}, {"view-0", "view-1", "view-2"})
        self.assertTrue(all(row["planned_steps"] == 8 for _, row in progress))
        self.assertEqual(checkpoints[-1][0], 8)
        self.assertEqual(checkpoints[-1][1]["coverage"]["minimum_pair_visits"], 2)
        self.assertEqual(stats["observation_storage_device"], "cpu")
        self.assertEqual(stats["mask_transfer_policy"], "selected_channels_per_step")
        json.dumps(stats)

    def test_incomplete_cycle_reports_uncovered_evidence_instead_of_full_coverage(self):
        truth = torch.ones(2, 7)
        obs = observations_for(truth, [[0, 1], [1, 0]])
        obs[1]["confidence"] = [0.] * 6 + [1.]
        _, stats = refine_semantics(torch.full_like(truth, .5), obs, identity_render,
                                   config={"steps": 1, "sampling_schedule": "view_cycle"})
        coverage = stats["coverage"]
        self.assertEqual(coverage["sweep_steps"], 4)
        self.assertEqual(coverage["completed_sweeps"], 0)
        self.assertEqual(coverage["observed_frame_class_pairs"], 8)
        self.assertGreater(len(coverage["uncovered_observed_frame_class_pairs"]), 0)
        self.assertEqual(coverage["minimum_pair_visits"], 0)
        self.assertEqual(stats["view_class_visits"]["view-1"][:6], [0] * 6)

    def test_compact_observations_match_dense_loss_and_reduce_mask_storage(self):
        truth = torch.tensor([[1., 0., 1., 0.], [0., 1., 1., 0.], [0., 0., 0., 0.]])
        initial = torch.full_like(truth, .4)
        dense = observations_for(truth, [[0, 1, 2], [2, 1, 0]], [True, False, True, False])
        dense[1]["observed"] = [False, True, True, False]
        compact = copy.deepcopy(dense)
        for obs, indices in zip(compact, ([2, 0], [1, 2])):
            obs["class_indices"] = indices
            obs["targets"] = obs["targets"][indices]
            obs["observed"] = [obs["observed"][c] for c in indices]
            obs["confidence"] = [obs["confidence"][c] for c in indices]
        config = {"steps": 12, "sampling_schedule": "view_cycle", "seed": 33, "log_every": 1}
        whole, whole_stats = refine_semantics(initial, dense, identity_render, config=config)
        small, small_stats = refine_semantics(initial, compact, identity_render, config=config)
        torch.testing.assert_close(whole, small, rtol=0, atol=0)
        torch.testing.assert_close(small[:, 3], initial[:, 3], rtol=0, atol=0)
        self.assertEqual(whole_stats["history"], small_stats["history"])
        self.assertEqual(whole_stats["view_class_visits"], small_stats["view_class_visits"])
        self.assertLess(small_stats["cpu_observation_tensor_bytes"], whole_stats["cpu_observation_tensor_bytes"])

    def test_compact_view_cycle_resume_reconstructs_partial_sweep_exactly(self):
        truth = torch.ones(3, 7)
        obs = observations_for(truth, [[0, 1, 2], [2, 1, 0]])
        for observation in obs:
            observation["class_indices"] = list(range(7))
        common = {"sampling_schedule": "view_cycle", "seed": 9, "log_every": 1,
                  "channels_per_step": 2, "regularization_samples": 2}
        initial = torch.full_like(truth, .6)
        whole, whole_stats = refine_semantics(initial, obs, identity_render, [(0, 1)],
                                              {**common, "steps": 16})
        states = {}
        refine_semantics(initial, obs, identity_render, [(0, 1)], {**common, "steps": 5},
                         state_callback=lambda step, state: states.update({step: state}))
        self.assertEqual(states[5]["schedule_state"]["step_in_sweep"], 5)
        resumed, resumed_stats = refine_semantics(initial, obs, identity_render, [(0, 1)],
                                                   {**common, "steps": 5, "minimum_sweeps": 2},
                                                   resume_state=states[5])
        torch.testing.assert_close(whole, resumed, rtol=0, atol=0)
        self.assertEqual(whole_stats["history"], resumed_stats["history"])
        self.assertEqual(whole_stats["view_class_visits"], resumed_stats["view_class_visits"])
        self.assertEqual(resumed_stats["completed_steps"], 16)
        with self.assertRaisesRegex(ValueError, "incompatible"):
            refine_semantics(initial, obs, identity_render, [(0, 1)],
                             {**common, "steps": 16, "sampling_schedule": "legacy"},
                             resume_state=states[5])

    def test_old_legacy_resume_config_without_schedule_fields_is_compatible(self):
        truth = torch.tensor([[1.], [0.]])
        initial = torch.full_like(truth, .5)
        obs = observations_for(truth)
        states = {}
        refine_semantics(initial, obs, identity_render, config={"steps": 2},
                         state_callback=lambda step, state: states.update({step: state}))
        states[2]["compatibility_config"].pop("sampling_schedule")
        states[2].pop("schedule_state")
        resumed, stats = refine_semantics(initial, obs, identity_render, config={"steps": 5},
                                          resume_state=states[2])
        whole, _ = refine_semantics(initial, obs, identity_render, config={"steps": 5})
        torch.testing.assert_close(resumed, whole, rtol=0, atol=0)
        self.assertEqual(stats["resumed_from_step"], 2)

    def test_compact_indices_and_schedule_validation(self):
        initial = torch.full((2, 3), .5)
        for indices in ([0, 0], [-1, 1], [0, 3], [False, 1], [0.0, 1.0]):
            with self.subTest(indices=indices):
                observation = observations_for(torch.ones(2, 2))[0]
                observation["class_indices"] = indices
                with self.assertRaisesRegex(ValueError, "class_indices"):
                    refine_semantics(initial, [observation], identity_render, config={"steps": 1})
        with self.assertRaisesRegex(ValueError, "sampling_schedule"):
            SemanticRefinementConfig(sampling_schedule="random").validate()
        with self.assertRaisesRegex(ValueError, "minimum_sweeps"):
            SemanticRefinementConfig(minimum_sweeps=1).validate()
        empty = observations_for(torch.ones(2, 3))[0]
        empty.update(class_indices=[], targets=torch.empty(0, 1, 2), observed=[], confidence=[])
        result, stats = refine_semantics(initial, [empty], identity_render,
                                        config={"steps": 1, "sampling_schedule": "view_cycle", "minimum_sweeps": 2})
        torch.testing.assert_close(result, initial, rtol=0, atol=0)
        self.assertEqual(stats["coverage"]["sweep_steps"], 0)


if __name__ == "__main__":
    unittest.main()
