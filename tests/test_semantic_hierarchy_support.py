"""Fresh hierarchy support is geometric, per edge, and exactly resumable on CPU."""
import copy
import hashlib

import numpy as np
import pytest
import torch

from backend.semantic_refinement import refine_semantics
from backend.semantic_worker import _hierarchy_geometry_support
from tests.test_semantic_refinement import identity_render, observations_for


@pytest.fixture(autouse=True)
def one_cpu_thread():
    previous = torch.get_num_threads(); torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def example():
    truth = torch.tensor([[1., 1.], [1., 0.], [0., 0.], [0., 0.]])
    observations = observations_for(truth, [[0, 1, 2, 3], [3, 2, 0, 1]])
    config = {'steps': 40, 'channels_per_step': 1, 'learning_rate': .1, 'log_every': 1,
              'sampling_schedule': 'view_cycle', 'seed': 2, 'regularization_samples': 2,
              'checkpoint_steps': (0, 17, 40)}
    return torch.full_like(truth, .05), observations, config


def test_fresh_uniform_gets_real_hierarchy_penalty_without_semantic_prior():
    initial, observations, config = example()
    result, stats = refine_semantics(initial, observations, identity_render, [(0, 1)], config,
                                    hierarchy_support_indices={(0, 1): [0]})
    assert stats['regularization_support_points'] == 0
    assert stats['hierarchy_geometry_support_points_per_edge'] == {'0:1': 1}
    assert max(row['hierarchy_loss'] for row in stats['history']) > 0
    assert all(row['prior_loss'] == 0 for row in stats['history'])
    assert torch.isfinite(result).all()
    assert torch.all(initial == .05)


def test_support_is_per_edge_and_cannot_regularize_an_unlisted_point():
    initial, observations, config = example()
    # A large violation exists at point0. Listed background points2/3 have equal
    # probabilities; the first loss must ignore that unrelated point0 violation.
    initial[0, 1] = .9
    _, stats = refine_semantics(initial, observations, identity_render, [(0, 1)],
        {**config, 'steps': 1, 'channels_per_step': 3}, hierarchy_support_indices={(0, 1): [2, 3]})
    assert all(row['hierarchy_loss'] == 0 for row in stats['history'])
    assert stats['hierarchy_support_identity']['edges'][0]['point_count'] == 2
    _, included = refine_semantics(initial, observations, identity_render, [(0, 1)],
        {**config, 'steps': 1, 'channels_per_step': 3}, hierarchy_support_indices={(0, 1): [0]})
    assert included['history'][0]['hierarchy_loss'] > .8


def test_explicit_geometry_support_exact_resume_and_changed_support_rejection():
    initial, observations, config = example()
    ids = {(0, 1): [0, 1]}
    whole, whole_stats = refine_semantics(initial, observations, identity_render, [(0, 1)], config,
                                         hierarchy_support_indices=ids)
    saved = {}
    refine_semantics(initial, observations, identity_render, [(0, 1)], {**config, 'steps': 17},
        hierarchy_support_indices=ids, state_callback=lambda step, state: saved.update({step: state}))
    resumed, stats = refine_semantics(initial, observations, identity_render, [(0, 1)], config,
        hierarchy_support_indices={(0, 1): [1, 0]}, resume_state=saved[17])
    torch.testing.assert_close(whole, resumed, rtol=0, atol=0)
    assert stats['history'] == whole_stats['history']
    assert stats['view_class_visits'] == whole_stats['view_class_visits']
    assert 'hierarchy_support_identity' in saved[17]
    for changed in (None, {}, {(0, 1): [0]}, {(0, 1): [1, 2]}):
        with pytest.raises(ValueError, match='incompatible'):
            refine_semantics(initial, observations, identity_render, [(0, 1)], config,
                             hierarchy_support_indices=changed, resume_state=saved[17])


def test_legacy_none_preserves_values_history_and_resume_schema():
    initial, observations, config = example(); initial.fill_(.6)
    states = {}
    baseline, before = refine_semantics(initial, observations, identity_render, [(0, 1)], config,
        state_callback=lambda step, state: states.update({step: state}))
    explicit_none, after = refine_semantics(initial, observations, identity_render, [(0, 1)], config,
                                           hierarchy_support_indices=None)
    torch.testing.assert_close(baseline, explicit_none, rtol=0, atol=0)
    assert before['history'] == after['history']
    assert 'hierarchy_support_identity' not in states[17]
    resumed, _ = refine_semantics(initial, observations, identity_render, [(0, 1)], config,
                                 resume_state=states[17])
    torch.testing.assert_close(baseline, resumed, rtol=0, atol=0)


@pytest.mark.parametrize('value', [{(0, 1): [0, 0]}, {(0, 1): [-1]}, {(0, 1): [4]},
    {(0, 1): [0.]}, {(0, 1): [True]}, {(0, 1): [[0]]}, {(1, 0): [0]}, {'bad': [0]}, []])
def test_invalid_support_fails_before_render(value):
    initial, observations, config = example()
    def forbidden(*args):
        pytest.fail('invalid support must fail before rendering')
    with pytest.raises(ValueError):
        refine_semantics(initial, observations, forbidden, [(0, 1)], config, hierarchy_support_indices=value)


def test_worker_support_requires_parent_child_intersection_and_two_distinct_views():
    records = [{'frame_id': frame, 'region_id': kind, 'mask': np.ones((1, 1), bool)}
               for frame in ('a', 'b', 'c') for kind in ('parent', 'child')]
    support = {('a', 'parent'): [0, 1, 2, 4], ('a', 'child'): [0, 1, 4],
               ('b', 'parent'): [0, 1, 2], ('b', 'child'): [0, 2, 4],
               ('c', 'parent'): [0, 1, 2, 4], ('c', 'child'): [1, 2, 4]}
    built = {'class_labels': ['P', 'C'],
        'tracks': [{'id': track, 'observations': [{'frame': frame, 'region_id': kind} for frame in ('a', 'b', 'c')]}
                   for track, kind in [('P', 'parent'), ('C', 'child')]],
        'diagnostics': {'matching_config': {'min_parent_views': 2}},
        'hierarchy': {'edges': [{'parent': 'P', 'child': 'C', 'per_view': [
            {'frame': 'a', 'supports': True, 'explicit_declaration': True},
            {'frame': 'b', 'supports': True, 'explicit_declaration': True},
            {'frame': 'c', 'supports': False, 'explicit_declaration': False}]}]}}
    calls = []
    def load(row, mask):
        key = row['frame_id'], row['region_id']; calls.append(key); return support[key]
    mapping, audit = _hierarchy_geometry_support(built, records, load)
    assert mapping[(0, 1)].tolist() == [0]
    assert all(frame != 'c' for frame, _ in calls)
    assert audit[0]['sha256'] == hashlib.sha256(np.asarray([0], np.int64).tobytes()).hexdigest()
    assert audit[0]['minimum_distinct_views'] == 2


def test_missing_geometry_support_never_falls_back_to_prior_or_all_points():
    initial, observations, config = example(); initial.fill_(.85)
    _, stats = refine_semantics(initial, observations, identity_render, [(0, 1)], config,
                               hierarchy_support_indices={(0, 1): []})
    assert stats['regularization_support_points'] == 4
    assert stats['hierarchy_geometry_support_points_per_edge'] == {'0:1': 0}
    assert all(row['hierarchy_loss'] == 0 for row in stats['history'])
    assert any(row['prior_loss'] > 0 for row in stats['history'])


def test_worker_passes_repeated_visible_child_support_without_fabricating_prior(tmp_path, monkeypatch):
    from backend import semantic_worker as worker, semantic_refinement
    from tests.test_semantic_worker import FakeGaussians, camera, make_ply, masks
    model = FakeGaussians()
    with torch.no_grad():
        model._xyz.copy_(torch.tensor([[(i + .5 - 3) / 3, 0., 1.] for i in range(6)]))
    ply = tmp_path / 'source.ply'; before = make_ply(ply)
    def load(*args):
        return model, lambda cam, colors: colors.T.reshape(3, 1, 6), torch.device('cpu')
    monkeypatch.setattr(worker, '_load_upstream_model', load)
    rows = [{**record, 'level': 'part' if record['label'] == 'part' else 'object',
             'annotation_source': 'manual', 'object_id': record['label'],
             **({'parent_id': 'object', 'parent_relation': 'part_of'} if record['label'] == 'part' else {})}
            for record in masks()]
    actual = semantic_refinement.refine_semantics
    captured = []
    def optimize(initial, *args, **kwargs):
        assert torch.all(initial == .05)
        captured.append(kwargs['hierarchy_support_indices'])
        assert next(iter(captured[-1].values())).tolist() == [0, 1]
        return actual(initial, *args, **kwargs)
    monkeypatch.setattr(semantic_refinement, 'refine_semantics', optimize)
    result = worker.refine_upstream_semantics(ply, [camera('a.png'), camera('b.png')], rows,
        tmp_path / 'run', {'semantic_steps': 400, 'expected_gaussian_count': 6,
            'semantic_granularity': 'multilevel', 'sampling_schedule': 'view_cycle',
            'granularity_config': {'min_shared_points': 2}})
    assert captured and result['metadata']['initialization']['kind'] == 'uniform_unknown'
    assert result['stats']['regularization_support_points'] == 0
    assert result['metadata']['hierarchy_geometry_support'][0]['point_count'] == 2
    assert result['metadata']['geometry_unchanged']
    assert hashlib.sha256(ply.read_bytes()).hexdigest() == before
    state = torch.load(result['resume_path'], weights_only=True)
    assert state['worker_identity']['hierarchy_geometry_support'] == result['metadata']['hierarchy_geometry_support']
    assert 'hierarchy_support_identity' in state['optimizer_state']
