from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image
import pytest
import torch

from backend.semantic_views import (file_sha256, resolve_image_reference,
                                    select_semantic_views, validate_mask_sources)
from backend.semantics import discover_objects


def photos(tmp_path, count):
    paths = []
    for index in range(count):
        path = tmp_path / f'{index:03d}.png'
        Image.new('RGB', (8, 6), (index % 256, 0, 0)).save(path)
        paths.append(str(path))
    return paths


def test_default_all_preserves_every_supplied_training_view(tmp_path):
    paths = photos(tmp_path, 31)
    selected, report = select_semantic_views(reversed(paths))
    assert selected == paths
    assert report['selected_count'] == report['eligible_count'] == 31
    assert report['policy'] == 'all' and report['all_eligible_selected']
    assert report['evaluation_annotations_used'] is False


def test_sampled_is_explicit_deterministic_and_spans_input(tmp_path):
    paths = photos(tmp_path, 31)
    cfg = {'semantic_view_mode': 'sampled', 'semantic_max_views': 8}
    selected, report = select_semantic_views(paths, cfg)
    assert selected == select_semantic_views(reversed(paths), cfg)[0]
    assert len(selected) == 8 and selected[0] == paths[0] and selected[-1] == paths[-1]
    assert {x['reason'] for x in report['excluded_images']} == {'sampled_out'}


def test_existing_fixed_benchmark_subset_is_never_expanded(tmp_path):
    paths = photos(tmp_path, 40)
    frozen = paths[::5]
    assert select_semantic_views(frozen)[0] == frozen


def test_legacy_explicit_limit_is_preserved_until_all_is_requested(tmp_path):
    paths = photos(tmp_path, 13)
    old_order = list(reversed(paths))
    selected, report = select_semantic_views(old_order, {'max_images': 6})
    assert selected == old_order[:6] and report['selection_method'] == 'legacy_prefix_limit'
    assert select_semantic_views(paths, {'max_images': 6, 'semantic_view_mode': 'all'})[0] == paths


def test_train_allowlist_and_holdouts_are_applied_before_sampling(tmp_path):
    paths = photos(tmp_path, 10)
    selected, report = select_semantic_views(paths, {
        'allowed_image_paths': [Path(p).name for p in paths[:7]],
        'held_out_image_paths': [paths[3], paths[8]],
    })
    assert selected == paths[:3] + paths[4:7]
    excluded = {x['image_path']: x['reason'] for x in report['excluded_images']}
    assert excluded[paths[3]] == excluded[paths[8]] == 'held_out'
    assert excluded[paths[7]] == 'not_in_training_allowlist'
    assert report['split_source'] == 'explicit_training_allowlist'


def test_empty_allowlist_does_not_fall_back_to_all(tmp_path):
    paths = photos(tmp_path, 2)
    assert select_semantic_views(paths, {'allowed_image_paths': []})[0] == []


def test_explicit_foreign_path_never_reuses_same_basename(tmp_path):
    paths = photos(tmp_path, 1)
    foreign = tmp_path / 'test' / '000.png'
    with pytest.raises(ValueError, match='outside'):
        resolve_image_reference(foreign, paths)
    with pytest.raises(ValueError, match='outside'):
        select_semantic_views(paths, {'allowed_image_paths': [str(foreign)]})


def test_ambiguous_basename_requires_exact_paths(tmp_path):
    a, b = tmp_path / 'train' / 'same.png', tmp_path / 'test' / 'same.png'
    with pytest.raises(ValueError, match='one training'):
        resolve_image_reference('same.png', [a, b])
    assert resolve_image_reference(str(a), [a, b]) == str(a)


@pytest.mark.parametrize('cfg', [
    {'semantic_view_mode': 'auto'}, {'semantic_view_mode': 'sampled', 'semantic_max_views': 0},
    {'semantic_view_mode': 'sampled', 'semantic_max_views': True},
    {'allowed_image_paths': 'all'}, {'held_out_image_paths': 'test.png'},
])
def test_invalid_selection_never_silently_selects_all(cfg):
    with pytest.raises(ValueError):
        select_semantic_views([], cfg)


def test_mask_hashes_bind_exact_image_and_mask_content(tmp_path):
    paths = photos(tmp_path, 2)
    mask = tmp_path / 'mask.png'
    Image.new('L', (8, 6), 255).save(mask)
    record = {'image_name': Path(paths[0]).name, 'source_image_sha256': file_sha256(paths[0]),
              'mask_path': str(mask), 'mask_sha256': file_sha256(mask)}
    result = validate_mask_sources([record], paths, require_hash=True)[0]
    assert result['image_path'] == paths[0] and result['source_image_binding'] == 'sha256_verified'
    assert 'image_path' not in record
    with pytest.raises(ValueError, match='image SHA256 mismatch'):
        validate_mask_sources([{**record, 'image_name': Path(paths[1]).name}], paths)
    Image.new('L', (8, 6), 0).save(mask)
    with pytest.raises(ValueError, match='Mask SHA256 mismatch'):
        validate_mask_sources([record], paths)


def test_legacy_hash_policy_is_explicit(tmp_path):
    paths = photos(tmp_path, 1)
    record = {'image_path': paths[0]}
    assert validate_mask_sources([record], paths)[0]['source_image_binding'] == 'legacy_unbound'
    with pytest.raises(ValueError, match='missing'):
        validate_mask_sources([record], paths, require_hash=True)


class Batch(dict):
    def __getattr__(self, name):
        return self[name]

    def to(self, device):
        return self


def fake_local_models(monkeypatch, detections=1, masks=True):
    class DetectorProcessor:
        def __call__(self, images, **kwargs):
            self.size = images.size
            return Batch(input_ids=torch.ones(1, 2, dtype=torch.int64))

        def post_process_grounded_object_detection(self, out, input_ids, threshold, text_threshold, target_sizes):
            return [{'boxes': torch.tensor([[0., 0., 4., 3.]] * detections),
                     'scores': torch.full((detections,), .8), 'text_labels': ['chair'] * detections}]

    class SamProcessor:
        def __call__(self, images, **kwargs):
            return {'original_sizes': torch.tensor([images.size[::-1]]),
                    'reshaped_input_sizes': torch.tensor([images.size[::-1]])}

        def post_process_masks(self, prediction, original, reshaped):
            h, w = original[0].tolist()
            return [torch.ones(detections, 1, h, w, dtype=torch.bool)]

    detector = lambda **kwargs: SimpleNamespace()
    sampler = lambda **kwargs: SimpleNamespace(pred_masks=torch.zeros(1))
    monkeypatch.setattr('backend.semantics._load_local_models', lambda *a: (
        DetectorProcessor(), detector, SamProcessor() if masks else None, sampler if masks else None))


def test_all_view_local_teacher_streams_masks_without_global_candidate_truncation(tmp_path, monkeypatch):
    paths = photos(tmp_path, 4)
    fake_local_models(monkeypatch, detections=32)
    result = discover_objects(paths, {'provider': 'local', 'device': 'cpu',
                                     'mask_output_dir': str(tmp_path / 'teacher_masks')})
    assert len(result['objects']) == len(result['masks']) == 128
    assert result['coverage']['analyzed_count'] == result['coverage']['masked_count'] == 4
    assert result['coverage']['all_selected_analyzed']
    assert len({r['mask_path'] for r in result['masks']}) == 128
    for record in result['masks']:
        assert 'mask' not in record
        assert record['original_image_size'] == [8, 6] and record['full_frame']
        assert record['source_image_sha256'] == file_sha256(record['image_path'])
        assert record['mask_sha256'] == file_sha256(record['mask_path'])
    assert len(validate_mask_sources(result['masks'], paths, require_hash=True)) == 128


def test_discovery_respects_training_split_and_reports_no_mask_teacher(tmp_path, monkeypatch):
    paths = photos(tmp_path, 9)
    fake_local_models(monkeypatch, masks=False)
    result = discover_objects(paths, {'provider': 'local', 'device': 'cpu',
        'allowed_image_paths': paths[:7], 'held_out_image_paths': paths[6:8],
        'semantic_view_mode': 'sampled', 'semantic_max_views': 3})
    assert result['coverage']['eligible_count'] == 6
    assert result['coverage']['analyzed_count'] == 3
    assert result['coverage']['masked_count'] == 0 and result['masks'] == []
    assert all(x['image_path'] not in paths[6:] for x in result['coverage']['images'])


def test_vision_inventory_batches_every_selected_view_and_keeps_local_ids_distinct(tmp_path, monkeypatch):
    paths = photos(tmp_path, 13)
    calls = []
    def inventory(messages, config):
        calls.append(messages)
        return {'objects': [{'id': 'same-local-id', 'label': 'chair'}]}
    monkeypatch.setattr('backend.semantics.llm_json', inventory)
    result = discover_objects(paths, {'provider': 'vision_api', 'allow_remote_images': True,
                                     'semantic_view_mode': 'all', 'vision_batch_size': 6})
    assert [len(call[0]['content']) - 1 for call in calls] == [6, 6, 1]
    assert result['coverage']['analyzed_count'] == 13
    assert result['coverage']['masked_count'] == 0
    assert len({item['id'] for item in result['objects']}) == 3


def test_selection_failure_never_calls_models(tmp_path, monkeypatch):
    paths = photos(tmp_path, 2)
    monkeypatch.setattr('backend.semantics._local_inventory', lambda *a: pytest.fail('model must not run'))
    result = discover_objects(paths, {'allowed_image_paths': ['missing.png']})
    assert result['available'] is False
