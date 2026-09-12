"""Real HTTP import/discovery persistence with CPU-only model doubles."""
import importlib
import io
import json
from pathlib import Path
from types import SimpleNamespace

from fastapi.testclient import TestClient
from PIL import Image
import pytest

from test_semantic_views import fake_local_models

api = importlib.import_module('backend.app')


@pytest.fixture
def environment(tmp_path, monkeypatch):
    monkeypatch.setattr(api, 'DATA', tmp_path)
    monkeypatch.setattr(api, 'analyze_images', lambda paths, cfg: {'image_count': len(paths)})
    monkeypatch.setattr(api, 'make_plan', lambda analysis, cfg: {'warnings': []})
    fake_local_models(monkeypatch)
    client = TestClient(api.app)
    files = []
    for index in range(15):
        buffer = io.BytesIO()
        Image.new('RGB', (8, 6), (index * 10, 0, 0)).save(buffer, 'PNG')
        files.append(('files', (f'view-{index}.png', buffer.getvalue(), 'image/png')))
    result = client.post('/api/projects', files=files)
    assert result.status_code == 200, result.text
    project = result.json()
    return SimpleNamespace(client=client, p=tmp_path / project['id'], pid=project['id'],
                           monkeypatch=monkeypatch)


def analyze(env, **config):
    result = env.client.post(f'/api/projects/{env.pid}/analyze', json={
        'semantics': True, 'device': 'cpu', 'provider': 'local', **config})
    assert result.status_code == 200, result.text
    return result.json()


def test_mode_switch_rebuilds_teacher_coverage_and_streams_new_masks(environment):
    env = environment
    # The strict label mapper is independently tested; the fake logits here do
    # not implement DINO token scores, so supply a deterministic mapped result.
    env.monkeypatch.setattr('backend.semantics._aligned_detector_labels', lambda *args: (
        ['chair'], [{}], {'accepted_labels': 1}))
    fast = analyze(env, mode='fast')
    assert fast['semantic_coverage']['selected_count'] == fast['semantic_coverage']['analyzed_count'] == 12
    assert fast['semantic_profile']['semantic_steps'] == 400
    first = json.loads((env.p / 'masks.json').read_text())
    assert len(first) == 12 and all('mask' not in record for record in first)
    assert all(Path(record['mask_path']).is_file() and record['mask_sha256'] for record in first)
    assert len({Path(record['mask_path']).parent for record in first}) == 1
    balanced = analyze(env, mode='balanced')
    second = json.loads((env.p / 'masks.json').read_text())
    assert len(second) == 15
    assert balanced['semantic_coverage']['selected_count'] == balanced['semantic_coverage']['masked_count'] == 15
    assert balanced['semantic_coverage']['all_selected_have_masks']
    assert balanced['semantic_profile']['semantic_steps'] == 1200
    assert not ({record['mask_path'] for record in first} & {record['mask_path'] for record in second})
    assert all(record['source_image_sha256'] for record in second)
    timing = balanced['semantic_timing']
    assert all(timing[key] >= 0 for key in ('selection_seconds', 'discovery_seconds', 'mask_write_seconds', 'total_seconds'))
    assert timing['mask_write_seconds'] > 0 and timing['gpu_training_included'] is False
    assert abs(sum(timing[key] for key in ('selection_seconds', 'discovery_seconds', 'mask_write_seconds')) - timing['total_seconds']) < 1e-6
    stored = json.loads((env.p / 'inventory.json').read_text())
    assert stored['coverage'] == balanced['semantic_coverage'] and stored['timing'] == timing


def test_all_override_fast_preserves_manual_masks_across_reanalysis(environment):
    env = environment
    env.monkeypatch.setattr('backend.semantics._aligned_detector_labels', lambda *args: (
        ['chair'], [{}], {'accepted_labels': 1}))
    imported = env.client.post(f'/api/projects/{env.pid}/masks', json={
        'image_name': '0000.jpg', 'label': 'user part', 'mask': [[1, 0], [0, 0]]})
    assert imported.status_code == 200
    result = analyze(env, mode='fast', semantic_view_mode='all')
    assert result['semantic_coverage']['selected_count'] == 15
    rows = json.loads((env.p / 'masks.json').read_text())
    assert len(rows) == 16
    assert sum(row.get('annotation_source') == 'manual' for row in rows) == 1
    original = (env.p / 'masks.json').read_bytes()
    def unavailable(*args):
        raise OSError('test detector unavailable')
    env.monkeypatch.setattr('backend.semantics._local_inventory', unavailable)
    failed = analyze(env, mode='balanced')
    assert json.loads((env.p / 'masks.json').read_text()) == json.loads(original)
    assert failed['semantic_timing']['discovery_seconds'] is None
    assert not failed['semantic_coverage']['pixel_mask_supervision_available']


def test_remote_all_view_inventory_is_not_reported_as_pixel_mask_supervision(environment):
    env = environment
    calls = []
    def inventory(messages, config):
        calls.append(messages)
        return {'objects': [{'label': 'chair'}]}
    env.monkeypatch.setattr('backend.semantics.llm_json', inventory)
    result = analyze(env, provider='vision_api', allow_remote_images=True, mode='balanced')
    assert len(calls) == 3
    coverage = result['semantic_coverage']
    assert coverage['analyzed_count'] == coverage['selected_count'] == 15
    assert coverage['masked_count'] == 0
    assert not coverage['pixel_mask_supervision_available'] and not coverage['all_selected_have_masks']
    assert result['semantic_timing']['mask_write_seconds'] == 0
