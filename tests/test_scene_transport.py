"""Exact, bounded JSON transport and PLY import budgets; no GPU execution."""
import copy
import json
from pathlib import Path
import sys

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from backend.scene_transport import CHUNK_FORMAT, MANIFEST_FORMAT, LINES_FORMAT, read_scene_lines, write_viewer_scene


def scene(count=5):
    return {'gaussians': [{'position': [i, .5, -2], 'color': [.1, .2, .3], 'scale': [.1]*3,
                           'rotation': [1, 0, 0, 0], 'opacity': .7, 'source_index': i*7,
                           'source': 'inferred' if i%2 else 'observed', 'sh_degree': 3,
                           'sh': [j*.00001-i*.01 for j in range(48)],
                           'semantic_memberships': ['cup', 'handle'] if i%2 else [],
                           'object_id': f'instance-{i}'} for i in range(count)],
            'cameras': [{'frame_id': '真实帧.png'}], 'objects': [{'id': 'cup', 'label': '杯子'}],
            'metadata': {'source_gaussian_count': count*7, 'preview_sampled': True}}


def test_chunks_preserve_all_values_global_order_and_header(tmp_path):
    original = scene()
    canonical = tmp_path/'scene.json'
    canonical.write_text(json.dumps(original))
    before = canonical.read_bytes()
    result = write_viewer_scene(original, canonical, threshold=2, chunk_size=2)
    manifest = json.loads(result.read_text())
    assert manifest['format'] == MANIFEST_FORMAT
    assert manifest['gaussian_count'] == 5
    assert [row['offset'] for row in manifest['chunks']] == [0, 2, 4]
    assert [row['count'] for row in manifest['chunks']] == [2, 2, 1]
    actual = []
    for row in manifest['chunks']:
        path = tmp_path/row['path']
        assert path.is_relative_to(tmp_path)
        chunk = json.loads(path.read_text())
        assert chunk['format'] == CHUNK_FORMAT
        assert chunk['offset'] == len(actual)
        assert chunk['count'] == len(chunk['gaussians'])
        actual.extend(chunk['gaussians'])
    assert {**manifest['scene'], 'gaussians': actual} == original
    assert canonical.read_bytes() == before
    assert [g['source_index'] for g in actual] == [0, 7, 14, 21, 28]


def test_small_scene_keeps_legacy_path_without_extra_writes(tmp_path):
    canonical = tmp_path/'scene.json'
    canonical.write_text('{}')
    assert write_viewer_scene(scene(), canonical) == canonical
    assert list(tmp_path.iterdir()) == [canonical]


def test_default_threshold_and_25000_record_bound(tmp_path):
    original = {'gaussians': [{'source_index': i} for i in range(50001)]}
    path = write_viewer_scene(original, tmp_path/'scene.json')
    manifest = json.loads(path.read_text())
    assert [row['count'] for row in manifest['chunks']] == [25000, 25000, 1]


def test_republication_does_not_overwrite_chunks_already_in_use(tmp_path):
    original = scene()
    first = write_viewer_scene(original, tmp_path/'scene.json', threshold=1, chunk_size=2)
    prior = json.loads(first.read_text())
    snapshots = {(tmp_path/row['path']): (tmp_path/row['path']).read_bytes() for row in prior['chunks']}
    changed = scene();changed['gaussians'][0]['source_index'] = 100
    second = write_viewer_scene(changed, tmp_path/'scene.json', threshold=1, chunk_size=2)
    current = json.loads(second.read_text())
    assert {r['path'] for r in prior['chunks']}.isdisjoint(r['path'] for r in current['chunks'])
    assert all(path.read_bytes() == value for path, value in snapshots.items())


def test_failed_serialization_preserves_previous_manifest_and_cleans_own_chunks(tmp_path):
    original = scene()
    path = write_viewer_scene(original, tmp_path/'scene.json', threshold=1, chunk_size=2)
    before, entries = path.read_bytes(), set(tmp_path.iterdir())
    original['gaussians'][-1]['sh'][0] = float('nan')
    with pytest.raises(ValueError):
        write_viewer_scene(original, tmp_path/'scene.json', threshold=1, chunk_size=2)
    assert path.read_bytes() == before
    assert set(tmp_path.iterdir()) == entries


@pytest.mark.parametrize('size', [0, -1, 25001, True])
def test_chunk_limit_cannot_be_bypassed(tmp_path, size):
    with pytest.raises(ValueError):
        write_viewer_scene(scene(), tmp_path/'scene.json', chunk_size=size)


@pytest.fixture
def api_client(tmp_path, monkeypatch):
    import backend.app as app_module
    monkeypatch.setattr(app_module, 'DATA', tmp_path)
    return app_module, TestClient(app_module.app)


@pytest.mark.parametrize('mode,expected', [('full',None), ('fast',100000), ('balanced',250000), ('fine',500000)])
def test_ply_import_passes_selected_budget_and_exposes_chunk_manifest(api_client, monkeypatch, mode, expected):
    module, client = api_client
    calls = []
    def read(path, max_points):
        calls.append(max_points)
        return scene(3)
    monkeypatch.setattr(module, 'read_ply', read)
    monkeypatch.setattr(module, 'write_viewer_scene', lambda value, path: write_viewer_scene(value, path, threshold=1, chunk_size=2))
    result = client.post('/api/import', files={'file': ('model.ply', b'ply mock')}, data={'mode': mode})
    assert result.status_code == 200, result.text
    body = result.json()
    assert calls == [expected]
    assert body['scene_url'].endswith('/scene.json')
    assert body['viewer_scene_url'].endswith('/scene.viewer.json')
    canonical = client.get(body['scene_url']).json()
    assert canonical['metadata']['preview_point_budget'] == expected
    assert canonical['metadata']['preview_mode'] == mode
    manifest = client.get(body['viewer_scene_url']).json()
    assert manifest['gaussian_count'] == 3
    assert manifest['scene']['metadata']['source_gaussian_count'] == 21


def test_unknown_ply_mode_rejected_before_reader_called(api_client, monkeypatch):
    module, client = api_client
    def forbidden(*args):
        raise AssertionError('invalid mode reached PLY reader')
    monkeypatch.setattr(module, 'read_ply', forbidden)
    result = client.post('/api/import', files={'file': ('model.ply', b'ply mock')}, data={'mode':'ultra'})
    assert result.status_code == 422


def test_default_ply_mode_is_full_and_small_json_stays_backward_compatible(api_client, monkeypatch):
    module, client = api_client
    calls = []
    monkeypatch.setattr(module, 'read_ply', lambda path, maximum: calls.append(maximum) or scene(1))
    result = client.post('/api/import', files={'file': ('model.ply', b'ply mock')})
    assert calls == [None]
    assert result.status_code == 200
    result = client.post('/api/import', files={'file': ('model.json', json.dumps(scene(1)), 'application/json')})
    body = result.json()
    assert body['scene_url'] == body['viewer_scene_url']
    assert client.get(body['scene_url']).json()['gaussians'] == scene(1)['gaussians']


def test_full_resource_limit_rejects_without_publishing(api_client, monkeypatch, tmp_path):
    module, client = api_client
    import backend.scene_limits as limits
    monkeypatch.setattr(limits, 'MAX_SCENE_GAUSSIANS', 2)
    monkeypatch.setattr(module, 'read_ply', lambda *args: scene(3))
    response = client.post('/api/import', files={'file': ('full.ply', b'ply mock')})
    assert response.status_code == 422
    assert '不会自动抽样' in response.text
    assert not list(tmp_path.iterdir())


def test_streamed_upload_limit_removes_partial_project(api_client, monkeypatch, tmp_path):
    module, client = api_client
    import backend.scene_limits as limits
    monkeypatch.setattr(limits, 'MAX_PLY_IMPORT_BYTES', 5)
    response = client.post('/api/import', files={'file': ('too-large.ply', b'123456')})
    assert response.status_code == 413
    assert not list(tmp_path.iterdir())


def test_limits_accept_fine_scene_and_reject_over_capacity_without_sampling():
    from backend.scene_limits import validate_gaussian_count, scene_point_budget
    validate_gaussian_count(2_162_260)
    validate_gaussian_count(3_000_000)
    assert scene_point_budget() is None
    with pytest.raises(ValueError, match='不会自动抽样'):
        validate_gaussian_count(3_000_001)


def test_all_training_modes_default_to_full_display():
    from backend.app import Config
    from backend.scene_limits import scene_point_budget
    for mode in ('fast','balanced','fine'):
        assert scene_point_budget(Config(mode=mode).viewer_quality) is None


def lines_bytes(original):
    header={key:value for key,value in original.items() if key!='gaussians'}
    records=[{'format':LINES_FORMAT,'gaussian_count':len(original['gaussians']),'scene':header}]
    for offset in range(0,len(original['gaussians']),2):
        values=original['gaussians'][offset:offset+2]
        records.append({'format':CHUNK_FORMAT,'offset':offset,'count':len(values),'gaussians':values})
    return ('\n'.join(json.dumps(value,ensure_ascii=False) for value in records)+'\n').encode('utf-8')


def test_jsonl_import_preserves_sh_global_rows_edits_and_metadata(api_client):
    module,client=api_client
    original=scene(5);original['gaussians'][2].update(hidden=True,priority=True)
    response=client.post('/api/import',files={'file':('complete.splat.jsonl',lines_bytes(original))})
    assert response.status_code==200,response.text
    result=client.get(response.json()['scene_url']).json()
    assert result['gaussians']==original['gaussians']
    assert result['objects']==original['objects']
    assert result['cameras']==original['cameras']
    assert all(result['metadata'][key]==value for key,value in original['metadata'].items())


def test_jsonl_larger_than_ply_and_json_cap_still_imports(api_client,monkeypatch):
    import backend.scene_limits as limits
    module,client=api_client
    monkeypatch.setattr(limits,'MAX_PLY_IMPORT_BYTES',5)
    monkeypatch.setattr(limits,'MAX_JSON_IMPORT_BYTES',5)
    response=client.post('/api/import',files={'file':('full.jsonl',lines_bytes(scene(3)))})
    assert response.status_code==200,response.text


def test_format_limits_cover_real_full_bonsai_export_without_large_fixture():
    from backend.scene_limits import import_file_limit,MAX_SCENE_LINE_BYTES
    assert import_file_limit('.ply')==1024**3
    assert import_file_limit('.json')==256*1024**2
    assert import_file_limit('.jsonl')==4*1024**3
    assert import_file_limit('.jsonl')>1_372_757_800
    assert MAX_SCENE_LINE_BYTES==64*1024**2


@pytest.mark.parametrize('alter',[
    lambda value:value[:-1],
    lambda value:value.replace(b'"offset": 2',b'"offset": 1'),
    lambda value:value.replace(b'"count": 2',b'"count": 3',1),
    lambda value:value.replace(b'"gaussian_count": 5',b'"gaussian_count": 2000001'),
    lambda value:value+b'{}\n',
    lambda value:value.replace('杯子'.encode(),b'\xff'),
    lambda value:value.split(b'\n',1)[0]+b'\n',
])
def test_jsonl_rejects_truncated_misaligned_oversized_or_invalid_input(tmp_path,alter):
    path=tmp_path/'invalid.jsonl';path.write_bytes(alter(lines_bytes(scene(5))))
    with pytest.raises(ValueError):read_scene_lines(path)


def test_jsonl_byte_line_cap_is_enforced_before_decode(tmp_path):
    path=tmp_path/'long.jsonl';path.write_bytes(lines_bytes(scene()))
    with pytest.raises(ValueError,match='单行'):
        read_scene_lines(path,max_line_bytes=32)
