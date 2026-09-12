"""The result shelf is explicitly registered, bounded, and never loads models."""
import copy
import json
from pathlib import Path
import sys

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from backend.result_library import LIBRARY_FORMAT, MAX_REGISTRY_BYTES, list_saved_results

PID = 'a' * 32


def make_project(root, name='scene.viewer.json'):
    project = root / PID
    project.mkdir(exist_ok=True)
    (project / 'project.json').write_text('{"images":[]}')
    scene = project / name
    scene.parent.mkdir(parents=True, exist_ok=True)
    scene.write_text('{"format":"splat-studio-scene-chunks/1"}')
    return {'id': 'bonsai-fine', 'title': '盆栽 · 精细', 'description': '历史 A100 结果',
            'project_id': PID, 'scene_path': name,
            'metrics': [{'label': 'PSNR', 'value': 31.2, 'scope': '历史结果'}],
            'notes': ['新版语义精度尚未重测']}


def catalog(root, rows):
    (root / 'result-library.json').write_text(json.dumps({'format': LIBRARY_FORMAT, 'results': rows}))


def test_missing_registry_does_not_enumerate_projects(tmp_path, monkeypatch):
    make_project(tmp_path)
    def forbidden(*args, **kwargs):
        pytest.fail('The catalogue must not enumerate directories')
    monkeypatch.setattr(Path, 'glob', forbidden)
    monkeypatch.setattr(Path, 'iterdir', forbidden)
    assert list_saved_results(tmp_path) == {'results': []}


@pytest.mark.parametrize('filename', ['scene.json', 'runs/scene.viewer.json', '测试 结果.json'])
def test_valid_scene_or_manifest_is_listed_without_reading_model(tmp_path, monkeypatch, filename):
    row = make_project(tmp_path, filename)
    catalog(tmp_path, [row])
    (tmp_path / PID / filename).write_bytes(b'not parsed as JSON by the result shelf')
    original = Path.open
    def guarded(path, *args, **kwargs):
        assert path.name == 'result-library.json', f'Unexpected model read: {path}'
        return original(path, *args, **kwargs)
    monkeypatch.setattr(Path, 'open', guarded)
    result = list_saved_results(tmp_path)['results']
    assert len(result) == 1
    assert result[0]['title'] == row['title']
    assert result[0]['metrics'] == row['metrics']
    assert result[0]['notes'] == row['notes']
    assert result[0]['scene_url'].startswith(f'/api/projects/{PID}/files/')
    assert 'scene_path' not in result[0]
    if filename.startswith('测试'):
        assert '%20' in result[0]['scene_url']


@pytest.mark.parametrize('filename', ['../scene.json', '/scene.json', 'runs/../scene.json',
    'runs//scene.json', './scene.json', 'runs\\scene.json', 'scene.json?x=1',
    'scene.json#x', '%2e%2e/scene.json', 'http://host/scene.json', 'scene.ply'])
def test_unsafe_scene_paths_are_omitted_without_losing_good_entry(tmp_path, filename):
    good = make_project(tmp_path)
    bad = dict(good, id='unsafe', scene_path=filename)
    catalog(tmp_path, [bad, good])
    assert [r['id'] for r in list_saved_results(tmp_path)['results']] == ['bonsai-fine']


@pytest.mark.parametrize('target', ['project', 'marker', 'scene', 'directory'])
def test_symlink_escape_is_not_advertised(tmp_path, target):
    root = tmp_path / 'data';root.mkdir()
    outside = tmp_path / 'outside';outside.mkdir()
    good = make_project(root)
    (outside / 'scene.json').write_text('{}')
    (outside / 'project.json').write_text('{}')
    project = root / PID
    if target == 'project':
        for child in project.iterdir(): child.unlink()
        project.rmdir();project.symlink_to(outside, target_is_directory=True)
        good['scene_path'] = 'scene.json'
    elif target == 'marker':
        (project/'project.json').unlink();(project/'project.json').symlink_to(outside/'project.json')
    elif target == 'scene':
        (project/'scene.viewer.json').unlink();(project/'scene.viewer.json').symlink_to(outside/'scene.json')
    else:
        (project/'linked').symlink_to(outside, target_is_directory=True)
        good['scene_path'] = 'linked/scene.json'
    catalog(root, [good])
    assert list_saved_results(root) == {'results': []}


@pytest.mark.parametrize('missing', ['project', 'marker', 'scene'])
def test_unavailable_entries_are_omitted(tmp_path, missing):
    row = make_project(tmp_path)
    if missing == 'project': row['project_id'] = 'b'*32
    elif missing == 'marker': (tmp_path/PID/'project.json').unlink()
    else: (tmp_path/PID/'scene.viewer.json').unlink()
    catalog(tmp_path, [row])
    assert list_saved_results(tmp_path) == {'results': []}


@pytest.mark.parametrize('field,value', [
    ('id', '../x'), ('project_id', '../x'), ('title', 'x'*121),
    ('description', 'x'*1201), ('notes', ['x']*21), ('notes', ['x'*601]),
    ('metrics', [{'label': 'x', 'value': float('nan')}]),
    ('metrics', [{'label': 'x', 'value': 10**999}]),
    ('metrics', [{'label': 'x', 'value': True}]),
    ('metrics', [{'label': 'x', 'value': 'ok', 'scope': 'x'*301}]),
])
def test_invalid_metadata_is_bounded(tmp_path, field, value):
    good = make_project(tmp_path)
    bad = copy.deepcopy(good);bad['id']='bad';bad[field] = value
    catalog(tmp_path, [bad, good])
    assert [r['id'] for r in list_saved_results(tmp_path)['results']] == ['bonsai-fine']


def test_duplicate_ids_only_listed_once(tmp_path):
    row = make_project(tmp_path)
    catalog(tmp_path, [row, row])
    assert len(list_saved_results(tmp_path)['results']) == 1


def test_oversized_and_malformed_registry_is_rejected(tmp_path):
    path = tmp_path/'result-library.json'
    for content in [b' '*(MAX_REGISTRY_BYTES+1), b'{', b'[]', b'{"format":"other","results":[]}']:
        path.write_bytes(content)
        with pytest.raises(ValueError): list_saved_results(tmp_path)
    catalog(tmp_path, [{}]*101)
    with pytest.raises(ValueError): list_saved_results(tmp_path)


def test_registry_symlink_cannot_read_outside_data(tmp_path):
    root=tmp_path/'data';root.mkdir()
    outside=tmp_path/'outside.json';outside.write_text('{}')
    (root/'result-library.json').symlink_to(outside)
    with pytest.raises(ValueError): list_saved_results(root)


def test_api_lists_existing_result_and_uses_existing_file_route(tmp_path, monkeypatch):
    import backend.app as app_module
    monkeypatch.setattr(app_module, 'DATA', tmp_path)
    client = TestClient(app_module.app)
    assert client.get('/api/results').json() == {'results': []}
    row = make_project(tmp_path)
    catalog(tmp_path, [row])
    result = client.get('/api/results')
    assert result.status_code == 200
    assert client.get(result.json()['results'][0]['scene_url']).status_code == 200
    (tmp_path/'result-library.json').write_text('{}')
    assert client.get('/api/results').status_code == 422
