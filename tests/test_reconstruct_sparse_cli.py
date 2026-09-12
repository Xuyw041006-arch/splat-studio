"""CPU contracts for the new Colab entrypoint; never execute a CUDA trainer."""
import ast
import importlib.util
import io
import json
from pathlib import Path, PurePosixPath
import shutil
import stat
import zipfile

import numpy as np
from PIL import Image
import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('sparse_cli_test_module', ROOT / 'scripts/reconstruct_sparse.py')
cli = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cli)


@pytest.fixture
def photo_project(tmp_path):
    raw = tmp_path / 'photos'
    raw.mkdir()
    image = Image.fromarray(np.arange(24 * 48 * 3, dtype=np.uint8).reshape(24, 48, 3))
    exif = Image.Exif()
    exif[274] = 6
    exif[41989] = 50
    image.save(raw / 'a.jpg', exif=exif)
    image.transpose(Image.Transpose.FLIP_LEFT_RIGHT).save(raw / 'b.png')
    project = tmp_path / 'project'
    manifest = cli.prepare_project(raw, project)
    return project, manifest, raw


def test_prepare_upright_pixels_original_identity_and_unique_names(photo_project):
    project, manifest, raw = photo_project
    first = manifest['images'][0]
    assert (first['width'], first['height']) == (24, 48)
    assert first['capture']['source_sha256'] == cli.sha256(raw / 'a.jpg')
    assert first['stored_sha256'] == cli.sha256(project / 'images/0000.png')
    assert first['stored_sha256'] != first['capture']['source_sha256']
    assert first['capture']['intrinsics_source'] == 'exif_35mm_approximate'
    assert first['capture']['K'][0][2] == 12
    assert [row['name'] for row in manifest['images']] == ['0000.png', '0001.png']
    template = json.loads((project / 'camera-template.json').read_text())
    assert template['cameras'][0]['source_sha256'] == first['capture']['source_sha256']
    assert cli.load_project(project) == manifest


def test_changed_stored_image_is_rejected(photo_project):
    project, _, _ = photo_project
    Image.new('RGB', (24, 48), 'red').save(project / 'images/0000.png')
    with pytest.raises(ValueError, match='identity changed'):
        cli.load_project(project)


def test_existing_project_never_overwritten(photo_project):
    project, _, raw = photo_project
    before = cli.sha256(project / 'project.json')
    with pytest.raises(ValueError, match='not empty'):
        cli.prepare_project(raw, project)
    assert cli.sha256(project / 'project.json') == before


@pytest.mark.parametrize('count', [1, 301])
def test_photo_count_gate_before_decode_or_training(tmp_path, count):
    raw = tmp_path / 'photos'
    raw.mkdir()
    for i in range(count):
        (raw / f'{i}.jpg').write_bytes(b'not decoded when count is invalid')
    with pytest.raises(ValueError, match='2–300'):
        cli.prepare_project(raw, tmp_path / 'project')


@pytest.mark.parametrize('value', [
    {'device': 'mps'}, {'device': 'cpu'}, {'semantic_strategy': 'joint'},
    {'completion': 'learned'}, {'provided_camera_w2c': []}, {'ground_truth_path': '/tmp/gt'},
])
def test_unsupported_config_never_silently_falls_back(value):
    with pytest.raises(ValueError):
        cli.validated_config(value)


def test_none_completion_and_actual_presets_are_preserved():
    from backend.planning import MODES
    for mode, steps in [('fast', 7000), ('balanced', 15000), ('fine', 22000)]:
        cfg = cli.validated_config({'mode': mode, 'sparse_completion': 'none'})
        assert cfg['sparse_completion'] == 'none'
        assert cfg['device'] == 'cuda'
        assert MODES[cfg['mode']]['iterations'] == steps


def mask_record(name='0000.png', **kwargs):
    return {'image_name': name, 'mask': np.ones((24, 12), bool), 'label': 'cup', **kwargs}


def test_full_frame_mask_normalization_and_priority_uses_entire_filename(photo_project, tmp_path):
    project, manifest, _ = photo_project
    masks = cli.persist_masks([mask_record(full_frame=True)], project, manifest)
    with Image.open(masks[0]['mask_path']) as mask:
        assert mask.size == (24, 48)
    paths = [str(project / 'images' / row['name']) for row in manifest['images']]
    cfg = cli.save_priority_masks(paths, masks, ['cup'], tmp_path / 'run')
    folder = Path(cfg['priority_masks_dir'])
    assert (folder / '0000.png.png').is_file()
    assert not (folder / '0000.png').exists()
    assert set(cfg['priority_masks']) == {paths[0]}
    assert masks[0]['source_image_sha256'] == manifest['images'][0]['stored_sha256']


@pytest.mark.parametrize('extra', [
    {'full_frame': False}, {'full_frame': 0}, {'coordinate_space': 'crop'},
    {'mask_origin': [0, 0]}, {'is_cropped': True}, {'mask': np.ones((12, 24), bool)},
    {'source_image_sha256': 'a' * 64}, {'image_name': 'not-an-input.png'},
])
def test_invalid_mask_coordinates_or_identity_rejected(photo_project, extra):
    project, manifest, _ = photo_project
    row = mask_record()
    row.update(extra)
    with pytest.raises(ValueError):
        cli.persist_masks([row], project, manifest)


def test_selected_name_without_mask_fails_before_training(photo_project, tmp_path):
    project, manifest, _ = photo_project
    masks = cli.persist_masks([mask_record()], project, manifest)
    with pytest.raises(ValueError, match='no actual training masks'):
        cli.save_priority_masks([str(project / 'images/0000.png')], masks, ['bear'], tmp_path / 'run')


@pytest.fixture
def streamed_mask_discovery(monkeypatch):
    import backend.semantics as semantics
    calls = []
    def discover(paths, cfg):
        folder = Path(cfg['mask_output_dir'])
        assert folder.parent.name == 'masks'
        assert not folder.exists()
        folder.mkdir(parents=True)
        path = folder / 'region.png'
        Image.fromarray(np.ones((24, 12), np.uint8)*255).save(path)
        calls.append(dict(cfg))
        return {'objects': [], 'warnings': [], 'source': 'mock-streamed-detector',
                'masks': [{'image_name': '0000.png', 'mask_path': str(path), 'label': 'cup',
                           'full_frame': True, 'coordinate_space': 'full_frame_pixels'}]}
    monkeypatch.setattr(semantics, 'discover_objects', discover)
    return calls


def test_cli_discovery_streams_to_new_directory_then_reuses_unchanged_cache(photo_project, streamed_mask_discovery):
    project, manifest, _ = photo_project
    cfg = {'provider': 'local', 'semantic_view_mode': 'all'}
    first = cli.cache_masks(project, manifest, cfg)
    assert len(streamed_mask_discovery) == 1
    assert 'mask' not in first[0]
    assert Path(first[0]['mask_path']).is_file()
    assert cli.cache_masks(project, manifest, cfg) == first
    assert len(streamed_mask_discovery) == 1
    changed = {**cfg, 'semantic_view_mode': 'sampled', 'semantic_max_views': 1}
    cli.cache_masks(project, manifest, changed)
    assert len(streamed_mask_discovery) == 2
    directories = [call['mask_output_dir'] for call in streamed_mask_discovery]
    assert directories[0] != directories[1]
    assert all((Path(directory)/'region.png').is_file() for directory in directories)


@pytest.mark.parametrize('change', [
    {'semantic_max_views': 2}, {'allowed_image_paths': ['0000.png']},
    {'held_out_image_paths': ['0001.png']}, {'max_images': 2},
    {'semantic_view_mode': 'all'}, {'detection_threshold': .6},
    {'max_detections_per_image': 5}, {'detector_model': 'different-detector'},
])
def test_changed_view_selection_or_teacher_invalidates_cached_masks(photo_project, streamed_mask_discovery, change):
    project, manifest, _ = photo_project
    cfg = {'provider': 'local', 'semantic_view_mode': 'sampled', 'semantic_max_views': 1}
    cli.cache_masks(project, manifest, cfg)
    cli.cache_masks(project, manifest, {**cfg, **change})
    assert len(streamed_mask_discovery) == 2


def test_resolved_profile_values_and_canonical_split_are_bound_in_cache(photo_project, streamed_mask_discovery):
    project, manifest, _ = photo_project
    cfg = {'provider': 'local', 'mode': 'balanced',
           'allowed_image_paths': ['0001.png', '0000.png'], 'held_out_image_paths': []}
    first = cli.cache_masks(project, manifest, cfg)
    signature = json.loads((project/'mask-cache.json').read_text())
    assert signature['format'] == 'sparse_cli_mask_cache_v2'
    assert signature['config']['semantic_view_mode'] == 'sampled'
    assert signature['config']['semantic_max_views'] == 24
    assert signature['config']['allowed_image_paths'] == ['0000.png', '0001.png']
    assert cli.cache_masks(project, manifest, {**cfg, 'allowed_image_paths': ['0000.png', '0001.png']}) == first
    assert len(streamed_mask_discovery) == 1


def test_pre_v2_cache_is_not_reused_for_full_view_discovery(photo_project, streamed_mask_discovery):
    project, manifest, _ = photo_project
    cfg = {'provider': 'local', 'semantic_view_mode': 'all'}
    cli.cache_masks(project, manifest, cfg)
    old = json.loads((project/'mask-cache.json').read_text())
    old.pop('format')
    old['config'].pop('semantic_max_views')
    cli.save_json(project/'mask-cache.json', old)
    cli.cache_masks(project, manifest, cfg)
    assert len(streamed_mask_discovery) == 2


@pytest.fixture
def mock_cuda_pipeline(monkeypatch):
    import backend.planning as planning
    import backend.upstream as upstream
    import backend.learned_geometry as learned
    monkeypatch.setattr(cli, 'require_cuda', lambda require_a100=False: {'gpu': 'CPU mock of A100', 'torch': 'mock'})
    monkeypatch.setattr(planning, 'capabilities', lambda: {'original_3dgs': True, 'cuda_semantic_refinement': True})
    monkeypatch.setattr(learned, 'geometry_availability', lambda cfg: {'available': True})
    calls = []
    def analyze(paths, cfg):
        assert len(paths) == 2
        assert all(Path(path).parent.name == 'images' for path in paths)
        assert cfg['intrinsics_sources'] == ['exif_35mm_approximate', 'heuristic']
        return {'image_count': 2, 'input_sha256': [cli.sha256(p) for p in paths]}
    monkeypatch.setattr(planning, 'analyze_images', analyze)
    monkeypatch.setattr(planning, 'make_plan', lambda a, c, h: {
        'can_reconstruct': True, 'warnings': [], 'recommended_backend': 'original_3dgs',
        'geometry_initialization_path': 'shared_photo_initialization', 'sparse': True, 'extreme': True})
    def reconstruct(paths, output, cfg, progress, cancelled):
        assert cfg['sparse_completion'] == 'none'
        assert cfg['sparse'] and cfg['extreme'] and cfg['device'] == 'cuda'
        assert not cfg.get('precomputed_poses')
        calls.append((paths, output, cfg))
        out = Path(output)
        ply = out / 'model/point_cloud/iteration_7000/point_cloud.ply'
        ply.parent.mkdir(parents=True)
        ply.write_bytes(b'complete PLY placeholder for routing test only')
        scene = {'gaussians': [], 'cameras': [{'image_name': '0000.png'}, {'image_name': '0001.png'}],
                 'metadata': {'backend': 'original_3dgs', 'ply_path': str(ply),
                              'initialization': {'actual_input_only': True},
                              'sparse_completion': {'status': 'disabled'}}}
        cli.save_json(out / 'scene.json', scene)
        progress(.95, 'Model saved (mock; no training)')
        return {'scene_path': str(out / 'scene.json'), 'ply_path': str(ply), 'backend': 'original_3dgs'}
    monkeypatch.setattr(upstream, 'reconstruct_upstream', reconstruct)
    return calls


def test_pipeline_actual_photo_routing_manifest_and_retained_runs(photo_project, mock_cuda_pipeline):
    project, _, _ = photo_project
    cfg = {'mode': 'fast', 'sparse_completion': 'none'}
    result = cli.run_pipeline(project, cfg, geometry_root=project.parent / 'geometry')
    assert result['status'] == 'complete'
    out = Path(result['scene_path']).parent
    record = json.loads((out / 'manifest.json').read_text())
    assert record['precomputed_scene_poses_used'] is False
    assert record['ground_truth_used'] is False
    assert record['requested_config']['sparse_completion'] == 'none'
    assert record['requested_config']['geometry_repo_path'] == str(project.parent / 'geometry/dust3r')
    assert record['full_ply_sha256'] == cli.sha256(result['ply_path'])
    assert record['rgb_ply_unchanged_by_semantics'] is True
    assert (out / 'initialization-summary.json').is_file()
    assert record['registered_views'] == 2
    second = cli.run_pipeline(project, cfg)
    assert second['id'] != result['id']
    assert (out / 'result.json').is_file()
    assert len(mock_cuda_pipeline) == 2


def test_iterative_semantics_calls_production_service_with_full_source(photo_project, mock_cuda_pipeline, monkeypatch):
    import backend.semantic_service as service
    project, manifest, _ = photo_project
    rows = cli.persist_masks([mask_record()], project, manifest)
    cli.save_json(project / 'masks.json', rows)
    monkeypatch.setattr(cli, 'cache_masks', lambda *args: rows)
    observed = []
    def refine(p, run_id, out, cfg, progress, cancelled):
        assert Path(p) == project and Path(out) == project / 'runs' / run_id
        scene = json.loads((Path(out) / 'scene.json').read_text())
        ply = Path(scene['metadata']['ply_path'])
        assert ply.is_file() and cfg['semantic_steps'] == 1200
        observed.append(cli.sha256(ply))
        artifacts = {'probabilities_path': str(Path(out) / 'full-probabilities.npz')}
        scene['metadata']['semantic_refinement_artifacts'] = artifacts
        cli.save_json(Path(out) / 'scene.json', scene)
        return {'scene_path': str(Path(out) / 'scene.json'), 'ply_path': str(ply)}
    monkeypatch.setattr(service, 'refine_scene_from_run', refine)
    result = cli.run_pipeline(project, {'mode': 'fast', 'sparse_completion': 'none',
        'semantics': True, 'semantic_refinement': 'cuda_iterative', 'semantic_steps': 1200})
    assert result['semantic_artifacts']['probabilities_path'].endswith('full-probabilities.npz')
    assert observed == [cli.sha256(result['ply_path'])]


def test_failed_initialization_retains_diagnostics_no_success(photo_project, mock_cuda_pipeline, monkeypatch):
    import backend.upstream as upstream
    from backend.reconstruction import ReconstructionError
    project, _, _ = photo_project
    def fail(*args):
        raise ReconstructionError('no parallax', code='insufficient_parallax', diagnostics={'registered_views': 0})
    monkeypatch.setattr(upstream, 'reconstruct_upstream', fail)
    with pytest.raises(ReconstructionError):
        cli.run_pipeline(project, {'sparse_completion': 'none'})
    out = next((project / 'runs').iterdir())
    status = json.loads((out / 'status.json').read_text())
    assert status['status'] == 'failed'
    assert status['error_code'] == 'insufficient_parallax'
    assert status['geometry_diagnostics'] == {'registered_views': 0}
    assert not (out / 'result.json').exists()
    assert not (project / 'latest-result.json').exists()


def test_cancel_before_initialization_never_calls_trainer(photo_project, mock_cuda_pipeline):
    project, _, _ = photo_project
    with pytest.raises(RuntimeError, match='Cancelled'):
        cli.run_pipeline(project, {}, cancelled=lambda: True)
    assert not mock_cuda_pipeline
    out = next((project / 'runs').iterdir())
    assert json.loads((out / 'status.json').read_text())['status'] == 'cancelled'


def test_no_cuda_is_explicit_and_never_starts_training(photo_project, monkeypatch):
    import torch
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: False)
    with pytest.raises(RuntimeError, match='No CPU/MPS'):
        cli.require_cuda()


def test_notebook_json_and_all_cells_compile_without_saved_execution():
    path = ROOT / 'colab/Splat_Studio_Sparse_A100.ipynb'
    notebook = json.loads(path.read_text())
    assert notebook['metadata']['splat_studio']['gpu_execution_status'] == 'not_executed'
    assert notebook['metadata']['colab']['gpuType'] == 'A100'
    for index, cell in enumerate(notebook['cells']):
        if cell['cell_type'] == 'code':
            assert cell['execution_count'] is None and cell['outputs'] == []
            ast.parse(''.join(cell['source']), filename=f'cell-{index}')
    text = ''.join(''.join(cell['source']) for cell in notebook['cells'])
    assert "'sparse_completion': SPARSE_COMPLETION" in text
    assert "'--no-weights'" in text
    assert "os.environ['PIP_CONSTRAINT']" in text
    assert '--require-a100' in text
    assert 'source_pin' in text and 'source_manifest' in text
    assert (ROOT / 'colab/Splat_Studio_Colab.ipynb').is_file()


def test_notebook_extractor_rejects_escape_and_changed_existing_content(tmp_path):
    notebook = json.loads((ROOT / 'colab/Splat_Studio_Sparse_A100.ipynb').read_text())
    functions = [node for cell in notebook['cells'] if cell['cell_type'] == 'code'
                 for node in ast.parse(''.join(cell['source'])).body
                 if isinstance(node, ast.FunctionDef) and node.name == 'safe_extract_zip']
    env = {'Path': Path, 'PurePosixPath': PurePosixPath, 'zipfile': zipfile, 'io': io, 'stat': stat, 'shutil': shutil}
    exec(compile(ast.Module(body=functions, type_ignores=[]), '<extractor-test>', 'exec'), env)
    def archive(name, payload):
        stream = io.BytesIO()
        with zipfile.ZipFile(stream, 'w') as opened:
            opened.writestr(name, payload)
        return stream.getvalue()
    with pytest.raises(ValueError):
        env['safe_extract_zip'](archive('../escape', 'bad'), tmp_path / 'root')
    env['safe_extract_zip'](archive('test.txt', 'first'), tmp_path / 'root')
    with pytest.raises(ValueError, match='已有解压文件'):
        env['safe_extract_zip'](archive('test.txt', 'changed'), tmp_path / 'root')
    assert (tmp_path / 'root/test.txt').read_text() == 'first'
