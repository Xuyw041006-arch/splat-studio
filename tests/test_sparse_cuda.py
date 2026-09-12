"""CPU fixtures verify the CUDA launch/data boundary; no GPU result is implied."""
import copy
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from backend.colmap_export import export_initialization_dataset, rotation_to_qvec, file_sha256
from backend.sparse_cuda import build_sparse_depth_anchors, patch_training_source, _DEPTH_HOOK, sparse_densification_settings
from backend import upstream

PROJECT = Path(__file__).resolve().parents[1]


def fixture_geometry(tmp_path, views=2, thumbnail=False):
    images = []; paths = []; cameras = []
    width, height = 64, 48
    pixels = np.random.default_rng(7).integers(0, 255, (height, width, 3), dtype=np.uint8)
    for i in range(views):
        path = tmp_path / f'input-{i}.png'; Image.fromarray(pixels).save(path); paths.append(path)
        image = Image.fromarray(pixels)
        if thumbnail: image.thumbnail((32, 24))
        images.append(np.asarray(image).copy())
        w, h = image.size
        matrix = np.eye(4); matrix[0, 3] = -.2 * i
        cameras.append({'image_index': i, 'image_path': str(path), 'width': w, 'height': h,
            'intrinsics': np.array([[w*.6, 0, w/2], [0, w*.6, h/2], [0, 0, 1.]]), 'world_to_camera': matrix})
    points = np.array([[-.3, -.2, 3.], [.3, -.2, 3.], [-.3, .2, 3.], [.3, .2, 3.]])
    observations = []
    for point in points:
        track = {}
        for c in cameras:
            xyz = c['world_to_camera'][:3, :3] @ point + c['world_to_camera'][:3, 3]
            projected = c['intrinsics'] @ xyz
            track[str(c['image_index'])] = (projected[:2] / projected[2]).tolist()
        observations.append(track)
    geometry = {'points': points, 'colors': np.full_like(points, .5), 'confidence': np.array([1., .8, .7, .6]),
        'sources': ['observed', 'observed', 'inferred', 'inferred'], 'observations': observations,
        'images': images, 'cameras': cameras, 'metadata': {'geometry_backend_applied': 'fixture', 'warnings': [],
        'sparse_completion': {'status': 'fixture_visible_only', 'unseen_surfaces_recovered': False}}}
    return geometry, paths


def reader():
    spec = importlib.util.spec_from_file_location('input_only_colmap_reader', PROJECT/'vendor/gaussian-splatting/utils/read_write_model.py')
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


def test_export_roundtrip_binary_tracks_provenance_and_canonical_grid(tmp_path):
    geometry, paths = fixture_geometry(tmp_path, views=3, thumbnail=True)
    geometry['cameras'] = geometry['cameras'][:2]
    result = export_initialization_dataset(geometry, paths, tmp_path/'dataset')
    model = Path(result['model_path']); model_reader = reader()
    cameras = model_reader.read_cameras_binary(model/'cameras.bin')
    images = model_reader.read_images_binary(model/'images.bin')
    points = model_reader.read_points3D_binary(model/'points3D.bin')
    assert len(cameras) == len(images) == 2
    assert set(p.name for p in (tmp_path/'dataset/images').iterdir()) == {p.name for p in paths[:2]}
    assert cameras[1].width == 64 and cameras[1].height == 48
    assert np.allclose(cameras[1].params, [38.4, 38.4, 32., 24.])
    assert np.allclose(images[1].xys, np.array([o['0'] for o in geometry['observations']])*2)
    for point in points.values():
        assert len(point.image_ids) == 2
        assert point.error < 1e-10
        for image_id, slot in zip(point.image_ids, point.point2D_idxs):
            assert images[image_id].point3D_ids[slot] == point.id
    metadata = json.loads((tmp_path/'dataset/initialization.json').read_text())
    assert metadata['external_full_scene_camera_or_pointcloud_used'] is False
    assert metadata['unregistered_input_indices'] == [2]
    assert metadata['initialization_source_counts'] == {'observed': 2, 'inferred': 2}
    assert metadata['pointcloud_sha256'] == file_sha256(model/'points3D.ply')
    assert metadata['input_manifest'][0]['source_sha256'] == file_sha256(paths[0])
    with np.load(metadata['point_provenance_path'], allow_pickle=False) as saved:
        assert saved['sources'].tolist() == geometry['sources']
    verified = upstream.registered_semantic_cameras(tmp_path/'dataset', model, PROJECT/'vendor/gaussian-splatting/utils/read_write_model.py')
    assert all(c['mask_pixel_coordinates_verified'] for c in verified)


@pytest.mark.parametrize('rotation', [np.eye(3), np.diag([1., -1., -1.]), np.array([[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]])])
def test_quaternion_uses_colmap_world_to_camera_convention(rotation):
    assert np.allclose(reader().qvec2rotmat(rotation_to_qvec(rotation)), rotation, atol=1e-10)


@pytest.mark.parametrize('corruption,match', [
    ('foreign_camera', '非当前输入'), ('pixels', '原图或完整画幅'), ('offcenter', '偏心主点'),
    ('rotation', 'SO\\(3\\)'), ('duplicate_camera', '唯一映射'), ('foreign_observation', '非输入视角'),
    ('duplicate_observation', '重复的数字索引'), ('confidence', '无效值'), ('baseline', '位移基线')])
def test_export_rejects_ambiguous_or_foreign_geometry(tmp_path, corruption, match):
    g, paths = fixture_geometry(tmp_path)
    if corruption == 'foreign_camera': g['cameras'][0]['image_path'] = str(tmp_path/'heldout.png')
    if corruption == 'pixels': g['images'][0][0, 0] = 0
    if corruption == 'offcenter': g['cameras'][0]['intrinsics'][0, 2] += .5
    if corruption == 'rotation': g['cameras'][0]['world_to_camera'][0, 0] = 2
    if corruption == 'duplicate_camera': g['cameras'][1] = copy.deepcopy(g['cameras'][0])
    if corruption == 'foreign_observation': g['observations'][0]['2'] = [1, 1]
    if corruption == 'duplicate_observation': g['observations'][0]['00'] = [1, 1]
    if corruption == 'confidence': g['confidence'][0] = np.nan
    if corruption == 'baseline': g['cameras'][1]['world_to_camera'] = np.eye(4)
    with pytest.raises(ValueError, match=match): export_initialization_dataset(g, paths, tmp_path/'dataset')


def test_export_refuses_existing_scene_and_honors_cancellation(tmp_path):
    g, paths = fixture_geometry(tmp_path)
    target = tmp_path/'dataset'; target.mkdir(); (target/'old-camera.txt').write_text('foreign')
    with pytest.raises(ValueError, match='目录必须为空'): export_initialization_dataset(g, paths, target)
    with pytest.raises(RuntimeError, match='取消'): export_initialization_dataset(g, paths, tmp_path/'cancelled', lambda: True)


def test_anchor_occlusion_missing_view_and_inferred_confidence(tmp_path):
    g, paths = fixture_geometry(tmp_path)
    # Behind the first point on the same ray: its nearer occluder is retained.
    g['points'][1] = g['points'][0] * 2
    g['observations'][1]['0'] = g['observations'][0]['0']
    g['observations'][2].pop('0')  # no detection/observation is no target
    exported = export_initialization_dataset(g, paths, tmp_path/'dataset')
    result = build_sparse_depth_anchors(exported, tmp_path/'anchors')
    view = result['metadata']['views'][paths[0].name]
    assert view['count'] == 2 and view['observed'] == view['inferred'] == 1
    assert result['metadata']['unobserved_pixels_supervised'] is False
    with np.load(tmp_path/'anchors'/view['path']) as data:
        assert np.allclose(sorted(data['weight']), [.15, 1.])
        assert np.allclose(data['inverse_depth'], 1/3)


def test_depth_hook_gradient_only_at_supported_pixels_and_scale_invariance(tmp_path):
    import torch
    g, paths = fixture_geometry(tmp_path)
    exported = export_initialization_dataset(g, paths, tmp_path/'dataset')
    anchors = build_sparse_depth_anchors(exported, tmp_path/'anchors')
    scope = {'torch': torch}
    exec(_DEPTH_HOOK.replace('__ANCHOR_MANIFEST__', repr(anchors['manifest_path'])), scope)
    depth = torch.zeros((1, 48, 64), requires_grad=True)
    loss = scope['splat_sparse_depth_loss'](depth, paths[0].name); loss.backward()
    assert loss.detach().item() > 0 and int(torch.count_nonzero(depth.grad)) == 4
    assert scope['splat_sparse_depth_loss'](torch.full_like(depth, 1/3), paths[0].name).item() < 1e-12
    with pytest.raises(RuntimeError, match='missing from input-only'): scope['splat_sparse_depth_loss'](depth, 'heldout.png')
    exported['points'] *= 5
    for camera in exported['cameras']:
        matrix = np.array(camera['world_to_camera']); matrix[:3, 3] *= 5; camera['world_to_camera'] = matrix.tolist()
    scaled = build_sparse_depth_anchors(exported, tmp_path/'scaled')
    scope2 = {'torch': torch}; exec(_DEPTH_HOOK.replace('__ANCHOR_MANIFEST__', repr(scaled['manifest_path'])), scope2)
    assert np.isclose(loss.detach().item(), scope2['splat_sparse_depth_loss'](torch.zeros_like(depth), paths[0].name).item())


def test_training_patch_uses_unique_markers_preserves_sampling_and_no_global_env(tmp_path):
    source = PROJECT/'vendor/gaussian-splatting/train.py'; before = file_sha256(source)
    patched = tmp_path/'train.py'
    meta = patch_training_source(source, patched, anchor_manifest=tmp_path/'manifest.json', priority_dir=tmp_path/'mask dir')
    text = patched.read_text()
    assert file_sha256(source) == before == meta['original_train_sha256']
    assert 'viewpoint_stack.pop(rand_idx)' in text
    assert 'splat_sparse_depth_loss(render_pkg["depth"], viewpoint_cam.image_name)' in text
    assert 'SPLAT_PRIORITY_MASK_DIR' not in text
    assert meta['sparse_depth_weight'] == .02 and meta['priority_loss'] is True
    invalid = tmp_path/'changed.py'; invalid.write_text(source.read_text().replace('        loss.backward()', '        loss.backward()\n        loss.backward()'))
    with pytest.raises(RuntimeError, match='backward marker'): patch_training_source(invalid, patched, anchor_manifest='x')
    with pytest.raises(ValueError, match='finite'): patch_training_source(source, patched, depth_weight=float('nan'))


@pytest.mark.parametrize('count,config,colmap,expected', [
    (2, {}, True, True), (8, {}, True, False), (8, {'sparse': True}, True, True),
    (16, {'geometry_backend': 'dust3r'}, True, True), (70, {}, False, True), (81, {}, False, False),
    (8, {'sparse_completion': 'learned_visible'}, True, True)])
def test_shared_routing(count, config, colmap, expected):
    assert upstream.use_shared_initialization(count, config, colmap) is expected


def test_upstream_consumes_shared_geometry_without_colmap_and_retains_provenance(tmp_path, monkeypatch):
    import torch
    from backend import reconstruction
    g, paths = fixture_geometry(tmp_path)
    calls = []; init_calls = []
    def initialize(inputs, config, progress, cancelled):
        init_calls.append(list(inputs)); return g
    def run(argv, cwd, log, progress, cancelled, stage, iteration_total=None):
        calls.append(argv)
        assert 'colmap' not in str(argv[0])
        assert '--densify_until_iter' in argv and argv[argv.index('--densify_until_iter')+1] == '3500'
        dataset = Path(argv[argv.index('-s')+1])
        assert (dataset/'sparse/0/points3D.bin').is_file()
        assert sorted(p.name for p in (dataset/'images').iterdir()) == sorted(p.name for p in paths)
        assert 'splat_sparse_depth_loss' in Path(argv[4]).read_text()
        assert 'install_gaussian_lineage' in Path(argv[4]).read_text()
        # A synthetic trainer output isolates launch/sidecar integration; no CUDA training occurs.
        from backend.gaussian_io import write_ply
        from backend.gaussian_lineage import write_lineage_sidecar
        target=Path(argv[argv.index('-m')+1])/'point_cloud/iteration_7000/point_cloud.ply'
        target.parent.mkdir(parents=True)
        write_ply({'gaussians':[{'position':p.tolist(),'color':[.5,.5,.5],'scale':[.01]*3,
            'rotation':[1,0,0,0],'opacity':.5} for p in g['points']]},target)
        write_lineage_sidecar(target,np.arange(len(g['points'])),np.array([1,1,2,2]),{'test_fixture':True})
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: True)
    monkeypatch.setattr(upstream.shutil, 'which', lambda *_: None)
    monkeypatch.setattr(reconstruction, 'initialize_geometry', initialize)
    monkeypatch.setattr(upstream, 'run_logged', run)
    result = upstream.reconstruct_upstream(paths, tmp_path/'run', {'mode': 'fast', 'sparse': True}, lambda *_: None, lambda: False)
    scene = json.loads(Path(result['scene_path']).read_text())
    assert init_calls == [paths] and len(calls) == 1
    assert scene['metadata']['sparse_depth']['total_anchors'] == 8
    assert scene['metadata']['initialization']['initialization_source_counts']['inferred'] == 2
    assert [p['source'] for p in scene['gaussians']]==g['sources']
    assert scene['metadata']['trainer_adapter']['initialization_lineage'] is True
    assert scene['metadata']['sparse_completion']['unseen_surfaces_recovered'] is False
    assert len(scene['cameras']) == 2


def test_external_completion_cannot_silently_disappear(tmp_path):
    with pytest.raises(RuntimeError, match='外部背面补全'):
        upstream.reconstruct_upstream([], tmp_path, {'completion': 'learned'}, lambda *_: None, lambda: False)


def test_densification_is_tighter_without_changing_training_budgets():
    assert [sparse_densification_settings(n)['densify_until_iter'] for n in (7000, 15000, 22000)] == [3500, 6000, 6000]
