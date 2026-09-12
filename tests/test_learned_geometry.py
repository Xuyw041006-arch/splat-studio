import importlib.util
import hashlib
import io
import json
from pathlib import Path

import numpy as np
import pytest

from backend import learned_geometry as learned


@pytest.mark.parametrize('shape', [(730, 988), (731, 1003), (1000, 1000), (988, 730)])
def test_padding_preserves_full_image_and_pixel_roundtrip(shape):
    rgb = np.full((*shape, 3), 180, np.uint8)
    prep = learned.preprocess_image(rgb)
    metadata = prep['metadata']
    assert not metadata['crop_applied']
    assert prep['valid'].sum() == np.prod(metadata['resized_size'])
    assert all(value % 16 == 0 for value in metadata['tensor_size'])
    h, w = shape
    pixels = np.array([[0, 0], [w - 1, 0], [0, h - 1], [w - 1, h - 1], [w / 2, h / 2]])
    mapped = learned.transform_pixels(pixels, prep['affine'])
    np.testing.assert_allclose(learned.transform_pixels(mapped, prep['inverse_affine']), pixels, atol=1e-10)
    x, y = np.rint(mapped).astype(int).T
    assert prep['valid'][y, x].all()
    principal = learned._model_principal_point(prep)
    intrinsic = np.array([[400, 0, principal[0]], [0, 400, principal[1]], [0, 0, 1]])
    original_k = prep['inverse_affine'] @ intrinsic
    np.testing.assert_allclose(original_k[:2, 2], [w / 2, h / 2], atol=1e-10)


def _fixture_plane():
    images = [np.full((64, 64, 3), 100, np.uint8) for _ in range(2)]
    preps = [learned.preprocess_image(image, 64) for image in images]
    intrinsic = np.array([[60., 0, 32], [0, 60., 32], [0, 0, 1]])
    poses = [np.eye(4), np.eye(4)]
    poses[1][0, 3] = .25
    depths = [np.full((64, 64), 3., np.float32) for _ in images]
    points = [learned._backproject(depth, intrinsic, pose) for depth, pose in zip(depths, poses)]
    confidence = [np.full((64, 64), 5., np.float32) for _ in images]
    return images, preps, [intrinsic] * 2, poses, depths, points, confidence


def test_visible_points_require_independent_view_depth_support():
    images, preps, ks, poses, depths, points, confidence = _fixture_plane()
    xyz, rgb, score, observations, stats = learned._assemble_points(points, depths, ks, poses, confidence, preps, images, {'geometry_max_points': 128}, None)
    assert len(xyz) == 128 and rgb.shape == xyz.shape
    assert all(len(track) == 2 for track in observations)
    assert np.all(score <= .49)
    assert stats['min_view_support'] == 2
    for point, track in zip(xyz, observations):
        for key, pixel in track.items():
            i = int(key)
            local = np.linalg.inv(poses[i]) @ np.r_[point, 1]
            projected = ks[i] @ local[:3]
            np.testing.assert_allclose(projected[:2] / projected[2], pixel, atol=1e-5)
    depths[1][:] = 100
    points[1] = learned._backproject(depths[1], ks[1], poses[1])
    with pytest.raises(learned.LearnedGeometryError, match='two-view depth support'):
        learned._assemble_points(points, depths, ks, poses, confidence, preps, images, {}, None)


def test_pnp_recovers_known_camera_and_never_substitutes_identity():
    rng = np.random.default_rng(3)
    xyz = rng.uniform([-1, -.7, 3], [1, .7, 5], (200, 3)).astype(np.float32)
    intrinsic = np.array([[400., 0, 256], [0, 400., 192], [0, 0, 1]])
    moved = xyz + [.3, -.1, .05]
    q = moved @ intrinsic.T
    pixels = q[:, :2] / q[:, 2:]
    pose, stats = learned.checked_pnp(xyz, pixels, intrinsic)
    np.testing.assert_allclose(pose[:3, 3], [.3, -.1, .05], atol=1e-4)
    assert stats['inliers'] >= 195

    class FailedPnP:
        SOLVEPNP_SQPNP = 1
        @staticmethod
        def solvePnPRansac(*args, **kwargs):
            return False, None, None, None
    with pytest.raises(learned.LearnedGeometryError) as result:
        learned.checked_pnp(xyz, pixels, intrinsic, cv2_module=FailedPnP)
    assert result.value.code == 'learned_pnp_failed'


def test_pnp_rejects_all_nan_or_insufficient_points():
    with pytest.raises(learned.LearnedGeometryError):
        learned.checked_pnp(np.full((30, 3), np.nan), np.zeros((30, 2)), np.eye(3))


def test_alignment_samples_keep_track_identity_and_ignore_padding():
    images, preps, ks, poses, depths, points, confidence = _fixture_plane()
    tracks = [{'0': [0., 0.], '1': [20., 21.]}, {'0': [-1., 10.]}, {'0': [64., 20.]}, {'0': [np.nan, 0.]}]
    samples = learned.alignment_samples(tracks, points, confidence, preps)
    assert len(samples) == 2
    assert [sample['track_index'] for sample in samples] == [0, 0]
    np.testing.assert_allclose(samples[1]['point'], points[1][21, 20])
    assert samples[1]['pixel'] == [20., 21.]


def test_availability_does_not_download_and_requires_explicit_boolean(tmp_path, monkeypatch):
    (tmp_path / 'dust3r/dust3r').mkdir(parents=True)
    (tmp_path / 'dust3r/croco/models').mkdir(parents=True)
    (tmp_path / 'dust3r/dust3r/model.py').write_text('')
    (tmp_path / 'dust3r/croco/models/croco.py').write_text('')
    monkeypatch.setattr(importlib.util, 'find_spec', lambda name: object())
    monkeypatch.setattr(learned, 'download_checkpoint', lambda *args: pytest.fail('Availability triggered a download'))
    config = {'geometry_cache_dir': str(tmp_path)}
    assert learned.geometry_availability(config)['available'] is False
    assert learned.geometry_availability({**config, 'allow_geometry_download': 'false'})['available'] is False
    status = learned.geometry_availability({**config, 'allow_geometry_download': True})
    assert status['available'] and status['requires_download']
    assert not learned.geometry_availability({**config, 'allow_geometry_download': True, 'geometry_model_repo': 'arbitrary/model'})['available']


def test_disconnected_camera_graph_is_not_a_success():
    reached, connected = learned._connectivity(3, [(0, 1)])
    assert not connected and reached == [0, 1]


def test_cancellation_and_calibration_rejection_precede_model_loading(monkeypatch):
    monkeypatch.setattr(learned, 'verify_assets', lambda *args: pytest.fail('Unexpected model access'))
    with pytest.raises(learned.ReconstructionCancelled):
        learned.initialize_learned_geometry(['x', 'y'], cancelled=lambda: True)
    with pytest.raises(learned.LearnedGeometryError) as result:
        learned.initialize_learned_geometry(['x', 'y'], {'intrinsics_original': [np.eye(3).tolist()]})
    assert result.value.code == 'learned_calibration_unsupported'


def test_exact_duplicate_inputs_rejected_without_model_load(tmp_path, monkeypatch):
    first, second = tmp_path / 'a.jpg', tmp_path / 'b.jpg'
    first.write_bytes(b'same'); second.write_bytes(b'same')
    monkeypatch.setattr(learned, 'verify_assets', lambda *args: pytest.fail('Unexpected model access'))
    with pytest.raises(learned.LearnedGeometryError) as result:
        learned.initialize_learned_geometry([str(first), str(second)])
    assert result.value.code == 'learned_duplicate_views'


@pytest.mark.parametrize('sources', [['heuristic', 'heuristic'], ['heuristic', 'exif_35mm_approximate']])
def test_approximate_app_intrinsics_do_not_block_learned_estimation(tmp_path, monkeypatch, sources):
    first, second = tmp_path / 'a.jpg', tmp_path / 'b.jpg'
    first.write_bytes(b'first'); second.write_bytes(b'second')
    def no_model(*args):
        raise RuntimeError('reached model validation')
    monkeypatch.setattr(learned, 'verify_assets', no_model)
    config = {'intrinsics_original': [np.eye(3).tolist()] * 2, 'intrinsics_sources': sources}
    with pytest.raises(RuntimeError, match='reached model validation'):
        learned.initialize_learned_geometry([str(first), str(second)], config)


def test_one_user_calibrated_camera_is_never_ignored():
    with pytest.raises(learned.LearnedGeometryError) as result:
        learned.initialize_learned_geometry(['a', 'b'], {'intrinsics_original': [np.eye(3).tolist()] * 2,
                                                       'intrinsics_sources': ['heuristic', 'user_calibrated']})
    assert result.value.code == 'learned_calibration_unsupported'


def _portable_source(tmp_path, monkeypatch):
    repo = tmp_path / 'portable'
    (repo / 'dust3r').mkdir(parents=True)
    (repo / 'dust3r/model.py').write_bytes(b'# pinned model source\n')
    manifest = {'code_commit': learned.CODE_COMMIT, 'croco_commit': learned.CROCO_COMMIT,
                'files': {'dust3r/model.py': learned.file_sha256(repo / 'dust3r/model.py')}}
    content = (json.dumps(manifest, sort_keys=True, indent=2) + '\n').encode()
    (repo / 'PINNED_SOURCE_MANIFEST.json').write_bytes(content)
    monkeypatch.setattr(learned, 'SOURCE_MANIFEST_SHA256', hashlib.sha256(content).hexdigest())
    monkeypatch.setattr(learned.subprocess, 'check_output', lambda *args, **kwargs: pytest.fail('Portable source required Git'))
    return repo, manifest


def test_portable_source_without_git_verifies_pinned_files_and_rejects_mutation(tmp_path, monkeypatch):
    repo, manifest = _portable_source(tmp_path, monkeypatch)
    assert learned.verify_source(repo) == manifest
    (repo / 'dust3r/model.py').write_bytes(b'# changed source\n')
    with pytest.raises(learned.LearnedGeometryError, match='source mismatch'):
        learned.verify_source(repo)


@pytest.mark.parametrize('extra', ['dust3r/unlisted.py', 'dust3r/native.so', 'dust3r/native.pyd', 'loose.pyc'])
def test_portable_source_rejects_unregistered_executable_files(tmp_path, monkeypatch, extra):
    repo, _ = _portable_source(tmp_path, monkeypatch)
    (repo / extra).write_bytes(b'unregistered executable')
    with pytest.raises(learned.LearnedGeometryError, match='Unregistered executable'):
        learned.verify_source(repo)


def test_portable_manifest_cannot_be_resigned_to_accept_modified_source(tmp_path, monkeypatch):
    repo, manifest = _portable_source(tmp_path, monkeypatch)
    (repo / 'dust3r/model.py').write_text('modified')
    manifest['files']['dust3r/model.py'] = learned.file_sha256(repo / 'dust3r/model.py')
    (repo / 'PINNED_SOURCE_MANIFEST.json').write_text(json.dumps(manifest))
    with pytest.raises(learned.LearnedGeometryError, match='manifest missing or modified'):
        learned.verify_source(repo)


def test_portable_source_rejects_external_symlink(tmp_path, monkeypatch):
    repo, _ = _portable_source(tmp_path, monkeypatch)
    external = tmp_path / 'external'
    external.mkdir()
    (repo / 'unregistered').symlink_to(external, target_is_directory=True)
    with pytest.raises(learned.LearnedGeometryError, match='Symlinked'):
        learned.verify_source(repo)


def test_readonly_source_environment_is_independent_of_writable_model_cache(tmp_path, monkeypatch):
    cache, bundle, explicit = tmp_path / 'cache', tmp_path / 'resources/dust3r', tmp_path / 'explicit'
    monkeypatch.setenv('SPLAT_GEOMETRY_HOME', str(cache))
    monkeypatch.setenv('SPLAT_GEOMETRY_REPO', str(bundle))
    root, repo, model = learned._paths({})
    assert root == cache and repo == bundle and model == cache / 'models/dust3r-512-dpt'
    assert learned._paths({'geometry_repo_path': str(explicit)})[1] == explicit


def test_cancelled_checkpoint_download_never_installs_partial_weights(tmp_path, monkeypatch):
    import urllib.request
    config = b'c' * 450
    weights = b'w' * (2 * 1024 * 1024)
    monkeypatch.setattr(learned, 'CONFIG_SHA256', hashlib.sha256(config).hexdigest())
    monkeypatch.setattr(learned, 'MODEL_SHA256', hashlib.sha256(weights).hexdigest())
    monkeypatch.setattr(learned, 'MODEL_BYTES', len(weights))
    state = {'cancel': False}
    class CancelAfterRead(io.BytesIO):
        def read(self, size=-1):
            value = super().read(size)
            state['cancel'] = True
            return value
    def response(request, timeout):
        return io.BytesIO(config) if request.full_url.endswith('config.json') else CancelAfterRead(weights)
    monkeypatch.setattr(urllib.request, 'urlopen', response)
    with pytest.raises(learned.ReconstructionCancelled):
        learned.download_checkpoint(tmp_path, cancelled=lambda: state['cancel'])
    assert (tmp_path / 'config.json').read_bytes() == config
    assert not (tmp_path / 'model.safetensors').exists()
    assert (tmp_path / 'model.safetensors.download').stat().st_size < len(weights)


def test_existing_bad_checkpoint_is_preserved_and_rejected(tmp_path, monkeypatch):
    import urllib.request
    (tmp_path / 'config.json').write_bytes(b'existing unrelated data')
    monkeypatch.setattr(urllib.request, 'urlopen', lambda *args, **kwargs: pytest.fail('Existing bad checkpoint was overwritten'))
    with pytest.raises(learned.LearnedGeometryError, match='Existing checkpoint'):
        learned.download_checkpoint(tmp_path)
    assert (tmp_path / 'config.json').read_bytes() == b'existing unrelated data'


def test_pair_direction_ratio_tie_prefers_more_checked_inliers():
    sparse = {'diagnostic': {'inlier_ratio': 1., 'inliers': 496, 'median_reprojection_px': .35}}
    dense = {'diagnostic': {'inlier_ratio': 1., 'inliers': 6000, 'median_reprojection_px': .48}}
    assert learned._best_checked_pair([sparse, dense]) is dense


def test_nonempty_point_cloud_does_not_certify_an_unsupported_third_camera():
    images, preps, ks, poses, depths, points, confidence = _fixture_plane()
    images.append(images[0].copy()); preps.append(preps[0]); ks.append(ks[0])
    third_pose = np.eye(4)
    third_pose[0, 3] = 1000
    poses.append(third_pose); depths.append(depths[0].copy()); confidence.append(confidence[0].copy())
    points.append(learned._backproject(depths[2], ks[2], poses[2]))
    with pytest.raises(learned.LearnedGeometryError, match='connected camera graph') as error:
        learned._assemble_points(points, depths, ks, poses, confidence, preps, images, {}, None)
    assert error.value.diagnostics['supported_points'] > 32
    assert error.value.diagnostics['registered_component'] == [0, 1]


def test_enlarged_small_images_have_no_negative_original_pixel_observations():
    images = [np.full((32, 32, 3), 100, np.uint8) for _ in range(2)]
    preps = [learned.preprocess_image(image, 64) for image in images]
    k = np.array([[60., 0, 32.5], [0, 60., 32.5], [0, 0, 1]])
    poses = [np.eye(4), np.eye(4)]
    poses[1][0, 3] = .25
    depths = [np.full((64, 64), 3., np.float32) for _ in images]
    points = [learned._backproject(depth, k, pose) for depth, pose in zip(depths, poses)]
    confidence = [np.full((64, 64), 5., np.float32) for _ in images]
    _, _, _, tracks, _ = learned._assemble_points(points, depths, [k, k], poses, confidence, preps, images, {}, None)
    assert all(0 <= value < 32 for track in tracks for pixel in track.values() for value in pixel)


def test_global_initialization_bypasses_upstream_identity_fallback_and_checks_all_cameras():
    import torch
    from types import SimpleNamespace
    images, preps, ks, poses, depths, pointmaps, confidence = _fixture_plane()
    scene = SimpleNamespace(imshapes=[(64, 64)] * 2, edges=[(0, 1)], pred_i={}, pred_j={}, conf_i={}, conf_j={},
                            im_conf=[torch.tensor(value) for value in confidence], min_conf_thr=3., device='cpu')
    called = {}
    def tree(*args, **kwargs):
        assert kwargs['has_im_poses'] is False
        return [torch.tensor(pointmap) for pointmap in pointmaps], [(0, 1)], None, None
    def apply(target, points, focals, camera_to_world):
        called.update({'points': points, 'focals': focals, 'poses': camera_to_world.numpy()})
    accepted = [{'i': 0, 'j': 1, 'intrinsics': ks}]
    result = learned._checked_global_initialization(scene, accepted, preps, 3., None, tree, apply)
    np.testing.assert_allclose(called['poses'], poses, atol=1e-4)
    assert len(result['cameras']) == 2 and result['identity_pose_fallback_allowed'] is False
    scene.im_conf[1][:] = 1.
    called.clear()
    with pytest.raises(learned.LearnedGeometryError) as failure:
        learned._checked_global_initialization(scene, accepted, preps, 3., None, tree, apply)
    assert failure.value.code == 'learned_pnp_failed'
    assert called == {}
