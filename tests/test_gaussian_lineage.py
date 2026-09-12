import ast
import hashlib
import json
from pathlib import Path
import sys
import types

import numpy as np
import pytest
from plyfile import PlyData, PlyElement

from backend import gaussian_lineage as lineage
from backend.gaussian_io import read_ply

ROOT = Path(__file__).resolve().parents[1] / 'vendor/gaussian-splatting'


def _seed(tmp_path, count=5):
    points = np.array([[float(i), .1 * i, 3.] for i in range(count)], np.float64)
    sources = np.array(['observed', 'inferred', 'unknown', 'observed', 'inferred'][:count])
    colors = np.array([[.1 + i * .1, .2, .3] for i in range(count)], np.float32)
    npz = tmp_path / 'initialization.npz'
    np.savez_compressed(npz, points=points, colors=colors, sources=sources)
    data = np.empty(count, dtype=[(name, 'f4') for name in ('x', 'y', 'z')])
    for i, name in enumerate(('x', 'y', 'z')):
        data[name] = points[:, i]
    ply = tmp_path / 'seed.ply'
    PlyData([PlyElement.describe(data, 'vertex')]).write(ply)
    return npz, ply, points, colors, sources


@pytest.fixture
def pinned_cpu_model(monkeypatch):
    """Execute pinned class bodies unchanged, with CPU allocation only.

    CUDA allocation and the initial nearest-neighbor scale kernel are isolated
    here; real clone/split/prune and optimizer row operations remain upstream.
    This is not a CUDA training or rasterization test.
    """
    import torch
    source_path = ROOT / 'scene/gaussian_model.py'
    tree = ast.parse(source_path.read_text())
    model = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == 'GaussianModel')
    code = compile(ast.Module(body=[model], type_ignores=[]), str(source_path), 'exec')
    class CpuAllocation:
        def __getattr__(self, name):
            value = getattr(torch, name)
            if name in {'zeros', 'ones', 'empty', 'tensor', 'eye'}:
                def factory(*args, **kwargs):
                    if kwargs.get('device') == 'cuda': kwargs['device'] = 'cpu'
                    return value(*args, **kwargs)
                return factory
            return value
    monkeypatch.setattr(torch.Tensor, 'cuda', lambda self, *args, **kwargs: self)
    module = types.ModuleType('_splat_lineage_pinned_cpu_fixture')
    module.__file__ = str(source_path)
    monkeypatch.setitem(sys.modules, module.__name__, module)
    module.__dict__.update(torch=CpuAllocation(), np=np, nn=torch.nn, os=__import__('os'), json=json,
        BasicPointCloud=object, RGB2SH=lambda rgb: (rgb - .5) / .28209479177387814,
        distCUDA2=lambda points: torch.ones(len(points)),
        inverse_sigmoid=lambda value: torch.log(value / (1 - value)),
        build_rotation=lambda rotation: torch.eye(3).repeat(len(rotation), 1, 1),
        build_scaling_rotation=lambda *args: None, strip_symmetric=lambda value: value,
        mkdir_p=lambda path: Path(path).mkdir(parents=True, exist_ok=True),
        PlyData=PlyData, PlyElement=PlyElement)
    exec(code, module.__dict__)
    return module.GaussianModel(0)


def _install_and_initialize(model, tmp_path):
    import torch
    npz, ply, points, colors, sources = _seed(tmp_path)
    state = lineage.install_gaussian_lineage(model, seed_npz=npz, seed_ply=ply, upstream_root=ROOT,
        expected_npz_sha256=lineage.file_sha256(npz), expected_ply_sha256=lineage.file_sha256(ply))
    model.create_from_pcd(types.SimpleNamespace(points=points, colors=colors), [types.SimpleNamespace(image_name='input.png')], 1.)
    parameters = [('xyz', '_xyz'), ('f_dc', '_features_dc'), ('f_rest', '_features_rest'),
                  ('opacity', '_opacity'), ('scaling', '_scaling'), ('rotation', '_rotation')]
    model.optimizer = torch.optim.Adam([{'name': name, 'params': [getattr(model, attr)]} for name, attr in parameters])
    model.percent_dense = .1
    model.xyz_gradient_accum = torch.zeros((5, 1))
    model.denom = torch.zeros((5, 1))
    model.tmp_radii = torch.ones(5)
    with torch.no_grad(): model._scaling[:] = torch.log(torch.tensor([.01, .2, .01, .2, .2]))[:, None]
    return state, sources


@pytest.mark.parametrize('optimizer_has_state', [False, True])
def test_real_pinned_clone_split_prune_inherit_exact_ids_and_preview_rows(tmp_path, pinned_cpu_model, optimizer_has_state):
    import torch
    model = pinned_cpu_model
    state, sources = _install_and_initialize(model, tmp_path)
    original_features = model._features_dc.detach().clone()
    if optimizer_has_state:
        for group in model.optimizer.param_groups:
            group['params'][0].grad = torch.zeros_like(group['params'][0])
        model.optimizer.step()
    model.densify_and_clone(torch.ones((5, 1)), .5, 1.)
    assert state['ids'].tolist() == [0, 1, 2, 3, 4, 0, 2]
    # Actual upstream split repeats selected [1,3,4] in blocks, three times.
    model.densify_and_split(torch.ones((5, 1)), .5, 1., N=3)
    expected = [0, 2, 0, 2, 1, 3, 4, 1, 3, 4, 1, 3, 4]
    assert state['ids'].tolist() == expected
    torch.testing.assert_close(model._features_dc, original_features[state['ids']])
    mask = torch.tensor([i % 3 == 0 for i in range(len(expected))])
    model.prune_points(mask)
    expected = [value for i, value in enumerate(expected) if not mask[i]]
    assert state['ids'].tolist() == expected
    torch.testing.assert_close(model._features_dc, original_features[state['ids']])
    output = tmp_path / 'final.ply'
    model.save_ply(output)
    original_output = tmp_path / 'original-save.ply'
    type(model).save_ply(model, original_output)
    assert output.read_bytes() == original_output.read_bytes()
    ids, codes, manifest = lineage.load_lineage_sidecar(output)
    assert ids.tolist() == expected
    assert [lineage.SOURCE_NAMES[value] for value in codes] == sources[ids].tolist()
    assert manifest['operation_counts'] == {'clone_appended': 2, 'split_appended': 9, 'pruned': 8}
    scene = read_ply(output, max_points=3)
    lineage.apply_lineage_to_preview(scene, output)
    for point in scene['gaussians']:
        row = point['source_index']
        assert point['initial_point_index'] == expected[row]
        assert point['source'] == sources[expected[row]]
    assert not scene['metadata']['gaussian_lineage']['nearest_neighbor_mapping_used']


def test_zero_selected_densification_preserves_lineage(tmp_path, pinned_cpu_model):
    import torch
    state, _ = _install_and_initialize(pinned_cpu_model, tmp_path)
    pinned_cpu_model.densify_and_clone(torch.zeros((5, 1)), .5, 1.)
    pinned_cpu_model.densify_and_split(torch.zeros((5, 1)), .5, 1.)
    assert state['ids'].tolist() == list(range(5))
    assert state['pending'] is None


def test_untracked_mutations_resume_and_invalid_prune_are_rejected(tmp_path, pinned_cpu_model):
    import torch
    model = pinned_cpu_model
    state, _ = _install_and_initialize(model, tmp_path)
    for name in ('capture', 'restore', 'load_ply'):
        with pytest.raises(RuntimeError, match='resume'):
            getattr(model, name)()
    with pytest.raises(RuntimeError, match='exact parent IDs'):
        model.densification_postfix(torch.empty((0, 3)))
    with pytest.raises(RuntimeError, match='direct Gaussian append'):
        model.cat_tensors_to_optimizer({})
    with pytest.raises(RuntimeError, match='direct Gaussian pruning'):
        model._prune_optimizer(torch.ones(5, dtype=torch.bool))
    with pytest.raises(RuntimeError, match='prune mask'):
        model.prune_points(torch.zeros(4, dtype=torch.bool))
    assert state['ids'].tolist() == list(range(5))


def test_seed_permutation_or_changed_hash_is_never_mapped_by_nearest_neighbor(tmp_path):
    npz, ply, points, colors, sources = _seed(tmp_path)
    np.savez_compressed(npz, points=points[::-1], sources=sources[::-1])
    with pytest.raises(RuntimeError, match='exact float32 row order'):
        lineage.prepare_lineage_tracking(npz, ply, ROOT, tmp_path / 'run')
    with pytest.raises(RuntimeError, match='changed after'):
        lineage._load_seed(npz, ply, '0' * 64, lineage.file_sha256(ply))


def test_sidecar_rejects_ply_change_array_corruption_and_bad_preview_index(tmp_path, pinned_cpu_model):
    _install_and_initialize(pinned_cpu_model, tmp_path)
    output = tmp_path / 'final.ply'
    pinned_cpu_model.save_ply(output)
    scene = read_ply(output, max_points=2)
    scene['gaussians'][0]['source_index'] = 10000
    with pytest.raises(RuntimeError, match='source_index'):
        lineage.apply_lineage_to_preview(scene, output)
    sidecar = Path(str(output) + '.lineage.npz')
    original = sidecar.read_bytes()
    sidecar.write_bytes(original + b'changed')
    with pytest.raises(RuntimeError, match='checksum'):
        lineage.load_lineage_sidecar(output)
    sidecar.write_bytes(original)
    output.write_bytes(output.read_bytes() + b'changed')
    with pytest.raises(RuntimeError, match='current full PLY'):
        lineage.load_lineage_sidecar(output)


def test_pinned_source_hash_and_markers_reject_changed_vendor(tmp_path):
    for relative in lineage.PINNED_SOURCE_SHA256:
        destination = tmp_path / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes((ROOT / relative).read_bytes())
    lineage.verify_upstream_sources(tmp_path)
    target = tmp_path / 'scene/gaussian_model.py'
    target.write_text(target.read_text().replace('.repeat(N, 1)', '.repeat_interleave(N, dim=0)'))
    with pytest.raises(RuntimeError, match='exact pinned'):
        lineage.verify_upstream_sources(tmp_path)


def test_prepare_produces_unindented_compilable_hash_guarded_runtime_hook(tmp_path):
    npz, ply, *_ = _seed(tmp_path)
    record = lineage.prepare_lineage_tracking(npz, ply, ROOT, tmp_path / 'run')
    assert record['hook_source'].startswith('import ')
    assert record['runtime_sha256'] == hashlib.sha256(Path(record['runtime_path']).read_bytes()).hexdigest()
    compile(record['hook_source'], '<hook>', 'exec')
    assert 'scene = Scene' not in record['hook_source']
    assert record['resume_supported'] is False


def test_copied_runtime_hook_installs_without_application_imports_and_rejects_tampering(tmp_path, pinned_cpu_model):
    npz, ply, *_ = _seed(tmp_path)
    record = lineage.prepare_lineage_tracking(npz, ply, ROOT, tmp_path / 'run')
    exec(record['hook_source'], {'gaussians': pinned_cpu_model})
    assert pinned_cpu_model._splat_lineage_state['ids'] is None
    runtime = Path(record['runtime_path'])
    runtime.write_bytes(runtime.read_bytes() + b'\n# altered\n')
    with pytest.raises(RuntimeError, match='runtime checksum'):
        exec(record['hook_source'], {'gaussians': object()})


def test_actual_loaded_seed_reordering_is_rejected_before_gaussian_creation(tmp_path, pinned_cpu_model):
    npz, ply, points, colors, _ = _seed(tmp_path)
    state = lineage.install_gaussian_lineage(pinned_cpu_model, seed_npz=npz, seed_ply=ply, upstream_root=ROOT,
        expected_npz_sha256=lineage.file_sha256(npz), expected_ply_sha256=lineage.file_sha256(ply))
    with pytest.raises(RuntimeError, match='exact initialization row order'):
        pinned_cpu_model.create_from_pcd(types.SimpleNamespace(points=points[::-1], colors=colors[::-1]), [], 1.)
    assert state['ids'] is None
    assert pinned_cpu_model.get_xyz.numel() == 0
