"""Exact initialization ancestry for the pinned 3DGS clone/split/prune code.

The runtime is self-contained so a copied CUDA trainer can import this file.
Labels describe an initial point's origin, not measured truth at an optimized
Gaussian's current position. No nearest-neighbor ancestry is ever inferred.
"""
from __future__ import annotations

import ast
import hashlib
import inspect
import json
from pathlib import Path
import textwrap
import types

import numpy as np

PINNED_SOURCE_SHA256 = {
    'scene/gaussian_model.py': '546ba41528a879af4d83bd280a3ef3cbcca9a5e9132204185aaa8f8ee9ffa310',
    'scene/__init__.py': '895d042f4dd2c6ee521dd985aea5c49d9c6f5c7379f76c4d537d8dc5468f18bf',
    'scene/dataset_readers.py': 'b06f01648fee893c4785ed38eeb934f91b79817b9dcbb694291c9162b0b6ba94',
}
SOURCE_NAMES = ('unknown', 'observed', 'inferred')
FORMAT = 'splat-gaussian-initialization-lineage/1'
SCOPE = ('Exact initial-point ancestry through clone/split/prune. Observed ancestry '
         'does not certify the optimized Gaussian as measured geometry; no unseen-surface recovery is asserted.')


def file_sha256(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def verify_upstream_sources(upstream_root):
    root = Path(upstream_root).resolve()
    for relative, expected in PINNED_SOURCE_SHA256.items():
        path = root / relative
        if not path.is_file() or file_sha256(path) != expected:
            raise RuntimeError('Lineage tracking requires the exact pinned upstream source: ' + relative)
    source = (root / 'scene/gaussian_model.py').read_text()
    tree = ast.parse(source)
    model = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == 'GaussianModel')
    methods = {node.name: ast.get_source_segment(source, node) for node in model.body if isinstance(node, ast.FunctionDef)}
    # These markers bind ancestry to the actual order used by the pinned code.
    for name, marker in (
        ('densify_and_clone', 'new_xyz = self._xyz[selected_pts_mask]'),
        ('densify_and_split', 'self.get_xyz[selected_pts_mask].repeat(N, 1)'),
        ('prune_points', 'valid_points_mask = ~mask'),
        ('save_ply', "PlyData([el]).write(path)"),
    ):
        if methods[name].count(marker) != 1:
            raise RuntimeError('Pinned lineage marker changed: ' + name)
    return root, methods


def _load_seed(seed_npz, seed_ply, expected_npz_sha256, expected_ply_sha256):
    from plyfile import PlyData
    if file_sha256(seed_npz) != expected_npz_sha256 or file_sha256(seed_ply) != expected_ply_sha256:
        raise RuntimeError('Initialization provenance or point cloud changed after lineage preparation')
    with np.load(seed_npz, allow_pickle=False) as data:
        points = np.asarray(data['points'], dtype=np.float32)
        sources = np.asarray(data['sources'])
    if points.ndim != 2 or points.shape[1] != 3 or len(points) < 1 or sources.shape != (len(points),):
        raise ValueError('Invalid initialization lineage array shapes')
    if not np.isfinite(points).all() or not set(sources.tolist()) <= set(SOURCE_NAMES):
        raise ValueError('Invalid initialization lineage coordinates or source labels')
    vertices = PlyData.read(str(seed_ply))['vertex']
    actual = np.stack([vertices[name] for name in ('x', 'y', 'z')], axis=1).astype(np.float32)
    if not np.array_equal(actual, points):
        raise RuntimeError('Initialization PLY and provenance NPZ disagree in exact float32 row order')
    codes = np.asarray([SOURCE_NAMES.index(str(value)) for value in sources], dtype=np.uint8)
    return points, codes


def _sidecar_paths(ply_path):
    path = Path(ply_path)
    return Path(str(path) + '.lineage.npz'), Path(str(path) + '.lineage.json')


def write_lineage_sidecar(ply_path, initial_point_indices, seed_source_codes, metadata):
    from plyfile import PlyData
    path = Path(ply_path)
    ids = np.asarray(initial_point_indices)
    seeds = np.asarray(seed_source_codes)
    if ids.ndim != 1 or ids.dtype.kind not in 'iu' or seeds.ndim != 1 or seeds.dtype.kind not in 'iu':
        raise ValueError('Lineage IDs and source codes must be one-dimensional integer arrays')
    if np.any(seeds > 2) or np.any(seeds < 0) or (len(ids) and (ids.min() < 0 or ids.max() >= len(seeds))):
        raise ValueError('Invalid lineage ancestor index or source code')
    if len(PlyData.read(str(path))['vertex']) != len(ids):
        raise RuntimeError('Lineage length does not match saved full PLY vertex count')
    sources = seeds[ids].astype(np.uint8)
    npz_path, json_path = _sidecar_paths(path)
    temporary = Path(str(npz_path) + '.tmp')
    with temporary.open('wb') as stream:
        np.savez_compressed(stream, initial_point_index=ids.astype(np.int64), source_code=sources)
    temporary.replace(npz_path)
    record = {**metadata, 'format': FORMAT, 'full_ply_sha256': file_sha256(path),
              'gaussian_count': len(ids), 'initial_point_count': len(seeds),
              'lineage_npz': npz_path.name, 'lineage_npz_sha256': file_sha256(npz_path),
              'source_encoding': {str(i): name for i, name in enumerate(SOURCE_NAMES)},
              'source_counts': {name: int(np.count_nonzero(sources == i)) for i, name in enumerate(SOURCE_NAMES)},
              'row_order': 'exact full PLY vertex order; preview must bind with source_index',
              'scope': SCOPE, 'nearest_neighbor_mapping_used': False}
    temporary_json = Path(str(json_path) + '.tmp')
    temporary_json.write_text(json.dumps(record, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    temporary_json.replace(json_path)
    return record


def load_lineage_sidecar(ply_path):
    from plyfile import PlyData
    npz_path, json_path = _sidecar_paths(ply_path)
    record = json.loads(json_path.read_text())
    if record.get('format') != FORMAT or record.get('full_ply_sha256') != file_sha256(ply_path):
        raise RuntimeError('Lineage manifest does not bind the current full PLY')
    if record.get('lineage_npz') != npz_path.name or record.get('lineage_npz_sha256') != file_sha256(npz_path):
        raise RuntimeError('Lineage array checksum or filename mismatch')
    with np.load(npz_path, allow_pickle=False) as data:
        ids, codes = data['initial_point_index'].copy(), data['source_code'].copy()
    count = len(PlyData.read(str(ply_path))['vertex'])
    if record.get('gaussian_count') != count or ids.shape != (count,) or codes.shape != (count,):
        raise RuntimeError('Lineage count does not match full PLY')
    if ids.dtype.kind not in 'iu' or codes.dtype.kind not in 'iu' or np.any(codes > 2) or np.any(codes < 0):
        raise RuntimeError('Lineage arrays have invalid types or source codes')
    seed_count = record.get('initial_point_count')
    if not isinstance(seed_count, int) or seed_count < 1 or (len(ids) and (ids.min() < 0 or ids.max() >= seed_count)):
        raise RuntimeError('Lineage ancestor indices are out of range')
    return ids, codes, record


def apply_lineage_to_preview(scene, ply_path):
    """Bind verified full-PLY rows; an absent or invalid sidecar is an error."""
    ids, codes, record = load_lineage_sidecar(ply_path)
    metadata = scene.setdefault('metadata', {})
    if metadata.get('source_ply_sha256') != record['full_ply_sha256'] or metadata.get('source_gaussian_count') != len(ids):
        raise RuntimeError('Preview does not refer to the lineage-bound full PLY')
    indices = [point.get('source_index') for point in scene.get('gaussians', [])]
    if any(type(index) is not int or not 0 <= index < len(ids) for index in indices) or len(set(indices)) != len(indices):
        raise RuntimeError('Preview source_index is missing, duplicated, or outside full PLY rows')
    for point, index in zip(scene['gaussians'], indices):
        point.update(source=SOURCE_NAMES[int(codes[index])], initial_point_index=int(ids[index]),
                     provenance_kind='initialization_lineage')
    metadata.update(gaussian_lineage=record, source_note=SCOPE)
    warning = '原始 PLY 未记录观测来源和语义置信度，均按未知处理。'
    metadata['warnings'] = [item for item in metadata.get('warnings', []) if item != warning]
    metadata['warnings'].append('来源颜色表示初始化点的继承谱系；优化后的位置与分裂高斯不能据此称为实测表面。')
    return scene


def install_gaussian_lineage(model, *, seed_npz, seed_ply, upstream_root, expected_npz_sha256, expected_ply_sha256):
    """Install before Scene creates the initial Gaussians. Resume is unsupported."""
    import torch
    root, methods = verify_upstream_sources(upstream_root)
    model_path = root / 'scene/gaussian_model.py'
    if Path(inspect.getfile(type(model))).resolve() != model_path:
        raise RuntimeError('GaussianModel is not loaded from the verified pinned source')
    if hasattr(model, '_splat_lineage_state'):
        raise RuntimeError('Lineage tracking cannot be installed twice')
    method_names = ('create_from_pcd', 'densify_and_clone', 'densify_and_split', 'densification_postfix',
                    'cat_tensors_to_optimizer', 'prune_points', '_prune_optimizer', 'save_ply', 'capture', 'restore', 'load_ply')
    originals = {name: getattr(model, name) for name in method_names}
    for name, bound in originals.items():
        if (name in model.__dict__ or getattr(bound, '__func__', None) is not type(model).__dict__.get(name)
                or Path(bound.__func__.__code__.co_filename).resolve() != model_path):
            raise RuntimeError('GaussianModel method was replaced before lineage installation: ' + name)
    seed_points, seed_codes = _load_seed(seed_npz, seed_ply, expected_npz_sha256, expected_ply_sha256)
    state = {'ids': None, 'pending': None, 'pruning': False, 'valid': True,
             'clone_appended': 0, 'split_appended': 0, 'pruned': 0}
    model._splat_lineage_state = state

    def check():
        if not state['valid'] or state['ids'] is None or state['ids'].ndim != 1 or len(state['ids']) != model.get_xyz.shape[0]:
            raise RuntimeError('Gaussian lineage and parameter row order are not synchronized')

    def initialize(self, pcd, *args, **kwargs):
        if state['ids'] is not None:
            raise RuntimeError('Cannot replace a lineage-tracked initialization')
        actual = np.asarray(pcd.points, dtype=np.float32)
        if not np.array_equal(actual, seed_points):
            raise RuntimeError('Upstream loaded points disagree with exact initialization row order')
        result = originals['create_from_pcd'](pcd, *args, **kwargs)
        if not np.array_equal(self.get_xyz.detach().cpu().numpy(), seed_points):
            raise RuntimeError('Gaussian creation changed initialization row order')
        state['ids'] = torch.arange(len(seed_points), dtype=torch.long, device=self.get_xyz.device)
        check()
        return result

    def set_pending(self, selected, repeats, operation):
        check()
        if state['pending'] is not None or selected.dtype != torch.bool or selected.shape != state['ids'].shape:
            raise RuntimeError('Unexpected or overlapping Gaussian append operation')
        # Tensor.repeat repeats the complete selected block: [a,b,a,b].
        # repeat_interleave would incorrectly attach split children [a,a,b,b].
        state['pending'] = state['ids'][selected].repeat(repeats)
        state['pending_operation'] = operation

    def append(self, *args, **kwargs):
        check()
        parents = state['pending']
        if parents is None or not args or len(parents) != args[0].shape[0]:
            raise RuntimeError('Densification lacks exact parent IDs; refusing guessed ancestry')
        try:
            result = originals['densification_postfix'](*args, **kwargs)
            state['ids'] = torch.cat((state['ids'], parents))
            state[state['pending_operation'] + '_appended'] += len(parents)
            state['pending'] = None
            check()
            return result
        except Exception:
            state['valid'] = False
            raise

    def cat_guard(self, *args, **kwargs):
        if state['pending'] is None:
            raise RuntimeError('Untracked direct Gaussian append is unsupported')
        return originals['cat_tensors_to_optimizer'](*args, **kwargs)

    def prune(self, mask):
        check()
        if state['pending'] is not None or mask.dtype != torch.bool or mask.shape != state['ids'].shape:
            raise RuntimeError('Invalid Gaussian lineage prune mask')
        retained = state['ids'][~mask]
        before = len(state['ids'])
        state['pruning'] = True
        try:
            result = originals['prune_points'](mask)
            state['ids'] = retained
            state['pruned'] += before - len(retained)
            check()
            return result
        except Exception:
            state['valid'] = False
            raise
        finally:
            state['pruning'] = False

    def prune_guard(self, *args, **kwargs):
        if not state['pruning']:
            raise RuntimeError('Untracked direct Gaussian pruning is unsupported')
        return originals['_prune_optimizer'](*args, **kwargs)

    def save(self, path):
        check()
        if state['pending'] is not None:
            raise RuntimeError('Cannot save while a Gaussian lineage append is pending')
        result = originals['save_ply'](path)
        write_lineage_sidecar(path, state['ids'].detach().cpu().numpy(), seed_codes,
            {'initialization_npz_sha256': expected_npz_sha256, 'initialization_ply_sha256': expected_ply_sha256,
             'pinned_upstream_source_sha256': PINNED_SOURCE_SHA256,
             'operation_counts': {name: state[name] for name in ('clone_appended', 'split_appended', 'pruned')}})
        return result

    def no_resume(self, *args, **kwargs):
        raise RuntimeError('Checkpoint/PLY resume without a validated ancestry state is unsupported by lineage tracking')

    replacements = {'create_from_pcd': initialize, '_splat_lineage_set_pending': set_pending,
                    'densification_postfix': append, 'cat_tensors_to_optimizer': cat_guard,
                    'prune_points': prune, '_prune_optimizer': prune_guard, 'save_ply': save,
                    'capture': no_resume, 'restore': no_resume, 'load_ply': no_resume}
    for name, repeats, operation in (('densify_and_clone', '1', 'clone'), ('densify_and_split', 'N', 'split')):
        source = methods[name]
        marker = '        self.densification_postfix('
        if source.count(marker) != 1:
            raise RuntimeError('Upstream append marker changed: ' + name)
        source = source.replace(marker, f'        self._splat_lineage_set_pending(selected_pts_mask, {repeats}, {operation!r})\n' + marker)
        namespace = {}
        exec(compile(textwrap.dedent(source), str(model_path) + ':lineage:' + name, 'exec'),
             originals[name].__func__.__globals__, namespace)
        replacements[name] = namespace[name]
    for name, method in replacements.items():
        setattr(model, name, types.MethodType(method, model))
    if model.get_xyz.numel():
        raise RuntimeError('Install Gaussian lineage before Scene/create_from_pcd, not after an unknown initialization')
    return state


def prepare_lineage_tracking(initialization_npz, initialization_ply, upstream_root, output_dir):
    """Return an unindented hook for immediately before ``scene = Scene(...)``."""
    root, _ = verify_upstream_sources(upstream_root)
    seed_npz, seed_ply = Path(initialization_npz).resolve(), Path(initialization_ply).resolve()
    npz_sha, ply_sha = file_sha256(seed_npz), file_sha256(seed_ply)
    _load_seed(seed_npz, seed_ply, npz_sha, ply_sha)
    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    runtime = output / 'gaussian_lineage_runtime.py'
    contents = Path(__file__).read_bytes()
    if runtime.exists() and runtime.read_bytes() != contents:
        raise RuntimeError('Refusing to replace a different per-run lineage runtime')
    runtime.write_bytes(contents)
    runtime_sha = file_sha256(runtime)
    hook = '\n'.join([
        'import hashlib as _splat_lineage_hashlib',
        'import importlib.util as _splat_lineage_importlib',
        'from pathlib import Path as _SplatLineagePath',
        f'_splat_lineage_runtime_path = _SplatLineagePath({str(runtime)!r})',
        f'if _splat_lineage_hashlib.sha256(_splat_lineage_runtime_path.read_bytes()).hexdigest() != {runtime_sha!r}:',
        '    raise RuntimeError("Per-run lineage runtime checksum changed")',
        '_splat_lineage_spec = _splat_lineage_importlib.spec_from_file_location("splat_gaussian_lineage_runtime", _splat_lineage_runtime_path)',
        '_splat_lineage_runtime = _splat_lineage_importlib.module_from_spec(_splat_lineage_spec)',
        '_splat_lineage_spec.loader.exec_module(_splat_lineage_runtime)',
        f'_splat_lineage_runtime.install_gaussian_lineage(gaussians, seed_npz={str(seed_npz)!r}, seed_ply={str(seed_ply)!r}, upstream_root={str(root)!r}, expected_npz_sha256={npz_sha!r}, expected_ply_sha256={ply_sha!r})',
    ]) + '\n'
    compile(hook, '<lineage-hook>', 'exec')
    return {'hook_source': hook, 'runtime_path': str(runtime), 'runtime_sha256': runtime_sha,
            'initialization_npz_sha256': npz_sha, 'initialization_ply_sha256': ply_sha,
            'pinned_upstream_source_sha256': PINNED_SOURCE_SHA256, 'scope': SCOPE,
            'resume_supported': False, 'lineage_format': FORMAT}
