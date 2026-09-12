"""Synthetic CUDA integration smoke; NEVER a dataset accuracy result or user ETA.

Uses the production worker and genuine original 3DGS CUDA forward/backward.
Three translated cameras see a known synthetic Gaussian sheet. Explicit coarse,
object and part masks exercise instance association and parent declarations.
The fresh run uses visible geometry to support hierarchy penalties, while a
second explicitly seeded run separately exercises prior anchoring.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
import time
import traceback

COMMIT = '54c035f7834b564019656c3e3fcc3646292f727d'
MODULES = ['backend/__init__.py', 'backend/semantic_worker.py', 'backend/semantic_refinement.py',
           'backend/semantic_targets.py', 'backend/semantic_granularity.py', 'backend/semantic_projection.py',
           'backend/semantics.py', 'backend/semantic_views.py']


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, allow_nan=False, indent=2))


def fixture(folder):
    import numpy as np
    from PIL import Image
    folder.mkdir(parents=True, exist_ok=False)
    points = np.asarray([[x, y, 3.] for y in np.linspace(-.55, .55, 16)
                         for x in np.linspace(-.8, .8, 24)], dtype=np.float32)
    fields = ['x', 'y', 'z', 'nx', 'ny', 'nz', *[f'f_dc_{i}' for i in range(3)],
              *[f'f_rest_{i}' for i in range(45)], 'opacity', 'scale_0', 'scale_1', 'scale_2',
              'rot_0', 'rot_1', 'rot_2', 'rot_3']
    values = np.zeros((len(points), len(fields)), dtype=np.float32)
    values[:, :3] = points
    values[:, fields.index('opacity')] = math.log(.8 / .2)
    values[:, fields.index('scale_0'):fields.index('scale_2') + 1] = math.log(.035)
    values[:, fields.index('rot_0')] = 1
    ply = folder / 'synthetic_complete.ply'
    with ply.open('w') as stream:
        stream.write('ply\nformat ascii 1.0\ncomment complete synthetic integration fixture; no reconstructed dataset\n')
        stream.write(f'element vertex {len(points)}\n')
        stream.write(''.join(f'property float {name}\n' for name in fields) + 'end_header\n')
        np.savetxt(stream, values, fmt='%.9g')
    masks, cameras = [], []
    selections = [('assembly', 'coarse', 'assembly', np.ones(len(points), bool), None),
        ('left', 'object', 'panel', points[:, 0] < 0, 'assembly'),
        ('right', 'object', 'panel', points[:, 0] > 0, 'assembly'),
        ('left-corner', 'part', 'corner', (points[:, 0] < 0) & (points[:, 1] < -.15), 'left'),
        ('right-corner', 'part', 'corner', (points[:, 0] > 0) & (points[:, 1] < -.15), 'right')]
    for index, center_x in enumerate((-.12, 0., .12)):
        frame = f'view-{index}.png'
        matrix = np.eye(4); matrix[0, 3] = -center_x
        camera = {'frame_id': frame, 'image_name': frame, 'width': 128, 'height': 128,
                  'fx': 150., 'fy': 150., 'cx': 64., 'cy': 64., 'world_to_camera': matrix.tolist()}
        cameras.append(camera)
        xyz = points + np.array([-center_x, 0., 0.])
        uv = xyz[:, :2] / xyz[:, 2:] * 150 + 64
        for oid, level, label, chosen, parent in selections:
            low = np.floor(uv[chosen].min(axis=0) - 1).astype(int)
            high = np.ceil(uv[chosen].max(axis=0) + 1).astype(int)
            mask = np.zeros((128, 128), np.uint8)
            mask[max(0, low[1]):min(128, high[1] + 1), max(0, low[0]):min(128, high[0] + 1)] = 255
            target = folder / f'{index}-{oid}.png'; Image.fromarray(mask).save(target)
            record = {'frame_id': frame, 'image_name': frame, 'mask_id': f'{index}-{oid}',
                      'region_id': f'{index}-{oid}', 'object_id': oid, 'level': level, 'label': label,
                      'confidence': 1., 'annotation_source': 'manual', 'full_frame': True,
                      'coordinate_space': 'full_frame_pixels', 'mask_path': str(target)}
            if parent:
                record.update(parent_id=parent, parent_relation='part_of' if level == 'part' else 'member_of',
                              parent_region_id=f'{index}-{parent}', relation='part_of' if level == 'part' else 'member_of',
                              relation_source='explicit_synthetic_fixture')
            masks.append(record)
    write_json(folder / 'cameras.json', cameras); write_json(folder / 'masks.json', masks)
    return ply, points, cameras, masks


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--project', default='/content/semantic-v3-code')
    parser.add_argument('--upstream', default='/content/semantic-v3-upstream')
    parser.add_argument('--output', default='/content/semantic-v3-cuda-smoke')
    parser.add_argument('--preflight-only', action='store_true', help='CPU geometry fixture check, not CUDA validation')
    args = parser.parse_args()
    project = Path(args.project).resolve(); root = Path(args.output).resolve()
    if root.exists():
        raise FileExistsError(f'Use a new output directory: {root}')
    root.mkdir(parents=True)
    sys.path.insert(0, str(project))
    import numpy as np
    from backend.semantic_granularity import build_granularity_targets
    from backend.semantic_projection import projection_support_loader
    report = {'kind': 'synthetic_native_cuda_integration_smoke', 'dataset_benchmark': False,
              'dataset_miou_measured': False, 'user_runtime_prediction': False, 'runs': [], 'status': 'running',
              'source_code_sha256': {name: sha(project / name) for name in MODULES}}
    started = time.perf_counter()
    try:
        ply, points, cameras, masks = fixture(root / 'inputs')
        protected = {str(path): sha(path) for path in (root / 'inputs').glob('*') if path.is_file()}
        ply_sha = sha(ply)
        frame_names = [camera['frame_id'] for camera in cameras]
        loader = projection_support_loader(points, {c['frame_id']: c for c in cameras}, opacity=np.full(len(points), .8))
        built = build_granularity_targets(masks, {'geometry_binding': {'source_ply_sha256': ply_sha,
            'gaussian_count': len(points)}}, support_loader=loader, allowed_frames=frame_names)
        assert len(built['class_labels']) == 5, built['class_labels']
        assert len(built['hierarchy']['edges']) == 4, built['hierarchy']
        assert all(len(track['observations']) == 3 for track in built['tracks']), built['tracks']
        from backend.semantic_worker import _hierarchy_geometry_support
        from backend.semantic_targets import _read_mask
        geometric_support, geometric_audit = _hierarchy_geometry_support(built,
            [{**row, 'mask': _read_mask(row, None)} for row in masks], loader)
        assert len(geometric_support) == 4 and all(len(ids) > 0 for ids in geometric_support.values())
        report['fixture'] = {'gaussians': len(points), 'training_views': frame_names, 'explicit_regions': len(masks),
                             'associated_tracks': len(built['tracks']), 'hierarchy_edges': len(built['hierarchy']['edges']),
                             'source_ply_sha256': ply_sha, 'source_is_synthetic': True,
                             'hierarchy_visible_geometry_support': geometric_audit}
        if args.preflight_only:
            report.update(status='cpu_fixture_preflight_only', cuda_executed=False)
            write_json(root / 'summary.json', report); print(json.dumps(report, indent=2)); return
        import torch
        import diff_gaussian_rasterization._C as native
        from backend.semantic_worker import refine_upstream_semantics
        if not torch.cuda.is_available() or 'A100' not in torch.cuda.get_device_name(0):
            raise RuntimeError('This timing smoke requires a genuine A100 CUDA runtime')
        revision = subprocess.check_output(['git', '-C', args.upstream, 'rev-parse', 'HEAD'], text=True).strip()
        if revision != COMMIT:
            raise RuntimeError('Unexpected original 3DGS revision')
        report['environment'] = {'python': sys.version, 'torch': torch.__version__, 'torch_cuda': torch.version.cuda,
            'gpu': torch.cuda.get_device_name(0), 'native_rasterizer': str(native.__file__), 'upstream_commit': revision}
        prior = np.full((len(points), len(built['class_labels'])), .05, np.float32)
        for edge in built['hierarchy']['edges']:
            prior[:, built['class_labels'].index(edge['child'])] = .85
        prior_path = root / 'declared_synthetic_prior.npz'
        np.savez_compressed(prior_path, probabilities=prior, classes=np.asarray(built['class_labels']),
                            source_ply_sha256=np.asarray(ply_sha))
        base = {'semantic_steps': 400, 'train_size': 128, 'preset': 'improved', 'seed': 13,
                'upstream_repo': args.upstream, 'expected_gaussian_count': len(points),
                'source_ply_sha256': ply_sha, 'semantic_granularity': 'multilevel',
                'sampling_schedule': 'view_cycle', 'minimum_sweeps': 3, 'training_frames': frame_names}
        for name, prior_config in [('fresh_uniform', {}), ('declared_synthetic_seeded', {'prior': {
                'path': str(prior_path), 'classes': built['class_labels'], 'source_ply_sha256': ply_sha}})]:
            print('RUN', name, '400 synthetic CUDA steps', flush=True)
            torch.cuda.synchronize(); before = time.perf_counter()
            result = refine_upstream_semantics(ply, cameras, masks, root / name, {**base, **prior_config},
                progress=lambda fraction, message: print(name, message, flush=True))
            torch.cuda.synchronize(); elapsed = time.perf_counter() - before
            metadata, stats = result['metadata'], result['stats']
            assert result['status'] == 'completed' and stats['completed_steps'] == 400
            assert metadata['geometry_unchanged'] and metadata['source_tensor_hashes'] == metadata['final_tensor_hashes']
            assert sorted(metadata['training_frames']) == sorted(frame_names)
            assert metadata['training_coverage']['missing_mask_views'] == []
            assert stats['coverage']['uncovered_observed_frame_class_pairs'] == []
            assert stats['coverage']['minimum_pair_visits'] >= 1
            assert len(metadata['hierarchy_edges']) == 4
            with np.load(result['probabilities_path'], allow_pickle=False) as data:
                probabilities = data['probabilities']
                assert probabilities.shape == (len(points), 5) and np.isfinite(probabilities).all()
                assert str(data['source_ply_sha256']) == ply_sha
            assert protected == {path: sha(path) for path in protected}
            maximum_hierarchy_loss = max(row['hierarchy_loss'] for row in stats['history'])
            geometric_counts = stats['hierarchy_geometry_support_points_per_edge']
            assert len(geometric_counts) == 4 and all(count > 0 for count in geometric_counts.values())
            if not prior_config:
                assert stats['regularization_support_points'] == 0
                assert metadata['initialization']['kind'] == 'uniform_unknown'
                assert all(row['prior_loss'] == 0 for row in stats['history'])
            if prior_config:
                assert stats['regularization_support_points'] > 0
                assert maximum_hierarchy_loss > 0, 'Seeded fixture did not exercise the hierarchy penalty'
            report['runs'].append({'name': name, 'status': 'completed', 'synthetic_prior_used': bool(prior_config),
                'worker_call_wall_seconds': elapsed, 'worker_timing': metadata['timing'],
                'optimizer_seconds': stats['training_seconds'], 'steps': stats['completed_steps'],
                'peak_cuda_allocated_bytes': stats['cuda_peak_allocated_bytes'], 'coverage': stats['coverage'],
                'regularization_support_points': stats['regularization_support_points'],
                'hierarchy_geometry_support_points_per_edge': geometric_counts,
                'hierarchy_support_identity': stats['hierarchy_support_identity'],
                'max_logged_hierarchy_loss': maximum_hierarchy_loss, 'source_ply_unchanged': True,
                'all_frozen_rgb_tensors_unchanged': True, 'catalog_path': result['catalog_path']})
            write_json(root / 'summary.json', report)
        report.update(status='completed', cuda_executed=True, all_inputs_unchanged=True,
                      total_smoke_seconds=time.perf_counter()-started,
                      limitations=['Synthetic fixture only; no dataset mIoU, RGB reconstruction quality or app ETA.',
                                   'Explicit synthetic masks; does not benchmark DINO/SAM extraction.',
                                   'Fresh uniform mode has no prior anchors; hierarchy support is independent visible geometry. Seeded run is separate.',
                                   'First run includes CUDA context/import warmup; repeat timing needs separately labeled runs.'])
        write_json(root / 'summary.json', report); print(json.dumps(report, indent=2), flush=True)
    except BaseException as exc:
        report.update(status='failed', error_type=type(exc).__name__, error=str(exc), traceback=traceback.format_exc())
        write_json(root / 'summary.json', report); raise


if __name__ == '__main__':
    main()
