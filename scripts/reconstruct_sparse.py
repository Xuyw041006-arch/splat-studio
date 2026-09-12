#!/usr/bin/env python3
"""Actual-photo CUDA reconstruction. No benchmark poses, GT, or MPS training.

Use --phase prepare to import photos and emit a calibration template; optionally
--phase discover to inspect cached object candidates before choosing priorities.
Each --phase run creates a new run, retaining failed and completed predecessors.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import io
import json
import os
from pathlib import Path
import signal
import sys
import time
import traceback
import uuid

PROJECT = Path(__file__).resolve().parents[1]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

import numpy as np
from PIL import Image, ImageOps
from backend.capture import photo_metadata, validate_calibration, geometry_config, capture_fingerprint

IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.webp', '.tif', '.tiff'}


def sha256(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.partial')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def prepare_project(image_dir, project_dir):
    """Keep raw-byte identity separately from lossless EXIF-upright RGB identity."""
    raw, project = Path(image_dir).resolve(), Path(project_dir).resolve()
    if not raw.is_dir():
        raise ValueError('--images must be a directory containing 2–300 photos')
    if project == raw or project.is_relative_to(raw):
        raise ValueError('--project must be outside the input photo directory')
    if project.exists() and any(project.iterdir()):
        raise ValueError('Project directory is not empty; reuse it without --images or choose a new directory')
    paths = sorted(p for p in raw.rglob('*') if p.is_file() and not p.is_symlink()
                   and p.suffix.lower() in IMAGE_EXTENSIONS and '__MACOSX' not in p.parts
                   and not p.name.startswith('._'))
    if not 2 <= len(paths) <= 300:
        raise ValueError(f'Expected 2–300 photos, found {len(paths)}')
    (project / 'images').mkdir(parents=True)
    items, total = [], 0
    for index, path in enumerate(paths):
        total += path.stat().st_size
        if path.stat().st_size > 30_000_000 or total > 1_500_000_000:
            raise ValueError('Photo limit is 30 MB each and 1.5 GB total')
        data = path.read_bytes()
        with Image.open(io.BytesIO(data)) as opened:
            if opened.width * opened.height > 48_000_000:
                raise ValueError(f'Photo exceeds 48 million pixels: {path.name}')
            upright = ImageOps.exif_transpose(opened).convert('RGB')
            name = f'{index:04d}.png'
            stored = project / 'images' / name
            upright.save(stored)
            items.append({'name': name, 'original_name': path.relative_to(raw).as_posix(),
                          'width': upright.width, 'height': upright.height,
                          'capture': photo_metadata(opened, upright, data),
                          'stored_sha256': sha256(stored), 'storage': 'lossless EXIF-upright RGB PNG'})
    manifest = {'id': uuid.uuid4().hex, 'created_at': time.time(), 'images': items,
                'import_method': 'actual input photos only; no supplied SfM or benchmark model read'}
    save_json(project / 'project.json', manifest)
    write_camera_template(project, manifest)
    return manifest


def write_camera_template(project, manifest):
    cfg = geometry_config(manifest, {})
    save_json(Path(project) / 'camera-template.json', {
        'coordinate_space': 'uploaded_pixels', 'cameras': [
            {'image_name': item['name'], 'width': item['width'], 'height': item['height'],
             'source_sha256': item['capture']['source_sha256'], 'K': k}
            for item, k in zip(manifest['images'], cfg['intrinsics_original'])]})


def load_project(project):
    project = Path(project).resolve()
    manifest = json.loads((project / 'project.json').read_text())
    if not 2 <= len(manifest['images']) <= 300:
        raise ValueError('Project must contain 2–300 actual photos')
    names = set()
    for item in manifest['images']:
        name = item['name']
        if not isinstance(name, str) or Path(name).name != name or name in names:
            raise ValueError('Image identities must be unique project basenames')
        names.add(name)
        path = project / 'images' / name
        if path.resolve() != path or not path.is_file() or sha256(path) != item.get('stored_sha256'):
            raise ValueError(f'Project photo identity changed or is missing: {name}')
        with Image.open(path) as image:
            if list(image.size) != [item['width'], item['height']]:
                raise ValueError(f'Project photo pixel dimensions changed: {name}')
    return manifest


def validated_config(value):
    # Reuse the desktop contract while refusing silent extra configuration keys.
    from backend.app import Config
    extras = {'strict_query_labels', 'semantic_max_image_size', 'part_labels'}
    if not isinstance(value, dict) or set(value) - set(Config.model_fields) - extras:
        raise ValueError('Unknown configuration fields; use the desktop Config contract')
    cfg = Config.model_validate(value).model_dump()
    if cfg['device'] not in {'auto', 'cuda'}:
        raise ValueError('This CLI is CUDA-only; CPU and MPS training are not supported')
    if cfg['semantic_strategy'] == 'joint':
        raise ValueError('Joint RGB/semantic training is not implemented; use posthoc semantics')
    if cfg['completion'] == 'learned':
        raise ValueError('Use sparse_completion for visible learned geometry; unseen-backside providers are unsupported')
    cfg.update({key: value[key] for key in extras if key in value})
    if 'strict_query_labels' in cfg and not isinstance(cfg['strict_query_labels'], bool):
        raise ValueError('strict_query_labels must be boolean')
    if 'semantic_max_image_size' in cfg and (type(cfg['semantic_max_image_size']) is not int or
                                           not 256 <= cfg['semantic_max_image_size'] <= 4096):
        raise ValueError('semantic_max_image_size must be an integer in 256–4096')
    if 'part_labels' in cfg and (not isinstance(cfg['part_labels'], list) or
                               not all(isinstance(item, str) for item in cfg['part_labels'])):
        raise ValueError('part_labels must be an array of strings')
    cfg.update(device='cuda', semantic_strategy='posthoc')
    return cfg


def require_cuda(require_a100=False):
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError('NVIDIA CUDA is required. No CPU/MPS training fallback is performed.')
    gpu = torch.cuda.get_device_name(0)
    if require_a100 and 'A100' not in gpu.upper():
        raise RuntimeError(f'Select an A100 runtime; current GPU: {gpu}')
    return {'gpu': gpu, 'torch': torch.__version__, 'cuda': torch.version.cuda,
            'python': sys.version, 'vram_bytes': torch.cuda.get_device_properties(0).total_memory}


def persist_masks(records, project, manifest):
    """Normalize explicitly full-frame masks before priority or semantic helpers."""
    by_name = {item['name']: item for item in manifest['images']}
    folder = Path(project) / 'masks' / uuid.uuid4().hex
    folder.mkdir(parents=True)
    saved = []
    for index, original in enumerate(records):
        record = dict(original)
        name = Path(record.get('image_name', record.get('image_path', ''))).name
        if name not in by_name:
            raise ValueError(f'Mask must reference an imported image name: {name}')
        for key in ('coordinate_space', 'mask_coordinate_space'):
            if key in record and record[key] not in {'full_frame', 'full_image', 'full_frame_pixels', 'full_image_pixels'}:
                raise ValueError('Cropped or unrecognized mask coordinates cannot be used')
        if any(key in record and record[key] is not True for key in ('full_frame', 'is_full_frame', 'mask_full_frame')):
            raise ValueError('Cropped masks cannot be used as full-frame supervision')
        if any(record.get(key) is not None and record.get(key) is not False for key in
               ('crop', 'crop_box', 'crop_bbox', 'crop_rect', 'crop_origin', 'crop_offset', 'crop_bounds',
                'mask_roi', 'roi', 'roi_origin', 'mask_offset', 'mask_origin', 'is_cropped')):
            raise ValueError('Crop transforms must be resolved before importing masks')
        if 'mask' in record:
            array = np.asarray(record.pop('mask'))
        else:
            with Image.open(record['mask_path']) as opened:
                array = np.asarray(opened.convert('L'))
        if array.ndim != 2 or not array.size or array.size > 48_000_000 or not np.isfinite(array).all():
            raise ValueError('Mask must be a finite, nonempty 2D full-frame array')
        item = by_name[name]
        if record.get('source_image_sha256') not in (None, item['stored_sha256']):
            raise ValueError('Mask source image SHA does not match this imported photo')
        if not isinstance(record.get('label'), str) or not record['label'].strip():
            raise ValueError('Each mask needs a nonempty label')
        h, w = array.shape
        # Match the worker's full-image thumbnail half-pixel rounding bound.
        lower = max((w - .5) / item['width'], (h - .5) / item['height'])
        upper = min((w + .5) / item['width'], (h + .5) / item['height'])
        if lower > upper + 1e-12:
            raise ValueError('Mask aspect ratio differs from its original image; cropped masks are rejected')
        path = folder / f'{index:05d}.png'
        Image.fromarray((array > 0).astype('uint8') * 255).resize(
            (item['width'], item['height']), Image.Resampling.NEAREST).save(path)
        record.update(image_name=name, image_path=str(Path(project) / 'images' / name), mask_path=str(path),
                      coordinate_space='full_frame_pixels', full_frame=True,
                      coordinate_source='explicit_full_frame' if original.get('full_frame') else 'legacy_full_frame_contract',
                      input_mask_shape=[h, w], mask_sha256=sha256(path), source_image_sha256=item['stored_sha256'])
        saved.append(record)
    if not saved:
        raise ValueError('No training masks were generated; names alone cannot supervise reconstruction')
    return saved


def cache_masks(project, manifest, cfg, imported=None):
    from backend.app import semantic_config
    from backend.semantics import discover_objects
    path = Path(project) / 'masks.json'
    detection_cfg = semantic_config(cfg)
    discovery_keys = ('provider', 'candidate_labels', 'manual_labels', 'strict_query_labels', 'part_labels',
                      'semantic_max_image_size', 'allow_remote_images', 'mode', 'semantic_budget',
                      'semantic_view_mode', 'semantic_max_views', 'semantic_granularity', 'max_images',
                      'allowed_image_paths', 'held_out_image_paths', 'detector_model', 'sam_model',
                      'detection_threshold', 'text_threshold', 'max_detections_per_image')
    # Bind resolved profile defaults as well as explicit limits/splits. An old
    # sampled cache must not masquerade as masks for a new all-view selection.
    selection_config = {key: detection_cfg.get(key) for key in discovery_keys}
    for key in ('allowed_image_paths', 'held_out_image_paths'):
        if isinstance(selection_config[key], (list, tuple, set)):
            selection_config[key] = sorted(str(value) for value in selection_config[key])
    signature = {'format': 'sparse_cli_mask_cache_v2', 'capture_fingerprint': capture_fingerprint(manifest),
                 'config': selection_config}
    cache_path = Path(project) / 'mask-cache.json'
    if imported:
        payload = json.loads(Path(imported).read_text())
        rows = payload.get('masks') if isinstance(payload, dict) else payload
        if not isinstance(rows, list):
            raise ValueError('Mask JSON must contain a list or {masks: [...]}')
        rows = [dict(row) for row in rows]
        for row in rows:
            if row.get('mask_path') and not Path(row['mask_path']).is_absolute():
                row['mask_path'] = str(Path(imported).resolve().parent / row['mask_path'])
        discovery = {'source': 'user_imported_full_frame_masks', 'objects': [], 'warnings': []}
        signature['imported_sha256'] = sha256(imported)
    elif path.is_file() and cache_path.is_file() and json.loads(cache_path.read_text()) == signature:
        rows = json.loads(path.read_text())
        for row in rows:
            if sha256(row['mask_path']) != row.get('mask_sha256'):
                raise ValueError('Cached mask changed; import corrected masks or choose a new project')
        return rows
    else:
        image_paths = [str(Path(project) / 'images' / item['name']) for item in manifest['images']]
        # The detector writes each completed mask immediately instead of keeping
        # every full-view binary array in memory. A fresh run retains prior files.
        mask_output_dir = Path(project) / 'masks' / ('discovery-' + uuid.uuid4().hex)
        result = discover_objects(image_paths, {**detection_cfg, 'generate_masks': True,
                                               'mask_output_dir': str(mask_output_dir)})
        rows = result.get('masks', [])
        discovery = {key: value for key, value in result.items() if key != 'masks'}
        save_json(Path(project) / 'discovery.json', discovery)
        if not rows:
            raise ValueError('Object discovery produced no masks: ' + '; '.join(discovery.get('warnings', [])))
    saved = persist_masks(rows, project, manifest)
    save_json(path, saved)
    save_json(cache_path, signature)
    save_json(Path(project) / 'discovery.json', discovery)
    return saved


def save_priority_masks(paths, records, priorities, output):
    from backend.semantics import build_priority_masks
    if not priorities:
        return {}
    known = {str(record.get(key, '')).strip().casefold() for record in records
             for key in ('object_id', 'mask_id', 'id', 'label')}
    missing = [label for label in priorities if label.strip().casefold() not in known]
    if missing:
        raise ValueError(f'Priority objects have no actual training masks: {missing}')
    masks = build_priority_masks(paths, records, priorities)
    if not masks:
        raise ValueError('Selected priority objects have no nonempty pixel masks')
    folder = Path(output) / 'priority_masks'
    folder.mkdir(parents=True)
    for path, mask in masks.items():
        # The upstream loader uses the full filename, including .png/.jpg.
        Image.fromarray(np.asarray(mask, dtype='uint8') * 255).save(folder / (Path(path).name + '.png'))
    return {'priority_masks': masks, 'priority_masks_dir': str(folder)}


def run_pipeline(project, cfg, *, require_a100=False, imported_masks=None, cancelled=None,
                 geometry_root=None, source_zip_sha256=None):
    from backend.app import semantic_hierarchy_config
    from backend.planning import analyze_images, make_plan, capabilities
    from backend.upstream import reconstruct_upstream
    from backend.gaussian_io import write_scene
    from backend.semantics import fuse_semantics
    from backend.semantic_service import refine_scene_from_run
    from backend.training_progress import TrainingProgress

    project = Path(project).resolve()
    manifest = load_project(project)
    run_id = uuid.uuid4().hex
    out = project / 'runs' / run_id
    out.mkdir(parents=True)
    started = time.perf_counter()
    cancelled = cancelled or (lambda: False)
    tracker = TrainingProgress()
    status = {'id': run_id, 'project_id': manifest['id'], 'status': 'running', 'phase': 'reconstruction',
              'started_at': time.time(), 'progress': 0.0}

    def progress(value, message):
        if cancelled():
            raise RuntimeError('Cancelled by caller')
        row = {'time': time.time(), 'elapsed_seconds': time.perf_counter() - started,
               'progress': min(.999, max(0., float(value))), 'message': str(message),
               'training_progress': tracker.update(str(message))}
        status.update(row)
        save_json(out / 'status.json', status)
        with (out / 'progress.jsonl').open('a') as stream:
            stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + '\n')
        print(json.dumps(row, ensure_ascii=False), flush=True)

    try:
        hardware = require_cuda(require_a100)
        cfg = geometry_config(manifest, validated_config(cfg))
        if geometry_root:
            cfg['geometry_cache_dir'] = str(Path(geometry_root).resolve())
            cfg['geometry_repo_path'] = str(Path(geometry_root).resolve() / 'dust3r')
        save_json(out / 'config.json', cfg)
        from backend.learned_geometry import geometry_availability
        capability = capabilities()
        capability['learned_geometry'] = geometry_availability(cfg)
        if not capability.get('original_3dgs'):
            raise RuntimeError('Original CUDA 3DGS extensions/source unavailable; run scripts/setup_upstream.py --cuda')
        if cfg['semantics'] and cfg['semantic_refinement'] == 'cuda_iterative' and not capability.get('cuda_semantic_refinement'):
            raise RuntimeError('CUDA semantic worker unavailable; requested optimization will not silently fall back')
        paths = [str(project / 'images' / item['name']) for item in manifest['images']]
        progress(.0, 'Analyze actual input photos and capture metadata')
        analysis = analyze_images(paths, cfg)
        plan = make_plan(analysis, cfg, capability)
        save_json(out / 'analysis.json', analysis)
        save_json(out / 'plan.json', plan)
        if not plan['can_reconstruct']:
            raise ValueError('Initialization preflight rejected input: ' + '; '.join(plan['warnings']))
        if plan['recommended_backend'] != 'original_3dgs':
            raise RuntimeError('CUDA-only entry point cannot use a local preview backend')
        if ('user_calibrated' in cfg['intrinsics_sources'] and
                plan['geometry_initialization_path'] == 'colmap_photo_initialization'):
            raise ValueError('Standard COLMAP path does not consume imported K; use <=80 photos and geometry_backend=sfm')
        cfg.update(sparse=plan['sparse'], extreme=plan['extreme'])
        records = []
        if cfg['semantics'] or cfg['priority_objects']:
            progress(.01, 'Load or generate full-frame object masks')
            records = cache_masks(project, manifest, cfg, imported_masks)
        cfg['masks'] = records
        if cfg['semantics'] and cfg['semantic_refinement'] == 'cuda_iterative':
            cfg = semantic_hierarchy_config(project, cfg, records)
        cfg.update(save_priority_masks(paths, records, cfg['priority_objects'], out))
        # Discovery models are not needed during RGB optimization. This process
        # can otherwise retain their GPU tensors while the trainer subprocess runs.
        from backend.semantics import _load_local_models
        _load_local_models.cache_clear()
        gc.collect()
        import torch
        torch.cuda.empty_cache()
        sources = {str(path.relative_to(PROJECT)): sha256(path)
                   for folder in ('backend', 'scripts') for path in sorted((PROJECT / folder).rglob('*.py'))}
        run_manifest = {'run_id': run_id, 'hardware': hardware, 'source_zip_sha256': source_zip_sha256,
                        'source_sha256': sources, 'capture_fingerprint': capture_fingerprint(manifest),
                        'input_images': manifest['images'], 'input_camera_origin': 'estimated from actual photos only',
                        'precomputed_scene_poses_used': False, 'ground_truth_used': False,
                        'requested_config': {key: value for key, value in cfg.items() if key not in ('masks', 'priority_masks')},
                        'mask_manifest_sha256': sha256(project / 'masks.json') if records else None,
                        'total_timing_scope': 'CUDA validation, input analysis, mask work, geometry, RGB, semantics, serialization; excludes prior setup/import'}
        save_json(out / 'manifest.json', run_manifest)
        result = reconstruct_upstream(paths, str(out), cfg, progress, cancelled)
        ply = Path(result['ply_path'])
        rgb_sha = sha256(ply)
        scene = json.loads(Path(result['scene_path']).read_text())
        if cfg['semantics']:
            if cfg['semantic_refinement'] == 'cuda_iterative':
                # The production service authenticates complete PLY/indices and
                # stores full raw/closed probability sidecars, without RGB training.
                result = refine_scene_from_run(project, run_id, out, cfg,
                    lambda value, message: progress(.95 + .04 * min(1., max(0., float(value))), message), cancelled)
                scene = json.loads(Path(result['scene_path']).read_text())
            else:
                progress(.95, 'Visibility-aware cross-view semantic projection')
                scene = fuse_semantics(scene, paths, cfg)
        if sha256(ply) != rgb_sha:
            raise RuntimeError('RGB PLY changed during semantic processing')
        if cancelled():
            raise RuntimeError('Cancelled by caller')
        scene.setdefault('metadata', {}).update(plan=plan, ply_path=str(ply),
            semantic_refinement_requested=cfg['semantic_refinement'],
            semantic_refinement_applied=cfg['semantic_refinement'] if cfg['semantics'] else None,
            semantics_requested=cfg['semantics'], job_kind='reconstruction', run_id=run_id)
        write_scene(scene, out / 'scene.json')
        initialization = scene['metadata'].get('initialization', {})
        save_json(out / 'initialization-summary.json', {
            'initialization': initialization, 'sparse_completion': scene['metadata'].get('sparse_completion'),
            'registered_views': len(scene.get('cameras', [])), 'input_views': len(paths),
            'sparse_depth': scene['metadata'].get('sparse_depth'),
            'unseen_surfaces_recovered': False})
        run_manifest.update(status='complete', total_seconds=time.perf_counter() - started,
            full_ply_sha256=rgb_sha, rgb_ply_unchanged_by_semantics=True,
            registered_views=len(scene.get('cameras', [])), input_views=len(paths),
            artifacts={'scene_path': str(out / 'scene.json'), 'ply_path': str(ply),
                       'gaussian_lineage': scene['metadata'].get('gaussian_lineage', {}),
                       'semantic_artifacts': scene['metadata'].get('semantic_refinement_artifacts', {})})
        for relative, expected in sources.items():
            if sha256(PROJECT / relative) != expected:
                raise RuntimeError('Source changed during run: ' + relative)
        save_json(out / 'manifest.json', run_manifest)
        status.update(status='complete', progress=1., elapsed_seconds=time.perf_counter() - started,
                      finished_at=time.time(), **run_manifest['artifacts'])
        save_json(out / 'status.json', status)
        save_json(out / 'result.json', status)
        save_json(project / 'latest-result.json', status)
        return status
    except BaseException as exc:
        status.update(status='cancelled' if cancelled() or isinstance(exc, KeyboardInterrupt) else 'failed',
                      error=str(exc), error_code=getattr(exc, 'code', None),
                      geometry_diagnostics=getattr(exc, 'diagnostics', {}),
                      elapsed_seconds=time.perf_counter() - started, finished_at=time.time())
        save_json(out / 'status.json', status)
        (out / 'error.log').write_text(traceback.format_exc())
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project', type=Path, required=True)
    parser.add_argument('--images', type=Path, help='Import a new project from this photo directory')
    parser.add_argument('--phase', choices=['prepare', 'discover', 'run'], default='run')
    parser.add_argument('--config', type=Path, help='JSON using desktop Config fields; device must be cuda or auto')
    parser.add_argument('--calibration', type=Path, help='Edited camera-template.json; all K use EXIF-upright pixels')
    parser.add_argument('--masks', type=Path, help='Optional full-frame mask records using imported image names')
    parser.add_argument('--geometry-root', type=Path)
    parser.add_argument('--upstream', type=Path)
    parser.add_argument('--require-a100', action='store_true')
    parser.add_argument('--source-zip-sha256')
    args = parser.parse_args(argv)
    if args.source_zip_sha256 and (len(args.source_zip_sha256) != 64 or any(c not in '0123456789abcdef' for c in args.source_zip_sha256)):
        parser.error('--source-zip-sha256 must be a lowercase SHA-256 digest')
    if args.upstream:
        os.environ['SPLAT_3DGS_REPO'] = str(args.upstream.resolve())
    if args.geometry_root:
        os.environ['SPLAT_GEOMETRY_HOME'] = str(args.geometry_root.resolve())
        os.environ['SPLAT_GEOMETRY_REPO'] = str(args.geometry_root.resolve() / 'dust3r')
    cfg = validated_config(json.loads(args.config.read_text()) if args.config else {})
    manifest = prepare_project(args.images, args.project) if args.images else load_project(args.project)
    if args.calibration:
        calibrated = validate_calibration(json.loads(args.calibration.read_text()), manifest['images'])
        for item in manifest['images']:
            item['calibrated_K'] = calibrated[item['name']]
        save_json(args.project / 'project.json', manifest)
        write_camera_template(args.project, manifest)
    if args.phase == 'prepare':
        print(json.dumps({'status': 'prepared', 'project': str(args.project.resolve()),
                          'images': len(manifest['images']), 'training_performed': False}))
        return 0
    cancel_state = {'cancelled': False}
    def cancel(signum, frame):
        cancel_state['cancelled'] = True
    signal.signal(signal.SIGINT, cancel)
    signal.signal(signal.SIGTERM, cancel)
    if args.phase == 'discover':
        require_cuda(args.require_a100)
        records = cache_masks(args.project.resolve(), manifest, cfg, args.masks)
        print(json.dumps({'status': 'discovered', 'mask_count': len(records),
                          'labels': sorted({row['label'] for row in records}), 'training_performed': False}, ensure_ascii=False))
    else:
        result = run_pipeline(args.project, cfg, require_a100=args.require_a100, imported_masks=args.masks,
            cancelled=lambda: cancel_state['cancelled'], geometry_root=args.geometry_root,
            source_zip_sha256=args.source_zip_sha256)
        print(json.dumps(result, ensure_ascii=False))
    return 0


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except Exception as error:
        print(json.dumps({'status': 'failed', 'error': str(error)}, ensure_ascii=False), file=sys.stderr)
        raise SystemExit(1)
