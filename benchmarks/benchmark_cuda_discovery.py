#!/usr/bin/env python3
"""Independent CUDA Grounding DINO + SAM latency on the SAME eight MPS train views.

Run AFTER timed reconstruction jobs finish; this process must not overlap them.
No GT image/mask is read, and existing training masks are never rewritten. The
only outputs are a separate latency JSON and (when allowed) public HF model cache.

Example in the already uploaded Colab support directory:
  python /content/splat-benchmark-tools/benchmarks/benchmark_cuda_discovery.py \
    --scene-dir /content/DATA/teatime \
    --split /content/OUTPUT/teatime/split.json \
    --training-masks /content/splat-benchmark-tools/training_masks.json \
    --output /content/cuda-discovery-latency.json --allow-model-download

Missing dependencies are reported, never installed automatically. MPS used
transformers==4.57.6; keep Colab's working CUDA torch/torchvision pair. --dry-run
validates the image selection/config without importing torch or running models.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import statistics
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# Verified against results/teatime/training_masks.json signature and the
# prepare_train_masks() configuration in benchmarks/run_benchmark.py. The
# portable package's training_masks.json omits vocabulary, hence this reference.
REFERENCE_IMAGES = [
    'frame_00003.jpg', 'frame_00026.jpg', 'frame_00050.jpg', 'frame_00072.jpg',
    'frame_00106.jpg', 'frame_00130.jpg', 'frame_00157.jpg', 'frame_00179.jpg',
]
REFERENCE_LABELS = [
    'apple', 'bag of cookies', 'bear nose', 'coffee', 'coffee mug', 'dall-e brand',
    'hooves', 'paper napkin', 'plate', 'sheep', 'stuffed bear', 'tea in a glass',
    'three cookies', 'yellow pouf',
]
KNOWN_GT_VIEWS = {
    'frame_00002.jpg', 'frame_00025.jpg', 'frame_00043.jpg',
    'frame_00107.jpg', 'frame_00129.jpg', 'frame_00140.jpg',
}
# The snapshots actually cached for the MPS reference. Pinning also prevents a
# later model-repository update from silently changing the latency comparison.
MODELS = [
    ('detector_model', 'IDEA-Research/grounding-dino-tiny', 'a2bb814dd30d776dcf7e30523b00659f4f141c71'),
    ('sam_model', 'facebook/sam-vit-base', '70c1a07f894ebb5b307fd9eaaee97b9dfc16068f'),
]
REFERENCE_CONFIG = {
    'provider': 'local', 'device': 'cuda', 'generate_masks': True,
    'candidate_labels': REFERENCE_LABELS, 'semantic_max_image_size': 640,
    'detection_threshold': .25, 'text_threshold': .25,
    'max_detections_per_image': 24,
    'part_labels': ['wheel', 'chair back', 'chair leg', 'door handle', 'car window'],
    'allow_model_download': False,  # Downloads are measured separately first.
}
MPS_REFERENCE_SECONDS = 22.409568166942336


def sha256(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def save_report(path, report):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    temporary.replace(path)


def selection(args):
    metadata_path = Path(args.training_masks).resolve()
    split_path = Path(args.split).resolve()
    raw = read_json(metadata_path)
    if not isinstance(raw, dict):
        raise ValueError('training-masks must be original or portable benchmark metadata JSON')
    raw = raw.get('semantics_training', raw)
    if raw.get('ground_truth_used') is not False:
        raise ValueError('Training mask metadata must explicitly declare ground_truth_used=false')
    signature = raw.get('signature', {})
    selected = signature.get('images') or raw.get('train_image_names')
    if not selected:
        selected = [m.get('image_path', m.get('image_name', '')) for m in raw.get('masks', [])]
    names = list(dict.fromkeys(Path(n).name for n in selected))
    if len(names) != 8 or set(names) != set(REFERENCE_IMAGES):
        raise ValueError(f'Expected the exact eight MPS train images; received {names}')
    vocabulary = signature.get('vocabulary', REFERENCE_LABELS)
    if vocabulary != REFERENCE_LABELS:
        raise ValueError('Vocabulary/order differs from the MPS discovery reference')
    split = read_json(split_path)
    split = split.get('split', split)
    train, test = ({Path(n).name for n in split[key]} for key in ('train', 'test'))
    if train & test or set(names) - train or set(names) & test or train & KNOWN_GT_VIEWS:
        raise ValueError('Actual train/test split overlaps or includes a known held-out GT view')
    image_dir = Path(args.scene_dir).resolve() / args.images_dir
    paths = [(image_dir / name).resolve() for name in REFERENCE_IMAGES]
    for path in paths:
        if path.parent != image_dir.resolve() or not path.is_file():
            raise FileNotFoundError(f'Missing expected training RGB: {path}')
    # Protect all input metadata/images and any existing PNG referenced by the
    # formal mask cache, even though no mask PNG is opened by this script.
    protected = {metadata_path, split_path, *paths}
    for mask in raw.get('masks', []):
        if mask.get('mask_path'):
            p = Path(mask['mask_path'])
            protected.add((p if p.is_absolute() else metadata_path.parent / p).resolve())
    if Path(args.output).resolve() in protected:
        raise ValueError('Latency output must be separate from every benchmark input/cache file')
    if args.warm_repeats < 1 or args.warm_repeats > 20:
        raise ValueError('warm-repeats must be between 1 and 20')
    return paths, {
        'metadata_file': str(metadata_path), 'metadata_sha256': sha256(metadata_path),
        'split_file': str(split_path), 'split_sha256': sha256(split_path),
        'image_names': REFERENCE_IMAGES, 'images_sha256': {p.name: sha256(p) for p in paths},
        'candidate_labels': vocabulary, 'ground_truth_used': False,
        'formal_masks_read_for_inference': False, 'formal_masks_modified': False,
        'config_source': 'MPS training_masks.json signature + run_benchmark.prepare_train_masks defaults',
    }


def prepare_weights(snapshot_download, cache_dir, allow_download):
    paths, rows = {}, []
    for key, model, revision in MODELS:
        started = time.perf_counter()
        cached_before = False
        required = ['config.json', 'model.safetensors', 'preprocessor_config.json']
        if key == 'detector_model':
            required += ['tokenizer_config.json', 'tokenizer.json', 'vocab.txt']
        try:
            local = Path(snapshot_download(model, revision=revision, cache_dir=cache_dir,
                                           local_files_only=True))
            cached_before = all((local / name).is_file() for name in required)
        except (OSError, ValueError):
            local = None
        if not cached_before:
            if not allow_download:
                raise RuntimeError(f'Missing cached {model}@{revision}; pass --allow-model-download to fetch public weights')
            local = Path(snapshot_download(
                model, revision=revision, cache_dir=cache_dir,
                allow_patterns=['*.json', '*.txt', '*.model', '*.safetensors'],
            ))
        if not all((local / name).is_file() for name in required):
            raise RuntimeError(f'Pinned snapshot lacks required weights/processor files: {model}')
        paths[key] = str(local)
        rows.append({'model': model, 'revision': revision, 'snapshot': str(local),
                     'already_cached': cached_before, 'download_requested': not cached_before,
                     'preparation_seconds': time.perf_counter() - started})
    return paths, rows


def mask_statistics(result, image_name, np):
    if result.get('available') is False:
        raise RuntimeError(f'Discovery failed for {image_name}: {result.get("warnings")}')
    if result.get('device') != 'cuda':
        raise RuntimeError(f'Expected actual device=cuda; {image_name} returned {result.get("device")}')
    masks = result.get('masks', [])
    nonempty = 0
    labels, shapes = {}, set()
    for record in masks:
        mask = np.asarray(record.get('mask'))
        if mask.ndim != 2 or not np.isfinite(mask).all():
            raise RuntimeError(f'Invalid 2D mask returned for {image_name}')
        nonempty += int(bool(mask.any()))
        label = str(record.get('label', ''))
        labels[label] = labels.get(label, 0) + 1
        shapes.add(tuple(mask.shape))
    if nonempty == 0:
        raise RuntimeError(f'No nonempty CUDA masks for reference training image {image_name}')
    return {'image': image_name, 'device': result['device'], 'mask_count': len(masks),
            'nonempty_mask_count': nonempty, 'labels': labels,
            'mask_shapes': [list(s) for s in sorted(shapes)],
            'warnings': result.get('warnings', [])}


def run_pass(name, paths, config, semantics, torch, np):
    rows = []
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    for index, path in enumerate(paths):
        torch.cuda.synchronize()
        tick = time.perf_counter()
        result = semantics.discover_objects([str(path)], config)
        torch.cuda.synchronize()  # Include asynchronous CUDA work in wall timing.
        elapsed = time.perf_counter() - tick
        row = mask_statistics(result, path.name, np)
        row['discovery_seconds'] = elapsed
        rows.append(row)
        del result
        print(f'{name} {index+1}/8 {path.name}: {elapsed:.4f}s, {row["nonempty_mask_count"]} nonempty masks', flush=True)
    torch.cuda.synchronize()
    return {'name': name, 'wall_seconds': time.perf_counter()-started,
            'discovery_calls_seconds': sum(r['discovery_seconds'] for r in rows),
            'nonempty_mask_count': sum(r['nonempty_mask_count'] for r in rows),
            'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
            'peak_reserved_bytes': torch.cuda.max_memory_reserved(), 'per_image': rows}


def run(args, report):
    paths, provenance = selection(args)
    report.update(inputs=provenance, config=REFERENCE_CONFIG.copy(),
                  models=[{'model': m, 'revision': r} for _, m, r in MODELS],
                  code_sha256={str(Path(__file__).relative_to(ROOT)): sha256(__file__),
                               'backend/semantics.py': sha256(ROOT/'backend/semantics.py')})
    if args.dry_run:
        report['status'] = 'dry_run_only_no_cuda_or_model_execution'
        return
    tick = time.perf_counter()
    try:
        import numpy as np
        import torch
    except (ImportError, RuntimeError, OSError) as exc:
        raise RuntimeError(f'CUDA torch/numpy import failed; no inference was attempted: {exc}') from exc
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is unavailable. This latency benchmark refuses MPS/CPU fallback.')
    try:
        import transformers
        from huggingface_hub import snapshot_download
        from backend import semantics
        # Validate the model classes now; torchvision/version failures remain
        # explicit dependency errors, never a silent CPU fallback.
        from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor, SamModel, SamProcessor
    except (ImportError, RuntimeError, OSError) as exc:
        raise RuntimeError('Missing/incompatible semantic dependencies. MPS used transformers==4.57.6. '
                           'Install compatible transformers, huggingface-hub, safetensors, Pillow and numpy; '
                           'keep the existing matching CUDA torch/torchvision pair. No automatic install was attempted. '
                           f'Original error: {exc}') from exc
    report['dependency_import_seconds'] = time.perf_counter()-tick
    if not hasattr(semantics, '_load_local_models') or not hasattr(semantics._load_local_models, 'cache_clear'):
        raise RuntimeError('Uploaded backend/semantics.py does not expose the expected cached model loader')
    report['runtime'] = {'python': platform.python_version(), 'platform': platform.platform(),
                         'torch': torch.__version__, 'torch_cuda': torch.version.cuda,
                         'transformers': transformers.__version__,
                         'huggingface_hub': importlib.metadata.version('huggingface-hub')}
    tick = time.perf_counter()
    torch.cuda.init()
    torch.cuda.synchronize()
    report['cuda_context_seconds'] = time.perf_counter()-tick
    gpu = torch.cuda.get_device_properties(torch.cuda.current_device())
    report['runtime'].update(device='cuda', gpu=gpu.name, gpu_total_memory_bytes=gpu.total_memory)
    cache_dir = str(Path(args.cache_dir).expanduser().resolve()) if args.cache_dir else None
    tick = time.perf_counter()
    snapshots, prepared = prepare_weights(snapshot_download, cache_dir, args.allow_model_download)
    report['weights_preparation_seconds'] = time.perf_counter()-tick
    report['weights'] = prepared
    save_report(args.output, report)
    config = {**REFERENCE_CONFIG, **snapshots, 'cache_dir': cache_dir}
    report['effective_config'] = config
    semantics._load_local_models.cache_clear()
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    tick = time.perf_counter()
    loaded = semantics._load_local_models(config['detector_model'], config['sam_model'], 'cuda', True, cache_dir)
    torch.cuda.synchronize()
    report['model_load_and_cuda_transfer_seconds'] = time.perf_counter()-tick
    model_devices = {label: str(next(model.parameters()).device)
                     for label, model in [('detector', loaded[1]), ('sam', loaded[3])]}
    if not all(device.startswith('cuda') for device in model_devices.values()):
        raise RuntimeError(f'Model parameters are not both on CUDA: {model_devices}')
    report['model_parameter_devices'] = model_devices
    report['cold_first_pass'] = run_pass('cold_first_pass', paths, config, semantics, torch, np)
    report['warm_passes'] = []
    save_report(args.output, report)
    for number in range(args.warm_repeats):
        report['warm_passes'].append(run_pass(f'warm_{number+1}', paths, config, semantics, torch, np))
        save_report(args.output, report)
    warm = [p['discovery_calls_seconds'] for p in report['warm_passes']]
    report['warm_summary'] = {'repeats': len(warm), 'eight_image_discovery_seconds_mean': statistics.mean(warm),
                              'eight_image_discovery_seconds_median': statistics.median(warm),
                              'eight_image_discovery_seconds_min': min(warm),
                              'eight_image_discovery_seconds_max': max(warm),
                              'per_image_discovery_seconds_mean': statistics.mean(warm)/len(paths)}
    report['status'] = 'completed_cuda_nonempty_masks_verified'


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--scene-dir', required=True, help='Teatime directory containing images/')
    parser.add_argument('--images-dir', default='images', help='RGB directory relative to scene-dir')
    parser.add_argument('--split', required=True, help='Actual reconstruction split.json or summary.json')
    parser.add_argument('--training-masks', required=True, help='Original or portable MPS training-mask metadata JSON')
    parser.add_argument('--output', required=True, help='Separate latency report JSON; never the formal mask cache')
    parser.add_argument('--cache-dir', help='HF hub cache directory; defaults to huggingface_hub configuration')
    parser.add_argument('--allow-model-download', action='store_true', help='Fetch only the two pinned public model snapshots if absent')
    parser.add_argument('--warm-repeats', type=int, default=3)
    parser.add_argument('--dry-run', action='store_true', help='Validate inputs without importing torch or running models')
    args = parser.parse_args()
    # Validate output separation BEFORE creating any report, including failures.
    try:
        selection(args)
    except Exception as exc:
        print(f'INPUT VALIDATION FAILED: {exc}', file=sys.stderr)
        return 2
    report = {'status': 'running', 'purpose': 'independent CUDA discovery latency, not reconstruction quality',
              'mps_reference': {'seconds': MPS_REFERENCE_SECONDS, 'device': 'mps',
                                'scope': 'one original 8-view prepare_train_masks run: model load/inference plus PNG mask writes; not warm-only'},
              'timing_definitions': {
                  'weights_preparation': 'cached snapshot validation or public model download; excludes pip installation',
                  'model_load': 'deserialize weights and transfer both models to CUDA; no image inference',
                  'cold_first_pass': 'first eight discovery calls after explicit model preload; includes first-use kernels, preprocessing and mask postprocessing',
                  'warm_passes': 'same eight images and model cache; synchronize CUDA around every discovery call; no PNG mask writing',
                  'wall_vs_calls': 'pass wall time additionally includes mask validation, statistics and logging; do not add wall and discovery_calls times',
              },
              'limitations': [
                  'Run only after timed GPU reconstruction jobs finish; script does not manage or stop other processes.',
                  'Pip dependency installation is not performed or timed; model cache preparation and cold/warm costs are separate.',
                  'MPS 22.409568s included PNG writes and a different torch/OS/runtime; do not present a warm CUDA/MPS ratio as a controlled device speedup.',
                  'Nonempty detections/masks validate execution, not segmentation accuracy; no GT is loaded.',
                  'Pinned image names/vocabulary validate the given split, but cannot prove an earlier training process followed it.',
              ]}
    try:
        run(args, report)
    except Exception as exc:
        report.update(status='failed', error=f'{type(exc).__name__}: {exc}')
        save_report(args.output, report)
        traceback.print_exc()
        return 1
    save_report(args.output, report)
    print(json.dumps({'status': report['status'], 'output': str(Path(args.output).resolve()),
                      'warm_summary': report.get('warm_summary')}, ensure_ascii=False, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
