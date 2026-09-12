#!/usr/bin/env python3
"""Sequential additional evaluation for the current A100 experiment."""
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time
import zipfile

ROOT = Path('/content')
STATUS = ROOT / 'extended-evaluation-status.json'


def status(phase, **values):
    row = {'phase': phase, 'utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), **values}
    STATUS.write_text(json.dumps(row, indent=2))
    print(json.dumps(row), flush=True)


def run(args, log):
    with (ROOT / log).open('w') as output:
        subprocess.run([str(x) for x in args], stdout=output, stderr=subprocess.STDOUT, check=True)


def main():
    status('waiting_for_training_and_semantic_evaluation')
    deadline = time.monotonic() + 7200
    while True:
        p = ROOT / 'benchmark-finalization-status.json'
        phase = json.loads(p.read_text()).get('phase') if p.exists() else None
        if phase == 'complete':
            break
        if phase == 'failed':
            raise RuntimeError('Earlier benchmark failed; extended evaluation will not conceal that failure')
        if time.monotonic() > deadline:
            raise TimeoutError('Earlier benchmark did not complete in two hours')
        time.sleep(10)
    status('full_resolution_evaluation')
    run([sys.executable, ROOT / 'evaluate_cuda_full_resolution.py',
         '--baseline-root', ROOT / 'benchmark-results', '--priority-root', ROOT / 'teatime-priority',
         '--output', ROOT / 'full-resolution-results'], 'full-resolution-evaluation.log')
    scale = ROOT / 'analyze_render_scale.py'
    if scale.exists():
        status('comparing_render_scales')
        run([sys.executable, scale, '--full-resolution-root', ROOT / 'full-resolution-results',
             '--output', ROOT / 'render-scale-analysis'], 'render-scale-analysis.log')
    status('preparing_discovery_dependencies')
    run([sys.executable, '-m', 'pip', 'install', 'transformers==4.57.6', 'safetensors'], 'discovery-dependencies.log')
    discovery = ROOT / 'splat-benchmark-tools/benchmarks/benchmark_cuda_discovery.py'
    shutil.copy2(ROOT / 'benchmark_cuda_discovery.py', discovery)
    status('measuring_cuda_discovery')
    run([sys.executable, discovery, '--scene-dir', ROOT / 'data/lerf-ovs/lerf_ovs/teatime',
         '--split', ROOT / 'benchmark-results/teatime/split.json',
         '--training-masks', ROOT / 'splat-benchmark-tools/training_masks.json',
         '--output', ROOT / 'cuda-discovery-latency.json', '--allow-model-download'], 'cuda-discovery.log')
    for scene in ('bonsai', 'teatime'):
        status('exporting_viewer_preview', scene=scene)
        model = ROOT / 'benchmark-results' / scene / 'fine'
        command = [sys.executable, ROOT / 'export_benchmark_viewer.py',
                   '--ply', model / 'point_cloud/iteration_22000/point_cloud.ply',
                   '--cameras', model / 'cameras.json', '--output', ROOT / 'previews' / scene,
                   '--max-gaussians', 100000, '--copy-ply']
        if scene == 'teatime':
            command += ['--semantics', ROOT / 'semantic-results/fine', '--priority-label', 'stuffed bear']
        run(command, f'{scene}-preview-export.log')
    status('bundling_extended_results')
    archive = ROOT / 'Splat-Studio-A100-extended-results.zip'
    with zipfile.ZipFile(archive, 'w', zipfile.ZIP_DEFLATED) as z:
        for folder in ('full-resolution-results', 'render-scale-analysis', 'previews'):
            for p in sorted((ROOT / folder).rglob('*')):
                if p.is_file() and p.suffix in {'.json', '.csv', '.png', '.npy', '.md', '.py', '.log'}:
                    z.write(p, p.relative_to(ROOT))
        for name in ('cuda-discovery-latency.json', 'cuda-discovery.log', 'full-resolution-evaluation.log',
                     'discovery-dependencies.log', 'finish_colab_extended.py', 'evaluate_cuda_full_resolution.py',
                     'benchmark_cuda_discovery.py', 'export_benchmark_viewer.py'):
            p = ROOT / name
            if p.exists():
                z.write(p, name)
    status('complete', archive=str(archive), bytes=archive.stat().st_size)


if __name__ == '__main__':
    try:
        main()
    except Exception as exc:
        status('failed', error=f'{type(exc).__name__}: {exc}')
        raise
