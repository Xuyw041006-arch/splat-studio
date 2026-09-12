#!/usr/bin/env python3
"""Finish an already running Colab benchmark, sequentially, then bundle results.

This file is uploaded alongside the documented priority and semantic tool packs.
It never starts concurrent GPU experiments, changes account settings, or mounts Drive.
"""
from pathlib import Path
import argparse
import json
import subprocess
import sys
import time
import zipfile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path('/content'))
    parser.add_argument('--wait-seconds', type=int, default=7200)
    args = parser.parse_args()
    root = args.root.resolve()
    baseline = root / 'benchmark-results'
    priority = root / 'teatime-priority'
    semantic = root / 'semantic-results'
    tools = root / 'splat-benchmark-tools'
    status_path = root / 'benchmark-finalization-status.json'

    def status(phase, **details):
        item = {'phase': phase, 'utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), **details}
        status_path.write_text(json.dumps(item, indent=2))
        print(json.dumps(item), flush=True)

    def run(command, log):
        log.parent.mkdir(parents=True, exist_ok=True)
        with log.open('w') as stream:
            subprocess.run([str(x) for x in command], stdout=stream, stderr=subprocess.STDOUT, check=True)

    try:
        status('waiting_for_six_reconstruction_runs')
        required = [baseline / scene / mode / 'result.json'
                    for scene in ('bonsai', 'teatime') for mode in ('fast', 'balanced', 'fine')]
        deadline = time.monotonic() + args.wait_seconds
        while not all(p.exists() for p in required):
            if time.monotonic() > deadline:
                raise TimeoutError('Reconstruction did not finish before the explicit wait deadline')
            time.sleep(10)
        status('training_priority_ablation')
        run([sys.executable, root / 'priority-tools/run_cuda_priority_ablation.py',
             '--baseline-model', baseline / 'teatime/balanced',
             '--training-masks', tools / 'training_masks.json', '--output', priority],
            root / 'priority-experiment.log')
        cases = [(mode, baseline / 'teatime' / mode, iterations)
                 for mode, iterations in [('fast', 7000), ('balanced', 15000), ('fine', 22000)]]
        cases.append(('balanced-priority', priority / 'balanced', 15000))
        for name, model, iterations in cases:
            destination = semantic / name
            if (destination / 'summary.json').exists():
                continue
            status('evaluating_full_model_semantics', model=name)
            run([sys.executable, tools / 'benchmarks/evaluate_cuda_semantics.py',
                 '--upstream', root / 'splat-benchmark/gaussian-splatting',
                 '--ply', model / 'point_cloud' / f'iteration_{iterations}/point_cloud.ply',
                 '--scene-dir', root / 'data/lerf-ovs/lerf_ovs/teatime',
                 '--split', baseline / 'teatime/split.json',
                 '--training-masks', tools / 'training_masks.json',
                 '--annotations', tools / 'annotations.json', '--eval-size', 256,
                 '--output', destination], semantic / f'{name}.log')
        status('bundling_results')
        archive = root / 'Splat-Studio-A100-benchmark-results.zip'
        # Full PLYs are exported separately so metric reports remain lightweight.
        with zipfile.ZipFile(archive, 'w', zipfile.ZIP_DEFLATED) as z:
            for directory in (baseline, priority, semantic):
                for path in sorted(directory.rglob('*')):
                    if path.is_file() and path.suffix.lower() in {'.json', '.jsonl', '.csv', '.png', '.npz', '.log', '.py', '.patch'}:
                        z.write(path, path.relative_to(root))
            for name in ('run_a100_experiment.py', 'a100-experiment.log', 'priority-experiment.log',
                         'cuda-build-verbose.log', 'data-download.log', 'teatime-download.log',
                         'run_cuda_benchmark.py', 'finish_colab_benchmark.py'):
                path = root / name
                if path.exists():
                    z.write(path, name)
        status('complete', results_archive=str(archive), archive_bytes=archive.stat().st_size)
    except Exception as exc:
        status('failed', error=str(exc))
        raise


if __name__ == '__main__':
    main()
