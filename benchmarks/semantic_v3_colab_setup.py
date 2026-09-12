"""Install only the pinned native 3DGS dependencies needed by the smoke test.

Run in a fresh Colab A100 runtime. Keep its existing PyTorch/CUDA stack; do not
install a replacement torch or silently substitute a mock rasterizer.
"""
from __future__ import annotations

import argparse
import importlib
import importlib.util
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import time

COMMIT = '54c035f7834b564019656c3e3fcc3646292f727d'
SUBMODULES = {'submodules/diff-gaussian-rasterization': '9c5c2028f6fbee2be239bc4c9421ff894fe4fbe0',
              'submodules/simple-knn': '86710c2d4b46680c02301765dd79e465819c8f19'}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--repo', default='/content/semantic-v3-upstream')
    parser.add_argument('--report-dir', default='/content/semantic-v3-setup')
    args = parser.parse_args()
    report_dir = Path(args.report_dir); report_dir.mkdir(parents=True, exist_ok=True)
    repo = Path(args.repo).resolve()
    started = time.perf_counter()
    report = {'kind': 'native_cuda_smoke_setup', 'python': sys.version,
              'platform': platform.platform(), 'upstream_commit': COMMIT,
              'submodules': SUBMODULES, 'commands': [], 'warnings': [], 'status': 'running'}
    env = dict(os.environ, MAX_JOBS='2', TORCH_CUDA_ARCH_LIST='8.0', PIP_DISABLE_PIP_VERSION_CHECK='1',
               NVCC_PREPEND_FLAGS='-include cstdint ' + os.environ.get('NVCC_PREPEND_FLAGS', ''))
    report['compiler_compatibility'] = 'Force-include standard cstdint for pinned rasterizer on CUDA 12.8; no source or numerical algorithm changes'
    def save():
        report['wall_seconds'] = time.perf_counter() - started
        (report_dir / 'setup.json').write_text(json.dumps(report, indent=2))
    def run(command, timeout=600):
        index = len(report['commands']); logfile = report_dir / f'command-{index:02d}.log'
        before = time.perf_counter()
        print('SETUP', index, ' '.join(map(str, command)), flush=True)
        try:
            with logfile.open('w') as stream:
                value = subprocess.run(list(map(str, command)), stdout=stream, stderr=subprocess.STDOUT,
                                       env=env, timeout=timeout, check=False)
            row = {'argv': list(map(str, command)), 'returncode': value.returncode,
                   'seconds': time.perf_counter() - before, 'log': str(logfile)}
            report['commands'].append(row); save()
            if value.returncode:
                print(logfile.read_text(errors='replace')[-12000:], flush=True)
                raise RuntimeError(f'Command {index} failed; see {logfile}')
            print('DONE', index, round(row['seconds'], 2), 'seconds', flush=True)
        except BaseException:
            save(); raise
    try:
        import torch
        from torch.utils.cpp_extension import CUDA_HOME
        report.update(torch=torch.__version__, torch_cuda=torch.version.cuda, cuda_home=CUDA_HOME)
        if not torch.cuda.is_available():
            raise RuntimeError('A genuine NVIDIA CUDA runtime is required')
        report.update(gpu=torch.cuda.get_device_name(0), capability=list(torch.cuda.get_device_capability(0)),
                      gpu_memory_bytes=torch.cuda.get_device_properties(0).total_memory)
        if 'A100' not in report['gpu']:
            raise RuntimeError('This smoke setup is explicitly configured for A100 / sm80')
        nvcc = shutil.which('nvcc') or (str(Path(CUDA_HOME) / 'bin/nvcc') if CUDA_HOME else None)
        if not nvcc or not Path(nvcc).is_file():
            raise RuntimeError('CUDA compiler nvcc is missing; native source builds cannot proceed')
        report['nvcc'] = subprocess.check_output([nvcc, '--version'], text=True)
        report['cxx'] = subprocess.check_output(['c++', '--version'], text=True).splitlines()[0]
        if sys.version_info >= (3, 13):
            report['warnings'].append('Python 3.13 uses a fresh native extension build; compatibility is established only if compilation and CUDA execution pass.')
        report['warnings'].append('PyTorch CUDA and local nvcc must be compatible. Build failures are retained; no runtime downgrade or fake renderer is used.')
        missing = [package for module, package in [('numpy', 'numpy'), ('PIL', 'Pillow'), ('plyfile', 'plyfile'),
            ('ninja', 'ninja'), ('cv2', 'opencv-python-headless'), ('setuptools', 'setuptools'), ('wheel', 'wheel')]
            if importlib.util.find_spec(module) is None]
        if missing:
            run([sys.executable, '-m', 'pip', 'install', *missing], 600)
        if not repo.exists():
            run(['git', 'init', repo], 60)
            run(['git', '-C', repo, 'remote', 'add', 'origin', 'https://github.com/graphdeco-inria/gaussian-splatting.git'], 60)
            run(['git', '-C', repo, 'fetch', '--depth', '1', 'origin', COMMIT], 240)
            run(['git', '-C', repo, 'checkout', '--detach', COMMIT], 60)
        if subprocess.check_output(['git', '-C', str(repo), 'rev-parse', 'HEAD'], text=True).strip() != COMMIT:
            raise RuntimeError('Existing upstream checkout is not the required pinned commit')
        run(['git', '-C', repo, 'submodule', 'update', '--init', '--recursive', '--', *SUBMODULES], 600)
        for relative, revision in SUBMODULES.items():
            actual = subprocess.check_output(['git', '-C', str(repo / relative), 'rev-parse', 'HEAD'], text=True).strip()
            if actual != revision:
                raise RuntimeError(f'Unexpected submodule revision: {relative}')
            run([sys.executable, '-m', 'pip', 'install', '-v', '--no-build-isolation', '--no-deps', str(repo / relative)], 900)
        report['native_extensions'] = {}
        for name in ('diff_gaussian_rasterization._C', 'simple_knn._C'):
            module = importlib.import_module(name)
            report['native_extensions'][name] = str(module.__file__)
        report['status'] = 'completed'
        report['execution_still_required'] = True
        save(); print(json.dumps(report, indent=2), flush=True)
    except BaseException as exc:
        report.update(status='failed', error_type=type(exc).__name__, error=str(exc)); save(); raise


if __name__ == '__main__':
    main()
