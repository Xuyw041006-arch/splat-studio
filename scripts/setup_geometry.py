#!/usr/bin/env python3
"""Install the pinned optional DUSt3R initializer, outside application binaries.

python scripts/setup_geometry.py --root /path/to/geometry-runtime
Use --no-weights for source/dependency setup only, --no-deps for an already
prepared interpreter, --status for a strictly read-only capability report.
The default project work/geometry directory is excluded from source releases.
CC BY-NC-SA 4.0 and upstream checkpoint/data conditions apply. This setup does
not install the CUDA 3DGS rasterizer or claim unseen-backside completion.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

PROJECT = Path(__file__).resolve().parents[1]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))
from backend.learned_geometry import (CODE_COMMIT, CROCO_COMMIT, MODEL_REPO, MODEL_REVISION, MODEL_SHA256,
                                      CONFIG_SHA256, MODEL_BYTES, SOURCE_MANIFEST_SHA256, download_checkpoint, file_sha256, geometry_availability)


def run(*args):
    subprocess.run([str(item) for item in args], check=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=PROJECT / 'work/geometry')
    parser.add_argument('--no-weights', action='store_true')
    parser.add_argument('--no-deps', action='store_true')
    parser.add_argument('--status', action='store_true')
    args = parser.parse_args(argv)
    root = args.root.expanduser().resolve()
    config = {'geometry_cache_dir': str(root)}
    if args.status:
        print(json.dumps(geometry_availability(config), indent=2))
        return 0
    started = time.perf_counter()
    root.mkdir(parents=True, exist_ok=True)
    repo = root / 'dust3r'
    if not repo.exists():
        run('git', '-c', 'core.autocrlf=false', 'clone', '--filter=blob:none', 'https://github.com/naver/dust3r.git', repo)
        run('git', '-C', repo, 'config', 'core.autocrlf', 'false')
        run('git', '-C', repo, 'checkout', '--detach', CODE_COMMIT)
    else:
        revision = subprocess.check_output(['git', '-C', str(repo), 'rev-parse', 'HEAD'], text=True).strip()
        dirty = subprocess.check_output(['git', '-C', str(repo), 'status', '--porcelain', '--untracked-files=no'], text=True).strip()
        if revision != CODE_COMMIT or dirty:
            raise SystemExit('Existing geometry source differs from the pinned checkout; select a new --root rather than overwrite it.')
    run('git', '-c', 'core.autocrlf=false', '-C', repo, 'submodule', 'update', '--init', '--recursive')
    actual_croco = subprocess.check_output(['git', '-C', str(repo / 'croco'), 'rev-parse', 'HEAD'], text=True).strip()
    if actual_croco != CROCO_COMMIT:
        raise SystemExit('CroCo submodule does not match the pinned revision')
    tracked = subprocess.check_output(['git', '-C', str(repo), 'ls-files', '--recurse-submodules'], text=True).splitlines()
    source_files = {name: file_sha256(repo / name) for name in sorted(tracked)}
    source_manifest = {'code_commit': CODE_COMMIT, 'croco_commit': CROCO_COMMIT, 'files': source_files}
    source_bytes = (json.dumps(source_manifest, sort_keys=True, indent=2) + '\n').encode()
    if hashlib.sha256(source_bytes).hexdigest() != SOURCE_MANIFEST_SHA256:
        raise SystemExit('Source contents do not match the independently pinned source manifest')
    (repo / 'PINNED_SOURCE_MANIFEST.json').write_bytes(source_bytes)
    if not args.no_deps:
        # Keep the installed torch/runtime version. GUI packages are omitted;
        # initializer imports the actual model/cloud optimizer, not Gradio demo.
        run(sys.executable, '-m', 'pip', 'install', 'roma', 'einops', 'trimesh', 'pyglet<2',
            'matplotlib', 'scipy', 'opencv-python', 'huggingface-hub>=0.22', 'safetensors', 'tqdm')
    model = root / 'models/dust3r-512-dpt'
    if not args.no_weights:
        download_checkpoint(model)
        if (model / 'model.safetensors').stat().st_size != MODEL_BYTES or file_sha256(model / 'model.safetensors') != MODEL_SHA256:
            raise SystemExit('Downloaded DUSt3R weight checksum differs from the official pinned metadata')
        if file_sha256(model / 'config.json') != CONFIG_SHA256:
            raise SystemExit('Downloaded DUSt3R configuration checksum differs from the official pinned metadata')
    from backend.learned_geometry import _runtime
    _runtime(repo)
    import dust3r.cloud_opt  # noqa: F401; verify actual initializer dependency chain
    files = {str(path.relative_to(repo)): hashlib.sha256(path.read_bytes()).hexdigest()
             for directory in (repo / 'dust3r', repo / 'croco/models') for path in directory.rglob('*.py')}
    manifest = {'code_repository': 'https://github.com/naver/dust3r', 'code_commit': CODE_COMMIT,
                'croco_commit': CROCO_COMMIT, 'model_repo': MODEL_REPO, 'model_revision': MODEL_REVISION,
                'model_sha256': MODEL_SHA256, 'model_bytes': MODEL_BYTES, 'config_sha256': CONFIG_SHA256,
                'license': 'CC-BY-NC-SA-4.0 plus upstream training-dataset/checkpoint terms',
                'model_downloaded': (model / 'model.safetensors').is_file(), 'weights_download_requested_this_setup': not args.no_weights,
                'model_path': str(model), 'repo_path': str(repo),
                'python': sys.version, 'source_sha256': files, 'setup_seconds': time.perf_counter() - started,
                'cuda_rope_compiled': False, 'inference_tested': False, 'backside_completion': False}
    manifest['portable_source_manifest_sha256'] = SOURCE_MANIFEST_SHA256
    (root / 'geometry-install.json').write_text(json.dumps(manifest, indent=2))
    print(json.dumps({'setup': manifest, 'capability': geometry_availability(config)}, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
