"""Freeze the local engine using the same Python environment as the app.

python scripts/build_backend.py --semantics --geometry --geometry-repo /path/to/dust3r
--build-dir can keep large intermediate binaries outside the source project.
Geometry builds include pinned source/dependencies, never the 2.3 GB checkpoint.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
CLOUD_RUNTIME_SUPPORT = ('scripts/cloud_worker.py', 'scripts/setup_upstream.py',
                         'scripts/setup_geometry.py', 'requirements.txt', 'requirements-semantic.txt')


def cloud_runtime_sources(root=None):
    """Readable portable source required to export Linux jobs from a frozen App."""
    root = Path(root or ROOT).resolve()
    files = sorted((root / 'backend').glob('*.py')) + [root / name for name in CLOUD_RUNTIME_SUPPORT]
    if not files or not (root / 'backend/__init__.py').is_file():
        raise RuntimeError('Cloud runtime backend source is incomplete')
    for path in files:
        if not path.is_file() or path.is_symlink() or path.resolve() != path:
            raise RuntimeError('Missing or linked cloud runtime source: ' + str(path))
    return sorted(files)


def stage_geometry(repo, destination):
    """Copy only files authenticated by the hard-coded upstream manifest SHA."""
    from backend.learned_geometry import verify_source, SOURCE_MANIFEST_SHA256, file_sha256
    repo = Path(repo).resolve(); destination = Path(destination).resolve()
    verify_source(repo)
    manifest_path = repo / 'PINNED_SOURCE_MANIFEST.json'
    manifest = json.loads(manifest_path.read_text())
    if destination.exists() and any(destination.iterdir()):
        # Reusing an authenticated, identical source tree keeps rebuilds fast;
        # unexpected or modified contents are rejected rather than overwritten.
        verify_source(destination)
    else:
        destination.mkdir(parents=True, exist_ok=True)
        for name, digest in manifest['files'].items():
            relative = Path(name)
            if relative.is_absolute() or '..' in relative.parts or '.git' in relative.parts:
                raise RuntimeError('Unsafe source manifest path: ' + name)
            target = destination / relative; target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(repo / relative, target)
            if file_sha256(target) != digest: raise RuntimeError('Geometry source changed while copying: ' + name)
        shutil.copyfile(manifest_path, destination / manifest_path.name)
    verify_source(destination)
    return {'source_manifest_sha256': SOURCE_MANIFEST_SHA256, 'source_files': len(manifest['files']),
            'source_bytes': sum((destination/name).stat().st_size for name in manifest['files']),
            'code_commit': manifest['code_commit'], 'croco_commit': manifest['croco_commit'],
            'checkpoint_included': False, 'bundle_repo_relative_path': 'geometry/dust3r'}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--semantics', action='store_true')
    parser.add_argument('--geometry', action='store_true')
    parser.add_argument('--geometry-repo', type=Path)
    parser.add_argument('--build-dir', type=Path, default=ROOT/'build')
    args = parser.parse_args(argv)
    sys.path.insert(0, str(ROOT))
    build = args.build_dir.resolve(); build.mkdir(parents=True, exist_ok=True)
    environment = dict(os.environ, PYINSTALLER_CONFIG_DIR=str(build/'pyinstaller-cache'),
                       MPLBACKEND='Agg', MPLCONFIGDIR=str(build/'matplotlib-cache'), PYTHONDONTWRITEBYTECODE='1')
    command = [sys.executable, '-m', 'PyInstaller', '--noconfirm', '--name', 'splat-backend', '--onedir',
        '--distpath', str(build/'backend'), '--workpath', str(build/'pyinstaller'),
        '--specpath', str(build), '--paths', str(ROOT)]
    dependencies = ['torch', 'cv2', 'uvicorn']
    if args.semantics: dependencies += ['transformers', 'torchvision', 'scipy']
    if args.geometry:
        from backend.learned_geometry import _paths
        repo = args.geometry_repo or _paths({})[1]
        geometry = stage_geometry(repo, build/'geometry/dust3r')
        (build/'geometry-package.json').write_text(json.dumps(geometry, indent=2)+'\n')
        dependencies += ['torchvision', 'scipy', 'roma', 'einops', 'trimesh', 'matplotlib',
                         'huggingface_hub', 'safetensors', 'PIL', 'tqdm', 'packaging', 'pyglet']
    for name in dict.fromkeys(dependencies): command += ['--collect-all', name]
    for path in sorted((ROOT/'backend').glob('*.py')):
        if path.name != '__init__.py': command += ['--hidden-import', 'backend.'+path.stem]
    runtime_sources = cloud_runtime_sources()
    runtime_manifest = {}
    for path in runtime_sources:
        relative = path.relative_to(ROOT)
        command += ['--add-data', str(path) + os.pathsep + str(Path('cloud-runtime') / relative.parent)]
        runtime_manifest[relative.as_posix()] = hashlib.sha256(path.read_bytes()).hexdigest()
    # The self-contained upstream trainer copies this authenticated runtime
    # from __file__; a PYZ entry alone does not provide readable source bytes.
    if (ROOT/'backend/gaussian_lineage.py').is_file():
        command += ['--add-data', str(ROOT/'backend/gaussian_lineage.py')+os.pathsep+'backend']
    command += [str(ROOT/'scripts/run_backend.py')]
    (build/'build-command.json').write_text(json.dumps({'argv': command, 'collect_all': list(dict.fromkeys(dependencies)),
        'geometry': args.geometry, 'semantics': args.semantics, 'checkpoint_included': False,
        'cloud_runtime_files': runtime_manifest}, indent=2)+'\n')
    subprocess.run(command, cwd=ROOT, env=environment, check=True)


if __name__ == '__main__':
    main()
