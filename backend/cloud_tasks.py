"""Offline, hash-bound CUDA task exchange. This module never starts a trainer.

The task contains only a project snapshot and an authenticated source runtime;
there are no credentials, local environment variables, RPCs or remote sessions.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import stat
import time
import uuid
import zipfile

FORMAT = 'splat-studio-cloud-task/1'
MAX_PACKAGE_BYTES = 4 * 1024**3
MAX_FILE_BYTES = 512 * 1024**2
MAX_MANIFEST_BYTES = 8 * 1024**2
MAX_FILES = 20000
RUNTIME_SUPPORT = ('scripts/cloud_worker.py', 'scripts/setup_upstream.py',
                   'scripts/setup_geometry.py', 'requirements.txt', 'requirements-semantic.txt')


def sha256(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def json_bytes(value):
    return (json.dumps(value, ensure_ascii=False, allow_nan=False, sort_keys=True, indent=2)+'\n').encode()


def relative_name(value):
    if not isinstance(value, str) or not value or len(value) > 512 or any(c in value for c in ('\\', ':', '\x00')):
        raise ValueError('云端任务包含无效相对文件路径')
    path = PurePosixPath(value)
    if path.is_absolute() or any(p in {'', '.', '..'} for p in value.split('/')):
        raise ValueError('云端任务路径不能越界或使用绝对路径')
    return value


def owned_file(root, value, *, max_bytes=MAX_FILE_BYTES):
    root = Path(root).absolute()
    raw = Path(value)
    path = raw if raw.is_absolute() else root / raw
    if root.resolve() != root or path.resolve() != path.absolute() or not path.is_relative_to(root) or not path.is_file():
        raise ValueError('云端任务文件必须位于本项目内，不能使用符号链接或外部路径')
    if path.stat().st_size > max_bytes:
        raise ValueError('云端任务文件超过资源上限')
    return path


def runtime_files(root):
    root = Path(root).absolute()
    backend = root/'backend'
    if not backend.is_dir() or not (backend/'app.py').is_file():
        raise ValueError('当前安装缺少可移植云端运行源码，请安装包含 cloud-runtime 的桌面版本')
    names = [p.relative_to(root).as_posix() for p in sorted(backend.glob('*.py'))]
    names += list(RUNTIME_SUPPORT)
    return {name: owned_file(root, name) for name in names}


def find_runtime(root):
    candidates = [Path(os.environ['SPLAT_CLOUD_RUNTIME_DIR'])] if os.environ.get('SPLAT_CLOUD_RUNTIME_DIR') else []
    candidates += [Path(root)/'cloud-runtime', Path(root)]
    for candidate in candidates:
        if (candidate/'scripts/cloud_worker.py').is_file():
            runtime_files(candidate)
            return candidate.absolute()
    raise ValueError('当前安装缺少云端 worker 源码；无法导出可执行任务包')


def cloud_plan(analysis, config):
    """Capture triage only: remote GPU, dependencies and ETA are still unknown."""
    from .planning import MODES
    n = analysis['image_count']; unique = analysis.get('unique_image_count', n)
    extreme = unique <= 2
    sparse = extreme or unique < 12 or analysis['connected_ratio'] < .75 or (
        config.get('view_span') is not None and config['view_span'] < 90)
    return {'execution_target': 'cloud', 'device': 'cuda', 'remote_connection_status': 'not_connected',
        'recommended_backend': 'cloud_cuda_pending', 'can_reconstruct': False,
        'can_export_cloud_task': unique >= 2, 'sparse': sparse, 'extreme': extreme,
        'strategy': 'two_view_completion' if extreme else 'sparse_regularized' if sparse else 'standard_3dgs',
        'reason': '依据照片数量、匹配覆盖和拍摄范围预检；实际初始化及 CUDA 能力由远端 worker 检查',
        'settings': dict(MODES[config.get('mode', 'balanced')]),
        'estimated_seconds': None, 'estimate_basis': '尚未连接远端 GPU；不使用本机 MPS 速度推算云端耗时',
        'completion_recommended': extreme, 'automatic_completion_status': 'remote_validation_pending',
        'learned_geometry_ready': False, 'geometry_backend_requested': config.get('geometry_backend', 'auto'),
        'sparse_completion_requested': config.get('sparse_completion', 'auto'),
        'semantic_refinement_requested': config.get('semantic_refinement', 'projection'),
        'planned_semantic_method': config.get('semantic_refinement', 'projection') if config.get('semantics') else None,
        'warnings': list(analysis.get('warnings', []))+['任务包仅待云端执行，尚未启动训练；请在 Colab 或服务器配置 CUDA worker 后运行。']}


README = '''Splat Studio CUDA cloud task (offline exchange)

This ZIP has NOT started training. No server connection is configured.
Extract it to a new directory, inspect manifest.json and the runtime source.
Use a CUDA Python environment on Colab or your own server. Install dependencies:
  python -m pip install -r runtime/requirements-semantic.txt
  python runtime/scripts/setup_upstream.py --cuda
COLMAP is also required for the standard large-scene path. Optional sparse
learned geometry: python runtime/scripts/setup_geometry.py
Downloads occur only when you run these setup commands; weights are not bundled.
For local semantic detection, pre-cache weights or select allow_model_download
in the App before export. Vision API credentials must be configured remotely;
no local credentials are included. Manual labels alone are not pixel masks.

CPU validation (no training):
  python runtime/scripts/cloud_worker.py TASK.zip --verify-only
CUDA execution (WORK must not exist, never reuse an earlier run directory):
  python runtime/scripts/cloud_worker.py TASK.zip --work-dir WORK
The worker refuses CPU/MPS/preview fallbacks and writes a result manifest.
Download WORK/results/scene.splat.jsonl and import it in the desktop App for all
Gaussians, all SH coefficients and semantic editing. Keep the complete results
folder (full PLY, semantic sidecars and SHA manifest) for reproducibility.
This protocol does not provide a live remote progress connection or automatic
result transfer. The App's exported task remains awaiting_remote_execution.
'''


def build_package(project_dir, config, destination, runtime_root, *, job_id=None):
    """Snapshot allowlisted inputs. Read-only to the original project/runtime."""
    p = Path(project_dir).absolute(); destination = Path(destination)
    if destination.exists(): raise ValueError('云端任务包不能覆盖已有文件')
    project_path = owned_file(p, 'project.json')
    source_manifest = json.loads(project_path.read_text())
    images = source_manifest.get('images', [])
    if not 2 <= len(images) <= 300: raise ValueError('云端重建需要本项目的 2 至 300 张原始照片')
    if config.get('execution_target') != 'cloud': raise ValueError('请先选择云端训练')
    if config.get('semantic_strategy') == 'joint': raise ValueError('当前云端任务不支持联合语义训练，请使用后置语义')
    if config.get('cloud_provider') not in {'colab', 'server'}: raise ValueError('无效云端 provider')
    pid = source_manifest.get('id')
    if not isinstance(pid, str) or not re.fullmatch('[a-f0-9]{32}', pid): raise ValueError('项目 ID 无效')
    jid = job_id or uuid.uuid4().hex
    if not re.fullmatch('[a-f0-9]{32}', jid): raise ValueError('云端任务 ID 无效')
    files = {}; image_names = set(); image_identities = set(); portable_images = []
    for item in images:
        name = relative_name(item.get('name'))
        if '/' in name or name.casefold() in image_identities: raise ValueError('照片名称无效或重复')
        image_names.add(name);image_identities.add(name.casefold())
        files['project/images/'+name] = owned_file(p, 'images/'+name)
        portable_images.append({k: v for k, v in item.items() if k in {'name','original_name','width','height','capture','calibrated_K'}})
    portable_project = {k: v for k, v in source_manifest.items() if k in {'id','created_at'}}
    title = config.get('scene_name') or source_manifest.get('name')
    if title:
        from .user_workflow import scene_name
        portable_project['name'] = scene_name(title)
    portable_project['images'] = portable_images
    files['project/project.json'] = json_bytes(portable_project)
    masks = []
    if (p/'masks.json').exists():
        records = json.loads(owned_file(p, 'masks.json').read_text())
        if not isinstance(records, list) or len(records) > MAX_FILES-1000: raise ValueError('掩码目录过大或格式无效')
        for record in records:
            entry = dict(record)
            name = entry.get('image_name') or Path(entry.get('image_path', '')).name
            if not isinstance(name,str) or name not in image_names: raise ValueError('掩码不属于本项目照片')
            if entry.get('image_path'):
                image_path = owned_file(p, entry['image_path'])
                if image_path != p/'images'/name: raise ValueError('掩码照片路径不一致')
            entry.update(image_name=name, image_path='images/'+name)
            if entry.get('mask_path'):
                path = owned_file(p, entry['mask_path'])
                if path.suffix.lower() not in {'.png', '.npy'}: raise ValueError('仅支持 PNG/NPY 像素掩码')
                relative = path.relative_to(p).as_posix()
                if not relative.startswith('masks/'): raise ValueError('掩码文件必须位于项目 masks 目录')
                files['project/'+relative] = path; entry['mask_path'] = relative
            elif 'mask' not in entry:
                raise ValueError('掩码记录没有像素监督，标签不能替代掩码')
            # Unknown external path fields are not portable inputs.
            if any(k.endswith('_path') and k not in {'image_path','mask_path'} for k in entry):
                raise ValueError('掩码包含尚未支持打包的外部路径字段')
            masks.append(entry)
    files['project/masks.json'] = json_bytes(masks)
    if config.get('priority_request') and config.get('priority_task_stage','train')=='train':
        from .priority_workflow import validate_approval
        validate_approval(p,config)
        files['project/priority-approval.json']=owned_file(p,'priority-approval.json')
    for name, path in runtime_files(runtime_root).items(): files['runtime/'+name] = path
    files['README.txt'] = README.encode()
    runtime_config = {**config, 'execution_target': 'local', 'device': 'cuda', 'viewer_quality': 'full'}
    manifest = {'format': FORMAT, 'id': jid, 'project_id': pid, 'created_at': time.time(),
        'execution_status': 'not_started', 'requested_config': dict(config), 'worker_config': runtime_config,
        'execution_conversion': 'cloud request -> local execution on remote CUDA worker; CPU/MPS fallback prohibited',
        'files': []}
    pending = destination.with_suffix('.partial')
    total = 0
    try:
        with zipfile.ZipFile(pending, 'x', compression=zipfile.ZIP_STORED, allowZip64=True) as archive:
            for name, source in sorted(files.items()):
                relative_name(name); digest = hashlib.sha256(); size = 0
                with archive.open(name, 'w', force_zip64=True) as target:
                    if isinstance(source, bytes):
                        chunks = (source,)
                        for chunk in chunks: target.write(chunk); digest.update(chunk); size += len(chunk)
                    else:
                        with source.open('rb') as stream:
                            while chunk := stream.read(1024**2):
                                size += len(chunk)
                                if size > MAX_FILE_BYTES or total+size > MAX_PACKAGE_BYTES: raise ValueError('云端任务包超过资源上限')
                                digest.update(chunk); target.write(chunk)
                total += size
                if size > MAX_FILE_BYTES or total > MAX_PACKAGE_BYTES: raise ValueError('云端任务包超过资源上限')
                manifest['files'].append({'path': name, 'bytes': size, 'sha256': digest.hexdigest()})
            if len(manifest['files']) > MAX_FILES: raise ValueError('云端任务文件数量超过上限')
            payload = json_bytes(manifest)
            if len(payload) > MAX_MANIFEST_BYTES: raise ValueError('云端任务清单过大')
            archive.writestr('manifest.json', payload)
        if pending.stat().st_size > MAX_PACKAGE_BYTES: raise ValueError('云端任务包超过 4 GiB 上限')
        pending.replace(destination)
    except Exception:
        pending.unlink(missing_ok=True); raise
    return manifest


def verify_package(package, runtime_root=None):
    """Reject zip slip, symlinks, duplicate names, bombs and mismatched source."""
    package = Path(package)
    if not package.is_file() or package.stat().st_size > MAX_PACKAGE_BYTES: raise ValueError('任务 ZIP 缺失或超过 4 GiB')
    with zipfile.ZipFile(package) as archive:
        infos = archive.infolist()
        if len(infos) > MAX_FILES+1: raise ValueError('任务 ZIP 文件过多')
        names = set(); total = 0
        for info in infos:
            name = relative_name(info.filename)
            mode = info.external_attr >> 16
            if name.casefold() in names or info.is_dir() or stat.S_IFMT(mode) not in {0, stat.S_IFREG} or info.flag_bits & 1:
                raise ValueError('任务 ZIP 包含重复路径、链接、目录或加密文件')
            names.add(name.casefold()); total += info.file_size
            if info.file_size > MAX_FILE_BYTES or total > MAX_PACKAGE_BYTES: raise ValueError('任务 ZIP 解压后超过资源上限')
        manifest_info = archive.getinfo('manifest.json')
        if manifest_info.file_size > MAX_MANIFEST_BYTES: raise ValueError('任务清单过大')
        manifest = json.loads(archive.read(manifest_info))
        if manifest.get('format') != FORMAT or manifest.get('execution_status') != 'not_started': raise ValueError('无效云端任务格式')
        for key in ('id','project_id'):
            if not isinstance(manifest.get(key),str) or not re.fullmatch('[a-f0-9]{32}',manifest[key]): raise ValueError('任务 ID 无效')
        records = manifest.get('files')
        if not isinstance(records, list) or len(records) > MAX_FILES: raise ValueError('任务清单无效')
        declared = set()
        for record in records:
            name = relative_name(record.get('path'))
            if name in declared or name == 'manifest.json': raise ValueError('任务清单含重复文件')
            declared.add(name)
            if type(record.get('bytes')) is not int or record['bytes'] < 0 or not re.fullmatch('[a-f0-9]{64}',str(record.get('sha256',''))):
                raise ValueError('任务文件哈希或长度无效')
            info = archive.getinfo(name)
            if info.file_size != record['bytes']: raise ValueError('任务文件长度不匹配')
            with archive.open(info) as stream:
                digest = hashlib.file_digest(stream, 'sha256').hexdigest()
            if digest != record['sha256']: raise ValueError('任务文件 SHA256 不匹配：'+name)
        if declared | {'manifest.json'} != {i.filename for i in infos}: raise ValueError('任务含未绑定 SHA 的文件或遗漏文件')
        cfg = manifest.get('worker_config', {})
        requested = manifest.get('requested_config', {})
        expected = {**requested, 'execution_target':'local', 'device':'cuda', 'viewer_quality':'full'}
        if requested.get('execution_target') != 'cloud' or cfg != expected: raise ValueError('任务必须显式转换为远端 CUDA，不允许回退')
        if runtime_root is not None:
            runtime = runtime_files(runtime_root)
            records_by_name = {r['path']:r for r in records}
            if {n for n in declared if n.startswith('runtime/')} != {'runtime/'+n for n in runtime}: raise ValueError('云端运行源码集合不匹配')
            for name, path in runtime.items():
                if sha256(path) != records_by_name['runtime/'+name]['sha256']: raise ValueError('worker 与任务运行源码版本不一致：'+name)
    return manifest


def extract_verified(package, destination, runtime_root=None):
    """Extract only after full validation, into a fresh private directory."""
    manifest = verify_package(package, runtime_root)
    destination = Path(destination).absolute()
    if destination.exists() or destination.parent.resolve() != destination.parent:
        raise ValueError('解包目标必须为全新的非链接目录，不能覆盖已有任务')
    destination.mkdir(mode=0o700)
    try:
        with zipfile.ZipFile(package) as archive:
            for record in manifest['files']:
                target = destination/record['path']; target.parent.mkdir(parents=True,exist_ok=True)
                with archive.open(record['path']) as source, target.open('xb') as stream:
                    shutil.copyfileobj(source, stream, length=1024**2)
                if target.stat().st_size != record['bytes'] or sha256(target) != record['sha256']:
                    raise ValueError('任务在验证后发生变化')
        (destination/'manifest.json').write_bytes(json_bytes(manifest))
    except Exception:
        shutil.rmtree(destination); raise
    return manifest
