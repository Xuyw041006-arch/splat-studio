#!/usr/bin/env python3
"""Run a hash-bound offline task on CUDA; no server, tunnels or CPU fallback.

python scripts/cloud_worker.py TASK.zip --verify-only
python scripts/cloud_worker.py TASK.zip --work-dir /content/new-splat-work
"""
from __future__ import annotations

import argparse
import importlib
import json
import os
from pathlib import Path
import shutil
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))
from backend.cloud_tasks import (extract_verified, verify_package, json_bytes, sha256,
                                 owned_file, relative_name)


def require_cuda(config):
    from backend.planning import capabilities
    cap = capabilities()
    if not cap.get('cuda') or not cap.get('original_3dgs'):
        raise ValueError('CUDA GPU 和原始 3DGS 扩展/源码必须可用；worker 禁止 CPU/MPS/预览回退。先运行 setup_upstream.py --cuda。')
    if config.get('semantics') and config.get('semantic_refinement') == 'cuda_iterative' and not cap.get('cuda_semantic_refinement'):
        raise ValueError('请求多轮语义，但 CUDA 语义扩展不可用；不会回退投影融合')
    import torch
    return {'device':'cuda','gpu_name':torch.cuda.get_device_name(0), 'torch':torch.__version__,
            'cuda_version':torch.version.cuda, 'original_3dgs':True,
            'cuda_semantic_refinement':bool(cap.get('cuda_semantic_refinement'))}


def materialize_project(extracted, data, manifest):
    """Only rewrite known project-relative inputs; never carry Mac paths over."""
    source = Path(extracted)/'project'; data = Path(data)
    project = json.loads(owned_file(source,'project.json').read_text())
    if project.get('id') != manifest['project_id']: raise ValueError('照片项目与任务 ID 不一致')
    images = project.get('images', [])
    if not 2 <= len(images) <= 300: raise ValueError('任务需要 2 至 300 张照片')
    names = set(); identities = set()
    for item in images:
        name = relative_name(item.get('name'))
        if '/' in name or name.casefold() in identities: raise ValueError('任务照片名称无效或重复')
        names.add(name);identities.add(name.casefold());owned_file(source,'images/'+name)
    masks = json.loads(owned_file(source,'masks.json').read_text())
    if not isinstance(masks,list): raise ValueError('任务 masks.json 格式无效')
    target = data/manifest['project_id']
    if target.exists(): raise ValueError('远端任务不能覆盖已有项目')
    for item in masks:
        name = relative_name(item.get('image_name'))
        if name not in names or item.get('image_path') != 'images/'+name:
            raise ValueError('任务掩码照片路径不匹配')
        owned_file(source,item['image_path'])
        item['image_path'] = str(target/'images'/name)
        if item.get('mask_path'):
            relative = relative_name(item['mask_path'])
            if not relative.startswith('masks/'): raise ValueError('任务掩码路径越界')
            owned_file(source,relative)
            item['mask_path'] = str(target/relative)
        elif 'mask' not in item: raise ValueError('任务掩码缺少像素监督')
        if any(key.endswith('_path') and key not in {'image_path','mask_path'} for key in item):
            raise ValueError('任务掩码包含未支持路径')
    data.mkdir(exist_ok=False)
    shutil.move(source,target)
    (target/'masks.json').write_bytes(json_bytes(masks))
    return target


def prepare_semantics(api, project, config):
    """Run actual teacher on remote CUDA, preserving explicit manual masks."""
    if not config.get('semantics') and not config.get('priority_objects'): return
    import numpy as np
    from backend.semantics import _load_mask
    masks_path = project/'masks.json'
    masks = json.loads(masks_path.read_text()) if masks_path.is_file() else []
    approved=False
    if config.get('priority_request') and config.get('priority_task_stage','train')=='train':
        from backend.priority_workflow import validate_approval
        approved=bool(validate_approval(project,config))
    if config.get('provider','local') == 'local' and not approved:
        cfg = api.semantic_config(config)
        defaults = ['chair','table','sofa','lamp','plant','book','cup','bottle','monitor','door','window','shelf','bag','car','building','tree']
        labels = list(dict.fromkeys((cfg.get('candidate_labels') or defaults)+config.get('priority_objects',[])))
        if labels: cfg['candidate_labels'] = labels
        cfg.update(device='cuda',strict_query_labels=True,generate_masks=True,
                   mask_output_dir=str(project/'masks'/'cloud-teacher'))
        result = api.discover_objects(api.paths_for(project),cfg)
        if result.get('available') is False:
            raise ValueError('远端语义检测未完成：'+'；'.join(result.get('warnings',[])))
        masks = [record for record in masks if record.get('annotation_source')=='manual'] + result.pop('masks',[])
        api.save_json(project/'inventory.json',result)
        api.save_json(masks_path,masks)
    if not masks:
        raise ValueError('没有真实像素掩码，无法执行语义或重点重建；请启用本地 CUDA 检测模型或提供手工掩码')
    supported = set()
    for record in masks:
        mask = _load_mask(record)
        if np.asarray(mask).any(): supported.add(api.normalize_label(record.get('label')))
    missing = {api.normalize_label(label) for label in config.get('priority_objects',[])} - supported
    if missing: raise ValueError('重点类别没有有效像素监督，不能静默执行普通重建：'+', '.join(sorted(missing)))


def export_results(api, project, job_id, target, manifest, package_sha256, hardware):
    """Stream existing viewer chunks into an importable full scene JSONL."""
    from backend.scene_transport import MANIFEST_FORMAT, CHUNK_FORMAT, LINES_FORMAT
    from backend.scene_limits import MAX_SCENE_LINE_BYTES, MAX_JSONL_IMPORT_BYTES, validate_gaussian_count
    target = Path(target)
    if target.exists(): raise ValueError('结果目录不能覆盖已有结果')
    target.mkdir()
    job = dict(api.jobs[job_id]); run = project/'runs'/job_id
    if job.get('status') != 'completed': raise ValueError('只有完成的任务可以导出结果')
    viewer = run/'scene.viewer.json'
    descriptor = json.loads((viewer if viewer.is_file() else run/'scene.json').read_text())
    if descriptor.get('format') == MANIFEST_FORMAT:
        header = descriptor['scene']; count = descriptor['gaussian_count']
        def records():
            offset = 0
            for row in descriptor['chunks']:
                path = owned_file(run,relative_name(row['path']))
                chunk = json.loads(path.read_text())
                if chunk.get('format') != CHUNK_FORMAT or chunk.get('offset') != offset or chunk.get('count') != len(chunk['gaussians']):
                    raise ValueError('结果分块记录不连续')
                offset += chunk['count']; yield chunk
            if offset != count: raise ValueError('结果高斯数量不一致')
    else:
        header = {key:value for key,value in descriptor.items() if key!='gaussians'}
        gaussian_values = descriptor['gaussians']; count = len(gaussian_values)
        def records():
            for offset in range(0,count,25000):
                yield {'format':CHUNK_FORMAT,'offset':offset,'count':min(25000,count-offset),'gaussians':gaussian_values[offset:offset+25000]}
    validate_gaussian_count(count)
    meta = header.setdefault('metadata',{})
    if meta.get('source_gaussian_count') != count or meta.get('backend') != 'original_3dgs':
        raise ValueError('远端输出不是完整原始 CUDA 模型，拒绝导出采样结果')
    ply = Path(meta['ply_path'])
    owned_file(project,ply,max_bytes=1024**3)
    actual_sha = sha256(ply)
    if actual_sha != meta.get('source_ply_sha256'): raise ValueError('远端完整 PLY 身份不匹配')
    shutil.copyfile(ply,target/'point_cloud.ply')
    meta.update(cloud_task={'id':manifest['id'],'package_sha256':package_sha256,
        'execution_target':'cloud','execution_device':'cuda','provider':manifest['requested_config']['cloud_provider'],
        'hardware':hardware,'requested_config':manifest['requested_config']},
        ply_path='point_cloud.ply')
    meta.pop('ply_url',None)
    # Archive every semantic sidecar exactly; normalize only the path metadata.
    artifacts = meta.get('semantic_refinement_artifacts',{})
    if isinstance(artifacts,dict):
        artifacts = dict(artifacts)
        if artifacts:
            import numpy as np
            catalog_path = owned_file(project,artifacts['catalog_path'])
            catalog = json.loads(catalog_path.read_text())
            labels = catalog.get('classes',[])
            if catalog.get('status')!='completed' or catalog.get('source_ply_sha256')!=actual_sha or catalog.get('gaussian_count')!=count or not labels:
                raise ValueError('语义 catalog 与完整模型身份不一致或未完成')
            for key in ('probabilities','closed_probabilities','membership','closed_membership'):
                declared = catalog.get('artifacts',{}).get(key,{})
                relative = relative_name(declared.get('path'))
                # Preserve the original catalog's relative references in semantic/.
                if '/' in relative or not relative.endswith('.npz'): raise ValueError('语义侧文件需要独立 NPZ 文件名')
                source = owned_file(catalog_path.parent,relative,max_bytes=4*1024**3)
                if sha256(source)!=declared.get('sha256'): raise ValueError('语义 NPZ 的 SHA 与 catalog 不一致')
                with np.load(source,allow_pickle=False) as data:
                    if str(data['source_ply_sha256'].item())!=actual_sha or data['classes'].tolist()!=labels:
                        raise ValueError('语义 NPZ 来源模型或类别顺序不一致')
                    if 'probabilities' in key:
                        values=data['probabilities']
                        if values.shape!=(count,len(labels)) or not np.isfinite(values).all() or (values<0).any() or (values>1).any():
                            raise ValueError('语义 NPZ 概率形状或范围无效')
                    else:
                        for index in range(len(labels)):
                            values=data[f'class_{index}']
                            if values.shape!=(count,) or values.dtype!=np.bool_: raise ValueError('语义成员 NPZ 形状或类型无效')
                path_key=key+'_path'
                if artifacts.get(path_key) and Path(artifacts[path_key]).resolve()!=source:
                    raise ValueError('场景语义路径与 catalog 声明冲突')
                artifacts[path_key]=str(source)
        portable = {}
        for key,value in artifacts.items():
            if key.endswith('_path') and isinstance(value,str):
                source = owned_file(project,value,max_bytes=4*1024**3)
                relative = 'semantic/'+source.name
                path = target/relative;path.parent.mkdir(exist_ok=True)
                if path.exists() and sha256(path)!=sha256(source): raise ValueError('语义文件名称冲突')
                shutil.copyfile(source,path);portable[key] = relative
            else: portable[key] = value
        meta['semantic_refinement_artifacts'] = portable
        if portable:
            from backend.semantic_package import write_semantic_package
            (target/'semantic/scene_context.json').write_bytes(json_bytes({'source_ply_sha256':actual_sha,
                'cameras':header.get('cameras',[]),'viewer':{key:meta[key] for key in ('viewer_camera','viewer_core_region','viewer_grid') if key in meta}}))
            write_semantic_package(target/'semantic',target/'semantics.zip')
    path = target/'scene.splat.jsonl'; total = 0
    with path.open('xb') as stream:
        for record in ({'format':LINES_FORMAT,'scene':header,'gaussian_count':count},):
            raw = json.dumps(record,ensure_ascii=False,allow_nan=False,separators=(',',':')).encode()+b'\n'
            if len(raw)-1 > MAX_SCENE_LINE_BYTES: raise ValueError('结果头信息超过传输上限')
            stream.write(raw);total += len(raw)
        for chunk in records():
            raw = json.dumps(chunk,ensure_ascii=False,allow_nan=False,separators=(',',':')).encode()+b'\n'
            total += len(raw)
            if len(raw)-1 > MAX_SCENE_LINE_BYTES or total > MAX_JSONL_IMPORT_BYTES: raise ValueError('完整结果超过 JSONL 传输上限，未降采样；请保留原始运行目录')
            stream.write(raw)
    (target/'task-manifest.json').write_bytes(json_bytes(manifest))
    (target/'job.json').write_bytes(json_bytes(job))
    evidence = {'status':'completed','format':'splat-studio-cloud-result/1','cloud_job_id':manifest['id'],
        'project_id':manifest['project_id'],'run_id':job_id,'package_sha256':package_sha256,
        'source_ply_sha256':actual_sha,'gaussian_count':count,'import_file':'scene.splat.jsonl',
        'complete_gaussians':True,'sh_preservation':'All source coefficients, no subsampling',
        'files':[{'path':p.relative_to(target).as_posix(),'bytes':p.stat().st_size,'sha256':sha256(p)}
                 for p in sorted(target.rglob('*')) if p.is_file()]}
    (target/'result-manifest.json').write_bytes(json_bytes(evidence))
    return evidence


def execute(package, work_dir):
    package = Path(package).resolve(); work = Path(work_dir).absolute()
    if work.exists() or work.parent.resolve()!=work.parent: raise ValueError('请选择不存在且不含符号链接的工作目录；不会覆盖/恢复其他任务')
    manifest = verify_package(package,ROOT)
    if 'backend.app' in sys.modules: raise ValueError('worker 必须在独立 Python 进程运行，不能复用桌面后端进程')
    hardware = require_cuda(manifest['worker_config'])
    work.mkdir(mode=0o700)
    state = {'status':'preparing','cloud_job_id':manifest['id'],'package_sha256':sha256(package),
             'hardware':hardware,'started_at':time.time()}
    status_path = work/'worker-status.json'
    status_path.write_bytes(json_bytes(state))
    try:
        extract_verified(package,work/'package',ROOT)
        project = materialize_project(work/'package',work/'data',manifest)
        os.environ['SPLAT_DATA_DIR'] = str(work/'data')
        api = importlib.import_module('backend.app')
        cfg = api.Config.model_validate(manifest['worker_config']).model_dump()
        if set(manifest['worker_config'])-set(cfg): raise ValueError('任务配置包含未知字段')
        cfg.update(strict_cuda=True,device='cuda',execution_target='local',viewer_quality='full')
        state.update(status='preparing_semantics');status_path.write_bytes(json_bytes(state))
        prepare_semantics(api,project,cfg)
        if cfg.get('priority_task_stage')=='review':
            from backend.priority_workflow import create_review,write_review_package
            review=create_review(project,cfg['priority_request'],json.loads((project/'masks.json').read_text()))
            review_path=work/'priority-review.zip';write_review_package(project,review,review_path)
            state.update(status='awaiting_user_confirmation',finished_at=time.time(),review_file=str(review_path),
                         review_sha256=sha256(review_path),matches=len(review['matches']),missing=review['missing'])
            status_path.write_bytes(json_bytes(state));return state
        jid = manifest['id']; api.jobs[jid] = {'id':jid,'project_id':manifest['project_id'],
            'status':'queued','progress':0.,'stage':'远端 CUDA 就绪','created_at':time.time(),'kind':'reconstruction',
            'execution_target':'cloud','execution_device':'cuda','package_sha256':state['package_sha256']}
        api.update_job(jid)
        (work/'worker-config.json').write_bytes(json_bytes(cfg))
        state.update(status='running',job_path=str(project/f'job-{jid}.json'));status_path.write_bytes(json_bytes(state))
        api.run_job(jid,cfg)
        if api.jobs[jid]['status']!='completed': raise RuntimeError(api.jobs[jid].get('error','CUDA task did not complete'))
        evidence = export_results(api,project,jid,work/'results',manifest,state['package_sha256'],hardware)
        state.update(status='completed',finished_at=time.time(),result=evidence)
        status_path.write_bytes(json_bytes(state));return state
    except Exception as exc:
        state.update(status='failed',error=str(exc),finished_at=time.time())
        status_path.write_bytes(json_bytes(state));raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('package',type=Path)
    parser.add_argument('--work-dir',type=Path)
    parser.add_argument('--verify-only',action='store_true')
    args = parser.parse_args(argv)
    try:
        if args.verify_only:
            manifest = verify_package(args.package,ROOT)
            print(json.dumps({'status':'verified_not_started','id':manifest['id'],
                'file_count':len(manifest['files']),'package_sha256':sha256(args.package)},ensure_ascii=False))
        elif args.work_dir:
            print(json.dumps(execute(args.package,args.work_dir),ensure_ascii=False))
        else: parser.error('--work-dir is required for CUDA execution; use --verify-only for CPU validation')
    except Exception as exc:
        print(json.dumps({'status':'failed','error':str(exc)},ensure_ascii=False),file=sys.stderr);return 1
    return 0


if __name__ == '__main__': raise SystemExit(main())
