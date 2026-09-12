"""Loopback-only desktop API, persisted projects, one GPU worker, no shell RPC."""
from __future__ import annotations
import json
import io
import os
import re
import shutil
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Literal
from urllib.parse import urlparse
import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, field_validator, model_validator
from PIL import Image, ImageOps
from starlette.concurrency import run_in_threadpool
from .planning import analyze_images, make_plan, capabilities
from .reconstruction import reconstruct
from .gaussian_io import read_ply, write_ply, write_scene
from .scene_transport import write_viewer_scene, read_scene_lines
from .scene_limits import import_file_limit, scene_point_budget, validate_gaussian_count
from .result_library import list_saved_results
from .semantics import discover_objects, fuse_semantics, build_priority_masks
from .commands import parse_command, apply_command
from .training_progress import TrainingProgress, fresh_training_progress
from .semantic_targets import normalize_label
from .capture import photo_metadata, validate_calibration, geometry_config, capture_fingerprint
from .cloud_tasks import cloud_plan, build_package, find_runtime, owned_file, sha256 as cloud_sha256
from .user_workflow import scene_name as normalize_scene_name, estimate_training

ROOT=Path(__file__).resolve().parents[1]
DATA=Path(os.environ.get('SPLAT_DATA_DIR',ROOT/'data')).resolve()
DATA.mkdir(parents=True,exist_ok=True)
Image.MAX_IMAGE_PIXELS=48_000_000
app=FastAPI(title='Splat Studio',version='0.1.10')
pool=ThreadPoolExecutor(max_workers=1,thread_name_prefix='splat-worker')
jobs={}; lock=threading.RLock()
TOKEN=os.environ.get('SPLAT_SESSION_TOKEN','')

class ApprovedHierarchyEdge(BaseModel):
    """Caller-declared label relation, still subject to training-mask evidence."""
    model_config={'extra':'forbid'}
    parent: str=Field(min_length=1,max_length=120)
    child: str=Field(min_length=1,max_length=120)
    relation: Literal['part_of','contents_of','member_of']
    source: Literal['user_explicit','user_manual_parent_id']='user_explicit'

    @field_validator('parent','child')
    @classmethod
    def normalize_reference(cls,value): return normalize_label(value)

    @model_validator(mode='after')
    def reject_self_relation(self):
        if self.parent==self.child: raise ValueError('父子标签不能相同；类别并集不支持同类实例之间的层级')
        return self

def validate_hierarchy_graph(edges):
    children={};relations={}
    for edge in edges:
        parent,child=edge['parent'],edge['child'];key=(parent,child)
        if key in relations and relations[key]!=edge['relation']:
            raise ValueError('同一父子标签不能同时声明不同关系')
        relations[key]=edge['relation'];children.setdefault(parent,set()).add(child)
    active=set();done=set()
    def visit(label):
        if label in active: raise ValueError('已批准层级不能包含循环')
        if label in done: return
        active.add(label)
        for child in children.get(label,()): visit(child)
        active.remove(label);done.add(label)
    for label in children: visit(label)

class Config(BaseModel):
    scene_name: str=Field(default='',max_length=64)
    full_quality: bool=False
    cloud_gpu: Literal['a100','other']='a100'
    measured_steps_per_second: float | None=Field(default=None,ge=.1,le=10000,allow_inf_nan=False)
    execution_target: Literal['local','cloud']='local'
    cloud_provider: Literal['colab','server']='colab'
    mode: Literal['fast','balanced','fine']='balanced'
    viewer_quality: Literal['full','fast','balanced','fine']='full'
    semantics: bool=False
    device: Literal['auto','cpu','mps','cuda']='auto'
    view_span: float | None=Field(default=None,ge=0,le=360)
    semantic_strategy: Literal['post','posthoc','joint']='post'
    semantic_refinement: Literal['projection','cuda_iterative']='cuda_iterative'
    semantic_steps: Literal[400,1200,2400]=1200
    semantic_budget: Literal['mode','manual']='mode'
    semantic_view_mode: Literal['auto','all','sampled']='auto'
    semantic_granularity: Literal['auto','flat','multilevel']='multilevel'
    approved_hierarchy: list[ApprovedHierarchyEdge]=Field(default_factory=list,max_length=100)
    priority_objects: list[str]=Field(default_factory=list,max_length=100)
    priority_request: str=Field(default='',max_length=1000)
    priority_approval_sha256: str=Field(default='',max_length=64)
    priority_refinement: Literal['weighted','detail']='weighted'
    priority_task_stage: Literal['train','review']='train'
    completion: Literal['none','auto','bounded_prior','learned']='auto'
    geometry_backend: Literal['auto','sfm','dust3r']='auto'
    sparse_completion: Literal['auto','none','learned_visible']='auto'
    allow_geometry_download: bool=False
    provider: Literal['local','manual','vision_api']='local'
    manual_labels: list[str]=Field(default_factory=list,max_length=100)
    allow_remote_images: bool=False
    allow_model_download: bool=False
    candidate_labels: list[str] | None=None

    @field_validator('scene_name')
    @classmethod
    def validate_scene_name(cls,value): return normalize_scene_name(value) if value else ''

    @field_validator('approved_hierarchy')
    @classmethod
    def validate_approved_graph(cls,edges):
        validate_hierarchy_graph([edge.model_dump() for edge in edges]);return edges

class SemanticJobRequest(BaseModel):
    model_config={'extra':'forbid'}
    source_run_id: str=Field(pattern=r'^[a-f0-9]{32}$')
    config: Config=Field(default_factory=Config)

def save_json(path,data):
    path=Path(path);tmp=path.with_suffix('.tmp')
    tmp.write_text(json.dumps(data,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8');tmp.replace(path)

def project(pid):
    if not re.fullmatch(r'[a-f0-9]{32}',pid): raise HTTPException(404,'项目不存在')
    p=DATA/pid
    if not (p/'project.json').is_file(): raise HTTPException(404,'项目不存在')
    return p

def paths_for(p): return [str(p/'images'/i['name']) for i in json.loads((p/'project.json').read_text())['images']]

def capture_manifest(p):
    manifest=json.loads((p/'project.json').read_text()); changed=False
    for item in manifest.get('images',[]):
        if 'width' in item and 'height' in item and item.get('capture'): continue
        path=Path(p)/'images'/item['name']
        if not path.is_file(): raise ValueError('旧项目缺少原始照片，不能恢复相机信息；请重新导入照片。')
        with Image.open(path) as image:
            upright=ImageOps.exif_transpose(image).convert('RGB')
            item.update(width=upright.width,height=upright.height,
                        capture=photo_metadata(image,upright,path.read_bytes()))
            item['capture']['source_note']='legacy project: SHA binds the stored uploaded image'
        changed=True
    if changed: save_json(p/'project.json',manifest)
    return manifest

def semantic_config(config):
    from .semantic_profiles import resolve_semantic_profile
    c=resolve_semantic_profile(config)
    if not c.get('candidate_labels'): c.pop('candidate_labels',None)
    if c.get('provider')=='vision_api':
        c.update(base_url=os.environ.get('SPLAT_VISION_BASE_URL',os.environ.get('SPLAT_LLM_BASE_URL','')),
                 model=os.environ.get('SPLAT_VISION_MODEL',os.environ.get('SPLAT_LLM_MODEL','')),
                 api_key_env='SPLAT_VISION_API_KEY')
    return c

def semantic_hierarchy_config(p,config,masks=None):
    """Map explicit manual parent IDs to labels; never approve automatic nesting."""
    cfg=dict(config)
    if masks is None:
        path=Path(p)/'masks.json'
        masks=json.loads(path.read_text()) if path.is_file() else []
    labels=set();labels_by_id={}
    for record in masks:
        label=normalize_label(record.get('label'));labels.add(label)
        if record.get('object_id'):
            labels_by_id.setdefault(str(record['object_id']),set()).add(label)
    manual=[]
    for record in masks:
        if record.get('annotation_source')!='manual' or not record.get('parent_id'): continue
        parent_id=str(record['parent_id']);parents=labels_by_id.get(parent_id,set())
        if len(parents)!=1:
            raise ValueError(f'手动父物品 ID {parent_id} 未找到唯一标签；请先标注父物品，并让同一 ID 跨视角保持同名')
        relation=record.get('parent_relation') or ('part_of' if record.get('level')=='part' else 'member_of')
        manual.append(ApprovedHierarchyEdge(parent=next(iter(parents)),child=record.get('label'),
            relation=relation,source='user_manual_parent_id').model_dump())
    manual_keys={(edge['parent'],edge['child'],edge['relation']) for edge in manual}
    supplied=[ApprovedHierarchyEdge.model_validate(edge).model_dump() for edge in cfg.get('approved_hierarchy',[])]
    for edge in supplied:
        if edge['parent'] not in labels or edge['child'] not in labels:
            raise ValueError('层级父子标签必须引用本项目已有训练掩码；名称不能补造缺失物品')
        if edge['source']=='user_manual_parent_id' and (edge['parent'],edge['child'],edge['relation']) not in manual_keys:
            raise ValueError('user_manual_parent_id 必须匹配本项目显式手动 parent_id 标注')
    combined=supplied+manual
    # Manual provenance wins when an API caller also approves the same label pair.
    unique={(edge['parent'],edge['child']):edge for edge in combined}
    if len(unique)>100: raise ValueError('单次语义优化最多支持 100 条显式层级关系')
    validate_hierarchy_graph(combined)
    cfg['approved_hierarchy']=[unique[key] for key in sorted(unique)]
    return cfg

@app.middleware('http')
async def protect_loopback(request: Request, call_next):
    if request.url.path.startswith('/api'):
        host=request.headers.get('host','').split(':')[0]
        if host not in {'127.0.0.1','localhost','testserver','[::1]'}: return JSONResponse({'detail':'Loopback access only'},403)
        origin=request.headers.get('origin')
        if origin and urlparse(origin).hostname not in {'127.0.0.1','localhost'}: return JSONResponse({'detail':'Origin rejected'},403)
        if TOKEN and request.headers.get('x-splat-token')!=TOKEN: return JSONResponse({'detail':'Session token required'},401)
    return await call_next(request)

@app.exception_handler(ValueError)
async def validation_error(request,exc): return JSONResponse({'detail':str(exc)},422)

@app.get('/api/health')
def health():
    cap=capabilities()
    return {'status':'ok','device':cap['device'],'capabilities':cap,'version':'0.1.10'}

@app.get('/api/projects')
def list_projects():
    return {'projects':[json.loads(p.read_text()) for p in sorted(DATA.glob('*/project.json'),key=lambda x:x.stat().st_mtime,reverse=True)][:100]}

@app.get('/api/results')
def saved_results():
    return list_saved_results(DATA)

@app.post('/api/projects')
async def upload_project(files: list[UploadFile]=File(...), scene_title: str=Form('')):
    title=normalize_scene_name(scene_title) if scene_title else ''
    if not 2<=len(files)<=300: raise HTTPException(422,'请选择 2 至 300 张照片；本地预览最多处理 80 张。')
    pid=uuid.uuid4().hex;p=DATA/pid;(p/'images').mkdir(parents=True)
    items=[];total=0
    try:
        for i,f in enumerate(files):
            data=await f.read(30_000_001);total+=len(data)
            if len(data)>30_000_000 or total>1_500_000_000: raise ValueError('照片超过 30 MB 或整个项目超过 1.5 GB。')
            import io
            with Image.open(io.BytesIO(data)) as image:
                if image.width*image.height>48_000_000: raise ValueError('单张照片最多 4800 万像素。')
                im=ImageOps.exif_transpose(image).convert('RGB')
                # Strip untrusted filenames and EXIF; keep a separate display-only original name.
                name=f'{i:04d}.jpg';im.save(p/'images'/name,quality=96)
                items.append({'name':name,'original_name':Path(f.filename or name).name,'width':im.width,'height':im.height,
                              'capture':photo_metadata(image,im,data),
                              'url':f'/api/projects/{pid}/files/images/{name}'})
        result={'id':pid,'images':items,'created_at':time.time(),'name':title};save_json(p/'project.json',result);return result
    except Exception:
        shutil.rmtree(p,ignore_errors=True);raise

@app.put('/api/projects/{pid}/name')
async def rename_project(pid: str,request: Request):
    p=project(pid);body=await request.json();name=normalize_scene_name(body.get('name'))
    record=json.loads((p/'project.json').read_text());record['name']=name;save_json(p/'project.json',record)
    return {'name':name}

@app.post('/api/projects/{pid}/plan')
async def plan_project(pid: str,config: Config):
    p=project(pid);cfg,fingerprint=project_geometry(p,config.model_dump())
    saved=p/'analysis.json'
    if not saved.is_file(): raise ValueError('请先完成照片分析。')
    analysis=json.loads(saved.read_text())['analysis']
    if analysis.get('capture_fingerprint')!=fingerprint: raise ValueError('照片或相机参数已变化，请重新分析。')
    hardware=capabilities()
    plan=cloud_plan(analysis,cfg) if config.execution_target=='cloud' else make_plan(analysis,cfg,hardware)
    if config.full_quality and config.execution_target=='local' and not hardware.get('original_3dgs'):
        plan['can_reconstruct']=False
    return {'plan':plan,'estimate':estimate_training(analysis,cfg,hardware)}

@app.get('/api/projects/{pid}/cameras')
def camera_template(pid: str):
    manifest=capture_manifest(project(pid))
    cfg=geometry_config(manifest,{})
    return {'coordinate_space':'uploaded_pixels','cameras':[
        {'image_name':i['name'],'width':i['width'],'height':i['height'],
         'source_sha256':i.get('capture',{}).get('source_sha256'), 'K':k}
        for i,k in zip(manifest['images'],cfg['intrinsics_original'])]}

@app.put('/api/projects/{pid}/cameras')
async def import_cameras(pid: str,request: Request):
    p=project(pid)
    if any(j['project_id']==pid and j['status'] in {'queued','running'} for j in jobs.values()):
        raise HTTPException(409,'重建期间不能更换相机参数')
    payload=await request.json()
    if len(json.dumps(payload))>500_000: raise HTTPException(413,'相机参数文件过大')
    manifest=capture_manifest(p)
    values=validate_calibration(payload,manifest['images'])
    for item in manifest['images']: item['calibrated_K']=values[item['name']]
    save_json(p/'project.json',manifest)
    (p/'analysis.json').unlink(missing_ok=True)
    return {'status':'ok','camera_count':len(values),'capture_fingerprint':capture_fingerprint(manifest)}

def project_geometry(p,config):
    manifest=capture_manifest(p)
    return geometry_config(manifest,config),capture_fingerprint(manifest)

@app.get('/api/projects/{pid}/files/{filename:path}')
def project_file(pid,filename):
    p=project(pid);f=(p/filename).resolve()
    if not f.is_relative_to(p) or not f.is_file() or f.suffix.lower() not in {'.jpg','.png','.json','.ply','.log','.zip'}:
        raise HTTPException(404,'文件不存在')
    return FileResponse(f)

@app.post('/api/projects/{pid}/analyze')
async def analyze(pid: str,config: Config):
    p=project(pid);cfg,fingerprint=project_geometry(p,config.model_dump());paths=paths_for(p)
    analysis=await run_in_threadpool(analyze_images,paths,cfg)
    analysis['capture_fingerprint']=fingerprint
    plan=cloud_plan(analysis,cfg) if config.execution_target=='cloud' else make_plan(analysis,cfg)
    inventory={'objects':[],'source':'disabled','warnings':[]}
    if config.semantics and config.execution_target=='cloud':
        # Cloud capture triage must not invoke local MPS detection or overwrite masks.
        saved=p/'inventory.json'
        cached=json.loads(saved.read_text()) if saved.is_file() else {}
        candidates=list(cached.get('objects',[]))
        known={normalize_label(item.get('label')) for item in candidates}
        for label in config.manual_labels:
            normalized=normalize_label(label)
            if normalized and normalized not in known:
                candidates.append({'id':'cloud-candidate-'+uuid.uuid5(uuid.NAMESPACE_URL,normalized).hex,
                    'label':label,'source':'manual','grounded':False,'level':'object','confidence':0.})
                known.add(normalized)
        inventory={'objects':candidates,'source':'cloud_pending',
                   'warnings':['语义检测将在 CUDA worker 上执行；已保存的掩码会随任务包携带。']}
    elif config.semantics:
        detection_cfg=semantic_config(cfg)
        detection_cfg['mask_output_dir']=str(p/'masks'/uuid.uuid4().hex)
        if config.provider=='local': detection_cfg['strict_query_labels']=True
        inventory=await run_in_threadpool(discover_objects,paths,detection_cfg)
        existing=json.loads((p/'masks.json').read_text()) if (p/'masks.json').exists() else []
        mask_records=[m for m in existing if m.get('annotation_source')=='manual']
        for i,mask in enumerate(inventory.pop('masks',[])):
            entry=dict(mask)
            if 'mask' in entry:
                array=np.asarray(entry.pop('mask'),dtype=np.uint8)*255
                folder=p/'masks';folder.mkdir(exist_ok=True);path=folder/f'{i:05d}.png';Image.fromarray(array).save(path)
                entry['mask_path']=str(path)
            mask_records.append(entry)
        save_json(p/'masks.json',mask_records if inventory.get('available') is not False else existing)
        save_json(p/'inventory.json',inventory)
    result={'analysis':analysis,'plan':plan,'objects':inventory['objects'],'semantic_source':inventory['source'],
            'semantic_coverage':inventory.get('coverage'), 'semantic_timing':inventory.get('timing'),
            'semantic_profile':semantic_config(cfg).get('semantic_profile') if config.semantics else None,
            'warnings':plan['warnings']+inventory.get('warnings',[])}
    save_json(p/'analysis.json',result)
    return result

@app.post('/api/projects/{pid}/masks')
async def import_mask(pid: str, request: Request):
    """Import a binary mask using JSON pixels; no caller-controlled filesystem paths."""
    p=project(pid);body=await request.json()
    if len(json.dumps(body))>12_000_000: raise HTTPException(413,'掩码过大')
    valid={Path(x).name for x in paths_for(p)}
    if body.get('image_name') not in valid: raise ValueError('image_name 必须来自此项目')
    parent_relation=body.get('parent_relation') or ('part_of' if body.get('level')=='part' else 'member_of')
    if body.get('parent_id') and parent_relation not in {'part_of','contents_of','member_of'}:
        raise ValueError('parent_relation 必须为 part_of、contents_of 或 member_of')
    mask=np.asarray(body.get('mask'),dtype=np.uint8)
    if mask.ndim!=2 or not mask.size or mask.size>4_000_000: raise ValueError('mask 必须是最多 400 万像素的二维 0/1 数组')
    name=uuid.uuid4().hex;folder=p/'masks';folder.mkdir(exist_ok=True);path=folder/(name+'.png');Image.fromarray((mask>0).astype('uint8')*255).save(path)
    records=json.loads((p/'masks.json').read_text()) if (p/'masks.json').exists() else []
    entry={'image_path':str(p/'images'/body['image_name']),'mask_path':str(path),'label':str(body.get('label','物品'))[:120],
           'mask_id':name,'annotation_source':'manual','level':body.get('level','object'),'confidence':1.0}
    if body.get('object_id'):entry['object_id']=str(body['object_id'])[:100]
    if body.get('parent_id'):
        entry['parent_id']=str(body['parent_id'])[:100];entry['parent_relation']=parent_relation
    records.append(entry);save_json(p/'masks.json',records);return {'mask_count':len(records),'object_id':entry.get('object_id',name),'mask_id':name}

@app.post('/api/projects/{pid}/priorities/review')
async def review_priorities(pid: str,config: Config):
    from .priority_workflow import parse_request,create_review,public_review
    p=project(pid);cfg=config.model_dump();requested=parse_request(cfg['priority_request'])
    if not requested:raise ValueError('请输入重点物品名称')
    masks=json.loads((p/'masks.json').read_text()) if (p/'masks.json').is_file() else []
    supported={str(m.get('label','')).casefold() for m in masks}
    wanted={r['label'] for r in requested}
    if not wanted.issubset(supported):
        if config.execution_target=='cloud':
            return {'status':'needs_cloud_review','requested':requested,'message':'先导出物品识别任务，在云端生成重点确认包；导回后查看真实区域并确认。'}
        detection={**semantic_config(cfg),'candidate_labels':list(wanted),'device':'cuda' if capabilities().get('cuda') else 'cpu',
                   'generate_masks':True,'strict_query_labels':True,'mask_output_dir':str(p/'masks'/uuid.uuid4().hex)}
        result=await run_in_threadpool(discover_objects,paths_for(p),detection)
        if result.get('available') is False:raise ValueError('重点识别未完成：'+'；'.join(result.get('warnings',[])))
        masks += result.pop('masks',[]);save_json(p/'masks.json',masks)
    result=await run_in_threadpool(create_review,p,cfg['priority_request'],masks)
    return public_review(p,result)

@app.post('/api/projects/{pid}/priorities/import')
async def import_priority_review(pid: str,file: UploadFile=File(...)):
    from .priority_workflow import import_review_package,public_review
    p=project(pid);raw=await file.read(512*1024**2+1)
    if len(raw)>512*1024**2:raise HTTPException(413,'重点确认包超过 512 MiB')
    review=await run_in_threadpool(import_review_package,p,io.BytesIO(raw))
    return public_review(p,review)

@app.post('/api/projects/{pid}/priorities/confirm')
async def confirm_priorities(pid: str,request: Request):
    from .priority_workflow import confirm_review
    body=await request.json()
    return confirm_review(project(pid),body.get('review_id'),body.get('selected_ids'))

def update_job(jid,**updates):
    with lock:
        if updates.get('status') in {'completed','failed','cancelled'}: updates['training_progress']=None
        jobs[jid].update(updates)
        if jobs[jid].get('started_at'): jobs[jid]['elapsed_seconds']=round(time.time()-jobs[jid]['started_at'],2)
        save_json(project(jobs[jid]['project_id'])/f'job-{jid}.json',jobs[jid])

def run_directory(p,run_id):
    if not isinstance(run_id,str) or not re.fullmatch(r'[a-f0-9]{32}',run_id):
        raise ValueError('source_run_id 必须是本项目的有效运行 ID')
    p=Path(p).absolute();runs=p/'runs';out=runs/run_id
    # Reject symlink substitutions, including aliases to another otherwise valid run.
    if p.resolve()!=p or runs.resolve()!=runs or out.resolve()!=out:
        raise ValueError('运行目录不能重定向到其他路径')
    return out

def validate_semantic_source(p,source_run_id):
    """Resolve a completed server-owned run; clients never supply artifact paths."""
    source=run_directory(p,source_run_id);p=Path(p).resolve()
    saved=p/f'job-{source_run_id}.json';scene_path=source/'scene.json'
    if saved.resolve()!=saved or scene_path.resolve()!=scene_path:
        raise ValueError('来源运行文件不能是路径重定向')
    if not saved.is_file() or not scene_path.is_file():
        raise HTTPException(404,'未找到本项目的来源运行')
    record=json.loads(saved.read_text())
    if record.get('id')!=source_run_id or record.get('project_id')!=p.name or record.get('status')!='completed':
        raise ValueError('只能对本项目已完成的运行继续语义优化')
    project_info=json.loads((p/'project.json').read_text())
    scene=json.loads(scene_path.read_text());meta=scene.get('metadata',{})
    backend=str(meta.get('backend',''))
    if not project_info.get('images') or not scene.get('cameras'):
        raise ValueError('继续语义优化需要原项目照片和相机；独立导入的 JSON/PLY 不具备这些训练数据')
    if backend!='original_3dgs' and 'cuda' not in backend.casefold():
        raise ValueError('多轮语义优化需要已有原始 CUDA 3DGS 模型；MPS/CPU 预览仅支持投影融合')
    raw_ply=meta.get('ply_path') or meta.get('source_ply_path') or meta.get('full_ply_path')
    if not isinstance(raw_ply,str) or not raw_ply:
        raise ValueError('来源运行没有可验证的完整 PLY 路径')
    ply=Path(raw_ply)
    if not ply.is_absolute(): ply=source/ply
    resolved=ply.resolve()
    if resolved!=ply.absolute() or not resolved.is_relative_to(p/'runs') or resolved.suffix.lower()!='.ply' or not resolved.is_file():
        raise ValueError('完整 PLY 必须保存在本项目运行目录中')
    masks=p/'masks.json'
    if masks.resolve()!=masks or not masks.is_file() or not json.loads(masks.read_text()):
        raise ValueError('请先为原项目照片生成或导入语义掩码')
    return source

def validate_run_result(p,out,result):
    p=Path(p).resolve();out=Path(out).resolve()
    scene_path=Path(result['scene_path']).resolve();ply_path=Path(result['ply_path']).resolve()
    if not scene_path.is_relative_to(out) or scene_path.suffix.lower()!='.json' or not scene_path.is_file():
        raise ValueError('语义服务必须在当前新运行内生成场景 JSON')
    if not ply_path.is_relative_to(p/'runs') or ply_path.suffix.lower()!='.ply' or not ply_path.is_file():
        raise ValueError('模型路径超出本项目运行目录')
    return {**result,'scene_path':str(scene_path),'ply_path':str(ply_path)}

def refine_scene_from_run(p,source_run_id,out,cfg,progress,cancelled):
    if cancelled(): raise RuntimeError('任务已取消')
    if not capabilities().get('cuda_semantic_refinement',False):
        raise ValueError('多轮语义优化需要 NVIDIA CUDA 和原始 3DGS 依赖；当前设备仅支持原投影融合')
    run_directory(p,source_run_id)
    from .semantic_service import refine_scene_from_run as service
    result=service(p,source_run_id,out,cfg,progress,cancelled)
    if cancelled(): raise RuntimeError('任务已取消')
    return validate_run_result(p,out,result)

def publish_scene(jid,p,out,result,scene,**metadata):
    result=validate_run_result(p,out,result)
    scene.setdefault('metadata',{}).update({'project_id':jobs[jid]['project_id'],'job_id':jid,
        'scene_name':json.loads((p/'project.json').read_text()).get('name',''),
        'ply_path':result['ply_path'],
        'ply_url':f"/api/projects/{jobs[jid]['project_id']}/files/"+str(Path(result['ply_path']).relative_to(p)),
        **metadata})
    canonical_scene=Path(out)/'scene.json'
    write_scene(scene,canonical_scene)
    scene_url=f"/api/projects/{jobs[jid]['project_id']}/files/"+str(canonical_scene.relative_to(p))
    viewer_path=write_viewer_scene(scene,canonical_scene)
    viewer_url=f"/api/projects/{jobs[jid]['project_id']}/files/"+str(viewer_path.relative_to(p))
    update_job(jid,viewer_scene_url=viewer_url)
    return scene_url

def run_job(jid,config):
    job=jobs[jid];p=project(job['project_id'])
    cancelled=lambda:bool(jobs[jid].get('cancel_requested'))
    if cancelled(): update_job(jid,status='cancelled',stage='已取消');return
    update_job(jid,status='running',started_at=time.time(),stage='分析输入',progress=0.)
    training_tracker=TrainingProgress()
    def progress(value,message):
        # Older local trainers include a raw one-step ETA; use the stable structured estimate instead.
        message=re.sub(r'\s*·\s*measured ETA\s+[0-9.]+s','',str(message))
        training=training_tracker.update(message)
        title='优化三维高斯' if training else message
        update_job(jid,progress=min(.97,float(value)),stage=title,message=message,training_progress=training)
    try:
        if config.get('execution_target','local')!='local':
            raise ValueError('云端任务必须导出后交给 CUDA worker，本机队列不会代替执行')
        out=run_directory(p,jid)
        paths=paths_for(p);cfg,fingerprint=project_geometry(p,config)
        if cfg.get('semantic_strategy')=='joint': raise ValueError('联合语义训练尚未实现；请使用后置语义。本版不会将普通 RGB 训练冒充联合训练。')
        analysis=json.loads((p/'analysis.json').read_text())['analysis'] if (p/'analysis.json').exists() else {}
        if analysis.get('capture_fingerprint')!=fingerprint: analysis=analyze_images(paths,cfg)
        plan=make_plan(analysis,cfg);cfg['sparse']=plan['sparse'];cfg['extreme']=plan['extreme']
        if cfg.get('strict_cuda') and (cfg.get('device')!='cuda' or plan['recommended_backend']!='original_3dgs'):
            raise ValueError('云端 worker 要求原始 CUDA 3DGS，拒绝 CPU/MPS/预览回退')
        if not plan['can_reconstruct']: raise ValueError('当前照片无法可靠初始化，且学习式几何不可用；请补拍有重叠和位移的照片，或安装并启用几何模型。')
        cfg['semantic_strategy']='posthoc'
        if cfg['semantics']:
            # Product tasks always execute hierarchy and cross-view confidence;
            # the historical projection/flat baselines remain library APIs only.
            cfg.update(semantic_granularity='multilevel',semantic_refinement='cuda_iterative')
            if plan['recommended_backend']!='original_3dgs' or not capabilities().get('cuda_semantic_refinement',False):
                raise ValueError('完整层级语义训练需要 NVIDIA CUDA。请选择云端训练，当前任务不会退回平级语义或本机预览。')
        masks=json.loads((p/'masks.json').read_text()) if (cfg['semantics'] or cfg['priority_objects']) and (p/'masks.json').exists() else []
        from .priority_workflow import validate_approval
        priority_approval=validate_approval(p,cfg)
        cfg['masks']=masks
        if cfg['semantics'] and cfg.get('semantic_refinement')=='cuda_iterative':
            cfg=semantic_hierarchy_config(p,cfg,masks)
        if cfg.get('completion')=='learned':
            from .completion import make_provider
            cfg['completion_provider']=make_provider(paths,out,cancelled)
        if cfg['priority_objects'] and masks:
            priority_records=[m for m in masks if m.get('annotation_source')=='priority_confirmed' and m.get('mask_id') in priority_approval['selected_ids']] if priority_approval else masks
            cfg['priority_masks']=build_priority_masks(paths,priority_records,cfg['priority_objects'])
        if cfg['priority_objects'] and not cfg.get('priority_masks'):
            raise ValueError('重点物品没有有效加权掩码；云端任务不能以普通重建冒充重点重建')
        if plan['recommended_backend']=='original_3dgs':
            if cfg.get('priority_masks'):
                folder=out/'priority_masks';folder.mkdir(parents=True,exist_ok=True)
                for path,mask in cfg['priority_masks'].items(): Image.fromarray(np.asarray(mask,dtype=np.uint8)*255).save(folder/(Path(path).name+'.png'))
                cfg['priority_masks_dir']=str(folder)
            from .upstream import reconstruct_upstream
            result=reconstruct_upstream(paths,str(out),cfg,progress,cancelled)
        else:
            result=reconstruct(paths,str(out),cfg,progress,cancelled)
        if cancelled(): raise RuntimeError('任务已取消')
        scene=json.loads(Path(result['scene_path']).read_text())
        semantic_method=None;fallback_reason=None
        if cfg['semantics']:
            wants_iterative=cfg.get('semantic_refinement')=='cuda_iterative'
            if wants_iterative and result.get('backend')=='original_3dgs' and capabilities().get('cuda_semantic_refinement',False):
                result=refine_scene_from_run(p,jid,out,cfg,lambda v,m:progress(.95+.04*float(v),m),cancelled)
                scene=json.loads(Path(result['scene_path']).read_text());semantic_method='cuda_iterative'
            else:
                raise ValueError('完整语义训练未执行，任务未完成；请检查 CUDA 训练依赖。')
        if cancelled(): raise RuntimeError('任务已取消')
        scene_url=publish_scene(jid,p,out,result,scene,plan=plan,semantics_requested=cfg['semantics'],
            semantic_refinement_requested=cfg.get('semantic_refinement','projection'),
            semantic_refinement_applied=semantic_method,semantic_refinement_fallback_reason=fallback_reason,
            job_kind='reconstruction')
        update_job(jid,status='completed',progress=1.,stage='重建已完成',scene_url=scene_url,finished_at=time.time())
    except Exception as exc:
        update_job(jid,status='cancelled' if cancelled() else 'failed',error=str(exc),
                   error_code=getattr(exc,'code',None),geometry_diagnostics=getattr(exc,'diagnostics',{}),
                   stage='已取消' if cancelled() else '重建失败',finished_at=time.time())

def enqueue_job(pid,config,worker,source_run_id=None):
    p=project(pid)
    with lock:
        if any(j['project_id']==pid and j['status'] in {'queued','running'} for j in jobs.values()): raise HTTPException(409,'此项目已有任务在运行')
        if len([j for j in jobs.values() if j['status'] in {'queued','running'}])>=4: raise HTTPException(429,'队列已满')
        jid=uuid.uuid4().hex
        jobs[jid]={'id':jid,'project_id':pid,'status':'queued','progress':0.,'stage':'等待 GPU','elapsed_seconds':0.,'created_at':time.time(),
                   'kind':'semantic_refinement' if source_run_id else 'reconstruction'}
        if source_run_id: jobs[jid]['source_run_id']=source_run_id
        update_job(jid)
        pool.submit(worker,jid,config)
        return dict(jobs[jid])

@app.post('/api/projects/{pid}/jobs')
def start_job(pid: str,config: Config):
    if config.execution_target=='cloud': return export_cloud_job(pid,config)
    p=project(pid);cfg=config.model_dump()
    if config.full_quality and not capabilities().get('original_3dgs'):
        raise ValueError('本机尚未配置 NVIDIA 训练环境，请选择云端或先完成设备配置。')
    if config.scene_name:
        record=json.loads((p/'project.json').read_text());record['name']=config.scene_name;save_json(p/'project.json',record)
    from .priority_workflow import validate_approval
    validate_approval(p,cfg)
    if cfg['semantics'] and cfg['semantic_refinement']=='cuda_iterative':cfg=semantic_hierarchy_config(p,cfg)
    return enqueue_job(pid,cfg,run_job)

@app.post('/api/projects/{pid}/cloud-jobs')
def export_cloud_job(pid: str,config: Config):
    p=project(pid)
    if config.execution_target!='cloud': raise ValueError('导出云端任务需要 execution_target=cloud')
    capture_manifest(p)
    cfg=config.model_dump()
    if config.scene_name:
        record=json.loads((p/'project.json').read_text());record['name']=config.scene_name;save_json(p/'project.json',record)
    if cfg['semantics'] and cfg['semantic_refinement']=='cuda_iterative': cfg=semantic_hierarchy_config(p,cfg)
    jid=uuid.uuid4().hex;root=DATA/'.cloud-jobs';folder=root/jid
    if root.resolve()!=root or folder.resolve()!=folder: raise ValueError('云端任务目录不能是路径重定向')
    folder.mkdir(parents=True,exist_ok=False)
    filename=f'splat-cloud-{jid}.zip';package=folder/filename
    try:
        build_package(p,cfg,package,find_runtime(ROOT),job_id=jid)
        record={'id':jid,'project_id':pid,'status':'awaiting_remote_execution',
            'execution_status':'not_started','execution_target':'cloud','cloud_provider':config.cloud_provider,
            'remote_connection_status':'not_connected','created_at':time.time(),'filename':filename,
            'package_url':f'/api/cloud-jobs/{jid}/package','package_sha256':cloud_sha256(package),
            'package_bytes':package.stat().st_size,
            'message':'云端任务包已生成，尚未启动训练；请在 Colab 或服务器运行 CUDA worker，然后导入完整结果。',
            'worker_command_hint':f'python runtime/scripts/cloud_worker.py {filename} --work-dir splat-cloud-{jid}',
            'instructions':['下载 ZIP 并解压至新目录，按 README 安装 CUDA 依赖和模型。',
                '在 Colab 或服务器执行 worker 命令；没有云端连接，本机不会代替训练。',
                '完成后下载 results/scene.splat.jsonl，在 App 导入完整高斯、球谐系数和语义。',
                '保留 results 内完整 PLY、语义文件和哈希清单；此状态不会自动同步远端进度。']}
        save_json(folder/'status.json',record)
        return record
    except Exception:
        shutil.rmtree(folder,ignore_errors=True);raise

def cloud_job_folder(jid):
    if not re.fullmatch(r'[a-f0-9]{32}',jid): raise HTTPException(404,'云端任务不存在')
    path=DATA/'.cloud-jobs'/jid
    if path.resolve()!=path or not path.is_dir(): raise HTTPException(404,'云端任务不存在')
    return path

@app.get('/api/cloud-jobs/{jid}')
def get_cloud_job(jid: str):
    folder=cloud_job_folder(jid)
    path=owned_file(folder,'status.json')
    return json.loads(path.read_text())

@app.get('/api/cloud-jobs/{jid}/package')
def download_cloud_package(jid: str):
    record=get_cloud_job(jid);folder=cloud_job_folder(jid)
    path=owned_file(folder,f'splat-cloud-{jid}.zip',max_bytes=4*1024**3)
    return FileResponse(path,media_type='application/zip',filename=record['filename'])

def run_semantic_job(jid,config):
    job=jobs[jid];p=project(job['project_id'])
    cancelled=lambda:bool(jobs[jid].get('cancel_requested'))
    if cancelled(): update_job(jid,status='cancelled',stage='已取消',finished_at=time.time());return
    update_job(jid,status='running',started_at=time.time(),stage='准备已有模型的语义优化',progress=0.)
    def progress(value,message):
        update_job(jid,progress=max(0.,min(.97,float(value))),stage='后置多轮语义优化',message=str(message),training_progress=None)
    try:
        out=run_directory(p,jid)
        validate_semantic_source(p,job['source_run_id'])
        out.mkdir(parents=True,exist_ok=True)
        cfg={**config,'semantics':True,'semantic_strategy':'posthoc','semantic_refinement':'cuda_iterative','device':'cuda'}
        cfg=semantic_hierarchy_config(p,cfg)
        result=refine_scene_from_run(p,job['source_run_id'],out,cfg,progress,cancelled)
        scene=json.loads(Path(result['scene_path']).read_text())
        if cancelled(): raise RuntimeError('任务已取消')
        scene_url=publish_scene(jid,p,out,result,scene,semantics_requested=True,
            semantic_refinement_requested='cuda_iterative',semantic_refinement_applied='cuda_iterative',
            semantic_refinement_fallback_reason=None,source_run_id=job['source_run_id'],job_kind='semantic_refinement',rgb_retrained=False)
        update_job(jid,status='completed',progress=1.,stage='语义优化已完成',scene_url=scene_url,finished_at=time.time())
    except Exception as exc:
        update_job(jid,status='cancelled' if cancelled() else 'failed',error=str(exc),
                   stage='已取消' if cancelled() else '语义优化失败',finished_at=time.time())

@app.post('/api/projects/{pid}/semantic-jobs')
def start_semantic_job(pid: str,request: SemanticJobRequest):
    p=project(pid)
    if request.config.execution_target=='cloud':
        raise ValueError('现有模型的云端语义优化尚未支持任务包；请使用照片云端重建，或在原始 CUDA 项目中执行语义 worker。不会回退本机。')
    if request.config.semantic_strategy=='joint':
        raise ValueError('联合语义训练尚未实现；已有模型只支持后置语义优化')
    if request.config.device not in {'auto','cuda'} or not capabilities().get('cuda_semantic_refinement',False):
        raise ValueError('已有模型的多轮语义优化需要 NVIDIA CUDA；MPS/CPU 仅支持原投影融合')
    validate_semantic_source(p,request.source_run_id)
    cfg=semantic_hierarchy_config(p,request.config.model_dump())
    return enqueue_job(pid,cfg,run_semantic_job,request.source_run_id)

@app.get('/api/jobs/{jid}')
def get_job(jid: str):
    if jid not in jobs: raise HTTPException(404,'任务不存在')
    with lock:
        result=dict(jobs[jid])
        if result['status']=='running':result['elapsed_seconds']=round(time.time()-result['started_at'],2)
        result['training_progress']=fresh_training_progress(result.get('training_progress'),result['status'])
        return result

@app.post('/api/jobs/{jid}/cancel')
def cancel_job(jid: str):
    get_job(jid);update_job(jid,cancel_requested=True);return {'id':jid,'cancel_requested':True}

@app.get('/api/demo/scene')
def demo(): raise HTTPException(410,'演示入口已移除，请新建或导入自己的场景')

@app.post('/api/commands')
async def commands(request: Request):
    body=await request.json();objects=body.get('objects',[])
    if not isinstance(objects,list) or len(objects)>500: raise ValueError('物品目录过大')
    cfg={'provider':'llm'} if body.get('use_llm') else {}
    command=await run_in_threadpool(parse_command,body.get('text',''),objects,cfg)
    outcome=apply_command({'gaussians':[],'objects':objects},command)
    return {**command,'target_ids':outcome.get('matched_ids',[]),'message':outcome.get('message','操作已解析')}

@app.post('/api/import')
async def import_scene(file: UploadFile=File(...), mode: Literal['full','fast','balanced','fine']=Form('full'), semantic: UploadFile | None=File(None)):
    suffix=Path(file.filename or '').suffix.lower()
    byte_limit=import_file_limit(suffix)
    pid=uuid.uuid4().hex;p=DATA/pid;p.mkdir();source=p/('import'+suffix)
    try:
        total=0
        with source.open('wb') as stream:
            while chunk:=await file.read(1024**2):
                total+=len(chunk)
                if total>byte_limit:
                    raise HTTPException(413,'文件超过当前格式上限：PLY 为 1 GiB，普通 JSON 为 256 MiB，分块 JSONL 为 4 GiB；大场景请使用 .splat.jsonl，最多 200 万高斯，不会自动抽样。')
                stream.write(chunk)
        point_budget=scene_point_budget(mode)
        if suffix=='.ply':
            scene=await run_in_threadpool(read_ply,source,point_budget)
        elif suffix=='.jsonl':
            scene=await run_in_threadpool(read_scene_lines,source)
        else:
            def load_json():
                with source.open(encoding='utf-8') as stream:return json.load(stream)
            scene=await run_in_threadpool(load_json)
        await run_in_threadpool(validate_scene,scene)
        if semantic is not None:
            if suffix!='.ply' or Path(semantic.filename or '').suffix.lower()!='.zip':
                raise ValueError('独立语义 ZIP 必须与原始高斯 PLY 配对')
            semantic_path=p/'semantic.zip';total=0
            with semantic_path.open('wb') as stream:
                while chunk:=await semantic.read(1024**2):
                    total+=len(chunk)
                    if total>2*1024**3:raise HTTPException(413,'语义包超过 2 GiB 上限')
                    stream.write(chunk)
            from .semantic_package import import_semantic_package
            scene=await run_in_threadpool(import_semantic_package,scene,semantic_path,p/'semantic')
        save_json(p/'project.json',{'id':pid,'images':[],'imported':True,'created_at':time.time()})
        scene.setdefault('metadata',{})['imported']=True
        if suffix=='.ply':
            scene['metadata'].update(ply_url=f'/api/projects/{pid}/files/import.ply',
                                     preview_mode=mode,preview_point_budget=point_budget)
        await run_in_threadpool(write_scene,scene,p/'scene.json')
        viewer_path=await run_in_threadpool(write_viewer_scene,scene,p/'scene.json')
        return {'id':pid,'scene_url':f'/api/projects/{pid}/files/scene.json',
                'viewer_scene_url':f'/api/projects/{pid}/files/'+str(viewer_path.relative_to(p))}
    except Exception:
        shutil.rmtree(p,ignore_errors=True);raise

def validate_scene(scene):
    if not isinstance(scene,dict) or not isinstance(scene.get('gaussians'),list): raise ValueError('场景需要高斯列表')
    validate_gaussian_count(len(scene['gaussians']))
    for g in scene['gaussians']:
        for name,n in [('position',3),('scale',3),('rotation',4),('color',3)]:
            a=np.asarray(g.get(name,[]),dtype=float)
            if a.shape!=(n,) or not np.isfinite(a).all(): raise ValueError(f'无效高斯字段 {name}')
        if not np.isfinite(g.get('opacity',float('nan'))) or not 0<=g['opacity']<=1 or min(g['scale'])<=0: raise ValueError('无效高斯尺度/透明度')
        if 'sh' in g:
            degree=g.get('sh_degree');coefficients=np.asarray(g['sh'],dtype=float)
            if type(degree) is not int or degree not in range(4) or coefficients.shape!=(3*(degree+1)**2,) or not np.isfinite(coefficients).all():
                raise ValueError('无效球谐系数，需要 SH0 至 SH3 的完整 RGB 系数')

# Mark jobs interrupted across restarts; there is no false resumption claim.
for saved in DATA.glob('*/job-*.json'):
    try:
        job=json.loads(saved.read_text())
        if job['status'] in {'running','queued'}:
            job.update(status='failed',error='应用上次退出时任务中断，请重新开始。');save_json(saved,job)
        jobs[job['id']]=job
    except (ValueError,KeyError): pass

dist=Path(os.environ.get('SPLAT_FRONTEND_DIR',ROOT/'dist'))
if dist.is_dir(): app.mount('/',StaticFiles(directory=dist,html=True),name='frontend')

def main():
    import uvicorn
    uvicorn.run(app,host='127.0.0.1',port=int(os.environ.get('SPLAT_PORT','8765')),log_level='info')

if __name__=='__main__': main()
