"""Project-scoped semantic-only refinement; never rerun or rewrite RGB training."""
from __future__ import annotations
import copy
import hashlib
import json
from pathlib import Path
import re
import uuid
import numpy as np
from .gaussian_io import read_ply, write_scene
from .semantic_sidecar import bind_semantic_sidecar
from .scene_limits import validate_gaussian_count


def _within(path,root):
    value=Path(path).resolve();base=Path(root).resolve()
    if not value.is_relative_to(base):raise ValueError('语义任务文件必须位于当前项目内')
    return value


def _sha(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def _ply_vertex_count(path):
    """Read the actual source count without decoding another full preview."""
    count=None
    with Path(path).open('rb') as stream:
        if stream.readline().strip()!=b'ply':raise ValueError('来源文件不是 PLY')
        for _ in range(4096):
            if stream.tell()>262144:raise ValueError('PLY header 过大')
            line=stream.readline()
            if not line:break
            fields=line.decode('ascii').strip().split()
            if fields==['end_header']:
                if count is None or count<1:raise ValueError('完整 PLY 没有高斯顶点')
                return count
            if fields[:2]==['element','vertex']:
                if count is not None or len(fields)!=3:raise ValueError('PLY 顶点数声明无效')
                count=int(fields[2])
    raise ValueError('PLY header 不完整')


_POINT_STATE_FIELDS=('hidden','visible','deleted','priority','reconstruction_weight',
                     'source','confidence','provenance')
_EDIT_FIELDS=('hidden','visible','deleted','priority','reconstruction_weight')


def _prepare_semantic_preview(original,ply,actual_hash):
    """Keep a trusted full-SH full scene; otherwise restore the complete PLY.

    A missing source hash cannot authenticate an old mapping. Old row order,
    approximate positions, equal preview sizes, and an inferred linspace schedule
    are deliberately not used to guess identity. The original run stays intact.
    """
    metadata=original.get('metadata',{})
    count=_ply_vertex_count(ply)
    validate_gaussian_count(count)
    rest_count=0
    with Path(ply).open('rb') as stream:
        for line in stream:
            if line.strip()==b'end_header':break
            fields=line.split()
            if len(fields)==3 and fields[0]==b'property' and fields[-1].startswith(b'f_rest_'):
                rest_count+=1
    source_degree=next((degree for degree in range(4) if rest_count==3*((degree+1)**2-1)),None)
    if source_degree is None:raise ValueError('完整 PLY 球谐布局无效')
    declared=metadata.get('source_gaussian_count')
    if declared is not None and (not isinstance(declared,int) or isinstance(declared,bool) or declared!=count):
        raise ValueError('来源场景的完整高斯点数与 PLY 实际顶点数不一致')
    points=original.get('gaussians')
    if not isinstance(points,list) or not points:raise ValueError('来源场景没有有效的高斯预览')
    indexed={}
    for point in points:
        if not isinstance(point,dict):raise ValueError('来源高斯记录无效')
        index=point.get('source_index')
        if index is None:continue
        if not isinstance(index,int) or isinstance(index,bool) or not 0<=index<count:
            raise ValueError('来源 source_index 必须是完整 PLY 范围内的整数')
        if index in indexed:raise ValueError('来源 source_index 重复，无法可靠保留编辑状态')
        indexed[index]=point
    identity_verified=metadata.get('source_ply_sha256')==actual_hash
    verified=indexed if identity_verified else {}
    complete_sh=all(point.get('sh_degree')==source_degree and len(point.get('sh',[]))==3*(source_degree+1)**2 for point in points)
    reuse=identity_verified and len(indexed)==len(points)==count and complete_sh
    if reuse:
        preview=copy.deepcopy(original)
        migrated=len(points)
        state_migrated=sum(any(key in point for key in _POINT_STATE_FIELDS) for point in points)
        unsampled=0
    else:
        preview=read_ply(ply,max_points=None)
        if preview['metadata']['source_ply_sha256']!=actual_hash or preview['metadata']['source_gaussian_count']!=count:
            raise ValueError('重新读取预览时完整 PLY 发生变化')
        if len(preview['gaussians'])!=count or preview['metadata'].get('preview_sh_degree')!=source_degree:
            raise ValueError('全量语义刷新必须保留完整高斯和全部球谐系数，拒绝采样预览')
        preserved={k:copy.deepcopy(v) for k,v in metadata.items()
                   if k not in {'sampling','class_catalog','semantics','semantic_sidecar',
                                'semantic_refinement_artifacts','scene_url','unsupported_classes','classes_lost_in_preview'}}
        preview['metadata']={**preserved,**preview['metadata']}
        preview['objects']=copy.deepcopy(original.get('objects',[]))
        preview['cameras']=copy.deepcopy(original.get('cameras',[]))
        migrated=state_migrated=0
        for point in preview['gaussians']:
            previous=verified.get(point['source_index'])
            if previous is None:continue
            migrated+=1
            copied=False
            for key in _POINT_STATE_FIELDS:
                if key in previous:
                    point[key]=copy.deepcopy(previous[key]);copied=True
            state_migrated+=int(copied)
        unsampled=len(verified)-migrated
    unmapped=[point for point in points if not identity_verified or point.get('source_index') is None]
    diagnostic={'method':'preserved_indexed_scene' if reuse else 'regenerated_preview',
                'source_identity_verified_for_old_mapping':identity_verified,
                'full_source_gaussian_count':count,'original_preview_count':len(points),
                'original_indexed_count':len(indexed),'verified_original_index_count':len(verified),
                'new_preview_count':len(preview['gaussians']),'matched_verified_index_count':migrated,
                'points_with_preserved_state':state_migrated,'unmapped_original_point_count':len(unmapped),
                'unmapped_original_points_with_edit_state':sum(any(key in point for key in _EDIT_FIELDS) for point in unmapped),
                'verified_original_indices_not_in_new_preview':unsampled,
                'original_geometry_and_sampling_preserved':reuse,
                'migration_rule':'Verified source_index only; never row order, nearest position, or guessed sampling'}
    preview['metadata'].update(source_ply_sha256=actual_hash,source_gaussian_count=count,
                               preview_gaussian_count=len(preview['gaussians']),
                               preview_sampled=False,display_mode='full',
                               preview_sh_degree=source_degree,source_sh_degree=source_degree,
                               semantic_preview_preservation=diagnostic)
    preview['metadata'].pop('scene_url',None)
    if not reuse:
        preview['metadata'].setdefault('warnings',[]).append(
            f'已从完整 PLY 恢复全部高斯及 SH{source_degree} 球谐；仅按已验证 source_index 保留 {migrated} 个匹配点的状态。'
            f'{len(unmapped)} 个旧点无法验证映射；这些状态未猜配到其他点，原运行保持不变。')
    return preview


def _preserve_category_state(original,scene):
    """Preserve display flags only for unambiguous, exact category unions."""
    diagnostic=scene['metadata']['semantic_preview_preservation']
    if not diagnostic['source_identity_verified_for_old_mapping']:
        diagnostic['preserved_category_state_labels']=[]
        return
    old={}
    for obj in original.get('objects',[]):
        if obj.get('semantic_granularity')=='category_union':old.setdefault(obj.get('label'),[]).append(obj)
    preserved=[]
    for obj in scene.get('objects',[]):
        candidates=old.get(obj.get('label'),[])
        if obj.get('semantic_granularity')!='category_union' or len(candidates)!=1:continue
        for key in ('visible','deleted','priority'):
            if key in candidates[0]:obj[key]=copy.deepcopy(candidates[0][key])
        preserved.append(obj['label'])
    diagnostic['preserved_category_state_labels']=preserved


def refine_scene_from_run(project_dir,source_run_id,output_dir,config,progress=None,cancelled=None):
    from .semantic_profiles import worker_semantic_config, resolve_semantic_profile
    if not isinstance(source_run_id,str) or not re.fullmatch(r'[a-f0-9]{32}',source_run_id):
        raise ValueError('来源运行 ID 无效')
    project=Path(project_dir).resolve();runs=project/'runs'
    if not runs.resolve().is_relative_to(project):raise ValueError('运行目录必须位于当前项目内')
    source=_within(runs/source_run_id,runs);out=_within(output_dir,runs)
    if out.parent!=runs or not re.fullmatch(r'[a-f0-9]{32}',out.name):raise ValueError('输出必须属于当前项目的运行目录')
    scene_file=_within(source/'scene.json',source)
    original=json.loads(scene_file.read_text())
    metadata=original.get('metadata',{})
    if metadata.get('backend')!='original_3dgs':raise ValueError('多轮语义需要本应用保存的原始 CUDA 完整模型')
    if not metadata.get('ply_path'):raise ValueError('来源结果缺少完整 PLY，不能优化采样 JSON')
    ply=_within(metadata['ply_path'],runs)
    if ply.name!='point_cloud.ply' or not ply.is_file():raise ValueError('完整 CUDA PLY 不可用')
    actual_hash=_sha(ply)
    if metadata.get('source_ply_sha256') not in (None,actual_hash):raise ValueError('源 PLY 与重建记录哈希不一致')
    cameras=copy.deepcopy(original.get('cameras',[]))
    if not cameras:raise ValueError('多轮语义需要已注册相机')
    masks_file=project/'masks.json'
    masks=json.loads(masks_file.read_text()) if masks_file.exists() else []
    if not isinstance(masks,list) or not masks:raise ValueError('请先识别或标注物品，名称不能代替训练掩码')
    camera_by_name={}
    for camera in cameras:
        name=Path(camera.get('image_name',camera.get('image_path',''))).name
        if not name or name in camera_by_name:raise ValueError('相机文件名不唯一，无法验证掩码对应关系')
        camera_by_name[name]=camera
    teacher_update=None
    if 'semantic_budget' in config or 'semantic_view_mode' in config:
        from .semantic_views import select_semantic_views, validate_mask_sources
        from .semantics import discover_objects
        resolved=resolve_semantic_profile(config)
        if not resolved.get('candidate_labels'):resolved.pop('candidate_labels',None)
        photo_paths=[str(project/'images'/name) for name in camera_by_name]
        if not all(Path(path).is_file() for path in photo_paths):
            raise ValueError('全视角语义需要当前已注册相机对应的原项目照片')
        chosen,coverage=select_semantic_views(photo_paths,resolved)
        names={Path(path).name for path in chosen}
        known={Path(row.get('image_path',row.get('image_name',''))).name for row in masks}
        missing=[path for path in chosen if Path(path).name not in known]
        if missing and config.get('provider','local')=='local':
            if any(camera_by_name[Path(path).name].get('mask_pixel_coordinates_verified') is not True for path in missing):
                raise ValueError('补充视角的照片与相机像素坐标尚未验证')
            if progress:progress(.95,f'补充 {len(missing)} 个训练视角的语义掩码')
            if cancelled and cancelled():raise RuntimeError('任务已取消')
            teacher=discover_objects(missing,{**resolved,'device':'cuda','semantic_view_mode':'all',
                'strict_query_labels':True,'generate_masks':True,
                'mask_output_dir':str(project/'masks'/uuid.uuid4().hex)})
            teacher_update={key:teacher.get(key) for key in ('source','coverage','timing','warnings')}
            if teacher.get('available') is False:
                raise ValueError('全视角掩码准备未完成：'+'; '.join(teacher.get('warnings',[])))
            masks.extend(teacher.get('masks',[]))
            temporary=masks_file.with_suffix('.tmp');temporary.write_text(json.dumps(masks,ensure_ascii=False));temporary.replace(masks_file)
        masks=[row for row in masks if Path(row.get('image_path',row.get('image_name',''))).name in names]
        masks=validate_mask_sources(masks,chosen)
        cameras=[camera for name,camera in camera_by_name.items() if name in names]
        camera_by_name={name:camera for name,camera in camera_by_name.items() if name in names}
    selected=[];skipped=[]
    for item in masks:
        record=dict(item)
        name=Path(record.get('image_path',record.get('image_name',''))).name
        camera=camera_by_name.get(name)
        if camera is None:
            skipped.append(name);continue
        if camera.get('mask_pixel_coordinates_verified') is not True:
            raise ValueError('掩码与注册相机的像素坐标尚未验证，需先确认去畸变/裁剪映射；不能直接按名称错配')
        if record.get('mask_path'):
            record['mask_path']=str(_within(record['mask_path'],project))
        record['image_name']=name;record['image_path']=name
        selected.append(record)
    if not selected:raise ValueError('没有已注册且坐标匹配的训练掩码')
    if cancelled and cancelled():raise RuntimeError('任务已取消')
    # Validate old mapping before spending CUDA time. Complete reliable previews
    # retain their exact geometry, sample order, provenance and per-point edits.
    preview=_prepare_semantic_preview(original,ply,actual_hash)
    out.mkdir(parents=True,exist_ok=True)
    cfg={**worker_semantic_config(config),'preset':'improved',
         'expected_gaussian_count':preview['metadata']['source_gaussian_count'],'source_ply_sha256':actual_hash,
         'approved_hierarchy':config.get('approved_hierarchy',[])}
    prior_meta=metadata.get('semantic_refinement_artifacts')
    if prior_meta and prior_meta.get('probabilities_path') and prior_meta.get('catalog_path'):
        prior_path=_within(prior_meta['probabilities_path'],runs)
        prior_catalog_path=_within(prior_meta['catalog_path'],runs)
        catalog=json.loads(prior_catalog_path.read_text())
        if catalog.get('source_ply_sha256')!=actual_hash:raise ValueError('已有语义概率与当前 PLY 不匹配')
        cfg['prior']={'path':str(prior_path),'source_ply_sha256':actual_hash,'classes':catalog['classes']}
    from .semantic_worker import refine_upstream_semantics
    result=refine_upstream_semantics(ply,cameras,selected,out/'semantic_refinement',cfg,progress,cancelled)
    catalog=json.loads(Path(result['catalog_path']).read_text())
    stats=result.get('stats') or json.loads(Path(result['stats_path']).read_text())
    with np.load(result['membership_path'],allow_pickle=False) as data:
        if str(data['source_ply_sha256'].item())!=actual_hash:raise ValueError('语义文件的模型身份校验失败')
        declared=data['classes'].tolist()
        memberships=np.column_stack([data[f'class_{i}'] for i in range(len(declared))])
    with np.load(result['probabilities_path'],allow_pickle=False) as data:
        if str(data['source_ply_sha256'].item())!=actual_hash or data['classes'].tolist()!=declared:
            raise ValueError('语义概率与成员文件身份不一致')
        probabilities=data['probabilities'].copy()
    preview['cameras']=cameras
    preview['metadata'].update(backend='original_3dgs',ply_path=str(ply))
    labels=catalog['classes'];edges=[]
    for edge in catalog.get('hierarchy_edges',[]):
        edges.append({'parent_idx':labels.index(edge['parent']),'child_idx':labels.index(edge['child']),
                      'relation':edge['relation'],'source':'caller_approved_and_multiview_mask_supported'})
    bound=bind_semantic_sidecar(preview,catalog,memberships,membership_classes=declared,
                                probabilities=probabilities,approved_edges=edges)
    scene=bound['scene']
    _preserve_category_state(original,scene)
    scene['metadata'].update({'semantic_refinement_requested':'cuda_iterative','semantic_refinement_applied':'cuda_iterative',
        'semantic_steps':stats.get('completed_steps'),'source_run_id':source_run_id,
        'semantic_timing':catalog.get('timing'), 'semantic_teacher_update':teacher_update,
        'semantic_coverage':stats.get('coverage'),
        'semantic_granularity':catalog.get('semantic_granularity'),
        'hierarchy_executed':catalog.get('hierarchy_executed'),
        'cross_view_confidence':catalog.get('cross_view_confidence'),
        'semantic_refinement_artifacts':{key:str(result[key]) for key in ('catalog_path','probabilities_path','membership_path','closed_membership_path','stats_path')},
        'semantic_training_unregistered_mask_frames':sorted(set(skipped)),
        'rgb_retrained_for_semantic_refinement':False,'source_ply_sha256':actual_hash})
    if _sha(ply)!=actual_hash:raise RuntimeError('原始 PLY 在语义任务中发生变化')
    path=out/'scene.json';write_scene(scene,path)
    return {'scene_path':str(path),'ply_path':str(ply),'semantic_artifacts':result}
