"""Portable, source-bound semantic packages; no pickle or executable imports."""
import hashlib
import json
from pathlib import Path, PurePosixPath
import shutil
import stat
import zipfile
import numpy as np
from .semantic_sidecar import bind_semantic_sidecar

MAX_BYTES=2*1024**3


def sha256(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def _relative(value):
    if not isinstance(value,str) or not value or '\\' in value or ':' in value or '\x00' in value:
        raise ValueError('语义包包含无效路径')
    path=PurePosixPath(value)
    if path.is_absolute() or any(p in ('','.','..') for p in value.split('/')):
        raise ValueError('语义包路径不能越界')
    return path


def write_semantic_package(folder,destination):
    folder=Path(folder);destination=Path(destination)
    catalog=json.loads((folder/'catalog.json').read_text())
    if catalog.get('status')!='completed':raise ValueError('只能导出已完成的语义训练')
    names={'catalog.json'}
    for item in catalog['artifacts'].values():
        name=str(_relative(item['path']))
        if sha256(folder/name)!=item['sha256']:raise ValueError('语义文件校验失败')
        names.add(name)
    for name in ('stats.json','targets.json','granularity.json','scene_context.json'):
        if (folder/name).is_file():names.add(name)
    with zipfile.ZipFile(destination,'x',compression=zipfile.ZIP_STORED) as archive:
        for name in sorted(names):archive.write(folder/name,name)
    return destination


def import_semantic_package(scene,archive_path,destination):
    destination=Path(destination)
    destination.mkdir(exist_ok=False)
    try:
        with zipfile.ZipFile(archive_path) as archive:
            members=archive.infolist()
            if len(members)>200 or sum(m.file_size for m in members)>MAX_BYTES:
                raise ValueError('语义包解压大小或文件数超过上限')
            paths=set();catalogs=[]
            for member in members:
                name=member.filename.rstrip('/') if member.is_dir() else member.filename
                _relative(name)
                if name.casefold() in paths or stat.S_ISLNK(member.external_attr>>16):
                    raise ValueError('语义包不能包含重名文件或符号链接')
                paths.add(name.casefold())
                if member.is_dir():continue
                if member.flag_bits&1:raise ValueError('不支持加密语义包')
                if Path(name).suffix not in {'.json','.npz'}:raise ValueError('语义包仅支持 JSON 与 NPZ 数据文件')
                if Path(name).name=='catalog.json':catalogs.append(name)
            if len(catalogs)!=1:raise ValueError('语义包需要唯一的 catalog.json')
            for member in members:
                if member.is_dir():continue
                path=destination/member.filename;path.parent.mkdir(parents=True,exist_ok=True)
                with archive.open(member) as source,path.open('xb') as target:shutil.copyfileobj(source,target,1024**2)
        catalog_path=destination/catalogs[0]
        if catalog_path.stat().st_size>16*1024**2:raise ValueError('语义目录文件过大')
        catalog=json.loads(catalog_path.read_text());meta=scene.get('metadata',{})
        count=meta.get('source_gaussian_count');labels=catalog.get('classes',[])
        if catalog.get('status')!='completed' or catalog.get('source_ply_sha256')!=meta.get('source_ply_sha256') or catalog.get('gaussian_count')!=count:
            raise ValueError('语义包与当前点云不匹配：需要相同 PLY SHA256 和高斯点数')
        if not labels or len(labels)>512 or count*len(labels)*5>MAX_BYTES:
            raise ValueError('语义类别数或完整概率矩阵超过内存上限')
        loaded={}
        for kind in ('probabilities','membership','closed_probabilities','closed_membership'):
            record=catalog.get('artifacts',{}).get(kind)
            if not isinstance(record,dict):raise ValueError('语义包缺少完整概率或成员文件')
            source=catalog_path.parent/str(_relative(record.get('path')))
            if not source.is_file() or sha256(source)!=record.get('sha256'):raise ValueError('语义文件 SHA256 校验失败')
            # NPZ is itself ZIP; bound decompressed arrays before NumPy allocation.
            with zipfile.ZipFile(source) as zipped:
                if sum(m.file_size for m in zipped.infolist())>MAX_BYTES:raise ValueError('NPZ 展开数组超过内存上限')
            with np.load(source,allow_pickle=False) as data:
                if data['classes'].tolist()!=labels or str(data['source_ply_sha256'].item())!=meta['source_ply_sha256']:
                    raise ValueError('语义列顺序或来源身份不一致')
                if 'probabilities' in kind:
                    values=data['probabilities']
                    if values.shape!=(count,len(labels)) or values.dtype.kind!='f' or not np.isfinite(values).all() or (values<0).any() or (values>1).any():
                        raise ValueError('语义概率形状或范围无效')
                else:
                    columns=[data[f'class_{i}'] for i in range(len(labels))]
                    if any(v.shape!=(count,) or v.dtype!=np.bool_ for v in columns):raise ValueError('语义成员数组形状无效')
                    values=np.column_stack(columns)
                if kind in {'probabilities','membership'}:loaded[kind]=values
        edges=[]
        for edge in catalog.get('hierarchy_edges',[]):
            if edge.get('observation_criteria_met') is not True:raise ValueError('语义包包含未通过观测验证的层级边')
            edges.append({'parent_idx':labels.index(edge['parent']),'child_idx':labels.index(edge['child']),
                          'relation':edge['relation'],'source':edge.get('source','multiview_supported')})
        bound=bind_semantic_sidecar(scene,catalog,loaded['membership'],membership_classes=labels,
                                   probabilities=loaded['probabilities'],approved_edges=edges)
        result=bound['scene']
        context_path=catalog_path.parent/'scene_context.json'
        if context_path.is_file():
            context=json.loads(context_path.read_text())
            if context.get('source_ply_sha256')!=meta['source_ply_sha256']:raise ValueError('场景相机与点云身份不一致')
            cameras=context.get('cameras',[])
            if not isinstance(cameras,list) or len(cameras)>300:raise ValueError('导入相机数量无效')
            result['cameras']=cameras
            for key,value in context.get('viewer',{}).items():
                if key in {'viewer_camera','viewer_core_region','viewer_grid'}:result['metadata'][key]=value
        result['metadata'].update(semantic_package_sha256=sha256(archive_path),
            semantic_granularity=catalog.get('semantic_granularity'),
            cross_view_confidence=catalog.get('cross_view_confidence'),
            hierarchy_executed=catalog.get('hierarchy_executed',False),
            semantic_import_validated=True)
        return result
    except BaseException:
        shutil.rmtree(destination,ignore_errors=True)
        raise
