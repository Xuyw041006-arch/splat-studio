"""Grounded, reviewable priority requests. Text never substitutes for pixels."""
from __future__ import annotations
import hashlib
import io
import json
from pathlib import Path
import re
import uuid
import zipfile
import numpy as np
from PIL import Image, ImageOps
from .cloud_tasks import json_bytes, sha256, owned_file, relative_name
from .semantics import _load_mask, validate_mask_sources

ALIASES = {
    '熊':'stuffed bear','小熊':'stuffed bear','玩具熊':'stuffed bear','毛绒熊':'stuffed bear',
    '熊耳':'bear ear','熊耳朵':'bear ear','熊鼻子':'bear nose','杯子':'cup','茶杯':'cup',
    '咖啡杯':'coffee mug','杯把':'mug handle','杯柄':'mug handle','茶':'glass of tea',
    '盘子':'plate','饼干':'three cookies','纸巾':'paper napkin','羊':'sheep',
    '椅子':'chair','桌子':'table','灯':'lamp','台灯':'lamp','植物':'plant','书':'book',
    '沙发':'sofa','瓶子':'bottle','显示器':'monitor',
}
FORMAT='splat-priority-review/1'

def parse_request(text):
    if not isinstance(text,str) or len(text)>1000: raise ValueError('重点描述最多 1000 个字符')
    text=re.sub(r'^(?:请|帮我|把|将|优先重建|重点重建|精细重建|加强|增强|优先|重点|重建|一下|这些|物品|物体|：|:|\s)+','',text.strip())
    values=[s.strip(' 。.!！') for s in re.split(r'[,，、;；\n]+',text) if s.strip(' 。.!！')]
    if len(values)>20: raise ValueError('每次最多选择 20 类重点物品')
    rows=[];seen=set()
    for value in values:
        key=' '.join(value.casefold().split());label=ALIASES.get(key,key)
        if label in seen:continue
        seen.add(label);rows.append({'input':value,'label':label,'mapping':'dictionary' if key in ALIASES else 'literal'})
    return rows

def image_identity(project):
    p=Path(project);manifest=json.loads((p/'project.json').read_text())
    return [{'name':i['name'],'sha256':sha256(owned_file(p,'images/'+i['name']))} for i in manifest['images']]

def create_review(project,request,records):
    p=Path(project);requested=parse_request(request)
    if not requested: raise ValueError('请输入要精细重建的物品名称')
    paths=[str(p/'images'/i['name']) for i in json.loads((p/'project.json').read_text())['images']]
    records=validate_mask_sources(records,paths)
    review_id=uuid.uuid4().hex;folder=p/'priority-reviews'/review_id;folder.mkdir(parents=True)
    matches=[];wanted={r['label'] for r in requested}
    for record in records:
        label=' '.join(str(record.get('label','')).casefold().split())
        label=ALIASES.get(label,label)
        if label not in wanted:continue
        path=Path(record['image_path']);im=ImageOps.exif_transpose(Image.open(path)).convert('RGB')
        mask=np.asarray(Image.fromarray(_load_mask(record).astype('uint8')*255).resize(im.size,Image.Resampling.NEAREST))>0
        if not mask.any():continue
        # Each review row is one real mask in one photo. No guessed cross-view instance ID.
        rid=uuid.uuid4().hex;mask_path=folder/(rid+'.png');Image.fromarray(mask.astype('uint8')*255).save(mask_path)
        rgb=np.asarray(im).copy();tint=np.array([74,210,156]);rgb[mask]=(.5*rgb[mask]+.5*tint).astype('uint8')
        preview=Image.fromarray(rgb);preview.thumbnail((800,600));preview.save(folder/(rid+'-preview.jpg'),quality=90)
        yy,xx=np.where(mask)
        matches.append({'id':rid,'label':label,'image_name':path.name,'confidence':float(record.get('confidence',1)),
            'area_fraction':float(mask.mean()),'bounds':[int(xx.min()),int(yy.min()),int(xx.max()+1),int(yy.max()+1)],
            'touches_border':bool(mask[0].any() or mask[-1].any() or mask[:,0].any() or mask[:,-1].any()),
            'mask_file':rid+'.png','mask_sha256':sha256(mask_path),'preview_file':rid+'-preview.jpg'})
    matched={m['label'] for m in matches}
    result={'format':FORMAT,'id':review_id,'request':request,'requested':requested,'images':image_identity(p),
            'matches':matches,'missing':[r['label'] for r in requested if r['label'] not in matched],
            'status':'awaiting_confirmation','identity_scope':'per-photo masks; no automatic same-instance claim'}
    (folder/'review.json').write_bytes(json_bytes(result))
    return result

def read_review(project,review_id):
    if not re.fullmatch('[a-f0-9]{32}',str(review_id)):raise ValueError('重点确认记录无效')
    p=Path(project);folder=p/'priority-reviews'/review_id
    result=json.loads(owned_file(p,folder/'review.json').read_text())
    if result['images']!=image_identity(p):raise ValueError('照片已变化，请重新匹配重点物品')
    for row in result['matches']:
        if sha256(owned_file(folder,row['mask_file']))!=row['mask_sha256']:raise ValueError('重点掩码已变化，请重新确认')
    return result

def confirm_review(project,review_id,selected):
    p=Path(project);review=read_review(p,review_id)
    if not isinstance(selected,list) or not selected or len(selected)!=len(set(selected)):raise ValueError('请至少确认一个实际区域')
    by_id={m['id']:m for m in review['matches']}
    if any(i not in by_id for i in selected):raise ValueError('选择包含不属于当前确认记录的区域')
    labels={by_id[i]['label'] for i in selected};missing={r['label'] for r in review['requested']}-labels
    if missing:raise ValueError('以下重点没有确认区域：'+', '.join(sorted(missing))+'。请修改描述或补充标注。')
    rows=[];mask_dir=p/'masks'/'priority-confirmed';mask_dir.mkdir(parents=True,exist_ok=True)
    for rid in selected:
        row=by_id[rid];source=p/'priority-reviews'/review_id/row['mask_file'];target=mask_dir/(rid+'.png');target.write_bytes(source.read_bytes())
        rows.append({'image_path':str(p/'images'/row['image_name']),'image_name':row['image_name'],
            'mask_path':str(target),'mask_id':rid,'label':row['label'],'confidence':row['confidence'],
            'annotation_source':'priority_confirmed','priority_review_id':review_id,'level':'object',
            'source_image_sha256':sha256(p/'images'/row['image_name'])})
    existing=json.loads((p/'masks.json').read_text()) if (p/'masks.json').is_file() else []
    # An explicit review also excludes rejected detections of these labels from
    # later semantic supervision; do not silently restore a rejected instance.
    existing=[r for r in existing if r.get('annotation_source')!='priority_confirmed'
        and ALIASES.get(str(r.get('label','')).casefold(),str(r.get('label','')).casefold()) not in labels]+rows
    (p/'masks.json').write_bytes(json_bytes(existing))
    signature=hashlib.sha256(json_bytes({'images':review['images'],'selection':[(i,by_id[i]['mask_sha256']) for i in sorted(selected)]})).hexdigest()
    approval={'review_id':review_id,'selected_ids':selected,'labels':sorted(labels),'images':review['images'],'sha256':signature,'status':'confirmed'}
    (p/'priority-approval.json').write_bytes(json_bytes(approval));return approval

def validate_approval(project,config):
    """Portable confirmation is bound to exact pixels, mask content and selection."""
    if not config.get('priority_request'):return None # Historical explicit priority_objects remains a library path.
    p=Path(project);path=p/'priority-approval.json'
    if not path.is_file():raise ValueError('重点物品尚未确认：请先匹配照片中的区域并确认')
    approval=json.loads(path.read_text())
    if approval['sha256']!=config.get('priority_approval_sha256') or approval['images']!=image_identity(p):raise ValueError('重点确认与当前照片不一致，请重新确认')
    if {r['label'] for r in parse_request(config['priority_request'])}!=set(approval['labels']):raise ValueError('重点描述已变化，请重新匹配')
    if set(config.get('priority_objects',[]))!=set(approval['labels']):raise ValueError('重点训练标签与确认不一致')
    records=json.loads((p/'masks.json').read_text());ids=set(approval['selected_ids'])
    selected=[r for r in records if r.get('annotation_source')=='priority_confirmed' and r.get('mask_id') in ids]
    if {r['mask_id'] for r in selected}!=ids:raise ValueError('重点确认掩码缺失')
    digest=hashlib.sha256(json_bytes({'images':approval['images'],'selection':sorted((r['mask_id'],sha256(owned_file(p,r['mask_path']))) for r in selected)})).hexdigest()
    if digest!=approval['sha256']:raise ValueError('重点掩码内容已变化，确认失效')
    return approval

def write_review_package(project,review,destination):
    p=Path(project);folder=p/'priority-reviews'/review['id'];files={'review.json':json_bytes(review)}
    # Carry teacher masks for subsequent semantic training, excluding filesystem paths.
    masks=json.loads((p/'masks.json').read_text());portable=[]
    for index,r in enumerate(masks):
        row={k:v for k,v in r.items() if not k.endswith('_path') and k!='mask'}
        row['image_name']=Path(r['image_path']).name;name=f'teacher/{index:06}.png'
        b=io.BytesIO();Image.fromarray(_load_mask(r).astype('uint8')*255).save(b,format='PNG');files[name]=b.getvalue();row['mask_file']=name;portable.append(row)
    files['teacher.json']=json_bytes(portable)
    for row in review['matches']:
        for key in ('mask_file','preview_file'):files[row[key]]=owned_file(folder,row[key]).read_bytes()
    with zipfile.ZipFile(destination,'x',zipfile.ZIP_DEFLATED) as z:
        for name,data in files.items():z.writestr(name,data)

def import_review_package(project,source):
    p=Path(project)
    with zipfile.ZipFile(source) as z:
        infos=z.infolist();names=[i.filename for i in infos]
        if len(names)!=len(set(names)) or len(names)>12000 or sum(i.file_size for i in infos)>512*1024**2:raise ValueError('重点确认包超过限制或有重复文件')
        for name in names:relative_name(name)
        if any(i.file_size>48*1024**2 for i in infos):raise ValueError('重点确认包单文件过大')
        review=json.loads(z.read('review.json'))
        if review.get('format')!=FORMAT or review.get('images')!=image_identity(p):raise ValueError('确认包与当前输入照片不匹配')
        rid=review.get('id');
        if not re.fullmatch('[a-f0-9]{32}',str(rid)):raise ValueError('重点确认包 ID 无效')
        folder=p/'priority-reviews'/rid
        if folder.exists():return read_review(p,rid)
        # Validate all payloads before publishing a review or replacing teacher masks.
        staged={};records=[]
        for row in review['matches']:
            for key in ('mask_file','preview_file'):
                name=relative_name(row[key]);data=z.read(name)
                if '/' in name:raise ValueError('预览路径无效')
                with Image.open(io.BytesIO(data)) as im:
                    if im.width*im.height>48_000_000:raise ValueError('确认图像过大')
                    im.verify()
                staged[name]=data
            if hashlib.sha256(staged[row['mask_file']]).hexdigest()!=row['mask_sha256']:raise ValueError('确认包掩码校验失败')
        valid={i['name'] for i in review['images']}
        for index,row in enumerate(json.loads(z.read('teacher.json'))):
            if row['image_name'] not in valid:raise ValueError('教师掩码不属于当前照片')
            name=relative_name(row.pop('mask_file'));raw=z.read(name)
            with Image.open(io.BytesIO(raw)) as im:
                if im.width*im.height>48_000_000:raise ValueError('教师掩码过大')
                im.verify()
            target=p/'masks'/('review-'+rid)/f'{index:06}.png';staged[str(target)]=raw
            records.append({**row,'image_path':str(p/'images'/row['image_name']),'mask_path':str(target)})
    folder.mkdir(parents=True)
    for name,data in staged.items():
        target=Path(name) if Path(name).is_absolute() else folder/name;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(data)
    (folder/'review.json').write_bytes(json_bytes(review))
    existing=json.loads((p/'masks.json').read_text()) if (p/'masks.json').is_file() else []
    (p/'masks.json').write_bytes(json_bytes([r for r in existing if r.get('annotation_source')=='manual']+records))
    return read_review(p,rid)

def public_review(project,review):
    pid=Path(project).name
    return {**review,'matches':[{**r,'preview_url':f'/api/projects/{pid}/files/priority-reviews/{review["id"]}/{r["preview_file"]}'} for r in review['matches']]}
