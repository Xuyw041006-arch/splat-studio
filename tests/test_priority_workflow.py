import io,json,zipfile
from pathlib import Path
import numpy as np
from PIL import Image
import pytest
from backend.priority_workflow import (parse_request,create_review,confirm_review,validate_approval,
    write_review_package,import_review_package,read_review)
from backend.priority_detail import detail_profile
from backend.sparse_cuda import patch_training_source

def project(tmp):
    tmp.mkdir();(tmp/'images').mkdir();(tmp/'masks').mkdir()
    images=[];records=[]
    for i in range(2):
        name=f'{i:04}.jpg';Image.new('RGB',(60,40),(100+i,120,90)).save(tmp/'images'/name)
        mask=np.zeros((40,60),np.uint8);mask[4:29,20:40]=255;mp=tmp/'masks'/f'{i}.png';Image.fromarray(mask).save(mp)
        images.append({'name':name,'width':60,'height':40})
        records.append({'image_path':str(tmp/'images'/name),'mask_path':str(mp),'mask_id':f'm{i}','label':'stuffed bear','confidence':.7})
    (tmp/'project.json').write_text(json.dumps({'id':'a'*32,'images':images}));(tmp/'masks.json').write_text(json.dumps(records));return records

def approve(p,records):
    r=create_review(p,'重点重建熊',records);a=confirm_review(p,r['id'],[m['id'] for m in r['matches']])
    cfg={'priority_request':'重点重建熊','priority_objects':a['labels'],'priority_approval_sha256':a['sha256']}
    return r,a,cfg

def test_priority_text_maps_requests_without_claiming_grounding():
    assert [x['label'] for x in parse_request('重点重建熊，咖啡杯')]==['stuffed bear','coffee mug']
    assert parse_request('unknown object')[0]['mapping']=='literal'
    assert parse_request('熊、玩具熊')[0]['label']=='stuffed bear'
    assert len(parse_request('熊、玩具熊'))==1

def test_review_confirm_is_bound_to_selected_pixels_and_images(tmp_path):
    p=tmp_path/'p';records=project(p);r,a,cfg=approve(p,records)
    assert len(r['matches'])==2 and not r['missing']
    assert validate_approval(p,cfg)['sha256']==a['sha256']
    chosen=json.loads((p/'masks.json').read_text())[-1];Image.new('L',(60,40),255).save(chosen['mask_path'])
    with pytest.raises(ValueError,match='掩码内容'):validate_approval(p,cfg)

def test_missing_target_blocks_confirmation_and_training(tmp_path):
    p=tmp_path/'p';records=project(p);r=create_review(p,'熊、咖啡杯',records)
    assert r['missing']==['coffee mug']
    with pytest.raises(ValueError,match='没有确认区域'):confirm_review(p,r['id'],[r['matches'][0]['id']])
    with pytest.raises(ValueError,match='尚未确认'):validate_approval(p,{'priority_request':'熊'})

def test_rejected_priority_detections_do_not_reenter_semantics(tmp_path):
    p=tmp_path/'p';records=project(p);r=create_review(p,'熊',records)
    confirm_review(p,r['id'],[r['matches'][0]['id']])
    masks=json.loads((p/'masks.json').read_text())
    assert len(masks)==1 and masks[0]['mask_id']==r['matches'][0]['id']

def test_review_rejects_foreign_selection_and_changed_photos(tmp_path):
    p=tmp_path/'p';records=project(p);r=create_review(p,'熊',records)
    with pytest.raises(ValueError,match='不属于'):confirm_review(p,r['id'],['foreign'])
    Image.new('RGB',(60,40),'red').save(p/'images/0000.jpg')
    with pytest.raises(ValueError,match='照片已变化'):read_review(p,r['id'])

def test_cloud_review_roundtrip_requires_matching_input(tmp_path):
    p=tmp_path/'p';records=project(p);r=create_review(p,'熊',records);z=tmp_path/'review.zip';write_review_package(p,r,z)
    q=tmp_path/'q';project(q)
    imported=import_review_package(q,z);assert len(imported['matches'])==2
    a=confirm_review(q,imported['id'],[m['id'] for m in imported['matches']])
    assert validate_approval(q,{'priority_request':'熊','priority_objects':a['labels'],'priority_approval_sha256':a['sha256']})
    Image.new('RGB',(60,40),'black').save(q/'images/0000.jpg')
    with pytest.raises(ValueError,match='不匹配'):import_review_package(q,z)

def test_review_zip_rejects_escaping_paths(tmp_path):
    p=tmp_path/'p';project(p);blob=io.BytesIO()
    with zipfile.ZipFile(blob,'w') as z:z.writestr('../escape','bad')
    blob.seek(0)
    with pytest.raises(ValueError,match='越界'):import_review_package(p,blob)

def test_native_api_import_and_confirmation_roundtrip(tmp_path,monkeypatch):
    import importlib
    from fastapi.testclient import TestClient
    api=importlib.import_module('backend.app')
    p=tmp_path/'source';records=project(p);review=create_review(p,'熊',records)
    z=tmp_path/'review.zip';write_review_package(p,review,z)
    q=tmp_path/('a'*32);project(q);monkeypatch.setattr(api,'DATA',tmp_path)
    client=TestClient(api.app)
    response=client.post(f'/api/projects/{q.name}/priorities/import',files={'file':('review.zip',z.read_bytes(),'application/zip')})
    assert response.status_code==200,response.text
    imported=response.json();assert len(imported['matches'])==2
    assert client.get(imported['matches'][0]['preview_url']).status_code==200
    response=client.post(f'/api/projects/{q.name}/priorities/confirm',json={'review_id':imported['id'],'selected_ids':[r['id'] for r in imported['matches']]})
    assert response.status_code==200,response.text
    assert response.json()['status']=='confirmed'

@pytest.mark.parametrize('mode,total,steps',[('fast',7000,400),('balanced',15000,1200),('fine',22000,2400)])
def test_detail_stays_inside_original_total_budget(mode,total,steps):
    p=detail_profile(mode,total);assert total-p['start']+1==steps
    assert p['start']>6000 and p['growth_fraction']<=.25

def test_detail_patch_compiles_against_pinned_original_trainer(tmp_path):
    source=Path(__file__).resolve().parents[1]/'vendor/gaussian-splatting/train.py'
    target=tmp_path/'train.py';r=patch_training_source(source,target,priority_dir=tmp_path,priority_detail=detail_profile('balanced',15000))
    assert r['priority_detail']['steps']==1200
    code=target.read_text();compile(code,str(target),'exec')
    assert code.count('priority_detail.finish(')==1 and code.count('priority_detail.densify(')==1

def test_priority_densification_accepts_upstream_index_visibility_without_broadcasting():
    import torch
    from types import SimpleNamespace
    from backend.priority_detail import TRAINING_HOOK
    scope={'torch':torch,'splat_np':np}
    exec(TRAINING_HOOK.replace('__DETAIL_CONFIG__','{}').replace('__DETAIL_MASK_DIR__',repr('/unused')),scope)
    detail=scope['SplatPriorityDetail'].__new__(scope['SplatPriorityDetail'])
    detail.config={'start':1,'steps':600,'interval':100,'max_new_points':2,'growth_fraction':.5}
    detail.limit=None;detail.added=0;detail.scene=SimpleNamespace(cameras_extent=1)
    mask=torch.zeros((6,6),dtype=torch.bool);mask[2:4,2:4]=True;detail.mask=lambda cam:mask
    class Points:
        get_xyz=torch.tensor([[0.,0.,0.],[.8,0.,0.],[0.,0.,0.],[.9,.9,0.]])
        get_opacity=torch.full((4,1),.8)
        def densify_and_clone(self,grads,*args):
            self.chosen=torch.where(grads[:,0]>0)[0].tolist()
            self.get_xyz=torch.cat([self.get_xyz,self.get_xyz[self.chosen]])
        def densify_and_split(self,*args):pass
    points=Points();cam=SimpleNamespace(full_proj_transform=torch.eye(4),image_width=6,image_height=6)
    viewspace=SimpleNamespace(grad=torch.tensor([[2.,0.,0.],[10.,0.,0.],[1.,0.,0.],[9.,0.,0.]]))
    detail.densify(points,cam,viewspace,torch.tensor([[0],[2],[3]]),torch.tensor([1.,0.,1.,1.]),1)
    assert points.chosen==[0] and detail.added==1 and detail.limit==2
