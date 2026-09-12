import hashlib
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import pytest
from backend.gaussian_io import write_ply,read_ply,write_scene
from backend.semantic_service import refine_scene_from_run

SOURCE='a'*32
TARGET='b'*32

def project(tmp_path):
    p=tmp_path/'project';run=p/'runs'/SOURCE
    ply=run/'model/point_cloud/iteration_15000/point_cloud.ply'
    write_ply({'gaussians':[{'position':[0,0,2],'color':[.5,.2,.3],'scale':[.01]*3,'rotation':[1,0,0,0],'opacity':.8},
                            {'position':[.1,0,2],'color':[.2,.8,.3],'scale':[.01]*3,'rotation':[1,0,0,0],'opacity':.8}]},ply)
    scene=read_ply(ply);scene['metadata'].update(backend='original_3dgs',ply_path=str(ply))
    scene['cameras']=[{'image_name':'view.png','width':4,'height':4,'intrinsics':{'fx':4,'fy':4,'cx':2,'cy':2},
        'world_to_camera':np.eye(4).tolist(),'mask_pixel_coordinates_verified':True}]
    write_scene(scene,run/'scene.json')
    (p/'masks.json').write_text(json.dumps([{'image_name':'view.png','label':'bear','mask':[[1,1],[1,1]]}]))
    return p,ply


def fake_worker(monkeypatch,called):
    from backend import semantic_worker
    def run(ply,cameras,masks,out,cfg,progress,cancelled):
        called.append(cfg)
        assert cfg['expected_gaussian_count']==2
        out=Path(out);out.mkdir()
        sha=hashlib.sha256(Path(ply).read_bytes()).hexdigest()
        catalog={'source_ply_sha256':sha,'gaussian_count':2,'classes':['bear'],'hierarchy_edges':[]}
        (out/'catalog.json').write_text(json.dumps(catalog));(out/'stats.json').write_text('{}')
        values=np.array([[.9],[.1]],np.float32)
        np.savez(out/'probabilities.npz',probabilities=values,classes=np.array(['bear']),source_ply_sha256=np.array(sha))
        np.savez(out/'semantic_membership.npz',class_0=np.array([True,False]),classes=np.array(['bear']),source_ply_sha256=np.array(sha))
        np.savez(out/'closed_semantic_membership.npz',class_0=np.array([True,False]))
        return {key:str(out/name) for key,name in [('catalog_path','catalog.json'),('probabilities_path','probabilities.npz'),
            ('membership_path','semantic_membership.npz'),('closed_membership_path','closed_semantic_membership.npz'),('stats_path','stats.json')]}
    monkeypatch.setattr(semantic_worker,'refine_upstream_semantics',run)


def test_semantic_only_preserves_ply_and_reuses_complete_prior(tmp_path,monkeypatch):
    p,ply=project(tmp_path);before=ply.read_bytes();called=[];fake_worker(monkeypatch,called)
    result=refine_scene_from_run(p,SOURCE,p/'runs'/TARGET,{'semantic_steps':400})
    scene=json.loads(Path(result['scene_path']).read_text())
    assert ply.read_bytes()==before
    assert scene['metadata']['rgb_retrained_for_semantic_refinement'] is False
    assert len(scene['gaussians'])==2 and [g['source_index'] for g in scene['gaussians']]==[0,1]
    assert len(scene['gaussians'][0]['semantic_ids'])==2
    assert scene['gaussians'][1]['semantic_ids']==['scene']
    assert scene['gaussians'][0]['source']=='unknown'
    result2=refine_scene_from_run(p,TARGET,p/'runs'/('c'*32),{'semantic_steps':1200})
    assert called[1]['prior']['source_ply_sha256']==hashlib.sha256(before).hexdigest()
    assert called[1]['prior']['classes']==['bear']
    assert Path(result2['ply_path'])==ply


def test_unverified_mask_coordinates_rejected_before_training(tmp_path,monkeypatch):
    p,ply=project(tmp_path);f=p/'runs'/SOURCE/'scene.json';scene=json.loads(f.read_text())
    scene['cameras'][0]['mask_pixel_coordinates_verified']=False;write_scene(scene,f)
    called=[];fake_worker(monkeypatch,called)
    with pytest.raises(ValueError,match='像素坐标'):
        refine_scene_from_run(p,SOURCE,p/'runs'/TARGET,{})
    assert called==[]


def test_model_identity_and_paths_are_enforced(tmp_path):
    p,ply=project(tmp_path)
    with pytest.raises(ValueError,match='ID'):
        refine_scene_from_run(p,'../outside',p/'runs'/TARGET,{})
    with pytest.raises(ValueError,match='当前项目'):
        refine_scene_from_run(p,SOURCE,tmp_path/TARGET,{})
    f=p/'runs'/SOURCE/'scene.json';scene=json.loads(f.read_text())
    scene['metadata']['source_ply_sha256']='0'*64;write_scene(scene,f)
    with pytest.raises(ValueError,match='哈希'):
        refine_scene_from_run(p,SOURCE,p/'runs'/TARGET,{})


def test_cancel_before_worker(tmp_path,monkeypatch):
    p,ply=project(tmp_path);called=[];fake_worker(monkeypatch,called)
    with pytest.raises(RuntimeError,match='取消'):
        refine_scene_from_run(p,SOURCE,p/'runs'/TARGET,{},cancelled=lambda:True)
    assert called==[]


def test_complete_indexed_preview_keeps_sample_order_geometry_and_edit_state(tmp_path,monkeypatch):
    p,ply=project(tmp_path);source=p/'runs'/SOURCE/'scene.json'
    original=json.loads(source.read_text());original['gaussians'].reverse()
    original['gaussians'][0].update(hidden=True,reconstruction_weight=2.,source='observed',
                                    confidence=.6,provenance='retained_source_record',sh=[1.,-2.,3.]+[0.]*45)
    original['objects']=[{'id':'scene','label':'My scene','level':'scene','parent_id':None,'visible':True},
                         {'id':'old-bear','label':'bear','semantic_granularity':'category_union',
                          'visible':False,'deleted':True,'priority':True}]
    write_scene(original,source);source_bytes=source.read_bytes();ply_bytes=ply.read_bytes()
    called=[];fake_worker(monkeypatch,called)
    from backend import semantic_service
    def do_not_resample(*args,**kwargs):raise AssertionError('A complete indexed preview must not be regenerated')
    monkeypatch.setattr(semantic_service,'read_ply',do_not_resample)
    result=refine_scene_from_run(p,SOURCE,p/'runs'/TARGET,{})
    scene=json.loads(Path(result['scene_path']).read_text())
    assert [g['source_index'] for g in scene['gaussians']]==[1,0]
    for before,after in zip(original['gaussians'],scene['gaussians']):
        assert all(after[key]==value for key,value in before.items())
    bear=next(obj for obj in scene['objects'] if obj['label']=='bear')
    assert bear['visible'] is False and bear['deleted'] is True and bear['priority'] is True
    info=scene['metadata']['semantic_preview_preservation']
    assert info['method']=='preserved_indexed_scene' and info['original_geometry_and_sampling_preserved']
    assert info['matched_verified_index_count']==2 and info['unmapped_original_point_count']==0
    assert info['preserved_category_state_labels']==['bear']
    assert source.read_bytes()==source_bytes and ply.read_bytes()==ply_bytes


def test_partial_mapping_migrates_by_source_index_never_old_row_number(tmp_path,monkeypatch):
    p,ply=project(tmp_path);source=p/'runs'/SOURCE/'scene.json'
    original=json.loads(source.read_text());original['gaussians'].reverse()
    original['gaussians'][0].update(hidden=True,reconstruction_weight=2.,source='observed')  # Verified source index 1.
    original['gaussians'][1].pop('source_index')
    original['gaussians'][1].update(hidden=True,reconstruction_weight=4.,source='inferred')
    write_scene(original,source);source_bytes=source.read_bytes()
    called=[];fake_worker(monkeypatch,called)
    result=refine_scene_from_run(p,SOURCE,p/'runs'/TARGET,{})
    scene=json.loads(Path(result['scene_path']).read_text());points=scene['gaussians']
    assert [g['source_index'] for g in points]==[0,1]
    assert 'hidden' not in points[0] and 'reconstruction_weight' not in points[0]
    assert points[0]['source']=='unknown'
    assert points[1]['hidden'] is True and points[1]['reconstruction_weight']==2. and points[1]['source']=='observed'
    info=scene['metadata']['semantic_preview_preservation']
    assert info['method']=='regenerated_preview' and info['matched_verified_index_count']==1
    assert info['unmapped_original_point_count']==1 and info['unmapped_original_points_with_edit_state']==1
    assert source.read_bytes()==source_bytes


@pytest.mark.parametrize('missing',['indices','hash'])
def test_unknown_mapping_is_not_guessed_even_when_counts_and_order_match(tmp_path,monkeypatch,missing):
    p,ply=project(tmp_path);source=p/'runs'/SOURCE/'scene.json'
    original=json.loads(source.read_text())
    for point in original['gaussians']:
        point.update(hidden=True,reconstruction_weight=2.)
        if missing=='indices':point.pop('source_index')
    if missing=='hash':original['metadata'].pop('source_ply_sha256')
    write_scene(original,source)
    called=[];fake_worker(monkeypatch,called)
    result=refine_scene_from_run(p,SOURCE,p/'runs'/TARGET,{})
    scene=json.loads(Path(result['scene_path']).read_text())
    assert all('hidden' not in point and 'reconstruction_weight' not in point for point in scene['gaussians'])
    info=scene['metadata']['semantic_preview_preservation']
    assert info['matched_verified_index_count']==0 and info['unmapped_original_point_count']==2
    assert info['unmapped_original_points_with_edit_state']==2 and not info['original_geometry_and_sampling_preserved']
    assert any('未猜配' in warning for warning in scene['metadata']['warnings'])


@pytest.mark.parametrize('bad_index',[0,-1,2,True,1.0,'1'])
def test_bad_existing_source_index_rejected_before_cuda_worker(tmp_path,monkeypatch,bad_index):
    p,ply=project(tmp_path);source=p/'runs'/SOURCE/'scene.json'
    scene=json.loads(source.read_text());scene['gaussians'][1]['source_index']=bad_index;write_scene(scene,source)
    called=[];fake_worker(monkeypatch,called)
    with pytest.raises(ValueError,match='source_index'):
        refine_scene_from_run(p,SOURCE,p/'runs'/TARGET,{})
    assert called==[]
    assert not (p/'runs'/TARGET).exists()


@pytest.mark.parametrize('bad_count',[3,2.,True,0])
def test_actual_ply_header_count_checked_before_cuda_worker(tmp_path,monkeypatch,bad_count):
    p,ply=project(tmp_path);source=p/'runs'/SOURCE/'scene.json'
    scene=json.loads(source.read_text());scene['metadata']['source_gaussian_count']=bad_count;write_scene(scene,source)
    called=[];fake_worker(monkeypatch,called)
    with pytest.raises(ValueError,match='点数'):
        refine_scene_from_run(p,SOURCE,p/'runs'/TARGET,{})
    assert called==[]


def test_instance_display_flags_do_not_hide_an_entire_new_category(tmp_path,monkeypatch):
    p,ply=project(tmp_path);source=p/'runs'/SOURCE/'scene.json';scene=json.loads(source.read_text())
    scene['objects']=[{'id':'one-bear','label':'bear','level':'object','visible':False,'deleted':True}]
    write_scene(scene,source)
    called=[];fake_worker(monkeypatch,called)
    result=refine_scene_from_run(p,SOURCE,p/'runs'/TARGET,{})
    output=json.loads(Path(result['scene_path']).read_text())
    bear=next(obj for obj in output['objects'] if obj['label']=='bear')
    assert bear['visible'] is True and not bear.get('deleted',False)
    assert output['metadata']['semantic_preview_preservation']['preserved_category_state_labels']==[]


def test_accidental_sampling_in_full_refresh_is_rejected_before_training(tmp_path,monkeypatch):
    p,ply=project(tmp_path);source=p/'runs'/SOURCE/'scene.json';scene=json.loads(source.read_text())
    scene['gaussians'][0]['hidden']=True  # Verified source index 0.
    scene['gaussians'][1].pop('source_index')
    write_scene(scene,source)
    called=[];fake_worker(monkeypatch,called)
    from backend import semantic_service
    original_reader=semantic_service.read_ply
    def sample_only_index_one(path,max_points):
        preview=original_reader(path,max_points)
        preview['gaussians']=preview['gaussians'][1:]
        preview['metadata'].update(preview_gaussian_count=1,preview_sampled=True)
        return preview
    monkeypatch.setattr(semantic_service,'read_ply',sample_only_index_one)
    with pytest.raises(ValueError,match='拒绝采样'):
        refine_scene_from_run(p,SOURCE,p/'runs'/TARGET,{})
    assert called==[]


@pytest.mark.parametrize('loss',['sampled','dc_only'])
def test_semantic_refresh_restores_full_model_and_sh_from_old_preview(tmp_path,monkeypatch,loss):
    p,ply=project(tmp_path);source=p/'runs'/SOURCE/'scene.json'
    old=json.loads(source.read_text());old['gaussians'][0]['hidden']=True
    if loss=='sampled':old['gaussians']=old['gaussians'][:1]
    else:
        for point in old['gaussians']:point.pop('sh');point.pop('sh_degree')
    write_scene(old,source)
    called=[];fake_worker(monkeypatch,called)
    result=refine_scene_from_run(p,SOURCE,p/'runs'/TARGET,{})
    output=json.loads(Path(result['scene_path']).read_text())
    assert [g['source_index'] for g in output['gaussians']]==[0,1]
    assert all(g['sh_degree']==3 and len(g['sh'])==48 for g in output['gaussians'])
    assert output['gaussians'][0]['hidden'] is True


def test_unverified_old_model_identity_does_not_migrate_category_flags(tmp_path,monkeypatch):
    p,ply=project(tmp_path);source=p/'runs'/SOURCE/'scene.json';scene=json.loads(source.read_text())
    scene['metadata'].pop('source_ply_sha256')
    scene['objects']=[{'id':'old','label':'bear','semantic_granularity':'category_union','visible':False,'priority':True}]
    write_scene(scene,source)
    called=[];fake_worker(monkeypatch,called)
    result=refine_scene_from_run(p,SOURCE,p/'runs'/TARGET,{})
    output=json.loads(Path(result['scene_path']).read_text());bear=next(o for o in output['objects'] if o['label']=='bear')
    assert bear['visible'] is True and bear['priority'] is False
    assert output['metadata']['semantic_preview_preservation']['preserved_category_state_labels']==[]
