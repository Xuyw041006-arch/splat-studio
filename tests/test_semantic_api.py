"""API orchestration checks use a stub semantic service, never allocate a GPU."""
import importlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace
from PIL import Image

from fastapi.testclient import TestClient
import pytest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
api_module=importlib.import_module('backend.app')


class DeferredPool:
    def __init__(self): self.calls=[]
    def submit(self,worker,*args): self.calls.append((worker,args))
    def run_next(self):
        worker,args=self.calls.pop(0)
        worker(*args)


@pytest.fixture
def env(tmp_path,monkeypatch):
    pool=DeferredPool()
    monkeypatch.setattr(api_module,'DATA',tmp_path)
    monkeypatch.setattr(api_module,'jobs',{})
    monkeypatch.setattr(api_module,'pool',pool)
    monkeypatch.setattr(api_module,'capabilities',lambda:{'cuda_semantic_refinement':True})
    pid='a'*32;rid='b'*32;p=tmp_path/pid;source=p/'runs'/rid
    (p/'images').mkdir(parents=True);source.mkdir(parents=True)
    image=Image.new('RGB',(16,16));image.save(p/'images'/'0000.jpg')
    capture=api_module.photo_metadata(image,image,(p/'images'/'0000.jpg').read_bytes())
    api_module.save_json(p/'project.json',{'id':pid,'images':[{'name':'0000.jpg','width':16,'height':16,'capture':capture}]})
    api_module.save_json(p/'masks.json',[{'image_name':'0000.jpg','label':'cup','mask':[[1]]}])
    ply=source/'model'/'point_cloud'/'iteration_15000'/'point_cloud.ply'
    ply.parent.mkdir(parents=True);ply.write_bytes(b'ply\nfull original model\n')
    scene={'gaussians':[{'position':[0,0,0],'scale':[1,1,1],'rotation':[1,0,0,0],'color':[1,0,0],'opacity':.5}],
           'objects':[],'cameras':[{'image_name':'0000.jpg'}],
           'metadata':{'backend':'original_3dgs','ply_path':str(ply),'project_id':pid,'job_id':rid}}
    api_module.save_json(source/'scene.json',scene)
    api_module.save_json(p/f'job-{rid}.json',{'id':rid,'project_id':pid,'status':'completed'})
    calls=[]
    def service(project,source_run_id,out,cfg,progress,cancelled):
        calls.append({'project':project,'source_run_id':source_run_id,'out':out,'cfg':cfg,'cancelled':cancelled})
        assert not cancelled()
        progress(.5,'semantic checkpoint 400')
        # Returning a nested scene verifies canonical publishing to out/scene.json.
        target=Path(out)/'semantic'/'refined.json';target.parent.mkdir(parents=True,exist_ok=True)
        refined=json.loads((Path(project)/'runs'/source_run_id/'scene.json').read_text())
        refined['objects']=[{'id':'cup','label':'cup'}]
        api_module.save_json(target,refined)
        return {'scene_path':str(target),'ply_path':refined['metadata']['ply_path']}
    module=SimpleNamespace(refine_scene_from_run=service)
    monkeypatch.setitem(sys.modules,'backend.semantic_service',module)
    return SimpleNamespace(client=TestClient(api_module.app),p=p,pid=pid,rid=rid,source=source,ply=ply,
                           pool=pool,calls=calls,service=module,monkeypatch=monkeypatch)


def submit(env,**config):
    return env.client.post(f'/api/projects/{env.pid}/semantic-jobs',json={'source_run_id':env.rid,'config':config})


def test_semantic_only_reuses_original_ply_and_shared_queue(env):
    original_ply=env.ply.read_bytes();original_scene=(env.source/'scene.json').read_bytes()
    env.monkeypatch.setattr(api_module,'reconstruct',lambda *args,**kwargs:pytest.fail('RGB training must not run'))
    response=submit(env,semantic_steps=2400)
    assert response.status_code==200,response.text
    job=response.json();jid=job['id']
    assert jid!=env.rid and job['kind']=='semantic_refinement' and job['status']=='queued'
    assert len(env.pool.calls)==1 and env.pool.calls[0][0] is api_module.run_semantic_job
    assert submit(env).status_code==409
    assert env.client.post(f'/api/projects/{env.pid}/jobs',json={}).status_code==409
    env.pool.run_next()
    finished=env.client.get('/api/jobs/'+jid).json()
    assert finished['status']=='completed',finished
    scene=env.client.get(finished['scene_url']).json();meta=scene['metadata']
    assert finished['scene_url'].endswith(f'/runs/{jid}/scene.json')
    assert meta['source_run_id']==env.rid and meta['job_id']==jid
    assert meta['rgb_retrained'] is False and meta['semantic_refinement_applied']=='cuda_iterative'
    assert meta['ply_path']==str(env.ply)
    assert env.client.get(meta['ply_url']).content==original_ply
    assert env.ply.read_bytes()==original_ply and (env.source/'scene.json').read_bytes()==original_scene
    assert env.calls[0]['cfg']['semantic_steps']==2400
    assert env.calls[0]['cfg']['device']=='cuda'
    assert env.calls[0]['cfg']['semantic_strategy']=='posthoc'


@pytest.mark.parametrize('config',[{'device':'mps'},{'device':'cpu'},{'semantic_strategy':'joint'},
                                  {'semantic_steps':30000},{'semantic_refinement':'unknown'}])
def test_unsupported_configuration_is_rejected_before_queue(env,config):
    assert submit(env,**config).status_code==422
    assert not env.pool.calls


def test_cuda_capability_and_defaults(env):
    defaults=api_module.Config()
    assert defaults.semantic_refinement=='cuda_iterative' and defaults.semantic_steps==1200
    env.monkeypatch.setattr(api_module,'capabilities',lambda:{'cuda_semantic_refinement':False})
    response=submit(env)
    assert response.status_code==422 and 'NVIDIA CUDA' in response.text
    assert not env.pool.calls


@pytest.mark.parametrize('bad_id',['../scene','B'*32,'b'*31,'b'*33,'/tmp/model.ply'])
def test_source_id_cannot_be_a_path(env,bad_id):
    response=env.client.post(f'/api/projects/{env.pid}/semantic-jobs',json={'source_run_id':bad_id})
    assert response.status_code==422 and not env.pool.calls


def test_external_paths_and_other_project_runs_are_rejected(env,tmp_path):
    endpoint=f'/api/projects/{env.pid}/semantic-jobs'
    assert env.client.post(endpoint,json={'source_run_id':'c'*32}).status_code==404
    assert env.client.post(endpoint,json={'source_run_id':env.rid,'ply_path':'/tmp/model.ply'}).status_code==422
    external=tmp_path/'foreign.ply';external.write_bytes(b'ply')
    scene_path=env.source/'scene.json';scene=json.loads(scene_path.read_text())
    scene['metadata']['ply_path']=str(external);api_module.save_json(scene_path,scene)
    assert submit(env).status_code==422 and not env.pool.calls


@pytest.mark.parametrize('target',['run','ply','scene','project'])
def test_symlink_aliases_are_rejected(env,tmp_path,target):
    path={'run':env.source,'ply':env.ply,'scene':env.source/'scene.json','project':env.p}[target]
    moved=tmp_path/('moved-'+target)
    path.rename(moved);path.symlink_to(moved,target_is_directory=moved.is_dir())
    assert submit(env).status_code==422 and not env.pool.calls


@pytest.mark.parametrize('missing',['images','cameras','masks','completed','cuda_backend'])
def test_imported_or_incomplete_sources_cannot_be_refined(env,missing):
    if missing=='images': api_module.save_json(env.p/'project.json',{'id':env.pid,'images':[]})
    elif missing=='masks': (env.p/'masks.json').unlink()
    elif missing=='completed': api_module.save_json(env.p/f'job-{env.rid}.json',{'id':env.rid,'project_id':env.pid,'status':'running'})
    else:
        path=env.source/'scene.json';scene=json.loads(path.read_text())
        if missing=='cameras':scene['cameras']=[]
        else:scene['metadata']['backend']='torch_preview_mps'
        api_module.save_json(path,scene)
    assert submit(env).status_code==422 and not env.pool.calls


def test_service_failure_is_persisted_without_publishing_success(env):
    def fail(*args): raise ValueError('mask and undistorted camera coordinates do not match')
    env.service.refine_scene_from_run=fail
    jid=submit(env).json()['id'];env.pool.run_next()
    job=env.client.get('/api/jobs/'+jid).json()
    assert job['status']=='failed' and 'coordinates' in job['error']
    assert 'scene_url' not in job
    assert json.loads((env.p/f'job-{jid}.json').read_text())['status']=='failed'


@pytest.mark.parametrize('when',['queued','running'])
def test_cancellation_reaches_service_and_never_publishes_success(env,when):
    jid=submit(env).json()['id']
    if when=='queued':
        assert env.client.post('/api/jobs/'+jid+'/cancel').status_code==200
    else:
        def cancel(project,source_run_id,out,cfg,progress,cancelled):
            api_module.update_job(jid,cancel_requested=True)
            assert cancelled()
            raise RuntimeError('任务已取消')
        env.service.refine_scene_from_run=cancel
    env.pool.run_next();job=env.client.get('/api/jobs/'+jid).json()
    assert job['status']=='cancelled' and 'scene_url' not in job
    assert not env.calls


def test_service_result_cannot_reference_foreign_artifact(env,tmp_path):
    original_service=env.service.refine_scene_from_run
    foreign=tmp_path/'foreign.ply';foreign.write_bytes(b'ply')
    def outside(*args):return {**original_service(*args),'ply_path':str(foreign)}
    env.service.refine_scene_from_run=outside
    jid=submit(env).json()['id'];env.pool.run_next()
    assert env.client.get('/api/jobs/'+jid).json()['status']=='failed'


def test_output_directory_substitution_fails_job_before_service(env,tmp_path):
    jid=submit(env).json()['id']
    foreign=tmp_path/'foreign-output';foreign.mkdir()
    (env.p/'runs'/jid).symlink_to(foreign,target_is_directory=True)
    env.pool.run_next();job=env.client.get('/api/jobs/'+jid).json()
    assert job['status']=='failed' and not env.calls and not list(foreign.iterdir())


@pytest.mark.parametrize('backend,requested,applied',[('original_3dgs','cuda_iterative','cuda_iterative'),
                                                   ('original_3dgs','projection','cuda_iterative'),
                                                   ('torch_preview','cuda_iterative',None)])
def test_new_rgb_job_requires_full_cuda_semantics_without_flat_fallback(env,backend,requested,applied):
    env.monkeypatch.setattr(api_module,'analyze_images',lambda paths,cfg=None:{})
    env.monkeypatch.setattr(api_module,'make_plan',lambda analysis,cfg:{'sparse':False,'extreme':False,'can_reconstruct':True,'recommended_backend':backend})
    def reconstruct(paths,out,cfg,progress,cancelled):
        out=Path(out);out.mkdir(parents=True,exist_ok=True)
        ply=out/'full.ply';ply.write_bytes(b'ply\nnew RGB model')
        scene=json.loads((env.source/'scene.json').read_text())
        scene['metadata'].update(ply_path=str(ply),backend=backend)
        api_module.save_json(out/'scene.json',scene)
        return {'scene_path':str(out/'scene.json'),'ply_path':str(ply),'backend':backend}
    env.monkeypatch.setattr(api_module,'reconstruct',reconstruct)
    env.monkeypatch.setitem(sys.modules,'backend.upstream',SimpleNamespace(reconstruct_upstream=reconstruct))
    projection=[]
    env.monkeypatch.setattr(api_module,'fuse_semantics',lambda scene,*args:projection.append(True) or scene)
    response=env.client.post(f'/api/projects/{env.pid}/jobs',json={'semantics':True,'semantic_refinement':requested})
    jid=response.json()['id'];env.pool.run_next();job=env.client.get('/api/jobs/'+jid).json()
    if applied is None:
        assert job['status']=='failed' and 'CUDA' in job['error']
        assert not env.calls and not projection
        return
    assert job['status']=='completed',job
    meta=env.client.get(job['scene_url']).json()['metadata']
    assert meta['semantic_refinement_requested']=='cuda_iterative' and meta['semantic_refinement_applied']==applied
    if applied=='cuda_iterative':
        assert len(env.calls)==1 and env.calls[0]['source_run_id']==jid and not projection
    else:assert projection and not env.calls
    assert not meta['semantic_refinement_fallback_reason']


def test_planning_declares_mps_projection_fallback_without_changing_rgb_iterations():
    from backend.planning import make_plan
    analysis={'image_count':24,'connected_ratio':1,'max_pair_inliers':40,'warnings':[]}
    hardware={'device':'mps','mps':True,'cuda':False,'original_3dgs':False,'colmap':False,'cuda_semantic_refinement':False}
    plan=make_plan(analysis,{'mode':'fine','device':'mps','semantics':True,'semantic_refinement':'cuda_iterative'},hardware)
    assert plan['planned_semantic_method']=='projection'
    assert plan['settings']['iterations']==22000
    assert any('多轮语义' in warning and '投影' in warning for warning in plan['warnings'])


def test_capability_probe_requires_cuda_and_dependencies_without_training(tmp_path,monkeypatch):
    from backend import planning
    import torch
    repository=tmp_path/'upstream';(repository/'scene').mkdir(parents=True)
    (repository/'scene'/'gaussian_model.py').write_text('raise RuntimeError("capability must not import trainer")')
    (repository/'train.py').write_text('raise RuntimeError("capability must not execute trainer")')
    monkeypatch.setenv('SPLAT_3DGS_REPO',str(repository))
    monkeypatch.setattr(torch.cuda,'is_available',lambda:True)
    monkeypatch.setattr(torch.backends.mps,'is_available',lambda:False)
    monkeypatch.setattr(torch.cuda,'init',lambda:pytest.fail('probe must not start CUDA training'))
    monkeypatch.setattr(planning.importlib.util,'find_spec',lambda module:object())
    assert planning.capabilities()['cuda_semantic_refinement'] is True
    monkeypatch.setattr(planning.importlib.util,'find_spec',lambda module:None)
    assert planning.capabilities()['cuda_semantic_refinement'] is False
    monkeypatch.setattr(torch.cuda,'is_available',lambda:False)
    monkeypatch.setattr(planning.importlib.util,'find_spec',lambda module:object())
    assert planning.capabilities()['cuda_semantic_refinement'] is False


def manual_parent_masks(env,**child_updates):
    endpoint=f'/api/projects/{env.pid}/masks'
    parent={'image_name':'0000.jpg','label':'Coffee Mug','object_id':'mug-1','mask':[[1,1],[1,1]]}
    child={'image_name':'0000.jpg','label':'Coffee','object_id':'coffee-1','parent_id':'mug-1',
           'parent_relation':'contents_of','level':'part','mask':[[1,0],[0,0]],**child_updates}
    assert env.client.post(endpoint,json=parent).status_code==200
    response=env.client.post(endpoint,json=child)
    assert response.status_code==200,response.text
    return {'parent':'coffee mug','child':'coffee','relation':'contents_of','source':'user_manual_parent_id'}


def test_manual_parent_id_is_passed_as_explicit_label_approval(env):
    expected=manual_parent_masks(env)
    stored=json.loads((env.p/'masks.json').read_text())
    assert stored[-1]['annotation_source']=='manual' and stored[-1]['parent_relation']=='contents_of'
    # Repeated observations collapse to one class relation; no auto nesting is approved.
    stored.append(dict(stored[-1]))
    stored.append({'label':'teapot','object_id':'teapot-1','parent_id':'mug-1',
                   'annotation_source':'grounded_sam','mask':[[1]]})
    api_module.save_json(env.p/'masks.json',stored)
    response=submit(env);assert response.status_code==200,response.text
    assert env.pool.calls[0][1][1]['approved_hierarchy']==[expected]
    env.pool.run_next();assert env.calls[0]['cfg']['approved_hierarchy']==[expected]
    assert api_module.jobs[response.json()['id']]['status']=='completed'


def test_explicit_api_hierarchy_is_typed_and_normalized(env):
    masks=json.loads((env.p/'masks.json').read_text());masks.append({'label':'Coffee','mask':[[1]]})
    api_module.save_json(env.p/'masks.json',masks)
    edge={'parent':'  CUP ','child':'ＣＯＦＦＥＥ','relation':'contents_of','source':'user_explicit'}
    response=submit(env,approved_hierarchy=[edge]);assert response.status_code==200,response.text
    env.pool.run_next()
    assert env.calls[0]['cfg']['approved_hierarchy']==[{'parent':'cup','child':'coffee','relation':'contents_of','source':'user_explicit'}]


@pytest.mark.parametrize('edge',[
    {'parent':'cup','child':'cup','relation':'part_of'},
    {'parent':' ','child':'coffee','relation':'part_of'},
    {'parent':'cup','child':'coffee','relation':'invented_relation'},
    {'parent':'cup','child':'coffee','relation':'part_of','source':'automatic_containment'},
    {'parent':'cup','child':'sheep','relation':'part_of'},
    {'parent':'cup','child':'coffee','relation':'part_of','source':'user_manual_parent_id'},
    {'parent':'cup','child':'coffee','relation':'part_of','mask_path':'/tmp/mask.png'},
])
def test_invalid_unobserved_or_unapproved_hierarchy_does_not_queue(env,edge):
    assert submit(env,approved_hierarchy=[edge]).status_code==422
    assert not env.pool.calls


def test_explicit_hierarchy_cycles_are_rejected_before_queue(env):
    edges=[{'parent':'cup','child':'coffee','relation':'part_of'},
           {'parent':'coffee','child':'cup','relation':'part_of'}]
    assert submit(env,approved_hierarchy=edges).status_code==422 and not env.pool.calls


@pytest.mark.parametrize('issue',['missing_parent','ambiguous_parent','cycle'])
def test_manual_parent_reference_must_be_unambiguous_and_acyclic(env,issue):
    manual_parent_masks(env,parent_id='missing' if issue=='missing_parent' else 'mug-1')
    masks=json.loads((env.p/'masks.json').read_text())
    if issue=='ambiguous_parent':masks.append({'label':'teapot','object_id':'mug-1','mask':[[1]],'annotation_source':'manual'})
    elif issue=='cycle':masks[-2].update(parent_id='coffee-1',parent_relation='part_of')
    api_module.save_json(env.p/'masks.json',masks)
    assert submit(env).status_code==422 and not env.pool.calls


def test_legacy_manual_part_defaults_to_part_of_and_invalid_relation_is_not_saved(env):
    manual_parent_masks(env)
    masks=json.loads((env.p/'masks.json').read_text());masks[-1].pop('parent_relation')
    api_module.save_json(env.p/'masks.json',masks)
    approved=api_module.semantic_hierarchy_config(env.p,api_module.Config().model_dump())['approved_hierarchy']
    assert approved[0]['relation']=='part_of'
    before=(env.p/'masks.json').read_bytes()
    response=env.client.post(f'/api/projects/{env.pid}/masks',json={'image_name':'0000.jpg','parent_id':'mug-1','parent_relation':'bad','mask':[[1]]})
    assert response.status_code==422 and (env.p/'masks.json').read_bytes()==before


def test_cuda_reconstruction_receives_full_filename_priority_masks_and_manual_hierarchy(env):
    from PIL import Image
    import numpy as np
    expected=manual_parent_masks(env)
    env.monkeypatch.setattr(api_module,'analyze_images',lambda paths,cfg=None:{})
    env.monkeypatch.setattr(api_module,'make_plan',lambda analysis,cfg:{'sparse':False,'extreme':False,'can_reconstruct':True,'recommended_backend':'original_3dgs'})
    env.monkeypatch.setattr(api_module,'build_priority_masks',lambda paths,masks,ids:{paths[0]:np.array([[1,0],[0,1]],dtype=bool)})
    def reconstruct(paths,out,cfg,progress,cancelled):
        folder=Path(cfg['priority_masks_dir'])
        # Original COLMAP CameraInfo.image_name includes the image extension.
        lookup=folder/(Path(paths[0]).name+'.png')
        assert lookup.name=='0000.jpg.png' and lookup.is_file()
        assert not (folder/'0000.png').exists()
        assert np.array_equal(np.asarray(Image.open(lookup)),np.array([[255,0],[0,255]],dtype=np.uint8))
        assert cfg['approved_hierarchy']==[expected]
        out=Path(out);ply=out/'point_cloud.ply';ply.write_bytes(b'ply\nnew RGB model')
        scene=json.loads((env.source/'scene.json').read_text());scene['metadata']['ply_path']=str(ply)
        api_module.save_json(out/'scene.json',scene)
        return {'scene_path':str(out/'scene.json'),'ply_path':str(ply),'backend':'original_3dgs'}
    env.monkeypatch.setitem(sys.modules,'backend.upstream',SimpleNamespace(reconstruct_upstream=reconstruct))
    response=env.client.post(f'/api/projects/{env.pid}/jobs',json={'semantics':True,'semantic_refinement':'cuda_iterative','priority_objects':['coffee-1']})
    assert response.status_code==200,response.text
    jid=response.json()['id'];env.pool.run_next()
    assert api_module.jobs[jid]['status']=='completed',api_module.jobs[jid]
    assert env.calls[0]['cfg']['approved_hierarchy']==[expected]


def test_local_analysis_uses_strict_query_alignment_for_new_teacher_masks(env):
    seen=[]
    env.monkeypatch.setattr(api_module,'analyze_images',lambda paths,cfg=None:{})
    env.monkeypatch.setattr(api_module,'make_plan',lambda analysis,cfg:{'warnings':[]})
    def discover(paths,cfg):
        seen.append(cfg)
        return {'objects':[],'masks':[],'source':'local','available':True}
    env.monkeypatch.setattr(api_module,'discover_objects',discover)
    response=env.client.post(f'/api/projects/{env.pid}/analyze',json={
        'semantics':True,'provider':'local','candidate_labels':['coffee mug','coffee'],'strict_query_labels':False})
    assert response.status_code==200,response.text
    assert seen[0]['strict_query_labels'] is True
    assert seen[0]['candidate_labels']==['coffee mug','coffee']
