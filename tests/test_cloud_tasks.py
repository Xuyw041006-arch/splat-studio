"""CPU tests of real task ZIPs, path/hash boundaries, and full result exchange."""
import importlib
import json
from pathlib import Path
import shutil
import stat
import subprocess
import sys
from types import SimpleNamespace
import zipfile

import numpy as np
from PIL import Image
import pytest
from fastapi.testclient import TestClient

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from backend import cloud_tasks as cloud
from scripts import cloud_worker as worker
api=importlib.import_module('backend.app')


@pytest.fixture
def env(tmp_path,monkeypatch):
    monkeypatch.setattr(api,'DATA',tmp_path)
    monkeypatch.setattr(api,'jobs',{})
    calls=[]
    monkeypatch.setattr(api,'pool',SimpleNamespace(submit=lambda *args:calls.append(args)))
    p=tmp_path/('a'*32);(p/'images').mkdir(parents=True)
    items=[]
    for i in range(2):
        image=Image.new('RGB',(16,16),(i*100,10,20));name=f'{i:04d}.jpg';path=p/'images'/name
        image.save(path)
        capture=api.photo_metadata(image,image,path.read_bytes())
        items.append({'name':name,'width':16,'height':16,'capture':capture,'url':'/local-only'})
    items[0]['calibrated_K']=[[20,0,8],[0,20,8],[0,0,1]]
    api.save_json(p/'project.json',{'id':p.name,'images':items,'created_at':1})
    (p/'masks').mkdir();Image.fromarray(np.ones((16,16),dtype=np.uint8)*255).save(p/'masks/cup.png')
    masks=[{'image_path':str(p/'images/0000.jpg'),'mask_path':str(p/'masks/cup.png'),'label':'cup',
            'annotation_source':'manual','confidence':1.,'level':'object','object_id':'cup','mask_id':'cup1'}]
    api.save_json(p/'masks.json',masks)
    cfg=api.Config(execution_target='cloud',device='mps',semantics=True,provider='manual',priority_objects=['cup']).model_dump()
    return SimpleNamespace(p=p,root=tmp_path,cfg=cfg,calls=calls,client=TestClient(api.app),monkeypatch=monkeypatch)


def package(env):
    path=env.root/'task.zip';manifest=cloud.build_package(env.p,env.cfg,path,ROOT)
    return path,manifest


def rewritten(path, change, name='changed.zip'):
    target=path.with_name(name)
    with zipfile.ZipFile(path) as source,zipfile.ZipFile(target,'w') as archive:
        entries={info.filename:source.read(info.filename) for info in source.infolist()}
        change(entries)
        for name,raw in entries.items():archive.writestr(name,raw)
    return target


def test_real_portable_roundtrip_preserves_calibration_masks_and_source(env):
    before=(env.p/'project.json').read_bytes(),(env.p/'masks.json').read_bytes()
    path,manifest=package(env)
    verified=cloud.verify_package(path,ROOT)
    assert verified==manifest
    assert verified['worker_config']['device']=='cuda'
    assert verified['requested_config']['device']=='mps'
    assert verified['worker_config']['execution_target']=='local'
    assert verified['worker_config']['viewer_quality']=='full'
    extracted=env.root/'unpacked';cloud.extract_verified(path,extracted,ROOT)
    assert (extracted/'project/images/0000.jpg').read_bytes()==(env.p/'images/0000.jpg').read_bytes()
    portable=json.loads((extracted/'project/project.json').read_text())
    assert portable['images'][0]['calibrated_K']==[[20,0,8],[0,20,8],[0,0,1]]
    assert 'url' not in portable['images'][0]
    raw_masks=(extracted/'project/masks.json').read_text()
    assert str(env.p) not in raw_masks
    remote=worker.materialize_project(extracted,env.root/'remote-data',manifest)
    masks=json.loads((remote/'masks.json').read_text())
    assert masks[0]['mask_path']==str(remote/'masks/cup.png')
    assert Path(masks[0]['mask_path']).read_bytes()==(env.p/'masks/cup.png').read_bytes()
    assert before==((env.p/'project.json').read_bytes(),(env.p/'masks.json').read_bytes())


def test_cpu_worker_cli_validates_source_without_cuda_or_training(env):
    path,_=package(env)
    result=subprocess.run([sys.executable,str(ROOT/'scripts/cloud_worker.py'),str(path),'--verify-only'],capture_output=True,text=True)
    assert result.returncode==0,result.stderr
    assert json.loads(result.stdout)['status']=='verified_not_started'
    assert not (env.root/'results').exists()


def test_cloud_api_exports_independent_job_without_local_queue(env):
    response=env.client.post(f'/api/projects/{env.p.name}/cloud-jobs',json=env.cfg)
    assert response.status_code==200,response.text
    job=response.json();assert job['status']=='awaiting_remote_execution'
    assert job['execution_status']=='not_started' and not env.calls and not api.jobs
    assert env.client.get('/api/cloud-jobs/'+job['id']).json()==job
    download=env.client.get(job['package_url']);assert download.status_code==200
    assert cloud.hashlib.sha256(download.content).hexdigest()==job['package_sha256']
    assert env.client.get('/api/jobs/'+job['id']).status_code==404
    response=env.client.post(f'/api/projects/{env.p.name}/jobs',json=env.cfg)
    assert response.json()['status']=='awaiting_remote_execution' and not env.calls
    assert env.client.post(f'/api/projects/{env.p.name}/cloud-jobs',json={'execution_target':'local'}).status_code==422


def test_cloud_analysis_skips_local_gpu_plan_and_teacher_preserves_masks(env):
    env.monkeypatch.setattr(api,'analyze_images',lambda *args:{'image_count':2,'unique_image_count':2,'connected_ratio':1.,'warnings':[]})
    env.monkeypatch.setattr(api,'make_plan',lambda *args:pytest.fail('local GPU plan called'))
    env.monkeypatch.setattr(api,'discover_objects',lambda *args:pytest.fail('local teacher called'))
    original=(env.p/'masks.json').read_bytes()
    response=env.client.post(f'/api/projects/{env.p.name}/analyze',json={**env.cfg,'manual_labels':['cup']})
    assert response.status_code==200,response.text
    plan=response.json()['plan']
    assert plan['can_export_cloud_task'] and not plan['can_reconstruct']
    assert plan['estimated_seconds'] is None and plan['recommended_backend']=='cloud_cuda_pending'
    assert response.json()['semantic_source']=='cloud_pending'
    assert response.json()['objects'][0]['label']=='cup' and not response.json()['objects'][0]['grounded']
    assert (env.p/'masks.json').read_bytes()==original


def test_cloud_semantic_only_refuses_without_fallback(env):
    response=env.client.post(f'/api/projects/{env.p.name}/semantic-jobs',json={'source_run_id':'b'*32,'config':env.cfg})
    assert response.status_code==422 and '不会回退' in response.json()['detail'] and not env.calls


@pytest.mark.parametrize('field,value',[('execution_target','remote'),('cloud_provider','ssh')])
def test_cloud_config_rejects_undefined_modes(env,field,value):
    assert env.client.post(f'/api/projects/{env.p.name}/cloud-jobs',json={**env.cfg,field:value}).status_code==422


@pytest.mark.parametrize('name',['../escape','/absolute','a/../../escape','a\\b','C:/a','a//b','a/./b'])
def test_zip_path_traversal_rejected(env,name):
    path,_=package(env)
    unsafe=rewritten(path,lambda items:items.update({name:b'bad'}))
    with pytest.raises(ValueError):cloud.extract_verified(unsafe,env.root/'unsafe')
    assert not (env.root/'unsafe').exists()


def test_unbound_extra_file_and_tampered_photo_rejected(env):
    path,_=package(env)
    extra=rewritten(path,lambda items:items.update({'extra.txt':b'no hash'}))
    with pytest.raises(ValueError,match='未绑定'):cloud.verify_package(extra)
    tampered=rewritten(path,lambda items:items.update({'project/images/0000.jpg':b'bad'}),'tampered.zip')
    with pytest.raises(ValueError,match='长度|SHA256'):cloud.verify_package(tampered)


def test_zip_symlink_duplicate_and_resource_limits(env):
    path,_=package(env)
    unsafe=env.root/'link.zip';shutil.copyfile(path,unsafe)
    with zipfile.ZipFile(unsafe,'a') as archive:
        info=zipfile.ZipInfo('link');info.create_system=3;info.external_attr=(stat.S_IFLNK|0o777)<<16
        archive.writestr(info,'/tmp/secret')
    with pytest.raises(ValueError,match='链接'):cloud.verify_package(unsafe)
    duplicate=env.root/'duplicate.zip';shutil.copyfile(path,duplicate)
    with zipfile.ZipFile(duplicate,'a') as archive:archive.writestr('README.TXT',b'duplicate case')
    with pytest.raises(ValueError,match='重复'):cloud.verify_package(duplicate)
    env.monkeypatch.setattr(cloud,'MAX_FILE_BYTES',10)
    with pytest.raises(ValueError,match='资源上限'):cloud.verify_package(path)


def test_local_symlink_and_external_mask_paths_refused(env):
    real=env.p/'masks/cup.png';saved=real.read_bytes();real.unlink();real.symlink_to(env.p/'images/0000.jpg')
    with pytest.raises(ValueError,match='符号链接'):package(env)
    assert not (env.root/'task.zip').exists()
    real.unlink();real.write_bytes(saved)
    masks=json.loads((env.p/'masks.json').read_text());masks[0]['mask_path']='/etc/passwd';api.save_json(env.p/'masks.json',masks)
    with pytest.raises(ValueError,match='外部路径'):package(env)


def test_worker_config_tampering_and_source_mismatch_refused(env):
    path,_=package(env)
    def tamper(items):
        value=json.loads(items['manifest.json']);value['worker_config']['device']='mps'
        items['manifest.json']=cloud.json_bytes(value)
    unsafe=rewritten(path,tamper)
    with pytest.raises(ValueError,match='CUDA'):cloud.verify_package(unsafe)
    fake=env.root/'different-runtime';(fake/'backend').mkdir(parents=True)
    for name,source in cloud.runtime_files(ROOT).items():
        target=fake/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(source,target)
    (fake/'backend/app.py').write_text('#different source')
    with pytest.raises(ValueError,match='版本不一致'):cloud.verify_package(path,fake)


def test_existing_work_is_never_overwritten_and_non_cuda_refused(env):
    path,_=package(env);target=env.root/'existing';target.mkdir();(target/'edits').write_text('keep')
    with pytest.raises(ValueError,match='覆盖'):cloud.extract_verified(path,target)
    assert (target/'edits').read_text()=='keep'
    import backend.planning
    env.monkeypatch.setattr(backend.planning,'capabilities',lambda:{'cuda':False,'device':'mps'})
    with pytest.raises(ValueError,match='禁止'):worker.require_cuda(env.cfg)


def test_remote_mask_validation_requires_real_priority_pixels(env):
    worker.prepare_semantics(api,env.p,env.cfg)
    bad={**env.cfg,'priority_objects':['missing']}
    with pytest.raises(ValueError,match='有效像素监督'):worker.prepare_semantics(api,env.p,bad)
    api.save_json(env.p/'masks.json',[])
    with pytest.raises(ValueError,match='没有真实像素掩码'):worker.prepare_semantics(api,env.p,env.cfg)


def test_strict_cuda_run_rejects_preview_before_trainer(env):
    jid='b'*32;api.jobs[jid]={'id':jid,'project_id':env.p.name,'status':'queued'}
    env.monkeypatch.setattr(api,'analyze_images',lambda *args:{})
    env.monkeypatch.setattr(api,'make_plan',lambda *args:{'sparse':False,'extreme':False,'recommended_backend':'torch_preview'})
    env.monkeypatch.setattr(api,'reconstruct',lambda *args:pytest.fail('preview executed'))
    api.run_job(jid,{**env.cfg,'execution_target':'local','strict_cuda':True,'device':'cuda'})
    assert api.jobs[jid]['status']=='failed' and '拒绝 CPU/MPS/预览回退' in api.jobs[jid]['error']


@pytest.mark.parametrize('missing',[False,True])
def test_priority_without_semantics_reaches_cuda_weighted_masks(env,missing):
    jid='b'*32;api.jobs[jid]={'id':jid,'project_id':env.p.name,'status':'queued'}
    env.monkeypatch.setattr(api,'analyze_images',lambda *args:{})
    env.monkeypatch.setattr(api,'make_plan',lambda *args:{'sparse':False,'extreme':False,'can_reconstruct':True,'recommended_backend':'original_3dgs'})
    captured=[]
    if missing:api.save_json(env.p/'masks.json',[])
    import backend.upstream
    def trainer(paths,out,cfg,*args):
        captured.append(cfg)
        assert cfg['priority_masks'] and Path(cfg['priority_masks_dir']).is_dir()
        assert (Path(cfg['priority_masks_dir'])/'0000.jpg.png').is_file()
        raise RuntimeError('weighted trainer reached; test stops before GPU')
    env.monkeypatch.setattr(backend.upstream,'reconstruct_upstream',trainer)
    api.run_job(jid,{**env.cfg,'semantics':False,'execution_target':'local','device':'cuda','strict_cuda':True})
    assert api.jobs[jid]['status']=='failed'
    if missing:assert not captured and '不能以普通重建冒充' in api.jobs[jid]['error']
    else:assert len(captured)==1 and 'weighted trainer reached' in api.jobs[jid]['error']


def test_full_result_jsonl_sh_and_semantics_roundtrip(env):
    from backend.gaussian_io import write_ply,write_scene,read_ply
    from backend.scene_transport import write_viewer_scene,read_scene_lines
    jid='b'*32;run=env.p/'runs'/jid;run.mkdir(parents=True)
    values=[{'position':[i*.1,0,2],'scale':[.1,.1,.1],'rotation':[1,0,0,0],'color':[.5,.3,.2],
             'opacity':.8,'sh_degree':3,'sh':[float(j+i)/100 for j in range(48)]} for i in range(5)]
    ply=run/'point_cloud.ply';write_ply({'gaussians':values},ply)
    scene=read_ply(ply);scene['metadata'].update(backend='original_3dgs',ply_path=str(ply))
    scene['objects']=[{'id':'cup','label':'cup'}];scene['gaussians'][0]['semantic_ids']=['cup']
    write_scene(scene,run/'scene.json');write_viewer_scene(scene,run/'scene.json',threshold=0,chunk_size=2)
    api.jobs[jid]={'id':jid,'project_id':env.p.name,'status':'completed'}
    manifest={'id':jid,'project_id':env.p.name,'requested_config':env.cfg}
    result=worker.export_results(api,env.p,jid,env.root/'results',manifest,'f'*64,{'device':'cuda'})
    decoded=read_scene_lines(env.root/'results/scene.splat.jsonl')
    assert decoded['gaussians']==scene['gaussians']
    assert result['gaussian_count']==5 and result['complete_gaussians']
    assert (env.root/'results/point_cloud.ply').read_bytes()==ply.read_bytes()
    response=env.client.post('/api/import',files={'file':('scene.splat.jsonl',(env.root/'results/scene.splat.jsonl').read_bytes())})
    assert response.status_code==200,response.text
    restored=env.client.get(response.json()['scene_url']).json()
    assert restored['gaussians']==scene['gaussians'] and restored['objects']==scene['objects']


@pytest.mark.parametrize('fault',[None,'foreign_model','escape','conflicting_path','wrong_count'])
def test_all_four_semantic_sidecars_export_with_model_binding(env,fault):
    from backend.gaussian_io import write_ply,write_scene,read_ply
    jid='b'*32;run=env.p/'runs'/jid;run.mkdir(parents=True)
    ply=run/'point_cloud.ply';write_ply({'gaussians':[{'position':[0,0,1],'scale':[.1]*3,'rotation':[1,0,0,0],
        'color':[.5]*3,'opacity':.8,'sh_degree':3,'sh':[.1]*48}]},ply)
    scene=read_ply(ply);digest=cloud.sha256(ply)
    folder=run/'semantic';folder.mkdir()
    artifacts={};common={'source_ply_sha256':np.asarray(digest),'classes':np.asarray(['cup'])}
    for key,name in [('probabilities','probabilities.npz'),('closed_probabilities','closed_probabilities.npz'),
                     ('membership','semantic_membership.npz'),('closed_membership','closed_semantic_membership.npz')]:
        data={**common}
        if fault=='foreign_model' and key=='closed_probabilities':data['source_ply_sha256']=np.asarray('0'*64)
        if 'probabilities' in key:data['probabilities']=np.asarray([[.75]],dtype=np.float32)
        else:data['class_0']=np.asarray([True])
        np.savez_compressed(folder/name,**data)
        artifacts[key]={'path':name,'sha256':cloud.sha256(folder/name)}
    catalog={'status':'completed','source_ply_sha256':digest,'gaussian_count':1,'classes':['cup'],'artifacts':artifacts}
    if fault=='escape':catalog['artifacts']['closed_probabilities']['path']='../outside.npz'
    if fault=='wrong_count':catalog['gaussian_count']=2
    api.save_json(folder/'catalog.json',catalog);api.save_json(folder/'stats.json',{'completed_steps':1200})
    # Match semantic_service's existing five paths: closed probabilities absent.
    meta_artifacts={'catalog_path':str(folder/'catalog.json'),'stats_path':str(folder/'stats.json')}
    meta_artifacts.update({key+'_path':str(folder/artifacts[key]['path']) for key in ('probabilities','membership','closed_membership')})
    if fault=='conflicting_path':
        other=run/'other';other.mkdir();shutil.copyfile(folder/'probabilities.npz',other/'probabilities.npz')
        meta_artifacts['probabilities_path']=str(other/'probabilities.npz')
    scene['metadata'].update(backend='original_3dgs',ply_path=str(ply),semantic_refinement_artifacts=meta_artifacts)
    write_scene(scene,run/'scene.json');api.jobs[jid]={'status':'completed'}
    args=(api,env.p,jid,env.root/'results',{'id':jid,'project_id':env.p.name,'requested_config':env.cfg},'f'*64,{'device':'cuda'})
    if fault:
        with pytest.raises(ValueError):worker.export_results(*args)
    else:
        evidence=worker.export_results(*args)
        names={record['path'] for record in evidence['files']}
        assert 'semantic/closed_probabilities.npz' in names
        for record in artifacts.values():
            path=env.root/'results/semantic'/record['path']
            assert cloud.sha256(path)==record['sha256']
