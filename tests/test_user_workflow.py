import importlib
import io
import json
from types import SimpleNamespace
import zipfile

import pytest
from PIL import Image
from fastapi.testclient import TestClient
from backend.user_workflow import scene_name, estimate_training

api=importlib.import_module('backend.app')

@pytest.mark.parametrize('value',['','../a','a/b','a:b','CON','Lpt1.txt','x\x00y','name.','x'*65])
def test_invalid_cross_platform_names(value):
    with pytest.raises(ValueError):scene_name(value)

def test_cjk_name_preserved():
    assert scene_name('  我的茶桌  ')=='我的茶桌'

def test_estimates_use_compute_device_and_change_with_quality():
    analysis={'image_count':6,'unique_image_count':6,'connected_ratio':1,'sampled_images':[{'width':988,'height':730}]}
    cfg={'execution_target':'cloud','cloud_gpu':'a100'}
    values=[estimate_training(analysis,{**cfg,'mode':mode}) for mode in ['fast','balanced','fine']]
    assert values[0]['seconds'][0]<values[1]['seconds'][0]<values[2]['seconds'][0]
    assert all(v['device_verified'] is False for v in values)
    assert estimate_training(analysis,{'execution_target':'local'},{'original_3dgs':False})['available'] is False
    uncalibrated=estimate_training(analysis,{**cfg,'cloud_gpu':'other'})
    assert '通用' in uncalibrated['basis']
    supplied=estimate_training(analysis,{**cfg,'cloud_gpu':'other','measured_steps_per_second':100})
    assert '你提供' in supplied['basis']
    for bad in [float('nan'),float('inf'),0,-3]:
        with pytest.raises(ValueError):estimate_training(analysis,{**cfg,'measured_steps_per_second':bad})

@pytest.fixture
def env(tmp_path,monkeypatch):
    monkeypatch.setattr(api,'DATA',tmp_path)
    monkeypatch.setattr(api,'jobs',{})
    calls=[]
    monkeypatch.setattr(api,'pool',SimpleNamespace(submit=lambda *a:calls.append(a)))
    monkeypatch.setattr(api,'capabilities',lambda:{'device':'mps','original_3dgs':False,'cuda':False})
    client=TestClient(api.app)
    data=io.BytesIO();Image.new('RGB',(24,24),'blue').save(data,format='PNG')
    response=client.post('/api/projects',files=[('files',('one.png',data.getvalue(),'image/png')),('files',('two.png',data.getvalue(),'image/png'))],data={'scene_title':'我的茶桌'})
    assert response.status_code==200
    pid=response.json()['id'];p=tmp_path/pid
    _,fingerprint=api.project_geometry(p,api.Config().model_dump())
    analysis={'image_count':2,'unique_image_count':2,'connected_ratio':1,'warnings':[],'sampled_images':[{'width':24,'height':24}],'capture_fingerprint':fingerprint}
    api.save_json(p/'analysis.json',{'analysis':analysis})
    return SimpleNamespace(client=client,p=p,pid=pid,calls=calls)

def test_rename_and_cloud_snapshot_retain_name(env):
    res=env.client.put(f'/api/projects/{env.pid}/name',json={'name':'工作室 茶桌'})
    assert res.status_code==200
    response=env.client.post(f'/api/projects/{env.pid}/cloud-jobs',json={'execution_target':'cloud','scene_name':'工作室 茶桌','semantics':False,'device':'cuda','full_quality':True})
    assert response.status_code==200,response.text
    file=env.client.get(response.json()['package_url'])
    with zipfile.ZipFile(io.BytesIO(file.content)) as package:
        manifest=json.loads(package.read('manifest.json'))
        assert manifest['requested_config']['scene_name']=='工作室 茶桌'
        assert manifest['worker_config']['scene_name']=='工作室 茶桌'
        assert json.loads(package.read('project/project.json'))['name']=='工作室 茶桌'
    assert not env.calls

def test_plan_does_not_run_detection_or_training(env,monkeypatch):
    def forbidden(*a,**k):raise AssertionError('Planning must not start recognition')
    monkeypatch.setattr(api,'discover_objects',forbidden)
    res=env.client.post(f'/api/projects/{env.pid}/plan',json={'execution_target':'cloud','scene_name':'我的茶桌','semantics':True,'cloud_gpu':'a100'})
    assert res.status_code==200,res.text
    assert res.json()['estimate']['available'] is True
    assert not env.calls
    manifest=json.loads((env.p/'project.json').read_text());manifest['images'][0]['calibrated_K']=[[30,0,12],[0,30,12],[0,0,1]];api.save_json(env.p/'project.json',manifest)
    assert env.client.post(f'/api/projects/{env.pid}/plan',json={'execution_target':'cloud'}).status_code==422

def test_full_quality_blocks_mps_fallback(env):
    res=env.client.post(f'/api/projects/{env.pid}/jobs',json={'full_quality':True,'device':'cuda'})
    assert res.status_code==422
    assert not env.calls
