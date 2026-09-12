import importlib
import io
import json
import os
from pathlib import Path
import sys
import tempfile
import time
import numpy as np
from PIL import Image
from fastapi.testclient import TestClient

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
os.environ['SPLAT_DATA_DIR']=tempfile.mkdtemp(prefix='splat-api-test-')
api_module=importlib.import_module('backend.app')
client=TestClient(api_module.app)

def images():
    from scripts.smoke_reconstruction import make_photogrammetry_fixture
    paths=make_photogrammetry_fixture(Path(tempfile.mkdtemp()))
    return [('files',(Path(p).name,Path(p).read_bytes(),'image/png')) for p in paths]

def test_upload_rejects_bad_images_and_origin():
    response=client.post('/api/projects',files=[('files',('a.txt',b'abc')),('files',('b.txt',b'abc'))],headers={'Origin':'https://evil.example'})
    assert response.status_code==403
    assert client.post('/api/projects',files=[images()[0]]).status_code==422

def test_photos_to_real_training_and_scene_export():
    response=client.post('/api/projects',files=images());assert response.status_code==200,response.text
    pid=response.json()['id'];config={'mode':'fast','device':'cpu','semantics':False}
    result=client.post(f'/api/projects/{pid}/analyze',json=config)
    assert result.status_code==200,result.text
    assert result.json()['analysis']['max_pair_inliers']>=16
    assert result.json()['plan']['strategy']=='two_view_completion'
    task=client.post(f'/api/projects/{pid}/jobs',json=config).json()
    deadline=time.monotonic()+30
    while time.monotonic()<deadline:
        job=client.get('/api/jobs/'+task['id']).json()
        if job['status'] in ('completed','failed','cancelled'):break
        time.sleep(.03)
    assert job['status']=='completed',job
    scene=client.get(job['scene_url']).json()
    assert len(scene['gaussians'])>10
    assert scene['metadata']['semantics_requested'] is False
    assert scene['metadata']['final_l1']<scene['metadata']['initial_l1']
    assert client.get(scene['metadata']['ply_url']).content.startswith(b'ply')

def test_manual_masks_persist_after_reanalysis_and_commands_validate():
    pid=client.post('/api/projects',files=images()).json()['id']
    body={'image_name':'0000.jpg','label':'面板','mask':[[1,0],[0,1]],'object_id':'panels'}
    assert client.post(f'/api/projects/{pid}/masks',json=body).status_code==200
    result=client.post(f'/api/projects/{pid}/analyze',json={'semantics':True,'provider':'manual','manual_labels':['面板'],'device':'cpu'})
    assert result.status_code==200,result.text
    masks=json.loads((api_module.DATA/pid/'masks.json').read_text());assert len(masks)==1
    objects=[{'id':'chair','label':'椅子','visible':True}]
    action=client.post('/api/commands',json={'text':'隐藏椅子','objects':objects})
    assert action.status_code==200,action.text
    assert action.json()['target_ids']==['chair']
    assert action.json()['action']=='hide'
    assert client.post(f'/api/projects/{pid}/masks',json={**body,'image_name':'../../secret'}).status_code==422

def test_duplicate_photos_not_usable_geometry_and_invalid_scene():
    pair=images();pid=client.post('/api/projects',files=[pair[0],pair[0]]).json()['id']
    result=client.post(f'/api/projects/{pid}/analyze',json={'device':'cpu'}).json()
    assert result['plan']['can_reconstruct'] is False
    result=client.post('/api/import',files={'file':('bad.json',json.dumps({'gaussians':[{'position':[float('nan'),0,0]}]}),'application/json')})
    assert result.status_code==422
