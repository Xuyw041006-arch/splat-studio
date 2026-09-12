from pathlib import Path
from types import SimpleNamespace
import importlib.util
import json
import os
import sys
import zipfile
import numpy as np
import pytest
from PIL import Image
from backend import planning, cloud_tasks
from backend.app import Config

SCRIPTS=Path(__file__).resolve().parents[1]/'scripts'
sys.path.insert(0,str(SCRIPTS))
import splat_env
from task_exchange import unpack_task

def test_windows_unicode_photos_bypass_narrow_opencv_io(tmp_path,monkeypatch):
    folder=tmp_path/'中文照片 with spaces';folder.mkdir();paths=[]
    image=np.random.default_rng(2).integers(0,256,(120,180,3),dtype=np.uint8)
    for i in range(2):
        path=folder/f'视角{i}.jpg';Image.fromarray(np.roll(image,i*5,axis=1)).save(path);paths.append(str(path))
    monkeypatch.setattr(planning,'os',SimpleNamespace(name='nt',environ=os.environ))
    monkeypatch.setattr(planning.cv2,'imread',lambda *a:pytest.fail('narrow-path imread called'))
    result=planning.analyze_images(paths)
    assert result['image_count']==2 and result['unique_image_count']==2

def task(tmp_path):
    project=tmp_path/('a'*32);(project/'images').mkdir(parents=True)
    rows=[]
    for i in range(2):
        name=f'{i:04d}.jpg';Image.new('RGB',(20,20),(i*80,10,20)).save(project/'images'/name)
        rows.append({'name':name,'width':20,'height':20})
    (project/'project.json').write_text(json.dumps({'id':project.name,'images':rows,'created_at':1}))
    (project/'masks.json').write_text('[]')
    cfg=Config(execution_target='cloud',cloud_provider='server',cloud_gpu='other',sparse_completion='none').model_dump()
    archive=tmp_path/'task.zip';cloud_tasks.build_package(project,cfg,archive,SCRIPTS.parent)
    return archive

def test_helper_prepares_real_task_verifies_source_and_does_not_train(tmp_path,monkeypatch):
    archive=task(tmp_path)
    monkeypatch.setattr(splat_env,'configure',lambda *a:{'python':sys.executable,'env':{}})
    base=tmp_path/'环境 with spaces'
    monkeypatch.setattr(sys,'argv',['splat_env.py','prepare','--task',str(archive),'--base',str(base)])
    splat_env.main()
    batch=next(base.glob('task-*'));job=json.loads((batch/'job.json').read_text())
    assert not Path(job['work']).exists()
    assert Path(job['task']).read_bytes()==archive.read_bytes()
    cloud_tasks.verify_package(job['task'],job['runtime'])

def test_helper_rejects_corrupt_task_before_execution(tmp_path):
    archive=task(tmp_path);broken=tmp_path/'broken.zip'
    with zipfile.ZipFile(archive) as z,zipfile.ZipFile(broken,'w') as out:
        for name in z.namelist():out.writestr(name,b'changed' if name=='project/masks.json' else z.read(name))
    destination=tmp_path/'unpacked'
    with pytest.raises(ValueError):unpack_task(broken,destination)
    assert not destination.exists()

def test_helper_rejects_path_escape_before_execution(tmp_path):
    archive=tmp_path/'bad.zip'
    with zipfile.ZipFile(archive,'w') as z:z.writestr('../escaped','bad')
    with pytest.raises(ValueError):unpack_task(archive,tmp_path/'unpacked')
    assert not (tmp_path/'escaped').exists()
