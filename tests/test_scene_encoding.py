"""Complete-SH scene export preserves exact values and atomic publication."""
import json
import math
import pytest
from backend.gaussian_io import write_scene


def test_complete_sh_and_semantics_survive_multiple_batches(tmp_path):
    point={'source_index':0,'position':[.125,-0.,1e-20],
           'sh':[math.sin(i)/7 for i in range(48)],'sh_degree':3,
           'semantic_ids':['物品：茶杯','semantic-class-abc'],
           'semantic_probabilities':{'semantic-class-abc':.5000000000001}}
    scene={'cameras':[],'gaussians':[{**point,'source_index':i} for i in range(25003)],
           'metadata':{'name':'茶具 / Teatime','geometry_modified':False},'objects':[]}
    path=tmp_path/'scene.json';write_scene(scene,path)
    expected=json.dumps(scene,ensure_ascii=False,allow_nan=False,separators=(',', ':'))
    assert path.read_text()==expected
    restored=json.loads(path.read_text())
    assert restored['gaussians'][-1]['source_index']==25002
    assert restored['gaussians'][-1]['sh']==point['sh']
    assert restored['gaussians'][-1]['semantic_probabilities']==point['semantic_probabilities']


@pytest.mark.parametrize('bad_value',[float('nan'),float('inf'),float('-inf')])
def test_late_invalid_geometry_preserves_existing_scene(tmp_path,bad_value):
    path=tmp_path/'scene.json';path.write_text('{"valid":"previous result"}')
    scene={'gaussians':[{'position':[0,0,1]}]*25000+[{'position':[bad_value,0,1]}]}
    with pytest.raises(ValueError):write_scene(scene,path)
    assert path.read_text()=='{"valid":"previous result"}'
    assert list(tmp_path.iterdir())==[path]


@pytest.mark.parametrize('scene',[{}, {'gaussians':[],'metadata':{'name':'空场景'}},
                                   {'metadata':{'note':'before'},'gaussians':[{'sh':[0.]*48}]}])
def test_empty_and_small_scene_format_remains_compatible(tmp_path,scene):
    path=tmp_path/'scene.json';write_scene(scene,path)
    assert path.read_text()==json.dumps(scene,ensure_ascii=False,allow_nan=False,separators=(',', ':'))
