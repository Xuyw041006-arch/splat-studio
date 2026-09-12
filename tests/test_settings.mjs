import assert from 'node:assert/strict';
import test from 'node:test';
import {sparsePanelState,semanticProfilePreview,geometryOptionsForPhotoSet,sceneDisplayInfo} from '../frontend/settings.js';

test('sparse controls require analyzed photos and relevant sparse evidence',()=>{
  assert.equal(sparsePanelState(null,{sparse:true}).visible,false);
  assert.equal(sparsePanelState({},null).visible,false);
  assert.equal(sparsePanelState({image_count:2},{}).visible,false);
  assert.equal(sparsePanelState({image_count:24},{strategy:'standard_3dgs',sparse:false}).visible,false);
  for(const plan of [{sparse:true},{extreme:true},{completion_recommended:true},{strategy:'two_view_completion'},{strategy:'sparse_regularized'}]){
    assert.equal(sparsePanelState({},plan).visible,true);
  }
});

test('sparse recommendation displays analyzed reason without changing settings',()=>{
  const plan={extreme:true,reason:'仅有两个独立视角',geometry_backend_requested:'sfm'};
  const before=JSON.stringify(plan);
  assert.equal(sparsePanelState({},plan).reason,plan.reason);
  assert.match(sparsePanelState({},plan).title,/极少/);
  assert.equal(JSON.stringify(plan),before);
  assert.equal('allow_geometry_download' in sparsePanelState({},plan),false);
});

test('mode budgets distinguish sampled fast/balanced and all-view fine coverage',()=>{
  const fast=semanticProfilePreview('fast'),balanced=semanticProfilePreview('balanced'),fine=semanticProfilePreview('fine');
  assert.deepEqual([fast.steps,fast.views,fast.limit,fast.size,fast.sweeps,fast.granularity],[400,'sampled',12,192,1,'multilevel']);
  assert.deepEqual([balanced.steps,balanced.views,balanced.size,balanced.sweeps],[1200,'sampled',256,2]);
  assert.deepEqual([fine.steps,fine.views,fine.size,fine.sweeps,fine.granularity],[2400,'all',384,3,'multilevel']);
});

test('explicit all-view, granularity and manual step overrides survive mode choices',()=>{
  const settings=semanticProfilePreview('fast','manual',2400,'all','multilevel');
  assert.equal(settings.views,'all');assert.equal(settings.granularity,'multilevel');assert.equal(settings.steps,2400);
  assert.equal(settings.size,256);assert.equal(settings.sweeps,0);
  const balanced=semanticProfilePreview('balanced','mode',400,'sampled','multilevel');
  assert.equal(balanced.steps,1200);assert.equal(balanced.views,'sampled');assert.equal(balanced.granularity,'multilevel');
});

test('photo-set reset removes old forced geometry paths while preserving consent',()=>{
  const old={geometry_backend:'dust3r',sparse_completion:'learned_visible',completion:'learned',allow_geometry_download:true,semantic_view_mode:'all'};
  const changed=geometryOptionsForPhotoSet(old);
  assert.deepEqual(changed,{...old,geometry_backend:'auto',sparse_completion:'auto',completion:'auto'});
  assert.equal(old.geometry_backend,'dust3r');
  assert.equal(geometryOptionsForPhotoSet({...old,allow_geometry_download:false}).allow_geometry_download,false);
});

test('display reports actual loaded SH degrees and source count even for older metadata',()=>{
  const quality=sceneDisplayInfo({metadata:{preview_sh_degree:0,source_gaussian_count:100}},
    {loaded:20,total:40},[{shDegree:3},{shDegree:3}]);
  assert.equal(quality.color,'SH3');assert.equal(quality.loaded,20);assert.equal(quality.source,100);assert.equal(quality.partial,true);
  const mixed=sceneDisplayInfo({metadata:{}},{loaded:3,total:3},[{shDegree:0},{shDegree:3},{shDegree:-1}]);
  assert.equal(mixed.color,'SH0/SH3 + 固定颜色');assert.equal(mixed.partial,false);
});

test('absent SH coefficients cannot be reported as restored higher-order color',()=>{
  const quality=sceneDisplayInfo({metadata:{preview_sh_degree:3,source_gaussian_count:'unknown'}},
    {loaded:8,total:9},[{shDegree:-1}]);
  assert.equal(quality.color,'固定颜色');assert.equal(quality.source,9);assert.deepEqual(quality.shDegrees,[]);
  assert.equal(sceneDisplayInfo({metadata:{preview_sh_degree:0}},{loaded:1,total:1},[{shDegree:-1}]).color,'SH0 固定颜色');
});
