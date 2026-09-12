import assert from 'node:assert/strict';
import test from 'node:test';
import * as THREE from 'three';
import {GaussianViewer} from '../frontend/viewer.js';
import {semanticIndex, gaussianMembership, membershipVisible, visibilitySnapshot, restoreVisibility, sceneWithVisibility} from '../frontend/scene_visibility.js';

const objects=[{id:'scene',level:'scene',parent_id:null},{id:'A',level:'object',parent_id:'scene'},
  {id:'B',level:'object',parent_id:'scene'},{id:'part-A',level:'part',parent_id:'A'}];
const index=semanticIndex(objects);
const g=ids=>({position:[0,0,1],scale:[.1,.2,.3],rotation:[1,0,0,0],color:[.2,.4,.6],opacity:.9,semantic_ids:ids,source:'unknown'});
const record=ids=>({...g(ids),...gaussianMembership(g(ids),index)});
const fixtures=[g(['A']),g(['A','B']),g(['B']),g([]),g(['part-A'])];

// Exercise the real renderer's load/packing path without a browser or WebGL context.
function cpuViewer(){
  const v=Object.create(GaussianViewer.prototype);
  Object.assign(v,{records:[],objects:[],objectMap:new Map(),hiddenIds:new Set(),bounds:new THREE.Box3(),
    scene:new THREE.Scene(),grid:new THREE.GridHelper(),camera:new THREE.PerspectiveCamera(48,1,.01,1000),
    sourceFilter:'inferred',isolationIds:new Set(['old']),resetView(){},resize(){}});
  return v;
}
test('isolation keeps shared A+B membership; hide/delete A removes its full union',()=>{
  assert.equal(membershipVisible(record(['A','B']),new Set(),new Set(['A'])),true);
  assert.equal(membershipVisible(record(['B']),new Set(),new Set(['A'])),false);
  assert.equal(membershipVisible(record(['A','B']),new Set(['A'])),false);
  assert.equal(membershipVisible(record(['B']),new Set(['A'])),true);
  assert.equal(membershipVisible(record(['part-A']),new Set(),new Set(['A'])),true);
});
test('ancestors of every member are included; scene root hide includes unlabeled splats',()=>{
  assert.deepEqual([...record(['A','B']).semanticIds].sort(),['A','B','scene']);
  assert.equal(record(['part-A']).semanticIds.has('A'),true);
  for(const gaussian of fixtures){
    const r={...gaussian,...gaussianMembership(gaussian,index)};
    assert.equal(membershipVisible(r,new Set(['scene'])),false);
    assert.equal(membershipVisible(r,new Set(),new Set(['scene'])),true);
  }
  assert.equal(membershipVisible(record([]),new Set(['__unlabeled__'])),false);
  assert.equal(membershipVisible(record(['A']),new Set(['__unlabeled__'])),true);
});
test('undo restores independent isolation and explicit hide sets without shared references',()=>{
  const hidden=new Set(['B']),isolation=new Set(['A']);
  const before=visibilitySnapshot(hidden,isolation);hidden.clear();isolation.add('B');
  const restored=restoreVisibility(before);
  assert.deepEqual([...restored.hidden],['B']);assert.deepEqual([...restored.isolation],['A']);
  assert.equal(membershipVisible(record(['A','B']),restored.hidden,restored.isolation),false);
  assert.equal(restoreVisibility(visibilitySnapshot(new Set(),null)).isolation,null);
});
test('export/reimport preserves isolation without hiding the shared A+B Gaussian',()=>{
  const scene={objects,gaussians:fixtures,metadata:{backend:'CUDA',preview_sh_degree:0,synthetic_demo:false}};
  const exported=JSON.parse(JSON.stringify(sceneWithVisibility(scene,new Set(),new Set(['A']))));
  assert.equal(exported.objects.find(o=>o.id==='B').visible,true);
  assert.equal(exported.gaussians[1].hidden,false);
  const viewer=cpuViewer();viewer.load(exported);
  assert.deepEqual(viewer.records.filter(r=>viewer.recordVisible(r)).map(r=>r.index),[0,1,4]);
  viewer.setIsolation(null);assert.equal(viewer.records.filter(r=>viewer.recordVisible(r)).length,5);
  assert.equal(exported.metadata.backend,'CUDA');assert.equal(exported.metadata.synthetic_demo,false);
  viewer.clear();
});
test('new scene resets an earlier source/isolation filter and unknown stats stay unknown',()=>{
  const viewer=cpuViewer();viewer.load({objects,gaussians:fixtures});
  assert.equal(viewer.sourceFilter,'all');assert.equal(viewer.isolationIds,null);assert.equal(viewer.visibleCount,5);
  assert.deepEqual(viewer.getObjectStats('scene'),{count:5,inferred:0,unknown:5,confidence:null});
  viewer.setSourceFilter('observed');viewer.load({objects,gaussians:[g(['A','B'])]});
  assert.equal(viewer.visibleCount,1);assert.equal(viewer.sourceFilter,'all');viewer.clear();
});
test('100k real-format records pack without an additional sample or invented owner',()=>{
  const viewer=cpuViewer();const gaussians=Array.from({length:100000},(_,i)=>({...g(['A','B']),source_index:i*3}));
  const result=viewer.load({objects,gaussians,metadata:{source_gaussian_count:300000,preview_sh_degree:0}});
  assert.equal(result.loaded,100000);assert.equal(result.sampled,false);assert.equal(viewer.visibleCount,100000);
  assert.equal(viewer.records[99999].index,99999);assert.equal(viewer.records[0].id,null);
  viewer.setIsolation(new Set(['A']));viewer.sortSplats();assert.equal(viewer.visibleCount,100000);
  viewer.setHidden(new Set(['scene']));viewer.sortSplats();assert.equal(viewer.visibleCount,0);viewer.clear();
});
