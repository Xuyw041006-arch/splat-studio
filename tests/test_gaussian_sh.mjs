import assert from 'node:assert/strict';
import test from 'node:test';
import * as THREE from 'three';
import {evaluateSH, gaussianLoadPlan, packSHTexture, shTextureLayout, shBasis, validateSH} from '../frontend/gaussian_sh.js';
import {GaussianViewer} from '../frontend/viewer.js';

const close=(actual,expected,tolerance=1e-12)=>{
  assert.equal(actual.length,expected.length);
  actual.forEach((value,index)=>assert.ok(Math.abs(value-expected[index])<tolerance,`${value} != ${expected[index]}`));
};

test('all degree 0–3 terms match pinned upstream NumPy eval_sh golden values',()=>{
  // Generated independently with the pinned utils/sh_utils.py at commit
  // 54c035f7834b564019656c3e3fcc3646292f727d, float64 coefficients sin(i+1)*.8.
  const sh=Array.from({length:48},(_,i)=>Math.sin(i+1)*.8);
  const golden=[
    {direction:[1,2,3],degrees:[[.689899665794515,.28303581977304615,.7256559423753526],
      [.6232041648638009,.3915422400076482,.5845270392788271],
      [.5381674373547681,.4830669315659744,.4942647896788503],
      [.14518197216689172,.947566333602389,0]]},
    {direction:[-2,1,.5],degrees:[[.689899665794515,.28303581977304615,.7256559423753526],
      [.28860205901365843,.7354221518127405,.2604894298631376],
      [.7566941989002229,.45677542710486674,.32609464513168435],
      [.8731991886468706,.32138987968352123,.4688961613586822]]},
    {direction:[0,0,1],degrees:[[.689899665794515,.28303581977304615,.7256559423753526],
      [.7450609381287642,.3416201246677014,.5582870400800588],
      [1.0765937888805213,0,1.044643736821332],
      [1.3274680190856762,0,1.552703027499165]]},
  ];
  for(const row of golden)row.degrees.forEach((expected,degree)=>close(evaluateSH(sh,degree,row.direction),expected));
});

test('camera-to-point orientation flips odd SH bands and preserves even ones',()=>{
  const forward=shBasis([1,2,3]),reverse=shBasis([-1,-2,-3]);
  forward.forEach((value,index)=>{
    const sign=(index>=1&&index<4)||index>=9?-1:1;
    close([reverse[index]],[sign*value]);
  });
  close(shBasis([1,0,0],1),[.28209479177387814,0,0,-.4886025119029199]);
  close(shBasis([0,1,0],1),[.28209479177387814,-.4886025119029199,0,0]);
});

test('SH validates channel-major coefficient extents and preserves HDR clamp semantics',()=>{
  for(const degree of [0,1,2,3]){
    const length=(degree+1)**2*3;
    assert.equal(validateSH(new Float32Array(length),degree),length/3);
    close(evaluateSH(new Float32Array(length),degree,[0,0,1]),[.5,.5,.5]);
  }
  close(evaluateSH([10,-10,0],0,[0,0,1]),[3.3209479177387813,0,.5]);
  for(const [coefficients,degree] of [[[1,2],0],[new Array(12).fill(0),2],[new Array(48).fill(0),4],[[1,NaN,1],0],[[1e300,1,1],0]]){
    assert.throws(()=>validateSH(coefficients,degree));
  }
  assert.throws(()=>shBasis([0,0,0]));
});

test('texture bank packs RGB per basis and crosses rows without reordering source identities',()=>{
  const sh=Array.from({length:48},(_,i)=>i+.25);
  const bank=packSHTexture([{sh,sh_degree:3},{sh:[.1,.2,.3],sh_degree:0}],16);
  assert.equal(bank.width,16);assert.equal(bank.height,2);assert.equal(bank.count,2);
  close(Array.from(bank.data.slice(0,4)),[.25,16.25,32.25,0]);
  close(Array.from(bank.data.slice(15*4,16*4)),[15.25,31.25,47.25,0]);
  close(Array.from(bank.data.slice(16*4,17*4)),[.1,.2,.3,0],1e-7);
  assert.ok(bank.data.slice(17*4).every(value=>value===0));
  assert.equal(bank.bytes,16*2*4*4);
  assert.throws(()=>packSHTexture(new Array(257).fill({sh:[0,0,0],sh_degree:0}),16),/GPU/);
  const empty=packSHTexture([]);assert.equal(empty.width,1);assert.equal(empty.height,1);
});

test('all supplied Gaussians are eligible by default; optional explicit budget is exact',()=>{
  const full=gaussianLoadPlan(300001);
  assert.equal(full.selectedCount,300001);assert.equal(full.sampled,false);assert.equal(full.omittedByBudget,0);
  let last=-1,count=0;for(const index of full.indices()){assert.equal(index,++last);count++;}
  assert.equal(count,300001);
  const budget=gaussianLoadPlan(11,4);
  assert.deepEqual([...budget.indices()],[0,2,5,8]);
  assert.equal(budget.selectedCount,4);assert.equal(budget.omittedByBudget,7);assert.equal(budget.sampled,true);
  assert.equal(budget.maxGaussians,4);
  assert.deepEqual([...gaussianLoadPlan(0).indices()],[]);
  for(const value of [0,-1,NaN,Infinity,1.1])assert.throws(()=>gaussianLoadPlan(5,value));
});

function cpuViewer(){
  const viewer=Object.create(GaussianViewer.prototype);
  Object.assign(viewer,{records:[],objects:[],objectMap:new Map(),hiddenIds:new Set(),bounds:new THREE.Box3(),
    scene:new THREE.Scene(),grid:new THREE.GridHelper(),camera:new THREE.PerspectiveCamera(48,1,.01,1000),
    sourceFilter:'all',isolationIds:null,resetView(){},resize(){},renderer:{capabilities:{maxTextureSize:16}}});
  return viewer;
}
const gaussian=(z,extra={})=>({position:[0,0,z],scale:[.1,.1,.1],color:[.2,.3,.4],opacity:.8,
  semantic_ids:['object'],...extra});

test('real viewer sorting and visibility retain SH record indices and DC fallback',()=>{
  const viewer=cpuViewer(),coefficients=Array.from({length:48},(_,i)=>i/100);
  const result=viewer.load({objects:[{id:'object',level:'object'}],gaussians:[
    gaussian(-1,{sh:[.1,.2,.3],sh_degree:0,source:'observed'}),
    gaussian(-3,{sh:coefficients,sh_degree:3,source:'inferred'}),
    gaussian(-2,{source:'unknown'})]});
  assert.equal(result.loaded,3);assert.equal(result.shGaussians,2);assert.equal(result.sampled,false);
  assert.deepEqual(Array.from(viewer.arrays.shInfo),[1,3,-1,-1,0,0]);
  assert.equal(viewer.material.uniforms.shTexture.value,viewer.shTexture);
  assert.equal(viewer.material.uniforms.shTextureWidth.value,16);
  viewer.setSourceFilter('observed');viewer.sortSplats();
  assert.equal(viewer.visibleCount,1);assert.deepEqual(Array.from(viewer.arrays.shInfo.slice(0,2)),[0,0]);
  viewer.setSourceFilter('all');viewer.setHidden(['object']);viewer.sortSplats();assert.equal(viewer.visibleCount,0);
  viewer.setHidden([]);viewer.setIsolation(['object']);viewer.select('object');viewer.sortSplats();
  assert.equal(viewer.visibleCount,3);assert.ok(viewer.arrays.highlight.every(value=>value===1));
  let disposed=0;viewer.shTexture.addEventListener('dispose',()=>disposed++);viewer.clear();assert.equal(disposed,1);
});

test('viewer exposes explicit budget and rejects malformed SH without silent DC substitution',()=>{
  const viewer=cpuViewer();
  const result=viewer.load({gaussians:[gaussian(-1),gaussian(-2),gaussian(-3)]},{maxGaussians:2});
  assert.equal(result.loaded,2);assert.equal(result.sampled,true);assert.equal(result.omittedByBudget,1);
  assert.equal(result.maxGaussians,2);viewer.clear();
  assert.throws(()=>viewer.load({gaussians:[gaussian(-1,{sh:[0,0],sh_degree:3})]}),/高斯 0/);
  viewer.clear();
});

test('full SH3 texture budgets fit modern texture dimensions with every coefficient retained',()=>{
  for(const count of [1065515,1706789]){
    const layout=shTextureLayout(count,16,16384);
    assert.ok(layout.bytes>=count*256);assert.ok(layout.bytes<count*256+layout.width*16);
    assert.ok(layout.width<=16384&&layout.height<=16384);
    assert.equal(gaussianLoadPlan(count).selectedCount,count);
  }
  assert.throws(()=>shTextureLayout(1065515,16,4096),/没有抽样或降低球谐阶数/);
  const degreeZero=packSHTexture([{sh:[.1,.2,.3],sh_degree:0}],16);
  assert.equal(degreeZero.coefficientStride,1);assert.equal(degreeZero.count,1);
  const inactive=Array.from({length:48},(_,i)=>i+.25),bank=packSHTexture([{sh:inactive,sh_degree:0}],16);
  assert.equal(bank.coefficientStride,16);assert.equal(bank.data[15*4+2],inactive[47]);
});

test('renderer reuses source vectors and shared semantic groups and releases CPU banks on clear',()=>{
  const viewer=cpuViewer(),shared=gaussian(-2,{sh:[.1,.2,.3],sh_degree:0}),source={objects:[{id:'object',level:'object'}],gaussians:[shared,{...shared,position:[0,0,-3]}]};
  viewer.load(source);
  assert.equal(viewer.records[0].position,shared.position);assert.equal(viewer.records[0].scale,shared.scale);assert.equal(viewer.records[0].color,shared.color);
  assert.equal(viewer.records[0].semanticIds,viewer.records[1].semanticIds);
  assert.equal(viewer.material.uniforms.shCoefficientStride.value,1);
  assert.deepEqual(Array.from(viewer.arrays.shInfo),[1,0,0,0]);
  viewer.clear();assert.equal(viewer.arrays,null);assert.equal(viewer.sortOrder,null);assert.equal(viewer.sortDepth,null);
});
