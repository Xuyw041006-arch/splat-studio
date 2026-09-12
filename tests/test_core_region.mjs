import assert from 'node:assert/strict';
import test from 'node:test';
import * as THREE from 'three';
import {coreRegionSettings,fitCoreRegion,insideCoreRegion,CORE_REGION_FORMAT} from '../frontend/core_region.js';
import {GaussianViewer} from '../frontend/viewer.js';
import {sceneVisibilityExport} from '../frontend/scene_visibility.js';
import {writeSceneLines,readSceneLines} from '../frontend/scene_transport.js';

const gaussian=(position,ids=[])=>({position,scale:[.1,.1,.1],rotation:[1,0,0,0],color:[.2,.3,.4],opacity:.8,semantic_ids:ids,sh_degree:3,sh:Array.from({length:48},(_,i)=>i/100)});
const objects=[{id:'scene',level:'scene'},{id:'cup',label:'cup',parent_id:'scene',level:'object'},{id:'plate',label:'plate',parent_id:'scene',level:'object'}];
const points=Array.from({length:100},(_,i)=>gaussian([Math.cos(i)*.2,Math.sin(i)*.2,(i%7)*.01],i<50?['cup']:['plate']));
points.push(gaussian([100,100,100],['cup']));
function viewer(){const v=Object.create(GaussianViewer.prototype);Object.assign(v,{records:[],objects:[],objectMap:new Map(),hiddenIds:new Set(),bounds:new THREE.Box3(),scene:new THREE.Scene(),grid:new THREE.GridHelper(),camera:new THREE.PerspectiveCamera(48,1,.01,1000),sourceFilter:'all',isolationIds:null,resetView(){},resize(){},renderer:{capabilities:{maxTextureSize:256}}});return v;}

test('robust scene center and radius reject distant floaters without moving any input',()=>{
  const records=points.map(g=>({position:g.position,semanticIds:new Set(g.semantic_ids)})),before=JSON.stringify(records.map(r=>r.position));
  const region=fitCoreRegion(records),settings=coreRegionSettings();
  assert.ok(Math.hypot(...region.center)<.1);assert.ok(region.radius<.5);
  assert.equal(insideCoreRegion([100,100,100],region,settings),false);
  assert.equal(insideCoreRegion(region.center,region,settings),true);
  assert.equal(insideCoreRegion([100,100,100],region,{...settings,enabled:false}),true);
  assert.equal(JSON.stringify(records.map(r=>r.position)),before);
  assert.equal(fitCoreRegion([]),null);
});

test('million-point fitting uses bounded statistics while visibility can consider every row',()=>{
  const shared={position:[1,2,3],semanticIds:new Set(['cup'])},records=new Array(1065515).fill(shared);
  records[records.length-1]={position:[1e8,1e8,1e8],semanticIds:new Set(['noise'])};
  const region=fitCoreRegion(records);assert.equal(region.eligible_count,1065515);assert.equal(region.sample_count,16384);
  assert.deepEqual(region.center,[1,2,3]);assert.equal(insideCoreRegion(records.at(-1).position,region,coreRegionSettings()),false);
  const selected=fitCoreRegion(records,['cup']);assert.equal(selected.eligible_count,1065514);assert.deepEqual(selected.center,[1,2,3]);
});

test('core filtering intersects semantic visibility and preserves all loaded records and coefficients',()=>{
  const v=viewer(),source={objects,gaussians:structuredClone(points)};const before=JSON.stringify(source.gaussians);
  const loaded=v.load(source);assert.equal(loaded.loaded,101);assert.equal(v.records.length,101);assert.equal(v.coreIncluded,100);assert.equal(v.visibleCount,100);
  v.setIsolation(['cup']);v.sortSplats();assert.equal(v.visibleCount,50);
  v.setHidden(['cup']);v.sortSplats();assert.equal(v.visibleCount,0);
  v.setHidden([]);v.setIsolation(null);const previous=v.getCoreSettings();v.setCoreSettings({...previous,enabled:false});v.sortSplats();assert.equal(v.visibleCount,101);
  v.setCoreSettings(previous);v.sortSplats();assert.equal(v.visibleCount,100);
  v.setCoreSettings({...previous,target:'objects',object_ids:['cup']});assert.equal(v.coreRegion.eligible_count,51);
  const region=v.coreRegion;v.setCoreSettings({...v.getCoreSettings(),multiplier:.5});assert.equal(v.coreRegion,region);
  v.sortSplats();assert.ok(v.visibleCount<100);assert.equal(JSON.stringify(source.gaussians),before);
  assert.equal(v.shTexture.image.data[100*16*4+15*4+2],Math.fround(source.gaussians[100].sh[47]));v.clear();
});

test('export and reimport retain core settings separately without baking spatial hiding into Gaussians',async()=>{
  const first=viewer(),source={objects,gaussians:structuredClone(points)};first.load(source);
  const settings={...first.getCoreSettings(),target:'objects',object_ids:['plate'],multiplier:.75};first.setCoreSettings(settings);first.sortSplats();const visible=first.visibleCount;
  const prepared=sceneVisibilityExport(source,new Set(),null);prepared.scene.metadata.viewer_core_region=first.getCoreSettings();
  const output=[];await writeSceneLines(prepared.scene,{write:line=>output.push(line),close(){}},{chunkSize:25,mapGaussian:prepared.mapGaussian});
  const restored=await readSceneLines(new Blob(output));assert.equal(restored.gaussians.length,101);assert.ok(restored.gaussians.every(g=>!g.hidden));
  assert.deepEqual(restored.gaussians.map(g=>g.sh),source.gaussians.map(g=>g.sh));
  const second=viewer();second.load(restored);assert.deepEqual(second.getCoreSettings(),settings);assert.equal(second.visibleCount,visible);
  second.setCoreSettings({...settings,enabled:false});second.sortSplats();assert.equal(second.visibleCount,101);first.clear();second.clear();
});

test('core settings reject invalid ranges and safely fall back when an object no longer exists',()=>{
  const settings=coreRegionSettings();assert.equal(settings.enabled,true);assert.equal(settings.format,CORE_REGION_FORMAT);
  for(const change of [{multiplier:0},{multiplier:Infinity},{multiplier:4.1},{enabled:'yes'},{target:'remote'},{object_ids:'cup'}])assert.throws(()=>coreRegionSettings({...settings,...change},objects),/无效/);
  const restored=coreRegionSettings({...settings,target:'objects',object_ids:['missing']},objects);assert.equal(restored.target,'scene');assert.deepEqual(restored.object_ids,[]);
});

test('saved empty-object cores explicitly become scene cores; nonfinite fits and distance overflow never admit far points',()=>{
  const v=viewer(),saved={...coreRegionSettings(),target:'objects',object_ids:['plate']};
  const result=v.load({objects,gaussians:[gaussian([0,0,0],['cup'])],metadata:{viewer_core_region:saved}});
  assert.equal(v.getCoreSettings().target,'scene');assert.deepEqual(v.getCoreSettings().object_ids,[]);assert.equal(result.coreWarnings.length,1);
  assert.throws(()=>v.setCoreSettings(saved),/没有可用于/);v.clear();
  assert.equal(insideCoreRegion([2e200,0,0],{center:[0,0,0],radius:1e200},{enabled:true,multiplier:1}),false);
  assert.equal(insideCoreRegion([.5e200,0,0],{center:[0,0,0],radius:1e200},{enabled:true,multiplier:1}),true);
  assert.throws(()=>fitCoreRegion([{position:[-1.7e308,-1.7e308,-1.7e308]},{position:[1.7e308,1.7e308,1.7e308]}]),/有限数值/);
});
