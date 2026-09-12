import test from 'node:test';
import assert from 'node:assert/strict';
import {savedViewerCamera,captureViewerCamera} from '../frontend/viewer_camera.js';
test('dataset pose preserves up-axis and derives fov from intrinsics without mutating it',()=>{
  const input={position:[1,2,3],forward:[0,0,2],up:[0,-3,0],height:730,intrinsics:[778,778,648,365]};
  const original=JSON.stringify(input),value=savedViewerCamera(input);
  assert.deepEqual(value.up,[0,-1,0]);assert.deepEqual(value.forward,[0,0,1]);
  assert.ok(value.fov>50&&value.fov<51);assert.equal(JSON.stringify(input),original);
});
test('invalid and degenerate poses fall back instead of corrupting camera navigation',()=>{
  const v={position:[0,0,0],forward:[0,0,1],up:[0,1,0],fov:50};
  for(const bad of [null,{}, {...v,up:[0,0,1]},{...v,forward:[0,0,0]},{...v,position:[NaN,0,0]},{...v,fov:Infinity},{...v,focus_distance:-1}])assert.equal(savedViewerCamera(bad),null);
  assert.deepEqual(savedViewerCamera(v).position,[0,0,0]);
});

test('registered world-to-camera pose opens from a real capture, preserving its coordinate frame',()=>{
 const camera={world_to_camera:[[0,0,1,-3],[0,1,0,-2],[-1,0,0,1],[0,0,0,1]],height:730,intrinsics:{fy:778}};
 const before=JSON.stringify(camera),v=captureViewerCamera([camera],[-3,2,3]);
 assert.deepEqual(v.position,[1,2,3]);assert.deepEqual(v.forward,[-1,0,0]);assert.equal(v.up[1],-1);assert.equal(v.distance,4);
 assert.equal(JSON.stringify(camera),before);
 const bad=structuredClone(camera);bad.world_to_camera[0][0]=2;
 assert.equal(captureViewerCamera([bad]),null);assert.deepEqual(captureViewerCamera([bad,camera],[-3,2,3]),v);
});
