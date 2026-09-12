import test from 'node:test';
import assert from 'node:assert/strict';
import {objectFocusBounds} from '../frontend/object_focus.js';

const point=(position,ids=['bear'])=>({position,semanticIds:new Set(ids),opacity:.8});
const close=(actual,expected)=>actual.forEach((value,i)=>assert.ok(Math.abs(value-expected[i])<1e-10,`${value} != ${expected[i]}`));

test('distant semantic outliers do not stretch a large-object camera frame',()=>{
  const ordinary=Array.from({length:100},(_,i)=>point([i/100,1+i/50,-1+i/200]));
  const records=[point([-1000,-2000,-3000]),...ordinary,point([5000,6000,7000])];
  const positions=records.map(r=>[...r.position]),memberships=records.map(r=>[...r.semanticIds]);
  const result=objectFocusBounds(records,['bear']);
  assert.equal(result.robust,true);assert.equal(result.count,102);
  close(result.min,[.0405,1.081,-.97975]);close(result.max,[.9495,2.899,-.52525]);
  assert.deepEqual(records.map(r=>r.position),positions);
  assert.deepEqual(records.map(r=>[...r.semanticIds]),memberships);
  assert.equal(records.length,102);assert.ok(records.every(r=>r.opacity===.8));
});

test('small selections keep their full extent and union memberships count only once',()=>{
  const records=[point([-2,3,1],['bear','part']),point([4,-1,9],['part']),
    point([1,5,-3],['bear']),point([1000,1000,1000],['other'])];
  const result=objectFocusBounds(records,['bear','part','bear']);
  assert.deepEqual(result,{min:[-2,-1,-3],max:[4,5,9],count:3,ignoredCount:0,robust:false,quantiles:[0,1]});
});

test('64 valid points enables exact per-axis 5th and 95th percentiles',()=>{
  const records=Array.from({length:64},(_,i)=>point([i,2*i,63-i]));
  const result=objectFocusBounds(records,['bear']);
  assert.equal(result.robust,true);close(result.min,[3.15,6.3,3.15]);close(result.max,[59.85,119.7,59.85]);
  const smaller=objectFocusBounds(records.slice(0,63),['bear']);
  assert.equal(smaller.robust,false);assert.deepEqual(smaller.min,[0,0,1]);assert.deepEqual(smaller.max,[62,124,63]);
});

test('nonfinite or malformed selected positions cannot corrupt camera bounds',()=>{
  const valid=Array.from({length:63},(_,i)=>point([i,1,2]));
  const invalid=[[NaN,0,0],[0,Infinity,0],[0,0,-Infinity],['1',2,3],[],null];
  const records=[...valid,...invalid.map(p=>point(p)),null,{position:[1,2,3]},point([999,999,999],['other'])];
  const result=objectFocusBounds(records,['bear']);
  assert.equal(result.count,63);assert.equal(result.ignoredCount,6);assert.equal(result.robust,false);
  assert.deepEqual(result.min,[0,1,2]);assert.deepEqual(result.max,[62,1,2]);
  assert.equal(objectFocusBounds(invalid.map(p=>point(p)),['bear']),null);
});

test('missing selections return no frame and single or coincident points remain valid',()=>{
  assert.equal(objectFocusBounds([],['bear']),null);
  assert.equal(objectFocusBounds([point([1,2,3])],['absent']),null);
  assert.equal(objectFocusBounds([point([1,2,3])],[]),null);
  assert.equal(objectFocusBounds(null,['bear']),null);
  assert.equal(objectFocusBounds([point([1,2,3])],null),null);
  const single=objectFocusBounds([point([1,2,3],['12'])],new Set([12]));
  assert.deepEqual(single.min,[1,2,3]);assert.deepEqual(single.max,[1,2,3]);
  const coincident=objectFocusBounds(Array.from({length:100},()=>point([1,2,3])),['bear']);
  assert.deepEqual(coincident.min,[1,2,3]);assert.deepEqual(coincident.max,[1,2,3]);
});
