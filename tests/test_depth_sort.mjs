import assert from 'node:assert/strict';
import test from 'node:test';
import {sortDepthIndices} from '../frontend/depth_sort.js';

test('float64 radix sorting exactly matches stable numeric ordering without depth quantization',()=>{
  const values=[0,-0,-Infinity,Infinity,Number.MIN_VALUE,-Number.MIN_VALUE,1,1+Number.EPSILON,1,-1,-1-Number.EPSILON,1e250,-1e250];
  let seed=374761393;
  for(let i=0;i<10000;i++){seed=(Math.imul(seed,1664525)+1013904223)>>>0;values.push((seed/2**32-.5)*10**((i%600)-300));}
  const depth=Float64Array.from(values),original=[...values.keys()],expected=original.slice().sort((a,b)=>values[a]-values[b]||a-b);
  const order=Uint32Array.from(original);assert.equal(sortDepthIndices(order,depth,new Uint32Array(order.length)),order);
  assert.deepEqual(Array.from(order),expected);assert.deepEqual(depth,Float64Array.from(values));
});

test('hidden-row subsets and empty scenes remain valid; malformed buffers fail explicitly',()=>{
  const depth=new Float64Array([1,-3,2,-4]),order=new Uint32Array([0,2]);
  sortDepthIndices(order,depth,new Uint32Array(4));assert.deepEqual(Array.from(order),[0,2]);
  sortDepthIndices(new Uint32Array(),new Float64Array(),new Uint32Array());
  assert.throws(()=>sortDepthIndices(new Uint32Array([1]),new Float64Array([0]),new Uint32Array(1)),/索引/);
  assert.throws(()=>sortDepthIndices(new Uint32Array([0]),new Float64Array([NaN]),new Uint32Array(1)),/深度/);
  assert.throws(()=>sortDepthIndices(new Uint32Array([0]),new Float64Array([0]),new Uint32Array()),/缓冲区/);
});
