import assert from 'node:assert/strict';
import test from 'node:test';
import {fetchSceneResource,sceneJsonParts,sceneLinesParts,iterateSceneLines,writeSceneLines,readSceneLines,SCENE_MANIFEST_FORMAT,SCENE_CHUNK_FORMAT,SCENE_LINES_FORMAT} from '../frontend/scene_transport.js';
import {sceneWithVisibility,sceneVisibilityExport,readIsolation} from '../frontend/scene_visibility.js';

const url='/api/projects/p/files/runs/r/scene.viewer.json';
const records=Array.from({length:5},(_,i)=>({position:[i,.25,-2],source_index:i*13,
  source:i%2?'inferred':'observed',sh_degree:3,sh:Array.from({length:48},(_,j)=>j*.01-i),
  semantic_memberships:['cup','handle'],object_id:`instance-${i}`}));
const header={objects:[{id:'cup',label:'杯子'}],cameras:[{frame_id:'照片.png'}],metadata:{source_gaussian_count:65}};
function fixture(){
  const manifest={format:SCENE_MANIFEST_FORMAT,gaussian_count:5,chunk_size:2,scene:structuredClone(header),
    chunks:[{path:'data/a.json',offset:0,count:2},{path:'data/b.json',offset:2,count:2},{path:'data/c.json',offset:4,count:1}]};
  const pages=new Map([[url,manifest]]);
  for(const chunk of manifest.chunks)pages.set('/api/projects/p/files/runs/r/'+chunk.path,
    {format:SCENE_CHUNK_FORMAT,offset:chunk.offset,count:chunk.count,gaussians:records.slice(chunk.offset,chunk.offset+chunk.count)});
  return {manifest,pages};
}

test('sequential chunks preserve exact SH, semantic identities, row order and header',async()=>{
  const {pages}=fixture();const calls=[];let active=0,peak=0;
  const read=async path=>{calls.push(path);active++;peak=Math.max(peak,active);await Promise.resolve();active--;return pages.get(path);};
  assert.deepEqual(await fetchSceneResource(url,read),{...header,gaussians:records});
  assert.equal(peak,1);assert.equal(calls.length,4);
  assert.deepEqual(calls.slice(1),['a','b','c'].map(n=>'/api/projects/p/files/runs/r/data/'+n+'.json'));
});

test('legacy plain scene needs only one read and remains unchanged',async()=>{
  const scene={...header,gaussians:records};let count=0;
  const result=await fetchSceneResource('/scene.json',async()=>{count++;return scene;});
  assert.equal(result,scene);assert.equal(count,1);
});

test('untrusted remote, encoded, absolute and parent paths rejected before chunk fetch',async()=>{
  for(const path of ['https://evil/a.json','//evil/a.json','/absolute.json','../secret','data/../secret',
                     '%2e%2e/secret','data/a.json?secret','data/a.json#fragment','data\\a.json']){
    const {manifest}=fixture();manifest.chunks[0].path=path;let reads=0;
    await assert.rejects(fetchSceneResource(url,async()=>{reads++;return manifest;}),/路径/);
    assert.equal(reads,1);
  }
});

test('bad count, ordering, duplicate paths and header fail before fetching chunks',async()=>{
  for(const alter of [m=>m.gaussian_count=6,m=>m.gaussian_count=2000001,m=>m.chunks[1].offset=1,
    m=>m.chunks[1].path=m.chunks[0].path,m=>m.chunks[0].count=3,m=>m.scene.gaussians=[],m=>m.chunk_size=25001]){
    const {manifest}=fixture();alter(manifest);let reads=0;
    await assert.rejects(fetchSceneResource(url,async()=>{reads++;return manifest;}));
    assert.equal(reads,1);
  }
});

test('chunk payload count and offset must agree with manifest',async()=>{
  for(const alter of [c=>c.offset=3,c=>c.count=1,c=>c.gaussians.pop(),c=>c.format='other',c=>c.gaussians[0]=null]){
    const {pages}=fixture();const chunk=structuredClone(pages.get('/api/projects/p/files/runs/r/data/a.json'));
    alter(chunk);pages.set('/api/projects/p/files/runs/r/data/a.json',chunk);
    await assert.rejects(fetchSceneResource(url,async path=>pages.get(path)),/清单不一致/);
  }
});

test('unknown manifest format does not become an empty scene',async()=>{
  await assert.rejects(fetchSceneResource(url,async()=>({format:'future',chunks:[]})),/可识别/);
});

test('bounded JSON export parts preserve scene without one giant serialization',()=>{
  const scene={...header,gaussians:records};
  const parts=sceneJsonParts(scene,{chunkSize:2});
  assert.deepEqual(JSON.parse(parts.join('')),scene);assert.ok(parts.length>3);
  assert.deepEqual(JSON.parse(sceneJsonParts({gaussians:[]}).join('')),{gaussians:[]});
  assert.deepEqual(scene.gaussians,records);
});

function byteStream(bytes,width=7){
  let offset=0;
  return new ReadableStream({pull(controller){
    if(offset===bytes.length){controller.close();return;}
    const end=Math.min(bytes.length,offset+width);controller.enqueue(bytes.slice(offset,end));offset=end;
  }});
}
const bytes=text=>new TextEncoder().encode(text);

test('JSONL streamed roundtrip preserves SH colors, source order, unicode and visibility edits',async()=>{
  const base={...header,gaussians:records.map(g=>({...g,semantic_ids:['cup']}))};
  const edited=sceneWithVisibility(base,new Set(['cup']),new Set(['cup']));
  const parts=sceneLinesParts(edited,{chunkSize:2});
  assert.equal(JSON.parse(parts[0]).format,SCENE_LINES_FORMAT);
  assert.ok(parts.every(line=>line.endsWith('\n')));
  // One-byte reads split every multibyte Chinese codepoint and JSON token.
  const loaded=await readSceneLines(byteStream(bytes(parts.join('')),1));
  assert.deepEqual(loaded,edited);
  assert.deepEqual([...readIsolation(loaded,loaded.objects)],['cup']);
  assert.ok(loaded.gaussians.every(g=>g.hidden));
  assert.deepEqual(loaded.gaussians.map(g=>g.source_index),[0,13,26,39,52]);
});

test('JSONL reads browser Blob streams without file.text or one whole-file string',async()=>{
  const scene={...header,gaussians:records};const blob=new Blob(sceneLinesParts(scene,{chunkSize:2}));
  blob.text=()=>{throw new Error('whole-file text must not be read');};
  assert.deepEqual(await readSceneLines(blob),scene);
});

test('JSONL export adaptively reduces chunk count to honor UTF-8 byte limit',async()=>{
  const scene={...header,gaussians:records};
  const single=Math.max(...sceneLinesParts(scene,{chunkSize:1}).map(line=>bytes(line.trimEnd()).length));
  const maximum=single+10;
  const parts=sceneLinesParts(scene,{chunkSize:5,maxLineBytes:maximum});
  assert.ok(parts.length>2);
  assert.ok(parts.every(line=>bytes(line.trimEnd()).length<=maximum));
  assert.deepEqual(await readSceneLines(new Blob(parts),{maxLineBytes:maximum}),scene);
});

test('JSONL truncated file, missing final newline and missing final chunk rejected',async()=>{
  const parts=sceneLinesParts({...header,gaussians:records},{chunkSize:2});
  for(const text of [parts.join('').slice(0,-10),parts.join('').slice(0,-1),parts.slice(0,-1).join('')]){
    await assert.rejects(readSceneLines(byteStream(bytes(text),3)),/截断|缺少/);
  }
});

test('JSONL invalid count, duplicated offset, wrong format and extra chunk rejected',async()=>{
  const parts=sceneLinesParts({...header,gaussians:records},{chunkSize:2});
  const parsed=parts.map(JSON.parse);
  for(const alter of [p=>p[0].gaussian_count=2000001,p=>p[1].offset=1,p=>p[2].offset=0,
                     p=>p[1].count=3,p=>p[1].format='unknown',p=>p.push(p[1])]){
    const values=structuredClone(parsed);alter(values);
    const text=values.map(value=>JSON.stringify(value)+'\n').join('');
    await assert.rejects(readSceneLines(byteStream(bytes(text),17)),/无效|不一致/);
  }
});

test('JSONL oversized line cancels input before unbounded accumulation',async()=>{
  let cancelled=false;
  const stream=new ReadableStream({start(controller){controller.enqueue(bytes(' '.repeat(201)));},
    cancel(){cancelled=true;}});
  await assert.rejects(readSceneLines(stream,{maxLineBytes:200}),/单行超过/);
  assert.ok(cancelled);
});

test('JSONL malformed UTF-8 and blank extra line are rejected',async()=>{
  await assert.rejects(readSceneLines(byteStream(new Uint8Array([0xc3,0x28,10]),1)),/UTF-8/);
  const parts=sceneLinesParts({...header,gaussians:records},{chunkSize:2});
  await assert.rejects(readSceneLines(new Blob([...parts,'\n'])),/有效 JSON/);
});

test('full Bonsai and Teatime counts pass sequential transport without missing source rows',async()=>{
  for(const count of [1065515,1706789]){
    const chunkSize=25000,chunks=[];let calls=0,progress=0;
    for(let offset=0;offset<count;offset+=chunkSize)chunks.push({path:`chunks/${offset}.json`,offset,count:Math.min(chunkSize,count-offset)});
    const manifest={format:SCENE_MANIFEST_FORMAT,gaussian_count:count,chunk_size:chunkSize,scene:{objects:[]},chunks};
    const sh=Array.from({length:48},(_,i)=>i/100);
    const loaded=await fetchSceneResource('/scene.viewer.json',async url=>{
      calls++;if(url==='/scene.viewer.json')return manifest;
      const offset=Number(url.match(/(\d+)\.json$/)[1]),entry=chunks[offset/chunkSize];
      return {format:SCENE_CHUNK_FORMAT,...entry,gaussians:Array.from({length:entry.count},(_,i)=>({source_index:offset+i,sh_degree:3,sh}))};
    },{onProgress:value=>{assert.ok(value.loaded>=progress);progress=value.loaded;}});
    assert.equal(loaded.gaussians.length,count);assert.equal(progress,count);assert.equal(calls,chunks.length+1);
    for(let i=0;i<count;i++)assert.equal(loaded.gaussians[i].source_index,i);
    assert.equal(loaded.gaussians[count-1].sh,sh);
  }
});

test('plain scenes and invalid caller capacities cannot bypass the full-scene safety guard',async()=>{
  await assert.rejects(fetchSceneResource('/scene.json',async()=>({gaussians:new Array(2000001)})),/超过当前完整显示上限/);
  await assert.rejects(fetchSceneResource('/scene.json',async()=>({gaussians:[]}),{maxGaussians:2000001}),/上限无效/);
});

test('streaming export applies visibility per chunk and honors writer backpressure',async()=>{
  const original={...header,gaussians:records.map(g=>({...g,semantic_ids:['cup']}))};
  const prepared=sceneVisibilityExport(original,new Set(['cup']),new Set(['cup']));
  assert.equal(prepared.scene.gaussians,original.gaussians);
  let mapped=0,pending=false,closed=false;const written=[];
  await writeSceneLines(prepared.scene,{async write(line){assert.equal(pending,false);pending=true;const before=mapped;await Promise.resolve();assert.equal(mapped,before);written.push(line);pending=false;},async close(){closed=true;}},
    {chunkSize:2,mapGaussian:g=>{mapped++;return prepared.mapGaussian(g);}});
  assert.equal(mapped,records.length);assert.ok(closed);
  const restored=await readSceneLines(new Blob(written));
  assert.deepEqual(restored,sceneWithVisibility(original,new Set(['cup']),new Set(['cup'])));
  assert.ok(original.gaussians.every(g=>g.hidden===undefined));
  let lazyCount=0;const iterator=iterateSceneLines(original,{chunkSize:2,mapGaussian:g=>{lazyCount++;return g;}});
  iterator.next();assert.equal(lazyCount,0);iterator.next();assert.equal(lazyCount,2);iterator.return();assert.equal(lazyCount,2);
});

test('failed streaming export aborts without creating subsequent chunks or closing successfully',async()=>{
  let writes=0,mapped=0,aborted=false,closed=false;
  await assert.rejects(writeSceneLines({...header,gaussians:records},{write(){if(++writes===2)throw new Error('disk full');},close(){closed=true;},abort(){aborted=true;}},
    {chunkSize:2,mapGaussian:g=>{mapped++;return g;}}),/disk full/);
  assert.equal(mapped,2);assert.equal(writes,2);assert.ok(aborted);assert.equal(closed,false);
});
