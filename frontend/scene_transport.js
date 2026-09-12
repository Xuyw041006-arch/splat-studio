// Bounded transport avoids V8's single-JSON-string limit. The final scene still
// occupies browser/GPU memory; this loader never samples or renumbers records.
import { MAX_SCENE_GAUSSIANS, assertSceneCapacity } from './scene_limits.js';
export const SCENE_MANIFEST_FORMAT='splat-studio-scene-chunks/1';
export const SCENE_CHUNK_FORMAT='splat-studio-scene-chunk/1';
export const SCENE_LINES_FORMAT='splat-studio-scene-lines/1';

function object(value){return value!==null&&typeof value==='object'&&!Array.isArray(value);}
function positiveInteger(value){return Number.isSafeInteger(value)&&value>0;}
function relativeChunkPath(path){
  return typeof path==='string'&&/^[A-Za-z0-9._-]+(?:\/[A-Za-z0-9._-]+)*$/.test(path)&&
    path.split('/').every(part=>part!=='.'&&part!=='..');
}

export async function fetchSceneResource(url,fetchJson,{maxGaussians=MAX_SCENE_GAUSSIANS,onProgress=()=>{}}={}){
  if(typeof url!=='string'||!url||typeof fetchJson!=='function')throw new Error('场景地址或读取器无效。');
  assertSceneCapacity(0,maxGaussians);
  const manifest=await fetchJson(url);
  if(object(manifest)&&Array.isArray(manifest.gaussians)){assertSceneCapacity(manifest.gaussians.length,maxGaussians);onProgress({loaded:manifest.gaussians.length,total:manifest.gaussians.length});return manifest;}
  if(!object(manifest)||manifest.format!==SCENE_MANIFEST_FORMAT)throw new Error('场景不是可识别的 JSON 或分块清单。');
  assertSceneCapacity(manifest.gaussian_count,maxGaussians);
  if(!positiveInteger(manifest.gaussian_count)||manifest.gaussian_count>maxGaussians||
     !positiveInteger(manifest.chunk_size)||manifest.chunk_size>25000||
     !object(manifest.scene)||Object.hasOwn(manifest.scene,'gaussians')||
     !Array.isArray(manifest.chunks)||!manifest.chunks.length||manifest.chunks.length>1000)
    throw new Error('场景分块清单的总数、图幅或头信息无效。');
  let offset=0;const seen=new Set();
  // Validate the complete manifest BEFORE following any of its paths. Remote,
  // absolute, percent-encoded, query and traversal paths are deliberately absent.
  for(const chunk of manifest.chunks){
    if(!object(chunk)||!relativeChunkPath(chunk.path)||seen.has(chunk.path)||
       !Number.isSafeInteger(chunk.offset)||chunk.offset!==offset||
       !positiveInteger(chunk.count)||chunk.count>manifest.chunk_size)
      throw new Error('场景分块路径、顺序或数量无效。');
    seen.add(chunk.path);offset+=chunk.count;
  }
  if(offset!==manifest.gaussian_count)throw new Error('场景分块总数不一致。');
  const cleanUrl=url.split(/[?#]/,1)[0];
  const base=cleanUrl.slice(0,cleanUrl.lastIndexOf('/')+1);
  const gaussians=new Array(manifest.gaussian_count);
  onProgress({loaded:0,total:manifest.gaussian_count});
  for(const entry of manifest.chunks){
    const chunk=await fetchJson(base+entry.path);
    if(!object(chunk)||chunk.format!==SCENE_CHUNK_FORMAT||chunk.offset!==entry.offset||chunk.count!==entry.count||
       !Array.isArray(chunk.gaussians)||chunk.gaussians.length!==entry.count||!chunk.gaussians.every(object))
      throw new Error('场景分块内容与清单不一致，请重新生成或导入模型。');
    for(let i=0;i<entry.count;i++)gaussians[entry.offset+i]=chunk.gaussians[i];
    onProgress({loaded:entry.offset+entry.count,total:manifest.gaussian_count});
  }
  return {...manifest.scene,gaussians};
}

export function sceneJsonParts(scene,{chunkSize=25000}={}){
  if(!object(scene)||!Array.isArray(scene.gaussians)||!positiveInteger(chunkSize)||chunkSize>25000)
    throw new Error('无法导出无效场景。');
  assertSceneCapacity(scene.gaussians.length);
  const header={...scene};delete header.gaussians;
  const prefix=JSON.stringify(header);
  const parts=[prefix.slice(0,-1)+(Object.keys(header).length?',':'')+'"gaussians":['];
  for(let start=0;start<scene.gaussians.length;start+=chunkSize){
    if(start)parts.push(',');
    parts.push(JSON.stringify(scene.gaussians.slice(start,start+chunkSize)).slice(1,-1));
  }
  parts.push(']}');return parts;
}

export function* iterateSceneLines(scene,{chunkSize=25000,maxLineBytes=64*1024*1024,mapGaussian=null}={}){
  if(!object(scene)||!Array.isArray(scene.gaussians)||!positiveInteger(scene.gaussians.length)||
     !positiveInteger(chunkSize)||chunkSize>25000||
     !positiveInteger(maxLineBytes)||maxLineBytes>64*1024*1024)
    throw new Error('无法导出无效的分块场景。');
  assertSceneCapacity(scene.gaussians.length);
  const header={...scene};delete header.gaussians;
  const encode=new TextEncoder();
  const first=JSON.stringify({format:SCENE_LINES_FORMAT,gaussian_count:scene.gaussians.length,scene:header});
  if(encode.encode(first).byteLength>maxLineBytes)throw new Error('场景头信息超过 JSONL 单行上限。');
  yield first+'\n';
  function* append(offset,count){
    const records=scene.gaussians.slice(offset,offset+count);
    const value=JSON.stringify({format:SCENE_CHUNK_FORMAT,offset,count,gaussians:mapGaussian?records.map(mapGaussian):records});
    if(encode.encode(value).byteLength>maxLineBytes){
      if(count===1)throw new Error('单个高斯记录超过 JSONL 单行上限，无法安全导出。');
      const half=Math.floor(count/2);yield* append(offset,half);yield* append(offset+half,count-half);return;
    }
    yield value+'\n';
  }
  for(let offset=0;offset<scene.gaussians.length;offset+=chunkSize)yield* append(offset,Math.min(chunkSize,scene.gaussians.length-offset));
}

export function sceneLinesParts(scene,options={}){
  return [...iterateSceneLines(scene,options)];
}

export async function writeSceneLines(scene,writer,{onProgress=()=>{},...options}={}){
  if(typeof writer?.write!=='function'||typeof writer?.close!=='function')throw new Error('分块导出写入器无效。');
  let lines=0;
  try{
    // Backpressure is deliberate: only construct the next JSON string after
    // the previous chunk has been written. Never accumulate a whole-file Blob.
    for(const line of iterateSceneLines(scene,{maxLineBytes:64*1024*1024-1,...options})){
      await writer.write(line);lines++;onProgress({lines});
    }
    await writer.close();return {lines};
  }catch(error){try{await writer.abort?.();}catch{}throw error;}
}

export async function readSceneLines(fileOrStream,{maxGaussians=MAX_SCENE_GAUSSIANS,maxLineBytes=64*1024*1024,onProgress=()=>{}}={}){
  if(!positiveInteger(maxGaussians)||maxGaussians>MAX_SCENE_GAUSSIANS||!positiveInteger(maxLineBytes)||maxLineBytes>64*1024*1024)
    throw new Error('JSONL 读取上限无效。');
  const stream=typeof fileOrStream?.stream==='function'?fileOrStream.stream():fileOrStream;
  if(typeof stream?.getReader!=='function')throw new Error('当前环境无法流式读取 JSONL 文件。');
  const reader=stream.getReader();let decoder=new TextDecoder('utf-8',{fatal:true});
  let lineParts=[],lineBytes=0,lineNumber=0,header=null,gaussians=null,offset=0;
  function append(bytes){
    if(!bytes.length)return;
    lineBytes+=bytes.length;
    if(lineBytes>maxLineBytes)throw new Error('JSONL 单行超过 64 MiB 上限，文件不是受支持的分块场景。');
    lineParts.push(decoder.decode(bytes,{stream:true}));
  }
  function finishLine(){
    const text=lineParts.join('')+decoder.decode();lineNumber++;
    lineParts=[];lineBytes=0;decoder=new TextDecoder('utf-8',{fatal:true});
    let value;
    try{value=JSON.parse(text);}catch{throw new Error(`JSONL 第 ${lineNumber} 行不完整或不是有效 JSON。`);}
    if(header===null){
      if(object(value))assertSceneCapacity(value.gaussian_count,maxGaussians);
      if(!object(value)||value.format!==SCENE_LINES_FORMAT||!positiveInteger(value.gaussian_count)||
         value.gaussian_count>maxGaussians||!object(value.scene)||Object.hasOwn(value.scene,'gaussians'))
        throw new Error('JSONL 场景格式、总数或头信息无效。');
      header=value;gaussians=new Array(value.gaussian_count);onProgress({loaded:0,total:value.gaussian_count});return;
    }
    if(!object(value)||value.format!==SCENE_CHUNK_FORMAT||!Number.isSafeInteger(value.offset)||
       value.offset!==offset||!positiveInteger(value.count)||value.count>25000||
       offset+value.count>header.gaussian_count||!Array.isArray(value.gaussians)||
       value.gaussians.length!==value.count||!value.gaussians.every(object))
      throw new Error('JSONL 场景分块顺序、数量或内容不一致。');
    for(let i=0;i<value.count;i++)gaussians[offset+i]=value.gaussians[i];
    offset+=value.count;
    onProgress({loaded:offset,total:header.gaussian_count});
  }
  try{
    while(true){
      const {done,value}=await reader.read();if(done)break;
      if(!(value instanceof Uint8Array))throw new Error('JSONL 流必须提供 UTF-8 字节。');
      let start=0;
      for(let i=0;i<value.length;i++)if(value[i]===10){append(value.subarray(start,i));finishLine();start=i+1;}
      append(value.subarray(start));
    }
    // Exported lines always end in a newline. A missing terminal newline is
    // treated as truncation, even if the partial last JSON happened to parse.
    if(lineBytes||lineParts.length)throw new Error('JSONL 文件末尾被截断，缺少完整的结束换行。');
    if(header===null||offset!==header.gaussian_count)throw new Error('JSONL 文件缺少场景头或完整的高斯分块。');
    return {...header.scene,gaussians};
  }catch(error){
    try{await reader.cancel(error);}catch{}
    if(error instanceof TypeError)throw new Error('JSONL 包含无效的 UTF-8 字节。');
    throw error;
  }finally{reader.releaseLock();}
}
