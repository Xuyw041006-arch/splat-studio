import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { semanticIndex, gaussianMembership, membershipVisible, readIsolation } from './scene_visibility.js';
import { gaussianSHShader, gaussianLoadPlan, packSHTexture, validateSH } from './gaussian_sh.js';
import { savedViewerCamera, captureViewerCamera } from './viewer_camera.js';
import { objectFocusBounds } from './object_focus.js';
import { assertSceneCapacity } from './scene_limits.js';
import { sortDepthIndices } from './depth_sort.js';
import { coreRegionSettings, fitCoreRegion, insideCoreRegion } from './core_region.js';
const DEFAULT_SCALE=[.015,.015,.015],DEFAULT_COLOR=[.7,.8,.72],DEFAULT_ROTATION=[1,0,0,0];

function normalizedVector(values,fallback,min,max,absolute=false){
  const source=values||fallback;
  // PLY/validated chunk vectors are already normalized. Reuse them instead of
  // allocating millions of identical position/scale/color arrays on the heap.
  if(source.length===3&&source.every(v=>Number.isFinite(v)&&v>=min&&v<=max))return source;
  return [0,1,2].map(i=>Math.max(min,Math.min(max,absolute?Math.abs(Number(source[i]))||fallback[i]:Number(source[i])||0)));
}

// Each instance is a screen-space ellipse obtained by projecting its full 3D
// Gaussian covariance. Back-to-front alpha compositing follows the camera.
const vertexShader = `
precision highp float;
attribute vec3 center;
attribute vec3 splatScale;
attribute vec4 quaternion;
attribute vec4 rgba;
attribute float highlight;
uniform vec2 viewport;
uniform float nearClip;
varying vec2 gaussianUV;
varying vec4 splatColor;
${gaussianSHShader}
vec3 rotateQ(vec3 v, vec4 q) { return v + 2.0 * cross(q.xyz, cross(q.xyz, v) + q.w * v); }
void main() {
  vec4 p = modelViewMatrix * vec4(center, 1.0);
  if (p.z >= -nearClip || rgba.a < 0.001) { gl_Position = vec4(2.0,2.0,2.0,1.0); gaussianUV=vec2(4.0); splatColor=vec4(0.0); return; }
  vec3 a = mat3(modelViewMatrix) * rotateQ(vec3(splatScale.x, 0., 0.), quaternion);
  vec3 b = mat3(modelViewMatrix) * rotateQ(vec3(0., splatScale.y, 0.), quaternion);
  vec3 c = mat3(modelViewMatrix) * rotateQ(vec3(0., 0., splatScale.z), quaternion);
  vec2 focal = vec2(projectionMatrix[0][0], projectionMatrix[1][1]) * viewport * 0.5;
  float invZ = 1.0 / -p.z;
  // Match the original rasterizer's computeCov2D field-of-view guard.
  // Only the covariance Jacobian is bounded; the center and stored Gaussian
  // remain unchanged. Off-screen centers must not stretch across the viewport.
  vec2 slopeLimit = 1.3 / vec2(projectionMatrix[0][0], projectionMatrix[1][1]);
  vec2 slope = clamp(p.xy * invZ, -slopeLimit, slopeLimit);
  vec3 jx = vec3(focal.x * invZ, 0., focal.x * slope.x * invZ);
  vec3 jy = vec3(0., focal.y * invZ, focal.y * slope.y * invZ);
  vec2 pa = vec2(dot(jx,a),dot(jy,a));
  vec2 pb = vec2(dot(jx,b),dot(jy,b));
  vec2 pc = vec2(dot(jx,c),dot(jy,c));
  float xx = pa.x*pa.x + pb.x*pb.x + pc.x*pc.x + 0.3;
  float yy = pa.y*pa.y + pb.y*pb.y + pc.y*pc.y + 0.3;
  float xy = pa.x*pa.y + pb.x*pb.y + pc.x*pc.y;
  float halfTrace = 0.5 * (xx + yy);
  float delta = sqrt(max(0., 0.25 * (xx-yy)*(xx-yy) + xy*xy));
  float lambda1 = max(0.1, halfTrace + delta);
  float lambda2 = max(0.1, halfTrace - delta);
  vec2 major = abs(xy) > 0.0001 ? normalize(vec2(xy, lambda1-xx)) : (xx >= yy ? vec2(1.,0.) : vec2(0.,1.));
  vec2 minor = vec2(-major.y, major.x);
  gaussianUV = position.xy * 3.0;
  vec2 offset = major * min(sqrt(lambda1), 2048.) * gaussianUV.x + minor * min(sqrt(lambda2),2048.) * gaussianUV.y;
  vec4 clip = projectionMatrix * p;
  clip.xy += offset / viewport * 2.0 * clip.w;
  gl_Position = clip;
  vec3 appearance = rgba.rgb;
  if (shInfo.x >= 0.0) {
    vec3 direction = (modelMatrix * vec4(center,1.0)).xyz - cameraPosition;
    direction = dot(direction,direction) > 0.000000000001 ? normalize(direction) : vec3(0.0,0.0,1.0);
    appearance = evaluateGaussianSH(direction, int(shInfo.x), int(shInfo.y));
  }
  splatColor = vec4(mix(appearance, vec3(0.74, 0.96, 0.81), highlight * 0.22), rgba.a);
}`;
const fragmentShader = `
precision highp float;
varying vec2 gaussianUV;
varying vec4 splatColor;
void main() {
  float exponent = dot(gaussianUV,gaussianUV);
  if (exponent > 9.0) discard;
  float alpha = min(0.99, splatColor.a * exp(-0.5 * exponent));
  if (alpha < 0.0039) discard;
  gl_FragColor = vec4(splatColor.rgb * alpha, alpha);
}`;

export class GaussianViewer {
  constructor(canvas, {onPick = () => {}, onStats = () => {}, onError = () => {}} = {}) {
    this.canvas=canvas; this.onPick=onPick; this.onStats=onStats; this.onError=onError;
    this.renderer=new THREE.WebGLRenderer({canvas,alpha:true,antialias:false,powerPreference:'high-performance'});
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
    this.renderer.setClearColor(0x101819,0);
    this.scene=new THREE.Scene();
    this.camera=new THREE.PerspectiveCamera(48,1,0.01,10000);
    this.camera.position.set(5,3.5,6.5);
    this.controls=new OrbitControls(this.camera,canvas);
    this.controls.enableDamping=true; this.controls.dampingFactor=.075;
    this.controls.target.set(0,0.8,0); this.controls.minDistance=.05; this.controls.maxDistance=1000;
    this.controls.addEventListener('change',()=>{this.sortDirty=true;});
    this.grid=new THREE.GridHelper(30,60,0x3e6350,0x345645);
    this.grid.material.transparent=true; this.grid.material.opacity=.23;
    this.grid.material.depthWrite=false; this.grid.renderOrder=-1;
    this.scene.add(this.grid);
    this.records=[]; this.objects=[]; this.objectMap=new Map(); this.hiddenIds=new Set();
    this.sourceFilter='all'; this.isolationIds=null; this.selectedId=null; this.sortDirty=false; this.lastSort=0;
    this.bounds=new THREE.Box3(); this.lastStats=0; this.frameCount=0; this.disposed=false;
    this.resizeObserver=new ResizeObserver(()=>this.resize()); this.resizeObserver.observe(canvas.parentElement);
    this.resize();
    this.pointerDown=null;
    canvas.addEventListener('pointerdown',e=>{if(e.button===0)this.pointerDown={x:e.clientX,y:e.clientY};});
    canvas.addEventListener('pointerup',e=>{if(this.pointerDown && Math.hypot(e.clientX-this.pointerDown.x,e.clientY-this.pointerDown.y)<5)this.pick(e); this.pointerDown=null;});
    canvas.addEventListener('webglcontextlost',e=>{e.preventDefault();this.onError('图形上下文已丢失，请重新打开应用或减小场景规模。');});
    this.animate=this.animate.bind(this); requestAnimationFrame(this.animate);
  }
  resize(){
    const width=Math.max(1,this.canvas.parentElement.clientWidth),height=Math.max(1,this.canvas.parentElement.clientHeight);
    this.renderer.setSize(width,height,false); this.camera.aspect=width/height; this.camera.updateProjectionMatrix();
    if(this.material)this.material.uniforms.viewport.value.set(width*this.renderer.getPixelRatio(),height*this.renderer.getPixelRatio());
    this.sortDirty=true;
  }
  load(data,{maxGaussians=null}={}){
    if(!Array.isArray(data.gaussians))throw new Error('场景缺少 gaussians 数组。');
    assertSceneCapacity(data.gaussians.length);
    this.clear(); this.initialView=savedViewerCamera(data.metadata?.viewer_camera);
    // The original CUDA rasterizer excludes camera-space depth <= 0.2.
    // Keep that convention for its trained results; otherwise near-camera
    // Gaussians unseen by the trainer can cover the scene with large streaks.
    this.camera.near=data.metadata?.backend==='original_3dgs'?0.2:0.01;
    this.objects=data.objects || []; this.objectMap=new Map(this.objects.map(o=>[String(o.id),o]));
    this.hiddenIds=new Set(this.objects.filter(o=>o.visible===false||o.hidden).map(o=>String(o.id)));
    this.isolationIds=readIsolation(data,this.objects);
    const index=semanticIndex(this.objects);
    const input=data.gaussians,plan=gaussianLoadPlan(input.length,maxGaussians),shRecords=[],memberships=new Map();
    let invalid=0;
    for(const i of plan.indices()){
      const g=input[i],p=g.position;
      if(!p || p.length!==3 || !p.every(Number.isFinite)){invalid++;continue;}
      const s=g.scale || g.scaling || DEFAULT_SCALE;
      const q=g.rotation || DEFAULT_ROTATION;
      const color=g.color || DEFAULT_COLOR;
      const id=g.object_id==null?null:String(g.object_id);
      let shIndex=-1,shDegree=-1;
      if(g.sh!=null){
        try{validateSH(g.sh,g.sh_degree);}catch(error){throw new Error(`高斯 ${i}：${error.message}`);}
        shIndex=shRecords.length;shDegree=g.sh_degree;shRecords.push(g);
      }
      const memberKey=g.object_id==null&&!g.semantic_ids?.length&&!g.semantic_path?.length?'':JSON.stringify([g.object_id??null,g.semantic_ids||null,g.semantic_path||null]);
      let membership=memberships.get(memberKey);
      if(!membership){membership=gaussianMembership(g,index);if(memberships.size<65536)memberships.set(memberKey,membership);}
      const {semanticIds,unlabeled}=membership;
      const quaternion=[Number(q[1])||0,Number(q[2])||0,Number(q[3])||0,Number(q[0])||0],length=Math.hypot(...quaternion);
      if(length<.1)quaternion.splice(0,4,0,0,0,1);else for(let k=0;k<4;k++)quaternion[k]/=length;
      this.records.push({position:p,scale:normalizedVector(s,DEFAULT_SCALE,.00001,1000,true),quaternion,color:normalizedVector(color,DEFAULT_COLOR,0,1),opacity:Math.min(.995,Math.max(0,Number(g.opacity??1))),id,semanticIds,unlabeled,source:g.source || 'unknown',hidden:!!g.hidden,confidence:g.semantic_confidence??g.confidence??null,index:i,shIndex,shDegree});
      this.bounds.min.x=Math.min(this.bounds.min.x,p[0]);this.bounds.max.x=Math.max(this.bounds.max.x,p[0]);
      this.bounds.min.y=Math.min(this.bounds.min.y,p[1]);this.bounds.max.y=Math.max(this.bounds.max.y,p[1]);
      this.bounds.min.z=Math.min(this.bounds.min.z,p[2]);this.bounds.max.z=Math.max(this.bounds.max.z,p[2]);
    }
    const n=this.records.length;
    this.coreSettings=coreRegionSettings(data.metadata?.viewer_core_region,this.objects);
    this.sceneCoreRegion=fitCoreRegion(this.records);
    this.initialView ||= captureViewerCamera(data.cameras,this.sceneCoreRegion?.center);
    this.coreWarnings=[];this.coreRegion=this.sceneCoreRegion;
    if(this.coreSettings.target==='objects'){
      const region=fitCoreRegion(this.records,this.coreSettings.object_ids);
      if(region)this.coreRegion=region;
      else{this.coreSettings={...this.coreSettings,target:'scene',object_ids:[]};this.coreWarnings.push('保存的物品核心没有对应高斯，已明确切换为场景核心。');}
    }
    if(this.coreRegion&&!Number.isFinite(this.coreRegion.radius*this.coreSettings.multiplier))throw new Error('核心区域半径超出有限数值范围。');
    const shBank=packSHTexture(shRecords,this.renderer?.capabilities?.maxTextureSize??4096);
    this.shTexture=new THREE.DataTexture(shBank.data,shBank.width,shBank.height,THREE.RGBAFormat,THREE.FloatType);
    this.shTexture.internalFormat='RGBA32F';this.shTexture.minFilter=THREE.NearestFilter;this.shTexture.magFilter=THREE.NearestFilter;
    this.shTexture.generateMipmaps=false;this.shTexture.colorSpace=THREE.NoColorSpace;this.shTexture.needsUpdate=true;
    const geometry=new THREE.InstancedBufferGeometry();
    geometry.setAttribute('position',new THREE.Float32BufferAttribute([-1,-1,0,1,-1,0,1,1,0,-1,1,0],3));
    geometry.setIndex([0,1,2,0,2,3]);
    this.arrays={center:new Float32Array(n*3),splatScale:new Float32Array(n*3),quaternion:new Float32Array(n*4),rgba:new Float32Array(n*4),highlight:new Float32Array(n),shInfo:new Float32Array(n*2)};
    this.sortOrder=new Uint32Array(n);this.sortScratch=new Uint32Array(n);this.sortDepth=new Float64Array(n);
    for(const [name,array] of Object.entries(this.arrays))geometry.setAttribute(name,new THREE.InstancedBufferAttribute(array,name==='highlight'?1:name==='shInfo'?2:name==='rgba'||name==='quaternion'?4:3).setUsage(THREE.DynamicDrawUsage));
    this.material=new THREE.ShaderMaterial({vertexShader,fragmentShader,uniforms:{viewport:{value:new THREE.Vector2()},nearClip:{value:this.camera.near},shTexture:{value:this.shTexture},shTextureWidth:{value:shBank.width},shCoefficientStride:{value:shBank.coefficientStride}},transparent:true,depthTest:true,depthWrite:false,blending:THREE.CustomBlending,blendSrc:THREE.OneFactor,blendDst:THREE.OneMinusSrcAlphaFactor,blendEquation:THREE.AddEquation,side:THREE.DoubleSide});
    this.mesh=new THREE.Mesh(geometry,this.material); this.mesh.frustumCulled=false; this.mesh.renderOrder=2; this.scene.add(this.mesh);
    this.grid.position.y=this.bounds.isEmpty()?-.04:this.bounds.min.y-.03;
    if(!this.bounds.isEmpty()){const size=this.bounds.getSize(new THREE.Vector3()).length();this.grid.scale.setScalar(Math.max(.05,size/10));}
    this.resetView(false); this.resize(); this.sortDirty=true; this.sortSplats();
    return {loaded:n,total:input.length,invalid,sampled:plan.sampled,maxGaussians:plan.maxGaussians,
      omittedByBudget:plan.omittedByBudget,shGaussians:shRecords.length,shTextureBytes:shBank.bytes,
      attributeBytes:n*68,sortBytes:n*16,estimatedGPUBufferBytes:shBank.bytes+n*68,coreWarnings:this.coreWarnings};
  }
  clear(){
    if(this.mesh){this.scene.remove(this.mesh);this.mesh.geometry.dispose();this.material.dispose();this.mesh=null;this.material=null;}
    if(this.shTexture){this.shTexture.dispose();this.shTexture=null;}
    this.arrays=null;this.sortOrder=null;this.sortScratch=null;this.sortDepth=null;
    this.records=[];this.objects=[];this.objectMap.clear();this.hiddenIds.clear();this.bounds.makeEmpty();this.selectedId=null;
    this.sourceFilter='all';this.isolationIds=null;this.visibleCount=0;
    this.coreSettings=null;this.coreRegion=null;this.sceneCoreRegion=null;this.coreIncluded=0;this.coreWarnings=[];
  }
  recordVisible(r){return insideCoreRegion(r.position,this.coreRegion,this.coreSettings)&&membershipVisible(r,this.hiddenIds,this.isolationIds,this.sourceFilter);}
  sortSplats(){
    if(!this.mesh)return;
    this.camera.updateMatrixWorld();const m=this.camera.matrixWorldInverse.elements;
    const records=this.records,depth=this.sortDepth,order=this.sortOrder;let count=0,coreIncluded=0;
    for(let i=0;i<records.length;i++){
      const r=records[i];if(!insideCoreRegion(r.position,this.coreRegion,this.coreSettings))continue;coreIncluded++;
      if(!membershipVisible(r,this.hiddenIds,this.isolationIds,this.sourceFilter))continue;
      depth[i]=m[2]*r.position[0]+m[6]*r.position[1]+m[10]*r.position[2]+m[14];order[count++]=i;
    }
    const visible=order.subarray(0,count);sortDepthIndices(visible,depth,this.sortScratch);
    const a=this.arrays;
    for(let i=0;i<count;i++){
      const r=records[visible[i]],j=i*4;a.center.set(r.position,i*3);a.splatScale.set(r.scale,i*3);a.quaternion.set(r.quaternion,j);
      a.rgba[j]=r.color[0];a.rgba[j+1]=r.color[1];a.rgba[j+2]=r.color[2];a.rgba[j+3]=r.opacity;
      a.highlight[i]=this.selectedId&&r.semanticIds.has(this.selectedId)?1:0;a.shInfo[i*2]=r.shIndex;a.shInfo[i*2+1]=r.shDegree;
    }
    for(const name of Object.keys(a))this.mesh.geometry.attributes[name].needsUpdate=true;
    this.mesh.geometry.instanceCount=count;this.visibleCount=count;this.coreIncluded=coreIncluded;this.sortDirty=false;this.lastSort=performance.now();
  }
  getCoreSettings(){return this.coreSettings?{...this.coreSettings,object_ids:[...this.coreSettings.object_ids]}:coreRegionSettings();}
  setCoreSettings(value){
    const next=coreRegionSettings(value,this.objects),previous=this.coreSettings;let region=this.coreRegion;
    if(!previous||previous.target!==next.target||previous.object_ids.length!==next.object_ids.length||previous.object_ids.some((id,i)=>id!==next.object_ids[i])){
      region=next.target==='objects'?fitCoreRegion(this.records,next.object_ids):this.sceneCoreRegion;
      if(!region&&next.target==='objects')throw new Error('所选物品没有可用于确定核心区域的高斯。');
    }
    if(region&&!Number.isFinite(region.radius*next.multiplier))throw new Error('核心区域半径超出有限数值范围。');
    this.coreRegion=region;this.coreSettings=next;this.sortDirty=true;
  }
  setHidden(ids){this.hiddenIds=new Set([...ids].map(String));this.sortDirty=true;}
  setIsolation(ids){this.isolationIds=ids===null?null:new Set([...ids].map(String));this.sortDirty=true;}
  setSourceFilter(source){this.sourceFilter=source;this.sortDirty=true;}
  select(id){this.selectedId=id==null?null:String(id);this.sortDirty=true;}
  setGrid(visible){this.grid.visible=visible;}
  getObjectStats(id){
    let count=0,inferred=0,unknown=0,confidenceSum=0,confidenceCount=0;id=String(id);
    for(const r of this.records){if(!r.semanticIds.has(id))continue;count++;if(r.source==='inferred')inferred++;else if(r.source!=='observed')unknown++;
      if(Number.isFinite(r.confidence)){confidenceSum+=r.confidence;confidenceCount++;}}
    return {count,inferred,unknown,confidence:confidenceCount?confidenceSum/confidenceCount:null};
  }
  resetView(smooth=true){
    // OrbitControls captures the up-axis at construction. A saved dataset pose
    // needs new controls, including when returning from an unrelated scene.
    this.controls.dispose();this.camera.up.fromArray(this.initialView?.up||[0,1,0]);
    this.camera.fov=this.initialView?.fov||48;this.controls=new OrbitControls(this.camera,this.canvas);
    this.controls.enableDamping=true;this.controls.dampingFactor=.075;this.controls.minDistance=.05;this.controls.maxDistance=1000;
    this.controls.addEventListener('change',()=>{this.sortDirty=true;});this.tween=null;
    if(this.initialView){
      const v=this.initialView;this.tween=null;
      this.camera.position.fromArray(v.position);this.camera.up.fromArray(v.up);this.camera.fov=v.fov;
      this.controls.target.copy(this.camera.position).addScaledVector(new THREE.Vector3(...v.forward),v.distance);
      this.camera.updateProjectionMatrix();this.controls.update();this.sortDirty=true;return;
    }
    const box=this.bounds.isEmpty()?new THREE.Box3(new THREE.Vector3(-2,0,-2),new THREE.Vector3(2,2,2)):this.bounds;
    this.frameBox(box,smooth,true);
  }
  focus(ids){
    const bounds=objectFocusBounds(this.records,ids);
    if(bounds)this.frameBox(new THREE.Box3(new THREE.Vector3(...bounds.min),new THREE.Vector3(...bounds.max)),true,false);
  }
  frameBox(box,smooth,resetDirection){
    const center=box.getCenter(new THREE.Vector3()),size=box.getSize(new THREE.Vector3());
    const radius=Math.max(.15,size.length()/2);
    const fov=Math.min(this.camera.fov*Math.PI/180,2*Math.atan(Math.tan(this.camera.fov*Math.PI/360)*this.camera.aspect));
    const distance=radius/Math.sin(fov/2)*1.22;
    const dir=resetDirection?new THREE.Vector3(1,.65,1.1).normalize():this.camera.position.clone().sub(this.controls.target).normalize();
    const position=center.clone().addScaledVector(dir,distance);
    if(smooth)this.tween={start:performance.now(),fromPosition:this.camera.position.clone(),fromTarget:this.controls.target.clone(),position,target:center};
    else{this.camera.position.copy(position);this.controls.target.copy(center);this.controls.update();}
    this.controls.maxDistance=Math.max(100,distance*8);this.camera.far=Math.max(1000,distance*20);this.camera.updateProjectionMatrix();
  }
  pick(event){
    if(!this.records.length)return;
    const rect=this.canvas.getBoundingClientRect(),x=event.clientX-rect.left,y=event.clientY-rect.top;
    const v=new THREE.Vector3();let best=null,bestScore=Infinity;
    for(const r of this.records){
      if(!r.id||!this.recordVisible(r)||r.opacity<.05)continue;
      v.set(...r.position).project(this.camera);if(v.z< -1||v.z>1)continue;
      const sx=(v.x+1)*rect.width/2,sy=(1-v.y)*rect.height/2;
      const distance=Math.hypot(sx-x,sy-y);
      if(distance<17){const score=distance+v.z*4;if(score<bestScore){bestScore=score;best=r.id;}}
    }
    if(best)this.onPick(best);
  }
  animate(now){
    if(this.disposed)return;
    requestAnimationFrame(this.animate);
    if(this.tween){const t=Math.min(1,(now-this.tween.start)/520),e=1-Math.pow(1-t,3);this.camera.position.lerpVectors(this.tween.fromPosition,this.tween.position,e);this.controls.target.lerpVectors(this.tween.fromTarget,this.tween.target,e);if(t>=1)this.tween=null;this.sortDirty=true;}
    this.controls.update();
    if(this.sortDirty && now-this.lastSort>65)this.sortSplats();
    this.renderer.render(this.scene,this.camera);this.frameCount++;
    if(now-this.lastStats>1200){this.onStats({fps:Math.round(this.frameCount*1000/(now-this.lastStats)),visible:this.visibleCount||0,total:this.records.length,coreIncluded:this.coreIncluded,coreEnabled:!!this.coreSettings?.enabled});this.frameCount=0;this.lastStats=now;}
  }
  dispose(){this.disposed=true;this.resizeObserver.disconnect();this.controls.dispose();this.clear();this.grid.geometry.dispose();this.grid.material.dispose();this.renderer.dispose();}
}
