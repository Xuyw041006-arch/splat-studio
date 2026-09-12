// Optional saved viewing pose; it affects navigation, never reconstructed data.
export function savedViewerCamera(value){
  const vector=v=>Array.isArray(v)&&v.length===3&&v.every(x=>typeof x==='number'&&Number.isFinite(x));
  if(!value||!vector(value.position)||!vector(value.forward)||!vector(value.up))return null;
  const norm=v=>Math.hypot(...v),fn=norm(value.forward),un=norm(value.up);
  if(fn<1e-8||un<1e-8)return null;
  const forward=value.forward.map(x=>x/fn),up=value.up.map(x=>x/un);
  if(Math.abs(forward.reduce((sum,x,i)=>sum+x*up[i],0))>.999)return null;
  const fov=value.fov??(value.height&&value.intrinsics?.[1]?2*Math.atan(value.height/(2*value.intrinsics[1]))*180/Math.PI:48);
  if(typeof fov!=='number'||!Number.isFinite(fov)||fov<5||fov>150)return null;
  const distance=value.focus_distance??4;
  if(typeof distance!=='number'||!Number.isFinite(distance)||distance<=.01||distance>10000)return null;
  return {position:[...value.position],forward,up,fov,distance};
}

// Registered camera coordinates share the original PLY world coordinate system.
// Prefer a photographed direction when sparse data cannot constrain the back.
export function captureViewerCamera(cameras,center=[0,0,0]){
  if(!Array.isArray(cameras))return null;
  for(const camera of cameras){
    const m=camera?.world_to_camera,fy=camera?.intrinsics?.fy,h=camera?.height;
    if(!Array.isArray(m)||m.length!==4||m.some(row=>!Array.isArray(row)||row.length!==4||row.some(v=>typeof v!=='number'||!Number.isFinite(v))))continue;
    if(m[3].some((v,i)=>Math.abs(v-(i===3?1:0))>1e-5))continue;
    if(!Number.isFinite(fy)||fy<=0||!Number.isFinite(h)||h<=0)continue;
    let rigid=true;
    for(let i=0;i<3;i++)for(let j=0;j<3;j++)if(Math.abs(m[i].slice(0,3).reduce((s,v,k)=>s+v*m[j][k],0)-(i===j?1:0))>1e-3)rigid=false;
    const det=m[0][0]*(m[1][1]*m[2][2]-m[1][2]*m[2][1])-m[0][1]*(m[1][0]*m[2][2]-m[1][2]*m[2][0])+m[0][2]*(m[1][0]*m[2][1]-m[1][1]*m[2][0]);
    if(!rigid||Math.abs(det-1)>1e-3)continue;
    const position=[0,1,2].map(k=>-m.slice(0,3).reduce((s,row)=>s+row[k]*row[3],0));
    const forward=m[2].slice(0,3),up=m[1].slice(0,3).map(v=>-v);
    const depth=center.reduce((s,v,k)=>s+(v-position[k])*forward[k],0);
    const result=savedViewerCamera({position,forward,up,fov:2*Math.atan(h/(2*fy))*180/Math.PI,focus_distance:Number.isFinite(depth)&&depth>.05?Math.min(10000,depth):4});
    if(result)return result;
  }
  return null;
}
