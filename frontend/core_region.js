// Non-destructive spatial visibility only. The fitting sample estimates a
// robust sphere; it is never the set of Gaussians retained for rendering/export.
export const CORE_REGION_FORMAT='splat-studio-core-region/1';
export const DEFAULT_CORE_MULTIPLIER=1.25;
const SAMPLE_LIMIT=16384;
const validPosition=p=>Array.isArray(p)&&p.length===3&&p.every(Number.isFinite);
const quantile=(sorted,q)=>{
  const index=(sorted.length-1)*q,low=Math.floor(index),fraction=index-low;
  if(!fraction)return sorted[low];
  return sorted[low]*(1-fraction)+sorted[Math.min(low+1,sorted.length-1)]*fraction;
};

export function coreRegionSettings(value=null,objects=[]){
  const initial={format:CORE_REGION_FORMAT,enabled:true,multiplier:DEFAULT_CORE_MULTIPLIER,target:'scene',object_ids:[]};
  if(value===null||value===undefined)return initial;
  if(!value||typeof value!=='object'||Array.isArray(value)||value.format!==CORE_REGION_FORMAT||
    typeof value.enabled!=='boolean'||!Number.isFinite(value.multiplier)||value.multiplier<.25||value.multiplier>4||
    !['scene','objects'].includes(value.target)||!Array.isArray(value.object_ids)||value.object_ids.length>100)
    throw new Error('核心区域显示设置无效；完整高斯数据未修改。');
  const known=new Set(objects.map(o=>String(o.id)));
  const ids=[...new Set(value.object_ids.map(String))].filter(id=>known.has(id));
  return {...initial,enabled:value.enabled,multiplier:value.multiplier,target:value.target==='objects'&&ids.length?'objects':'scene',object_ids:value.target==='objects'?ids:[]};
}

export function fitCoreRegion(records,objectIds=[]){
  if(!Array.isArray(records))throw new Error('核心区域缺少高斯记录。');
  const targets=objectIds.length?new Set(objectIds.map(String)):null;
  const eligible=record=>{
    if(!validPosition(record?.position))return false;
    if(!targets)return true;
    for(const id of targets)if(record.semanticIds?.has(id))return true;
    return false;
  };
  let total=0;for(const record of records)if(eligible(record))total++;
  if(!total)return null;
  const sampleCount=Math.min(SAMPLE_LIMIT,total),positions=new Float64Array(sampleCount*3),axis=new Float64Array(sampleCount);
  let ordinal=0,selected=0,next=0;
  for(const record of records){
    if(!eligible(record))continue;
    if(ordinal===next){
      positions.set(record.position,selected*3);selected++;
      next=sampleCount===1?Infinity:Math.floor(selected*(total-1)/(sampleCount-1));
    }
    ordinal++;if(selected===sampleCount)break;
  }
  const center=[];
  for(let component=0;component<3;component++){
    for(let i=0;i<sampleCount;i++)axis[i]=positions[i*3+component];
    axis.sort();center.push(quantile(axis,.5));
  }
  for(let i=0;i<sampleCount;i++)axis[i]=Math.hypot(positions[i*3]-center[0],positions[i*3+1]-center[1],positions[i*3+2]-center[2]);
  axis.sort();
  // Small selections keep all centers. For large scenes, a radial 90th
  // percentile prevents a small population of distant floaters setting scale.
  const q=sampleCount>=32?.9:1,radius=Math.max(1e-5,quantile(axis,q));
  if(!center.every(Number.isFinite)||!Number.isFinite(radius))throw new Error('核心区域的位置范围超出有限数值表示，无法安全过滤。');
  return {center,radius,eligible_count:total,sample_count:sampleCount,fit_quantile:q};
}

export function insideCoreRegion(position,region,settings){
  if(!settings?.enabled||!region)return true;
  const x=position[0]-region.center[0],y=position[1]-region.center[1],z=position[2]-region.center[2];
  const radius=region.radius*settings.multiplier;
  return Number.isFinite(radius)&&radius>=0&&Math.hypot(x,y,z)<=radius;
}
