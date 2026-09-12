// Camera framing only: keep every Gaussian and semantic membership untouched.
export function objectFocusBounds(records, ids){
  if(!Array.isArray(records)||!(Array.isArray(ids)||ids instanceof Set))return null;
  const targets=[...new Set([...ids].map(String))];
  if(!targets.length)return null;
  const axes=[[],[],[]];let ignoredCount=0;
  for(const record of records){
    if(typeof record?.semanticIds?.has!=='function'||!targets.some(id=>record.semanticIds.has(id)))continue;
    const p=record.position;
    if(!Array.isArray(p)||p.length!==3||!p.every(v=>typeof v==='number'&&Number.isFinite(v))){ignoredCount++;continue;}
    for(let axis=0;axis<3;axis++)axes[axis].push(p[axis]);
  }
  const count=axes[0].length;
  if(!count)return null;
  const robust=count>=64;
  const quantile=(sorted,q)=>{
    const index=(sorted.length-1)*q,low=Math.floor(index),fraction=index-low;
    if(!fraction)return sorted[low];
    return sorted[low]*(1-fraction)+sorted[low+1]*fraction;
  };
  const min=[],max=[];
  for(const axis of axes){
    axis.sort((a,b)=>a-b);
    min.push(quantile(axis,robust ? .05 : 0));
    max.push(quantile(axis,robust ? .95 : 1));
  }
  return {min,max,count,ignoredCount,robust,quantiles:robust?[.05,.95]:[0,1]};
}
