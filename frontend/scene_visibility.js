// Shared by rendering and export: a Gaussian may belong to several class unions.
// Hiding any member removes it; isolation retains the union of selected members.
export function semanticIndex(objects=[]){
  const map=new Map(objects.map(o=>[String(o.id),o]));
  const roots=objects.filter(o=>o.level==='scene'&&o.parent_id==null);
  return {map,sceneRoot:roots.length===1?String(roots[0].id):null};
}
export function gaussianMembership(g,index){
  const direct=new Set([...(g.semantic_ids||[]),...(g.semantic_path||[])].map(String));
  if(g.object_id!=null)direct.add(String(g.object_id));
  const unlabeled=![...direct].some(id=>index.map.get(id)?.level!=='scene');
  const ids=new Set(direct);
  for(const first of direct){
    let current=first;const visited=new Set();
    while(current!=null&&!visited.has(current)){
      visited.add(current);ids.add(current);
      const parent=index.map.get(current)?.parent_id;current=parent==null?null:String(parent);
    }
  }
  // A single explicit scene root owns the whole scene, including unlabeled splats.
  if(index.sceneRoot!=null)ids.add(index.sceneRoot);
  return {semanticIds:ids,unlabeled};
}
export function membershipVisible(record,hiddenIds,isolationIds=null,sourceFilter='all'){
  if(record.hidden||record.unlabeled&&hiddenIds.has('__unlabeled__'))return false;
  if(sourceFilter!=='all'&&(record.source||'unknown')!==sourceFilter)return false;
  for(const id of record.semanticIds)if(hiddenIds.has(id))return false;
  if(isolationIds!==null){for(const id of record.semanticIds)if(isolationIds.has(id))return true;return false;}
  return true;
}
export function readIsolation(scene,objects){
  const ids=scene.metadata?.editor_visibility?.isolated_ids;
  if(!Array.isArray(ids))return null;
  const valid=new Set(objects.map(o=>String(o.id)));
  return new Set(ids.map(String).filter(id=>valid.has(id)));
}
export function visibilitySnapshot(hiddenIds,isolationIds){
  return {hidden:[...hiddenIds],isolated:isolationIds===null?null:[...isolationIds]};
}
export function restoreVisibility(snapshot){
  return {hidden:new Set(snapshot.hidden),isolation:snapshot.isolated===null?null:new Set(snapshot.isolated)};
}
export function sceneVisibilityExport(scene,hiddenIds,isolationIds){
  const index=semanticIndex(scene.objects||[]);
  const objectHidden=id=>{
    let current=String(id);const visited=new Set();
    while(current!=null&&!visited.has(current)){
      if(hiddenIds.has(current))return true;
      visited.add(current);const parent=index.map.get(current)?.parent_id;current=parent==null?null:String(parent);
    }
    return false;
  };
  return {scene:{...scene,
    objects:(scene.objects||[]).map(o=>({...o,visible:!objectHidden(o.id)})),
    metadata:{...scene.metadata,editor_visibility:{isolated_ids:isolationIds===null?null:[...isolationIds]}}
  },mapGaussian:g=>({...g,hidden:hiddenIds.size?!membershipVisible({...g,...gaussianMembership(g,index)},hiddenIds):!!g.hidden})};
}
export function sceneWithVisibility(scene,hiddenIds,isolationIds){
  const prepared=sceneVisibilityExport(scene,hiddenIds,isolationIds);
  return {...prepared.scene,gaussians:scene.gaussians.map(prepared.mapGaussian)};
}
