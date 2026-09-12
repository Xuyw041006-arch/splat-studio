export function nameError(value){
  const name=String(value||'').trim();
  if(!name)return '先给场景起个名字。';
  if(name.length>64||/[<>:"/\\|?*\x00-\x1f]/.test(name)||name.endsWith('.'))return '最多 64 个字，请勿使用 / \\ : * ? 等文件名符号。';
  if(/^(CON|PRN|AUX|NUL|COM[1-9]|LPT[1-9])(?:\..*)?$/i.test(name))return '请换一个名称，这个名称由系统保留。';
  return '';
}
export function sceneFilename(value){
  const name=String(value||'场景').trim().replace(/[<>:"/\\|?*\x00-\x1f]/g,'_').replace(/[. ]+$/,'').slice(0,64)||'场景';
  return `${nameError(name)?'场景-'+name:name}.splat.jsonl`;
}
export function nextSetupStep(step,semantics){
  return ({photos:'analysis',analysis:'semantics',semantics:semantics?'priority':'settings',priority:'settings',settings:'confirm'})[step]||step;
}
export function previousSetupStep(step,semantics){
  return ({analysis:'photos',semantics:'analysis',priority:'semantics',settings:semantics?'priority':'semantics',confirm:'settings',cloud:'confirm',training:'confirm'})[step]||'photos';
}
export function photoRecommendation(analysis,viewSpan=null){
  if(!analysis)return null;
  const n=analysis.unique_image_count??analysis.image_count;
  if(n<2)return {kind:'blocked',title:'还需要不同视角的照片',message:'重复照片无法增加视角，请至少选择两张不同照片。'};
  const weak=analysis.connected_ratio<.75;
  if(n<=2||weak)return {kind:'completion',title:'建议稀疏重建，并尝试视角补全',message:n<=2?'只有两个视角，建议补充有照片依据的可见区域。未拍到的背面仍可能缺失。':'照片之间的重叠不足，建议补充几何；也可以返回添加相邻视角。'};
  if(n<12||viewSpan!==null&&viewSpan<90)return {kind:'sparse',title:'建议使用稀疏重建',message:'照片数量或拍摄范围较少，但已有重叠。先使用这些照片重建；需要时再检查补全。'};
  return {kind:'direct',title:'可以直接重建',message:'照片数量和重叠满足常规重建的初步条件。'};
}
export function importAsRgb(scene){
  const metadata={...scene.metadata};
  for(const k of ['editor_visibility','semantic_refinement','semantic_refinement_artifacts','semantic_model','semantic_classes','semantic_hierarchy'])delete metadata[k];
  metadata.semantic_view='rgb';
  return {...scene,objects:[],metadata,gaussians:scene.gaussians.map(g=>{
    const copy={...g};for(const k of ['object_id','semantic_ids','semantic_path','semantic_probabilities','semantic_confidence'])delete copy[k];return copy;
  })};
}
export function hasObjectSemantics(scene){return Array.isArray(scene?.objects)&&scene.objects.some(o=>o.level!=='scene');}
