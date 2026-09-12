// Product navigation is independent of training and never loads a test scene.
export function importRequirements({geometry, semantics, geometryOnly=false}) {
  if (!geometry) return '请选择高斯点云或完整场景文件。';
  if (!/\.(ply|json|jsonl)$/i.test(geometry.name)) return '高斯点云使用 PLY；完整场景使用 JSON 或 JSONL。';
  if (/\.ply$/i.test(geometry.name) && !geometryOnly && !semantics) return '请同时选择与该点云配套的语义包 ZIP。';
  if (semantics && !/\.zip$/i.test(semantics.name)) return '语义包需要为 ZIP 文件。';
  if (semantics && !/\.ply$/i.test(geometry.name)) return '独立语义包需要与原始 PLY 配对；完整场景已经包含语义。';
  return null;
}

export function initSceneFlow({isBusy,onNew,onImport,onResume,hasScene}) {
  const root=document.querySelector('#app');
  const home=document.querySelector('#scene-home');
  const importer=document.querySelector('#scene-import');
  const setPhase=phase=>{
    if(!['home','new','import','editor'].includes(phase))throw new Error('Unknown scene workflow');
    root.dataset.workflow=phase;home.hidden=phase!=='home';importer.hidden=phase!=='import';
    document.querySelector('#resume-scene').hidden=!hasScene();
    requestAnimationFrame(()=>window.dispatchEvent(new Event('resize')));
  };
  const guarded=fn=>()=>{if(!isBusy())fn();};
  document.querySelector('#new-scene').onclick=guarded(()=>{onNew();setPhase('new');});
  document.querySelector('#existing-scene').onclick=guarded(()=>{setPhase('import');showImportType();});
  document.querySelector('#home-button').onclick=guarded(()=>setPhase('home'));
  document.querySelector('.brand').onclick=event=>{event.preventDefault();if(!isBusy())setPhase('home');};
  document.querySelector('#import-back').onclick=guarded(()=>{if(document.querySelector('#import-type-step').hidden)showImportType();else setPhase('home');});
  document.querySelector('#resume-scene').onclick=guarded(()=>{onResume();setPhase('editor');});
  const geometry=document.querySelector('#scene-geometry-file');
  const semantic=document.querySelector('#scene-semantic-file');
  const only=document.querySelector('#geometry-only');
  const submit=document.querySelector('#import-scene-submit');
  const status=document.querySelector('#import-scene-status');
  function showImportType(){document.querySelector('#import-type-step').hidden=false;document.querySelector('#import-files-step').hidden=true;}
  function openImport(mode){
    setPhase('import');only.checked=mode==='rgb';geometry.value='';semantic.value='';
    document.querySelector('#import-type-step').hidden=true;document.querySelector('#import-files-step').hidden=false;
    document.querySelector('#import-files-heading').textContent=only.checked?'导入三维外观':'导入可识别物品的场景';
    document.querySelector('#import-files-description').textContent=only.checked?'选择完整场景文件或高斯点云（JSONL、JSON、PLY）。':'选择完整场景 JSONL / JSON；若使用 PLY，请同时选择配套语义 ZIP。';refresh();
  }
  document.querySelector('#import-rgb-choice').onclick=()=>openImport('rgb');
  document.querySelector('#import-semantic-choice').onclick=()=>openImport('semantic');
  function refresh(){
    const integrated=!!geometry.files[0]&&!/\.ply$/i.test(geometry.files[0].name);
    semantic.disabled=only.checked||integrated;
    document.querySelector('#semantic-file-field').hidden=integrated||only.checked;
    status.textContent=geometry.files[0]?(importRequirements({geometry:geometry.files[0],semantics:semantic.disabled?null:semantic.files[0],geometryOnly:only.checked})||'文件已选好。'):'';
    submit.disabled=isBusy()||!!importRequirements({geometry:geometry.files[0],semantics:semantic.disabled?null:semantic.files[0],geometryOnly:only.checked});
  }
  for(const el of [geometry,semantic,only])el.onchange=refresh;
  submit.onclick=async()=>{
    if(isBusy())return;
    submit.disabled=true;status.textContent='正在校验并导入完整场景…';
    try{await onImport(geometry.files[0],semantic.disabled?null:semantic.files[0],{geometryOnly:only.checked});setPhase('editor');}
    catch(error){status.textContent=error.message;}
    finally{submit.disabled=false;}
  };
  setPhase('home');refresh();
  return {setPhase,refresh,openImport};
}
