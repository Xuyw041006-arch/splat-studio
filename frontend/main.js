import {installI18n} from './i18n.js';
import { initSettings, sparsePanelState, semanticProfilePreview, geometryOptionsForPhotoSet, sceneDisplayInfo } from './settings.js';
import { GaussianViewer } from './viewer.js';
import { readIsolation, visibilitySnapshot, restoreVisibility, sceneVisibilityExport } from './scene_visibility.js';
import { estimateBasisText, trainingEtaText } from './task_progress.js';
import { fetchSceneResource, sceneJsonParts, readSceneLines, writeSceneLines } from './scene_transport.js';
import { initSceneFlow } from './scene_flow.js';
import { initCreationWizard } from './creation_wizard.js';
import { nameError,sceneFilename,importAsRgb,hasObjectSemantics } from './user_workflow.js';
import { initPriorityFlow } from './priority_flow.js';
import { createCoreRegionControls } from './core_region_controls.js';
import { trainingPriorityLabel, trainingPriorityLabels } from './training_priorities.js';
import { executionConfig, executionControls, cloudPackageDownload, requestCloudPackage } from './execution_target.js';

const $=selector=>document.querySelector(selector);
const $$=selector=>[...document.querySelectorAll(selector)];
const iconPaths={upload:'M12 16V3m0 0L7 8m5-5l5 5M4 14v6h16v-6',download:'M12 3v13m0 0l5-5m-5 5l-5-5M4 17v4h16v-4',sparkles:'M12 3l2.5 6.5L21 12l-6.5 2.5L12 21l-2.5-6.5L3 12l6.5-2.5L12 3zM20 2v4m-2-2h4',bolt:'M13 2L4 14h7l-1 8 10-12h-7l1-8z',sliders:'M4 6h16M4 12h16M4 18h16M8 3v6M16 9v6M10 15v6',diamond:'M12 3l9 9-9 9-9-9 9-9zM12 3l4 9-4 9-4-9 4-9zM3 12h18',orbit:'M21 12c0 5-4 9-9 9s-9-4-9-9 4-9 9-9m9 0l-9 9m4-9h5v5',cpu:'M8 8h8v8H8zM5 5h14v14H5zM8 2v3m8-3v3M8 19v3m8-3v3M2 8h3m-3 8h3m14-8h3m-3 8h3',plus:'M12 5v14M5 12h14',scan:'M8 3H3v5m13-5h5v5M3 16v5h5m8 0h5v-5M7 7h10v10H7z',play:'M8 4l12 8-12 8V4z',clock:'M21 12a9 9 0 1 1-18 0 9 9 0 0 1 18 0zM12 7v5l3 2',route:'M5 3v13a5 5 0 0 0 10 0V8m-4 4l4-4 4 4M3 3h4',grid:'M3 3h18v18H3zM3 9h18M3 15h18M9 3v18M15 3v18',maximize:'M8 3H3v5m13-5h5v5M3 16v5h5m8 0h5v-5',help:'M21 12a9 9 0 1 1-18 0 9 9 0 0 1 18 0zM9.1 9a3 3 0 0 1 5.8 1c0 2-3 2-3 4m.1 3h0',cube:'M12 3l9 5v9l-9 5-9-5V8l9-5zM3 8l9 5 9-5M12 13v9M8 5l9 5',mouse:'M8 2h8a3 3 0 0 1 3 3v11a7 7 0 0 1-14 0V5a3 3 0 0 1 3-3zM12 2v6',focus:'M8 3H3v5m13-5h5v5M3 16v5h5m8 0h5v-5M16 12a4 4 0 1 1-8 0 4 4 0 0 1 8 0z',eye:'M2 12s3.5-7 10-7 10 7 10 7-3.5 7-10 7S2 12 2 12zM15 12a3 3 0 1 1-6 0 3 3 0 0 1 6 0z',eyeOff:'M3 3l18 18M10 5c7-1 12 7 12 7a17 17 0 0 1-3 4M7 6a19 19 0 0 0-5 6s4 7 10 7c1 0 3-.3 4-1M10 10a3 3 0 0 0 4 4',arrowUp:'M12 19V5m0 0l-6 6m6-6l6 6',search:'M16 10a6 6 0 1 1-12 0 6 6 0 0 1 12 0zM15 15l6 6',layers:'M12 3l10 5-10 5L2 8l10-5zM2 12l10 5 10-5M2 16l10 5 10-5',undo:'M3 8h11a7 7 0 0 1 0 14M3 8l5-5M3 8l5 5',isolate:'M7 3H3v4m14-4h4v4M3 17v4h4m10 0h4v-4M8 8h8v8H8z',close:'M6 6l12 12M18 6L6 18',chevron:'M7 10l5 5 5-5',folder:'M3 6h7l2 2h9v12H3V6z'};
function icon(name){return `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="${iconPaths[name]||iconPaths.cube}"/></svg>`;}
function hydrateIcons(root=document){root.querySelectorAll('[data-icon]').forEach(el=>{if(el.querySelector('svg'))return;el.insertAdjacentHTML('afterbegin',icon(el.dataset.icon));});}
hydrateIcons();
const state={sceneName:'',files:[],fileUrls:[],projectId:null,analysis:null,plan:null,sparseContext:null,semanticAnalysis:null,renderQuality:null,inventory:[],priorities:new Set(),mode:'balanced',busy:false,jobId:null,pollTimer:null,scene:null,sceneUrl:null,ownedRun:null,objects:[],selectedId:null,hidden:new Set(),isolation:null,history:[],collapsed:new Set(),source:'all',health:null};
let viewer,coreControls,sceneFlow,priorityFlow,wizard;
try{viewer=new GaussianViewer($('#scene-canvas'),{onPick:selectObject,onStats:stats=>{const q=state.renderQuality;$('#renderer-status').textContent=`WebGL · ${stats.fps} FPS${stats.total?` · ${formatNumber(stats.visible)} 可见高斯`:''}${q?` · 已载入 ${formatNumber(q.loaded)} / ${formatNumber(q.source)} · ${q.color}`:''}`;renderCoreControls();},onError:message=>toast(message,true)});}catch(error){toast(`无法初始化 WebGL：${error.message}`,true);$('#renderer-status').textContent='WebGL 不可用';}
const BASE=window.SPLAT_BACKEND_URL||'';
async function api(path,options={}){
  const url=/^https?:/.test(path)?path:`${BASE}${path}`;
  let response;
  try{response=await fetch(url,options);}catch(error){throw new Error(options.body instanceof FormData?'照片读取或上传失败，请重新选择文件，并确认本地引擎仍在运行。':'无法连接本地重建引擎，请检查后端是否已启动。');}
  let data;
  try{data=await response.json();}catch{throw new Error(`引擎返回无法读取的响应 (${response.status})。`);}
  if(!response.ok){let detail=data.detail||data.message||data.error||`请求失败 (${response.status})`;if(typeof detail!=='string')detail=JSON.stringify(detail);throw new Error(detail);}
  return data;
}
const jsonOptions=data=>({method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(data)});
function toast(message,error=false){const el=document.createElement('div');el.className=`toast${error?' error':''}`;el.textContent=message;$('#toast-container').append(el);setTimeout(()=>el.remove(),error?8500:4300);}
function escapeHtml(value){return String(value??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));}
function formatNumber(n){return Number(n||0).toLocaleString('zh-CN');}
function duration(seconds){if(seconds<60)return `${Math.max(1,Math.round(seconds))} 秒`;if(seconds<3600)return `${Math.ceil(seconds/60)} 分钟`;return `${(seconds/3600).toFixed(1).replace(/\.0$/,'')} 小时`;}
function estimateText(plan){
  if(!plan)return '分析素材后显示';
  const raw=plan.estimated_seconds??plan.estimate_seconds??plan.estimated_time_seconds??plan.estimated_time??plan.estimate;
  let min,max;
  if(Array.isArray(raw)){[min,max]=raw;}else if(raw && typeof raw==='object'){min=raw.min??raw.minimum??raw.low;max=raw.max??raw.maximum??raw.high;}else if(typeof raw==='number'){min=raw;max=raw;}else if(typeof raw==='string')return raw;
  min??=plan.estimated_seconds_min??plan.eta_min_seconds;max??=plan.estimated_seconds_max??plan.eta_max_seconds;
  if(min==null&&plan.estimated_minutes!=null){if(Array.isArray(plan.estimated_minutes)){min=plan.estimated_minutes[0]*60;max=plan.estimated_minutes[1]*60;}else{min=max=plan.estimated_minutes*60;}}
  if(min==null && max==null)return '当前设备尚无基准';
  min??=max;max??=min;return `${Number(min)===Number(max)?duration(min):`${duration(min)}–${duration(max)}`} · 粗估`;
}
function selectedExecution(){return executionConfig($('#execution-target').value,$('#cloud-provider').value,$('#device').value);}
function executionView(){const selected=selectedExecution();return executionControls({target:selected.execution_target,provider:selected.cloud_provider,device:selected.device,photoCount:state.files.length,busy:state.busy,plan:state.plan,health:state.health,ownedRun:state.ownedRun});}
function config(){const angle=Number($('#view-span').value);return {...selectedExecution(),mode:state.mode,semantics:$('#semantics').checked,view_span:$('#view-span').value.trim()===''?null:Math.max(0,Math.min(360,angle)),semantic_strategy:$('#semantic-strategy').value,semantic_refinement:'cuda_iterative',semantic_steps:Number($('#semantic-steps')?.value||1200),semantic_budget:$('#semantic-budget')?.value||'mode',semantic_view_mode:$('#semantic-view-mode')?.value||'auto',semantic_granularity:'multilevel',priority_objects:trainingPriorityLabels(state.priorities,[...state.inventory,...state.objects]),completion:$('#completion-mode')?.value||'auto',geometry_backend:$('#geometry-backend')?.value||'auto',sparse_completion:$('#sparse-completion')?.value||'auto',allow_geometry_download:$('#allow-geometry-download')?.checked||false,provider:$('#semantic-provider')?.value||'local',allow_model_download:$('#allow-model-download')?.checked||false,allow_remote_images:$('#allow-remote-images')?.checked||false,manual_labels:($('#candidate-labels')?.value||'').split(/[,，]/).map(s=>s.trim()).filter(Boolean),candidate_labels:($('#candidate-labels')?.value||'').trim()?$('#candidate-labels').value.split(/[,，]/).map(s=>s.trim()).filter(Boolean):null,...priorityFlow?.config(),...wizard?.config(),...(!$('#semantics').checked?{priority_request:'',priority_objects:[],priority_approval_sha256:'',priority_refinement:'weighted'}:{})};}
async function ensureProject(){
  if(state.files.length<2)throw new Error('请先添加至少两张照片。');
  const invalid=nameError(state.sceneName);if(invalid)throw new Error(invalid);
  if(!state.projectId){const body=new FormData();body.append('scene_title',state.sceneName);state.files.forEach(file=>body.append('files',file));const result=await api('/api/projects',{method:'POST',body});state.projectId=result.id||result.project_id;}
  await api(`/api/projects/${state.projectId}/name`,{...jsonOptions({name:state.sceneName}),method:'PUT'});
  return state.projectId;
}
function clearCloudPackage(){if(state.cloudPackage&&$('#execution-target').value==='cloud')$('#setup-status').textContent='照片、设置或标注已变化，请重新导出云端任务包。';state.cloudPackage=null;$('#cloud-package-result').hidden=true;$('#cloud-package-download').removeAttribute('href');}
function refreshExecutionControls(){
  const view=executionView(),cloud=view.cloud;
  $('#local-device-field').hidden=cloud;$('#device-detail').hidden=cloud;$('#cloud-execution-panel').hidden=!cloud;
  $('#device').disabled=state.busy||cloud;$('#execution-target').disabled=state.busy;$('#cloud-provider').disabled=state.busy;
  $('#cloud-execution-hint').textContent=$('#cloud-provider').value==='colab'?'任务包下载到本机后，由你上传 Colab 并按说明运行。尚未连接远程 GPU，导出不会开始训练。':'将任务包传到你自己的 NVIDIA GPU 服务器，按包内说明运行，再下载结果导入。当前没有直连服务器。';
  $('#reconstruct-button').innerHTML=icon(cloud?'download':'play')+view.primaryLabel;
  $('#reconstruct-button').hidden=!cloud&&!state.plan;$('#reconstruct-button').disabled=(cloud?!view.canExport:!view.canLocalReconstruct)||($('#semantics').checked&&priorityFlow&&!priorityFlow.ready());
  $('#analyze-button').innerHTML=icon('scan')+(cloud?'分析照片 · 可选':'分析照片');
  $('#analyze-button').classList.toggle('primary',!cloud);$('#analyze-button').classList.toggle('secondary',cloud);
  if(cloud)$('#estimate').textContent=view.estimateLabel;
  const geometry=state.health?.capabilities?.learned_geometry;
  $('#geometry-status').textContent=cloud?'几何模型及权重将在云端运行时校验；本机资源状态不代表云端。':geometry?.available?'DUSt3R 资源就绪 · 推断时校验权重与几何':geometry?.source_available?'几何代码已安装；需要本地权重或允许首次下载。':'几何模型未安装：运行 scripts/setup_geometry.py 后再使用自动补全。';
  $('#completion-mode').querySelector('[value="learned"]').disabled=cloud||!state.health?.capabilities?.learned_completion;
  $('#semantic-provider').querySelector('[value="local"]').textContent=cloud?'云端 Grounding DINO + SAM':'本地 Grounding DINO + SAM';
  $('#model-download-label').textContent=cloud?'允许云端运行时首次下载识别模型':'允许首次下载本地模型（约 1 GB）';
  $('#recognition-location-hint').textContent=cloud?'照片分析仅检查输入与视角；自动识别在云端运行任务时进行。任务包保留照片与所选设置。':'本地识别无需上传照片。首次使用需要下载模型，后续使用缓存。';
  $('#allow-model-download').disabled=state.busy||$('#semantic-provider').value!=='local';
  if(cloud)$('#engine-status').textContent='云端训练待手动执行 · 本机负责导出与查看';
  else if(state.health){const device=state.health.device?.name||state.health.device?.type||state.health.device||state.health.capabilities?.recommended_device||'自动';$('#engine-status').textContent=`${typeof device==='string'?device.toUpperCase():'本地计算设备'} · 本地引擎就绪`;}
  else $('#engine-status').textContent='本地引擎未连接';
}
function refreshSemanticControls(){
  const choice=$('#semantic-refinement'),button=$('#semantic-refine-button');if(!choice||!button)return;
  const execution=executionView(),available=execution.iterativeSelectable;
  choice.querySelector('[value="cuda_iterative"]').disabled=!available;choice.disabled=state.busy;
  const canRefine=execution.canRefineCurrent;
  $('#semantic-steps').disabled=state.busy||$('#semantic-budget').value!=='manual'||(!canRefine&&(!available||choice.value!=='cuda_iterative'));
  const profile=semanticProfilePreview(state.mode,$('#semantic-budget').value,$('#semantic-steps').value,$('#semantic-view-mode').value,$('#semantic-granularity').value);
  $('#semantic-profile-hint').textContent=`${profile.views==='all'?'全部训练视角':`最多 ${profile.limit} 个抽样视角`} · ${profile.steps} 步基础预算 · ${profile.granularity==='multilevel'?'多粒度层级匹配':'平级语义'}。为覆盖有效训练对，实际步数可能增加。`; 
  button.disabled=state.busy||!canRefine;
  $('#semantic-refinement-hint').textContent=execution.cloud?'云端重建后执行层级匹配、跨视角置信度与多轮语义优化。':'每个档位均执行层级和跨视角置信度处理；完整语义训练需要 NVIDIA CUDA，本机不可用时请选择云端。';
  $('#semantic-refine-hint').textContent=execution.cloud?'当前云端导出用于所选照片的新重建任务。已有模型的独立语义训练需要完整模型、原照片、相机和掩码；此按钮不会在本机代为执行。':!available?'当前设备不能对已有模型执行实验性 CUDA 多轮语义优化。':!state.ownedRun?'仅支持本应用已完成的 CUDA 运行；独立导入的 JSON/PLY 不具备原项目训练数据。':`使用当前运行的完整模型与已缓存掩码，以 ${profile.steps} 步为基础预算进行实验性语义优化；RGB 保持不变，生成新运行。轮数增加不保证精度提升。`;
}
function updateSparsePanel(){
  const panel=$('#sparse-settings');if(!panel)return;
  const context=state.analysis&&state.plan?{analysis:state.analysis,plan:state.plan}:state.sparseContext;
  const display=sparsePanelState(context?.analysis,context?.plan);
  panel.hidden=!display.visible;
  $('#sparse-settings-title').textContent=display.title;
  $('#sparse-reason').textContent=(display.visible&&!state.analysis?'当前照片上次分析：':'')+display.reason;
  // Hiding a recommendation never resets the user's choices or consents.
}
function resetGeometryForPhotos(){
  state.sparseContext=null;
  const options=geometryOptionsForPhotoSet({});
  for(const [selector,key] of [['#geometry-backend','geometry_backend'],['#sparse-completion','sparse_completion'],['#completion-mode','completion']]){
    if($(selector))$(selector).value=options[key];
  }
  // Download consent is a user preference and is not changed with the photos.
}
function renderSemanticAnalysis(){
  const element=$('#semantic-analysis-detail');if(!element)return;
  const detail=state.semanticAnalysis;element.hidden=!$('#semantics').checked||!detail;
  if(element.hidden)return;
  const coverage=detail.coverage,profile=detail.profile,timing=detail.timing;
  $('#semantic-coverage-text').textContent=coverage?`已分析 ${coverage.analyzed_count} / ${coverage.selected_count} 个所选视角；可用输入 ${coverage.eligible_count} 个，${coverage.masked_count} 个视角有像素掩码。${coverage.pixel_mask_supervision_available?'':'当前仅有候选名称或尚未得到掩码，不能据此进行像素监督。'}`:'当前识别方式没有自动生成像素监督，请使用已有掩码或手动标注。';
  const seconds=value=>typeof value==='number'&&Number.isFinite(value)?`${value.toFixed(2)} 秒`:'未记录';
  $('#semantic-timing-text').textContent=timing?`语义准备实测 ${seconds(timing.total_seconds)} · 识别 ${seconds(timing.discovery_seconds)} · 保存掩码 ${seconds(timing.mask_write_seconds)}。不含后续高斯与语义优化。`:'语义准备耗时尚未记录。';
  $('#semantic-analysis-profile').textContent=profile?`多轮优化预算：${profile.semantic_steps} 步起，监督长边 ${profile.train_size} 像素；${profile.semantic_view_mode==='all'?'全部训练视角':'抽样训练视角'}，${profile.semantic_granularity==='multilevel'?'多粒度层级匹配':'平级语义'}。覆盖有效训练对所需步数可能更高。`:'';
}
function setBusy(busy){state.busy=busy;$('#command-input').disabled=busy;$('#command-send').disabled=busy||!state.objects.length;$('#analyze-button').disabled=busy||state.files.length<2;$('#reconstruct-button').disabled=busy||state.plan?.can_reconstruct===false;$('#reanalyze-button').disabled=busy;$('#home-button').disabled=busy;$('#photo-input').disabled=busy;$('#clear-photos').disabled=busy;$('#add-photos').disabled=busy;$$('.quality-option').forEach(b=>b.disabled=busy);['#semantics','#device','#view-span','#semantic-strategy','#completion-mode','#geometry-backend','#sparse-completion','#allow-geometry-download','#camera-template','#camera-import','#semantic-provider','#candidate-labels','#allow-model-download','#allow-remote-images','#manual-mask-button','#semantic-budget','#semantic-view-mode','#semantic-granularity'].forEach(s=>{if($(s))$(s).disabled=busy;});if($('#import-button'))$('#import-button').disabled=busy;refreshSemanticControls();refreshExecutionControls();renderCoreControls();priorityFlow?.refresh();wizard?.sync();}
function invalidateAnalysis(){state.analysis=null;state.plan=null;state.semanticAnalysis=null;clearCloudPackage();updateSparsePanel();$('#analysis-summary').hidden=true;$('#inventory-section').hidden=true;$('#analyze-button').hidden=false;$('#reconstruct-button').hidden=true;$('#reanalyze-button').hidden=true;$('#estimate').textContent='分析素材后显示';$('#setup-status').textContent=state.files.length<2?'请选择至少 2 张具有重叠区域的照片':executionView().cloud?'可先检查照片，也可直接导出云端任务；导出不会开始训练。':'先分析照片，再确定适合的重建策略';$('#setup-status').classList.remove('error');setBusy(state.busy);}
function clearScene(){priorityFlow?.result(null);state.renderQuality=null;state.scene=null;state.sceneUrl=null;state.ownedRun=null;state.objects=[];state.hidden.clear();state.isolation=null;state.history=[];state.selectedId=null;viewer?.clear();$('#welcome-overlay').hidden=false;$('#scene-chips').hidden=true;$('#viewport-mode').hidden=true;$('#selection-chip').hidden=true;$('#object-detail').hidden=true;$('#export-button').disabled=true;$('#command-send').disabled=true;$('#scene-metadata').textContent='本地工作空间';renderTree();refreshSemanticControls();coreControls?.clear();}
function renderCoreControls(){
  if(!state.scene||!viewer){coreControls?.clear();return;}
  const quality=state.renderQuality;
  coreControls?.render({settings:viewer.getCoreSettings(),objects:state.objects,selectedObject:state.objects.find(o=>String(o.id)===state.selectedId),
    loaded:quality?.loaded??viewer.records.length,total:quality?.source??viewer.records.length,visible:viewer.visibleCount,coreIncluded:viewer.coreIncluded,busy:state.busy,pending:viewer.sortDirty});
  if(quality)$('#splat-count').textContent=`可见 ${formatNumber(viewer.visibleCount)} · 已载入 ${formatNumber(quality.loaded)} / 来源 ${formatNumber(quality.source)}`;
}
function changeCoreSettings(changes){
  if(state.busy||!state.scene||!viewer)return;
  const previous=viewer.getCoreSettings();
  try{viewer.setCoreSettings({...previous,...changes});snapshot(previous);renderCoreControls();}
  catch(error){toast(error.message,true);renderCoreControls();}
}
coreControls=createCoreRegionControls({parent:$('.viewer-panel'),before:$('#canvas-wrap'),
  onToggle:enabled=>changeCoreSettings({enabled}),onRange:multiplier=>changeCoreSettings({multiplier}),
  onScene:()=>changeCoreSettings({enabled:true,target:'scene',object_ids:[]}),
  onSelected:()=>{if(state.selectedId)changeCoreSettings({enabled:true,target:'objects',object_ids:[state.selectedId]});},
  onAll:()=>{if(!state.busy)applyAction('show_all');}});
function sceneReadProgress({loaded,total}){$('#scene-metadata').textContent=`正在完整载入 ${formatNumber(loaded)} / ${formatNumber(total)} 个高斯及原始球谐系数…`;}
async function fetchViewerScene(url){
  // Release the preceding scene's JS references, CPU arrays and GPU textures
  // before accumulating the next million-point model on a 16 GiB machine.
  clearScene();
  return fetchSceneResource(url,api,{onProgress:sceneReadProgress});
}
function addFiles(files){
  if(state.busy){toast('当前任务正在处理，请等待完成后再更换照片。');return;}
  const valid=[...files].filter(f=>/^image\/(jpeg|png|webp)$/.test(f.type)||/\.(jpe?g|png|webp)$/i.test(f.name));
  if(valid.length<files.length)toast('已跳过不支持的文件。请选择 JPG、PNG 或 WEBP 照片。',true);
  const keys=new Set(state.files.map(f=>`${f.name}:${f.size}:${f.lastModified}`));let added=0;
  for(const f of valid){const key=`${f.name}:${f.size}:${f.lastModified}`;if(!keys.has(key)){keys.add(key);state.files.push(f);state.fileUrls.push(URL.createObjectURL(f));added++;}}
  if(added){state.projectId=null;resetGeometryForPhotos();state.priorities.clear();clearScene();invalidateAnalysis();renderPhotos();$('#project-title').textContent=state.sceneName||'新建场景';$('#project-title').toggleAttribute('data-i18n-ignore',!!state.sceneName);wizard?.photosChanged();}
}
function renderPhotos(){
  $('#camera-settings').hidden=state.files.length<2;
  $('#photo-count').textContent=`${state.files.length} 张照片`;
  $('#dropzone').classList.toggle('has-photos',state.files.length>0);$('#photo-actions').hidden=!state.files.length;
  const grid=$('#photo-grid');grid.replaceChildren();
  state.files.slice(0,11).forEach((file,index)=>{const div=document.createElement('div');div.className='photo-thumb';const img=document.createElement('img');img.src=state.fileUrls[index];img.alt=file.name;img.title=file.name;const remove=document.createElement('button');remove.type='button';remove.textContent='×';remove.title=`移除 ${file.name}`;remove.setAttribute('aria-label',remove.title);remove.onclick=()=>{if(state.busy)return;URL.revokeObjectURL(state.fileUrls[index]);state.files.splice(index,1);state.fileUrls.splice(index,1);state.projectId=null;resetGeometryForPhotos();invalidateAnalysis();renderPhotos();wizard?.invalidateCapture();};div.append(img,remove);grid.append(div);});
  if(state.files.length>11){const extra=document.createElement('div');extra.className='photo-more';extra.textContent=`+${state.files.length-11}`;grid.append(extra);}
  $('#analyze-button').disabled=state.files.length<2||state.busy;priorityFlow?.refresh();
}
$('#photo-input').addEventListener('change',e=>{addFiles(e.target.files);e.target.value='';});
$('#add-photos').onclick=()=>$('#photo-input').click();
$('#clear-photos').onclick=()=>{if(state.busy)return;state.fileUrls.forEach(u=>URL.revokeObjectURL(u));state.files=[];state.fileUrls=[];state.projectId=null;resetGeometryForPhotos();state.priorities.clear();invalidateAnalysis();renderPhotos();$('#project-title').textContent=state.sceneName||'新建场景';$('#project-title').toggleAttribute('data-i18n-ignore',!!state.sceneName);wizard?.invalidateCapture();};
$('#dropzone').addEventListener('keydown',e=>{if(e.key==='Enter'||e.key===' '){e.preventDefault();$('#photo-input').click();}});
$('#dropzone').addEventListener('dragover',e=>{e.preventDefault();$('#dropzone').classList.add('drag-over');});
$('#dropzone').addEventListener('dragleave',()=>$('#dropzone').classList.remove('drag-over'));
$('#dropzone').addEventListener('drop',e=>{e.preventDefault();$('#dropzone').classList.remove('drag-over');addFiles(e.dataTransfer.files);});
window.addEventListener('dragover',e=>e.preventDefault());window.addEventListener('drop',e=>e.preventDefault());
$$('.quality-option').forEach(button=>button.onclick=()=>{state.mode=button.dataset.mode;$$('.quality-option').forEach(b=>{b.classList.toggle('selected',b===button);b.setAttribute('aria-checked',b===button?'true':'false');});invalidateAnalysis();});
$('#semantics').onchange=()=>{$('#semantic-settings').hidden=!$('#semantics').checked;invalidateAnalysis();};
$('#device').onchange=invalidateAnalysis;$('#view-span').onchange=invalidateAnalysis;$('#execution-target').onchange=invalidateAnalysis;$('#cloud-provider').onchange=invalidateAnalysis;
$('#semantic-strategy').onchange=()=>{$('#strategy-hint').textContent=$('#semantic-strategy').value==='joint'?'联合 RGB 与语义训练尚未实现。':'在重建完成后处理语义；类别包含候选不等于已验证的物理部件。';invalidateAnalysis();};
async function analyze(){
  if(state.files.length<2)return;
  setBusy(true);$('#setup-status').textContent='读取照片并评估视角覆盖…';$('#setup-status').classList.remove('error');
  try{
    await ensureProject();
    const result=await api(`/api/projects/${state.projectId}/analyze`,jsonOptions({...config(),semantics:false,priority_request:'',priority_objects:[]}));
    state.analysis=result.analysis||result;state.plan=result.plan||state.analysis.plan||{};state.inventory=[...(result.objects||state.analysis.objects||[]),...(state.manualProject===state.projectId?state.manualInventory||[]:[])];
    state.sparseContext={analysis:state.analysis,plan:state.plan};state.semanticAnalysis={coverage:result.semantic_coverage,timing:result.semantic_timing,profile:result.semantic_profile};
    renderAnalysis(result);
    $('#analyze-button').hidden=true;$('#reconstruct-button').hidden=false;$('#reanalyze-button').hidden=false;
    $('#setup-status').textContent=executionView().cloud?'照片预检完成。导出任务后，在云端运行；尚未验证远端 GPU 或开始训练。':state.plan.can_reconstruct===false?'当前素材或模型条件不足，请按提示补拍或配置几何模型。':sparsePanelState(state.analysis,state.plan).visible?'检测到少视角素材，请查看补全建议，再开始重建。':'素材分析完成，可调整重点物品后开始重建';
    return result;
  }catch(error){toast(error.message,true);$('#setup-status').textContent=error.message;$('#setup-status').classList.add('error');return null;}
  finally{setBusy(false);}
}
$('#analyze-button').onclick=analyze;$('#reanalyze-button').onclick=analyze;
const strategyNames={two_view_completion:'双视角 · 几何验证与自动补全',sparse_regularized:'稀疏视角重建',standard_3dgs:'标准多视角 3DGS',sparse:'稀疏视角重建',sparse_completion:'稀疏重建 + 补全',completion:'极少视角 · 补全优先',dense:'多视角精细重建',standard:'标准多视角重建',two_view:'双视角重建 + 补全',regular:'标准多视角重建'};
function renderAnalysis(result){
  const plan=state.plan,analysis=state.analysis;
  $('#analysis-summary').hidden=false;updateSparsePanel();renderSemanticAnalysis();
  const strategy=plan.strategy||plan.reconstruction_strategy||plan.pipeline||'自动重建方案';
  $('#plan-name').textContent=plan.label||plan.name||strategyNames[strategy]||strategy;
  const pieces=[];
  const desc=plan.description||plan.reason||analysis.summary||analysis.description;
  if(desc)pieces.push(typeof desc==='string'?desc:JSON.stringify(desc));
  if(executionView().cloud)pieces.push('执行位置：云端任务包 · GPU 尚未连接或验证');else if(plan.recommended_backend||plan.backend)pieces.push(`计算路径：${plan.recommended_backend||plan.backend}`);
  if(analysis.unique_image_count!=null)pieces.push(`独立照片 ${analysis.unique_image_count} / ${analysis.image_count}`);
  if(sparsePanelState(analysis,plan).visible){
    if(executionView().cloud)pieces.push('稀疏几何与补全将在云端运行时验证');
    else if(plan.automatic_completion_status)pieces.push(`自动补全：${({ready:'模型就绪，将按几何结果检查',unavailable:'模型未就绪',not_requested:'关闭',automatic_if_needed:'按需触发'})[plan.automatic_completion_status]||plan.automatic_completion_status}`);
  }
  if(analysis.registered_views!=null)pieces.push(`有效视角 ${analysis.registered_views} 个`);
  $('#plan-description').textContent=pieces.join(' · ')||`已分析 ${state.files.length} 张照片，具体重建进度将由引擎实时返回。`;
  $('#estimate').textContent=executionView().cloud?'云端运行后估计':estimateText(plan);
  $('#estimate-basis').textContent=executionView().cloud?'尚未确定远端 GPU、输入传输与模型准备耗时；不使用本机 MPS/CPU 基准推算云端时间。':estimateBasisText(plan);
  const warnings=[...(Array.isArray(result.warnings)?result.warnings:[]),...(Array.isArray(plan.warnings)?plan.warnings:[]),...(Array.isArray(analysis.warnings)?analysis.warnings:[])];
  $('#plan-warnings').replaceChildren();[...new Set(warnings.map(w=>typeof w==='string'?w:w.message||JSON.stringify(w)))].forEach(w=>{const p=document.createElement('p');p.textContent=w;$('#plan-warnings').append(p);});
  $('#inventory-section').hidden=!$('#semantics').checked;
  $('#inventory-list').replaceChildren();
  state.inventory.forEach((item,index)=>{const id=String(item.id??item.label??index),key=trainingPriorityLabel(item);const label=document.createElement('label');label.className='inventory-item';const input=document.createElement('input');input.type='checkbox';input.checked=state.priorities.has(id)||state.priorities.has(key);input.disabled=!key;input.onchange=()=>{state.priorities.delete(id);state.priorities.delete(key);if(input.checked&&key)state.priorities.add(key);clearCloudPackage();};label.append(input,document.createTextNode(item.label||item.name||id));$('#inventory-list').append(label);});
  $('#inventory-note').textContent=state.inventory.length?(result.object_source||analysis.object_source||'候选物品需由用户确认；跨视角关联后形成统一语义标识。'):'尚未得到可靠物品候选。可配置视觉模型后重新分析；无可靠语义时不会虚构物品。';
}
function stageLabel(stage){return {queued:'等待计算资源',preparing:'准备场景',analyzing:'分析照片',features:'提取与匹配特征',sfm:'恢复相机位姿',colmap:'恢复相机位姿',reconstructing:'优化三维高斯',training:'优化三维高斯',semantics:'融合跨视角语义',segmenting:'理解场景物品',completion:'检查并补充推断几何',exporting:'导出高斯场景',completed:'重建完成',failed:'重建未完成',cancelled:'任务已取消'}[stage]||stage||'重建中';}
async function exportCloudTask(override=null){
  if(!executionView().canExport)return;
  setBusy(true);clearCloudPackage();$('#setup-status').textContent='正在本机整理照片与配置，生成云端任务包…';$('#setup-status').classList.remove('error');
  try{
    await ensureProject();
    const {result}=await requestCloudPackage({api,files:state.files,projectId:state.projectId,config:override||config(),onProject:id=>{state.projectId=id;}});
    const download=cloudPackageDownload(result,BASE);state.cloudPackage=result;
    const link=$('#cloud-package-download');link.href=download.url;link.download=download.filename;
    $('#cloud-package-message').textContent=download.message;
    $('#cloud-package-sha').textContent=`SHA256 ${download.sha256}`;
    $('#cloud-package-command').textContent=result.worker_command_hint?`运行命令：${result.worker_command_hint}`:'按任务包内 README 的步骤安装环境并运行。';
    $('#cloud-package-result').hidden=false;link.click();wizard?.cloudPrepared(override?.priority_task_stage==='review');
    $('#setup-status').textContent=override?.priority_task_stage==='review'?'识别任务已导出。在 Colab 运行后导回 priority-review.zip；确认区域后再生成重建任务。':'训练任务包已生成 · 等待在云端运行，当前没有训练进程';
    toast(override?.priority_task_stage==='review'?'物品识别任务已生成，尚未开始重建。':'训练任务包已生成，完成后导入完整 JSONL 结果。');
  }catch(error){$('#setup-status').textContent=error.message;$('#setup-status').classList.add('error');toast(error.message,true);}
  finally{setBusy(false);}
}
async function startReconstruction(){
  if(executionView().cloud)return exportCloudTask();
  if(!state.projectId||state.busy||state.plan?.can_reconstruct===false)return;
  setBusy(true);clearScene();$('#welcome-overlay').hidden=true;$('#job-overlay').hidden=false;$('#job-stage').textContent='提交重建任务';$('#job-message').textContent='准备计算资源…';$('#job-progress').style.width='0%';$('#job-progress-label').textContent='0%';$('#job-elapsed').textContent='已用时 0 秒';$('#cancel-job').disabled=false;
  lastJobTelemetry=null;renderJobEta();
  try{const job=await api(`/api/projects/${state.projectId}/jobs`,jsonOptions(config()));state.jobId=job.id||job.job_id;if(!state.jobId)throw new Error('引擎未返回任务标识。');pollJob();}catch(error){finishJobError(error.message);}
}
$('#reconstruct-button').onclick=startReconstruction;
async function pollJob(){
  if(!state.jobId)return;
  try{
    const job=await api(`/api/jobs/${state.jobId}`);
    lastJobTelemetry=job;renderJobEta();
    $('#job-stage').textContent=stageLabel(job.stage||job.status);$('#job-message').textContent=job.message||'引擎正在处理，请稍候。';
    const progress=Number(job.progress);if(Number.isFinite(progress)){$('#job-progress').style.width=`${Math.max(0,Math.min(100,progress*100))}%`;$('#job-progress-label').textContent=`${Math.round(Math.max(0,Math.min(100,progress*100)))}%`;}
    $('#job-elapsed').textContent=`已用时 ${duration(Number(job.elapsed_seconds)||0)}`;
    if(['completed','complete','succeeded','success','done'].includes(job.status)){
      if(!job.scene_url)throw new Error('任务已结束，但没有生成可加载的场景。');
      const scene=await fetchViewerScene(job.viewer_scene_url||job.scene_url);loadScene(scene,{url:job.scene_url,demo:false});$('#job-overlay').hidden=true;state.jobId=null;setBusy(false);const semanticOnly=job.kind==='semantic_refinement';$('#setup-status').textContent=`${semanticOnly?'语义优化完成':'重建完成'} · 用时 ${duration(Number(job.elapsed_seconds)||0)}`;toast(semanticOnly?'语义已更新，RGB 模型保持不变。':'场景已生成，可以旋转、缩放和编辑物品。');wizard?.completed();
    }else if(['failed','error'].includes(job.status)){finishJobError(job.error||job.message||'重建失败，请检查引擎输出。');}
    else if(['cancelled','canceled'].includes(job.status)){$('#job-overlay').hidden=true;$('#welcome-overlay').hidden=!state.scene;state.jobId=null;setBusy(false);$('#setup-status').textContent='任务已取消，已有场景与素材已保留。';toast('任务已取消。');wizard?.failed('任务已取消，可以调整设置后重新开始。');}
    else{state.pollTimer=setTimeout(pollJob,1200);}
  }catch(error){finishJobError(error.message);}
}
let lastJobTelemetry=null;
function renderJobEta(){const text=trainingEtaText(state.jobId?lastJobTelemetry:null,duration);$('#job-eta').textContent=text;$('#job-eta').hidden=!text;}
// Expire stale telemetry even when a fetch or subprocess produces no new response.
setInterval(renderJobEta,1000);
function finishJobError(message){clearTimeout(state.pollTimer);state.jobId=null;setBusy(false);$('#job-overlay').hidden=true;$('#welcome-overlay').hidden=!state.scene;$('#setup-status').textContent=String(message);$('#setup-status').classList.add('error');toast(String(message),true);wizard?.failed(String(message));}
$('#cancel-job').onclick=async()=>{if(!state.jobId)return;$('#cancel-job').disabled=true;try{await api(`/api/jobs/${state.jobId}/cancel`,{method:'POST'});$('#job-message').textContent='正在取消，等待引擎释放资源…';}catch(error){toast(error.message,true);$('#cancel-job').disabled=false;}};

function loadScene(scene,{url=null,demo=false}={}){
  if(!viewer)throw new Error('WebGL 渲染器不可用。');
  
  let result;try{result=viewer.load(scene);}catch(error){viewer.clear();if(error instanceof RangeError)throw new Error('内存不足，无法完整载入此模型。没有自动抽样或降低球谐阶数；请关闭其他大型场景后重试。');throw error;}
  state.scene=scene;state.sceneName=scene.metadata?.scene_name||state.sceneName||'场景';$('#project-title').textContent=state.sceneName;$('#project-title').setAttribute('data-i18n-ignore','');$('#app').dataset.hasSemantics=String(hasObjectSemantics(scene));state.sceneUrl=url;state.objects=scene.objects||[];state.selectedId=null;state.hidden=new Set(state.objects.filter(o=>o.visible===false||o.hidden).map(o=>String(o.id)));state.history=[];state.collapsed.clear();state.source='all';
  const grid=scene.metadata?.viewer_grid!==false;viewer.setGrid(grid);$('#grid-toggle').classList.toggle('active',grid);
  $('#command-feedback').textContent=state.objects.length?'用语言探索场景。支持查找、聚焦、隐藏、隔离与撤销。':'当前模型没有语义标识，可旋转和缩放查看。';$('#command-input').value='';
  state.isolation=readIsolation(scene,state.objects);
  $('#welcome-overlay').hidden=true;$('#scene-chips').hidden=false;$('#viewport-mode').hidden=false;$('#export-button').disabled=false;$('#command-send').disabled=!state.objects.length;
  const synthetic=demo||!!scene.metadata?.synthetic_demo||scene.metadata?.source==='demo';
  const ownUrl=typeof url==='string'?url.match(/^\/api\/projects\/([a-f0-9]{32})\/files\/runs\/([a-f0-9]{32})\/scene\.json$/):null;
  const metadata=scene.metadata||{};const cudaScene=metadata.backend==='original_3dgs'||/cuda/i.test(metadata.backend||'');
  state.ownedRun=!synthetic&&ownUrl&&metadata.project_id===ownUrl[1]&&metadata.job_id===ownUrl[2]&&cudaScene&&Array.isArray(scene.cameras)&&scene.cameras.length?{projectId:ownUrl[1],runId:ownUrl[2]}:null;
  const quality=sceneDisplayInfo(scene,result,viewer.records);state.renderQuality=quality;
  const badge=synthetic?'演示场景 · 人工生成':quality.partial?'场景载入不完整':'场景外观'; 
  $('#scene-badge').classList.toggle('demo',synthetic);$('#scene-badge').innerHTML=`<i></i>${badge}`;
  $('#scene-badge').title=`已载入 ${formatNumber(quality.loaded)} / 来源 ${formatNumber(quality.source)} 个高斯 · ${quality.color}。显示效果取决于输入模型保留的几何与颜色。`; 
  $('#splat-count').textContent=`${formatNumber(quality.loaded)} / ${formatNumber(quality.source)} GAUSSIANS`; 
  $('#scene-metadata').textContent=`${synthetic?'人工生成演示 · 非照片重建结果':scene.metadata?.backend||scene.metadata?.device||'3DGS'} · 已载入 ${formatNumber(quality.loaded)} / 来源 ${formatNumber(quality.source)} 个高斯 · ${quality.color}`;
  $('#scene-metadata').title=[scene.metadata?.source_note,...(scene.metadata?.warnings||[])].filter(Boolean).join('\n');
  const geometry=metadata.initialization?.initialization||metadata;
  if(geometry.registered_views!=null)$('#scene-metadata').textContent+=` · 相机 ${geometry.registered_views}/${geometry.input_views??geometry.registered_views}`;
  const completion=metadata.sparse_completion;
  if(completion){
    const descriptions={completed_visible_only:`已补充 ${completion.added_points||0} 个推断点`,learned_visible_initialization:'已用学习式可见几何初始化',learned_visible_initialization_after_sfm_failure:'传统几何不足，已切换学习式可见几何',unavailable:'补全模型未就绪，保留传统几何',rejected:'补全未通过对齐检查，保留传统几何',failed_preserved_observed:'补全未通过检查，保留传统几何',disabled:'自动补全已关闭',not_needed_for_input_view_count:'本次未触发补全'};
    const description=descriptions[completion.status]||`补全状态：${completion.status}`;
    $('#scene-metadata').textContent+=` · ${description}`;
    $('#scene-metadata').title+=`\n${description}；未拍摄背面保持未知。`;
    if(!['disabled','not_needed_for_input_view_count'].includes(completion.status))toast(description);
  }
  $$('.viewport-mode button').forEach(b=>b.classList.toggle('active',b.dataset.source==='all'));
  $('#object-search').value='';$('#selection-chip').hidden=true;$('#object-detail').hidden=true;$('#undo-button').disabled=true;renderTree();refreshSemanticControls();
  if(result.sampled)toast(`场景 JSON 含 ${formatNumber(result.total)} 个高斯。当前显示 ${formatNumber(result.loaded)} 个；JSON 导出保留其输入数据，完整模型以原始 PLY 为准。`);
  if(result.invalid)toast(`跳过了 ${result.invalid} 个无效高斯。`,true);
  for(const warning of result.coreWarnings||[])toast(warning,true);
  if(scene.metadata?.warnings)for(const warning of scene.metadata.warnings.filter(w=>!/^显示全部|^来源颜色表示/.test(w)).slice(0,2))toast(warning,true);
  priorityFlow?.result(scene);renderCoreControls();sceneFlow?.setPhase('editor');
  // Apply the capture pose after the editor has its final viewport dimensions.
  requestAnimationFrame(()=>{viewer.resize();viewer.resetView(false);});
}
$('#grid-toggle').onclick=()=>{const button=$('#grid-toggle');button.classList.toggle('active');viewer?.setGrid(button.classList.contains('active'));};
$('#reset-view').onclick=()=>viewer?.resetView();
const objectsToggle=document.createElement('button');objectsToggle.className='button ghost compact objects-toggle';objectsToggle.textContent='物品';objectsToggle.setAttribute('aria-label','打开物品面板');
$('#reset-view').before(objectsToggle);
const objectsClose=document.createElement('button');objectsClose.className='icon-button objects-close';objectsClose.textContent='×';objectsClose.setAttribute('aria-label','关闭物品面板');$('.objects-heading').append(objectsClose);
objectsToggle.onclick=()=>{$('.objects-panel').classList.add('drawer-open');$('#object-search').focus();};
objectsClose.onclick=()=>$('.objects-panel').classList.remove('drawer-open');
$$('.viewport-mode button').forEach(button=>button.onclick=()=>{state.source=button.dataset.source;viewer?.setSourceFilter(state.source);$$('.viewport-mode button').forEach(b=>b.classList.toggle('active',b===button));if(state.source==='inferred')toast('当前仅显示推断或补全的区域，其几何与外观未得到真实观测验证。');});
function objectMap(){return new Map(state.objects.map(o=>[String(o.id),o]));}
function descendants(ids){const result=new Set(ids.map(String));let changed=true;while(changed){changed=false;for(const o of state.objects){if(o.parent_id!=null && result.has(String(o.parent_id))&&!result.has(String(o.id))){result.add(String(o.id));changed=true;}}}return result;}
function explicitlyHidden(id){const map=objectMap();let current=String(id),seen=new Set();while(current&&!seen.has(current)){if(state.hidden.has(current))return true;seen.add(current);const p=map.get(current)?.parent_id;current=p==null?null:String(p);}return false;}
function effectivelyHidden(id){if(explicitlyHidden(id))return true;if(state.isolation===null)return false;const branch=descendants([String(id)]);return ![...branch].some(child=>state.isolation.has(child));}
function renderTree(){
  $('#object-count').textContent=String(state.objects.length);const tree=$('#object-tree');tree.replaceChildren();
  if(!state.objects.length){tree.innerHTML=`<div class="objects-empty"><span>${icon('layers')}</span><strong>空间里，不止有几何。</strong><p>${state.scene?'当前场景没有可靠语义标识。<br />启用语义理解后重新重建。':'启用语义理解后，<br />在这里探索物品与部件层级。'}</p></div>`;return;}
  const query=$('#object-search').value.trim().toLowerCase();const map=objectMap(),children=new Map();
  for(const o of state.objects){const parent=o.parent_id!=null&&map.has(String(o.parent_id))?String(o.parent_id):null;if(!children.has(parent))children.set(parent,[]);children.get(parent).push(o);}
  const matches=new Set(state.objects.filter(o=>!query||String(o.label||o.name||o.id).toLowerCase().includes(query)||(o.aliases||[]).some(a=>String(a).toLowerCase().includes(query))).map(o=>String(o.id)));
  if(query){for(const id of [...matches]){let item=map.get(id),seen=new Set();while(item?.parent_id!=null&&!seen.has(String(item.parent_id))){seen.add(String(item.parent_id));matches.add(String(item.parent_id));item=map.get(String(item.parent_id));}}}
  const colors=['#aed1b2','#cfb990','#8bb0b6','#b6a2b5','#9bae7f','#b9c8a6'];let count=0;const visited=new Set();
  function addRows(parent,depth){for(const object of children.get(parent)||[]){const id=String(object.id);if(visited.has(id)||!matches.has(id))continue;visited.add(id);count++;const hasChildren=children.has(id);const row=document.createElement('div');row.className=`object-row${state.selectedId===id?' selected':''}${effectivelyHidden(id)?' is-hidden':''}`;row.dataset.id=id;row.tabIndex=0;row.setAttribute('role','button');row.setAttribute('aria-label',`${object.label||object.name||id}${effectivelyHidden(id)?'，已隐藏':''}`);
      const inset=document.createElement('span');inset.className='object-indent';inset.style.width=`${Math.min(depth,4)*10}px`;
      const chevron=document.createElement('button');chevron.className=`object-chevron${state.collapsed.has(id)?' collapsed':''}`;chevron.setAttribute('aria-label',hasChildren?'展开或收起部件':'物体');chevron.innerHTML=hasChildren?icon('chevron'):'';chevron.disabled=!hasChildren;chevron.onclick=e=>{e.stopPropagation();state.collapsed.has(id)?state.collapsed.delete(id):state.collapsed.add(id);renderTree();};
      const dot=document.createElement('span');dot.className='object-dot';dot.style.background=object.color&&typeof object.color==='string'?object.color:colors[Math.abs(hashString(id))%colors.length];
      const label=document.createElement('span');label.className='object-label';label.textContent=object.label||object.name||id;label.title=label.textContent;
      const visibility=document.createElement('button');visibility.className='object-visibility';visibility.innerHTML=icon(effectivelyHidden(id)?'eyeOff':'eye');visibility.setAttribute('aria-label',`${effectivelyHidden(id)?'显示':'隐藏'}${label.textContent}`);visibility.onclick=e=>{e.stopPropagation();applyAction(effectivelyHidden(id)?'show':'hide',[id]);};
      row.append(inset,chevron,dot,label);if(hasChildren){const number=document.createElement('span');number.className='object-sub-count';number.textContent=String(children.get(id).length);row.append(number);}row.append(visibility);row.onclick=()=>selectObject(id);row.onkeydown=e=>{if(e.key==='Enter'||e.key===' '){e.preventDefault();selectObject(id);}};tree.append(row);
      if(hasChildren&&(!state.collapsed.has(id)||query))addRows(id,depth+1);
  }}addRows(null,0);
  if(!count)tree.innerHTML='<p class="no-results">没有找到匹配的物品</p>';
}
function hashString(string){let n=0;for(const c of string)n=(n*31+c.charCodeAt(0))|0;return n;}
function selectObject(id){
  id=String(id);const object=objectMap().get(id);if(!object)return;
  state.selectedId=id;viewer?.select(id);$('#selection-chip').hidden=false;$('#selection-label').textContent=object.label||object.name||id;$('#object-detail').hidden=false;$('#object-detail-name').textContent=object.label||object.name||id;
  const stats=viewer.getObjectStats(id);const consistent=object.cross_view_confidence;$('#object-confidence').textContent=consistent!=null?`跨视角可信度 ${Math.round(consistent*100)}%`:stats.confidence==null?'置信度未知':`模型分数 ${Math.round(stats.confidence*100)}%`;
  $('#object-detail-stats').textContent=`${formatNumber(stats.count)} 个高斯 · ${stats.unknown===stats.count&&stats.count?'观测来源未知':stats.unknown?`${formatNumber(stats.inferred)} 个推断 / ${formatNumber(stats.unknown)} 个来源未知`:`${stats.count?Math.round(stats.inferred/stats.count*100):0}% 推断区域`}`;
  renderTree();renderCoreControls();
}
$('#object-search').addEventListener('input',renderTree);
function snapshot(core=viewer?.getCoreSettings()){state.history.push({...visibilitySnapshot(state.hidden,state.isolation),core,source:state.source});if(state.history.length>60)state.history.shift();$('#undo-button').disabled=false;}
function applyAction(action,ids=[],command={}){
  ids=ids.map(String);const map=objectMap();ids=ids.filter(id=>map.has(id));
  if(action==='undo'){if(!state.history.length){toast('没有可以撤销的场景操作。');return false;}const saved=state.history.pop();Object.assign(state,restoreVisibility(saved));if(saved.core)viewer?.setCoreSettings(saved.core);state.source=saved.source||'all';viewer?.setSourceFilter(state.source);}
  else if(action==='focus'){if(ids.length){selectObject(ids[0]);viewer?.focus(ids);}return;}
  else if(action==='search'){if(ids.length){selectObject(ids[0]);viewer?.focus(ids);}return;}
  else if(action==='priority'){const labels=ids.map(id=>trainingPriorityLabel(map.get(id))).filter(Boolean);if(command.priority===false){priorityFlow?.reset();toast('已清除待训练的重点配置。');return true;}if(labels.length){priorityFlow?.startFromObjects(labels);toast('重点目标已带入重建流程，请匹配原照片中的区域。');}else toast('没有匹配到可用于重建的物品。',true);return !!labels.length;}
  else if(action==='hide'||action==='delete'){snapshot();ids.forEach(id=>state.hidden.add(id));}
  else if(action==='show'){snapshot();for(const id of descendants(ids)){state.hidden.delete(id);state.isolation?.add(id);let item=map.get(id),seen=new Set();while(item?.parent_id!=null&&!seen.has(String(item.parent_id))){const p=String(item.parent_id);seen.add(p);state.hidden.delete(p);item=map.get(p);}}}
  else if(action==='isolate'){if(!ids.length)return;snapshot();state.hidden.clear();state.isolation=descendants(ids);}
  else if(action==='show_all'){snapshot();state.hidden.clear();state.isolation=null;state.source='all';viewer?.setSourceFilter('all');if(viewer)viewer.setCoreSettings({...viewer.getCoreSettings(),enabled:false});}
  else{toast('当前尚不支持这条操作。',true);return;}
  viewer?.setHidden(state.hidden);viewer?.setIsolation(state.isolation);renderTree();renderCoreControls();$$('.viewport-mode button').forEach(b=>b.classList.toggle('active',b.dataset.source===state.source));$('#undo-button').disabled=!state.history.length;
  return true;
}
$('#undo-button').onclick=()=>{const changed=applyAction('undo');$('#command-feedback').textContent=changed?'已撤销上一次操作。':'没有可撤销的操作。';};
$('#focus-selection').onclick=$('#detail-focus').onclick=()=>{if(state.selectedId)applyAction('focus',[state.selectedId]);};
$('#hide-selection').onclick=$('#detail-hide').onclick=()=>{if(state.selectedId)applyAction('hide',[state.selectedId]);};
$('#detail-isolate').onclick=()=>{if(state.selectedId)applyAction('isolate',[state.selectedId]);};
async function sendCommand(){
  const text=$('#command-input').value.trim();if(!text)return;if(!state.objects.length){toast('请先加载具有语义标识的场景。');return;}
  $('#command-send').disabled=true;$('#command-feedback').textContent='正在理解指令…';
  try{const result=await api('/api/commands',jsonOptions({text,use_llm:$('#use-llm')?.checked||false,objects:state.objects.map(o=>({...o,visible:!effectivelyHidden(String(o.id))})),selected_ids:state.selectedId?[state.selectedId]:[]}));const allowed=new Set(['hide','delete','show','isolate','focus','priority','undo','search','show_all']);const applied=allowed.has(result.action)?applyAction(result.action,result.target_ids||result.targets||[],result):false;$('#command-feedback').textContent=result.action==='undo'?(applied?'已撤销上一次场景操作。':'没有可以撤销的场景操作。'):(result.message||'操作已执行。');$('#command-provider').textContent=result.source==='llm'?'模型辅助':'本地指令';$('#command-input').value='';if(result.action==='none'||result.action==='unknown')toast(result.message||'没有理解这条指令，请提供具体物品名称。');}
  catch(error){$('#command-feedback').textContent=error.message;toast(error.message,true);}finally{$('#command-send').disabled=!state.objects.length;}
}
$('#command-send').onclick=sendCommand;$('#command-input').addEventListener('keydown',e=>{if(e.key==='Enter'&&!e.isComposing)sendCommand();});
$('#help-button').onclick=()=>$('#help-dialog').showModal();$('#close-help').onclick=()=>$('#help-dialog').close();$('#help-dialog').addEventListener('click',e=>{if(e.target===$('#help-dialog')){const rect=e.target.getBoundingClientRect();if(e.clientX<rect.left||e.clientX>rect.right||e.clientY<rect.top||e.clientY>rect.bottom)e.target.close();}});
window.addEventListener('keydown',e=>{if(['INPUT','TEXTAREA','SELECT'].includes(document.activeElement?.tagName))return;if((e.metaKey||e.ctrlKey)&&e.key.toLowerCase()==='z'){e.preventDefault();applyAction('undo');}else if(e.key.toLowerCase()==='f'&&state.selectedId){applyAction('focus',[state.selectedId]);}else if((e.key.toLowerCase()==='h'||e.key==='Delete'||e.key==='Backspace')&&state.selectedId){e.preventDefault();applyAction('hide',[state.selectedId]);}else if(e.key==='Escape'&&state.scene&&!$('#help-dialog').open){applyAction('show_all');}});
async function saveScene(){
  if(!state.scene||state.busy)return;
  const source=state.scene,name=sceneFilename(state.sceneName||source.metadata?.scene_name);
  setBusy(true);
  try{
    const prepared=sceneVisibilityExport({...source,objects:state.objects},new Set(state.hidden),state.isolation===null?null:new Set(state.isolation));
    Object.assign(prepared.scene.metadata,{scene_name:state.sceneName,exported_at:new Date().toISOString(),editor:'Splat Studio',visibility_edits:true,viewer_core_region:viewer.getCoreSettings()});
    let writer;const native=window.splatDesktop;
    if(typeof native?.beginSceneExport==='function'){
      const token=await native.beginSceneExport(name);if(token===null)return;
      writer={write:text=>native.writeSceneExport(token,text),close:()=>native.finishSceneExport(token),abort:()=>native.abortSceneExport(token)};
    }else if(typeof window.showSaveFilePicker==='function'){
      const handle=await window.showSaveFilePicker({suggestedName:name,types:[{description:'完整三维场景',accept:{'application/x-ndjson':['.jsonl']}}]});writer=await handle.createWritable();
    }else{
      if(source.gaussians.length>50000)throw new Error('请在原生 App 中保存这个大型场景。');
      const parts=[];writer={write:text=>parts.push(text),abort:()=>{parts.length=0;},close:()=>{const url=URL.createObjectURL(new Blob(parts,{type:'application/x-ndjson'}));const a=document.createElement('a');a.href=url;a.download=name;a.click();setTimeout(()=>URL.revokeObjectURL(url),3000);}};
    }
    await writeSceneLines(prepared.scene,writer,{mapGaussian:prepared.mapGaussian});toast(`已保存 ${name}，可再次导入。`);
  }catch(error){if(error.name!=='AbortError')toast(`保存失败：${error.message}`,true);}
  finally{setBusy(false);}
}
$('#export-button').onclick=saveScene;
$('#export-button').title='保存完整三维外观、物品语义（如有）和当前显示设置，可再次导入。';
// Both import forms enter the editor only after all selected files validate.
sceneFlow=initSceneFlow({isBusy:()=>state.busy,hasScene:()=>!!state.scene,onResume:()=>{},
  onNew:()=>{priorityFlow?.reset();clearScene();state.projectId=null;state.files=[];state.fileUrls.forEach(URL.revokeObjectURL);state.fileUrls=[];state.inventory=[];state.priorities.clear();$('#photo-input').value='';renderPhotos();invalidateAnalysis();$('#project-title').textContent='新建场景';$('#project-title').removeAttribute('data-i18n-ignore');$('#semantics').checked=false;wizard?.reset();},
  onImport:async(file,semantic,{geometryOnly=false}={})=>{
    if(!file)throw new Error('请选择场景文件');
    setBusy(true);
    try{
      let scene,url=null;
      if(/\.jsonl$/i.test(file.name))scene=await readSceneLines(file,{onProgress:sceneReadProgress});
      else if(/\.json$/i.test(file.name)){if(file.size>256*1024*1024)throw new Error('普通 JSON 超过 256 MiB，请导入分块 JSONL。');scene=JSON.parse(await file.text());}
      else{const body=new FormData();body.append('file',file);body.append('mode','full');if(semantic)body.append('semantic',semantic);const result=await api('/api/import',{method:'POST',body});url=result.scene_url;scene=await fetchViewerScene(result.viewer_scene_url||url);}
      if(!geometryOnly&&!hasObjectSemantics(scene))throw new Error('这个文件没有物品语义，请返回选择“仅 RGB 外观”，或导入配套的语义包。');
      if(geometryOnly)scene=importAsRgb(scene);
      state.sceneName=scene.metadata?.scene_name||file.name.replace(/(?:\.splat)?\.[^.]+$/,'');scene.metadata={...scene.metadata,scene_name:state.sceneName};
      loadScene(scene,{url});toast('场景已打开。');wizard?.completed();
    }finally{setBusy(false);}
  }});
async function loadHealth(){
  try{const health=await api('/api/health');state.health=health;const geometry=health.capabilities?.learned_geometry;$('#geometry-status').textContent=geometry?.available?'DUSt3R 资源就绪 · 推断时校验权重与几何':geometry?.source_available?'几何代码已安装；需要本地权重或允许首次下载。':'几何模型未安装：运行 scripts/setup_geometry.py 后再使用自动补全。';$('#completion-mode').querySelector('[value="learned"]').disabled=!health.capabilities?.learned_completion;$('#connection-status').className='connection-status connected';$('#connection-status').innerHTML='<i></i>已就绪';const device=health.device?.name||health.device?.type||health.device||health.capabilities?.recommended_device||'自动';const text=typeof device==='string'?device.toUpperCase():'本地计算设备';$('#engine-status').textContent=`${text} · 本地引擎就绪`;$('#device-detail').innerHTML=icon('cpu')+`<span>${escapeHtml(health.device_detail||health.message||`${text} 可用 · 以实际重建路径为准`)}</span>`;if(!health.capabilities?.cuda){$('#execution-target').value='cloud';$('#device').value='cuda';}if(health.capabilities?.llm||health.capabilities?.vision_llm)$('#command-provider').textContent='模型可用';}
  catch(error){state.health=null;$('#connection-status').className='connection-status error';$('#connection-status').innerHTML='<i></i>引擎未连接';$('#engine-status').textContent='本地引擎未连接';$('#device-detail').innerHTML=icon('cpu')+'<span>请启动本地后端，或通过桌面应用打开工作台。</span>';$('#setup-status').textContent='引擎未连接 · 可先选择照片';}
  finally{refreshSemanticControls();refreshExecutionControls();wizard?.sync();}
}
initSettings({state,api,jsonOptions,renderAnalysis,invalidateAnalysis,toast,onProjectChanged:clearCloudPackage,onCaptureChanged:()=>wizard?.cameraChanged(),ensureProject});
$('#semantic-refine-button').onclick=async()=>{
  const source=state.ownedRun;if(executionView().cloud||state.busy||!source||$('#semantic-refine-button').disabled)return;
  setBusy(true);$('#job-overlay').hidden=false;$('#job-stage').textContent='提交语义优化任务';$('#job-message').textContent='复用完整 RGB 模型与项目训练数据…';$('#job-progress').style.width='0%';$('#job-progress-label').textContent='0%';$('#job-elapsed').textContent='已用时 0 秒';$('#cancel-job').disabled=false;
  lastJobTelemetry=null;renderJobEta();
  try{const job=await api(`/api/projects/${source.projectId}/semantic-jobs`,jsonOptions({source_run_id:source.runId,config:{...config(),semantics:true,device:'cuda',semantic_strategy:'posthoc',semantic_refinement:'cuda_iterative'}}));state.jobId=job.id||job.job_id;if(!state.jobId)throw new Error('引擎未返回任务标识。');pollJob();}catch(error){finishJobError(error.message);}
};
priorityFlow=initPriorityFlow({state,api,jsonOptions,getConfig:config,toast,setBusy,
  ensureProject,
  onChange:()=>{clearCloudPackage();refreshExecutionControls();wizard?.sync();},exportReview:exportCloudTask,
  onRebuild:()=>{if(!state.ownedRun||state.ownedRun.projectId!==state.projectId){state.projectId=null;state.files=[];state.fileUrls.forEach(URL.revokeObjectURL);state.fileUrls=[];$('#photo-input').value='';}clearScene();renderPhotos();invalidateAnalysis();sceneFlow.setPhase('new');$('#project-title').textContent='重点物品重建';wizard?.reset();}
});
wizard=initCreationWizard({state,config,analyze,ensureProject,
  plan:()=>api(`/api/projects/${state.projectId}/plan`,jsonOptions(config())),
  start:startReconstruction,save:saveScene,onImport:mode=>sceneFlow.openImport(mode),priority:priorityFlow,
  refreshControls:invalidateAnalysis,toast,health:loadHealth});
priorityFlow.refresh();refreshExecutionControls();refreshSemanticControls();wizard.sync();
loadHealth();
// Product startup always opens the scene chooser; no implicit result or demo load.

installI18n();
