/** Optional model configuration and explicit, geometry-grounded manual masks. */
export function sparsePanelState(analysis,plan){
  const strategy=String(plan?.strategy||plan?.reconstruction_strategy||'');
  const relevant=!!analysis&&!!plan&&(plan.sparse===true||plan.extreme===true||plan.completion_recommended===true||/sparse|two_view|completion/.test(strategy));
  return {visible:relevant,title:plan?.extreme?'极少视角 · 补全建议':'少视角 · 重建建议',
    reason:relevant?(plan.reason||analysis.summary||'照片覆盖不足，建议先恢复可靠几何，再检查是否需要补充可见区域。'):''};
}
export function semanticProfilePreview(mode='balanced',budget='mode',steps=1200,views='auto',granularity='auto'){
  const profile={fast:{steps:400,views:'sampled',limit:12,size:192,sweeps:1,granularity:'multilevel'},balanced:{steps:1200,views:'sampled',limit:24,size:256,sweeps:2,granularity:'multilevel'},fine:{steps:2400,views:'all',limit:24,size:384,sweeps:3,granularity:'multilevel'}}[mode]||{};
  return {...profile,...(budget==='manual'?{size:256,sweeps:0}:{}),steps:budget==='manual'?Number(steps):profile.steps,views:views==='auto'?profile.views:views,granularity:granularity==='auto'?profile.granularity:granularity};
}
export function geometryOptionsForPhotoSet(options){
  return {...options,geometry_backend:'auto',sparse_completion:'auto',completion:'auto'};
}
export function sceneDisplayInfo(scene,result,records){
  const loaded=Number(result.loaded)||0,inputCount=Number(result.total)||0;
  const declared=scene.metadata?.source_gaussian_count;
  const source=Number.isInteger(declared)&&declared>=inputCount?declared:inputCount;
  const degrees=new Set();let fixed=0;
  for(const record of records){if(Number.isInteger(record.shDegree)&&record.shDegree>=0&&record.shDegree<=3)degrees.add(record.shDegree);else fixed++;}
  const sh=[...degrees].sort((a,b)=>a-b).map(degree=>`SH${degree}`).join('/');
  const color=sh?(fixed?`${sh} + 固定颜色`:sh):scene.metadata?.preview_sh_degree===0?'SH0 固定颜色':'固定颜色';
  return {loaded,source,color,partial:loaded<source,shDegrees:[...degrees].sort((a,b)=>a-b)};
}
export function initSettings({state,api,jsonOptions,renderAnalysis,invalidateAnalysis,toast,onProjectChanged=()=>{},onCaptureChanged=()=>{}}){
  const $=s=>document.querySelector(s);
  $('#semantic-strategy').querySelector('[value="joint"]').disabled=true;
  $('#semantic-strategy').querySelector('[value="joint"]').textContent='联合语义训练 · 尚未实现';
  $('#strategy-hint').textContent='重建后处理语义；自动类别包含候选不等于已验证的物理实例或部件。';
  $('#semantic-settings').insertAdjacentHTML('beforeend',`
    <label class="field-label" for="semantic-refinement">后置语义处理</label>
    <select id="semantic-refinement" class="full-select"><option value="cuda_iterative">多轮层级语义 · NVIDIA CUDA</option></select>
    <label class="field-label" for="semantic-budget">语义计算预算</label>
    <select id="semantic-budget" class="full-select"><option value="mode">跟随快速 / 均衡 / 精细模式</option><option value="manual">手动设置优化步数</option></select>
    <label class="field-label" for="semantic-steps">手动优化步数</label>
    <select id="semantic-steps" class="full-select" disabled><option value="400">400 步</option><option value="1200" selected>1200 步</option><option value="2400">2400 步</option></select>
    <p id="semantic-profile-hint" class="field-hint"></p>
    <details><summary class="field-label">训练视角与语义粒度</summary>
      <label class="field-label" for="semantic-view-mode">训练图片覆盖</label><select id="semantic-view-mode" class="full-select"><option value="auto">跟随档位 · 12 / 24 / 全部视角</option><option value="all">全部训练视角</option><option value="sampled">抽样训练视角</option></select>
      <label class="field-label" for="semantic-granularity">语义粒度</label><select id="semantic-granularity" class="full-select"><option value="multilevel">层级匹配 + 跨视角置信度 · 始终启用</option></select>
      <p class="field-hint">全部视角逐张处理，不需要同时放入显存。独立评估视角不参与训练；多粒度匹配仍需跨视角证据验证。</p>
    </details>
    <p id="semantic-refinement-hint" class="field-hint">正在检测 NVIDIA CUDA。多轮优化属于实验性功能，轮数增加不保证精度提升。本机没有 CUDA 时请选择云端训练。</p>
    <button id="semantic-refine-button" class="button secondary full" type="button" disabled>仅优化当前模型的语义</button>
    <p id="semantic-refine-hint" class="field-hint">需要本应用已有的 CUDA 运行、原照片、相机与掩码；不会重训 RGB。</p>
    <label class="field-label" for="semantic-provider">识别方式</label>
    <select id="semantic-provider" class="full-select"><option value="local">本地 Grounding DINO + SAM</option><option value="manual">手动命名与标注</option><option value="vision_api">视觉大模型 · 自定义端点</option></select>
    <label class="option-check"><input id="allow-model-download" type="checkbox"><span id="model-download-label">允许首次下载本地模型（约 1 GB）</span></label>
    <p class="field-hint" id="recognition-location-hint">本地识别无需上传照片。首次使用需要下载模型，后续使用缓存。</p>
    <label class="field-label" for="candidate-labels">物品候选词 / 手动名称</label><input id="candidate-labels" class="config-input" placeholder="留空使用默认词表；如 chair, lamp, wheel">
    <label class="option-check" id="cloud-consent" hidden><input id="allow-remote-images" type="checkbox">将所选照片发送到我配置的模型端点</label>
    <p class="field-hint" id="cloud-hint" hidden>通过 SPLAT_VISION_BASE_URL / MODEL / API_KEY 环境变量配置。视觉模型提出名称后，仍需本地掩码或手动标注定位。</p>
    <button id="manual-mask-button" class="button secondary full" type="button">手动标注物品 / 部件</button>
  `);
  $('#analysis-summary').insertAdjacentHTML('afterend',`
    <section id="sparse-settings" class="analysis-summary" aria-labelledby="sparse-settings-title" hidden>
    <strong id="sparse-settings-title">少视角 · 重建建议</strong><p id="sparse-reason" class="field-hint"></p>
    <label class="field-label" for="sparse-completion">自动补全</label><select id="sparse-completion" class="full-select"><option value="auto">自动 · 少视角时补充可见区域</option><option value="none">关闭 · 仅所选初始化结果</option><option value="learned_visible">尝试学习式可见区域补全</option></select>
    <label class="option-check"><input id="allow-geometry-download" type="checkbox">允许首次下载几何权重（约 2.3 GB）</label>
    <p class="field-hint" id="geometry-status">检测几何模型资源中…</p>
    <p class="field-hint">相机位姿由照片估计。匹配可靠时保留观测点，补充通过一致性检查的推断点；未拍摄背面保持未知。模型代码需先安装。</p>
    <details id="sparse-expert-settings"><summary class="field-label">高级选项：初始化与补全</summary>
    <label class="field-label" for="geometry-backend">相机与几何初始化</label><select id="geometry-backend" class="full-select"><option value="auto">自动 · 优先传统几何</option><option value="sfm">传统 SfM</option><option value="dust3r">DUSt3R · 学习式几何</option></select>
    <select id="completion-mode" class="full-select" aria-label="附加补全后处理"><option value="auto">无额外后处理</option><option value="none">无额外后处理</option><option value="bounded_prior">局部几何扩展 · 非学习式</option><option value="learned">外接补全程序 · 仅本地预览</option></select></details>
    </section>
  `);
  $('#device-detail').insertAdjacentHTML('afterend',`<details id="camera-settings" hidden><summary class="field-label">相机参数 · 可选</summary><div class="detail-actions"><button id="camera-template" type="button" class="button secondary">下载内参模板</button><button id="camera-import" type="button" class="button secondary">导入内参 JSON</button></div><input id="camera-file" type="file" accept="application/json,.json" hidden><p class="field-hint">内参必须匹配 EXIF 转正后的照片编号、尺寸和 SHA。模板内默认值是估计值；导入标定值后请重新分析。不支持畸变和外部位姿。</p></details>`);
  $('#analysis-summary').insertAdjacentHTML('beforeend','<div id="semantic-analysis-detail" hidden><strong>语义训练覆盖</strong><p id="semantic-coverage-text" class="field-hint"></p><p id="semantic-timing-text" class="field-hint"></p><p id="semantic-analysis-profile" class="field-hint"></p></div>');
  $('.command-footer').insertAdjacentHTML('beforeend','<label class="option-check compact-check"><input id="use-llm" type="checkbox">使用已配置的 LLM</label>');
  const onProvider=()=>{
    const remote=$('#semantic-provider').value==='vision_api';$('#cloud-consent').hidden=!remote;$('#cloud-hint').hidden=!remote;
    $('#allow-model-download').disabled=$('#semantic-provider').value!=='local';invalidateAnalysis();
  };
  $('#semantic-provider').onchange=onProvider;
  ['#completion-mode','#geometry-backend','#sparse-completion','#allow-geometry-download','#candidate-labels','#allow-model-download','#allow-remote-images','#semantic-refinement','#semantic-steps','#semantic-budget','#semantic-view-mode','#semantic-granularity'].forEach(s=>$(s).onchange=invalidateAnalysis);
  async function cameraProject(){
    if(state.files.length<2)throw new Error('请先选择至少两张照片。');
    if(!state.projectId){const body=new FormData();state.files.forEach(f=>body.append('files',f));const p=await api('/api/projects',{method:'POST',body});state.projectId=p.id;}
    return state.projectId;
  }
  $('#camera-template').onclick=async()=>{if(state.busy)return;try{const pid=await cameraProject();const data=await api(`/api/projects/${pid}/cameras`);const url=URL.createObjectURL(new Blob([JSON.stringify(data,null,2)],{type:'application/json'}));const a=document.createElement('a');a.href=url;a.download='camera-intrinsics.json';a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);toast('模板已下载；请用实际标定值替换估计内参。');}catch(e){toast(e.message,true);}};
  $('#camera-import').onclick=()=>{if(!state.busy)$('#camera-file').click();};
  $('#camera-file').onchange=async()=>{try{const file=$('#camera-file').files[0];if(!file)return;if(file.size>500000)throw new Error('相机文件过大。');const pid=await cameraProject();const data=JSON.parse(await file.text());await api(`/api/projects/${pid}/cameras`,{...jsonOptions(data),method:'PUT'});invalidateAnalysis();toast('拍摄信息已更新。');onCaptureChanged();}catch(e){toast(e.message,true);}finally{$('#camera-file').value='';}};
  const dialog=document.createElement('dialog');dialog.id='mask-dialog';dialog.innerHTML=`
    <div class="dialog-heading"><h2>手动标注物品</h2><button id="mask-close" class="icon-button" aria-label="关闭标注">×</button></div>
    <p class="field-hint">在图上连续点击绘制多边形，再保存。相同物品跨照片使用相同 ID 与名称；不同实例使用不同 ID。平级语义按类别合并；多粒度模式依据可见几何匹配实例。同一 ID 仍需跨视角证据验证。</p>
    <label class="field-label" for="mask-photo">照片</label><select id="mask-photo" class="full-select"></select>
    <div class="mask-fields"><label>物品名称<input id="mask-label" class="config-input" value="重点物品"></label><label>跨视角物品 ID<input id="mask-object-id" class="config-input" value="manual-object-1"></label><label>层级<select id="mask-level" class="full-select"><option value="object">物品</option><option value="part">部件</option></select></label><label>父物品 ID（部件可填）<input id="mask-parent" class="config-input" placeholder="manual-object-1"></label></div>
    <label class="field-label" for="mask-parent-relation">当前标注与父物品的关系</label><select id="mask-parent-relation" class="full-select"><option value="member_of">属于该组 / 上级类别</option><option value="part_of">物理部件</option><option value="contents_of">容器内的物品</option></select>
    <p class="field-hint">填写父 ID 是你的显式层级声明，请先标注父物品。CUDA 多轮优化还会检查多视角掩码支持；投影重叠本身不能证明部件关系。</p>
    <canvas id="mask-canvas" width="512" height="384" aria-label="点击照片绘制物品多边形"></canvas><p class="field-hint" id="mask-feedback">至少绘制 3 个顶点。</p>
    <div class="detail-actions"><button id="mask-clear" class="button secondary">清除顶点</button><button id="mask-save" class="button primary">保存当前照片掩码</button></div>
  `;document.body.append(dialog);
  const canvas=$('#mask-canvas'),ctx=canvas.getContext('2d');let points=[],photo=null;
  function redraw(){ctx.clearRect(0,0,canvas.width,canvas.height);if(photo)ctx.drawImage(photo,0,0,canvas.width,canvas.height);if(points.length){ctx.beginPath();points.forEach(([x,y],i)=>i?ctx.lineTo(x,y):ctx.moveTo(x,y));ctx.closePath();ctx.fillStyle='#b6e5c744';ctx.strokeStyle='#d8ffe4';ctx.lineWidth=2;ctx.fill();ctx.stroke();points.forEach(([x,y])=>{ctx.beginPath();ctx.arc(x,y,3,0,Math.PI*2);ctx.fillStyle='#fff';ctx.fill();});}$('#mask-feedback').textContent=`${points.length} 个顶点 · 至少 3 个顶点可保存`;} 
  async function loadPhoto(){const index=Number($('#mask-photo').value);points=[];photo=new Image();photo.onload=()=>{const scale=Math.min(1,512/Math.max(photo.naturalWidth,photo.naturalHeight));canvas.width=Math.round(photo.naturalWidth*scale);canvas.height=Math.round(photo.naturalHeight*scale);redraw();};photo.src=state.fileUrls[index];}
  $('#manual-mask-button').onclick=async()=>{
    if(state.busy)return;if(state.files.length<2){toast('请先选择至少两张照片。');return;}
    try{
      if(!state.projectId){const body=new FormData();state.files.forEach(f=>body.append('files',f));const p=await api('/api/projects',{method:'POST',body});state.projectId=p.id;}
      if(state.manualProject!==state.projectId){state.manualProject=state.projectId;state.manualInventory=[];}
      const select=$('#mask-photo');select.replaceChildren();state.files.forEach((f,i)=>{const option=document.createElement('option');option.value=String(i);option.textContent=`${i+1}. ${f.name}`;select.append(option);});dialog.showModal();await loadPhoto();
    }catch(e){toast(e.message,true);}
  };
  $('#mask-photo').onchange=loadPhoto;$('#mask-close').onclick=()=>dialog.close();$('#mask-clear').onclick=()=>{points=[];redraw();};
  $('#mask-level').onchange=()=>{$('#mask-parent-relation').value=$('#mask-level').value==='part'?'part_of':'member_of';};
  canvas.onclick=e=>{const r=canvas.getBoundingClientRect();points.push([(e.clientX-r.left)*canvas.width/r.width,(e.clientY-r.top)*canvas.height/r.height]);redraw();};
  $('#mask-save').onclick=async()=>{
    if(points.length<3){toast('至少绘制三个顶点。');return;}
    const label=$('#mask-label').value.trim(),id=$('#mask-object-id').value.trim();if(!label||!id){toast('请填写物品名称与 ID。');return;}
    const buffer=document.createElement('canvas');buffer.width=canvas.width;buffer.height=canvas.height;const c=buffer.getContext('2d');c.beginPath();points.forEach(([x,y],i)=>i?c.lineTo(x,y):c.moveTo(x,y));c.closePath();c.fillStyle='white';c.fill();const rgba=c.getImageData(0,0,buffer.width,buffer.height).data;
    const mask=Array.from({length:buffer.height},(_,y)=>Array.from({length:buffer.width},(_,x)=>rgba[(y*buffer.width+x)*4+3]>0?1:0));
    const payload={image_name:String(Number($('#mask-photo').value)).padStart(4,'0')+'.jpg',label,object_id:id,level:$('#mask-level').value,mask};if($('#mask-parent').value.trim()){payload.parent_id=$('#mask-parent').value.trim();payload.parent_relation=$('#mask-parent-relation').value;}
    $('#mask-save').disabled=true;
    try{await api(`/api/projects/${state.projectId}/masks`,jsonOptions(payload));onProjectChanged();state.manualInventory=state.manualInventory||[];if(!state.manualInventory.some(o=>o.id===id))state.manualInventory.push({id,label,level:payload.level,source:'manual_mask'});state.priorities.add(label);state.inventory=[...state.inventory.filter(o=>o.id!==id),{id,label,source:'manual_mask'}];if(state.analysis){renderAnalysis({});}$('#mask-feedback').textContent='掩码已保存，并标记为重点物品。可切换照片继续标注同一物品。';toast('掩码已保存；重建时将应用该区域的权重。');}catch(e){toast(e.message,true);}finally{$('#mask-save').disabled=false;}
  };
}
