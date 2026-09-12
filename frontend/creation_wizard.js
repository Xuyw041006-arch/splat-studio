import {nameError,sceneFilename,nextSetupStep,previousSetupStep,photoRecommendation} from './user_workflow.js';

export function initCreationWizard({state,config,analyze,plan,ensureProject,start,save,onImport,priority,refreshControls,toast,health}){
  const $=s=>document.querySelector(s);
  let step='photos',analysis=null,estimate=null,error='',choosing=false,selectedSemantics=null;
  const root=document.createElement('section');root.id='creation-wizard';root.className='creation-wizard';
  root.innerHTML=`<nav class="wizard-path" aria-label="新建场景进度"><span>1 素材</span><i></i><span>2 场景理解</span><i></i><span>3 重建</span></nav>
    <div class="wizard-scroll">
    <article data-step="photos"><p class="wizard-kicker">创建场景</p><h1 tabindex="-1">先给场景起个名字</h1><label class="wizard-label" for="scene-name">场景名称</label><input id="scene-name" class="wizard-input" maxlength="64" placeholder="例如：我的书桌" autocomplete="off"><p id="scene-name-error" class="wizard-error"></p><div id="wizard-photo-fields"></div></article>
    <article data-step="analysis" hidden><p class="wizard-kicker">照片检查</p><h1 tabindex="-1">看看这些照片适合怎么重建</h1><div id="capture-loading" class="wizard-loading"><span class="spinner"></span>正在检查照片和视角重叠…</div><div id="capture-result" hidden><div class="capture-numbers"><div><strong id="capture-total"></strong><span>张照片</span></div><div><strong id="capture-unique"></strong><span>张不重复照片</span></div><div><strong id="capture-overlap"></strong><span>照片关联比例</span></div></div><div class="recommendation-card"><span class="recommendation-icon">✓</span><div><h2 id="capture-recommendation"></h2><p id="capture-reason"></p></div></div><label class="wizard-check" id="capture-completion-field"><input id="wizard-completion" type="checkbox">需要时尝试补全</label><p id="capture-extra" class="wizard-muted"></p><details class="wizard-details"><summary>调整拍摄信息</summary><div id="wizard-capture-fields"></div></details></div></article>
    <article data-step="semantics" hidden><p class="wizard-kicker">场景理解</p><h1 tabindex="-1">需要识别场景里的物品吗？</h1><div class="wizard-choice-grid" role="radiogroup" aria-label="是否识别物品"><button data-semantic-choice="no" class="wizard-choice" role="radio" aria-checked="false"><span class="choice-symbol">◈</span><h2>仅重建外观</h2><p>查看、旋转和缩放三维场景</p></button><button data-semantic-choice="yes" class="wizard-choice" role="radio" aria-checked="false"><span class="choice-symbol">⌕</span><h2>同时识别物品</h2><p>还能查找、选中和隐藏物品</p></button></div></article>
    <article data-step="priority" hidden><p class="wizard-kicker">重要物品 · 可选</p><h1 tabindex="-1">哪些物品需要更清晰？</h1><p class="wizard-subtitle">填写物品名称，稍后核对照片中的区域。留空则平均处理整个场景。</p><div id="wizard-priority-fields"></div><details class="wizard-details"><summary>识别与手动标注设置</summary><div id="wizard-semantic-fields"></div></details></article>
    <article data-step="settings" hidden><p class="wizard-kicker">重建设置</p><h1 tabindex="-1">选择效果与计算设备</h1><label class="wizard-label">重建质量</label><div id="wizard-quality-fields"></div><label class="wizard-label" for="execution-target">重建设备</label><div id="wizard-device-fields"></div><div id="wizard-cloud-profile"><label class="wizard-label" for="cloud-gpu">云端将使用的 GPU</label><select id="cloud-gpu" class="full-select"><option value="a100">NVIDIA A100</option><option value="other">其他 NVIDIA GPU</option></select></div><details class="wizard-details"><summary>提供实测速率，让预估更准确（可选）</summary><label class="wizard-label" for="measured-speed">同档位训练速度 · 步 / 秒</label><input id="measured-speed" class="wizard-input" type="number" min="0.1" max="10000" step="any" placeholder="未知则留空"></details><div class="device-ready"><span id="wizard-device-status"></span><button class="text-button" id="device-setup-help">如何配置？</button><button class="text-button" id="device-recheck">重新检测</button></div><label class="wizard-check"><input id="wizard-download-models" type="checkbox">首次使用时下载所需资源（最多约 3.3 GB）</label></article>
    <article data-step="confirm" hidden><p class="wizard-kicker">最后确认</p><h1 tabindex="-1">准备好开始了吗？</h1><dl id="wizard-confirm-summary" class="wizard-summary"></dl><div class="estimate-card"><span>预计计算时间</span><strong id="wizard-estimate">计算中…</strong><p id="wizard-estimate-basis"></p></div><div id="wizard-priority-confirm"><h2>核对重要物品</h2><p class="wizard-muted">确认绿色区域选对了物品，再开始重建。</p><div id="wizard-review-actions"></div></div><p id="wizard-execution-note" class="wizard-muted"></p></article>
    <article data-step="training" hidden><p class="wizard-kicker">正在重建</p><h1 tabindex="-1">正在生成你的场景</h1><div id="wizard-job-fields"></div></article>
    <article data-step="cloud" hidden><p class="wizard-kicker">云端重建</p><h1 tabindex="-1">任务已准备好</h1><p class="wizard-subtitle">还未启动训练。把任务包放到选好的云端设备运行，完成后把结果导入这里。</p><ol class="cloud-next-steps"><li><strong>保存任务包</strong><span>照片和所选设置已打包。</span></li><li><strong>在云端运行</strong><span>首次使用先完成环境配置，再运行包内命令。</span><button id="cloud-setup-help" class="text-button">查看配置与运行方法</button></li><li><strong>导入完成的结果</strong><span>选择云端输出的 scene.splat.jsonl。</span><button id="cloud-import-result" class="button primary">导入重建结果</button></li></ol><div id="wizard-cloud-result"></div></article>
    </div><footer class="wizard-footer"><p id="wizard-feedback" role="status"></p><div><button id="wizard-back" class="button secondary">上一步</button><button id="wizard-next" class="button primary">下一步</button></div></footer>`;
  $('.workspace').before(root);
  const move=(sel,dest)=>$(dest).append($(sel));
  move('.source-section','#wizard-photo-fields');
  const photoSection=$('.source-section');
  for(const el of [...photoSection.children])if(el.matches('.field-label,.angle-field,.field-hint'))$('#wizard-capture-fields').append(el);
  move('#camera-settings','#wizard-capture-fields');
  move('.quality-options','#wizard-quality-fields');
  move('#execution-target','#wizard-device-fields');
  move('#cloud-provider','#wizard-device-fields');
  move('.priority-setup','#wizard-priority-fields');
  for(const sel of ['.priority-buttons','#priority-status','#priority-cloud-actions','#priority-open-review'])move(sel,'#wizard-review-actions');
  move('#semantic-settings','#wizard-semantic-fields');
  move('#job-overlay','#wizard-job-fields');move('#cloud-package-result','#wizard-cloud-result');
  root.querySelector('.cloud-next-steps li').append($('#wizard-cloud-result'));
  $('#device').value='cuda';
  $('#priority-request').placeholder='例如：熊、咖啡杯';
  $('#priority-match').textContent='识别并核对区域';$('#priority-clear').textContent='不设重点';
  $('#priority-export-review').textContent='下载物品识别任务';
  $('.priority-setup .section-title').hidden=true;
  for(const el of $('.priority-setup').querySelectorAll('.section-description,.field-hint,.priority-option'))el.hidden=true;
  $('#semantic-strategy').hidden=true;$('#strategy-hint').hidden=true;document.querySelector('label[for="semantic-strategy"]').hidden=true;
  $('#wizard-semantic-fields').querySelector('#semantic-refine-button').hidden=true;
  $('#wizard-semantic-fields').querySelector('#semantic-refine-hint').hidden=true;
  for(const id of ['semantic-refinement','semantic-budget','semantic-steps','semantic-profile-hint','semantic-refinement-hint']){
    $('#'+id).hidden=true;document.querySelector(`label[for="${id}"]`)?.setAttribute('hidden','');
  }
  $('#semantic-view-mode').closest('details').hidden=true;
  $('#semantic-provider').querySelector('[value="local"]').textContent='自动识别';
  $('#semantic-provider').querySelector('[value="vision_api"]').textContent='使用已配置的视觉服务';
  $('#dropzone strong').textContent='添加照片';$('#dropzone>small').textContent='至少 2 张 · JPG、PNG、WEBP';
  for(const b of document.querySelectorAll('.quality-option')){b.title='';b.querySelector('small').textContent={fast:'先看大致效果',balanced:'速度与细节兼顾',fine:'保留更多细节'}[b.dataset.mode];}
  $('#execution-target').setAttribute('aria-label','重建设备');$('#cloud-provider').setAttribute('aria-label','云端运行位置');
  $('#execution-target').querySelector('[value="local"]').textContent='本机 NVIDIA GPU';$('#execution-target').querySelector('[value="cloud"]').textContent='云端 GPU';
  $('#cloud-provider').querySelector('[value="server"]').textContent='其他 NVIDIA 云服务器';

  const help=document.createElement('dialog');help.id='device-help-dialog';help.className='device-help-dialog';
  help.innerHTML=`<div class="dialog-heading"><h2>配置重建设备</h2><button class="icon-button" aria-label="关闭设备帮助">×</button></div><div class="help-device-tabs"><button data-help-device="local">本机</button><button data-help-device="colab">Colab</button><button data-help-device="server">其他云服务器</button></div><div id="device-help-content"></div>`;document.body.append(help);help.querySelector('.dialog-heading button').onclick=()=>help.close();
  const snippets={
    local:['本机需要 NVIDIA GPU','Mac 可以查看和保存场景，完整重建请选择云端。Windows / Linux 的 NVIDIA 电脑需安装匹配的驱动、CUDA 编译工具与 PyTorch。',[
      '在项目源码目录创建 Python 3.11 环境，安装 GPU 版 PyTorch，然后安装项目依赖。',
      '运行下列命令准备训练环境。确认 NVIDIA GPU 检测通过后，以该 Python 启动 App。',
      'Windows 原生 App 可通过 SPLAT_PYTHON 指定 Python 路径，并通过 SPLAT_3DGS_REPO 指定训练仓库；配置后重启 App。'],
      'python -m pip install -r requirements-semantic.txt\npython scripts/setup_upstream.py --cuda\npython scripts/setup_geometry.py\npython -c "import torch; print(torch.cuda.is_available())"\n# Development app: set SPLAT_PYTHON, then run npm run desktop\n# Windows PowerShell:\n$env:SPLAT_PYTHON="C:\\splat-env\\Scripts\\python.exe"\n$env:SPLAT_3DGS_REPO="C:\\SplatStudio\\vendor\\gaussian-splatting"\n& "C:\\SplatStudio\\Splat Studio.exe"'],
    colab:['使用 Colab 的 GPU','在 Colab 中更改运行时类型，选择 GPU；若要采用这里的 A100 预估，请实际选择 A100。',[
      '上传 App 导出的任务 ZIP，并解压到新的目录。',
      '按包内 README.txt 安装环境。在 Colab 单元格里，命令前加 !；先切换到解压后的目录。',
      '完成后下载 results/scene.splat.jsonl；若运行的是物品识别任务，则下载 priority-review.zip 回来确认。'],
      '!python -m pip install -r runtime/requirements-semantic.txt\n!python runtime/scripts/setup_upstream.py --cuda\n!python runtime/scripts/setup_geometry.py\n!python runtime/scripts/cloud_worker.py /content/task.zip --work-dir /content/training-run'],
    server:['使用其他 NVIDIA 云服务器','准备一台支持 CUDA 的 NVIDIA GPU 机器。推荐 Linux、至少 16 GB 显存；较大场景需要更多显存。',[
      '安装匹配的驱动、CUDA 编译工具与 GPU 版 PyTorch。用 nvidia-smi 和 nvcc --version 检查。',
      '上传任务 ZIP，解压到新的目录，按包内 README.txt 安装环境并运行 worker。',
      '下载 results/scene.splat.jsonl 后回到 App 导入。当前采用文件交换，不需要向 App 提供服务器密码。'],
      'python -m pip install -r runtime/requirements-semantic.txt\npython runtime/scripts/setup_upstream.py --cuda\npython runtime/scripts/setup_geometry.py\npython runtime/scripts/cloud_worker.py /path/task.zip --work-dir /path/training-run']};
  function showHelp(kind){
    const [title,description,items,code]=snippets[kind],content=$('#device-help-content');content.replaceChildren();
    const h=document.createElement('h3'),p=document.createElement('p'),ol=document.createElement('ol'),details=document.createElement('details'),summary=document.createElement('summary'),pre=document.createElement('pre');
    h.textContent=title;p.textContent=description;for(const item of items){const li=document.createElement('li');li.textContent=item;ol.append(li);}summary.textContent='安装与运行命令';pre.textContent=code;details.append(summary,pre);content.append(h,p,ol,details);
    help.querySelectorAll('[data-help-device]').forEach(b=>b.classList.toggle('active',b.dataset.helpDevice===kind));if(!help.open)help.showModal();
  }
  help.querySelectorAll('[data-help-device]').forEach(b=>b.onclick=()=>showHelp(b.dataset.helpDevice));
  $('#device-setup-help').onclick=$('#cloud-setup-help').onclick=()=>showHelp($('#execution-target').value==='local'?'local':$('#cloud-provider').value);
  $('#device-recheck').onclick=async()=>{await health();sync();};
  $('#cloud-import-result').onclick=()=>onImport($('#semantics').checked?'semantic':'rgb');
  $('#scene-name').oninput=()=>{state.sceneName=$('#scene-name').value.trim();$('#project-title').textContent=state.sceneName||'新建场景';$('#project-title').toggleAttribute('data-i18n-ignore',!!state.sceneName);sync();};
  $('#wizard-completion').onchange=()=>{$('#sparse-completion').value=$('#wizard-completion').checked?'auto':'none';refreshControls();};
  $('#wizard-download-models').onchange=()=>{for(const sel of ['#allow-model-download','#allow-geometry-download'])$(sel).checked=$('#wizard-download-models').checked;refreshControls();};
  root.querySelectorAll('[data-semantic-choice]').forEach(b=>b.onclick=()=>{selectedSemantics=b.dataset.semanticChoice==='yes';$('#semantics').checked=selectedSemantics;$('#semantics').dispatchEvent(new Event('change'));sync();});
  $('#wizard-back').onclick=()=>{if(!state.busy)go(previousSetupStep(step,$('#semantics').checked));};
  $('#wizard-next').onclick=async()=>{
    if(state.busy||choosing)return;
    error='';
    if(step==='photos'){await checkPhotos();return;}
    if(step==='semantics'&&selectedSemantics===null)return;
    if(step==='settings'){
      const speed=$('#measured-speed').value;
      if(speed&&(!Number.isFinite(Number(speed))||Number(speed)<.1||Number(speed)>10000)){error='请输入有效的实测速率，或留空。';sync();return;}
      choosing=true;go('confirm');
      try{await ensureProject();const result=await plan();state.plan=result.plan;estimate=result.estimate;}
      catch(e){error=e.message;estimate=null;}finally{choosing=false;sync();}return;
    }
    if(step==='confirm'){
      if($('#semantics').checked&&!priority.ready())return;
      if($('#execution-target').value==='local')go('training');
      await start();sync();return;
    }
    go(nextSetupStep(step,$('#semantics').checked));
  };
  function go(next){step=next;error='';sync();root.querySelector('.wizard-scroll').scrollTop=0;root.querySelector(`[data-step="${step}"] h1`)?.focus({preventScroll:true});}
  async function checkPhotos(){
    if(nameError(state.sceneName)){error=nameError(state.sceneName);sync();return;}
    if(state.files.length<2){error='请至少添加两张不同视角的照片。';sync();return;}
    go('analysis');analysis=null;sync();
    const result=await analyze();
    if(result){analysis=result.analysis;const rec=photoRecommendation(analysis,$('#view-span').value.trim()?Number($('#view-span').value):null);$('#wizard-completion').checked=rec.kind==='completion'||rec.kind==='sparse';$('#sparse-completion').value=$('#wizard-completion').checked?'auto':'none';}
    else error='照片分析未完成。请返回检查素材后重试。';
    sync();
  }
  function summaryRow(label,value){const row=document.createElement('div'),dt=document.createElement('dt'),dd=document.createElement('dd');dt.textContent=label;dd.textContent=value;if(label==='场景')dd.setAttribute('data-i18n-ignore','');row.append(dt,dd);$('#wizard-confirm-summary').append(row);}
  const formatTime=s=>s<60?`${Math.ceil(s)} 秒`:s<3600?`${Math.ceil(s/60)} 分钟`:`${(s/3600).toFixed(1)} 小时`;
  function sync(){
    root.querySelectorAll('[data-step]').forEach(p=>p.hidden=p.dataset.step!==step);
    root.dataset.step=step;
    const sem=$('#semantics').checked,cloud=$('#execution-target').value==='cloud';
    $('#photo-input').disabled=state.busy||!!nameError(state.sceneName);
    $('#scene-name').disabled=state.busy;$('#scene-name-error').textContent=state.sceneName?nameError(state.sceneName):'';
    $('#wizard-feedback').textContent=error;
    $('#wizard-back').hidden=['photos','training'].includes(step);$('#wizard-back').disabled=state.busy||choosing;
    const next=$('#wizard-next');next.hidden=['training','cloud'].includes(step);next.disabled=state.busy||choosing;
    next.textContent=step==='analysis'?'确认方案':step==='confirm'?(cloud?'准备云端重建':'开始重建'):step==='photos'?'分析照片':'下一步';
    if(step==='photos')next.disabled ||= !!nameError(state.sceneName)||state.files.length<2;
    if(step==='semantics')next.disabled ||= selectedSemantics===null;
    const rec=photoRecommendation(analysis,$('#view-span').value.trim()?Number($('#view-span').value):null);
    if(step==='analysis'){
      $('#capture-loading').hidden=!!analysis||!!error;$('#capture-result').hidden=!analysis;
      next.disabled ||= !analysis||rec?.kind==='blocked';
      if(analysis){$('#capture-total').textContent=analysis.image_count;$('#capture-unique').textContent=analysis.unique_image_count;$('#capture-overlap').textContent=`${Math.round(analysis.connected_ratio*100)}%`;$('#capture-recommendation').textContent=rec.title;$('#capture-reason').textContent=rec.message;$('#capture-completion-field').hidden=rec.kind==='direct'||rec.kind==='blocked';$('#capture-extra').textContent=analysis.duplicate_count?`有 ${analysis.duplicate_count} 张重复照片，不计为新视角。`:'';}
    }
    root.querySelectorAll('[data-semantic-choice]').forEach(b=>{const selected=selectedSemantics!==null&&(b.dataset.semanticChoice==='yes')===selectedSemantics;b.classList.toggle('selected',selected);b.setAttribute('aria-checked',String(selected));b.disabled=state.busy;});
    $('#cloud-provider').hidden=!cloud;$('#wizard-cloud-profile').hidden=!cloud;
    $('#wizard-device-status').textContent=cloud?'云端设备按你的选择估时，运行前需确认实际 GPU。':state.health?.capabilities?.original_3dgs?`已就绪 · ${state.health.capabilities.gpu_name||'NVIDIA GPU'}`:'本机无法进行完整重建，请配置 NVIDIA GPU 或选择云端。';
    for(const id of ['cloud-gpu','measured-speed','wizard-download-models','wizard-completion'])$('#'+id).disabled=state.busy;
    if(step==='confirm'){
      $('#wizard-confirm-summary').replaceChildren();summaryRow('场景',state.sceneName);summaryRow('照片',`${state.files.length} 张 · ${rec?.kind==='direct'?'直接重建':'稀疏重建'}${$('#wizard-completion').checked?' · 按需补全':''}`);summaryRow('物品识别',sem?($('#priority-request').value.trim()?`已开启 · 重点：${$('#priority-request').value.trim()}`:'已开启 · 不设重点'):'关闭');summaryRow('质量',{fast:'快速',balanced:'均衡',fine:'精细'}[state.mode]);summaryRow('设备',cloud?`${$('#cloud-provider').value==='colab'?'Colab':'云服务器'} · ${$('#cloud-gpu').value==='a100'?'A100':'其他 NVIDIA GPU'}`:'本机 NVIDIA GPU');
      $('#wizard-estimate').textContent=choosing?'正在估算…':estimate?.available?estimate.seconds.map(formatTime).join('–'):'暂不可预估';$('#wizard-estimate-basis').textContent=estimate?[estimate.basis,estimate.scope].filter(Boolean).join(' '):'';
      $('#wizard-priority-confirm').hidden=!sem||!$('#priority-request').value.trim();
      $('#wizard-execution-note').textContent=cloud?'下一步会生成任务包，在云端运行后再导入结果。':'完成后可保存场景，也可立即查看。';
      next.disabled ||= !state.plan||!!error||!cloud&&!state.plan.can_reconstruct||sem&&!priority.ready();
    }
    const progress=['photos','analysis'].includes(step)?0:['semantics','priority'].includes(step)?1:2;root.querySelectorAll('.wizard-path span').forEach((p,i)=>p.classList.toggle('active',i<=progress));
  }
  const complete=document.createElement('dialog');complete.className='completion-dialog';complete.innerHTML='<p class="wizard-kicker">重建成果</p><h2>场景已准备好</h2><p id="complete-name"></p><p id="complete-contents"></p><div class="detail-actions"><button id="complete-later" class="button secondary">继续查看</button><button id="complete-save" class="button primary">保存成果</button></div>';document.body.append(complete);
  $('#complete-later').onclick=()=>complete.close();$('#complete-save').onclick=async()=>{complete.close();await save();};
  sync();
  return {sync,step:()=>step,config:()=>({scene_name:state.sceneName||'',full_quality:true,cloud_gpu:$('#cloud-gpu').value,measured_steps_per_second:$('#measured-speed').value?Number($('#measured-speed').value):null}),
    reset(){selectedSemantics=null;state.sceneName='';$('#scene-name').value='';analysis=null;estimate=null;go('photos');},
    photosChanged(){analysis=null;estimate=null;if(state.files.length>=2&&!state.busy&&!nameError(state.sceneName))void checkPhotos();else go('photos');},
    invalidateCapture(){analysis=null;estimate=null;sync();},
    cameraChanged(){analysis=null;estimate=null;void checkPhotos();},
    failed(message){go('confirm');error=message;sync();},
    cloudPrepared(review){if(!review)go('cloud');},
    completed(){const name=state.scene?.metadata?.scene_name||state.sceneName||'场景';$('#complete-name').textContent=`保存为 ${sceneFilename(name)}`;$('#complete-contents').textContent=state.objects.length?'一个完整场景文件，包含三维外观、物品语义、层级和当前显示设置。可再次导入继续操作。':'一个完整场景文件，包含三维外观和当前显示设置。可再次导入查看。';if(!complete.open)complete.showModal();}
  };
}
