export function priorityReady(request,approval){return !request.trim()||approval?.status==='confirmed';}
export function comparisonRows(evidence){
  if(evidence?.status!=='completed'||!Array.isArray(evidence.before)||!Array.isArray(evidence.after))return [];
  const valid=r=>r&&typeof r.image_name==='string'&&Number.isFinite(r.roi_psnr)&&Number.isFinite(r.roi_l1)
    &&typeof r.image==='string'&&r.image.length<=2e6&&/^data:image\/(jpeg|png);base64,[A-Za-z0-9+/=]+$/.test(r.image);
  const before=new Map(evidence.before.filter(valid).map(r=>[r.image_name,r]));
  return evidence.after.filter(r=>valid(r)&&before.has(r.image_name)).slice(0,20).map(after=>({before:before.get(after.image_name),after}));
}

export function initPriorityFlow({state,api,jsonOptions,ensureProject,getConfig,onChange,exportReview,onRebuild,toast,setBusy}){
  let review=null,approval=null,photoKey='',pendingLabels=[];
  const section=document.createElement('section');section.className='setup-section priority-setup';
  section.innerHTML=`<div class="section-title"><h2><span class="number">03</span> 重点物品精细重建</h2><span class="pill">可选</span></div>
    <p class="section-description">输入值得更多细节的物品，查看照片中的匹配区域，确认后参与训练。</p>
    <label class="field-label" for="priority-request">要重点重建什么？</label>
    <input id="priority-request" class="priority-input" maxlength="1000" placeholder="例如：熊、咖啡杯；或重点重建熊" />
    <label class="priority-option"><input id="priority-detail" type="checkbox" checked /> 高分辨率细化与有限增密</label>
    <p class="field-hint">在所选档位总步数内，安排 400 / 1200 / 2400 步细化；保存同视角增强前后结果。</p>
    <div class="priority-buttons"><button id="priority-match" class="button secondary">匹配照片中的物品</button><button id="priority-clear" class="text-button">清除重点</button></div>
    <p id="priority-status" class="field-hint" aria-live="polite">不填写时按普通场景训练。</p>
    <div id="priority-cloud-actions" hidden><button id="priority-export-review" class="button secondary full">导出物品识别任务</button><label class="button secondary full priority-file">导入云端重点确认包<input id="priority-import-review" type="file" accept=".zip" /></label><p class="field-hint">先在 Colab 运行识别任务，再导回确认包查看区域。此步骤不会开始 RGB 重建。</p></div>
    <button id="priority-open-review" class="button secondary full" hidden>查看匹配并确认区域</button>`;
  document.querySelector('#inventory-section').before(section);
  const $=id=>document.getElementById(id),input=$('priority-request'),status=$('priority-status');
  const dialog=document.createElement('dialog');dialog.className='priority-review-dialog';
  dialog.innerHTML=`<div class="dialog-heading"><h2>确认重点物品的实际区域</h2><button class="button secondary" id="priority-close">关闭</button></div><p class="field-hint">绿色覆盖为识别区域。逐张检查、取消错误区域；同名物品可能有多个实例，系统不把它们自动认定为同一件。</p><p id="priority-review-summary"></p><div class="priority-review-grid"></div><div class="priority-confirm-bar"><button id="priority-confirm" class="button primary">确认所选区域用于训练</button><span id="priority-confirm-feedback" aria-live="polite"></span></div>`;
  document.body.append(dialog);dialog.querySelector('#priority-close').onclick=()=>dialog.close();
  function invalidate(message='重点描述已变化，请重新匹配并确认。'){
    approval=null;review=null;pendingLabels=[];$('priority-open-review').hidden=true;$('priority-cloud-actions').hidden=true;
    status.textContent=input.value.trim()?message:'不填写时按普通场景训练。';onChange();
  }
  input.oninput=()=>invalidate();$('priority-detail').onchange=onChange;
  $('priority-clear').onclick=()=>{input.value='';state.priorities.clear();invalidate();};
  function showReview(value){
    review=value;approval=null;input.value=value.request;pendingLabels=value.requested.map(x=>x.label);
    $('priority-cloud-actions').hidden=true;$('priority-open-review').hidden=false;
    status.textContent=`已匹配 ${value.matches.length} 个照片区域，等待你的确认。`;
    const grid=dialog.querySelector('.priority-review-grid');grid.replaceChildren();
    for(const row of value.matches){
      const card=document.createElement('label');card.className='priority-review-card';
      const image=document.createElement('img');image.src=row.preview_url;image.alt=`${row.image_name} 中的 ${row.label}`;
      const check=document.createElement('input');check.type='checkbox';check.checked=true;check.value=row.id;check.setAttribute('aria-label',`使用 ${row.image_name} 的 ${row.label} 区域 ${row.id.slice(0,6)}`);
      const caption=document.createElement('span');caption.textContent=`${row.label} · ${row.image_name} · 检测分数 ${Math.round(row.confidence*100)}% · 占图 ${(row.area_fraction*100).toFixed(1)}%${row.touches_border?' · 接触画面边缘，请仔细检查':''}`;
      card.append(image,check,caption);grid.append(card);
    }
    dialog.querySelector('#priority-review-summary').textContent=value.missing.length?`尚未匹配：${value.missing.join('、')}。请修改描述或补充手动标注后重新匹配。`:`重点类别：${pendingLabels.join('、')}。这些分数不是分割准确率。`;
    dialog.querySelector('#priority-confirm').disabled=!!value.missing.length||!value.matches.length;
    dialog.querySelector('#priority-confirm-feedback').textContent='';onChange();dialog.showModal();
  }
  $('priority-open-review').onclick=()=>{if(review&&!dialog.open)dialog.showModal();};
  $('priority-match').onclick=async()=>{
    if(state.busy||!input.value.trim())return;
    setBusy(true);status.textContent='正在匹配实际照片区域…';
    try{
      await ensureProject();
      const result=await api(`/api/projects/${state.projectId}/priorities/review`,jsonOptions(getConfig()));
      if(result.status==='needs_cloud_review'){
        pendingLabels=result.requested.map(x=>x.label);status.textContent=result.message;$('priority-cloud-actions').hidden=false;
      }else showReview(result);
    }catch(e){status.textContent=e.message;toast(e.message,true);}
    finally{setBusy(false);}
  };
  $('priority-export-review').onclick=async()=>{
    if(state.busy)return;
    await exportReview({...getConfig(),priority_objects:pendingLabels,priority_task_stage:'review',priority_approval_sha256:''});
  };
  $('priority-import-review').onchange=async event=>{
    const file=event.target.files[0];if(!file)return;
    setBusy(true);
    try{await ensureProject();const body=new FormData();body.append('file',file);showReview(await api(`/api/projects/${state.projectId}/priorities/import`,{method:'POST',body}));}
    catch(e){status.textContent=e.message;toast(e.message,true);}finally{event.target.value='';setBusy(false);}
  };
  dialog.querySelector('#priority-confirm').onclick=async()=>{
    if(state.busy)return;
    const selected=[...dialog.querySelectorAll('.priority-review-grid input:checked')].map(e=>e.value);
    setBusy(true);
    try{
      approval=await api(`/api/projects/${state.projectId}/priorities/confirm`,jsonOptions({review_id:review.id,selected_ids:selected}));
      state.priorities=new Set(approval.labels);status.textContent=`已确认 ${selected.length} 个照片区域：${approval.labels.join('、')}。下一步开始训练或导出云端训练任务。`;
      dialog.close();onChange();
    }catch(e){dialog.querySelector('#priority-confirm-feedback').textContent=e.message;}finally{setBusy(false);}
  };
  const resultButton=document.createElement('button');resultButton.id='priority-evidence-button';resultButton.className='button secondary compact';resultButton.textContent='重点增强前后';resultButton.hidden=true;
  document.querySelector('.view-toolbar').prepend(resultButton);
  const resultDialog=document.createElement('dialog');resultDialog.className='priority-result-dialog';
  resultDialog.innerHTML='<div class="dialog-heading"><h2>重点重建效果检查</h2><button class="button secondary">关闭</button></div><p class="priority-result-summary"></p><div class="priority-result-pairs"></div><p class="field-hint">同一训练相机、同一目标区域的细化前后对照。指标不代表未见视角质量；若局部误差上升，会如实显示。</p>';
  document.body.append(resultDialog);resultDialog.querySelector('button').onclick=()=>resultDialog.close();resultButton.onclick=()=>resultDialog.showModal();
  const rebuild=document.createElement('button');rebuild.id='priority-rebuild';rebuild.className='button secondary full';rebuild.textContent='为所选物品配置精细重建';
  document.querySelector('#object-detail').append(rebuild);
  rebuild.onclick=()=>{
    const object=state.objects.find(o=>String(o.id)===state.selectedId);
    if(object)startFromObjects([object.semantic_key||object.canonical_label||object.label||object.name]);
  };
  function startFromObjects(labels){input.value=labels.join('、');invalidate('重点目标已带入新建流程。请提供原始照片，匹配并确认后开始重建。');onRebuild();}
  return {
    ready:()=>priorityReady(input.value,approval),
    config:()=>input.value.trim()?{priority_request:input.value.trim(),priority_objects:approval?.labels||pendingLabels,
      priority_approval_sha256:approval?.sha256||'',priority_refinement:$('priority-detail').checked?'detail':'weighted',priority_task_stage:'train'}:{priority_request:'',priority_refinement:'weighted',priority_approval_sha256:'',priority_task_stage:'train'},
    reset(){input.value='';invalidate();},startFromObjects,
    refresh(){
      const key=state.files.map(f=>`${f.name}:${f.size}:${f.lastModified}`).join('|');
      if(key!==photoKey){photoKey=key;if(approval||review)invalidate('照片已变化，请重新匹配并确认重点区域。');}
      section.hidden=state.files.length<2&&!input.value.trim();
      $('priority-match').disabled=state.busy||state.files.length<2;
      for(const element of [input,$('priority-detail'),$('priority-export-review'),$('priority-import-review')])element.disabled=state.busy;
    },
    result(scene){
      const evidence=scene?.metadata?.priority_enhancement,rows=comparisonRows(evidence);resultButton.hidden=!rows.length;
      const pairs=resultDialog.querySelector('.priority-result-pairs');pairs.replaceChildren();if(!rows.length)return;
      const number=value=>Number.isFinite(value)?value:'未记录';
      resultDialog.querySelector('.priority-result-summary').textContent=`重点：${Array.isArray(evidence.labels)?evidence.labels.filter(x=>typeof x==='string').join('、'):'未记录'} · 细化 ${number(evidence.profile?.steps)} 步（计入总预算 ${number(evidence.profile?.total_iterations)} 步） · 新增 ${number(evidence.added_gaussians)} / 上限 ${number(evidence.growth_limit)} 个高斯`;
      for(const {before,after} of rows){
        const section=document.createElement('section'),title=document.createElement('h3');title.textContent=`${after.image_name} · 目标区域 PSNR ${before.roi_psnr.toFixed(2)} → ${after.roi_psnr.toFixed(2)} dB`;section.append(title);
        const pair=document.createElement('div');pair.className='priority-pair';
        for(const [row,label] of [[before,`细化前 · ${evidence.before_iteration} 步`],[after,`细化后 · ${evidence.after_iteration} 步`]]){
          if(!/^data:image\/(jpeg|png);base64,[A-Za-z0-9+/=]+$/.test(row.image)||row.image.length>2e6)continue;
          const figure=document.createElement('figure'),image=document.createElement('img'),caption=document.createElement('figcaption');image.src=row.image;image.alt=label;caption.textContent=label;figure.append(image,caption);pair.append(figure);
        }
        section.append(pair);pairs.append(section);
      }
    }
  };
}
