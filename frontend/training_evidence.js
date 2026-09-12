// Training facts travel with the model. Missing or unpaired measurements never
// become claimed improvements; this panel also works for imported result files.
const finite=value=>typeof value==='number'&&Number.isFinite(value);
const positiveInt=value=>Number.isSafeInteger(value)&&value>0;
const hash=value=>typeof value==='string'&&/^[a-f0-9]{64}$/.test(value);
const labels=value=>Array.isArray(value)?[...new Set(value.filter(x=>typeof x==='string'&&x.trim()).map(x=>x.trim()))].slice(0,100):[];
const signed=(value,digits)=>`${value>0?'+':''}${value.toFixed(digits)}`;

export function trainingEvidenceView(metadata={}){
  const evidence=metadata.training_evidence;
  if(!evidence||evidence.status!=='completed')return null;
  const sourceHash=metadata.source_ply_sha256||metadata.ply_sha256;
  if(!hash(sourceHash)||!hash(evidence.model_sha256)||sourceHash!==evidence.model_sha256)return null;
  const comparison=evidence.comparison||{};
  const paired=comparison.paired===true&&comparison.evaluation==='heldout'&&
    hash(comparison.split_sha256)&&hash(evidence.baseline_model_sha256)&&
    Number.isSafeInteger(comparison.seed);
  const priorities=labels(evidence.priority_objects);
  const measurements=[];
  if(paired)for(const roi of (Array.isArray(evidence.roi_metrics)?evidence.roi_metrics:[]).slice(0,100)){
    if(!roi||!priorities.includes(roi.label)||!positiveInt(roi.views))continue;
    const psnr=finite(roi.baseline_psnr)&&finite(roi.priority_psnr)
      ?`${roi.baseline_psnr.toFixed(2)} → ${roi.priority_psnr.toFixed(2)} dB（${signed(roi.priority_psnr-roi.baseline_psnr,2)}）`:null;
    const ssim=[roi.baseline_ssim,roi.priority_ssim].every(x=>finite(x)&&x>=-1&&x<=1)
      ?`${roi.baseline_ssim.toFixed(4)} → ${roi.priority_ssim.toFixed(4)}（${signed(roi.priority_ssim-roi.baseline_ssim,4)}）`:null;
    if(psnr||ssim)measurements.push({label:roi.label,views:roi.views,psnr,ssim});
  }
  const times=[];
  for(const [key,title] of [['baseline_rgb','普通重建'],['priority_rgb','重点增强重建'],['teacher','语义掩码准备'],['semantic','语义优化']]){
    const value=evidence.timing_seconds?.[key];
    if(finite(value)&&value>=0)times.push({title,seconds:value});
  }
  const semantic=[];
  const sm=evidence.semantic_metrics;
  if(sm?.unit==='percent'&&sm.evaluation==='heldout'&&sm.source_ply_sha256===sourceHash&&
    hash(sm.split_sha256)&&sm.split_sha256===comparison.split_sha256)for(const [key,title] of [['miou','mIoU'],['boundary_iou','边界 IoU']]){
    const value=evidence.semantic_metrics[key];
    if(finite(value)&&value>=0&&value<=100)semantic.push({title,value});
  }
  return {priorities,paired,measurements,times,semantic,
    mode:({fast:'快速',balanced:'均衡',fine:'精细'})[evidence.mode]||'自定义',
    rgbSteps:positiveInt(evidence.rgb_steps)?evidence.rgb_steps:null,
    semanticSteps:positiveInt(evidence.semantic_steps)?evidence.semantic_steps:null,
    trainViews:positiveInt(evidence.train_views)?evidence.train_views:null,
    heldoutViews:positiveInt(evidence.heldout_views)?evidence.heldout_views:null,
    method:evidence.priority_method==='normalized_pixel_weight_1_plus_2M'
      ?'重点物品掩码内的 RGB 误差权重为 3，其他区域为 1，并按总权重归一化。物品边缘也参与优化。':null,
    roiSource:typeof comparison.roi_source==='string'?comparison.roi_source.slice(0,500):null};
}

export function createTrainingEvidence({parent,onFocus=()=>{},document:doc=globalThis.document}){
  const panel=doc.createElement('section');panel.className='training-evidence result-summary';panel.hidden=true;
  panel.setAttribute('aria-label','重要物品重建实测');parent.append(panel);
  const text=(tag,value,container=panel)=>{const el=doc.createElement(tag);el.textContent=value;container.append(el);return el;};
  return {clear(){panel.hidden=true;panel.replaceChildren();},render(scene){
    panel.replaceChildren();const evidence=trainingEvidenceView(scene?.metadata);panel.hidden=!evidence;if(!evidence)return;
    text('h3','重要物品精细重建');
    const budget=[`${evidence.mode}模式`,evidence.rgbSteps&&`RGB ${evidence.rgbSteps.toLocaleString()} 步`,evidence.semanticSteps&&`语义 ${evidence.semanticSteps.toLocaleString()} 步`].filter(Boolean);
    text('p',budget.join(' · '));
    if(evidence.trainViews&&evidence.heldoutViews)text('small',`${evidence.trainViews} 个训练视角 · ${evidence.heldoutViews} 个独立评估视角`);
    if(evidence.method)text('p',evidence.method);
    for(const label of evidence.priorities){
      const object=(scene.objects||[]).find(o=>o.semantic_label===label||o.label===label||o.name===label||o.id===label);
      const button=text('button',object?.label||label);button.type='button';button.className='button secondary compact';
      button.disabled=!object;button.title=object?'聚焦这个重要物品':'当前模型没有对应语义标识';
      if(object)button.onclick=()=>onFocus(String(object.id));
    }
    if(evidence.measurements.length){
      text('p','独立评估视角的物品区域：普通重建 → 重点增强');
      for(const row of evidence.measurements){const group=doc.createElement('div');panel.append(group);text('strong',`${row.label} · ${row.views} 个视角`,group);if(row.psnr)text('p',`PSNR ${row.psnr}`,group);if(row.ssim)text('p',`SSIM ${row.ssim}`,group);}
      if(evidence.roiSource)text('small',`区域来源：${evidence.roiSource}`);
    }else text('p','尚无可比较的独立评估指标，不能据此判断重点增强是否提高画质。');
    if(evidence.times.length){const details=doc.createElement('details');panel.append(details);text('summary','实测耗时',details);for(const row of evidence.times)text('p',`${row.title}：${row.seconds.toFixed(1)} 秒`,details);}
    for(const row of evidence.semantic)text('p',`语义 ${row.title}：${row.value.toFixed(2)}%`);
  }};
}
