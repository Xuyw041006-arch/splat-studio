export function estimateBasisText(plan){
  return `预估依据（非本任务实测）：${plan?.estimate_basis||'当前设备缺少针对本场景的标定，只能提供粗略规划范围。'}`;
}
export function trainingEtaText(job,duration,now=Date.now()){
  const p=job?.training_progress;
  if(job?.status!=='running'||!p)return '';
  const age=now/1000-Number(p.updated_at);
  if(p.phase==='stale'||!Number.isFinite(age)||age<0||age>45)return '训练进度暂未更新 · ETA 已暂停';
  if(p.phase!=='training'||!(p.total>p.iteration))return '';
  if(p.eta_seconds==null||!Number.isFinite(Number(p.eta_seconds)))return '训练速率采样中 · 暂无稳定 ETA';
  const source=p.eta_basis==='tqdm_observed_rate'?'近期日志速率':'近期实测迭代';
  return `训练循环剩余约 ${duration(Math.max(0,Number(p.eta_seconds)))} · 据${source}估算；不含验证、保存和融合`;
}
