/** Training location is independent from this computer's detected devices. */
export function executionConfig(target='local',provider='colab',device='auto'){
  if(!['local','cloud'].includes(target))throw new Error('未知的训练位置。');
  if(!['colab','server'].includes(provider))throw new Error('未知的云端训练方式。');
  if(!['auto','mps','cuda','cpu'].includes(device))throw new Error('未知的本机计算设备。');
  return {execution_target:target,cloud_provider:provider,device};
}

export function executionControls({target='local',provider='colab',device='auto',photoCount=0,busy=false,plan=null,health=null,ownedRun=null}={}){
  executionConfig(target,provider,device);
  const cloud=target==='cloud';
  const localCuda=!!health?.capabilities?.cuda_semantic_refinement&&['auto','cuda'].includes(device);
  return {cloud,canExport:cloud&&!busy&&photoCount>=2&&plan?.can_export_cloud_task!==false,
    canLocalReconstruct:!cloud&&!busy&&!!plan&&plan.can_reconstruct!==false,
    iterativeSelectable:cloud||localCuda,canRefineCurrent:!cloud&&!busy&&localCuda&&!!ownedRun,
    primaryLabel:cloud?'导出云端任务包':'开始重建',
    estimateLabel:cloud?'云端运行后估计':'分析素材后显示',
    connectionText:cloud?'未连接云端 GPU · 手动运行任务包':'本机训练'};
}

export function cloudPackageDownload(result,base=''){
  if(!result||result.status!=='awaiting_remote_execution'||result.execution_target!=='cloud'||
      !['colab','server'].includes(result.cloud_provider)||typeof result.id!=='string'||result.id.trim()!==result.id||! /^[A-Za-z0-9_-]+$/.test(result.id)){
    throw new Error('云端任务包响应不完整；没有启动任何训练。');
  }
  if(typeof result.package_sha256!=='string'||result.package_sha256.length!==64||!/^[a-f0-9]{64}$/.test(result.package_sha256))throw new Error('云端任务包缺少有效 SHA256。');
  if(result.execution_status!=null&&result.execution_status!=='not_started')throw new Error('任务包响应包含不一致的执行状态。');
  const path=result.package_url;
  if(typeof path!=='string'||path.trim()!==path||!path.startsWith(`/api/cloud-jobs/${result.id}/`)||
      !/^\/api\/cloud-jobs\/[A-Za-z0-9_-]+\/[A-Za-z0-9_.-]+$/.test(path)||
      ['.','..'].includes(path.split('/').at(-1)))throw new Error('云端任务包下载地址无效。');
  if(typeof base!=='string'||base&&!/^https?:\/\/(?:127\.0\.0\.1|localhost|\[::1\])(?::\d+)?$/.test(base)){
    throw new Error('任务包只能从当前本机引擎下载。');
  }
  const filename=result.filename??`splat-studio-${result.cloud_provider}-${result.id}.zip`;
  if(typeof filename!=='string'||filename.trim()!==filename||filename.length>200||!/^[A-Za-z0-9][A-Za-z0-9_.-]*\.zip$/.test(filename))throw new Error('云端任务包文件名无效。');
  return {url:base+path,filename,
    sha256:result.package_sha256,message:result.message||'任务包已生成，等待你在云端运行。'};
}

/** The cloud action may upload locally and export, but never starts a local job. */
export async function requestCloudPackage({api,files,projectId=null,config,onProject=()=>{}}){
  const selection=executionConfig(config?.execution_target,config?.cloud_provider,config?.device);
  if(selection.execution_target!=='cloud')throw new Error('云端导出需要先选择云服务器 GPU。');
  if(typeof api!=='function'||!Array.isArray(files)||files.length<2)throw new Error('请先选择至少两张照片。');
  const validId=id=>typeof id==='string'&&id.trim()===id&&/^[A-Za-z0-9_-]+$/.test(id);
  if(projectId!=null&&!validId(projectId))throw new Error('项目标识无效。');
  if(!projectId){
    const body=new FormData();files.forEach(file=>body.append('files',file));
    const created=await api('/api/projects',{method:'POST',body});
    projectId=created?.id||created?.project_id;
    if(!validId(projectId))throw new Error('引擎未返回有效项目标识。');
    onProject(projectId);
  }
  const result=await api(`/api/projects/${projectId}/cloud-jobs`,{method:'POST',
    headers:{'Content-Type':'application/json'},body:JSON.stringify(config)});
  cloudPackageDownload(result);
  if(result.cloud_provider!==selection.cloud_provider||result.project_id!=null&&result.project_id!==projectId){
    throw new Error('云端任务包与当前项目或云端选择不一致。');
  }
  return {projectId,result};
}
