import assert from 'node:assert/strict';
import test from 'node:test';
import {executionConfig, executionControls, cloudPackageDownload, requestCloudPackage} from '../frontend/execution_target.js';

const hash='a'.repeat(64);
const response=(overrides={})=>({status:'awaiting_remote_execution',execution_target:'cloud',
  cloud_provider:'colab',id:'cloud-fixture-1',package_sha256:hash,
  package_url:'/api/cloud-jobs/cloud-fixture-1/package',...overrides});
const health={capabilities:{cuda_semantic_refinement:true}};

test('execution defaults and cloud choices retain the local device preference',()=>{
  assert.deepEqual(executionConfig(),{execution_target:'local',cloud_provider:'colab',device:'auto'});
  for(const target of ['local','cloud'])for(const provider of ['colab','server'])for(const device of ['auto','mps','cuda','cpu']){
    assert.deepEqual(executionConfig(target,provider,device),{execution_target:target,cloud_provider:provider,device});
  }
  assert.equal(executionConfig('cloud','server','mps').device,'mps');
});

test('invalid execution enums are rejected rather than silently falling back',()=>{
  for(const value of ['remote','LOCAL','',null,false,1])assert.throws(()=>executionConfig(value,'colab','auto'));
  for(const value of ['aws','COLAB','',null,false,1])assert.throws(()=>executionConfig('local',value,'auto'));
  for(const value of ['gpu','CUDA','',null,false,1])assert.throws(()=>executionConfig('cloud','colab',value));
  assert.throws(()=>executionControls({target:'remote',photoCount:10}));
});

test('cloud export remains available when only local reconstruction and local CUDA are unavailable',()=>{
  const plan={can_reconstruct:false,can_export_cloud_task:true};
  const input={target:'cloud',provider:'colab',device:'mps',photoCount:2,plan,
    health:{capabilities:{cuda_semantic_refinement:false}},ownedRun:{id:'previous-local-run'}};
  const before=JSON.stringify(input);
  const controls=executionControls(input);
  assert.equal(controls.cloud,true);
  assert.equal(controls.canExport,true);
  assert.equal(controls.canLocalReconstruct,false);
  assert.equal(controls.iterativeSelectable,true);
  assert.equal(controls.canRefineCurrent,false);
  assert.equal(JSON.stringify(input),before);
  assert.match(controls.primaryLabel,/导出/);
  assert.match(controls.estimateLabel,/云端/);
  assert.match(controls.connectionText,/未连接/);
});

test('cloud export obeys photo count, busy state, and an explicit export prohibition',()=>{
  const base={target:'cloud',photoCount:2,health:null};
  assert.equal(executionControls(base).canExport,true);
  assert.equal(executionControls({...base,photoCount:1}).canExport,false);
  assert.equal(executionControls({...base,photoCount:0}).canExport,false);
  assert.equal(executionControls({...base,busy:true}).canExport,false);
  assert.equal(executionControls({...base,plan:{can_export_cloud_task:false}}).canExport,false);
  assert.equal(executionControls({...base,provider:'server',device:'cpu',plan:{can_reconstruct:false}}).canExport,true);
});

test('local reconstruction requires a local plan and never falls back to cloud export',()=>{
  for(const plan of [null,undefined,{can_reconstruct:false}]){
    const controls=executionControls({target:'local',photoCount:20,plan,health});
    assert.equal(controls.canLocalReconstruct,false);
    assert.equal(controls.canExport,false);
  }
  assert.equal(executionControls({target:'local',plan:{can_reconstruct:true}}).canLocalReconstruct,true);
  assert.equal(executionControls({target:'local',plan:{},busy:true}).canLocalReconstruct,false);
});

test('local iterative refinement requires CUDA capability and a matching device choice',()=>{
  for(const device of ['auto','cuda']){
    const controls=executionControls({target:'local',device,health,ownedRun:{id:'owned'}});
    assert.equal(controls.iterativeSelectable,true);
    assert.equal(controls.canRefineCurrent,true);
    assert.equal(executionControls({target:'local',device,health,ownedRun:null}).canRefineCurrent,false);
    assert.equal(executionControls({target:'local',device,health,ownedRun:{id:'owned'},busy:true}).canRefineCurrent,false);
  }
  for(const device of ['mps','cpu']){
    const controls=executionControls({target:'local',device,health,ownedRun:{id:'owned'}});
    assert.equal(controls.iterativeSelectable,false);
    assert.equal(controls.canRefineCurrent,false);
  }
  for(const missing of [null,{}, {capabilities:{}}, {capabilities:{cuda_semantic_refinement:false}}]){
    assert.equal(executionControls({target:'local',device:'cuda',health:missing,ownedRun:{id:'owned'}}).canRefineCurrent,false);
  }
});

test('cloud selection never refines an existing local run, even on a local CUDA machine',()=>{
  for(const provider of ['colab','server'])for(const device of ['auto','mps','cuda','cpu']){
    const controls=executionControls({target:'cloud',provider,device,health,ownedRun:{id:'owned-local'}});
    assert.equal(controls.canRefineCurrent,false);
    assert.equal(controls.canLocalReconstruct,false);
    assert.equal(controls.iterativeSelectable,true);
  }
});

test('download requires the explicit awaiting execution state and preserves the archive hash',()=>{
  for(const provider of ['colab','server']){
    const result=response({cloud_provider:provider});
    const before=JSON.stringify(result);
    const download=cloudPackageDownload(result);
    assert.equal(download.url,result.package_url);
    assert.equal(download.filename,`splat-studio-${provider}-cloud-fixture-1.zip`);
    assert.equal(download.sha256,hash);
    assert.match(download.message,/等待/);
    assert.equal(JSON.stringify(result),before);
  }
  assert.equal(cloudPackageDownload(response({package_url:'/api/cloud-jobs/cloud-fixture-1/download'})).url,
    '/api/cloud-jobs/cloud-fixture-1/download');
});

test('download retains the backend archive filename used by the worker command',()=>{
  assert.equal(cloudPackageDownload(response({filename:'splat-cloud-fixture-1.zip'})).filename,'splat-cloud-fixture-1.zip');
  for(const filename of ['../task.zip','folder/task.zip','.task.zip','task.exe','task.zip\n',[],42]){
    assert.throws(()=>cloudPackageDownload(response({filename})));
  }
});

test('missing, completed, local, or malformed cloud responses are never treated as download success',()=>{
  for(const result of [null,undefined,{}, response({status:'completed'}),response({status:'running'}),
    response({execution_target:'local'}),response({cloud_provider:'remote'}),
    response({id:''}),response({id:' '}),response({id:null}),
    response({package_sha256:''}),response({package_sha256:'a'.repeat(63)}),
    response({package_sha256:['a'.repeat(64)]}),
    response({package_sha256:'g'.repeat(64)})])assert.throws(()=>cloudPackageDownload(result));
  for(const id of [123,true])assert.throws(()=>cloudPackageDownload(response({id,package_url:`/api/cloud-jobs/${id}/package`})));
});

test('download rejects remote URLs and any path not bound to the returned job id',()=>{
  for(const package_url of ['https://example.com/api/cloud-jobs/cloud-fixture-1/package',
    '//example.com/api/cloud-jobs/cloud-fixture-1/package','javascript:alert(1)',
    '/api/cloud-jobs/other-job/package','/api/cloud-jobs/cloud-fixture-1',
    '/api/cloud-jobs/cloud-fixture-1/../package','/api/cloud-jobs/cloud-fixture-1/..',
    '/api/cloud-jobs/cloud-fixture-1/.','/api/cloud-jobs/cloud-fixture-1/%2e%2e',
    '/api/cloud-jobs/cloud-fixture-1/package/extra','/api/cloud-jobs/cloud-fixture-1/..\\package',
    '/api/cloud-jobs/cloud-fixture-1/package?redirect=https://example.com','/api/cloud-jobs/cloud-fixture-1/package#fragment']){
    assert.throws(()=>cloudPackageDownload(response({package_url})),package_url);
  }
});

test('only an explicit local engine origin may prefix a download path',()=>{
  for(const base of ['http://localhost','https://127.0.0.1:8443','http://[::1]:8765']){
    assert.equal(cloudPackageDownload(response(),base).url,base+response().package_url);
  }
  for(const base of ['https://example.com','//localhost','http://localhost.example.com',
    'http://localhost@evil.example','http://127.0.0.1/path','http://localhost/','file:///tmp']){
    assert.throws(()=>cloudPackageDownload(response(),base),base);
  }
  for(const base of [null,false,0,{},[]])assert.throws(()=>cloudPackageDownload(response(),base));
});

const photos=()=>[new File(['image-one'],'one.jpg',{type:'image/jpeg'}),new File(['image-two'],'two.jpg',{type:'image/jpeg'})];
const cloudConfig=()=>({...executionConfig('cloud','colab','mps'),mode:'balanced',semantics:true,
  semantic_refinement:'cuda_iterative',semantic_steps:1200,priority_objects:['cup']});

test('new cloud project uploads locally then exports only, preserving requested training settings',async()=>{
  const files=photos(),config=cloudConfig(),calls=[],projects=[];
  const api=async(path,options)=>{
    calls.push({path,options});
    return path==='/api/projects'?{id:'project-one'}:response({project_id:'project-one',execution_status:'not_started'});
  };
  const result=await requestCloudPackage({api,files,config,onProject:id=>projects.push(id)});
  assert.deepEqual(calls.map(call=>call.path),['/api/projects','/api/projects/project-one/cloud-jobs']);
  const uploaded=calls[0].options.body.getAll('files');assert.deepEqual(uploaded.map(file=>file.name),['one.jpg','two.jpg']);
  assert.equal(await uploaded[1].text(),'image-two');
  assert.deepEqual(JSON.parse(calls[1].options.body),config);
  assert.deepEqual(projects,['project-one']);assert.equal(result.projectId,'project-one');
  assert.equal(result.result.execution_status,'not_started');assert.equal(config.device,'mps');
});

test('existing cloud project exports without reuploading or creating a local job',async()=>{
  const calls=[];
  await requestCloudPackage({api:async(path)=>{calls.push(path);return response({project_id:'existing'});},
    files:photos(),projectId:'existing',config:cloudConfig(),onProject:()=>assert.fail('Must retain existing project')});
  assert.deepEqual(calls,['/api/projects/existing/cloud-jobs']);
});

test('cloud export errors retain the uploaded project and never fall back to local training',async()=>{
  const calls=[],projects=[];
  await assert.rejects(requestCloudPackage({api:async(path)=>{
    calls.push(path);if(path==='/api/projects')return {id:'saved'};throw new Error('package failed');
  },files:photos(),config:cloudConfig(),onProject:id=>projects.push(id)}),/package failed/);
  assert.deepEqual(calls,['/api/projects','/api/projects/saved/cloud-jobs']);assert.deepEqual(projects,['saved']);
});

test('invalid location or photos cannot issue a cloud request',async()=>{
  const api=async()=>assert.fail('Invalid selection must not make a request');
  await assert.rejects(requestCloudPackage({api,files:photos(),config:executionConfig()}));
  await assert.rejects(requestCloudPackage({api,files:photos().slice(0,1),config:cloudConfig()}));
  await assert.rejects(requestCloudPackage({api,files:photos(),config:cloudConfig(),projectId:'other/../project'}));
});

test('mismatched project/provider or running response is never reported as a ready package',async()=>{
  for(const result of [response({project_id:'other'}),response({cloud_provider:'server'}),response({execution_status:'running'})]){
    const calls=[];
    await assert.rejects(requestCloudPackage({api:async(path)=>{calls.push(path);return result;},
      files:photos(),projectId:'current',config:cloudConfig()}));
    assert.deepEqual(calls,['/api/projects/current/cloud-jobs']);
  }
});
