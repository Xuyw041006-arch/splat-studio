import test from 'node:test';
import assert from 'node:assert/strict';
import {trainingEvidenceView} from '../frontend/training_evidence.js';

function metadata(){return {source_ply_sha256:'a'.repeat(64),training_evidence:{
  status:'completed',model_sha256:'a'.repeat(64),baseline_model_sha256:'b'.repeat(64),
  mode:'balanced',rgb_steps:15000,semantic_steps:1245,train_views:170,heldout_views:20,
  priority_objects:['coffee mug'],priority_method:'normalized_pixel_weight_1_plus_2M',
  comparison:{paired:true,evaluation:'heldout',seed:0,split_sha256:'c'.repeat(64),roi_source:'heldout annotations'},
  roi_metrics:[{label:'coffee mug',views:4,baseline_psnr:30,priority_psnr:29.5,baseline_ssim:0.8,priority_ssim:0.85}],
  timing_seconds:{baseline_rgb:100,priority_rgb:102,semantic:10,teacher:40},
  semantic_metrics:{unit:'percent',miou:50,boundary_iou:30,evaluation:'heldout',source_ply_sha256:'a'.repeat(64),split_sha256:'c'.repeat(64)}
}};}

test('a planned or differently bound result cannot present completed training evidence',()=>{
  assert.equal(trainingEvidenceView({}),null);
  const item=metadata();item.training_evidence.status='running';assert.equal(trainingEvidenceView(item),null);
  item.training_evidence.status='completed';item.source_ply_sha256='d'.repeat(64);assert.equal(trainingEvidenceView(item),null);
  delete item.source_ply_sha256;assert.equal(trainingEvidenceView(item),null);
});
test('semantic scores require their own heldout split and model binding',()=>{
  for(const change of [{evaluation:'train'},{source_ply_sha256:'d'.repeat(64)},{split_sha256:null},{split_sha256:'d'.repeat(64)}]){
    const item=metadata();Object.assign(item.training_evidence.semantic_metrics,change);
    assert.equal(trainingEvidenceView(item).semantic.length,0);
  }
});
test('paired heldout results preserve both regressions and improvements and actual steps',()=>{
  const view=trainingEvidenceView(metadata());assert.equal(view.rgbSteps,15000);assert.equal(view.semanticSteps,1245);
  assert.match(view.measurements[0].psnr,/-0.50/);assert.match(view.measurements[0].ssim,/\+0.0500/);
  assert.equal(view.times.length,4);assert.equal(view.semantic[0].value,50);
});
test('unpaired or training-view measurements do not claim an advantage',()=>{
  for(const change of [{paired:false},{evaluation:'train'},{split_sha256:null},{seed:null}]){
    const item=metadata();Object.assign(item.training_evidence.comparison,change);
    assert.equal(trainingEvidenceView(item).measurements.length,0);
  }
});
test('invalid metrics and unrelated objects are excluded rather than shown as zero',()=>{
  const item=metadata();item.training_evidence.roi_metrics.push({label:'unrelated',views:2,baseline_psnr:0,priority_psnr:100});
  item.training_evidence.roi_metrics[0].priority_psnr=NaN;item.training_evidence.roi_metrics[0].priority_ssim=5;
  item.training_evidence.timing_seconds.semantic=-1;item.training_evidence.semantic_metrics.miou=Infinity;
  const view=trainingEvidenceView(item);assert.equal(view.measurements.length,0);assert.equal(view.times.length,3);
  assert.equal(view.semantic.length,1);
});
