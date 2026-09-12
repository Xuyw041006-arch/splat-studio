import test from 'node:test';
import assert from 'node:assert/strict';
import {nameError,sceneFilename,nextSetupStep,previousSetupStep,photoRecommendation,importAsRgb,hasObjectSemantics} from '../frontend/user_workflow.js';

test('RGB skips priority, semantic path includes it, return preserves branch',()=>{
 assert.equal(nextSetupStep('semantics',false),'settings');
 assert.equal(nextSetupStep('semantics',true),'priority');
 assert.equal(previousSetupStep('settings',false),'semantics');
 assert.equal(previousSetupStep('settings',true),'priority');
 assert.equal(previousSetupStep('confirm',true),'settings');
});
test('named output uses a portable filename and preserves Chinese',()=>{
 assert.equal(nameError('我的茶桌'),'');assert.equal(sceneFilename('我的茶桌'),'我的茶桌.splat.jsonl');
 for(const value of ['','../x','x:y','CON','name.','x'.repeat(65)])assert.ok(nameError(value));
 assert.equal(sceneFilename('../../x'),'.._.._x.splat.jsonl');
 assert.match(sceneFilename('CON'),/^场景-/);
});
test('capture recommendation handles duplicates, two views and sufficient coverage',()=>{
 assert.equal(photoRecommendation({image_count:9,unique_image_count:1,connected_ratio:1}).kind,'blocked');
 assert.equal(photoRecommendation({image_count:2,unique_image_count:2,connected_ratio:1}).kind,'completion');
 assert.equal(photoRecommendation({image_count:6,unique_image_count:6,connected_ratio:1}).kind,'sparse');
 assert.equal(photoRecommendation({image_count:20,unique_image_count:20,connected_ratio:1}).kind,'direct');
 assert.equal(photoRecommendation({image_count:20,unique_image_count:20,connected_ratio:1},45).kind,'sparse');
 assert.equal(photoRecommendation({image_count:20,unique_image_count:20,connected_ratio:.5}).kind,'completion');
});
test('RGB import removes semantic controls and memberships without mutating original model',()=>{
 const original={objects:[{id:'bear',level:'object'}],metadata:{editor_visibility:{isolated_ids:['bear']}},gaussians:[{position:[1,2,3],sh:[1,2],object_id:'bear',semantic_ids:['bear'],semantic_path:['bear'],source:'observed'}]};
 assert.equal(hasObjectSemantics(original),true);
 const rgb=importAsRgb(original);assert.equal(hasObjectSemantics(rgb),false);
 assert.equal(rgb.gaussians[0].object_id,undefined);assert.equal(rgb.metadata.editor_visibility,undefined);
 assert.deepEqual(rgb.gaussians[0].sh,[1,2]);assert.equal(original.gaussians[0].object_id,'bear');
});
