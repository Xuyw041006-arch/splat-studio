import test from 'node:test';
import assert from 'node:assert/strict';
import {priorityReady,comparisonRows} from '../frontend/priority_flow.js';
test('text alone never enables priority training',()=>{
 assert.equal(priorityReady('熊',null),false);
 assert.equal(priorityReady('熊',{status:'awaiting_confirmation'}),false);
 assert.equal(priorityReady('熊',{status:'confirmed'}),true);
 assert.equal(priorityReady('',null),true);
});
test('comparison keeps same-camera before/after pairing',()=>{
 const a={image_name:'a',roi_psnr:20,roi_l1:.1,image:'data:image/jpeg;base64,YQ=='},b={...a,image_name:'b',roi_psnr:21};
 assert.deepEqual(comparisonRows({status:'completed',before:[a],after:[b,a]}),[{before:a,after:a}]);
 assert.deepEqual(comparisonRows({status:'running',before:[a],after:[a]}),[]);
});
test('untrusted imported evidence cannot crash the viewer or load remote images',()=>{
 assert.deepEqual(comparisonRows({status:'completed',before:{},after:[]}),[]);
 const bad={image_name:'a',roi_psnr:'bad',image:'https://example.com/image.jpg'};
 assert.deepEqual(comparisonRows({status:'completed',before:[null,bad],after:[null,bad]}),[]);
});
