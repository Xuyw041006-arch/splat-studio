import assert from 'node:assert/strict';
import test from 'node:test';
import {trainingPriorityLabel,trainingPriorityLabels} from '../frontend/training_priorities.js';

test('priority training uses canonical mask labels rather than viewer IDs or translated aliases',()=>{
  const objects=[{id:'semantic-class-abc',semantic_key:'stuffed bear',label:'熊玩偶',aliases:['毛绒熊'],level:'object'},
    {id:'candidate-2',label:'coffee mug',aliases:['咖啡杯'],level:'object'},{id:'scene',label:'场景',level:'scene'}];
  assert.equal(trainingPriorityLabel(objects[0]),'stuffed bear');
  assert.deepEqual(trainingPriorityLabels(new Set(['semantic-class-abc','candidate-2','scene']),objects),['stuffed bear','coffee mug']);
  assert.deepEqual(trainingPriorityLabels(new Set(['stuffed bear','STUFFED BEAR','coffee mug']),objects),['stuffed bear','coffee mug']);
  assert.equal(trainingPriorityLabel({id:'orphan'}),null);
});
