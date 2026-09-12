import test from 'node:test';
import assert from 'node:assert/strict';
import {importRequirements} from '../frontend/scene_flow.js';
test('PLY requires matching semantic package unless geometry-only is selected',()=>{
 assert.match(importRequirements({geometry:{name:'scene.ply'}}),/配套/);
 assert.equal(importRequirements({geometry:{name:'scene.ply'},semantics:{name:'semantics.zip'}}),null);
 assert.equal(importRequirements({geometry:{name:'scene.PLY'},geometryOnly:true}),null);
 assert.match(importRequirements({geometry:{name:'scene.ply'},semantics:{name:'bad.json'}}),/ZIP/);
});
test('integrated scene import is self-contained and unsupported inputs stay blocked',()=>{
 assert.equal(importRequirements({geometry:{name:'scene.jsonl'}}),null);
 assert.match(importRequirements({geometry:{name:'scene.json'},semantics:{name:'semantic.zip'}}),/完整场景/);
 assert.match(importRequirements({geometry:{name:'image.jpg'}}),/PLY/);
 assert.match(importRequirements({}),/请选择/);
});
