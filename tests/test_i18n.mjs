import test from 'node:test';
import assert from 'node:assert/strict';
import {parseHTML} from 'linkedom';
import {installI18n,translate} from '../frontend/i18n.js';
import {createRequire} from 'node:module';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
const require=createRequire(import.meta.url);
const {createLanguagePreferences,installLanguageIPC}=require('../desktop/language.cjs');
const settle=()=>new Promise(r=>setImmediate(r));
test('English UI, nested progress and opaque user names',()=>{
  assert.equal(translate('开始重建'),'Start reconstruction');
  assert.equal(translate('已用时 15 分钟'),'Elapsed 15 min');
  assert.equal(translate('4 分钟–15 分钟'),'4 min–15 min');
  assert.equal(translate('2 张 · 稀疏重建 · 按需补全'),'2 photos · Sparse reconstruction · Completion as needed');
  assert.equal(translate('已开启 · 重点：快速'),'Enabled · Priority: 快速');
  assert.equal(translate('保存为 我的书桌.splat.jsonl'),'Save as 我的书桌.splat.jsonl');
  assert.equal(translate('已确认 3 个照片区域：快速。下一步开始训练或导出云端训练任务。'),'Confirmed 3 photo regions: 快速. Start training or export a cloud training task next.');
  assert.equal(translate('开始重建','zh'),'开始重建');
});
test('Switch language in-place, preserve values, node identity and interactions',async()=>{
  const {document}=parseHTML('<html><body><button data-language="zh">中文</button><button data-language="en">EN</button><h1>先给场景起个名字</h1><input id="name" value="均衡" placeholder="例如：我的书桌"><span data-i18n-ignore>快速</span><span class="object-label">场景</span><code>快速</code><button id="next">下一步</button><p id="status">已用时 1 分钟</p></body></html>');
  const stored=new Map(),storage={getItem:k=>stored.get(k),setItem:(k,v)=>stored.set(k,v)};
  const ui=installI18n(document,{storage,native:{setLanguage:()=>{}}});let clicks=0;
  const button=document.querySelector('#next');button.onclick=()=>clicks++;
  document.querySelector('[data-language="en"]').click();await settle();
  assert.equal(document.documentElement.lang,'en');assert.equal(document.querySelector('h1').textContent,'Give your scene a name');
  assert.equal(document.querySelector('#name').value,'均衡');assert.equal(document.querySelector('#name').placeholder,'For example: My desk');
  assert.equal(document.querySelector('[data-i18n-ignore]').textContent,'快速');assert.equal(document.querySelector('.object-label').textContent,'场景');assert.equal(document.querySelector('code').textContent,'快速');
  assert.equal(document.querySelector('#status').textContent,'Elapsed 1 min');
  document.querySelector('#status').textContent='已用时 2 分钟';await settle();assert.equal(document.querySelector('#status').textContent,'Elapsed 2 min');
  button.click();assert.equal(clicks,1);assert.equal(button,document.querySelector('#next'));
  ui.setLanguage('zh');await settle();assert.equal(button.textContent,'下一步');assert.equal(document.querySelector('#status').textContent,'已用时 2 分钟');
  assert.equal(document.querySelector('#name').placeholder,'例如：我的书桌');assert.equal(stored.get('splat-studio-language'),'zh');ui.destroy();
});
test('Persist native language across random-port restarts; reject unknown values and IPC callers',()=>{
  const directory=fs.mkdtempSync(path.join(os.tmpdir(),'splat-locale-'));try{
    const filename=path.join(directory,'language.json'),prefs=createLanguagePreferences(filename);prefs.set('en');
    const restored=createLanguagePreferences(filename);assert.equal(restored.get(),'en');assert.equal(restored.t('导出完整高斯场景'),'Save complete Gaussian scene');assert.throws(()=>restored.set('../../x'));
    const handlers={},frame={url:'http://127.0.0.1:9000/'},contents={mainFrame:frame},win={webContents:contents};
    installLanguageIPC({ipcMain:{handle:(k,v)=>handlers[k]=v},win,origin:'http://127.0.0.1:9000',preferences:restored});
    assert.throws(()=>handlers['language:set']({sender:{},senderFrame:frame},'zh'));
    assert.equal(handlers['language:set']({sender:contents,senderFrame:frame},'zh'),'zh');
  }finally{fs.rmSync(directory,{recursive:true,force:true});}
});
