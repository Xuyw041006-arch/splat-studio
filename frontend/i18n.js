import catalog from './locales/en.json' with {type:'json'};

export const LANGUAGE_KEY='splat-studio-language';
export const normalizeLanguage=value=>value==='en'?'en':'zh';
const escapeRegExp=s=>s.replace(/[.*+?^${}()|[\]\\]/g,'\\$&');
// Dynamic UI messages use numbered placeholders. Captured user data must never
// be translated, even when an object is named exactly like a button (e.g. 快速).
const opaque={
  '以“{0}”为核心':[0], '保存为 {0}':[0], '移除 {0}':[0], '已保存 {0}，可再次导入。':[0],
  '{0} 中的 {1}':[0,1], '使用 {0} 的 {1} 区域 {2}':[0,1,2],
  '{0} · {1} · 检测分数 {2}% · 占图 {3}%{4}':[0,1],
  '尚未匹配：{0}。请修改描述或补充手动标注后重新匹配。':[0],
  '重点类别：{0}。这些分数不是分割准确率。':[0],
  '已确认 {0} 个照片区域：{1}。下一步开始训练或导出云端训练任务。':[1],
  '已开启 · 重点：{0}':[0],
  '重点：{0} · 细化 {1} 步（计入总预算 {2} 步） · 新增 {3} / 上限 {4} 个高斯':[0],
  '{0} · 目标区域 PSNR {1} → {2} dB':[0], '运行命令：{0}':[0],
  '以下重点没有确认区域：{0}。请修改描述或补充标注。':[0], '无法解码照片 {0}':[0],
};
const patterns=Object.entries(catalog).filter(([s])=>/\{\d+\}/.test(s)).map(([source,target])=>{
  const ids=[];let last=0,pattern='';
  for(const match of source.matchAll(/\{(\d+)\}/g)){pattern+=escapeRegExp(source.slice(last,match.index))+'([\\s\\S]*?)';ids.push(Number(match[1]));last=match.index+match[0].length;}
  return {source,target,ids,re:new RegExp('^'+pattern+escapeRegExp(source.slice(last))+'$'),specificity:source.replace(/\{\d+\}/g,'').length};
}).sort((a,b)=>b.specificity-a.specificity);
export function translate(text,language='en',depth=0){
  if(language!=='en'||typeof text!=='string'||!/[\u3400-\u9fff]/u.test(text)||depth>6)return text;
  const leading=text.match(/^\s*/)[0],trailing=text.match(/\s*$/)[0],value=text.trim();
  if(Object.hasOwn(catalog,value))return leading+catalog[value]+trailing;
  if(/\d (?:秒|分钟|小时)–/.test(value))return leading+value.split('–').map(s=>translate(s,'en',depth+1)).join('–')+trailing;
  for(const item of patterns){const match=value.match(item.re);if(!match)continue;
    const args={};item.ids.forEach((id,i)=>{args[id]=opaque[item.source]?.includes(id)?match[i+1]:translate(match[i+1],'en',depth+1);});
    return leading+item.target.replace(/\{(\d+)\}/g,(_,id)=>args[id]??'')+trailing;
  }
  // Composite status bars are assembled from independently translated fragments.
  const parts=value.split(/(\s*·\s*|–|(?<=。)\s*|\n)/);
  if(parts.length>1)return leading+parts.map((s,i)=>i%2?s:translate(s,'en',depth+1)).join('')+trailing;
  return text;
}

const ignored='script,style,pre,code,textarea,[data-i18n-ignore],.object-label,#selection-label,#object-detail-name,#mask-photo,.photo-thumb img';
const attributes=['title','placeholder','aria-label','alt'];
/** Localize the imperative UI at its DOM boundary, without changing application
 * state, input values, API payloads, node identity, event handlers, or selection.
 * Only changed nodes are processed; renderer updates never rescan the document.
 */
export function installI18n(doc=document,{storage=globalThis.localStorage,native=globalThis.splatDesktop}={}){
  let language='zh';try{language=normalizeLanguage(storage?.getItem(LANGUAGE_KEY));}catch{}
  const originals=new WeakMap();let observer;
  const ignoredNode=node=>!!(node.nodeType===1?node:node.parentElement)?.closest?.(ignored);
  function localizeValue(node,key,read,write){
    if(ignoredNode(node))return;
    const current=read();if(!current)return;
    let record=originals.get(node);if(!record){record={};originals.set(node,record);}
    let item=record[key];if(!item||current!==item.output){item={source:current,output:current};record[key]=item;}
    const output=translate(item.source,language);item.output=output;if(current!==output)write(output);
  }
  function visit(node){
    if(ignoredNode(node))return;
    if(node.nodeType===3)localizeValue(node,'text',()=>node.textContent,v=>{node.textContent=v;});
    else if(node.nodeType===1||node.nodeType===9){
      if(node.nodeType===1)for(const key of attributes)if(node.hasAttribute(key))localizeValue(node,key,()=>node.getAttribute(key),v=>node.setAttribute(key,v));
      for(const child of node.childNodes)visit(child);
    }
  }
  function refresh(){
    visit(doc.documentElement);doc.documentElement.lang=language==='en'?'en':'zh-CN';
    doc.querySelectorAll('[data-language]').forEach(b=>{const selected=b.dataset.language===language;b.setAttribute('aria-pressed',String(selected));b.classList.toggle('active',selected);});
    doc.title=language==='en'?'Splat Studio · 3D reconstruction':'Splat Studio · 三维重建工作台';
  }
  function setLanguage(value,persist=true){
    language=normalizeLanguage(value);if(persist){try{storage?.setItem(LANGUAGE_KEY,language);}catch{}void native?.setLanguage?.(language);}
    refresh();doc.dispatchEvent(new (doc.defaultView?.CustomEvent||CustomEvent)('splat-language-change',{detail:{language}}));
  }
  doc.querySelectorAll('[data-language]').forEach(b=>b.addEventListener('click',()=>setLanguage(b.dataset.language)));
  observer=new (doc.defaultView?.MutationObserver||MutationObserver)(records=>{
    const changed=new Set();for(const r of records){if(r.type==='childList'){for(const n of r.addedNodes)changed.add(n);}else changed.add(r.target);}
    for(const node of changed)if(node.isConnected)visit(node);
  });
  observer.observe(doc.documentElement,{subtree:true,childList:true,characterData:true,attributes:true,attributeFilter:attributes});
  refresh();
  // Native preferences survive the app's random loopback port across launches.
  Promise.resolve(native?.getLanguage?.()).then(value=>{if(value==='zh'||value==='en')setLanguage(value,false);}).catch(()=>{});
  return {setLanguage,getLanguage:()=>language,refresh,destroy:()=>observer.disconnect()};
}
