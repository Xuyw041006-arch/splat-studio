// Only explicitly registered, local results appear here. Metrics retain their
// own scope so a historical full-model score is not presented as a new UI test.
export async function initResultLibrary({api,openScene,isBusy,toast}){
  const button=document.createElement('button');button.id='results-button';button.className='button secondary compact';button.textContent='测试结果';button.hidden=true;
  document.querySelector('#demo-button').before(button);
  const dialog=document.createElement('dialog');dialog.className='results-dialog';dialog.setAttribute('aria-label','选择测试结果');
  const heading=document.createElement('h2');heading.textContent='打开真实测试结果';
  const intro=document.createElement('p');intro.textContent='选择场景后可以直接旋转、缩放。带语义的结果还支持搜索、聚焦、隐藏和撤销。';
  const close=document.createElement('button');close.className='button ghost';close.textContent='关闭';close.onclick=()=>dialog.close();
  const list=document.createElement('div');list.className='result-cards';dialog.append(heading,intro,list,close);document.body.append(dialog);
  const summary=document.createElement('section');summary.className='result-summary';summary.hidden=true;summary.setAttribute('aria-label','当前测试结果');
  document.querySelector('.setup-scroll').prepend(summary);
  const loading=document.createElement('div');loading.className='result-loading';loading.hidden=true;loading.setAttribute('role','status');document.querySelector('#canvas-wrap').append(loading);
  let entries=[];
  function metrics(parent,entry){
    for(const metric of entry.metrics||[]){const row=document.createElement('p');row.className='result-metric';
      const label=document.createElement('span');label.textContent=metric.label;
      const value=document.createElement('strong');value.textContent=String(metric.value);
      row.append(label,value);parent.append(row);
      if(metric.scope){const scope=document.createElement('small');scope.textContent=metric.scope;parent.append(scope);}
    }
  }
  function showSummary(entry){
    summary.replaceChildren();const title=document.createElement('h3');title.textContent=entry.title;
    const description=document.createElement('p');description.textContent=entry.description;
    const badge=document.createElement('span');badge.className='eyebrow';badge.textContent='真实模型 · 测试结果';summary.append(badge,title,description);metrics(summary,entry);
    const details=document.createElement('details');const name=document.createElement('summary');name.textContent='结果说明';details.append(name);
    for(const text of entry.notes||[]){const note=document.createElement('p');note.textContent=text;details.append(note);}summary.append(details);summary.hidden=false;
  }
  async function select(entry){
    if(isBusy())return;dialog.close();loading.textContent=`正在载入 ${entry.title}…`;loading.hidden=false;
    try{await openScene(entry);showSummary(entry);const url=new URL(window.location.href);url.searchParams.set('result',entry.id);history.replaceState(null,'',url);toast('真实测试结果已载入，可以开始操作。');}
    catch(error){toast(`结果载入失败：${error.message}`,true);}
    finally{loading.hidden=true;}
  }
  button.onclick=()=>{if(!isBusy())dialog.showModal();};
  try{
    const data=await api('/api/results');entries=Array.isArray(data.results)?data.results:[];
    for(const entry of entries){const card=document.createElement('article');card.className='result-card';
      const title=document.createElement('h3');title.textContent=entry.title;const desc=document.createElement('p');desc.textContent=entry.description;
      card.append(title,desc);metrics(card,entry);const open=document.createElement('button');open.className='button primary';open.textContent=`打开 ${entry.title}`;open.onclick=()=>select(entry);card.append(open);list.append(card);
    }
    button.hidden=!entries.length;
    const requested=new URLSearchParams(window.location.search).get('result');
    if(requested){const entry=entries.find(item=>item.id===requested);if(entry)await select(entry);else toast('这个测试结果当前不可用，请从“测试结果”重新选择。',true);}
  }catch(error){if(new URLSearchParams(window.location.search).has('result'))toast(`无法读取测试结果：${error.message}`,true);}
  return {clear(){summary.hidden=true;const url=new URL(window.location.href);url.searchParams.delete('result');history.replaceState(null,'',url);}};
}
