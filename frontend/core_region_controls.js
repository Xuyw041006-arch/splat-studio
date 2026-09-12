export function createCoreRegionControls({parent,before,onToggle,onRange,onScene,onSelected,onAll}){
  const element=document.createElement('section');element.className='core-region-controls';element.hidden=true;element.setAttribute('aria-label','核心区域显示');
  const row=document.createElement('div');row.className='core-region-row';
  const toggleLabel=document.createElement('label');toggleLabel.className='core-region-toggle';
  const toggle=document.createElement('input');toggle.type='checkbox';toggle.setAttribute('aria-label','仅显示核心区域');
  toggleLabel.append(toggle,document.createTextNode('仅显示核心区域'));
  const rangeLabel=document.createElement('label');rangeLabel.className='core-region-range';rangeLabel.textContent='范围';
  const range=document.createElement('input');range.type='range';range.min='.25';range.max='4';range.step='.05';range.setAttribute('aria-label','核心区域范围倍率');
  const value=document.createElement('output');rangeLabel.append(range,value);
  const scene=document.createElement('button');scene.type='button';scene.className='button ghost compact';scene.textContent='场景核心';
  const selected=document.createElement('button');selected.type='button';selected.className='button ghost compact core-selected-object';selected.textContent='以所选物品为核心';
  const all=document.createElement('button');all.type='button';all.className='button secondary compact';all.textContent='显示全部';all.title='关闭范围、来源和语义可见性过滤；可以撤销。';
  row.append(toggleLabel,rangeLabel,scene,selected,all);
  const status=document.createElement('p');status.className='core-region-status';status.setAttribute('aria-live','polite');
  element.append(row,status);parent.insertBefore(element,before);
  const multiplierText=number=>`${Number(number).toFixed(2).replace(/0$/,'')}×`;
  toggle.onchange=()=>onToggle(toggle.checked);
  range.oninput=()=>{value.textContent=multiplierText(range.value);};
  // Commit once on release/keyboard change; dragging never launches repeated
  // million-point sorts or allocations while the thumb is moving.
  range.onchange=()=>onRange(Number(range.value));
  scene.onclick=onScene;selected.onclick=onSelected;all.onclick=onAll;
  return {clear(){element.hidden=true;},render({settings,selectedObject,objects=[],loaded,total,visible,coreIncluded,busy=false,pending=false}){
    if(!settings){element.hidden=true;return;}element.hidden=false;
    toggle.checked=settings.enabled;toggle.disabled=busy;
    if(document.activeElement!==range){range.value=String(settings.multiplier);value.textContent=multiplierText(settings.multiplier);}
    range.disabled=busy||!settings.enabled;scene.disabled=busy;selected.disabled=busy||!selectedObject;all.disabled=busy;
    scene.classList.toggle('active',settings.target==='scene');selected.classList.toggle('active',settings.target==='objects');
    selected.title=selectedObject?`以“${selectedObject.label||selectedObject.name||selectedObject.id}”为核心`:'先在右侧选择一个物品';
    const target=settings.target==='objects'?objects.filter(o=>settings.object_ids.includes(String(o.id))).map(o=>o.label||o.name||o.id).join('、'):'场景';
    const number=n=>Number(n||0).toLocaleString('zh-CN');
    status.textContent=`${settings.enabled?`${target}核心 · 范围内 ${number(coreIncluded)}`:'范围过滤已关闭'} · 当前可见 ${pending?'更新中':number(visible)} · 已载入 ${number(loaded)} / 来源 ${number(total)}。完整高斯与 SH 保留。`;
  }};
}
