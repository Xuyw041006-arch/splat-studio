// Viewer identifiers/translated aliases are not necessarily 2D mask labels.
export function trainingPriorityLabel(object){
  if(!object||object.level==='scene')return null;
  const label=object.semantic_key??object.canonical_label??object.label??object.name;
  return typeof label==='string'&&label.trim()?label.trim():null;
}
export function trainingPriorityLabels(selected,objects=[]){
  const byId=new Map(objects.map(object=>[String(object.id),object])),result=[],seen=new Set();
  for(const value of selected){
    const string=String(value),label=byId.has(string)?trainingPriorityLabel(byId.get(string)):string.trim();
    if(!label||seen.has(label.toLowerCase()))continue;
    seen.add(label.toLowerCase());result.push(label);
  }
  return result;
}
