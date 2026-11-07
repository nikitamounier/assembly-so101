export const TASKS=['motor','horn','screw','retainer','yoke'];
export const FAMILIES=['loose','pickup','inhand','partial','near'];
export const familyLabel=f=>({loose:'Loose',pickup:'Pickup',inhand:'In hand',partial:'Partial',near:'Near'})[f]||f;
export function reviewOf(reviews,id){return reviews[id]?.status||'unreviewed';}
export function filterEntries(entries,filters,reviews={}){
 const query=(filters.search||'').toLowerCase().trim();
 const filtered=entries.filter(e=>['bank','task','family','split','status'].every(k=>!filters[k]||e[k]===filters[k])&&(!filters.review||reviewOf(reviews,e.id)===filters.review)&&(!query||`${e.id} ${e.seed} ${e.task} ${e.family}`.toLowerCase().includes(query)));
 const order=filters.sort||'source';
 return filtered.sort((a,b)=>{
  if(order==='overlap'||order==='depth'){const key=order==='overlap'?'material_mm':'depth_mm';return (b[key]??-Infinity)-(a[key]??-Infinity)||a.id.localeCompare(b.id);}
  if(order==='seed')return a.seed-b.seed||a.id.localeCompare(b.id);
  return TASKS.indexOf(a.task)-TASKS.indexOf(b.task)||FAMILIES.indexOf(a.family)-FAMILIES.indexOf(b.family)||a.split.localeCompare(b.split)||a.seed-b.seed||a.id.localeCompare(b.id);
 });
}
export function visibleWindow(length,scrollTop,height,columns,rowHeight,overscan=2){
 const rows=Math.ceil(length/columns),first=Math.max(0,Math.floor(scrollTop/rowHeight)-overscan),last=Math.min(rows,Math.ceil((scrollTop+height)/rowHeight)+overscan);
 return {start:first*columns,end:Math.min(length,last*columns),offset:first*rowHeight,total:rows*rowHeight};
}
