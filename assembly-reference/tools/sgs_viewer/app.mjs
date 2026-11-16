import {TASKS,FAMILIES,familyLabel,filterEntries,visibleWindow,reviewOf} from './core.mjs';
const $=id=>document.getElementById(id);
const escape=s=>String(s??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
let manifest,filtered=[],selected=null,view='side',inspector,inspectorPromise,detailSequence=0,columns=2,rowHeight=240,framePending=false,reviews={},storageKey;
const details=new Map(),filterKeys=['bank','task','family','split','status','review','search','sort'];
const title=e=>`${e.task[0].toUpperCase()+e.task.slice(1)} / ${familyLabel(e.family)}`;
const filters=()=>Object.fromEntries(filterKeys.map(k=>[k,$(k).value]));
function failure(error){$('error').hidden=false;$('error').textContent=error.message||String(error);}
async function json(path){const response=await fetch(path);if(!response.ok)throw Error(`${path}: HTTP ${response.status}`);return response.json();}
function setURL(){const params=new URLSearchParams();for(const [key,value]of Object.entries(filters()))if(key==='bank'||(value&&key!=='sort'))params.set(key,value);if(selected)params.set('state',selected.id);history.replaceState(null,'',`${location.pathname}?${params}`);}
function layout(){const width=$('scroller').clientWidth;columns=Math.max(1,Math.floor((width-12)/215));rowHeight=(width-24-(columns-1)*12)/columns*.7+79;$('tiles').style.gridTemplateColumns=`repeat(${columns},minmax(0,1fr))`;drawGrid();}
function drawGrid(){
 const scroller=$('scroller'),window=visibleWindow(filtered.length,scroller.scrollTop,scroller.clientHeight,columns,rowHeight);
 $('spacer').style.height=`${window.total+24}px`;$('tiles').style.transform=`translateY(${window.offset}px)`;
 $('tiles').innerHTML=filtered.slice(window.start,window.end).map(e=>{
  const review=reviewOf(reviews,e.id),badge=e.status==='qualified'?e.split.toUpperCase():`${e.split.toUpperCase()} · BASELINE`;
  return `<button class="card ${selected?.id===e.id?'selected':''}" data-id="${e.id}" data-review="${review}" aria-pressed="${selected?.id===e.id}" style="height:${rowHeight-12}px" title="${escape(e.bank)} · ${e.id}"><img src="${escape(e.thumb)}" loading="lazy" decoding="async" alt="${escape(title(e))}, ${e.split}, seed ${e.seed}"><span class="badge">${badge}</span>${review!=='unreviewed'?`<span class="review-mark">${review==='flagged'?'⚑':'✓'}</span>`:''}<span class="caption"><strong>${escape(title(e))}</strong><span class="small"><span>seed ${e.seed}</span><span>${e.id.slice(0,7)}</span></span></span></button>`;
 }).join('');
 for(const image of $('tiles').querySelectorAll('img'))image.onerror=()=>image.closest('.card').classList.add('image-error');
 $('empty').hidden=filtered.length!==0;
}
function coverage(){
 const f=filters();f.task='';f.family='';const entries=filterEntries(manifest.entries,f,reviews);const counts=new Map();
 for(const e of entries){const k=`${e.task}/${e.family}`;const c=counts.get(k)||{train:0,eval:0};c[e.split]++;counts.set(k,c);}
 $('coverage-note').textContent='Train / Eval · click a cell to browse';
 $('coverage').innerHTML=`<table aria-label="Task and family coverage"><thead><tr><th>Task</th>${FAMILIES.map(f=>`<th>${familyLabel(f)}</th>`).join('')}</tr></thead><tbody>${TASKS.map(task=>`<tr><th>${task}</th>${FAMILIES.map(family=>{const c=counts.get(`${task}/${family}`)||{train:0,eval:0};return `<td><button data-task="${task}" data-family="${family}" class="${$('task').value===task&&$('family').value===family?'active':''}" title="${task} / ${family}: ${c.train} train, ${c.eval} eval">${c.train}<span>/ ${c.eval}</span></button></td>`;}).join('')}</tr>`).join('')}</tbody></table>`;
}
function reviewsSummary(){const ids=new Set(manifest.entries.map(e=>e.id));const relevant=Object.entries(reviews).filter(([id])=>ids.has(id));$('review-count').textContent=`${relevant.filter(([,r])=>r.status==='reviewed').length} reviewed · ${relevant.filter(([,r])=>r.status==='flagged').length} flagged`;}
function update(reset=true,keepSelection=true){
 filtered=filterEntries(manifest.entries,filters(),reviews);$('count').textContent=`${filtered.length.toLocaleString()} of ${manifest.entries.length.toLocaleString()} entries`;
 if(reset)$('scroller').scrollTop=0;coverage();reviewsSummary();
 const same=keepSelection&&selected&&filtered.find(e=>e.id===selected.id);
 select(same||filtered[0]||null,false);drawGrid();setURL();
}
function updateReviewButtons(){const status=selected?reviewOf(reviews,selected.id):'unreviewed';$('mark-reviewed').setAttribute('aria-pressed',status==='reviewed');$('mark-flagged').setAttribute('aria-pressed',status==='flagged');$('review-state').textContent=status;$('mark-reviewed').disabled=$('mark-flagged').disabled=!selected;}
async function getDetail(entry){
 if(details.has(entry.id))return details.get(entry.id);
 const value=await json(entry.detail);if(value.id!==entry.id)throw Error('Detail identity mismatch');details.set(entry.id,value);if(details.size>8)details.delete(details.keys().next().value);return value;
}
async function select(entry,scroll=true){
 selected=entry;const sequence=++detailSequence,index=filtered.findIndex(e=>e.id===entry?.id);
 $('selection-number').textContent=entry?`STATE ${index+1} OF ${filtered.length}`:'NO MATCHING STATES';$('selection-title').textContent=entry?title(entry):'No selection';
 $('previous').disabled=index<=0;$('next').disabled=index<0||index>=filtered.length-1;updateReviewButtons();drawGrid();setURL();
 if(!entry){$('preview').hidden=true;$('three').hidden=true;$('visual-message').textContent='Adjust the filters to see states.';$('facts').replaceChildren();$('metadata').textContent='No state selected.';$('original').hidden=true;return;}
 if(scroll){const row=Math.floor(index/columns),top=row*rowHeight,bottom=top+rowHeight,s=$('scroller');if(top<s.scrollTop||bottom>s.scrollTop+s.clientHeight)s.scrollTop=top;}
 const mm=value=>value==null?'Not recorded':`${value.toFixed(4)} mm`;
 const facts=[['Collection',entry.bank],['Split / family',`${entry.split.toUpperCase()} / ${familyLabel(entry.family)}`],['Qualification',entry.status==='qualified'?'Saved qualification passed':'Older all-loose baseline; no prepared SGS receipt'],['Seed / arm',`${entry.seed} / ${entry.side||'not specified'}`],['Material overlap',mm(entry.material_mm)],['Native depth',mm(entry.depth_mm)],['State ID',entry.id],['Source entry',`${entry.archive} [${entry.archive_index}]`]];
 $('facts').innerHTML=facts.map(([key,value])=>`<dt>${key}</dt><dd>${escape(value)}</dd>`).join('');$('metadata').textContent='Loading saved metadata…';
 try{await showView();const detail=await getDetail(entry);if(sequence!==detailSequence)return;$('metadata').textContent=JSON.stringify(detail.metadata,null,2);}catch(error){if(sequence===detailSequence){$('visual-message').textContent=error.message;failure(error);}}
}
async function showView(){
 const entry=selected;if(!entry)return;const wanted=view;
 for(const button of document.querySelectorAll('[data-view]'))button.setAttribute('aria-pressed',button.dataset.view===view);
 $('preview').hidden=view==='3d';$('three').hidden=view!=='3d';$('focus-3d').hidden=$('reset-3d').hidden=view!=='3d';$('original').hidden=view==='3d';$('visual-message').textContent='';
 if(view!=='3d'){
  $('preview').onerror=()=>{$('visual-message').textContent='Image could not be loaded.';};$('preview').src=entry.images[view];$('preview').alt=`${title(entry)} · ${view} · ${entry.split} · seed ${entry.seed}`;$('original').href=entry.images[view];$('view-hint').textContent=view==='detail'?'Close-up · all geometry retained':'Native MuJoCo snapshot';return;
 }
 $('view-hint').textContent='Drag to orbit · wheel to zoom · Shift-drag to pan';$('visual-message').textContent='Loading native geometry…';
 if(!inspectorPromise)inspectorPromise=(async()=>{const {SnapshotInspector}=await import('./inspect.mjs');const instance=new SnapshotInspector($('three'));await instance.load(manifest.geometry);return instance;})().catch(error=>{inspectorPromise=null;throw error;});
 const [instance,detail]=await Promise.all([inspectorPromise,getDetail(entry)]);inspector=instance;
 if(selected?.id!==entry.id||view!==wanted)return;inspector.show(detail);$('visual-message').textContent='';
}
function mark(status){if(!selected)return;const id=selected.id;if(reviewOf(reviews,id)===status)delete reviews[id];else reviews[id]={status,updated_utc:new Date().toISOString()};try{localStorage.setItem(storageKey,JSON.stringify(reviews));}catch{failure(Error('Browser storage unavailable; export reviews to preserve them.'));}update(false);}
async function start(){
 manifest=await json('manifest.json');if(new Set(manifest.entries.map(e=>e.id)).size!==manifest.entries.length)throw Error('Export contains duplicate state IDs');
 storageKey=`sgs-reviews-v1:${manifest.environment.model_signature}`;try{const saved=JSON.parse(localStorage.getItem(storageKey)||'{}');if(saved&&typeof saved==='object'&&!Array.isArray(saved))for(const [id,row]of Object.entries(saved))if(row&&['reviewed','flagged'].includes(row.status))reviews[id]=row;}catch{}
 $('profile').textContent=manifest.environment.profile;$('total').textContent=`${manifest.entries.length.toLocaleString()} saved entries · ${(manifest.entries.length*3).toLocaleString()} images`;
 for(const bank of manifest.banks){const label=bank.distribution==='all_loose'?'Loose baseline':'Qualified SGS';$('bank').add(new Option(`${label} · ${bank.entries} · ${bank.name}`,bank.name));}for(const task of TASKS)$('task').add(new Option(task,task));for(const family of FAMILIES)$('family').add(new Option(familyLabel(family),family));
 const params=new URLSearchParams(location.search);$('bank').value=params.has('bank')?params.get('bank'):manifest.banks[0]?.name||'';
 for(const key of filterKeys)if(params.has(key))$(key).value=params.get(key);
 const linked=manifest.entries.find(e=>e.id===params.get('state'));if(linked)selected=linked;
 $('export-info').textContent=`Exported ${manifest.created_utc}. Model ${manifest.environment.model_signature}. ${manifest.banks.map(b=>`${b.name}: ${b.entries} entries`).join('; ')}. Loose-only evaluation aliases are counted once. Metrics come from saved qualification receipts, not a new simulation.`;
 for(const bank of manifest.excluded){const item=document.createElement('li');item.textContent=`${bank.name}: ${bank.reason}`;$('excluded').append(item);}
 for(const key of filterKeys)$(key).addEventListener(key==='search'?'input':'change',()=>update());
 $('clear').onclick=()=>{for(const key of filterKeys)$(key).value=key==='sort'?'source':'';update();};
 $('tiles').onclick=event=>{const button=event.target.closest('[data-id]');if(button)select(filtered.find(e=>e.id===button.dataset.id));};
 $('coverage').onclick=event=>{const button=event.target.closest('[data-task]');if(button){$('task').value=button.dataset.task;$('family').value=button.dataset.family;update();}};
 $('scroller').addEventListener('scroll',()=>{if(!framePending){framePending=true;requestAnimationFrame(()=>{framePending=false;drawGrid();});}},{passive:true});
 new ResizeObserver(layout).observe($('scroller'));
 const next=delta=>{const index=filtered.findIndex(e=>e.id===selected?.id),entry=filtered[index+delta];if(entry)select(entry);};
 $('previous').onclick=()=>next(-1);$('next').onclick=()=>next(1);
 document.addEventListener('keydown',event=>{if(/INPUT|SELECT|TEXTAREA|BUTTON/.test(event.target.tagName)||event.ctrlKey||event.metaKey)return;if(event.key==='ArrowRight'){event.preventDefault();next(1);}if(event.key==='ArrowLeft'){event.preventDefault();next(-1);}});
 for(const button of document.querySelectorAll('[data-view]'))button.onclick=()=>{view=button.dataset.view;showView().catch(error=>{failure(error);$('visual-message').textContent='3D unavailable. Side, top and close-up images remain available.';});};
 $('focus-3d').onclick=()=>inspector?.focus();$('reset-3d').onclick=()=>inspector?.reset();$('mark-reviewed').onclick=()=>mark('reviewed');$('mark-flagged').onclick=()=>mark('flagged');
 $('export-reviews').onclick=()=>{const ids=new Set(manifest.entries.map(e=>e.id)),current=Object.fromEntries(Object.entries(reviews).filter(([id])=>ids.has(id)));const blob=new Blob([JSON.stringify({model:manifest.environment.model_signature,exported_utc:new Date().toISOString(),reviews:current},null,2)],{type:'application/json'}),link=document.createElement('a');link.href=URL.createObjectURL(blob);link.download='sgs-reviews.json';link.click();setTimeout(()=>URL.revokeObjectURL(link.href),1000);};
 update();layout();
}
start().catch(failure);
