import './style.css';
import { latestTaskReplay, replayForSelection } from './task-replays.mjs';
import { ThreeSceneRenderer } from '../assembly-so101/web/src/viewer.js';
const $=id=>document.getElementById(id);
const tasks=['base_side_cover','shoulder_connector','assembly_pin','upper_arm_module','forearm_module'];
const names=['Base side cover','Shoulder connector','Assembly pin','Upper arm module','Forearm module'];
const families=['loose','pickup','inhand','partial','near'];
const familyNames=['Loose','Pickup','In hand / transport','Partial insertion','Near seated'];
const colors=['#608b72','#c49a4d','#b86d60','#8e79aa','#528f9b','#ac8062','#7f9046','#5674ab'];
const fmt=(v,d=0)=>v==null||!Number.isFinite(Number(v))?'—':Number(v).toLocaleString('en-US',{maximumFractionDigits:d});
const pct=v=>v==null?'—':(100*v).toFixed(1)+'%';
const esc=v=>String(v??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
let workspace={states:{},training:{}},workspaceRevision=null;
let live={history:[]},scope='stage',selected=null,metric='success',renderer,previewRequest=0,previewTask=tasks[0],selectedTask=tasks[0];
let episode=null,currentStage=null,previewPoses=null,previewScene=null,sceneMode=null,recordedScene=null,replayRequest=0,loadingReplay=null,replayMode=true,playing=false,playTime=0,lastFrameIndex=-1,lastTick=0,lastRender=0,dirty=2,sceneVisible=true;
let previewSceneRequest;
async function fetchScene(path){
 const response=await fetch(path,{signal:AbortSignal.timeout(60000)});
 if(!response.ok)throw Error('Scene unavailable');
 const scene=await response.json();
 if(scene.mesh_encoding==='base64-le32'){
  const bytes=value=>Uint8Array.from(atob(value),c=>c.charCodeAt(0));
  for(const mesh of Object.values(scene.meshes)){
   mesh.vertices=new Float32Array(bytes(mesh.vertices).buffer);
   mesh.faces=Array.from(new Uint32Array(bytes(mesh.faces).buffer));
  }
 }
 return scene;
}
const stageName=stage=>({'loose':'Loose scene','pickup':'Pickup','floor_grasp':'Floor grasp','aligned':'Aligned','entry':'Entry','near':'Near seated','left_support':'Left support','right_support':'Right support','transport':'Transport','far_side':'Far-side support','half':'Half inserted'}[Object.keys({'left_support':1,'right_support':1,'floor_grasp':1,'far_side':1,'transport':1,'aligned':1,'pickup':1,'entry':1,'near':1,'half':1,'loose':1}).find(k=>stage.includes(k))]||stage.replace(/^\d+_/,'').replaceAll('_',' '));
function stageRows(){
 const actual=(live.starting_states?.groups||[]).filter(g=>g.task===selectedTask);
 const catalog=Object.entries(live.starting_catalog||{}).filter(([stage,row])=>stage==='00_loose'||row.task===selectedTask).map(([stage])=>stage);
 return [...new Set([...catalog,...actual.map(g=>g.stage)])].sort().map(stage=>actual.find(g=>g.stage===stage)||{stage,task:selectedTask,entries:0,prototypes:0,variations:0});
}
const keys=()=>scope==='stage'?stageRows().map(r=>r.stage):scope==='task'?tasks:families;
const labels=()=>scope==='stage'?keys().map(stageName):scope==='task'?names:familyNames;
const allRows=()=>[...(live.previous_history||[]),...(live.history||[])];
const field=()=>scope+'_'+(metric==='return'?'return':'success');
const overall=()=>['fps','safety','kl'].includes(metric);
const plotRows=()=>scope==='stage'&&!overall()?(live.stage_history||[]).filter(r=>r.run===live.run&&r.iteration!=null).map(r=>({it:r.iteration,version:r.version,groups:r.groups})):allRows();
function points(key){
 const rows=plotRows();
 const value=r=>scope==='stage'&&!overall()?r.groups?.find(g=>g.task===selectedTask&&g.stage===key)?.success:overall()?r[metric]:r[field()]?.[key];
 return rows.map((r,i)=>{
  const v=value(r);
  if(v==null||!Number.isFinite(v))return {x:r.it,y:null,version:r.version};
  const values=$('smooth').checked?rows.slice(Math.max(0,i-4),i+1).filter(p=>p.version===r.version).map(value).filter(n=>n!=null&&Number.isFinite(n)):[v];
  return {x:r.it,y:values.reduce((a,b)=>a+b,0)/values.length,version:r.version};
 });
}
function path(points,x,y){let gap=true,version;return points.map(p=>{if(version!==p.version)gap=true;version=p.version;if(p.y==null){gap=true;return '';}const cmd=gap?'M':'L';gap=false;return `${cmd}${x(p.x).toFixed(2)},${y(p.y).toFixed(2)}`;}).join(' ');}
function chart(){
 const node=$('chart'),w=Math.max(320,node.clientWidth-28),h=260,l=49,r=16,t=23,b=34;
 const chosen=overall()?['overall']:selected==null?keys():[keys()[selected]];
 const series=chosen.filter(Boolean).map(k=>({key:k,points:points(k)})),valid=series.flatMap(s=>s.points).filter(p=>p.y!=null),rows=plotRows();
 let xmin=rows[0]?.it||0,xmax=rows.at(-1)?.it||1,ymin=valid.length?Math.min(...valid.map(p=>p.y)):0,ymax=valid.length?Math.max(...valid.map(p=>p.y)):1;
 if(xmin===xmax){xmin-=1;xmax+=1;}
 if(['success','safety'].includes(metric)){ymin=0;ymax=1;}else{const pad=Math.max((ymax-ymin)*.15,.001);ymin=Math.max(0,ymin-pad);ymax+=pad;}
 const x=v=>l+(v-xmin)/(xmax-xmin||1)*(w-l-r),y=v=>t+(ymax-v)/(ymax-ymin||1)*(h-t-b);
 let svg=`<svg viewBox="0 0 ${w} ${h}" aria-hidden="true">`;
 for(let i=0;i<=4;i++){const v=ymin+(ymax-ymin)*i/4,yy=y(v);svg+=`<line x1="${l}" x2="${w-r}" y1="${yy}" y2="${yy}" stroke="#e2e7dc" stroke-dasharray="3 4"/><text x="${l-8}" y="${yy+3}" text-anchor="end" fill="#899780" font-size="9">${['success','safety'].includes(metric)?Math.round(v*100)+'%':fmt(v,metric==='kl'?3:1)}</text>`;}
 for(let i=0;i<=4;i++){const v=xmin+(xmax-xmin)*i/4;svg+=`<text x="${x(v)}" y="${h-10}" text-anchor="middle" fill="#899780" font-size="9">${fmt(Math.round(v))}</text>`;}
 const boundary=live.migration?.parent_it??live.rollback_origin?.it;
 if(boundary!=null&&boundary>=xmin&&boundary<=xmax)svg+=`<line x1="${x(boundary)}" x2="${x(boundary)}" y1="${t}" y2="${h-b}" stroke="#ad8d57" stroke-dasharray="5 4"/><text x="${Math.min(w-145,x(boundary)+5)}" y="12" fill="#96743e" font-size="9">Run boundary · ${fmt(boundary)}</text>`;
 for(const s of series){const col=overall()?'#608b72':colors[keys().indexOf(s.key)];svg+=`<path d="${path(s.points,x,y)}" fill="none" stroke="${col}" stroke-width="2" stroke-linejoin="round"/>`;for(const p of s.points.filter(p=>p.y!=null))svg+=`<circle cx="${x(p.x)}" cy="${y(p.y)}" r="2.5" fill="${col}"/>`;}
 node.innerHTML=svg+'</svg>'+(!valid.length?'<span class="chart-empty">'+(metric==='return'?'Episode returns are not logged by this learner':scope==='stage'?'No sampled outcomes for this starting state yet':'Waiting for the first committed update')+'</span>':'')+'<div id="tooltip" class="chart-tooltip" hidden></div>';
 node.onpointermove=e=>{if(!rows.length)return;const v=xmin+Math.max(0,Math.min(1,(e.offsetX-l)/(w-l-r)))*(xmax-xmin);const row=rows.reduce((a,b)=>Math.abs(a.it-v)<Math.abs(b.it-v)?a:b);const tip=$('tooltip');tip.textContent=`${row.version} · iteration ${fmt(row.it)} · `+chosen.map(k=>{const val=scope==='stage'&&!overall()?row.groups?.find(g=>g.task===selectedTask&&g.stage===k)?.success:overall()?row[metric]:row[field()]?.[k];return `${overall()?metric:labels()[keys().indexOf(k)]}: ${['success','safety'].includes(metric)?pct(val):fmt(val,3)}`;}).join(' / ');tip.hidden=false;};node.onpointerleave=()=>{$('tooltip').hidden=true;};
 $('history-note').textContent=scope==='stage'&&!overall()?`${fmt(rows.length)} saved sampler snapshots · last ${live.starting_states?.history_per_entry||50} outcomes / entry`:(rows.length?`${fmt(rows.length)} committed updates · run versions kept separate`:'Waiting for training data');
 $('chart-title').textContent=({success:scope==='stage'?`Starting states · ${names[tasks.indexOf(selectedTask)]}`:`Success by ${scope==='task'?'assembly task':'reset family'}`,return:'Episode returns',safety:'Safety termination rate',fps:'Valid rollout throughput',kl:'Policy KL divergence'})[metric];
 $('legend').innerHTML=overall()?'<span class="muted">Overall active run</span>':keys().map((k,i)=>`<button data-series="${i}" style="--color:${colors[i]}" class="${selected!=null&&selected!==i?'inactive':''}" aria-pressed="${selected===i}"><i></i>${labels()[i]}</button>`).join('');
 $('legend').querySelectorAll('button').forEach(btn=>btn.onclick=()=>{selected=selected===Number(btn.dataset.series)?null:Number(btn.dataset.series);if(selected!=null&&(scope==='task'||scope==='stage'))setPreview(keys()[selected]);render();});
}
function taskCards(){
 const window=(live.history||[]).slice(-20);
 $('tasks').innerHTML=tasks.map((task,i)=>{
  let n=0,won=0;for(const row of window){const count=row.task_episodes?.[task]||0;if(count&&row.task_success?.[task]!=null){n+=count;won+=count*row.task_success[task];}}
  const ps=(live.history||[]).map(row=>({x:row.it,y:row.task_success?.[task],version:row.version})).filter(p=>p.y!=null).slice(-80);
  const spark=ps.length?path(ps,v=>2+(v-ps[0].x)/(ps.at(-1).x-ps[0].x||1)*94,v=>30-v*25):'';
  return `<button class="family ${selectedTask===task?'selected':''}" data-task="${i}" style="--color:${colors[i]}" aria-pressed="${selectedTask===task}"><div class="family-title"><span><i></i>${names[i]}</span><small>0${i+1}</small></div><div class="family-values"><div><strong>${n?pct(won/n):'—'}</strong><small>EPISODE SUCCESS</small></div><svg viewBox="0 0 100 34" aria-hidden="true"><path d="${spark}" fill="none" stroke="${colors[i]}" stroke-width="1.5"/></svg></div><div class="family-foot"><span>${fmt(n)} completed episodes</span><span>View starting states ↗</span></div></button>`;
 }).join('');
 $('tasks').querySelectorAll('button').forEach(b=>b.onclick=()=>selectTask(tasks[Number(b.dataset.task)]));
}
function selectTask(task,{scroll=false}={}){selectedTask=task;scope='stage';selected=null;const loading=setPreview(task);render();if(scroll)document.querySelector('.replay-panel').scrollIntoView({behavior:'smooth',block:'start'});return loading;}
const taskReplay=task=>latestTaskReplay(live.replays,task,live.run,live.model_signature);
function taskRollouts(){
 const node=$('task-rollouts');
 const revision=[live.run,selectedTask,...tasks.map(task=>taskReplay(task)?.id||'')].join('|');
 if(node.dataset.revision===revision)return;node.dataset.revision=revision;
 node.innerHTML=tasks.map((task,i)=>{
  const clip=taskReplay(task),outcome=clip?(clip.success?'Latest success':'Latest failure'):'No recording yet';
  return `<button class="task-rollout ${clip?(clip.success?'won':'failed'):''}" data-latest-task="${task}" ${clip?'':'disabled'} aria-label="${esc(names[i]+' · '+outcome)}"><span class="rollout-name">${names[i]}</span><span class="rollout-outcome">${clip?'▶ ':''}${outcome}</span><span class="rollout-detail">${clip?esc(stageName(clip.stage))+' · '+clip.duration_s.toFixed(2)+' s':'Waiting for a complete episode'}</span>${clip?`<small>Iteration ${fmt(clip.iteration)} · ${esc(new Date(clip.recorded_at*1000).toLocaleTimeString())}</small>`:''}</button>`;
 }).join('');
 node.querySelectorAll('button[data-latest-task]').forEach(button=>button.onclick=()=>{replayMode=true;$('view-mode').textContent='View reset';$('view-mode').setAttribute('aria-pressed','false');selectTask(button.dataset.latestTask,{scroll:true});});
}
function episodeStateSelector(){
 const node=$('preview-state'),rows=stageRows();
 const choices=[...new Set([...rows.map(row=>row.stage),...Object.values(live.replays||{}).filter(clip=>clip.task===selectedTask).map(clip=>clip.stage)])].sort();
 const revision=[selectedTask,...choices.map(stage=>stage+':'+(live.replays?.[selectedTask+'/'+stage]?.id||''))].join('|');
 if(node.dataset.revision!==revision){
  node.dataset.revision=revision;
  node.innerHTML='<option value="">Automatic · latest episode for this task</option>'+choices.map(stage=>{
   const clip=replayForSelection(live.replays,selectedTask,stage,live.run,live.model_signature);
   const status=clip?(clip.success?'success available':'failure available'):'no episode yet';
   return `<option value="${esc(stage)}">${esc(stage)} · ${status}</option>`;
  }).join('');
 }
 node.value=tasks.includes(previewTask)?'':previewTask;
}
function startingTable(){
 const rows=stageRows(),stats=live.starting_states;
 $('starting-title').textContent='Starting states · '+names[tasks.indexOf(selectedTask)];
 $('starting-count').textContent=`${rows.filter(r=>r.entries>0).length} active starts · ${fmt(rows.reduce((n,r)=>n+r.entries,0))} TRAIN entries`;
 $('starting-tabs').innerHTML=tasks.map((t,i)=>`<button data-environment="${t}" aria-pressed="${selectedTask===t}">${names[i]}</button>`).join('');
 $('starting-tabs').querySelectorAll('button').forEach(b=>b.onclick=()=>selectTask(b.dataset.environment));
 $('starting-rows').innerHTML=rows.map((r,i)=>{const clip=live.replays?.[selectedTask+'/'+r.stage];return `<tr class="${selected!=null&&scope==='stage'&&keys()[selected]===r.stage?'selected':''}"><td><button class="state-select" data-stage="${r.stage}"><i style="background:${colors[i]}"></i><span>${esc(stageName(r.stage))}<small>${esc(r.stage)}${r.entries?'':' · not in active bank'}</small></span></button></td><td>${fmt(r.entries)}<small>${fmt(r.prototypes)} nominal · ${fmt(r.variations)} variations${r.eval_entries!=null?' · '+fmt(r.eval_entries)+' EVAL held out':''}</small></td><td>${r.sampled_episodes==null?'—':fmt(r.sampled_episodes)}</td><td><b>${pct(r.success)}</b><small>${r.history_episodes!=null?fmt(r.history_wins)+' / '+fmt(r.history_episodes)+' retained outcomes':'Awaiting sampler data'}</small></td><td>${r.sampling_probability==null?'—':pct(r.sampling_probability)}</td><td>${r.visited_entries==null?'—':fmt(r.visited_entries)+' / '+fmt(r.entries)}</td><td><span class="replay-state ${clip?(clip.success?'won':'failed'):''}">${clip?(clip.success?'Success ▶':'Failure ▶'):'Waiting'}</span></td></tr>`;}).join('');
 $('starting-rows').querySelectorAll('button').forEach(b=>b.onclick=()=>{scope='stage';selected=stageRows().findIndex(r=>r.stage===b.dataset.stage);setPreview(b.dataset.stage);render();});
 $('starting-note').textContent=stats?.sampler_available?`Saved sampler at ${new Date(stats.sampler_saved_at*1000).toLocaleTimeString()}. Success uses the last ${stats.history_per_entry} outcomes per reset entry, weighted by retained outcome counts. Episodes sampled is the lifetime count in this sampler version. Sampling share is across all task/state entries.`:'Bank entries shown now; per-start outcomes are read from the small saved sampler artifact every 60 seconds.';
}
function render(){
 const last=live.history?.at(-1),fresh=live.connection==='live'&&Date.now()/1000-(live.received_at||0)<65;
 const stopped=live.phase==='user_stopped';
 const restoring=workspace.training?.state==='restoring',paused=stopped||['paused','restoring'].includes(workspace.training?.state);
 const handoff=live.handoff?.status==='intentional_handoff'&&live.handoff.old_run===live.run;
 const delayed=(live.training_age_s??0)>1200;
 $('status-dot').className='dot '+(fresh&&!delayed&&!handoff&&!paused?'live':'stale');
 $('status').textContent=stopped?`Training stopped · checkpoint ${fmt(last?.it)}`:paused?`${restoring?'Restoring training':'Training paused'} · checkpoint ${fmt(workspace.training.checkpoint)}`:handoff?'Checkpoint handoff in progress':!fresh?'Cluster observation unavailable':delayed?'Training update delayed':'Live from Chimera';
 $('updated').textContent=live.received_at?'Observed '+new Date(live.received_at*1000).toLocaleTimeString():'Awaiting data';
 $('iteration').textContent=fmt(last?.it??live.migration?.parent_it??live.rollback_origin?.it);$('transitions').textContent=fmt(last?.transitions??live.migration?.transitions??live.rollback_origin?.transitions);$('fps').textContent=fmt(last?.fps,1);
 $('phase').textContent=stopped?'Checkpoint saved · stopped by user':paused?(restoring?'Checkpoint saved · restoring training':'Checkpoint saved · training paused'):handoff?(live.learner_closed?'Checkpoint saved · preparing successor':'Finishing current PPO update'):last?`${live.version} · committed PPO update`:(live.migration||live.rollback_origin)?'Restored checkpoint · awaiting update':'Awaiting learner';
 $('envs').textContent=stopped?`0 / ${fmt(live.target_envs)} · stopped`:last?.workers?.active_envs!=null?`${fmt(last.workers.active_envs,0)} / ${fmt(live.target_envs)}`:'—';
 const runJobs=live.cpu_jobs?.filter(j=>j.state==='RUNNING')||[],pending=live.cpu_jobs?.filter(j=>j.state==='PENDING')||[];
 $('chimera').textContent=live.cpu_jobs?fmt(runJobs.reduce((n,j)=>n+j.cpus,0)):'—';$('chimera-note').textContent=`${runJobs.length} running leases · ${pending.length} pending · Slurm CPU units`;
 const modalFresh=live.modal?.status==='observed'&&Date.now()/1000-live.modal.observed_at<100;
 $('modal').textContent=modalFresh?fmt(live.modal.containers):'—';$('modal-note').textContent=modalFresh?'Observed containers, including any relay':'Modal API observation unavailable';
 if(live.experiment==='bc'){const sail=live.sail||{};$('modal').textContent=fmt(sail.reserved_cpus);$('modal-note').textContent=`${fmt(sail.finite_workers)} workers have returned finite controls`; }
 const req=live.modal_request,appLive=live.modal?.apps?.some(a=>a.app_id===req?.app_id);
 $('capacity-note').textContent=appLive&&req?.status==='workers_dispatched'?`${fmt(req.requested_physical_cores)} physical rollout cores requested. Actual contributing environments are measured above.`:'Modal goal: 4,092 physical rollout cores. Requested cores and live containers do not prove connected rollout capacity.';
 if(live.experiment==='bc')$('capacity-note').textContent=`500 Sail CPU target, including relay. Pool: ${live.sail?.status||'starting'}. Actual contributing environments are measured above.`;
 $('batch').textContent=last?`${fmt(last.valid_transitions)} / ${fmt(last.target_transitions)}`:'—';$('batch-note').textContent=`${fmt(live.steps||32)} steps / environment · ${fmt(live.worker_slots)} worker slots`;
 $('safety').textContent=pct(last?.safety);$('gpu').textContent=`GPU lease ${live.gpu_job||'—'} · one learner`;
 const rollback=live.rollback_origin;
 const continuation=handoff||live.migration||rollback;
 $('continuation').hidden=!continuation;
 $('continuation-title').textContent=handoff?'Preparing checkpoint-safe continuation':rollback?`Restored native checkpoint ${fmt(rollback.it)} · simplified-physics updates excluded`:`Continued from checkpoint ${fmt(live.migration?.parent_it)} · policy and optimizer preserved`;
 $('continuation-note').textContent=rollback?'Original native contacts, timestep and guards restored. Policy and optimizer resume from the pre-change checkpoint; rejected simplified-physics updates are excluded from this lineage.':handoff?(live.learner_closed?'The old learner has closed with its checkpoint saved. Retained Chimera leases stay owned while the successor is prepared.':'The current learner is finishing a checkpoint-safe update. Retained Chimera leases stay owned; prepared successor metrics are not shown as live training.'):'Bank and batch changes mark a new chart segment. Evaluation states stay held out. Inherited checkpoint counters are not a new PPO update.';
 if(paused){$('fps').textContent=restoring?'Restoring':'Paused';$('envs-caption').textContent='Last committed rollout · training paused';}else{$('envs-caption').textContent='Measured in the last rollout';}
 const warning=!fresh?'Showing the last received snapshot. The read-only cluster connection will retry.':live.errors?.length?'Worker launcher needs attention: '+live.errors.join(', '):delayed&&!handoff&&!paused?'No committed PPO update in over 20 minutes.':null;
 $('warning').hidden=!warning;$('warning').textContent=warning||'';
 $('run-path').textContent=live.run||'Waiting for the active run…';
 $('identity-chips').innerHTML=[`${live.version||'—'} · diagnostic`,`${fmt(live.training_rows)} task-conditioned TRAIN rows`,`${fmt(live.prototypes)} source starts`,`${fmt(live.checkpoint_bytes/1e6,1)} MB checkpoint`].map(v=>`<span>${esc(v)}</span>`).join('');
 $('identity-details').innerHTML=[['Model signature',live.model_signature],['Reset bank',live.bank],['Reset schema',live.bank_schema],['Checkpoint saved',live.checkpoint_saved_at?new Date(live.checkpoint_saved_at*1000).toLocaleString():null],['Admission',live.qualification]].map(([k,v])=>`<dt>${esc(k)}</dt><dd>${esc(v||'Not available')}</dd>`).join('');
 const option=$('metric').querySelector('[value="return"]');option.disabled=!live.available_metrics?.[scope+'_return'];option.textContent=option.disabled?'Episode return · not logged':'Episode return';
 document.querySelectorAll('[data-scope]').forEach(b=>b.setAttribute('aria-pressed',String(b.dataset.scope===scope)));
 $('all').textContent=scope==='stage'?'Show all starts':scope==='task'?'Show all tasks':'Show all families';chart();taskCards();taskRollouts();startingTable();episodeStateSelector();
}
async function setPreview(task){
 previewTask=task;episode=null;playing=false;currentStage=null;previewPoses=null;loadingReplay=null;replayRequest++;const isTask=tasks.includes(task);if(isTask)selectedTask=task;const owner=isTask?task:selectedTask;$('preview-task').value=owner;$('scene-title').textContent=names[tasks.indexOf(owner)]+(isTask?'':' · '+stageName(task));const request=++previewRequest;
 episodeStateSelector();
 const preferred=isTask?taskReplay(task):null;const previewKey=preferred?.stage||task;
 try{const response=await fetch('/api/preview?task='+encodeURIComponent(previewKey),{signal:AbortSignal.timeout(15000)});if(!response.ok)throw Error();const preview=await response.json();if(request!==previewRequest)return;currentStage=preview.stage;previewPoses=preview.poses;if(!replayMode)showReset();void syncReplay();}catch{if(request===previewRequest)$('scene-loading').textContent='Assembly preview unavailable';}
}
function showReset(){
 if(!renderer||!previewPoses)return;
 if(!previewScene){
  if(!previewSceneRequest)previewSceneRequest=fetchScene('/api/scene?format=compact').then(scene=>{previewScene=scene;if(!episode)showReset();}).catch(()=>{if(!episode)$('scene-loading').textContent='3D preview unavailable';}).finally(()=>{previewSceneRequest=null;});
  return;
 }
 if(sceneMode!=='preview'){renderer.setScene(previewScene);sceneMode='preview';}
 renderer.setSnapshot({poses:previewPoses});dirty=2;
 $('scene-loading').hidden=true;
 $('preview-stage').textContent=(currentStage||'').replaceAll('_',' ');$('episode-return').textContent='Drag to orbit · scroll to zoom';
 $('episode-badge').textContent=replayMode?'Waiting for episode':'Reset preview';$('episode-badge').classList.remove('failed');
 document.querySelector('.scene-tag').textContent='RESET PREVIEW · NOT A POLICY ROLLOUT';
 $('play').disabled=true;$('scrub').disabled=true;$('time').textContent='0.00 / 0.00s';
 $('replay-title').textContent=live.recording?'Waiting for a complete training episode':'Replay recorder awaiting activation';
 $('replay-note').textContent='This is the approved reset. A saved success will be played first; until one exists, a saved complete failure is used. Initial partial episodes and lost-worker fragments are excluded.';
}
async function syncReplay(){
 if(!replayMode||!currentStage||!renderer||document.hidden||!sceneVisible)return;
 const task=tasks.includes(previewTask)?previewTask:selectedTask;
 const preferred=tasks.includes(previewTask)?taskReplay(task):null;
 if(preferred&&preferred.stage!==currentStage){await setPreview(previewTask);return;}
 const meta=preferred||replayForSelection(live.replays,task,currentStage,live.run,live.model_signature);
 if(!meta){if(live.run){episode=null;playing=false;showReset();}return;}
 const key=live.run+'/'+task+'/'+currentStage;
 if(episode?.id===meta.id||loadingReplay?.key===key)return;
 const request=++replayRequest;loadingReplay={key,request};$('episode-badge').textContent='Loading episode…';
 try{
  const response=await fetch('/api/episode?task='+encodeURIComponent(task)+'&stage='+encodeURIComponent(currentStage),{cache:'no-store',signal:AbortSignal.timeout(55000)});
  if(response.status===202){if(request===replayRequest)$('episode-badge').textContent='Waiting for episode…';return;}
  if(!response.ok)throw Error('Episode temporarily unavailable');
  const clip=await response.json();if(!clip||request!==replayRequest)return;
  if(clip.task!==task||clip.stage!==currentStage||clip.identity.run!==live.run||!clip.complete_episode||clip.source!=='actual_training_episode')throw Error('Episode identity mismatch');
  if(recordedScene?.run!==clip.identity.run){
   $('episode-badge').textContent='Loading 3D model…';
   const scene=await fetchScene('/api/replay-scene?format=compact&run='+encodeURIComponent(clip.identity.run));
   if(request!==replayRequest)return;
   recordedScene={run:clip.identity.run,scene};
  }
  if(request!==replayRequest)return;
  if(sceneMode!=='replay'||renderer.replayRun!==clip.identity.run){renderer.setScene(recordedScene.scene);renderer.replayRun=clip.identity.run;}
  sceneMode='replay';episode=clip;playTime=0;lastFrameIndex=-1;playing=true;dirty=2;$('scene-loading').hidden=true;
  $('episode-badge').textContent=clip.success?'Successful episode':clip.terminal_reason==='safety'?'Guard termination':'Failed episode';$('episode-badge').classList.toggle('failed',!clip.success);
  document.querySelector('.scene-tag').textContent='ACTUAL TRAINING EPISODE · '+(clip.success?'SUCCESS':String(clip.terminal_reason==='safety'?'GUARD TERMINATION':clip.terminal_reason||'FAILURE').toUpperCase());
  $('preview-stage').textContent=stageName(clip.stage)+' · iteration '+fmt(clip.iteration);
  $('episode-return').textContent='Episode return '+Number(clip.episode_return).toFixed(4);
  $('replay-title').textContent=tasks.includes(previewTask)?(clip.success?'Latest saved success for this task':'Latest saved failure · no success saved for this task'):(clip.success?'Latest saved success from this starting state':'Latest saved failure · no successful episode saved for this start');
  const guardNote=clip.terminal_reason==='safety'?' Guard triggered; the exact cause was not recorded. Brief physics-substep violations or simulator warnings may not be visible in these saved frames.':'';
  $('replay-note').textContent=`Complete actual training episode · ${clip.duration_s.toFixed(2)} s · captured ${new Date(clip.recorded_at*1000).toLocaleTimeString()}. Playback uses saved poses; no simulation or policy inference runs on this computer.${guardNote}`;
  $('play').disabled=false;$('scrub').disabled=false;$('scrub').max=clip.duration_s;$('play').textContent='Ⅱ';showEpisodeFrame();
 }catch{
  if(request===replayRequest){$('episode-badge').textContent='Episode loading delayed';$('replay-title').textContent='Saved episode is temporarily unavailable; retrying';}
 }finally{if(loadingReplay?.request===request)loadingReplay=null;}
}
function showEpisodeFrame(){
 if(!episode||!renderer)return;
 const displayTime=Math.min(playTime,episode.duration_s);
 let lo=0,hi=episode.times.length-1;
 while(lo<hi){const mid=Math.ceil((lo+hi)/2);if(episode.times[mid]<=displayTime)lo=mid;else hi=mid-1;}
 if(lo!==lastFrameIndex){renderer.setSnapshot({poses:episode.frames[lo]});dirty=2;lastFrameIndex=lo;}$('scrub').value=displayTime;$('time').textContent=`${displayTime.toFixed(2)} / ${episode.duration_s.toFixed(2)}s`;
}
$('preview-state').onchange=e=>{replayMode=true;$('view-mode').textContent='View reset';$('view-mode').setAttribute('aria-pressed','false');if(!e.target.value){selectTask(selectedTask);return;}scope='stage';selected=stageRows().findIndex(row=>row.stage===e.target.value);setPreview(e.target.value);render();};
$('play').onclick=()=>{playing=!playing;$('play').textContent=playing?'Ⅱ':'▶';};
$('scrub').oninput=e=>{playTime=Number(e.target.value);showEpisodeFrame();};
$('view-mode').onclick=()=>{replayMode=!replayMode;$('view-mode').textContent=replayMode?'View reset':'View episode';$('view-mode').setAttribute('aria-pressed',String(!replayMode));if(replayMode){episode=null;syncReplay();}else{playing=false;replayRequest++;showReset();}};
function tick(now){
 if(renderer&&!document.hidden&&sceneVisible){
  if(episode&&playing&&replayMode){playTime+=(now-(lastTick||now))/1000*Number($('speed').value);if(playTime>episode.duration_s+.65)playTime=0;showEpisodeFrame();}
  renderer.controls.update();
  if(dirty>0&&now-lastRender>33){renderer.renderer.render(renderer.scene,renderer.camera);lastRender=now;dirty--;}
 }
 lastTick=now;requestAnimationFrame(tick);
}
requestAnimationFrame(tick);
document.addEventListener('visibilitychange',()=>{lastTick=0;dirty=2;if(!document.hidden)syncReplay();});
$('preview-task').innerHTML=tasks.map((t,i)=>`<option value="${t}">${names[i]}</option>`).join('');$('preview-task').onchange=e=>selectTask(e.target.value);$('reset-camera').onclick=()=>renderer?.preset('perspective');
$('metric').onchange=e=>{metric=e.target.value;render();};$('smooth').onchange=render;$('all').onclick=()=>{selected=null;render();};
document.querySelectorAll('[data-scope]').forEach(b=>b.onclick=()=>{scope=b.dataset.scope;selected=null;render();});
new ResizeObserver(()=>chart()).observe($('chart'));
async function poll(){try{const response=await fetch('/api/live',{cache:'no-store',signal:AbortSignal.timeout(30000)});if(!response.ok)throw Error();const oldRun=live.run;live=await response.json();render();if(oldRun&&oldRun!==live.run){recordedScene=null;void setPreview(previewTask);}else void syncReplay();}catch{$('status').textContent='Dashboard connection interrupted';$('status-dot').className='dot stale';}setTimeout(poll,3000);}
async function init(){poll();try{renderer=new ThreeSceneRenderer($('scene'));renderer.renderer.setAnimationLoop(null);renderer.controls.addEventListener('change',()=>{dirty=2;});new ResizeObserver(()=>{dirty=2;}).observe($('scene'));new IntersectionObserver(entries=>{sceneVisible=entries[0].isIntersecting;dirty=2;if(sceneVisible)syncReplay();}).observe($('scene'));void setPreview(previewTask);}catch{$('scene-loading').textContent='3D preview unavailable';}}
init();


function showWorkspaceTab(){
 const states=location.hash.startsWith('#states');
 const cube=location.hash.startsWith('#training/cube');
 const bc=location.hash.startsWith('#training/bc');
 $('bc-panel').hidden=!bc;
 $('bc-project').setAttribute('aria-current',bc?'page':'false');
 if(bc&&!$('bc-frame').getAttribute('src'))$('bc-frame').src='/bc/';
 document.body.classList.toggle('cube-workspace',cube);
 $('training-projects').hidden=states;
 $('cube-panel').hidden=!cube;
 $('assembly-project').setAttribute('aria-current',!cube&&!bc?'page':'false');
 $('cube-project').setAttribute('aria-current',cube?'page':'false');
 if(cube&&!$('cube-frame').getAttribute('src'))$('cube-frame').src='/cube/';
 $('training-panel').hidden=states||cube||bc;$('states-panel').hidden=!states;
 $('training-nav').setAttribute('aria-current',states?'false':'page');$('states-nav').setAttribute('aria-current',states?'page':'false');
 document.title=states?'Assembly · States & variations':cube?'Cube handoff · Training':'Assembly · Training';
 if(states&&!$('states-frame').getAttribute('src'))$('states-frame').src='/states/';
 if(!states)requestAnimationFrame(()=>{chart();dirty=2;});
}
function renderWorkspace(){
 const bank=workspace.states||{},pause=workspace.training||{};
 const train=Number.isFinite(bank.accepted_train)?bank.accepted_train:null,evals=Number.isFinite(bank.accepted_eval)?bank.accepted_eval:null;
 const targetTrain=bank.target_train||16000,targetEval=bank.target_eval||3200,total=(train??0)+(evals??0),target=targetTrain+targetEval;
 $('bank-counter').textContent=train==null||evals==null?'Accepted states · awaiting sync':`${fmt(total)} / ${fmt(target)} accepted`;
 $('bank-progress').max=target;$('bank-progress').value=total;
 $('bank-detail').textContent=`${fmt(train)} TRAIN · ${fmt(evals)} EVAL · ${bank.bank_status==='complete'?'bank complete':'generation in progress'}`;
 $('bank-quality').textContent=bank.available===false?'Viewer is updating · accepted snapshots remain saved':bank.snapshot_label||'Native overlap and held-grip retention · movement allowed';
 $('training-pause').hidden=!['paused','restoring'].includes(pause.state);$('pause-title').textContent=`${pause.state==='restoring'?'Restoring training':'Training temporarily paused'}${pause.checkpoint!=null?' · checkpoint '+fmt(pause.checkpoint):''}`;
 $('pause-reason').textContent=pause.reason||'The checkpoint is saved while the expanded reset bank is prepared.';
 const frame=$('states-frame');
 if(workspaceRevision!==null&&workspaceRevision!==bank.revision&&frame.getAttribute('src')){
  let hash='';try{const params=new URLSearchParams(frame.contentWindow.location.hash.slice(1));params.delete('reset');params.delete('bank');hash=params.toString();}catch{}
  $('states-loading').hidden=false;frame.src='/states/'+(hash?'#'+hash:'');
 }
 workspaceRevision=bank.revision??workspaceRevision;
}
async function pollWorkspace(){
 try{const response=await fetch('/api/workspace',{cache:'no-store'});workspace=await response.json();renderWorkspace();render();}catch{$('bank-quality').textContent='Bank status temporarily unavailable';}
 setTimeout(pollWorkspace,5000);
}
$('states-frame').onload=()=>{$('states-loading').hidden=true;};
window.addEventListener('hashchange',showWorkspaceTab);
showWorkspaceTab();pollWorkspace();
