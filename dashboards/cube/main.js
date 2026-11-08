import './style.css';
import { ThreeSceneRenderer } from '../../web/src/viewer.js';
const $ = id => document.getElementById(id);
const families = ['loose','left_pickup','converging_transport','pre_handoff','partial_right_contact','light_right_grasp','dual_grasp','post_handoff','right_transport','near_target','placed_held'];
const names = ['Loose','Left pickup','Converging transport','Pre-handoff','Partial right contact','Light right grasp','Dual grasp','Post-handoff','Right transport','Near target','Placed, held'];
const colors = ['#608b72','#a4a956','#ce9b47','#c77754','#d29b6b','#ab8656','#b95a65','#9b6b9e','#777db6','#4f99a2','#367663'];
const fmt = (n, digits=0) => n == null ? '—' : Number(n).toLocaleString('en-US',{maximumFractionDigits:digits});
const signed = n => n == null ? '—' : (n > 0 ? '+' : '') + (Math.abs(n)>0&&Math.abs(n)<0.0001?Number(n).toExponential(2):Number(n).toFixed(4));
let live = {history:[]}, selected = null, metric='return', episode=null, renderer=null, frame=0, playing=false, lastTick=0, replayRequest=0, initialPoses=null;
const rewardEpoch = iteration => (live.run_context?.reward_changes||[]).filter(change=>iteration>=change.first_iteration).length;
const field = () => ({return:'fam_return',reward:'fam_reward',success:'fam_success'})[metric];
const values = family => {
  const rows = live.history || [], key=field(), smooth=$('smooth').checked;
  return rows.map((r,i)=>{
    const value=r[key]?.[family];
    if (value == null || !Number.isFinite(value)) return {x:r.it,y:null,epoch:rewardEpoch(r.it)};
    const ns = smooth ? rows.slice(Math.max(0,i-9),i+1).filter(p=>rewardEpoch(p.it)===rewardEpoch(r.it)).map(p=>p[key]?.[family]).filter(v=>v!=null&&Number.isFinite(v)) : [value];
    return {x:r.it,y:ns.reduce((a,b)=>a+b,0)/ns.length,epoch:rewardEpoch(r.it)};
  });
};
function path(points, x, y) { let gap=true, previousEpoch; return points.map(p=>{if(previousEpoch!==undefined&&p.epoch!==previousEpoch)gap=true;previousEpoch=p.epoch;if(p.y==null){gap=true;return '';} const command=gap?'M':'L'; gap=false;return `${command}${x(p.x).toFixed(2)},${y(p.y).toFixed(2)}`;}).join(' '); }
function drawChart(){
  const chart=$('chart'), width=Math.max(chart.clientWidth-28,300),height=235,left=51,right=18,top=20,bottom=32;
  const chosen=selected===null?families:[families[selected]], series=chosen.map(f=>({f,points:values(f)}));
  const all=series.flatMap(s=>s.points).filter(p=>p.y!=null), rows=(live.history||[]).filter(r=>r[field()] && Object.values(r[field()]).some(v=>v!=null));
  let xmin=rows[0]?.it||0,xmax=rows.at(-1)?.it||1,ymin=all.length?Math.min(...all.map(p=>p.y)):0,ymax=all.length?Math.max(...all.map(p=>p.y)):1;
  if(metric==='success'){ymin=0;ymax=1;} else {const span=Math.max(ymax-ymin,metric==='return'?.01:.0001);ymin-=span*.15;ymax+=span*.15;}
  const x=v=>left+(v-xmin)/(xmax-xmin||1)*(width-left-right), y=v=>top+(ymax-v)/(ymax-ymin||1)*(height-top-bottom);
  let svg=`<svg viewBox="0 0 ${width} ${height}" aria-hidden="true">`;
  for(let t=0;t<=4;t++){const v=ymin+(ymax-ymin)*t/4,py=y(v);svg+=`<line x1="${left}" y1="${py}" x2="${width-right}" y2="${py}" stroke="#e9ede2" stroke-dasharray="3 4"/><text x="${left-9}" y="${py+3}" text-anchor="end" fill="#9ba58f" font-size="9">${metric==='success'?Math.round(v*100)+'%':(metric==='reward'?v.toExponential(1):v.toFixed(4))}</text>`;}
  for(let t=0;t<=4;t++){const v=xmin+(xmax-xmin)*t/4;svg+=`<text x="${x(v)}" y="${height-10}" text-anchor="middle" fill="#9ba58f" font-size="9">${Math.round(v).toLocaleString()}</text>`;}
  for(const change of live.run_context?.reward_changes||[]){
    if(change.first_iteration<xmin||change.first_iteration>xmax)continue;
    const px=x(change.first_iteration),tx=Math.min(px+6,width-135);
    svg+=`<line x1="${px}" y1="${top}" x2="${px}" y2="${height-bottom}" stroke="#ac8753" stroke-width="1.3" stroke-dasharray="5 4"/><text x="${tx}" y="11" fill="#8b6b3d" font-size="9">${change.kind==='bank'?'New resets':'New reward'} · ${fmt(change.first_iteration)}</text>`;
  }
  for(const s of series){const color=colors[families.indexOf(s.f)];svg+=`<path d="${path(s.points,x,y)}" fill="none" stroke="${color}" stroke-width="2" stroke-linejoin="round"/>`;const p=s.points.filter(p=>p.y!=null).at(-1);if(p)svg+=`<circle cx="${x(p.x)}" cy="${y(p.y)}" r="3" fill="${color}"/>`;}
  chart.innerHTML=svg+'</svg>'+(!all.length?'<span class="chart-empty">'+(live.summary?'Waiting for complete episodes in this family':'Reward recording is starting…')+'</span>':'')+'<div id="tooltip" class="chart-tooltip" hidden></div>';
  chart.onpointermove=e=>{if(!rows.length)return;const fraction=Math.max(0,Math.min(1,(e.offsetX-left)/(width-left-right)));const target=xmin+fraction*(xmax-xmin);const row=rows.reduce((a,b)=>Math.abs(a.it-target)<Math.abs(b.it-target)?a:b);const tip=$('tooltip');tip.textContent=`Iteration ${fmt(row.it)} · `+chosen.map(f=>`${names[families.indexOf(f)]}: ${metric==='success'?(row[field()]?.[f]==null?'—':(100*row[field()][f]).toFixed(1)+'%'):signed(row[field()]?.[f])}`).join(' / ');tip.hidden=false;};
  chart.onpointerleave=()=>{$('tooltip').hidden=true;};
  $('history-note').textContent=rows.length?`Latest ${fmt(rows.length)} updates · iterations ${fmt(xmin)}–${fmt(xmax)} · reset/reward versions separated`:'Waiting for training data';
}
function render(){
  const last=live.history?.at(-1), stale=live.connection!=='live'||(live.training_age_s??0)>60;
  $('status-dot').className='dot '+(stale?'stale':'live');
  $('status').textContent=live.connection==='live'?(stale?'Training data delayed':'Live from Chimera'):live.connection==='unavailable'?'Connection interrupted':'Connecting to Chimera';
  $('updated').textContent=live.received_at?`Updated ${new Date(live.received_at*1000).toLocaleTimeString([], {hour:'2-digit',minute:'2-digit',second:'2-digit'})}`:'Awaiting data';
  $('iteration').textContent=fmt(last?.it);$('transitions').textContent=last?fmt(last.transitions):'—';$('fps').textContent=fmt(last?.fps);$('jobs').textContent=live.target_cpu_jobs?`${live.running_cpu_jobs} / ${live.target_cpu_jobs}`:'—';$('phase').textContent=live.phase?.replaceAll('_',' ')||'Awaiting learner';$('jobs-caption').textContent=live.target_envs?`${fmt(live.target_envs)} target environments`:'Allocated worker pool';
  const context=live.run_context;
  $('continuation').hidden=!context;
  if(context){
    $('continuation-title').textContent=`Training ${context.rolled_back?"restored":"continued"} from checkpoint ${fmt(context.resume_checkpoint)} — policy and optimizer retained`;
    $('continuation-note').textContent=context.note || (context.rolled_back?`Continuous gripper contact is no longer required. Outside-circle table contact after pickup still disqualifies; initial pickup dragging is allowed. Superseded continuation archived.`:`Stricter reward rules began at iteration ${fmt(context.resume_checkpoint+1)}. Family counters restarted their reporting window; training iterations and transitions did not reset.`);
  }
  const started=live.summary?.recording_started;
  $('family-window').textContent=started?`Family stats since recorder resumed at ${new Date(started*1000).toLocaleTimeString([], {hour:'2-digit',minute:'2-digit',second:'2-digit'})}.`:'Family statistics for the current reporting window.';
  const warn=live.connection==='unavailable'?'The cluster read is temporarily unavailable. Showing the last received data.':live.errors?.length?`Training needs attention: ${live.errors.join(', ')}`:null;
  $('warning').hidden=!warn;$('warning').textContent=warn||'';
  $('metric-note').textContent=({return:'Mean undiscounted return of complete episodes',reward:'Mean raw environment reward per valid step',success:'Successful episodes / finished episodes'})[metric];
  if(!$('legend').children.length)$('legend').innerHTML=families.map((f,i)=>`<button style="--color:${colors[i]}" class="${selected!==null&&selected!==i?'inactive':''}" aria-pressed="${selected===i}" data-family="${i}"><i></i>${names[i]}</button>`).join('');
  const familyMarkup=families.map((f,i)=>{
    const stats=live.summary?.families?.[f],n=stats?.episodes||0,mean=n?stats.return_sum/n:null,rate=n?stats.successes/n:null;
    const ps=values(f).filter(p=>p.y!=null).slice(-80);const lo=ps.length?Math.min(...ps.map(p=>p.y)):0,hi=ps.length?Math.max(...ps.map(p=>p.y)):1;
    const spark=ps.length?path(ps,v=>(v-ps[0].x)/(ps.at(-1).x-ps[0].x||1)*96+2,v=>30-(v-lo)/(hi-lo||1)*25):'';
    return `<button class="family ${selected===i?'selected':''}" aria-pressed="${selected===i}" data-family="${i}" style="--color:${colors[i]}"><div class="family-title"><span><i></i>${names[i]}</span><small>${String(i+1).padStart(2,'0')}</small></div><div class="family-values"><div><strong>${signed(mean)}</strong><small>MEAN EPISODE RETURN</small></div><svg viewBox="0 0 100 34" aria-hidden="true"><path d="${spark}" fill="none" stroke="${colors[i]}" stroke-width="1.8"/></svg></div><div class="family-foot"><span>${fmt(n)} complete episodes</span><span>${rate==null?'—':(rate*100).toFixed(1)+'%'} success</span></div></button>`;
  }).join('');
  if(!$('families').children.length)$('families').innerHTML=familyMarkup;
  else{
    const template=document.createElement('template');template.innerHTML=familyMarkup;
    [...template.content.children].forEach((fresh,i)=>{
      const current=$('families').children[i];
      current.querySelector('strong').textContent=fresh.querySelector('strong').textContent;
      current.querySelector('.family-foot').innerHTML=fresh.querySelector('.family-foot').innerHTML;
      current.querySelector('svg').innerHTML=fresh.querySelector('svg').innerHTML;
    });
  }
  document.querySelectorAll('[data-family]').forEach(b=>{
    const i=Number(b.dataset.family);b.setAttribute('aria-pressed',String(selected===i));
    if(b.classList.contains('family'))b.classList.toggle('selected',selected===i);
    else b.classList.toggle('inactive',selected!==null&&selected!==i);
    b.onclick=()=>selectFamily(i);
  });
  drawChart();
}
$('metric').onchange=e=>{metric=e.target.value;render();};$('smooth').onchange=render;$('all').onclick=()=>selectFamily(null);new ResizeObserver(drawChart).observe($('chart'));
function clearEpisode(message, loading=false){
  episode=null;playing=false;frame=0;$('play').disabled=true;$('scrub').disabled=true;$('scrub').value=0;
  $('play').textContent='▶';$('time').textContent='0.00 / 0.00s';$('empty-replay').hidden=false;
  $('episode-badge').textContent=loading?'Loading':'Waiting';
  $('episode-badge').classList.remove('failed');
  $('empty-replay').querySelector('h3').textContent=loading?'Loading replay…':'No recorded success yet';
  $('empty-replay').querySelector('p').textContent=message;
  $('episode-family').textContent=selected===null?'Waiting for the latest success':`${names[selected]} · no saved replay yet`;
  $('episode-return').textContent='—';
  if(renderer&&initialPoses)renderer.setSnapshot({poses:initialPoses});
}
function selectFamily(index){
  selected=index;render();
  document.querySelector('.replay-panel h2').textContent=index===null?'Latest successful episode':`Latest success · ${names[index]}`;
  syncEpisode(true);
}
async function syncEpisode(force=false){
  const family=selected===null?null:families[selected];
  const id=family?live.episode_ids?.[family]:live.episode_id;
  if(!force&&id&&episode?.id===id)return;
  if(!force&&!id){
    ++replayRequest;
    clearEpisode(family?`No successful ${names[selected].toLowerCase()} replay has been saved yet.`:'The latest complete successful episode will appear here automatically.');
    return;
  }
  const request=++replayRequest;
  clearEpisode(family?`Loading the latest successful ${names[selected].toLowerCase()} episode.`:'Loading the latest successful episode.',true);
  try{
    const response=await fetch('/api/episode'+(family?'?family='+encodeURIComponent(family):''),{cache:'no-store'});
    if(!response.ok)throw Error('Replay unavailable');
    const clip=await response.json();
    if(request!==replayRequest||family!==(selected===null?null:families[selected]))return;
    if(!clip){clearEpisode(family?`No successful ${names[selected].toLowerCase()} replay has been saved yet.`:'The latest complete successful episode will appear here automatically.');return;}
    if(family&&clip.family!==family)throw Error('Replay family mismatch');
    episode=clip;frame=0;playing=true;$('empty-replay').hidden=true;
    const failed=clip.success===false;
    $('episode-badge').textContent=failed?'Failed diagnostic':'Successful';
    $('episode-badge').classList.toggle('failed',failed);
    const familyName=names[families.indexOf(clip.family)]||clip.family;
    document.querySelector('.replay-panel h2').textContent=failed?`Example failure · ${familyName}`:(family?`Latest success · ${familyName}`:'Latest successful episode');
    $('episode-family').textContent=failed?`${familyName} · checkpoint ${fmt(clip.iteration)} · diagnostic rollout`:`${familyName} · iteration ${fmt(clip.iteration)}`;
    $('episode-return').textContent=`Return ${signed(clip.episode_return)}`;$('play').disabled=false;$('scrub').disabled=false;
    $('scrub').max=clip.frames.length-1;$('play').textContent='Ⅱ';showFrame();
  }catch(error){if(request===replayRequest){clearEpisode('The replay is temporarily unavailable. Retrying with the next update.');$('episode-badge').textContent='Unavailable';}}
}
function showFrame(){if(!episode||!renderer)return;renderer.setSnapshot({poses:episode.frames[Math.min(Math.floor(frame),episode.frames.length-1)]});$('scrub').value=Math.floor(frame);$('time').textContent=`${(frame*episode.dt).toFixed(2)} / ${episode.duration_s.toFixed(2)}s`;}
$('play').onclick=()=>{playing=!playing;$('play').textContent=playing?'Ⅱ':'▶';};$('scrub').oninput=e=>{frame=Number(e.target.value);showFrame();};
function tick(now){if(episode&&playing){frame+=(now-(lastTick||now))/1000*Number($('speed').value)/episode.dt;if(frame>=episode.frames.length-1)frame=0;showFrame();}lastTick=now;requestAnimationFrame(tick);}requestAnimationFrame(tick);
async function poll(){try{const r=await fetch('/api/live',{cache:'no-store'});if(!r.ok)throw Error('Dashboard unavailable');live=await r.json();render();await syncEpisode();}catch(e){$('status').textContent='Dashboard connection interrupted';$('status-dot').className='dot stale';}setTimeout(poll,3000);}
async function init(){try{renderer=new ThreeSceneRenderer($('scene'));const [scene,initial]=await Promise.all([fetch('/api/scene').then(r=>r.json()),fetch('/api/initial').then(r=>r.json())]);renderer.setScene(scene);initialPoses=initial;renderer.setSnapshot({poses:initial});}catch(error){$('empty-replay').querySelector('small').textContent='3D rendering unavailable in this browser';}poll();}init();
