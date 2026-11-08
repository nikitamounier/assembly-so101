"""Reuse original self-contained zoomable viewer with fresh captures only."""
from pathlib import Path
import ast,json
H=Path(__file__).resolve().parent
old=ast.parse((H.parent/'native-flame-profile-20261107/build_report.py').read_text())
page=next(n.value.value for n in old.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='page' for t in n.targets))
style=page.split('<style>')[1].split('</style>')[0]
renderer=page[page.index('const select='):page.index('let r=DATA.result;')]
def root(n):return dict(name=n,value=0,children={})
def add(t,stack,w):
 if w<=0:return
 t['value']+=w
 for n in stack:t=t['children'].setdefault(n,root(n));t['value']+=w
 return t
def pack(t):return dict(name=t['name'],value=t['value'],children=[pack(x) for x in sorted(t['children'].values(),key=lambda x:-x['value'])])
views=[];rows=[]
pairs=json.loads((H/'pairs-instrumented.json').read_text());t=root('v13 native collision pairs — measured timers')
for r in pairs['rows']:
 for p in r['pairs']:
  n=' ↔ '.join(p['names']);s=p['stats'];add(t,[r['stage'],n,'Original SDF search'],s[1]);add(t,[r['stage'],n,'Surface supplement'],s[2]);rows.append(dict(stage=r['stage'],pair=n,bodies=' ↔ '.join(p['bodies']),sdf=s[1],supplement=s[2],calls=s[0],empty=s[6],budget=f'{s[7]:.0f}×{s[8]:.0f}'))
views.append(dict(title='v13 native worker → collision pairs (timer-based)',note='Fresh two-world/two-native-thread replay. Widths sum pair timers across threads and include end-step refresh; four separate saved stages. This is a measured timer hierarchy, NOT sampled C/C++ call stacks. Loose stage has zero SDF pair calls. Do not add this view to PPO or phase-clock times.',tree=pack(t)))
for v in ['v13','v12','v10']:
 d=json.loads((H/v/'learner.speedscope.json').read_text());p=d['profiles'][0];t=root(v+' learner / local workers — sampled Python wall stacks')
 for ids,w in zip(p['samples'],p['weights']):add(t,[d['shared']['frames'][i]['name'] for i in ids],w)
 views.append(dict(title=v+' PPO/learner (sampled Python)',note=p['name']+'. Actual32-step PPO; native worker calls are opaque within rollout. Local workers run in this learner process, so no separate remote worker stack is invented. Includes collection, optimizer and checkpoint; excludes initialization. GIL scheduling bias possible.',tree=pack(t)))
for v in ['v13','v10']:
 d=json.loads((H/(v+'-timed')/'panel.json').read_text());t=root(v+' native engine phases — timer hierarchy')
 for r in d['rows']:
  if r['warmup']:continue
  q=r['spans']['native_timers'];case=r['stage'];step=q['mjTIMER_STEP']['seconds'];used=0
  for label,key in [('Constraint solver','mjTIMER_CONSTRAINT'),('Velocity','mjTIMER_VELOCITY'),('Actuation','mjTIMER_ACTUATION'),('Integration advance','mjTIMER_ADVANCE')]:
   w=q[key]['seconds'];add(t,[case,label],w);used+=w
  collision=q['mjTIMER_POS_COLLISION']['seconds'];used+=collision
  broad=q['mjTIMER_COL_BROAD']['seconds'];narrow=q['mjTIMER_COL_NARROW']['seconds'];add(t,[case,'Collision','Broad phase'],broad);add(t,[case,'Collision','Narrow phase'],narrow);add(t,[case,'Collision','Other collision timer work'],max(0,collision-broad-narrow))
  assert used<=step+1e-5,(case,used,step)
  add(t,[case,'Other step: kinematics, inertia, constraint construction/projection, callbacks'],max(0,step-used))
 views.append(dict(title=v+' native worker → engine phases (timer-based)',note='Actual nested MuJoCo engine timers from a separate clock-enabled replay pass; warmup omitted. Selected nonoverlapping categories plus residual avoid double counting. Durations sum both native worker threads; phase widths are NOT PPO wall proportions. No C-stack unwinding was performed.',tree=pack(t)))
for v in ['v13','v12','v10']:
 p=json.loads((H/v/'ppo-result.json').read_text());t=root(v+' actual PPO spans');sp={s['name']:s['seconds'] for s in p['spans']};steps=sum(s['seconds'] for s in p['steps']);add(t,['Collect','BatchedAssemblyEnv.step (physics, refresh, reward/guards)'],steps);add(t,['Collect','Other collect: policy inference, episode reset, observation/bookkeeping'],max(0,sp['collect']-steps));add(t,['Optimizer update'],sp['update']);add(t,['Save checkpoint'],sp['save']);add(t,['Other training loop'],max(0,p['elapsed_s']-sum(sp.values())))
 views.append(dict(title=v+' actual PPO spans (wall timers)',note='Exact wrapper wall spans from one isolated32-step PPO update. These top-level spans cover the same run as the sampled Python view. Do not sum them with other views.',tree=pack(t)))
rows.sort(key=lambda x:-(x['sdf']+x['supplement']))
payload=json.dumps(dict(views=views,rows=rows)).replace('</','<\\/')
html='''<!doctype html><html><head><meta charset="utf-8"><title>Fresh hinge-limit flame profile</title><style>'''+style+'''</style></head><body><h1>Fresh hinge-limit flame profile</h1><p class="sub">November7,2026 · v10 baseline / v12 optimized collisions / v13 measured hinge limits ·0.5ms ×80substeps ·2logical CPU threads on1physical core. Isolated runs; live training untouched. Click any frame to zoom; reset to return.</p><div class="cards"><div class="card"><strong>205.0 →129.4s</strong><span>v10→mean v13 full32-step PPO</span></div><div class="card"><strong>97–98%</strong><span>v13 heavy-stage native time in collision</span></div><div class="card"><strong>No proven PPO gain over v12</strong><span>v12 mean128.0s; v13 mean129.4s; two runs each</span></div></div><h2>Flame graph</h2><select id="view"></select><button id="reset">Reset zoom</button><input id="search" placeholder="Search function or geometry"><div id="note"></div><div id="detail">Hover for exact time; click to zoom.</div><div id="graph"></div><h2>Fresh v13 collision pair attribution</h2><p class="sub">Native timer hierarchy from a separate private instrumented replay; geometry names are exact. Pair times sum threads and include end-step refresh. Instrumentation parity PASS:40 output arrays exact. Budget40×40 means invoked initial-point/iteration budget.</p><div class="scroll"><table id="pairs"></table></div><p><a href="README.md">Report and limitations</a> · <a href="v13/learner.speedscope.json">Raw v13 sampled stacks</a> · <a href="pairs-instrumented.json">Raw native pair timers</a></p><script>const DATA='''+payload+';\n'+renderer+'''
const cols=[['stage','Stage'],['pair','Geometry pair'],['sdf','SDF seconds'],['supplement','Supplement seconds'],['calls','Calls'],['empty','Empty-result calls'],['budget','Budget']];document.querySelector('#pairs').innerHTML='<thead><tr>'+cols.map(c=>`<th>${c[1]}</th>`).join('')+'</tr></thead><tbody>'+DATA.rows.map(r=>'<tr title="'+esc(r.bodies)+'">'+cols.map(([k])=>`<td>${typeof r[k]==='number'?(k==='sdf'||k==='supplement'?r[k].toFixed(5):r[k].toLocaleString()):esc(r[k])}</td>`).join('')+'</tr>').join('')+'</tbody>';choose();</script></body></html>'''
(H/'flamegraph.html').write_text(html)
print('Saved9zoomable views',len(html),'bytes')
