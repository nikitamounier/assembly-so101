import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'outputs/ppo-profile-20261106'
A={k:json.loads((OUT/k/'analysis.json').read_text()) for k in ['so101','assembly-nist']}
R={k:json.loads((OUT/k/'result.json').read_text()) for k in A}
P=json.loads((OUT/'pair-probe/result.json').read_text())
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'axes.spines.top':False,'axes.spines.right':False,'figure.facecolor':'white','axes.titleweight':'bold'})
orange='#e87832';blue='#426caa';teal='#258c87';red='#b54e56';gray='#b5bfca'
fig,ax=plt.subplots(2,2,figsize=(14,9.3),layout='constrained')
fig.suptitle('Full PPO iteration: SO101 vs original AssembleNIST',fontsize=21,fontweight='bold',ha='left',x=.04)
for i,(case,label,color) in enumerate([('so101','SO101',orange),('assembly-nist','NIST',teal)]):
 a=A[case];ax[0,0].barh(1-i,a['wall_s'],color=color,height=.45);ax[0,0].text(a['wall_s']+4,1-i,f"{a['wall_s']:.2f} s",va='center',fontweight='bold')
ax[0,0].set(yticks=[1,0],yticklabels=['SO101','NIST'],xlim=(0,345),xlabel='Seconds per full 262,144-transition iteration',title='39.9× total wall-time difference')
ax[0,0].grid(axis='x',alpha=.15)
rows=[]
for case in A:
 a=A[case];span={s['name']:s['duration_s'] for s in a['spans']};remote=a['remote_step_wall_sum_s'];rows.append([100*a['critical_worker_compute_sum_s']/a['wall_s'],100*(remote-a['critical_worker_compute_sum_s'])/a['wall_s'],100*(span['collect']-remote)/a['wall_s'],100*span['update']/a['wall_s'],100*(a['wall_s']-span['collect']-span['update'])/a['wall_s']])
left=np.zeros(2)
for i,(name,c) in enumerate([('Slowest worker',orange),('Reply overhead',gray),('Policy + GAE',blue),('PPO update',teal),('Checkpoint + other','#c4b7d9')]):
 v=np.array(rows)[:,i];ax[0,1].barh([1,0],v,left=left,color=c,label=name,height=.48);left+=v
ax[0,1].set(yticks=[1,0],yticklabels=['SO101','NIST'],xlim=(0,100),xlabel='% of measured iteration wall time',title='Rollout physics dominates the critical path')
ax[0,1].legend(fontsize=8,loc='lower center',bbox_to_anchor=(.5,-.35),ncol=3,frameon=False)
tot={}
for w in A['so101']['sampled_workers']:
 for name,v in w['mujoco_timers']['rollout_last_trajectory'].items():tot[name]=tot.get(name,0)+v['seconds']
native=tot['mjTIMER_STEP'];narrow=tot['mjTIMER_COL_NARROW']/native*100;solve=tot['mjTIMER_CONSTRAINT']/native*100;project=tot['mjTIMER_POS_PROJECT']/native*100
labels=['Narrow-phase collision','Constraint solve','Inertia projection','Other physics'];values=[narrow,solve,project,100-narrow-solve-project]
ax[1,0].barh(np.arange(4)[::-1],values,color=[orange,blue,teal,gray],height=.58)
for y,v in zip(np.arange(4)[::-1],values):ax[1,0].text(v+1,y,f'{v:.1f}%',va='center',fontweight='bold')
ax[1,0].set(yticks=np.arange(4)[::-1],yticklabels=labels,xlim=(0,105),xlabel='% of native stepping time in two sampled workers',title='SO101: 93% is narrow-phase collision')
top=P['pairs'][:2];pv=[r['seconds']/P['pair_sum_thread_seconds']*100 for r in top];pv.append(100-sum(pv))
ax[1,1].barh([2,1,0],pv,color=[orange,blue,gray],height=.55)
for y,v in zip([2,1,0],pv):ax[1,1].text(v+1,y,f'{v:.1f}%',va='center',fontweight='bold')
ax[1,1].set(yticks=[2,1,0],yticklabels=['Base ↔ motor\n0 contacts returned','Screw ↔ driver case','Other pairs'],xlim=(0,102),xlabel='% of collision-pair time (separate 44-env / 8-step probe)',title='Auxiliary probe: expensive empty-contact searches')
fig.savefig(OUT/'overview.png',dpi=160);plt.close(fig)
fig,axes=plt.subplots(2,1,figsize=(13,7),layout='constrained',sharex=True)
for axis,(case,label) in zip(axes,[('so101','SO101'),('assembly-nist','AssembleNIST')]):
 steps=R[case]['steps'];ct=np.array([s['worker_ct_s'] for s in steps]);w=np.array([s['wall_s'] for s in steps]);x=np.arange(1,33)
 axis.plot(x,w,label='Learner remote-step wall time',color=orange,lw=3);axis.plot(x,ct.max(1),label='Slowest worker compute phase',color=red,lw=1.5,ls='--');axis.plot(x,ct.mean(1),label='Mean worker compute phase',color=blue,lw=2)
 axis.fill_between(x,ct.mean(1),ct.max(1),color=orange,alpha=.12)
 axis.set(ylabel='Seconds / control step',title=f'{label}: one worker on GPU71FM was last on all 32 steps');axis.grid(alpha=.15)
axes[0].legend(loc='lower right',frameon=False,fontsize=9);axes[1].set_xlabel('Control step within captured iteration')
fig.savefig(OUT/'worker-barriers.png',dpi=160);plt.close(fig)
# Machine-readable overview, excluding raw tracing-hook diagnostics.
summary={case:{'iteration':a['iteration'],'total_s':a['wall_s'],'remote_step_s':a['remote_step_wall_sum_s'],'ppo_update_s':next(s['duration_s'] for s in a['spans'] if s['name']=='update'),'checkpoint_s':next(s['duration_s'] for s in a['spans'] if s['name']=='save'),'kernel_active_fraction':a['gpu']['kernel_busy_fraction_of_iteration'],'all_workers':187,'valid_transitions':262144,'qualified':a['qualified']} for case,a in A.items()}
(OUT/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
for case in A:
    for w in A[case]['sampled_workers']:
        meta=next(r for r in R[case]['worker_results'] if r['token']==w['token'])
        w['compute_views']=[p.name.replace('.json','.html') for p in sorted((OUT/case).glob(f"worker-{w['token']}-thread-*.speedscope.json")) if int(p.name.split('-thread-')[1].split('.')[0])!=meta['main_thread']]
DATA=json.dumps({k:{'summary':summary[k],'workers':A[k]['sampled_workers']} for k in A})
html='''<!doctype html><html><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>PPO profiles: SO101 vs AssembleNIST</title><style>
+*{box-sizing:border-box}body{margin:0;background:#f3f5f8;color:#1c2c40;font:15px system-ui}main{max-width:1600px;margin:auto;padding:28px}h1{font-size:30px;margin:0 0 12px}h2{font-size:20px}p{line-height:1.6;max-width:1150px}.cards{display:flex;gap:18px;flex-wrap:wrap}.card{background:white;border:1px solid #dce2e9;border-radius:12px;padding:20px;min-width:210px}.big{font-size:34px;font-weight:700}.sub{color:#596a7c;font-size:13px}button,select{padding:10px 14px;border:1px solid #bcc9d6;border-radius:7px;background:white;color:#1c2c40;font:inherit;cursor:pointer}button.active{background:#244869;color:white}nav{display:flex;gap:8px;flex-wrap:wrap;margin:14px 0}iframe{width:100%;height:680px;border:1px solid #d6dfe7;background:white;border-radius:10px}table{border-collapse:collapse;background:white;width:100%;max-width:1000px}td,th{padding:11px;border-bottom:1px solid #e0e5ea;text-align:left}img{width:100%;max-width:1400px;display:block;margin:20px 0}a{color:#216e9a}.note{background:#fff6df;border:1px solid #f0d492;padding:14px;border-radius:8px;line-height:1.6}.hidden{display:none}.small{font-size:13px}code{font:13px ui-monospace,monospace}</style><main>
+<h1>PPO performance: SO101 vs original AssembleNIST</h1><p>Both captures contain a complete <b>8,192 × 32 = 262,144-transition</b> iteration. Same H100, same 37 CPU leases plus spare GPU-host CPUs, 187 two-thread workers, unchanged allocation lifetimes. No worker loss or joins during either captured iteration.</p>
+<div class="cards"><div class="card"><div class="sub">SO101 · iteration 30</div><div class="big">301.53 s</div></div><div class="card"><div class="sub">AssembleNIST · iteration 2</div><div class="big">7.56 s</div></div><div class="card"><div class="sub">Native SO101 stepping</div><div class="big">93% collision</div></div><div class="card"><div class="sub">Same full-batch shape</div><div class="big">39.9×</div></div></div>
+<p><a href="README.md">Detailed findings, method and limitations</a> · <a href="summary.json">Timing summary</a> · <a href="hardware-match.json">Same-hardware proof</a></p>
+<img src="overview.png" alt="Measured iteration and native phase breakdown"><h2>Interactive flame graphs</h2><div class="note">Stack views sample <b>wall time</b>, including waits. A wide learner <code>select</code> frame means it is waiting for simulation replies. Worker native C++ calls appear at their Python call sites. The separate MuJoCo phase view is derived from nested native timers, <b>not native C stack sampling</b>. Concurrent worker and native-thread times must not be added to iteration wall time.</div>
+<nav id="cases"><button data-case="so101" class="active">SO101</button><button data-case="assembly-nist">AssembleNIST</button></nav><nav id="views"><button data-view="learner" class="active">Learner stacks</button><button data-view="worker">CPU worker stacks</button><button data-view="native">Native phase timers</button></nav><select id="worker" class="hidden"></select><p id="caption" class="small"></p><iframe id="graph" title="Interactive flame graph"></iframe><div id="missing" class="note hidden">The original NIST backend keeps active MuJoCo data in thread-local rollout buffers. This capture did not collect their internal phase counters; zero counters in the unused model buffers do not imply zero collision cost. Its worker stack view and outer native-rollout timings are valid.</div>
+<h2>Where synchronization costs time</h2><img src="worker-barriers.png" alt="Mean, slowest worker and round-trip timing over all 32 steps"><p>The shaded gap is the difference between mean worker compute phase and the slowest worker. It is an idealized balancing opportunity, not pure network overhead. The actual reply/dispatch residual is approximately 0.30 s for SO101 across the entire iteration.</p>
+<h2>Raw data</h2><ul><li><a href="so101/torch-trace.json.gz">SO101 Torch CPU/CUDA trace (gzip)</a> · <a href="so101/result.json">SO101 qualification and per-step data</a></li><li><a href="assembly-nist/torch-trace.json.gz">NIST Torch CPU/CUDA trace (gzip)</a> · <a href="assembly-nist/result.json">NIST qualification and per-step data</a></li><li><a href="pair-probe/result.json">Separate collision-pair probe, including bitwise parity check</a></li></ul>
+<p class="small">Source: SO101 frozen v13 / 85c006a; original AssembleNIST eb5d15e. Different tasks, start distributions and policy ages; this is a measured workload comparison, not a controlled single-variable physics ablation. The rejected asynchronous-native-sampler attempt is excluded. Training was restored after the comparison; no physics or PPO optimization was applied.</p></main><script>
+const data=DATA;let dataset='so101',view='learner';const selector=document.querySelector('#worker');function updateWorkers(){selector.replaceChildren();let o=new Option('Aggregate: two sampled main threads','workers-cpu.speedscope.html');selector.add(o);for(const w of data[dataset].workers){selector.add(new Option(`Worker ${w.worker_index} · mean ${w.mean_step_s.toFixed(3)} s/step · percentile ${w.pool_speed_percentile.toFixed(1)}`,w.speedscope.replace('.json','.html')));let i=0;for(const file of w.compute_views){selector.add(new Option(`Worker ${w.worker_index} · compute thread ${++i} (Python/native call sites)`,file))}}}function draw(){document.querySelectorAll('[data-case]').forEach(b=>b.classList.toggle('active',b.dataset.case===dataset));document.querySelectorAll('[data-view]').forEach(b=>b.classList.toggle('active',b.dataset.view===view));selector.classList.toggle('hidden',view!=='worker');const absent=view==='native'&&dataset!=='so101';document.querySelector('#missing').classList.toggle('hidden',!absent);document.querySelector('#graph').classList.toggle('hidden',absent);let file=view==='learner'?'wall.speedscope.html':view==='native'?'native-phases.speedscope.html':selector.value;document.querySelector('#graph').src=dataset+'/'+file;document.querySelector('#caption').textContent=view==='native'?'Nested native timer attribution across both sampled workers and all of their physics substeps; thread-summed durations.':view==='worker'?'Worker indices refer to connection slices in this captured iteration. Percentiles use all 187 workers; higher means slower.':'Main-thread learner wall stacks: click to zoom, hover for details, search by function name.'}document.querySelectorAll('[data-case]').forEach(b=>b.onclick=()=>{dataset=b.dataset.case;updateWorkers();draw()});document.querySelectorAll('[data-view]').forEach(b=>b.onclick=()=>{view=b.dataset.view;draw()});selector.onchange=draw;updateWorkers();draw();</script></html>'''.replace('\n+','\n').replace('DATA',DATA)
(OUT/'comparison.html').write_text(html)
from make_viewer_self_contained import make_self_contained
make_self_contained(OUT)
print('wrote overview.png, worker-barriers.png, summary.json, comparison.html')
