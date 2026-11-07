"""Decode raw native profiles and calculate timing summaries in an allocation."""
import collections
import json
from pathlib import Path
import pstats
import sys
import numpy as np
import vmprof

path=Path(sys.argv[1])
r=json.loads((path/'result.json').read_text())


def export(stacks,target,title):
    frames=[];indices={};samples=[];weights=[]
    for stack,weight in stacks:
        ids=[]
        for name in stack:
            if name not in indices:
                indices[name]=len(frames);frames.append(dict(name=name))
            ids.append(indices[name])
        samples.append(ids);weights.append(weight)
    value={'$schema':'https://www.speedscope.app/file-format-schema.json','shared':dict(frames=frames),'profiles':[dict(type='sampled',name=title,unit='seconds',startValue=0,endValue=sum(weights),samples=samples,weights=weights)],'activeProfileIndex':0}
    target.write_text(json.dumps(value)+'\n')


aggregate=[];workers=[]
for row in r['worker_results']:
    if not row.get('native_profile'):
        raw=path/f"worker-{row['token']}.wall.json"
        events=json.loads(raw.read_text())
        end=min(row['end_utc'],r['start_utc']+r['wall_s'])-row['start_utc']
        by_thread=collections.defaultdict(list)
        for at,tid,stack in events:by_thread[tid].append((at,stack))
        stacks=[]
        for tid,records in by_thread.items():
            prev=0.0;these=[]
            for at,stack in records:
                stop=min(at,end)
                if stop>prev:these.append((stack,stop-prev))
                prev=stop
                if at>=end:break
            if tid==row['main_thread']:stacks=these
            export(these,path/f"worker-{row['token']}-thread-{tid}.speedscope.json",f"Worker {row['token']} Python thread {tid}: wall time, including native call-site waits")
        export(stacks,path/f"worker-{row['token']}.speedscope.json",f"Worker {row['token']}: MAIN-thread wall stacks, native call sites")
        aggregate+=stacks
        phases=collections.defaultdict(lambda:dict(wall_s=0.0,cpu_s=0.0,calls=0))
        for event in row['phases']:
            phase=phases[event['name']];phase['wall_s']+=event['wall_s'];phase['cpu_s']+=event['cpu_s'];phase['calls']+=1
        totals=collections.defaultdict(lambda:np.zeros(len(row['timer_names'])))
        counts=collections.defaultdict(lambda:np.zeros(len(row['timer_names']),dtype=np.int64))
        previous={(d['kind'],str(d['index'])):d for d in row['initial_timers']}
        stepid=row['timer_names'].index('mjTIMER_STEP')
        for snapshots in row['timer_steps']:
            for d in snapshots:
                key=(d['kind'],str(d['index']));old=previous[key]
                reset=d['kind']=='rollout_last_trajectory' and d['number'][stepid]<=old['number'][stepid]
                durations=np.array(d['duration']) if reset else np.maximum(0,np.array(d['duration'])-old['duration'])
                numbers=np.array(d['number']) if reset else np.maximum(0,np.array(d['number'])-old['number'])
                totals[d['kind']]+=durations;counts[d['kind']]+=numbers;previous[key]=d
        native={kind:{name:dict(seconds=float(totals[kind][i]),calls=int(counts[kind][i])) for i,name in enumerate(row['timer_names'])} for kind in totals}
        reported=np.array([d['reported_ct_s'] for d in row['step_times']])
        all_ct=np.array([d['worker_ct_s'] for d in r['steps']])
        errors=np.max(np.abs(all_ct-reported[:,None]),axis=0);index=int(np.argmin(errors))
        assert errors[index]<1e-6,errors[index]
        mean=float(reported.mean());percentile=float((all_ct.mean(0)<=mean).mean()*100)
        workers.append({k:row[k] for k in ['token','job','host','rank','pid','steps','cpu_s']} | dict(main_wall_s=sum(w for _,w in stacks),compute_wall_s=float(reported.sum()),step_cpu_s=sum(d['cpu_s'] for d in row['step_times']),mean_step_s=mean,pool_speed_percentile=percentile,worker_index=index,phases=dict(phases),mujoco_timers=native,speedscope=f"worker-{row['token']}.speedscope.json",sampling=row['sampling']))
        continue
    raw=Path(row['native_profile'])
    s=vmprof.read_profile(str(raw))
    stacks=[];leaf=collections.Counter();inclusive=collections.Counter();unknown=0;count=0
    for trace,n,tid,memory in s.profiles:
        names=[s.adr_dict.get(a,f'<unresolved 0x{a:x}>') for a in trace]
        if any('py:sample:' in name and 'profile_worker.py' in name for name in names):continue
        weight=n*.0099
        if not names:continue
        stacks.append((names,weight));leaf[names[-1]]+=weight
        for name in set(names):inclusive[name]+=weight
        count+=n
        if '<native symbol ' in names[-1] or '<unresolved ' in names[-1]:unknown+=n
    dest=raw.with_suffix('.speedscope.json')
    export(stacks,dest,f"Worker CPU native/Python: {row['host']} / {row['token']}")
    aggregate+=stacks
    workers.append({**row,'samples':count,'sample_cpu_s':sum(w for _,w in stacks),'unresolved_leaf_fraction':unknown/max(1,count),'top_leaf':leaf.most_common(30),'top_inclusive':inclusive.most_common(40),'speedscope':dest.name,'vmprof_meta':s.meta})
export(aggregate,path/'workers-cpu.speedscope.json','Representative workers: summed MAIN-thread wall samples, including waits')
leaf=collections.Counter();inclusive=collections.Counter()
for stack,w in aggregate:
    leaf[stack[-1]]+=w
    for name in set(stack):inclusive[name]+=w
steps=r['steps'];ct=np.array([s['worker_ct_s'] for s in steps]);wall=np.array([s['wall_s'] for s in steps]);means=ct.mean(0)
tail=np.argmax(ct,axis=1);freq=collections.Counter(map(int,tail))
stats=pstats.Stats(str(path/'python.pstats'))
functions=[]
for (filename,line,name),(cc,nc,tt,ctotal,callers) in stats.stats.items():
    functions.append(dict(file=filename,line=line,name=name,calls=nc,self_s=tt,total_s=ctotal))
functions.sort(key=lambda x:x['self_s'],reverse=True)
ops=json.loads((path/'operators.json').read_text())
trace=json.loads((path/'torch-trace.json').read_text())
kernel_events=[e for e in trace['traceEvents'] if e.get('ph')=='X' and e.get('cat')=='kernel']
intervals=sorted((e['ts'],e['ts']+e['dur']) for e in kernel_events)
union=0;end=-float('inf')
for start,stop in intervals:
    union+=max(0,stop-max(start,end));end=max(end,stop)
gpu_summary=dict(kernel_count=len(kernel_events),kernel_sum_s=sum(e['dur'] for e in kernel_events)/1e6,kernel_union_s=union/1e6,kernel_busy_fraction_of_iteration=union/1e6/r['wall_s'])
summary=dict(qualified=r['qualified'],iteration=r['iteration'],wall_s=r['wall_s'],transitions=r['transitions'],spans=r['spans'],remote_profile=r['remote_profile'],worker_step_mean_s=float(ct.mean()),worker_step_max_mean_s=float(ct.max(1).mean()),remote_step_wall_sum_s=float(wall.sum()),straggler_wait_upper_bound_s=float((ct.max(1)-ct.mean(1)).sum()),critical_worker_compute_sum_s=float(ct.max(1).sum()),noncritical_worker_compute_sum_s=float(ct.sum()),step_wall_minus_slowest_worker_s=float((wall-ct.max(1)).sum()),worker_mean_step_percentiles_s={str(q):float(np.percentile(means,q)) for q in [0,50,90,99,100]},stragglers=[dict(worker=i,host=steps[0]['hosts'][i],steps_last=n,mean_step_s=float(means[i]),max_step_s=float(ct[:,i].max())) for i,n in freq.most_common()],slowest_workers=[dict(worker=int(i),host=steps[0]['hosts'][i],mean_step_s=float(means[i])) for i in np.argsort(means)[-15:][::-1]],sampled_workers=workers,native_leaf=leaf.most_common(40),native_inclusive=inclusive.most_common(60),python_self=functions[:50],python_cumulative=sorted(functions,key=lambda x:x['total_s'],reverse=True)[:50],cuda_self_total_s=sum(o['self_device_us'] for o in ops)/1e6,cuda_top=sorted(ops,key=lambda x:x['self_device_us'],reverse=True)[:30])
summary['gpu']=gpu_summary
(path/'analysis.json').write_text(json.dumps(summary,indent=2)+'\n')
print(json.dumps({k:summary[k] for k in ['qualified','iteration','wall_s','worker_step_mean_s','worker_step_max_mean_s','straggler_wait_upper_bound_s','step_wall_minus_slowest_worker_s','cuda_self_total_s']}))
