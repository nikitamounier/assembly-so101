import bisect,csv,json,subprocess,sys
from pathlib import Path
import numpy as np
OUT=Path(sys.argv[1]);stock=json.loads((OUT/'stock.json').read_text());diag=json.loads((OUT/'diagnostic.json').read_text());traced=json.loads((OUT/'traced.json').read_text());timed=json.loads((OUT/'timed.json').read_text());staged=json.loads((OUT/'stages.json').read_text())
checks={}
reference=np.load(OUT/'stock-states.npz')
for mode in ('diagnostic','traced','timed','stages'):
 actual=np.load(OUT/(mode+'-states.npz'));metadata=json.loads((OUT/(mode+'.json')).read_text())
 checks[mode]=dict(states_bitwise=bool(np.array_equal(reference['states'],actual['states'])),contact_counts_bitwise=bool(np.array_equal(reference['contacts'],actual['contacts'])),contact_hash_match=metadata['contact_sha256']==stock['contact_sha256'],max_state_delta=float(np.max(abs(reference['states']-actual['states']))))
assert all(v['states_bitwise'] and v['contact_counts_bitwise'] and v['contact_hash_match'] for v in checks.values()),checks
symbols=[]
for line in (OUT/'symbols.txt').read_text().splitlines():
 parts=line.split(maxsplit=2)
 if len(parts)==3:
  try:symbols.append((int(parts[0],16),parts[2]))
  except ValueError:pass
symbols.sort();addresses=[x[0] for x in symbols]
def symbol(address):
 if address<512*512:
  a,b=divmod(address-1,512);return f"pair: {traced['geom_names'][a] or a} ↔ {traced['geom_names'][b] or b}"
 offset=address-traced['base'];i=bisect.bisect_right(addresses,offset)-1
 return symbols[i][1] if i>=0 else hex(offset)
nodes={int(row['node']):dict(parent=int(row['parent']),name=symbol(int(row['address'],16)),ns=int(row['nanoseconds']),calls=int(row['calls'])) for row in csv.DictReader((OUT/'nodes.csv').open())}
for i,n in nodes.items():n['exclusive_ns']=n['ns']-sum(c['ns'] for c in nodes.values() if c['parent']==i)
assert all(n['exclusive_ns']>=0 for n in nodes.values())
def path(i):return ([] if nodes[i]['parent']==0 else path(nodes[i]['parent']))+[nodes[i]['name']]
roots={i:n for i,n in nodes.items() if n['parent']==0}
pairs=[]
for row in csv.DictReader((OUT/'pairs.csv').open()):
 r={k:int(v) for k,v in row.items()};r['name']=symbol(1+r['a']*512+r['b']);pairs.append(r)
pairs.sort(key=lambda r:-r['nanoseconds'])
def write_profile(name,stacks,title):
 frames=[];lookup={};samples=[];weights=[]
 for names,weight in stacks:
  if weight<=0:continue
  sample=[]
  for label in names:
   if label not in lookup:lookup[label]=len(frames);frames.append(dict(name=label))
   sample.append(lookup[label])
  samples.append(sample);weights.append(weight)
 data={'$schema':'https://www.speedscope.app/file-format-schema.json','shared':dict(frames=frames),'profiles':[dict(type='sampled',name=title,unit='seconds',startValue=0,endValue=sum(weights),samples=samples,weights=weights)]}
 (OUT/(name+'.speedscope.json')).write_text(json.dumps(data))
write_profile('native-detail',[(path(i),n['exclusive_ns']/1e9) for i,n in nodes.items()],'Native narrow phase — synchronous function tracing of selected pair calls')
# Coarse stage widths use a separate replay with the original MuJoCo/plugin
# binaries and three timestamps per adapter call. Only deeper proportions are
# estimated from the denser compiler trace.
stacks=[]
stages=[]
for row in csv.DictReader((OUT/'stages.csv').open()):
 r={k:int(v) for k,v in row.items()};r['name']=symbol(1+r['a']*512+r['b']);stages.append(r)
 for label,key in [('search budget selection / adapter setup','selector_ns'),('native collider','collider_ns'),('custom contact supplement','supplement_ns')]:
  candidates=[]
  for i,n in nodes.items():
   p=path(i)
   if p[0]!=r['name']:continue
   if (key=='collider_ns' and n['name'].startswith('mjc_') and nodes.get(n['parent'],{}).get('name')=='contact_search') or (key=='supplement_ns' and n['name']=='supplement_motor_base'):candidates.append(i)
  target=['whole narrow phase',r['name'],label]
  if not candidates:
   stacks.append((target+(['stock native collider (opaque)'] if key=='collider_ns' else []),r[key]/1e9));continue
  denom=sum(nodes[i]['ns'] for i in candidates)
  for root in candidates:
   rp=path(root)
   for i,n in nodes.items():
    p=path(i)
    if p[:len(rp)]==rp:stacks.append((target+p[len(rp)-1:],n['exclusive_ns']*r[key]/denom/1e9))
callback_s=sum(sum(r[k] for k in ('selector_ns','collider_ns','supplement_ns')) for r in stages)/1e9
residual=staged['narrow_thread_s']-callback_s
assert residual>=0,(residual,callback_s,staged['narrow_thread_s'])
stacks.append((['whole narrow phase','MuJoCo pair filtering / other stock colliders / contact insertion and timer residual (unexpanded)'],residual))
write_profile('whole-narrowphase',stacks,'Whole native narrow phase — measured coarse stages; estimated deeper proportions from traced calls')
functions={}
for i,n in nodes.items():
 r=functions.setdefault(n['name'],dict(name=n['name'],calls=0,inclusive_s=0,exclusive_s=0))
 r['calls']+=n['calls'];r['inclusive_s']+=n['ns']/1e9;r['exclusive_s']+=n['exclusive_ns']/1e9
stock_pairs=[]
for row in csv.DictReader((OUT/'timed-pairs.csv').open()):
 r={k:int(v) for k,v in row.items()};r['name']=symbol(1+r['a']*512+r['b']);stock_pairs.append(r)
events=[int(v) for v in (OUT/'surface-events.txt').read_text().splitlines()]
summary=dict(parity=checks,stock_wall_s=stock['wall_s'],diagnostic_wall_s=diag['wall_s'],traced_wall_s=traced['wall_s'],stock_timed_wall_s=timed['wall_s'],stage_wall_s=staged['wall_s'],stage_narrow_s=staged['narrow_thread_s'],stock_narrow_s=stock['narrow_thread_s'],traced_narrow_s=traced['narrow_thread_s'],callbacks_s=callback_s,outer_residual_s=residual,pairs=pairs,stock_pairs=stock_pairs,stages=stages,surface_events=dict(zip(['samples_tested','aabb_rejected','sdf_distance_evaluations','sdf_gradient_evaluations','bin_updates'],events)),functions=sorted(functions.values(),key=lambda r:-r['inclusive_s']),sampled_pair_calls=sum(r['selected'] for r in pairs),pair_calls=sum(r['calls'] for r in pairs),selected_interval=8,trace_nodes=nodes)
(OUT/'summary.json').write_text(json.dumps(summary,indent=2))
print(json.dumps({k:v for k,v in summary.items() if k not in ('trace_nodes','functions')},indent=2))
