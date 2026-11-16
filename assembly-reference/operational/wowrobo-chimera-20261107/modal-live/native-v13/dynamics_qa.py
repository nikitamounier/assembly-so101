"""Paired native timestep qualification, with unchanged real guards."""
from pathlib import Path
import json,importlib.util,time,hashlib,sys
import mujoco,numpy as np,torch
from assembly_policy.wowrobo import env as native
from assembly_policy.wowrobo.runtime import load_runtime
ROOT=Path(__file__).resolve().parent.parent;REL=ROOT/'native-v13';C=ROOT/'robotics-assembly-3.0'
variant=sys.argv[1];assert variant in ('baseline','candidate')
base=ROOT/('native-v10' if variant=='baseline' else 'native-v13');bank=base/('training-accepted-v10' if variant=='baseline' else 'training-accepted-v13');meta=json.loads((bank/'manifest.json').read_text());torch.set_num_threads(1)
m,profile,identity=load_runtime(meta['profile_path']);assert m.opt.timestep==.0005
def digest(a):
 a=np.ascontiguousarray(a);return dict(dtype=str(a.dtype),shape=list(a.shape),sha256=hashlib.sha256(a.tobytes()).hexdigest())
arrays={}
for name in dir(m):
 try:a=getattr(m,name)
 except (AttributeError,TypeError):continue
 if isinstance(a,np.ndarray):arrays[name]=digest(a)
options={}
for name in dir(m.opt):
 if name.startswith('_'):continue
 a=getattr(m.opt,name)
 if isinstance(a,np.ndarray):options[name]=a.tolist()
 elif isinstance(a,(int,float)):options[name]=a
 elif type(a).__module__=='mujoco._enums':options[name]=int(a)
samples=[];dynamics={'held':[],'near':[]}
for ti,task in enumerate(meta['task_names']):
 z=np.load(bank/(task+'_train.npz'));assert np.isfinite(z['states']).all() and z['states'].shape[1]==(162 if variant=='baseline' else 166)
 for stage in np.unique(z['stage']):
  indices=[i for i in np.flatnonzero(z['stage']==stage) if np.all(z['states'][i,159:162]==1)]
  if indices:samples.append((ti,str(stage),z['states'][indices[0]].copy()))
 for group,family in [('held',2),('near',4)]:
  indices=[i for i in np.flatnonzero(z['family']==family) if np.all(z['states'][i,159:162]==1)];assert indices,(task,group)
  i=indices[0];dynamics[group].append((ti,str(z['stage'][i]),z['states'][i].copy()))
def env(n):
 native.load_runtime=lambda _:(m,profile,identity)
 return native.BatchedAssemblyEnv(meta['task_names'],np.zeros(n,np.int64),native.EnvConfig(profile_path=meta['profile_path'],review_model=meta['review_model'],nthread=min(n,5)))
def grasp_pairs(d):
 result=set()
 for con in d.contact:
  bodies=tuple(sorted(m.body(int(m.geom_bodyid[int(g)])).name for g in con.geom))
  if con.dist<=.0001 and any('gripper' in x or 'moving_jaw' in x for x in bodies):result.add(bodies)
 return sorted(result)
initial=[];insertion=[];e=env(1)
try:
 for ti,stage,state in samples:
  e.set_states(np.array([0]),state[None],np.array([ti]));initial.append(dict(task=meta['task_names'][ti],stage=stage,obs={k:digest(v) for k,v in e.observe().items()},grasp=grasp_pairs(e.data[0]),reward=e.reward.snapshot(e.data[0],dict(task=meta['task_names'][ti]))))
 config=json.loads((C/'configs/wowrobo_mating_profile.json').read_text());steps=['cover','connector','pin','upper','final']
 for ti,stage,state in dynamics['near']:
  task=meta['task_names'][ti];address=int(m.jnt_qposadr[int(m.body_jntadr[m.body(task).id])]);pairs={frozenset((p['geom1'],p['geom2'])) for p in config['steps'][steps[ti]]}
  for offset in [np.zeros(3)]+[np.eye(3)[k]*sign*distance for distance in [.00025,.0005,.001,.003] for k in range(3) for sign in [-1,1]]:
   shifted=state.copy();shifted[1+address:1+address+3]+=offset;e.set_states(np.array([0]),shifted[None],np.array([ti]));depth=min([float(c.dist) for c in e.data[0].contact if frozenset(m.geom(int(g)).name for g in c.geom) in pairs]+[0]);insertion.append(dict(task=task,offset_m=offset.tolist(),depth_m=depth))
finally:e.close()
result=dict(variant=variant,identity=identity,arrays=arrays,options=options,initial=initial,insertion=insertion,dynamic={})
cache=REL/(variant+'-qualification.json')
if cache.exists():
 saved=json.loads(cache.read_text())
 if all(saved[k]==result[k] for k in ('identity','arrays','options','initial','insertion')):
  for group,row in saved.get('dynamic',{}).items():
   if all(v==999 for v in row['first_guard']+row['first_success']):result['dynamic'][group]=row

(REL/(variant+'-static.json')).write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(dict(event='static_complete',variant=variant,arrays=len(arrays),starts=len(initial),insertion_controls=len(insertion))),flush=True)
for group,states in dynamics.items():
 if group in result['dynamic']:
  print(json.dumps(dict(event='reused_complete_unterminated_group',variant=variant,group=group)),flush=True);continue
 e=env(len(states));S=np.array([s for _,_,s in states]);T=np.array([t for t,_,_ in states]);e.set_states(np.arange(len(states)),S,T);first_guard=np.full(len(states),999);first_success=np.full(len(states),999);maxdepth=np.zeros(len(states));poses=[];t=time.perf_counter()
 try:
  for k in range(12):
   action=np.zeros((len(states),12));action[:,[0,6]]=.02*np.sin(k*.25);reward,done,info=e.step(action);assert np.isfinite(reward).all() and not info['unstable'].any()
   for i in np.flatnonzero(info['safety']):first_guard[i]=min(first_guard[i],k)
   for i in np.flatnonzero(info['success']):first_success[i]=min(first_success[i],k)
   maxdepth=np.maximum(maxdepth,info['penetration']);poses.append(np.array([d.body(meta['task_names'][ti]).xpos.copy() for d,ti in zip(e.data,T)]))
   finished=np.flatnonzero(done)
   if len(finished):e.set_states(finished,S[finished],T[finished])
  row=dict(stages=[stage for _,stage,_ in states],first_guard=first_guard.tolist(),first_success=first_success.tolist(),maxdepth=maxdepth.tolist(),poses=np.array(poses).tolist(),final_grasp=[grasp_pairs(d) for d in e.data],seconds=time.perf_counter()-t);result['dynamic'][group]=row;print(json.dumps(dict(event='dynamic_group',variant=variant,group=group,seconds=row['seconds'],guard=row['first_guard'],success=row['first_success'])),flush=True)
 finally:e.close()
 (REL/(variant+'-qualification.json')).write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(dict(event='variant_complete',variant=variant,identity=identity)),flush=True)
