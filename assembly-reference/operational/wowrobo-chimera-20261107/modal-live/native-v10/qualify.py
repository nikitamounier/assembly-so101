"""Paired native timestep qualification, with unchanged real guards."""
from pathlib import Path
import json,importlib.util,time,hashlib
import mujoco,numpy as np,torch
from assembly_policy.wowrobo import env as native
from assembly_policy.wowrobo.runtime import load_runtime
ROOT=Path(__file__).resolve().parent.parent;REL=ROOT/'native-v10';C=ROOT/'robotics-assembly-3.0';bank=C/'assets_local/wowrobo/training-accepted-v6';meta=json.loads((bank/'manifest.json').read_text());torch.set_num_threads(1)
base,profile,oldid=load_runtime(meta['profile_path']);assert oldid=='8c4783bc443e8e9efad8f19890b6910a79299d434dbb3111af9f24733d850879'
spec=importlib.util.spec_from_file_location('_timestep_candidate',REL/'wowrobo_training_runtime.py');module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);candidate,_,newid=module.load_native(meta['profile_path']);models=[base,candidate];ids=[oldid,newid]
changed=[];checked=0
for name in dir(base):
 try:a,b=getattr(base,name),getattr(candidate,name)
 except (AttributeError,TypeError):continue
 if isinstance(a,np.ndarray):
  checked+=1
  if not np.array_equal(a,b,equal_nan=True):changed.append(name)
assert not changed,changed
for name in dir(base.opt):
 if name.startswith('_') or name=='timestep':continue
 a,b=getattr(base.opt,name),getattr(candidate.opt,name)
 if isinstance(a,(float,int,np.ndarray)) or type(a).__module__=='mujoco._enums':np.testing.assert_array_equal(a,b,err_msg=name)
assert base.opt.timestep==.00025 and candidate.opt.timestep==.0005
samples=[];dynamics={'held':[],'near':[]};data_by_task={}
for ti,task in enumerate(meta['task_names']):
 z=np.load(bank/(task+'_train.npz'));data_by_task[task]=z
 assert np.isfinite(z['states']).all() and z['states'].shape[1]==162
 for stage in np.unique(z['stage']):
  indices=[i for i in np.flatnonzero(z['stage']==stage) if np.all(z['states'][i,-3:]==1)]
  if indices:samples.append((ti,str(stage),z['states'][indices[0]].copy()))
 for group,family in [('held',2),('near',4)]:
  indices=[i for i in np.flatnonzero(z['family']==family) if np.all(z['states'][i,-3:]==1)]
  assert indices,(task,group)
  i=indices[0];dynamics[group].append((ti,str(z['stage'][i]),z['states'][i].copy()))

def envs(n):
 out=[]
 for m,sig in zip(models,ids):
  native.load_runtime=lambda _,m=m,sig=sig:(m,profile,sig)
  out.append(native.BatchedAssemblyEnv(meta['task_names'],np.zeros(n,np.int64),native.EnvConfig(profile_path=meta['profile_path'],review_model=meta['review_model'],nthread=min(n,2))))
 return out

def grasp_pairs(m,d):
 result=set()
 for con in d.contact:
  bodies=tuple(sorted(m.body(int(m.geom_bodyid[int(g)])).name for g in con.geom))
  if con.dist<=.0001 and any('gripper' in x or 'moving_jaw' in x for x in bodies):result.add(bodies)
 return result

initial=[];es=envs(1)
try:
 for ti,stage,state in samples:
  snapshots=[];pairs=[]
  for e in es:
   e.set_states(np.array([0]),state[None],np.array([ti]));snapshots.append(e.observe());pairs.append(grasp_pairs(e.m,e.data[0]))
  for k in snapshots[0]:np.testing.assert_array_equal(snapshots[0][k],snapshots[1][k],err_msg=stage+':'+k)
  assert pairs[0]==pairs[1],(stage,pairs)
  assert es[0].reward.snapshot(es[0].data[0],dict(task=meta['task_names'][ti]))==es[1].reward.snapshot(es[1].data[0],dict(task=meta['task_names'][ti]))
  initial.append(dict(task=meta['task_names'][ti],stage=stage,initial_obs_reward_and_grasp_contacts_exact=True))
 config=json.loads((C/'configs/wowrobo_mating_profile.json').read_text());steps=['cover','connector','pin','upper','final'];insert=[]
 for ti,stage,state in dynamics['near']:
  task=meta['task_names'][ti];m=base;address=int(m.jnt_qposadr[int(m.body_jntadr[m.body(task).id])]);pairs={frozenset((p['geom1'],p['geom2'])) for p in config['steps'][steps[ti]]}
  for offset in [np.zeros(3)]+[np.eye(3)[k]*sign*distance for distance in [.00025,.0005,.001,.003] for k in range(3) for sign in [-1,1]]:
   shifted=state.copy();shifted[1+address:1+address+3]+=offset;depth=[]
   for e in es:
    e.set_states(np.array([0]),shifted[None],np.array([ti]));depth.append(min([float(c.dist) for c in e.data[0].contact if frozenset(e.m.geom(int(g)).name for g in c.geom) in pairs]+[0]))
   assert abs(depth[0]-depth[1])<1e-12,(task,offset.tolist(),depth)
   insert.append(dict(task=task,offset_m=offset.tolist(),depths_m=depth))
finally:
 for e in es:e.close()
(REL/'static-qualification.json').write_text(json.dumps(dict(status='PASS',model_arrays_exact=checked,option_difference_only='timestep .00025 to .0005',starts=initial,insertion_controls=insert,all1143_payloads_finite=True),indent=2)+'\n');print(json.dumps(dict(event='static_PASS',arrays=checked,starts=len(initial),insertion_controls=len(insert))),flush=True)
results=[];failures=[]
for group,states in dynamics.items():
 es=envs(len(states));S=np.array([s for _,_,s in states]);T=np.array([t for t,_,_ in states]);timings=[];captures=[]
 try:
  for e in es:
   assert e.substeps in (160,80);e.set_states(np.arange(len(states)),S,T);first_guard=np.full(len(states),999);first_success=np.full(len(states),999);maxdepth=np.zeros(len(states));poses=[];t=time.perf_counter()
   for k in range(12):
    action=np.zeros((len(states),12));action[:,[0,6]]=.02*np.sin(k*.25)
    reward,done,info=e.step(action);assert np.isfinite(reward).all() and not info['unstable'].any()
    for i in np.flatnonzero(info['safety']):first_guard[i]=min(first_guard[i],k)
    for i in np.flatnonzero(info['success']):first_success[i]=min(first_success[i],k)
    maxdepth=np.maximum(maxdepth,info['penetration']);poses.append(np.array([d.body(meta['task_names'][ti]).xpos.copy() for d,ti in zip(e.data,T)]))
   timings.append(time.perf_counter()-t);captures.append(dict(first_guard=first_guard,first_success=first_success,maxdepth=maxdepth,poses=np.array(poses),final_grasp=[grasp_pairs(e.m,d) for d in e.data]));print(json.dumps(dict(event='dynamic_group',group=group,substeps=e.substeps,seconds=timings[-1],guard=first_guard.tolist(),success=first_success.tolist())),flush=True)
  b,c=captures
  for i,(ti,stage,state) in enumerate(states):
   common=min(int(b['first_guard'][i]),int(c['first_guard'][i]),12);delta=float(np.max(np.linalg.norm(b['poses'][:common,i]-c['poses'][:common,i],axis=1))) if common else 0.0
   new_failure=bool(b['first_guard'][i]==999 and c['first_guard'][i]!=999);lost=list(b['final_grasp'][i]-c['final_grasp'][i]) if b['first_guard'][i]==999 and c['first_guard'][i]==999 else []
   if new_failure or delta>.0005 or lost or (b['first_success'][i]!=999 and c['first_success'][i]==999):failures.append(dict(group=group,stage=stage,new_guard=new_failure,max_position_delta_m=delta,lost_grasp_pairs=lost))
   results.append(dict(group=group,task=meta['task_names'][ti],stage=stage,baseline_guard=int(b['first_guard'][i]),candidate_guard=int(c['first_guard'][i]),baseline_success=int(b['first_success'][i]),candidate_success=int(c['first_success'][i]),max_position_delta_m=delta,baseline_maxdepth=float(b['maxdepth'][i]),candidate_maxdepth=float(c['maxdepth'][i]),lost_grasp_pairs=lost))
 finally:
  for e in es:e.close()
result=dict(status='PASS' if not failures else 'FAIL',old_signature=oldid,new_signature=newid,physics_changed_only_timestep=True,control_dt=.04,old_substeps=160,new_substeps=80,dynamic_controls=240,dynamic_seconds_simulated=9.6,scope='10original held/near starts, paired0.48s command/retention dynamics,125fixed-pose insertion controls; not completeassemblyconvergencequalification',rows=results,failures=failures)
(REL/'qualification.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)
assert not failures,failures
