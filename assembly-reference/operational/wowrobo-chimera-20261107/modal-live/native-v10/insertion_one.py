"""Paired actual gripper commands toward original receiver-relative insertion goals."""
from pathlib import Path
import json,importlib.util,sys,time
import numpy as np,mujoco,torch
from assembly_policy.wowrobo import env as native
from assembly_policy.wowrobo.runtime import load_runtime
ROOT=Path(__file__).resolve().parent.parent;REL=ROOT/'native-v10';C=ROOT/'robotics-assembly-3.0';bank=C/'assets_local/wowrobo/training-accepted-v6';meta=json.loads((bank/'manifest.json').read_text());variant=sys.argv[1];torch.set_num_threads(1)
if variant=='baseline':m,profile,sig=load_runtime(meta['profile_path'])
else:
 spec=importlib.util.spec_from_file_location('_insert_candidate',REL/'wowrobo_training_runtime.py');mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod);m,profile,sig=mod.load_native(meta['profile_path'])
native.load_runtime=lambda _:(m,profile,sig);tasks=meta['task_names'];e=native.BatchedAssemblyEnv(tasks,np.arange(5),native.EnvConfig(profile_path=meta['profile_path'],review_model=meta['review_model'],nthread=5));S=[];stages=[]
for task in tasks:
 z=np.load(bank/(task+'_train.npz'));ix=next(i for i in np.flatnonzero(z['family']==4) if np.all(z['states'][i,-3:]==1));S.append(z['states'][ix]);stages.append(str(z['stage'][ix]))
S=np.array(S);e.set_states(np.arange(5),S,np.arange(5));commands=[];poses=[];errors=[];guard=np.full(5,999);finished=np.zeros(5,bool)
reference=json.loads((REL/'insertion-baseline.json').read_text()) if variant=='candidate' else None
right_dofs=m.jnt_dofadr[m.actuator_trnid[6:11,0]];site=m.site('right_gripperframe').id

def error(i):
 snap=e.reward.snapshot(e.data[i],dict(task=tasks[i]));return np.asarray(snap['target_box']['rotation_row_major']).reshape(3,3)@np.asarray(snap['checks'][-1]['position_error_m'])
errors.append([float(np.linalg.norm(error(i))) for i in range(5)]);t=time.perf_counter()
try:
 for k in range(6):
  action=np.zeros((5,12))
  if reference is not None:action=np.asarray(reference['commands'][k])
  else:
   for i,d in enumerate(e.data):
    if finished[i]:continue
    delta=-error(i);delta*=min(1,.0002/max(np.linalg.norm(delta),1e-12));jp=np.zeros((3,m.nv));jr=np.zeros_like(jp);mujoco.mj_jacSite(m,d,jp,jr,site);A=np.vstack((jp[:,right_dofs],.05*jr[:,right_dofs]));dq=np.linalg.solve(A.T@A+1e-8*np.eye(5),A.T@np.r_[delta,[0,0,0]]);action[i,6:11]=np.clip(dq/.035,-.1,.1)
  commands.append(action.tolist());r,done,info=e.step(action);assert np.isfinite(r).all() and not info['unstable'].any()
  poses.append([d.body(task).xpos.tolist() for d,task in zip(e.data,tasks)]);errors.append([float(np.linalg.norm(error(i))) for i in range(5)])
  for i in np.flatnonzero(info['safety']):guard[i]=min(guard[i],k)
  finished|=done
  ended=np.flatnonzero(done)
  if len(ended):e.set_states(ended,S[ended],ended)
 x=dict(variant=variant,signature=sig,steps=6,simulated_seconds=.24,stages=stages,commands=commands,poses=poses,errors_m=errors,first_guard=guard.tolist(),seconds=time.perf_counter()-t,scope='Small native right-gripper joint commands toward unchanged geometric seating target; paired response/guard check, not fullstrokeassemblycertification');(REL/('insertion-'+variant+'.json')).write_text(json.dumps(x,indent=2)+'\n');print(json.dumps({k:v for k,v in x.items() if k not in ('commands','poses')}),flush=True)
finally:e.close()
