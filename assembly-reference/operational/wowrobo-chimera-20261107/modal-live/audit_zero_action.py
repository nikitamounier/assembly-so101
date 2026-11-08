"""Distinguish reset/guard mismatch from untrained actions on the new insertion starts."""
from pathlib import Path
import json,numpy as np
from assembly_policy.wowrobo.env import BatchedAssemblyEnv,EnvConfig
from assembly_policy.wowrobo.bank import check_bank
ROOT=Path(__file__).resolve().parents[4];C=ROOT/'robotics-assembly-3.0';bank=C/'assets_local/wowrobo/training-accepted-v5';meta=check_bank(bank,allow_debug=True)
env=BatchedAssemblyEnv(meta['task_names'],np.zeros(2,dtype=np.int64),EnvConfig(profile_path=meta['profile_path'],review_model=meta['review_model'],nthread=2))
class Capture:
 def __init__(self,engine):self.engine=engine
 def __getattr__(self,name):return getattr(self.engine,name)
 def rollout(self,*a,**kw):
  result=self.engine.rollout(*a,**kw);self.depth=np.maximum(0,-result[1][:,:,env.sensor_adr].min(axis=1));return result
capture=Capture(env.engine);env.engine=capture;chosen=[]
for task,stage in [('upper_arm_module','21_upper_far_side'),('forearm_module','29_final_far_side')]:
 ti=meta['task_names'].index(task)
 with np.load(bank/(task+'_train.npz')) as z:
  ids=np.nonzero(z['stage']==stage)[0]
  assert len(ids)>2
  for index in [ids[0],ids[1],ids[-1]]:chosen.append((ti,stage,int(index),z['states'][index].copy()))
rows=[]
for a in range(0,len(chosen),2):
 group=chosen[a:a+2];env.set_states(np.arange(2),np.stack([r[3] for r in group]),np.array([r[0] for r in group]))
 reward,done,info=env.step(np.zeros((2,12)))
 for i,(ti,stage,index,state) in enumerate(group):
  rows.append(dict(task=meta['task_names'][ti],stage=stage,index=index,zero_action_safety=bool(info['safety'][i]),unstable=bool(info['unstable'][i]),max_contact_penetration_m=float(capture.depth[i]),native_warning=bool(env.native_warning[i]),reward=float(reward[i]),friction=state[-3:].tolist()))
  print(json.dumps(rows[-1]),flush=True)
result=dict(status='PASS' if not any(r['zero_action_safety'] for r in rows) else 'RESET_GUARD_MISMATCH',control_seconds=env.ecfg.control_dt,rows=rows)
(ROOT/'zero-action-audit-v5.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(dict(status=result['status'],cases=len(rows))),flush=True);env.close()
