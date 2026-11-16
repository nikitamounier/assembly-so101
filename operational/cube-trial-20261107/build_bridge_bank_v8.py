"""Add native partial-contact and light-grasp acquisition reset families."""
import json,hashlib,time
from pathlib import Path
import mujoco,numpy as np
from assembly_policy.cube_trial.core import model,_solve,placement_contacts
from assembly_policy.cube_trial.env import BatchedCubeEnv,EnvConfig
ROOT=Path.cwd();source=ROOT/'data/cube_trial/banks_policy_transport_v7';dest=ROOT/'data/cube_trial/banks_handoff_bridge_v8';dest.mkdir(exist_ok=True)
m=model();spec=mujoco.mjtState.mjSTATE_INTEGRATION;cube=m.geom('cube_geom').id;weight=float(m.body_mass[m.body('cube').id]*abs(m.opt.gravity[2]));rng=np.random.default_rng(202611073)
oldcontext={k:v.copy() for k,v in np.load(source/'reset-context.npz').items()};contexts={k:list(v) for k,v in oldcontext.items()};provenance=[];context_by_hash={};start=time.time()
def support(d):
 forces=np.zeros((2,3))
 for k,c in enumerate(d.contact):
  if cube not in (c.geom1,c.geom2):continue
  other=c.geom2 if c.geom1==cube else c.geom1;name=m.geom(other).name or ''
  if name.startswith(('left_','right_')):
   f=np.empty(6);mujoco.mj_contactForce(m,d,k,f);forces[int(name.startswith('right_'))]+=c.frame.reshape(3,3).T@f[:3]*(1 if c.geom2==cube else -1)
 return forces[:,2]
for split,n in [('train',128),('eval',32)]:
 old={k:v.copy() for k,v in np.load(source/f'cube_handoff_{split}.npz').items()};pre=np.flatnonzero(old['family']==3);new={9:[],10:[]};receipts={9:[],10:[]}
 for ix,entry in enumerate(pre):
  chosen={}
  for attempt in range(36):
   entry=int(pre[(ix+attempt//6)%len(pre)])
   e=BatchedCubeEnv(['cube_handoff'],[0],EnvConfig(nthread=1));e.set_states([0],old['states'][entry:entry+1]);d=e.data[0];solve=mujoco.MjData(m);mujoco.mj_setState(m,solve,e.state[0],spec)
   zoff=float(rng.uniform(-.0007,-.00015));_solve(m,solve,'right',d.body('cube').xpos+np.array([0,0,zoff]),d.body('cube').xmat.reshape(3,3).copy(),np.zeros(3));target=np.r_[solve.qpos[6:11],rng.uniform(.32,.37)];partial=[];light=[];close_step=float(rng.uniform(.00025,.0005))
   for step in range(110):
    if step>=30:target[5]=max(.2715,.287-(step-30)*close_step)
    act=np.zeros((1,12));act[0,6:]=np.clip((target-e.last_ctrl[0,6:])/np.r_[[.035]*5,.05],-2,2);_,done,info=e.step(act);c=placement_contacts(m,d);loads=c['pad_loads_n'];fz=support(d)
    common=not c['table_touching'] and not e.off_target_table_contact[0] and min(loads[:2])>1 and fz[0]>.55*weight and abs(fz.sum()-weight)<.08
    family=None
    if common and .015<loads[2]<.4 and loads[3]<=.001:
     distance=mujoco.mj_geomDistance(m,d,cube,m.geom('right_moving_pad').id,.02,None)
     if .0002<distance<.003:family=9
    if common and min(loads[2:])>.05 and max(loads[2:])<3.5 and max(loads[2:])<.65*min(loads[:2]):family=10
    if family is not None:
     state=e.state[0].copy();context=(e.hist[0].copy(),e.prev_action[0].copy(),float(d.time-e.last_right_grasp_time[0]) if np.isfinite(e.last_right_grasp_time[0]) else 1e6,bool(e.carry_started[0]));record=dict(source_bank=source.name,source_entry=int(entry),source_split=split,source_step=step+1,source_time_s=float(d.time),z_offset_m=zoff,pad_loads_n=loads.tolist(),left_vertical_support_n=float(fz[0]),right_vertical_support_n=float(fz[1]),cube_weight_n=weight,cube_position=d.body('cube').xpos.tolist(),right_jaw_rad=float(d.qpos[11]),native_contact_only=True)
     (partial if family==9 else light).append((state,context,record))
   e.close()
   if partial and light:
    chosen={9:partial[int(rng.integers(len(partial)))],10:light[int(rng.integers(len(light)))]};break
  assert chosen,(split,int(entry),'no qualified pair')
  for family,(state,context,record) in chosen.items():new[family].append(state);receipts[family].append(record);context_by_hash[hashlib.sha256(state.tobytes()).hexdigest()]=context
  if ix%16==0:print(json.dumps(dict(split=split,built=ix+1,elapsed_s=time.time()-start)),flush=True)
 out={k:np.concatenate([v,np.zeros((2*n,*v.shape[1:]),dtype=v.dtype)]) for k,v in old.items()};out['states'][:len(old['states'])]=old['states'];row=len(old['states'])
 for family in [9,10]:
  for j,(state,record) in enumerate(zip(new[family],receipts[family])):
   out['states'][row]=state;out['family'][row]=family
   if 'seed' in out:out['seed'][row]=(190000 if family==9 else 200000)+(0 if split=='train' else 1000000)+j
   # Retrieve the corresponding exact context selected with this snapshot.
   # Context arrays are retained separately below using its snapshot hash.
   record.update(split=split,row=row,family='partial_right_contact' if family==9 else 'light_right_grasp',holder=['left'] if family==9 else ['left','right'],state_sha256=hashlib.sha256(state.tobytes()).hexdigest());provenance.append(record);row+=1
 # states' contexts recorded in context_by_hash during selection.
 np.savez_compressed(dest/f'cube_handoff_{split}.npz',**out)

for r in provenance:
 key=r['state_sha256'];history,action,age,carry=context_by_hash[key]
 for k,v in [('state_sha256',key),('histories',history),('previous_actions',action),('right_grasp_age_s',age),('carry_started',carry)]:contexts[k].append(v)
np.savez_compressed(dest/'reset-context.npz',**{k:np.asarray(v,dtype=oldcontext[k].dtype) for k,v in contexts.items()})
manifest=json.loads((source/'manifest.json').read_text());manifest['family_names']+=['Partial right contact','Light right grasp'];manifest.update(reset_distribution='cube-handoff-acquisition-bridge-v8',reset_context_sha256=hashlib.sha256((dest/'reset-context.npz').read_bytes()).hexdigest(),new_families=['partial_right_contact','light_right_grasp'],existing_rows_unchanged=True,bridge_contact_contract='MuJoCo loaded native contacts; left supports>55%cubeweight; partial one pad, opposite clear>0.2mm; light both pads0.05–3.5N',train_per_family=128,eval_per_family=32)
(dest/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n');oldq=json.loads((source/'qualification.json').read_text());(dest/'qualification.json').write_text(json.dumps(oldq+provenance,indent=2)+'\n');(dest/'bridge-provenance.json').write_text(json.dumps(provenance,indent=2)+'\n');print(json.dumps(dict(done=True,new_states=len(provenance),elapsed=time.time()-start)),flush=True)
