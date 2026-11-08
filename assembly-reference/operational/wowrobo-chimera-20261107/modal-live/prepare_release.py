"""Prepare a separate scaled continuation; never change active run inputs."""
from pathlib import Path
import json,hashlib
import mujoco,numpy as np
ROOT=Path(__file__).resolve().parents[4];C=ROOT/'robotics-assembly-3.0';S=ROOT/'assembly-so101'
HERE=C/'operational/wowrobo-chimera-20261107/modal-live'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
profile=C/'outputs/equilibrium-model-20261107/profile-centered-v6/profile.json'
p=json.loads(profile.read_text());p.update(model_sha256=sha(p['model_path']),source_model_sha256=sha(p['audit_model_path']),training_episode_friction='native-userdata-v1');profile.write_text(json.dumps(p,indent=2)+'\n')
f=S/'assembly_policy/wowrobo/env.py';s=f.read_text()
def replace(a,b):
 global s
 if b in s:return
 assert a in s,a;s=s.replace(a,b)
replace('self.nstate = self.eq_end + (self.m.nu + self.m.nv if self.extra_reset else 0)','self.episode_friction = self.profile.get("training_episode_friction") == "native-userdata-v1"\n        self.nstate = self.eq_end + (self.m.nu + self.m.nv if self.extra_reset else 0) + (3 if self.episode_friction else 0)')
replace('self.state[i, self.eq_end+self.m.nu:]','self.state[i, self.eq_end+self.m.nu:self.eq_end+self.m.nu+self.m.nv]')
replace('d.ctrl[:] = self.last_ctrl[i]','d.ctrl[:] = self.last_ctrl[i]\n            if self.episode_friction: d.userdata[:3] = self.state[i,-3:]')
replace('controls = np.concatenate([ctrl, self.state[:, self.full_size : self.eq_end]], axis=1)','controls = np.concatenate([ctrl, self.state[:, self.full_size : self.eq_end], *([self.state[:,-3:]] if self.episode_friction else [])], axis=1)')
replace('control_spec=mujoco.mjtState.mjSTATE_CTRL | mujoco.mjtState.mjSTATE_EQ_ACTIVE,','control_spec=mujoco.mjtState.mjSTATE_CTRL | mujoco.mjtState.mjSTATE_EQ_ACTIVE | (mujoco.mjtState.mjSTATE_USERDATA if self.episode_friction else 0),')
f.write_text(s)
for name in ['wowrobo_point_features.json','wowrobo_reward_regions.json']:
 q=S/'configs'/name;x=json.loads(q.read_text());x['model_signature']=p['source_model_sha256'];q.write_text(json.dumps(x,indent=2)+'\n')
from assembly_policy.wowrobo.runtime import load_runtime
m,physical,signature=load_runtime(profile)
snap=HERE/'accepted-states';nominal=snap/'nominals';meta=json.loads((nominal/'stages/manifest.json').read_text());frozen=json.loads((snap/'manifest.json').read_text())
tasks=json.loads((S/'configs/wowrobo_reward_regions.json').read_text())['tasks'];tasks=list(tasks)
fullsize=mujoco.mj_stateSize(m,mujoco.mjtState.mjSTATE_FULLPHYSICS);rows={t:[] for t in tasks};records=[]
def add(stage,path,state_spec,friction,kind,source):
 with np.load(path) as z:
  state=z['states'][0];ctrl=z['controls'][0]
  # Source witnesses predate the three new per-world userdata parameters.
  source_spec=int(state_spec) & ~int(mujoco.mjtState.mjSTATE_USERDATA)
  d=mujoco.MjData(m);mujoco.mj_setState(m,d,state,source_spec)
  np.testing.assert_array_equal(ctrl,d.ctrl)
  assert not np.any(d.xfrc_applied) and not np.any(d.qfrc_applied)
  full=np.empty(fullsize);mujoco.mj_getState(m,d,full,mujoco.mjtState.mjSTATE_FULLPHYSICS)
  packed=np.r_[full,d.eq_active,d.ctrl,d.qacc_warmstart,[friction[k] for k in ['finger_part','part_table','part_part']]]
 task=meta['stage_by_task'][stage]['task'];destinations=tasks if task=='initial_scene' else [task]
 family=0 if stage=='00_loose' else 4 if 'near' in stage else 3 if 'half' in stage or 'entry' in stage else 1 if 'pickup' in stage or 'floor' in stage else 2
 for task in destinations:
  rows[task].append(dict(state=packed,stage=stage,family=family));records.append(dict(stage=stage,task=task,kind=kind,source=source,source_archive_sha256=sha(path),friction_multipliers=friction))
for stage in meta['task_names']:
 add(stage,nominal/'stages'/(stage+'_train.npz'),meta['state_spec'],dict(finger_part=1,part_table=1,part_part=1),'prototype',stage)
for r in frozen['records']:
 if r['split']!='train':continue
 path=snap/r['archive'];assert sha(path)==r['archive_sha256']
 add(r['stage'],path,mujoco.mjtState.mjSTATE_INTEGRATION,r['friction_multipliers'],'variation',r['archive'])
bank=C/'assets_local/wowrobo/training-accepted-v4';bank.mkdir(exist_ok=False)
for task,values in rows.items():np.savez_compressed(bank/(task+'_train.npz'),states=np.stack([v['state'] for v in values]),family=np.array([v['family'] for v in values]),stage=np.array([v['stage'] for v in values]))
manifest=dict(schema='wowrobo-fullphysics-eq-ctrl-warmstart-friction-v1',model_signature=signature,source_model_signature=sha(p['audit_model_path']),profile_path=str(profile),review_model=p['audit_model_path'],task_names=tasks,training_qualified=False,debug_unqualified=True,minimum_initial_reset_valid=True,stationarity_required=False,passed_source_stages=25,records=records,state_size=fullsize+m.neq+m.nu+m.nv+3,qualification='Accepted minimum-valid prototypes and TRAIN variations; exact episode friction preserved. EVAL held out. Full task/production admission separate.')
(bank/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
result=dict(status='prepared',bank=str(bank),profile=str(profile),model_signature=signature,prototypes=25,accepted_train=frozen['accepted_train'],heldout_eval=frozen['accepted_eval'],training_rows=sum(map(len,rows.values())),rows_by_task={k:len(v) for k,v in rows.items()},state_size=manifest['state_size'])
(ROOT/'reset-update-v4.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)
