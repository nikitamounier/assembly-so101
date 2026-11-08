"""Prepare an isolated matched-model/reset release; never edit the active run."""
from pathlib import Path
import json,hashlib,os,xml.etree.ElementTree as ET
import mujoco,numpy as np
ROOT=Path(__file__).resolve().parents[4];C=ROOT/'robotics-assembly-3.0';S=ROOT/'assembly-so101';HERE=Path(__file__).parent
OLD='/Users/nikitamounier/Documents/ChatGPT/robot assembly 2'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
for f,h in json.loads((HERE/'inputs.json').read_text()).items():assert sha(ROOT/f)==h,f
for f in json.loads((HERE/'inputs.json').read_text()):
 p=ROOT/f
 if p.suffix in ('.json','.xml'):p.write_text(p.read_text().replace(OLD,str(ROOT)))
profile=C/'outputs/equilibrium-model-20261107/profile-centered-v6/profile.json';p=json.loads(profile.read_text());p['model_sha256']=sha(p['model_path']);p['source_model_sha256']=sha(p['audit_model_path']);p['training_reset_payload']='fullphysics-eq-ctrl-warmstart-v1';profile.write_text(json.dumps(p,indent=2)+'\n')
# The bridge continues using the one frozen canonical factory, pointed at this profile.
f=S/'assembly_policy/wowrobo/runtime.py';s=f.read_text().replace('assets_local/wowrobo/physical-resets-repaired-cover-support-v2/profile.json','outputs/equilibrium-model-20261107/profile-centered-v6/profile.json').replace('assets_local/wowrobo/randomization-bank-repaired-cover-support-v2/scene.xml','outputs/equilibrium-model-20261107/profile-centered-v6/source.xml');f.write_text(s)
# Preserve actual saved helper controls and warmstart as well as full physics/equalities.
f=S/'assembly_policy/wowrobo/env.py';s=f.read_text();s=s.replace('self.nstate = self.full_size + self.m.neq','self.extra_reset = self.profile.get("training_reset_payload") == "fullphysics-eq-ctrl-warmstart-v1"\n        self.eq_end = self.full_size + self.m.neq\n        self.nstate = self.eq_end + (self.m.nu + self.m.nv if self.extra_reset else 0)\n        self.reset_bias = np.zeros((self.N, self.m.nu))\n        self.initial_warmstart = np.zeros((self.N, self.m.nv))\n        self.needs_warmstart = np.zeros(self.N, bool)')
s=s.replace('self.last_ctrl[i] = self.data[i].qpos[self.ctrl_adr]', 'self.last_ctrl[i] = (self.state[i, self.eq_end:self.eq_end+self.m.nu] if self.extra_reset else self.data[i].qpos[self.ctrl_adr])\n            self.reset_bias[i] = self.last_ctrl[i] - self.data[i].qpos[self.ctrl_adr]\n            if self.extra_reset:\n                self.initial_warmstart[i] = self.state[i, self.eq_end+self.m.nu:]\n                self.needs_warmstart[i] = True')
s=s.replace('self.state[i, self.full_size :]','self.state[i, self.full_size : self.eq_end]')
s=s.replace('ctrl = np.asarray([d.qpos[self.ctrl_adr] for d in self.data])','ctrl = self.last_ctrl.copy() if self.extra_reset else np.asarray([d.qpos[self.ctrl_adr] for d in self.data])')
s=s.replace('self.state[:, self.full_size :]','self.state[:, self.full_size : self.eq_end]')
s=s.replace('nstep=self.substeps,','nstep=self.substeps,\n            initial_warmstart=np.where(self.needs_warmstart[:,None], self.initial_warmstart, 0.0),')
s=s.replace('warning_during = any(', 'self.needs_warmstart[:] = False\n        warning_during = any(')
f.write_text(s)
from assembly_policy.wowrobo.runtime import load_runtime
m,physical,signature=load_runtime(profile)
review=mujoco.MjModel.from_xml_path(p['audit_model_path'])
# Correct reference fingerprints for centered frames. Source meshes are unchanged.
from assembly_policy.so101.wowrobo_obs import kinematic_fingerprint,mesh_fingerprint
feature=S/'configs/wowrobo_point_features.json';x=json.loads(feature.read_text());x['model_signature']=sha(p['audit_model_path']);x['kinematic_fingerprint']=kinematic_fingerprint(review)
for row in x['mesh_fingerprints']:row['sha256']=mesh_fingerprint(review,row['id'])
# Recompute seated target geometry from the state-migrated completed anchors.
reward=S/'configs/wowrobo_reward_regions.json';rc=json.loads(reward.read_text());rc['model_signature']=sha(p['audit_model_path'])
completed=C/'assets_local/wowrobo/minimum-validity-viewer-20261107-v1/completed'
for task,n in zip(rc['tasks'],['06','11','16','24','32']):
 z=np.load(next(completed.glob(n+'*.npz')));d=mujoco.MjData(review);mujoco.mj_setState(review,d,z['states'][0],mujoco.mjtState.mjSTATE_FULLPHYSICS);mujoco.mj_kinematics(review,d)
 goal=rc['tasks'][task];g=review.geom(goal['geom']).id;b=d.body(goal['receiver']);rot=b.xmat.reshape(3,3);center=d.geom_xpos[g]+d.geom_xmat[g].reshape(3,3)@np.asarray(goal['mesh_center_local']);goal['center_receiver_local']=(rot.T@(center-b.xpos)).tolist();goal['rotation_receiver_local']=(rot.T@d.geom_xmat[g].reshape(3,3)).tolist()
 if task in x['tasks']:
  x['tasks'][task]['center_receiver_local']=goal['center_receiver_local'];x['tasks'][task]['rotation_receiver_local']=goal['rotation_receiver_local']
feature.write_text(json.dumps(x,indent=2)+'\n');reward.write_text(json.dumps(rc,indent=2)+'\n')
meta=json.loads((C/'assets_local/wowrobo/minimum-validity-viewer-20261107-v1/stages/manifest.json').read_text());selected=json.loads((HERE/'selected.json').read_text());tasks=list(rc['tasks']);rows={t:[] for t in tasks};records=[];fullsize=mujoco.mj_stateSize(m,mujoco.mjtState.mjSTATE_FULLPHYSICS)
for stage,relative in sorted(selected.items()):
 report=json.loads((ROOT/relative).read_text());assert report['minimum_validity_pass'] and not report['minimum_validity_rejection_reasons'];assert report['physics_profile']['model_sha256']=='909e034485939b77037f9568e59d7d027e37238372e95995c1a8222803ee96ef'
 z=np.load((ROOT/relative).with_suffix('.npz'));state=z['states'][0];assert hashlib.sha256(state.tobytes()).hexdigest()==report['state_sha256'];ctrl=z['controls'][0];assert hashlib.sha256(ctrl.tobytes()).hexdigest()==report['control_sha256'];d=mujoco.MjData(m);mujoco.mj_setState(m,d,state,mujoco.mjtState(report['state_spec']));np.testing.assert_array_equal(ctrl,d.ctrl)
 assert not np.any(d.xfrc_applied) and not np.any(d.qfrc_applied),'No external reset forces allowed'
 full=np.empty(fullsize);mujoco.mj_getState(m,d,full,mujoco.mjtState.mjSTATE_FULLPHYSICS);packed=np.r_[full,d.eq_active,d.ctrl,d.qacc_warmstart];task=meta['stage_by_task'][stage]['task'];destinations=tasks if task=='initial_scene' else [task];assert all(t in tasks for t in destinations)
 for task in destinations:
  row=dict(stage=stage,state=packed,family=0 if stage=='00_loose' else 4 if 'near' in stage else 3 if 'half' in stage or 'entry' in stage else 1 if 'pickup' in stage or 'floor' in stage else 2);rows[task].append(row);records.append(dict(stage=stage,task=task,state_sha256=report['state_sha256'],report_sha256=sha(ROOT/relative),minimum_validity_pass=True,hold_seconds=report['hold_seconds']))
bank=C/'assets_local/wowrobo/training-minimum-24-v1';bank.mkdir(exist_ok=False)
for task,values in rows.items():
 assert values,task
 np.savez_compressed(bank/(task+'_train.npz'),states=np.stack([v['state'] for v in values]),family=np.array([v['family'] for v in values]),stage=np.array([v['stage'] for v in values]))
manifest=dict(schema='wowrobo-fullphysics-eq-ctrl-warmstart-v1',model_signature=signature,source_model_signature=sha(p['audit_model_path']),profile_path=str(profile),review_model=p['audit_model_path'],task_names=tasks,training_qualified=False,debug_unqualified=True,minimum_initial_reset_valid=True,stationarity_required=False,passed_source_stages=len(selected),records=records,qualification='24 passed minimum initial resets: no unintended overlaps and held parts retained; motion permitted. Full production/task admission remains separate.',state_size=fullsize+m.neq+m.nu+m.nv)
(bank/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
result=dict(model_signature=signature,stages=len(selected),training_rows=sum(map(len,rows.values())),rows_by_task={k:len(v) for k,v in rows.items()},bank=str(bank),profile=str(profile),review_model=p['audit_model_path'],warmstart_and_controls_preserved=True,physics_profile_original_sha256='909e034485939b77037f9568e59d7d027e37238372e95995c1a8222803ee96ef',missing_stage='21_upper_far_side')
(ROOT/'reset-update-v3.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)
