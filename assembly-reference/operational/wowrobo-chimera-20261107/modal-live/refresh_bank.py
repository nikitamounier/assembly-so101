from pathlib import Path
import json,hashlib
import mujoco,numpy as np
ROOT=Path(__file__).resolve().parents[4];C=ROOT/"robotics-assembly-3.0";S=ROOT/"assembly-so101";HERE=Path(__file__).parent
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
profile=C/"outputs/equilibrium-model-20261107/profile-centered-v6/profile.json"
p=json.loads(profile.read_text())
from assembly_policy.wowrobo.runtime import load_runtime
m,physical,signature=load_runtime(profile)
snap=HERE/'accepted-states-v5';nominal=snap/'nominals';meta=json.loads((nominal/'stages/manifest.json').read_text());frozen=json.loads((snap/'manifest.json').read_text())
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
bank=C/'assets_local/wowrobo/training-accepted-v5';bank.mkdir(exist_ok=False)
for task,values in rows.items():np.savez_compressed(bank/(task+'_train.npz'),states=np.stack([v['state'] for v in values]),family=np.array([v['family'] for v in values]),stage=np.array([v['stage'] for v in values]))
manifest=dict(schema='wowrobo-fullphysics-eq-ctrl-warmstart-friction-v1',model_signature=signature,source_model_signature=sha(p['audit_model_path']),profile_path=str(profile),review_model=p['audit_model_path'],task_names=tasks,training_qualified=False,debug_unqualified=True,minimum_initial_reset_valid=True,stationarity_required=False,passed_source_stages=25,records=records,state_size=fullsize+m.neq+m.nu+m.nv+3,qualification='Accepted minimum-valid prototypes and TRAIN variations; exact episode friction preserved. EVAL held out. Full task/production admission separate.')
(bank/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
result=dict(status='prepared',bank=str(bank),profile=str(profile),model_signature=signature,prototypes=25,accepted_train=frozen['accepted_train'],heldout_eval=frozen['accepted_eval'],training_rows=sum(map(len,rows.values())),rows_by_task={k:len(v) for k,v in rows.items()},state_size=manifest['state_size'])
(ROOT/'reset-update-v5.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)
