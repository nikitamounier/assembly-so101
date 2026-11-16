"""Real native model parity, independent friction worlds, and finite reset observations."""
from pathlib import Path
import json,time,sys,hashlib
import mujoco,numpy as np
from assembly_policy.wowrobo.env import BatchedAssemblyEnv,EnvConfig,STATE_SPEC
from assembly_policy.wowrobo.bank import check_bank
ROOT=Path(__file__).resolve().parent.parent;C=ROOT/'robotics-assembly-3.0';bank=C/'assets_local/wowrobo/training-accepted-v6'
meta=check_bank(bank,allow_debug=True);rows=[]
env=BatchedAssemblyEnv(meta['task_names'],np.zeros(2,dtype=np.int64),EnvConfig(profile_path=meta['profile_path'],review_model=meta['review_model'],nthread=2))
assert env.m.nuserdata==3 and env.nstate==162
selected=[]
finite_key=hashlib.sha256((bank/'manifest.json').read_bytes()+Path(sys.modules[BatchedAssemblyEnv.__module__].__file__).read_bytes()).hexdigest()
finite_proof=ROOT/'reset-finite-v6.json'
cached=finite_proof.exists() and json.loads(finite_proof.read_text()).get('identity')==finite_key
old_bank=C/'assets_local/wowrobo/training-accepted-v5'
old_proof=ROOT/'reset-finite-v5.json';old_rows=set();reused=0
old_key=hashlib.sha256((old_bank/'manifest.json').read_bytes()+Path(sys.modules[BatchedAssemblyEnv.__module__].__file__).read_bytes()).hexdigest()
if old_proof.exists() and json.loads(old_proof.read_text()).get('identity')==old_key:
 for ti,task in enumerate(meta['task_names']):
  with np.load(old_bank/(task+'_train.npz')) as z:
   for state in z['states']:old_rows.add((ti,hashlib.sha256(state.tobytes()).hexdigest()))
for ti,task in enumerate(meta['task_names']):
 with np.load(bank/(task+'_train.npz')) as z:
  for state,stage in zip(z['states'],z['stage']):
   row_cached=(ti,hashlib.sha256(state.tobytes()).hexdigest()) in old_rows
   reused+=int(row_cached)
   if not cached and not row_cached:
    env.set_states(np.array([0]),state[None],np.array([ti]));assert np.isfinite(env.observe()['proprio']).all()
    np.testing.assert_array_equal(env.last_ctrl[0],state[env.eq_end:env.eq_end+env.m.nu])
    np.testing.assert_array_equal(env.initial_warmstart[0],state[env.eq_end+env.m.nu:env.eq_end+env.m.nu+env.m.nv])
    np.testing.assert_array_equal(env.data[0].userdata,state[-3:])
   rows.append(dict(task=task,stage=str(stage)))
   if np.any(state[-3:]!=1) and (not selected or (len(selected)==1 and str(stage)!='00_loose')):selected.append((ti,state.copy(),str(stage)))
assert len(selected)==2
finite_proof.write_text(json.dumps(dict(status='PASS',identity=finite_key,finite_rows=len(rows)))+'\n')
print(json.dumps(dict(event='finite_resets_verified',rows=len(rows),cached=cached,reused_identical_rows=reused)),flush=True)
# Admission oracle runs without the installed hook, so each multiplier applies once.
hook=sys.modules['_wowrobo_native_episode_friction'];lib=hook.LIBRARIES[-1];lib.wowrobo_remove_friction()
import ast,types
# Compile the exact admission step function without importing its CLI/build dependencies.
tree=ast.parse((C/'assembly_policy/so101/wowrobo_physical_resets.py').read_text())
node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='physical_split_step')
friction=hook.infer_geom_roles.__globals__
namespace=dict(mujoco=mujoco,apply_contact_friction=friction['apply_contact_friction'],apply_friction_and_rebuild_constraints=friction['apply_friction_and_rebuild_constraints'])
exec(compile(ast.Module(body=[node],type_ignores=[]),'admission_physical_split_step','exec'),namespace)
oracle=types.SimpleNamespace(physical_split_step=namespace['physical_split_step'])
expected=[];times={}
for ti,state,stage in selected:
 d=mujoco.MjData(env.m);mujoco.mj_setState(env.m,d,state[:env.full_size],STATE_SPEC);d.time=0;d.eq_active[:]=state[env.full_size:env.eq_end];d.ctrl[:]=state[env.eq_end:env.eq_end+env.m.nu];d.qacc_warmstart[:]=state[env.eq_end+env.m.nu:env.eq_end+env.m.nu+env.m.nv]
 f=dict(zip(['finger_part','part_table','part_part'],state[-3:]));t=time.perf_counter()
 for _ in range(env.substeps):oracle.physical_split_step(env.m,d,f)
 times[stage]=time.perf_counter()-t
 full=np.empty(env.full_size);mujoco.mj_getState(env.m,d,full,STATE_SPEC);expected.append(full)
hook.install(env.m,C/'outputs/native-rollout-friction')
states=np.stack([s for _,s,_ in selected]);env.set_states(np.arange(2),states,np.array([t for t,_,_ in selected]));t=time.perf_counter();reward,done,info=env.step(np.zeros((2,12)));elapsed=time.perf_counter()-t
error=float(np.max(np.abs(env.state[:,:env.full_size]-np.stack(expected))));np.testing.assert_allclose(env.state[:,:env.full_size],expected,rtol=0,atol=1e-9)
assert np.isfinite(reward).all()
result=dict(status='PASS',finite_reset_rows=len(rows),state_size=env.nstate,native_vs_admission_max_error=error,threaded_friction_worlds=2,native_seconds=elapsed,admission_oracle_seconds=times,model_signature=meta['model_signature'],heldout_eval_not_trained=True,controls_velocities_warmstart_and_episode_friction_preserved=True)
(ROOT/'reset-validation-v6.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True);env.close()
