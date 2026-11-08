from pathlib import Path
import json,hashlib,time
import numpy as np,mujoco
from assembly_policy.wowrobo.env import BatchedAssemblyEnv,EnvConfig,STATE_SPEC
from assembly_policy.wowrobo.bank import check_bank
ROOT=Path(__file__).resolve().parents[4];C=ROOT/'robotics-assembly-3.0';bank=C/'assets_local/wowrobo/training-minimum-24-v1';meta=check_bank(bank,allow_debug=True)
env=BatchedAssemblyEnv(meta['task_names'],np.array([0]),EnvConfig(profile_path=meta['profile_path'],review_model=meta['review_model'],nthread=1));rows=[]
for ti,task in enumerate(meta['task_names']):
 z=np.load(bank/(task+'_train.npz'))
 for state,stage in zip(z['states'],z['stage']):
  env.set_states(np.array([0]),state[None],np.array([ti]));np.testing.assert_array_equal(env.last_ctrl[0],state[env.eq_end:env.eq_end+env.m.nu]);np.testing.assert_array_equal(env.initial_warmstart[0],state[env.eq_end+env.m.nu:]);assert np.isfinite(env.observe()['proprio']).all();rows.append(dict(task=task,stage=str(stage),state_size=len(state),reset_controls_exact=True,reset_warmstart_exact=True,finite_obs=True))
 # Compare first zero-action control interval with native rollout using the saved witness inputs.
 state=z['states'][-1];env.set_states(np.array([0]),state[None],np.array([ti]));expected=env.engine.rollout(env.m,env.work,env.state[:,:env.full_size].copy(),np.repeat(np.r_[env.last_ctrl[0],state[env.full_size:env.eq_end]][None,None,:],env.substeps,axis=1),control_spec=mujoco.mjtState.mjSTATE_CTRL|mujoco.mjtState.mjSTATE_EQ_ACTIVE,nstep=env.substeps,initial_warmstart=env.initial_warmstart.copy())[0][0,-1].copy()
 env.set_states(np.array([0]),state[None],np.array([ti]));reward,done,info=env.step(np.zeros((1,12)));error=float(np.max(np.abs(expected-env.state[0,:env.full_size])));np.testing.assert_allclose(expected,env.state[0,:env.full_size],rtol=0,atol=1e-10);assert np.isfinite(reward).all();print(json.dumps(dict(task=task,stage=str(z['stage'][-1]),zero_action_native_state_error=error,safety=bool(info['safety'][0]))),flush=True)
env.close()
result=dict(status='PASS',stages=meta['passed_source_stages'],training_rows=len(rows),reset_controls_and_warmstart_exact=True,zero_action_native_rollout_parity=True,model_signature=meta['model_signature'],scope='24passed source minimum checks, finite trainer observations/reset conversion and five native zero-action parity cases; not full task success or production learning qualification',rows=rows)
(ROOT/'reset-update-validation-v3.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k!='rows'}),flush=True)
