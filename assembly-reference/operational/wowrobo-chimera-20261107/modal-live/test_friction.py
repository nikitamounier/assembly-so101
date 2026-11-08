"""Compare threaded native friction with the admission split-step oracle."""
from pathlib import Path
import sys,time,json
import mujoco,numpy as np
from mujoco import rollout
ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT))
from assembly_policy.so101.wowrobo_physical_resets import physical_split_step
from assembly_policy.so101.wowrobo_rollout_friction import install

m=mujoco.MjModel.from_xml_string('<mujoco><size nuserdata="3"/><option integrator="implicitfast" timestep=".002"/><worldbody><geom name="floor" type="plane" size="1 1 .1" friction=".3 .005 .0001"/><body name="base_servo" pos="0 0 .05"><freejoint/><geom type="sphere" size=".05" friction=".3 .005 .0001"/></body></worldbody></mujoco>')
N=2;T=100;spec=mujoco.mjtState.mjSTATE_FULLPHYSICS
initial=[];expected=[];draws=[.97,1.03]
for factor in draws:
 d=mujoco.MjData(m);d.qvel[0]=.3
 state=np.empty(mujoco.mj_stateSize(m,spec));mujoco.mj_getState(m,d,state,spec);initial.append(state)
 rows=[]
 for _ in range(T):
  physical_split_step(m,d,dict(finger_part=1,part_table=factor,part_part=1))
  row=np.empty_like(state);mujoco.mj_getState(m,d,row,spec);rows.append(row)
 expected.append(rows)
lib=install(m,ROOT/'outputs/native-rollout-friction-20261107')
controls=np.array([[[1,f,1]]*T for f in draws])
try:
 with rollout.Rollout(nthread=2) as engine:
  actual,_=engine.rollout(m,[mujoco.MjData(m) for _ in range(2)],np.array(initial),controls,control_spec=mujoco.mjtState.mjSTATE_USERDATA,nstep=T)
 error=float(np.max(np.abs(actual-np.asarray(expected))))
 np.testing.assert_allclose(actual,expected,rtol=0,atol=1e-12)
 assert np.max(np.abs(actual[0]-actual[1]))>1e-6
 result=dict(status='PASS',threaded_worlds=N,substeps=T,native_vs_admission_max_error=error,independent_friction_draws=draws,control_spec_userdata=True)
 (Path(__file__).parent/'friction-toy-verification.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
finally:lib.wowrobo_remove_friction()
