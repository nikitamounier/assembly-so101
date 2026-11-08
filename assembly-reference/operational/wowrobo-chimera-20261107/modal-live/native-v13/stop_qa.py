"""Actual-model limit activation/residuals, including non-unity friction and noise."""
from pathlib import Path
import concurrent.futures,json
import mujoco,numpy as np
from assembly_policy.wowrobo.runtime import load_runtime
ROOT=Path(__file__).resolve().parent.parent;REL=ROOT/'native-v13';bank=REL/'training-accepted-v13'
meta=json.loads((bank/'manifest.json').read_text());m,profile,signature=load_runtime(meta['profile_path'])
with np.load(bank/'base_side_cover_train.npz') as z:state=z['states'][0].copy()
full=mujoco.mj_stateSize(m,mujoco.mjtState.mjSTATE_FULLPHYSICS);eqend=full+m.neq
joints=[m.joint(n).id for n in ('target_servo_2_shaft','target_servo_3_shaft')]
original=m.jnt_range.copy();amp=profile['hinge_limit_noise_rad'];jobs=[]
for k,joint in enumerate(joints):
    for endpoint in (0,1):
        for noise in (-.8*amp,0,.8*amp):
            for inside in (False,True):jobs.append((k,joint,endpoint,noise,inside))
def run(job):
    k,joint,endpoint,noise,inside=job;d=mujoco.MjData(m)
    mujoco.mj_setState(m,d,state[:full],mujoco.mjtState.mjSTATE_FULLPHYSICS)
    d.eq_active[:]=0  # Deliberate mechanism probe, not a candidate training reset.
    d.ctrl[:]=state[eqend:eqend+m.nu];d.userdata[:3]=[.9,1.1,.95]
    d.userdata[3:7]=0;d.userdata[3+2*k+endpoint]=noise
    boundary=original[joint,endpoint]+noise
    direction=1 if endpoint==0 else -1
    q=boundary+direction*(1e-5 if inside else -1e-5)
    d.qpos[m.jnt_qposadr[joint]]=q;d.qvel[:]=0
    mujoco.mj_forward(m,d)
    ix=np.flatnonzero((d.efc_type[:d.nefc]==int(mujoco.mjtConstraint.mjCNSTR_LIMIT_JOINT))&(d.efc_id[:d.nefc]==joint))
    assert len(ix)==(0 if inside else 1),(job,ix.tolist())
    residual=None
    if len(ix):
        residual=float(d.efc_pos[ix[0]]);assert abs(residual+1e-5)<1e-10,(job,residual)
        assert d.efc_force[ix[0]]>=0
    assert np.isfinite(d.qacc).all() and not any(w.number for w in d.warning)
    np.testing.assert_array_equal(m.jnt_range,original)
    return dict(joint=int(joint),endpoint=endpoint,noise_rad=noise,inside=inside,active=bool(len(ix)),constraint_residual=residual)
with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:rows=list(pool.map(run,jobs))
result=dict(status='PASS',physics_signature=signature,probes=len(rows),nominal_and_both_noise_signs=True,nonunity_friction=True,shared_ranges_unchanged=True,rows=rows,scope='Actual-model static limit activation/residual mechanism probes; other welds disabled only in these probes, not production.')
(REL/'stop-response.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k!='rows'}),flush=True)
