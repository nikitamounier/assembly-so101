"""Paired bounded native timing on current repaired CAD; never training admission."""
import hashlib
import json
from pathlib import Path
import signal
import time
import xml.etree.ElementTree as ET
import mujoco
import numpy as np
from assembly_policy.so101.wowrobo_contact_backend import load_backend,set_surface_mode
from assembly_policy.so101.wowrobo_physical_resets import physical_split_step
from assembly_policy.so101.wowrobo_friction import apply_contact_friction,infer_geom_roles,CATEGORIES
ROOT=Path(__file__).resolve().parents[2]
OUT=Path(__file__).resolve().parent
PROFILE=ROOT/'assets_local/wowrobo/physical-resets-repaired-v1'
signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('Bounded native probe exceeded180seconds')))
signal.alarm(180)
backend=load_backend(PROFILE/'grids/manifest.json',build_dir=OUT/'native')
set_surface_mode(1)
m=mujoco.MjModel.from_xml_path(str(PROFILE/'physical.xml'))
roles=infer_geom_roles(m)
unit=dict.fromkeys(CATEGORIES,1.)
noisy=dict(finger_part=1.02,part_table=.98,part_part=1.01)
def cached(d,multipliers):
    mujoco.mj_step1(m,d)
    apply_contact_friction(m,d,multipliers,geom_roles=roles)
    mujoco.mj_makeConstraint(m,d);mujoco.mj_island(m,d);mujoco.mj_projectConstraint(m,d)
    mujoco.mj_transmission(m,d);mujoco.mj_fwdVelocity(m,d);mujoco.mj_step2(m,d)
methods={'native_unit':lambda d:mujoco.mj_step(m,d),'split_unit':lambda d:physical_split_step(m,d,unit),
         'cached_roles_unit':lambda d:cached(d,unit),'split_noise':lambda d:physical_split_step(m,d,noisy),
         'cached_roles_noise':lambda d:cached(d,noisy)}
result=dict(scope='Same current rejected diagnostic states; timing and parity only, not valid training samples',
            mujoco=mujoco.__version__,model_sha256=hashlib.sha256((PROFILE/'physical.xml').read_bytes()).hexdigest(),
            backend=backend,surface_mode=1,timestep=m.opt.timestep,
            substeps_per_40ms_control=int(round(.04/m.opt.timestep)),nq=m.nq,nv=m.nv,ngeom=m.ngeom,
            rows=[],same_fixed_solver=True)
for stage in ('00_loose','14_pin_half'):
    z=np.load(ROOT/'outputs/repaired-reset-physics-20261107/default-window-probes'/f'{stage}.npz')
    seed=z['settled_state'];ctrl=z['settled_controls']
    measured={}
    for name,fn in methods.items():
        times=[];last=None
        for repeat in range(3):
            d=mujoco.MjData(m);mujoco.mj_setState(m,d,seed,mujoco.mjtState.mjSTATE_INTEGRATION);d.ctrl[:]=ctrl
            mujoco.mj_forward(m,d)
            start=time.perf_counter()
            for _ in range(8):fn(d)
            times.append(time.perf_counter()-start)
            last=np.empty_like(seed);mujoco.mj_getState(m,d,last,mujoco.mjtState.mjSTATE_INTEGRATION)
        measured[name]=dict(median_8step_s=float(np.median(times)),samples_s=times,state=last)
    for a,b in [('native_unit','split_unit'),('split_unit','cached_roles_unit'),('split_noise','cached_roles_noise')]:
        err=float(np.max(np.abs(measured[a]['state']-measured[b]['state'])))
        np.testing.assert_allclose(measured[a]['state'],measured[b]['state'],rtol=0,atol=1e-12)
        measured[b]['parity_vs_'+a]=err
    for row in measured.values():row.pop('state')
    result['rows'].append(dict(stage=stage,methods=measured,
        unit_split_over_native=measured['split_unit']['median_8step_s']/measured['native_unit']['median_8step_s'],
        noisy_role_cache_speedup=measured['split_noise']['median_8step_s']/measured['cached_roles_noise']['median_8step_s']))
    (OUT/'performance.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result['rows'][-1]),flush=True)
signal.alarm(0)
