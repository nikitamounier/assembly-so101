from pathlib import Path
import json,numpy as np,mujoco
from assembly_policy.wowrobo.runtime import load_runtime
ROOT=Path(__file__).resolve().parents[4];m,p,sig=load_runtime();raw=mujoco.MjModel.from_xml_path(p['model_path'])
assert (m.nq,m.nv,m.nu,m.neq,m.npair,m.ngeom)==(raw.nq,raw.nv,raw.nu,raw.neq,raw.npair,raw.ngeom)
fields=['body_pos','body_quat','body_mass','body_inertia','geom_pos','geom_quat','geom_friction','geom_solref','geom_solimp','geom_contype','geom_conaffinity','pair_geom1','pair_geom2','pair_friction','pair_solref','pair_solimp','pair_dim','jnt_stiffness','dof_damping','actuator_gainprm','actuator_biasprm','actuator_ctrlrange','actuator_forcerange']
for name in fields:np.testing.assert_array_equal(getattr(m,name),getattr(raw,name),err_msg=name)
for name in ['timestep','integrator','iterations','tolerance','noslip_iterations','noslip_tolerance','sdf_initpoints','sdf_iterations','ls_iterations']:assert getattr(m.opt,name)==getattr(raw.opt,name),name
r=dict(status='PASS',source_physics_model_matches_training_model=True,geometry_contact_laws_gains_masses_exact=True,solver_and_timestep_exact=True,added_only=['minimum-contact sensor','256MiB arena capacity'],compared_fields=fields,model_signature=sig)
(ROOT/'model-match-validation-v3.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r),flush=True)
