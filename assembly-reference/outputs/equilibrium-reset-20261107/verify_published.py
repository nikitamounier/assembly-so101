import hashlib,json
from pathlib import Path
import mujoco,numpy as np
from assembly_policy.so101.wowrobo_contact_backend import load_backend,set_surface_mode
from assembly_policy.so101.wowrobo_randomization import WowRoboRandomizer
root=Path('assets_local/wowrobo/initial-valid-resets-v1'); meta=json.loads((root/'stages/manifest.json').read_text());p=meta['physics_profile'];load_backend(p['grid_manifest']);set_surface_mode(1)
m=mujoco.MjModel.from_xml_string(Path(p['model_path']).read_text()); v=mujoco.MjModel.from_xml_string((root/'scene.xml').read_text()); maximum=0.; count=0
for stage in meta['task_names']:
 z=np.load(root/'physical-replays'/f'{stage}.npz');a=mujoco.MjData(m);mujoco.mj_setState(m,a,z['states'][0],mujoco.mjtState.mjSTATE_INTEGRATION);mujoco.mj_kinematics(m,a)
 saved=np.load(root/'stages'/f'{stage}_train.npz');b=mujoco.MjData(v);mujoco.mj_setState(v,b,saved['states'][0],mujoco.mjtState.mjSTATE_FULLPHYSICS);mujoco.mj_kinematics(v,b)
 np.testing.assert_array_equal(a.qvel,b.qvel);np.testing.assert_array_equal(a.ctrl,saved['controls'][0]);maximum=max(maximum,float(np.max(np.abs(a.xpos-b.xpos))))
 report=json.loads((root/'physical-replays'/f'{stage}.json').read_text());assert report['minimum_validity_pass'];count+=1
assert maximum<1e-12
sampler=WowRoboRandomizer(config='configs/wowrobo_randomization_minimum_valid.json');sampler.restore('23_upper_near')
result={'valid_states':count,'maximum_display_vs_physical_body_position_error_m':maximum,'physical_model_sha256':hashlib.sha256(Path(p['model_path']).read_bytes()).hexdigest(),'all_four_target_shafts_unactuated':all(m.joint(f'target_servo_{i}_shaft').id not in m.actuator_trnid[:,0] for i in range(1,5)),'stationarity_required':False}
assert count==25 and result['all_four_target_shafts_unactuated'];Path('outputs/equilibrium-reset-20261107/published-verification.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
