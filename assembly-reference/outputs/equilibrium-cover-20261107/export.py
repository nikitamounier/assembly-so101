import hashlib,json
from pathlib import Path
import mujoco,numpy as np
OUT=Path(__file__).parent;ROOT=OUT.parents[1]
m=mujoco.MjModel.from_xml_path(str(ROOT/'assets_local/wowrobo/photo-resets-repaired-cover-support-v2/scene.xml'))
names=['shoulder_pan','shoulder_lift','elbow_flex','wrist_flex','wrist_roll','gripper'];idx=np.array([m.joint('left_'+n).qposadr[0]for n in names]);acts=np.array([m.actuator('left_'+n).id for n in names]);mask=np.ones(m.nq,bool);mask[idx]=False
result={}
for name in ['pickup-capture-v1','aligned-capture-v2']:
 report=json.loads((OUT/(name+'.json')).read_text());bounds=json.loads((OUT/(name+'-boundaries.json')).read_text())
 with np.load(OUT/(name+'.npz'))as z:
  assert np.array_equal(z['initial_qpos'][mask],z['source_qpos'][mask]);q=z['initial_qpos'][idx];ctrl=z['initial_ctrl'][acts]
  result[report['stage']]={'recipe_name':name,'source_nominal_bank':'assets_local/wowrobo/photo-resets-repaired-cover-support-v2','physical_profile':'outputs/equilibrium-model-20261107/profile-bearing-v4','left_joint_order':names,'left_qpos':q.tolist(),'left_controls_with_gravity_preload':ctrl.tolist(),'source_state_sha256':hashlib.sha256(z['source_qpos'].tobytes()).hexdigest(),'only_six_LEFT_initial_coordinates_changed':True,'jaw_closing_preload_rad':report['preload_rad'],'apply_note':'Set LEFT qpos to recipe; set actual position controls to qpos plus gravity compensation once, then subtract jaw_closing_preload_rad from LEFT jaw command. Do not add gravity compensation again to supplied absolute controls. No target coordinate or RIGHT coordinate changes. Re-settle and independently hold the whole scene after applying; local cover hold alone is not whole-scene admission.','local_hold':{k:report[k]for k in ['minimum_loads_N','cover_hold_motion_m','cover_hold_rotation_deg','cover_settle_motion_m','controls_unchanged','hold_max_global_velocity']},'native_boundaries':{k:{'candidate_rejected':v['candidate_rejected'],'max_sampled_penetration_m':v['max_sampled_penetration_m']}for k,v in bounds.items()},'training_eligible':False}
(OUT/'capture-recipes.json').write_text(json.dumps(result,indent=2)+'\n')
