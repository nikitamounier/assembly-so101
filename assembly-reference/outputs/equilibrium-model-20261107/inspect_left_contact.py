import json
from pathlib import Path
import mujoco,numpy as np
from assembly_policy.so101.wowrobo_contact_backend import load_backend,set_surface_mode
from assembly_policy.so101.wowrobo_physical_resets import ROOT,WowRoboPhysicalValidator

out=ROOT/'outputs/equilibrium-model-20261107'
profile=json.loads((ROOT/'assets_local/wowrobo/physical-resets-repaired-cover-support-v2/profile.json').read_text())
load_backend(profile['grid_manifest']);set_surface_mode(1)
m=mujoco.MjModel.from_xml_path(profile['model_path'])
records=json.loads(Path(profile['grid_manifest']).read_text())['meshes']
z=np.load(ROOT/'outputs/cover-support-fix-20261107/physical-precontact/18_upper_floor_grasp.npz')
validator=object.__new__(WowRoboPhysicalValidator);validator.model=m;validator.grid_records=records
result={}
for label in ['input','settled']:
 d=mujoco.MjData(m)
 if label=='settled':
  mujoco.mj_setState(m,d,z['settled_state'],mujoco.mjtState(int(z['state_spec'])))
  d.ctrl[:]=z['settled_controls']
 else:
  d.qpos[:]=z['input_qpos'];d.ctrl[:]=z['input_controls']
 mujoco.mj_forward(m,d)
 site=m.site('left_gripperframe').id;frame=d.site_xmat[site].reshape(3,3)
 contacts=[]
 for i in range(d.ncon):
  c=d.contact[i];a,b=map(int,c.geom);ba,bb=int(m.geom_bodyid[a]),int(m.geom_bodyid[b])
  names=[m.body(k).name for k in (ba,bb)]
  if not any(n in ('left_gripper','left_moving_jaw_so101_v1')for n in names):continue
  hand_geom=a if names[0] in ('left_gripper','left_moving_jaw_so101_v1')else b
  hand_body=int(m.geom_bodyid[hand_geom]);hand_name=m.geom(hand_geom).name.removesuffix('__contact')
  local=(c.pos-d.site_xpos[site])@frame
  normal=c.frame[:3]*(1 if hand_geom==a else -1)
  sign=1 if m.body(hand_body).name=='left_gripper' else -1
  force=np.zeros(6);mujoco.mj_contactForce(m,d,i,force)
  mesh=records.get(hand_name,{}).get('source_mesh_name','')
  flags=dict(nonpositive_distance=bool(c.dist<=0),correct_mesh=any(k in mesh for k in ['wrist_roll_follower','moving_jaw']),pad_x=bool(-.035<local[0]<.004),pad_y=bool(abs(local[1])<.010),inward_normal=bool(sign*normal@frame[:,2]>.25))
  contacts.append(dict(index=i,geom_names=[m.geom(g).name for g in(a,b)],body_names=names,source_mesh=mesh,dist_m=float(c.dist),point_site_m=local.tolist(),normal_site=(normal@frame).tolist(),signed_inward_normal=float(sign*normal@frame[:,2]),force=force.tolist(),mask_flags=flags))
 result[label]=dict(time=float(d.time),site_position=d.site_xpos[site].tolist(),site_rotation=frame.tolist(),pad_loads=validator._loaded_pads(d,[('left','base_side_cover')]),contacts=contacts,qpos_left=[float(d.qpos[m.joint('left_'+n).qposadr[0]]) for n in ['shoulder_pan','shoulder_lift','elbow_flex','wrist_flex','wrist_roll','gripper']])
 print(label,result[label]['site_position'],result[label]['pad_loads'])
 for c in contacts:
  print(c['body_names'],round(c['dist_m']*1e6,1),np.round(c['point_site_m'],5),round(c['signed_inward_normal'],3),round(c['force'][0],3),c['mask_flags'])
(out/'left-contact.json').write_text(json.dumps(result,indent=2)+'\n')
