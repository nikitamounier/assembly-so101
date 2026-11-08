import json,hashlib,mujoco,numpy as np,importlib.util
from pathlib import Path
from assembly_policy.so101.wowrobo_obs import kinematic_fingerprint,mesh_fingerprint,PointTables
spec=importlib.util.spec_from_file_location('reg','robotics-assembly-3.0/assembly_policy/so101/wowrobo_housing_registration.py');reg=importlib.util.module_from_spec(spec);spec.loader.exec_module(reg)
r=Path('robotics-assembly-3.0/assets_local/wowrobo/initial-valid-resets-v1');before=mujoco.MjModel.from_xml_string(Path('robotics-assembly-3.0/assets_local/wowrobo/photo-resets-repaired-cover-support-v2/scene.xml').read_text());after=mujoco.MjModel.from_xml_string((r/'scene.xml').read_text())
for name in ['wowrobo_point_features','wowrobo_reward_regions']:
 c=json.loads(Path('assembly-so101/configs/'+name+'.json').read_text());c['model_signature']=hashlib.sha256((r/'scene.xml').read_bytes()).hexdigest()
 if 'kinematic_fingerprint' in c:
  assert c['kinematic_fingerprint']==kinematic_fingerprint(before)
  c['kinematic_fingerprint']=kinematic_fingerprint(after)
  c['mesh_fingerprints']=[{'id':i,'name':after.geom(i).name or '', 'sha256':mesh_fingerprint(after,i)} for i in range(after.ngeom) if after.geom_type[i]==mujoco.mjtGeom.mjGEOM_MESH and after.geom_group[i] in (0,2) and after.geom_rgba[i,3]>0]
 for task,t in c['tasks'].items():
  if 'reference_qpos' not in t:continue
  old=mujoco.MjData(before);old.qpos[:]=t['reference_qpos'];mujoco.mj_kinematics(before,old);orr=old.body(t['receiver']).xmat.reshape(3,3);world_center=old.body(t['receiver']).xpos+orr@np.asarray(t['center_receiver_local']);world_rotation=orr@np.asarray(t['rotation_receiver_local'])
  q,_=reg.migrate_qpos(before,after,np.asarray(t['reference_qpos']),propagate_final_with_receiver=task=='forearm_module');t['reference_qpos']=q.tolist();d=mujoco.MjData(after);d.qpos[:]=q;mujoco.mj_kinematics(after,d);rr=d.body(t['receiver']).xmat.reshape(3,3);t['center_receiver_local']=(rr.T@(world_center-d.body(t['receiver']).xpos)).tolist();t['rotation_receiver_local']=(rr.T@world_rotation).tolist()
 (r/(name+'.json')).write_text(json.dumps(c,indent=2)+'\n')
 if name.endswith('features'):tables=PointTables(after,config=c);print('Point contract rebuilt:',tables.n_pts,flush=True)
print('Bank-local viewer contracts generated')
