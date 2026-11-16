import json
from pathlib import Path
import mujoco,numpy as np,trimesh
from assembly_policy.so101.wowrobo_boundaries import world_mesh
ROOT=Path(__file__).resolve().parents[2];OUT=Path(__file__).parent;m=mujoco.MjModel.from_xml_path(str(ROOT/'assets_local/wowrobo/photo-resets-repaired-cover-support-v2/scene.xml'));d=mujoco.MjData(m);ranges=json.loads((OUT/'own-ranges.json').read_text());res={}
for stage,rows in ranges.items():
 servo=2 if stage.startswith('21')else 3;h=m.joint(f'target_servo_{servo}_shaft');body=m.body(f'target_servo_{servo}_pivot').id
 with np.load(ROOT/'assets_local/wowrobo/photo-resets-repaired-cover-support-v2/stages'/(stage+'_train.npz'))as z:mujoco.mj_setState(m,d,z['states'][0],mujoco.mjtState.mjSTATE_FULLPHYSICS)
 mujoco.mj_kinematics(m,d);q0=d.qpos.copy();res[stage]=[]
 for row in rows:
  if row['delta_rad']not in [0,.2,.4]:continue
  d.qpos[:]=q0;d.qpos[h.qposadr[0]]=row['theta_rad'];mujoco.mj_kinematics(m,d);axis=d.xaxis[h.id];anchor=d.xanchor[h.id];points=np.array([d.xpos[body]+d.xmat[body].reshape(3,3)@np.array(p['case_local'])for p in row['contacts']]);mesh=world_mesh(m,d,m.geom(f'servo_{servo}_case').id)
  if not len(points):continue
  near,dist,face=trimesh.proximity.closest_point(mesh,points);norm=mesh.face_normals[face];records=[]
  for p,pt,nn,dd,npnt in zip(row['contacts'],points,norm,dist,near):
   records.append({**p,'case_raw_surface_distance_m':float(dd),'case_raw_normal_dot_axis':float(nn@axis),'case_raw_hinge_normal_lever_m':float(nn@np.cross(axis,npnt-anchor)),'case_raw_normal':nn.tolist(),'case_raw_nearest':npnt.tolist()})
  res[stage].append({'theta_rad':row['theta_rad'],'delta_rad':row['delta_rad'],'contacts':records});print(stage,row['delta_rad'],[(round(x['case_raw_surface_distance_m']*1000,3),round(x['rad']*1000,1),round(x['ax']*1000,1),round(x['case_raw_normal_dot_axis'],3),round(x['case_raw_hinge_normal_lever_m']*1000,3))for x in records],flush=True)
(OUT/'raw-face-normals.json').write_text(json.dumps(res,indent=2)+'\n')
