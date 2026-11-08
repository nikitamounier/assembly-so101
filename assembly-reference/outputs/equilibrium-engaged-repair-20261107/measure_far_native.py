import json,ctypes
from pathlib import Path
import mujoco,numpy as np
from assembly_policy.so101.wowrobo_contact_backend import load_backend,set_surface_mode,surface_library
from assembly_policy.so101.wowrobo_boundaries import world_mesh
ROOT=Path(__file__).resolve().parents[2];OUT=Path(__file__).parent;OLD=ROOT/'outputs/equilibrium-engaged-20261107'
p=json.loads((ROOT/'outputs/equilibrium-model-20261107/profile-bearing-v5/profile.json').read_text());load_backend(p['grid_manifest']);set_surface_mode(1);m=mujoco.MjModel.from_xml_path(p['model_path']);d=mujoco.MjData(m)
f=surface_library().wowrobo_sdf_distance;ptr=ctypes.POINTER(ctypes.c_double);f.argtypes=[ctypes.c_void_p,ctypes.c_void_p,ctypes.c_int,ptr];f.restype=ctypes.c_double
r={x['id']:x for x in json.loads((ROOT/'assets_local/wowrobo/photo-resets-repaired-cover-support-v2/stage-receipts.json').read_text())};res={}
for stage in ['21_upper_far_side','22_upper_half','23_upper_near','29_final_far_side','30_final_half','31_final_near']:
 with np.load(OLD/(stage+'-initializer.npz'))as z:qnew=z['qpos'].copy()
 with np.load(ROOT/'assets_local/wowrobo/photo-resets-repaired-cover-support-v2/stages'/(stage+'_train.npz'))as z:mujoco.mj_setState(m,d,z['states'][0],mujoco.mjtState.mjSTATE_FULLPHYSICS);qold=d.qpos.copy()
 support=r[stage]['support_contact'];task=r[stage]['definition']['task'];bid=m.body(task).id;receiver=support['receiver_body'];gids=[g for g in range(m.ngeom)if m.geom_type[g]==mujoco.mjtGeom.mjGEOM_SDF and m.body(m.geom_bodyid[g]).name==receiver];rows={}
 for label,q in [('source',qold),('clear_hinge',qnew)]:
  d.qpos[:]=q;mujoco.mj_kinematics(m,d);mujoco.mj_comPos(m,d);points=[]
  for g in range(m.ngeom):
   if m.geom_type[g]==mujoco.mjtGeom.mjGEOM_MESH and m.geom(g).name.startswith(task+'_printed'):
    active=world_mesh(m,d,g);points.extend(active.vertices)
  points=np.asarray(points);local=(points-d.xpos[bid])@d.xmat[bid].reshape(3,3);center=np.array(support['far_point_piece_local_m']);values=[]
  for radius in [.015,.03,.05]:
   subset=points[np.linalg.norm(local-center,axis=1)<radius]
   if not len(subset):continue
   if len(subset)>500:subset=subset[np.linspace(0,len(subset)-1,500,dtype=int)]
   dist=np.array([min(f(m._address,d._address,g,np.ascontiguousarray(pt,dtype=float).ctypes.data_as(ptr))for g in gids)for pt in subset]);i=int(np.argmin(np.abs(dist)));values.append({'radius_m':radius,'count':len(subset),'signed_nearest_m':float(dist[i]),'minimum_signed_m':float(dist.min()),'point':subset[i].tolist()})
  rows[label]=values
 res[stage]=rows;print(stage,rows,flush=True)
(OUT/'far-native.json').write_text(json.dumps(res,indent=2)+'\n')
