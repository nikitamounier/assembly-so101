import json
from pathlib import Path
import mujoco,numpy as np,trimesh
from assembly_policy.so101.wowrobo_contact_backend import load_backend,set_surface_mode
from assembly_policy.so101.wowrobo_boundaries import world_mesh
ROOT=Path(__file__).resolve().parents[2];OUT=Path(__file__).parent
p=json.loads((ROOT/'outputs/equilibrium-model-20261107/profile-bearing-v5/profile.json').read_text());load_backend(p['grid_manifest']);set_surface_mode(1);m=mujoco.MjModel.from_xml_path(p['model_path']);d=mujoco.MjData(m)
r={x['id']:x for x in json.loads((ROOT/'assets_local/wowrobo/photo-resets-repaired-cover-support-v2/stage-receipts.json').read_text())};res={}
for stage in ['21_upper_far_side','22_upper_half','23_upper_near','29_final_far_side','30_final_half','31_final_near']:
 path=OUT/(stage+'-initializer.npz')
 if not path.exists():continue
 with np.load(path)as z:d.qpos[:]=z['qpos']
 mujoco.mj_kinematics(m,d);support=r[stage]['support_contact'];task=r[stage]['definition']['task'];bid=m.body(task).id
 receiver=support['receiver_body'];mesh=trimesh.util.concatenate([world_mesh(m,d,g)for g in range(m.ngeom)if m.geom_type[g]==mujoco.mjtGeom.mjGEOM_MESH and m.body(m.geom_bodyid[g]).name==receiver]);points=[]
 for g in range(m.ngeom):
  if m.geom_type[g]!=mujoco.mjtGeom.mjGEOM_MESH or not m.geom(g).name.startswith(task+'_printed'):continue
  active=world_mesh(m,d,g);points.extend(active.vertices);points.extend(active.triangles_center[::3])
 points=np.asarray(points);local=(points-d.xpos[bid])@d.xmat[bid].reshape(3,3);center=np.array(support['far_point_piece_local_m']);rows=[]
 for radius in [.015,.03,.05]:
  subset=points[np.linalg.norm(local-center,axis=1)<radius]
  if not len(subset):continue
  near,dist,_=trimesh.proximity.closest_point(mesh,subset);i=int(dist.argmin());rows.append({'patch_radius_m':radius,'sample_count':len(subset),'gap_m':float(dist[i]),'incoming_point_world':subset[i].tolist(),'receiver_point_world':near[i].tolist()})
 res[stage]=rows;print(stage,rows,flush=True)
(OUT/'far-patch-gaps.json').write_text(json.dumps(res,indent=2)+'\n')
