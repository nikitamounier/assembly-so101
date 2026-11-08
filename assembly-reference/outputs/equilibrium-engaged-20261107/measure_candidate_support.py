import json
from pathlib import Path
import mujoco,numpy as np,trimesh
from assembly_policy.so101.wowrobo_contact_backend import load_backend,set_surface_mode
from assembly_policy.so101.wowrobo_boundaries import world_mesh
ROOT=Path(__file__).resolve().parents[2];OUT=Path(__file__).parent
p=json.loads((ROOT/'outputs/equilibrium-model-20261107/profile-bearing-v5/profile.json').read_text());load_backend(p['grid_manifest']);set_surface_mode(1);m=mujoco.MjModel.from_xml_path(p['model_path']);d=mujoco.MjData(m)
r={x['id']:x for x in json.loads((ROOT/'assets_local/wowrobo/photo-resets-repaired-cover-support-v2/stage-receipts.json').read_text())}
res={}
for stage in ['21_upper_far_side','22_upper_half','23_upper_near','29_final_far_side','30_final_half','31_final_near']:
 path=OUT/(stage+'-initializer.npz')
 if not path.exists():continue
 with np.load(path)as z:d.qpos[:]=z['qpos']
 mujoco.mj_kinematics(m,d);support=r[stage]['support_contact'];task=r[stage]['definition']['task'];bid=m.body(task).id
 far=d.xpos[bid]+d.xmat[bid].reshape(3,3)@np.array(support['far_point_piece_local_m']);receiver=support['receiver_body'];geoms=[g for g in range(m.ngeom)if m.geom_type[g]==mujoco.mjtGeom.mjGEOM_MESH and m.body(m.geom_bodyid[g]).name==receiver]
 mesh=trimesh.util.concatenate([world_mesh(m,d,g)for g in geoms]);near,dist,_=trimesh.proximity.closest_point(mesh,[far]);res[stage]={'far_world':far.tolist(),'receiver_nearest':near[0].tolist(),'gap_m':float(dist[0])};print(stage,res[stage],flush=True)
(OUT/'candidate-support.json').write_text(json.dumps(res,indent=2)+'\n')
