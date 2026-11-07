import json
from pathlib import Path
import mujoco,numpy as np,trimesh
from assembly_policy.so101.wowrobo_boundaries import NativeBoundaryAudit,world_mesh
out=Path(__file__).resolve().parent;r=json.loads((out/'evaluated-14.json').read_text());m=mujoco.MjModel.from_xml_path('assets_local/wowrobo/photo-resets-repaired-v1/scene.xml');d=mujoco.MjData(m)
with np.load('assets_local/wowrobo/photo-resets-repaired-v1/stages/18_upper_floor_grasp_train.npz') as f:mujoco.mj_setState(m,d,f['states'][0],mujoco.mjtState.mjSTATE_FULLPHYSICS)
idx=[m.joint('left_'+n).qposadr[0] for n in ['shoulder_pan','shoulder_lift','elbow_flex','wrist_flex','wrist_roll','gripper']];d.qpos[idx]=r['q_left'];mujoco.mj_forward(m,d)
print('Rsite',d.site_xmat[m.site('left_gripperframe').id].reshape(3,3))
for p in r['pads']:
 mesh=world_mesh(m,d,p['geom']);near,dist,face=trimesh.proximity.closest_point(mesh,[p['target_point_world_m']]);print('normal',mesh.face_normals[face[0]],'dot',mesh.face_normals[face[0]]@p['target_normal_world'],'gap',dist)
