from pathlib import Path
import json,mujoco,numpy as np,trimesh
from assembly_policy.so101.wowrobo_boundaries import world_mesh
out=Path('outputs/insertion-fix-20261107/final/local-cad-v2');a=json.loads(Path('outputs/linear-insertion-aligned-20261107/final/alignment.json').read_text());m=mujoco.MjModel.from_xml_path('outputs/linear-insertion-20261107/source-snapshot/scene.xml');d=mujoco.MjData(m);d.qpos[:]=a['reference_qpos'];mujoco.mj_forward(m,d);rr=d.body('upper_arm_module').xmat.reshape(3,3);rp=d.body('upper_arm_module').xpos;cm=world_mesh(m,d,m.geom('servo_3_case').id);case=np.vstack([cm.vertices,cm.triangles_center]);case=(case-rp)@rr;rec=trimesh.load_mesh(out/'upper_arm_module_printed_0.obj',process=False);up=-np.array(a['axis_world'])@rr;allp=[];alld=[];alln=[];allq=[]
for depth in np.linspace(0,.025,51):
 p=case+depth*up;mask=np.all((p>=rec.bounds[0])&(p<=rec.bounds[1]),axis=1);p=p[mask];inside=rec.contains(p);p=p[inside]
 if len(p):
  near,dist,tri=trimesh.proximity.closest_point(rec,p);allp.append(p);alld.append(dist);alln.append(rec.face_normals[tri]);allq.append(np.full(len(p),depth))
np.savez_compressed(out/'case-sweep-inside-derived.npz',points=(np.vstack(allp)*1000 if allp else np.empty((0,3))),distances=(np.hstack(alld)*1000 if alld else np.empty(0)),normals=(np.vstack(alln) if alln else np.empty((0,3))),standoff=(np.hstack(allq)*1000 if allq else np.empty(0)));print('insidepoints',sum(len(p) for p in allp),'max_mm',max([max(x) for x in alld],default=0)*1000,flush=True)
