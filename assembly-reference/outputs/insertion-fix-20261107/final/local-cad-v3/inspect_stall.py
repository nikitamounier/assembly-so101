from pathlib import Path
import json,numpy as np,mujoco,trimesh,ctypes,igl
from assembly_policy.so101.wowrobo_contact_backend import load_backend,surface_library
from assembly_policy.so101.wowrobo_boundaries import world_mesh
from assembly_policy.so101.wowrobo_balanced_push import BalancedPush
from scripts.diagnose_wowrobo_insertion import configure
out=Path('outputs/insertion-fix-20261107/final/local-cad-v3');base=out/'coarse';load_backend(base/'grids/manifest.json');m=mujoco.MjModel.from_xml_path(str(base/'candidate-scene.xml'));active=configure(m,'forearm_module',['base_servo','base_side_cover','shoulder_connector','assembly_pin','upper_arm_module'],True);m.dof_damping[:]=1000;d=mujoco.MjData(m);s=np.load(base/'drive-state.npz');d.qpos[:]=s['qpos'];d.qvel[:]=s['qvel'];mujoco.mj_forward(m,d);a=json.loads((base/'alignment.json').read_text());axis=np.array(a['axis_world']);push=BalancedPush(m,d,active,axis,damping=0);push.apply(d,20);mujoco.mj_forward(m,d);fn=surface_library().wowrobo_sdf_distance;ptr=ctypes.POINTER(ctypes.c_double);fn.argtypes=[ctypes.c_void_p,ctypes.c_void_p,ctypes.c_int,ptr];fn.restype=ctypes.c_double
case=world_mesh(m,d,m.geom('servo_3_case').id);receiver=world_mesh(m,d,m.geom('upper_arm_module_printed_0').id);sourcecase=trimesh.load_mesh('assets_local/wowrobo/components-v2/meshes/servo_3_case.obj',process=False);bid=m.body('target_servo_3_pivot').id;rb=d.xmat[bid].reshape(3,3);bp=d.xpos[bid];rows=[]
def query(g,p):return float(fn(m._address,d._address,int(g),np.ascontiguousarray(p,dtype=float).ctypes.data_as(ptr)))
for i,c in enumerate(d.contact[:d.ncon]):
 f=np.zeros(6);mujoco.mj_contactForce(m,d,i,f);row={'geom1':m.geom(c.geom1).name,'geom2':m.geom(c.geom2).name,'native_dist_mm':float(c.dist*1000),'normal_N':float(f[0]),'normal_axis_dot':float(c.frame[:3]@axis),'world_pos_m':c.pos.tolist()}
 if 'servo_3_case__contact' in [row['geom1'],row['geom2']]:
  rc,rd,rt=trimesh.proximity.closest_point(receiver,c.pos[None]);cp,cd,ct=trimesh.proximity.closest_point(case,rc);local=(rc[0]-bp)@rb;w=float(igl.fast_winding_number(np.array(sourcecase.vertices),np.array(sourcecase.faces),local[None])[0]);row.update(receiver_surface_world_m=rc[0].tolist(),receiver_surface_case_source_mm=(local*1000).tolist(),receiver_surface_case_unsigned_mm=float(cd[0]*1000),receiver_surface_case_trimesh_inside=bool(case.contains(rc)[0]),receiver_surface_case_winding=w,receiver_surface_native_case_SDF_mm=query(m.geom('servo_3_case__contact').id,rc[0])*1000,receiver_surface_native_receiver_SDF_mm=query(m.geom('upper_arm_module_printed_0__contact').id,rc[0])*1000)
 rows.append(row)
r={'qpos_mm':float(d.qpos[0]*1000),'qvel':d.qvel.tolist(),'contacts':rows};(out/'stall-witnesses.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r,indent=2))
