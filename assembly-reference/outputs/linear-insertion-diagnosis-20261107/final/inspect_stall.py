from pathlib import Path
import json,numpy as np,mujoco,ctypes,trimesh
from assembly_policy.so101.wowrobo_contact_backend import load_backend,surface_library
from assembly_policy.so101.wowrobo_boundaries import world_mesh
from scripts.probe_wowrobo_aligned_force import descendants
root=Path.cwd();out=root/'outputs/linear-insertion-diagnosis-20261107/final';p=root/'outputs/linear-insertion-aligned-20261107/final';load_backend(root/'outputs/linear-insertion-updated-20261107/grids/manifest.json');m=mujoco.MjModel.from_xml_path(str(p/'guide-scene.xml'));d=mujoco.MjData(m);a=np.load(p/'final-state.npz');d.qpos[:]=a['qpos'];d.qvel[:]=a['qvel'];active=descendants(m,'forearm_module');fixtures=set().union(*(descendants(m,x) for x in ['base_servo','base_side_cover','shoulder_connector','assembly_pin','upper_arm_module']));m.geom_contype[:]=0;m.geom_conaffinity[:]=0
for g in range(m.ngeom):
 if m.geom_type[g]!=mujoco.mjtGeom.mjGEOM_SDF:continue
 b=int(m.geom_bodyid[g])
 if b in active:m.geom_contype[g]=1;m.geom_conaffinity[g]=2
 elif b in fixtures:m.geom_contype[g]=2;m.geom_conaffinity[g]=1
m.body_contype[:]=0;m.body_conaffinity[:]=0
for g in range(m.ngeom):
 b=int(m.geom_bodyid[g]);m.body_contype[b]|=m.geom_contype[g];m.body_conaffinity[b]|=m.geom_conaffinity[g]
mujoco.mj_forward(m,d);axis=np.array(json.loads((p/'alignment.json').read_text())['axis_world']);lib=surface_library();lib.wowrobo_sdf_distance.argtypes=[ctypes.c_void_p,ctypes.c_void_p,ctypes.c_int,np.ctypeslib.ndpointer(dtype=np.float64,shape=(3,))];lib.wowrobo_sdf_distance.restype=ctypes.c_double
rows=[]
for c in d.contact[:d.ncon]:
 row={'geom1':m.geom(c.geom1).name,'geom2':m.geom(c.geom2).name,'distance_m':float(c.dist),'position_world_m':c.pos.tolist(),'normal_world':c.frame[:3].tolist(),'normal_axis_dot':float(c.frame[:3]@axis),'friction_coefficients':c.friction.tolist(),'dim':int(c.dim),'geometry':[]}
 for g in [c.geom1,c.geom2]:
  name=m.geom(g).name;vg=m.geom(name.removesuffix('__contact')).id;mesh=world_mesh(m,d,vg);near,dist,tri=trimesh.proximity.closest_point(mesh,c.pos[None]);inside=mesh.contains(c.pos[None]);n=mesh.face_normals[tri[0]];row['geometry'].append({'name':name,'nearest_triangle_point_world_m':near[0].tolist(),'exact_unsigned_distance_m':float(dist[0]),'trimesh_inside':bool(inside[0]),'nearest_triangle_normal_world':n.tolist(),'triangle_normal_axis_dot':float(n@axis),'mesh_watertight':bool(mesh.is_watertight)})
 rows.append(row)
r={'source':'Aligned final-state at4seconds, re-forwarded static contact geometry (force response decomposed separately)','qpos':d.qpos.tolist(),'qvel':d.qvel.tolist(),'axis_world':axis.tolist(),'contacts':rows};(out/'stall-contact-geometry.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r,indent=2))
from assembly_policy.so101.wowrobo_balanced_push import BalancedPush
push=BalancedPush(m,d,active,axis,damping=10);push.apply(d,.5);mujoco.mj_forward(m,d);decomp=[]
for i,c in enumerate(d.contact[:d.ncon]):
 f=np.zeros(6);mujoco.mj_contactForce(m,d,i,f);frame=c.frame.reshape(3,3);sign=1 if int(m.geom_bodyid[c.geom2]) in active else -1;normal=sign*frame[0]*f[0];friction=sign*(frame[1]*f[1]+frame[2]*f[2]);row={'contact_id':i,'force_contact_frame':f.tolist(),'normal_world_N':normal.tolist(),'friction_world_N':friction.tolist(),'normal_axial_reaction_N':float(normal@axis),'friction_axial_reaction_N':float(friction@axis),'total_axial_reaction_N':float((normal+friction)@axis)};decomp.append(row)
r['static_force_decomposition']={'applied_N':.5,'qfrc_applied':d.qfrc_applied.tolist(),'qfrc_constraint':d.qfrc_constraint.tolist(),'qacc':d.qacc.tolist(),'contacts':decomp,'note':'Recomputed mj_forward at saved baselineq/qvel with matching0.5N balancedforce+COMdrag. Not a nofriction counterfactual.'};(out/'stall-contact-geometry.json').write_text(json.dumps(r,indent=2)+'\n');print('DECOMPOSITION',json.dumps(r['static_force_decomposition'],indent=2))
if d.ncon:
 c=d.contact[0];receivergeom=m.geom('upper_arm_module_printed_0').id;casegeom=m.geom('servo_3_case').id;receiver_mesh=world_mesh(m,d,receivergeom);case_mesh=world_mesh(m,d,casegeom);casepoint=trimesh.proximity.closest_point(case_mesh,c.pos[None])[0][0];rows=[]
 for delta in [-.0005,-.0002,-.0001,0,.0001,.0002,.0005]:
  x=casepoint+delta*axis;near,dist,tri=trimesh.proximity.closest_point(receiver_mesh,x[None]);inside=bool(receiver_mesh.contains(x[None])[0]);rows.append({'axial_increment_m':delta,'case_surface_point_world_m':x.tolist(),'inside_watertight_receiver':inside,'signed_distance_to_receiver_m':float(-dist[0] if inside else dist[0])})
 r['tracked_source_case_surface_point_exact_axial_sweep']=rows;(out/'stall-contact-geometry.json').write_text(json.dumps(r,indent=2)+'\n');print('EXACTPOINTSWEEP',json.dumps(rows,indent=2))
