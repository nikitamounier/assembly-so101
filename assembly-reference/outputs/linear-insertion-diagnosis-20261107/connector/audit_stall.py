from pathlib import Path
import json,numpy as np,mujoco,trimesh,igl
from scipy.spatial.transform import Rotation
from assembly_policy.so101.wowrobo_contact_backend import load_backend
ROOT=Path(__file__).resolve().parents[3];P=ROOT/'outputs/linear-insertion-aligned-20261107/connector';out=Path(__file__).parent
manifest=json.loads((ROOT/'outputs/linear-insertion-updated-20261107/grids/manifest.json').read_text());load_backend(ROOT/'outputs/linear-insertion-updated-20261107/grids/manifest.json');m=mujoco.MjModel.from_xml_path(str(P/'guide-scene.xml'));d=mujoco.MjData(m);z=np.load(P/'final-state.npz');d.qpos[:]=z['qpos'];d.qvel[:]=z['qvel'];r=json.loads((P/'receipt.json').read_text());m.geom_contype[:]=0;m.geom_conaffinity[:]=0
for name in r['active_contact_geoms']:
 i=m.geom(name).id;m.geom_contype[i]=1;m.geom_conaffinity[i]=2
for name in r['receiver_contact_geoms']:
 i=m.geom(name).id;m.geom_contype[i]=2;m.geom_conaffinity[i]=1
m.body_contype[:]=0;m.body_conaffinity[:]=0
for i in range(m.ngeom):
 b=int(m.geom_bodyid[i]);m.body_contype[b]|=m.geom_contype[i];m.body_conaffinity[b]|=m.geom_conaffinity[i]
mujoco.mj_forward(m,d);meshes={};trans={}
for name in manifest['meshes']:
 g=m.geom(name).id;mi=int(m.geom_dataid[g]);rm=Rotation.from_quat(m.mesh_quat[mi],scalar_first=True).as_matrix();rw=d.geom_xmat[g].reshape(3,3)@rm.T;pw=d.geom_xpos[g]-rw@m.mesh_pos[mi];trans[name]=(rw,pw);meshes[name]=trimesh.load_mesh(manifest['meshes'][name]['source_file'],process=False)
def local(name,world):
 R,p=trans[name];return (np.asarray(world)-p)@R
def world(name,local):
 R,p=trans[name];return np.asarray(local)@R.T+p
def exact(name,points):
 points=np.ascontiguousarray(points);mesh=meshes[name];_,distance,_=trimesh.proximity.closest_point(mesh,points);winding=igl.winding_number(np.ascontiguousarray(mesh.vertices),np.ascontiguousarray(mesh.faces,dtype=np.int64),points);return np.where(winding>.5,-distance,distance),winding
rows=[];axis=np.array(r['axis_world'])
for contact in d.contact:
 a=m.geom(contact.geom1).name.removesuffix('__contact');b=m.geom(contact.geom2).name.removesuffix('__contact');row={'a':a,'b':b,'native_dist_m':float(contact.dist),'native_normal_world':contact.frame[:3].tolist(),'native_normal_axis_dot':float(contact.frame[:3]@axis),'native_contact_position_world_m':contact.pos.tolist(),'source_surface_checks':[]}
 for source,target in [(a,b),(b,a)]:
  mesh=meshes[source];p=local(source,[contact.pos]);near,dist,face=trimesh.proximity.closest_point(mesh,p);w=world(source,near);value,winding=exact(target,local(target,w));normal_world=mesh.face_normals[int(face[0])]@trans[source][0].T;row['source_surface_checks'].append({'source':source,'target':target,'nearest_source_triangle':int(face[0]),'nearest_source_surface_world_m':w[0].tolist(),'nearest_source_surface_normal_world':normal_world.tolist(),'source_face_normal_axis_dot':float(normal_world@axis),'signed_distance_to_target_exact_m':float(value[0]),'winding_in_target':float(winding[0]),'contact_to_source_surface_m':float(dist[0])})
 rows.append(row)
result={'native_stall_travel_mm':r['final_axial_mm'],'axis_world':axis.tolist(),'native_contacts_recomputed':len(rows),'contacts':rows,'scope':'Exact source-triangle audit at native stall contacts; physical force attribution is separate matched ablation.'};(out/'stall-contact-surface-audit.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
