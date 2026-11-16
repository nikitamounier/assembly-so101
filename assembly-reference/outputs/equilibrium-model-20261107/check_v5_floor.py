import json
from pathlib import Path
import mujoco,numpy as np
from assembly_policy.so101.wowrobo_contact_backend import load_backend,set_surface_mode
from assembly_policy.so101.wowrobo_physical_resets import ROOT
out=ROOT/'outputs/equilibrium-model-20261107';p=json.loads((out/'profile-bearing-v5/profile.json').read_text());load_backend(p['grid_manifest']);set_surface_mode(1);m=mujoco.MjModel.from_xml_path(p['model_path']);old=mujoco.MjModel.from_xml_path(str(out/'profile-bearing-v4/physical.xml'))
z=np.load(ROOT/'outputs/equilibrium-reset-20261107/natural-v1/00_loose.npz');d=mujoco.MjData(m);mujoco.mj_setState(m,d,z['states'][0],mujoco.mjtState(int(z['state_spec'])));d.ctrl[:]=z['controls'][0];mujoco.mj_forward(m,d)
floor=m.geom('review_table').id;nonfloor_changes=[]
def allowed(mm,a,b):return bool(mm.geom_contype[a]&mm.geom_conaffinity[b] or mm.geom_contype[b]&mm.geom_conaffinity[a])
for a in range(m.ngeom):
 for b in range(a):
  if floor in(a,b):continue
  if allowed(m,a,b)!=allowed(old,a,b):nonfloor_changes.append([a,b])
rows={}
for c in d.contact:
 if floor not in c.geom:continue
 g=int(c.geom[1] if c.geom[0]==floor else c.geom[0]);name=m.geom(g).name;assert not name.endswith('__contact');assert m.geom_type[g]==mujoco.mjtGeom.mjGEOM_MESH
 mid=int(m.geom_dataid[g]);a,n=m.mesh_vertadr[mid],m.mesh_vertnum[mid];raw=m.mesh_vert[a:a+n];world=raw@d.geom_xmat[g].reshape(3,3).T+d.geom_xpos[g];normal=c.frame[:3]*(1 if int(c.geom[0])==floor else-1);sourcepoint=c.pos+normal*c.dist/2
 row=rows.setdefault(name,dict(contacts=0,min_contact_distance_m=float('inf'),minimum_raw_vertex_z_m=float(world[:,2].min()),maximum_contact_source_vertex_error_m=0.))
 row['contacts']+=1;row['min_contact_distance_m']=min(row['min_contact_distance_m'],float(c.dist));row['maximum_contact_source_vertex_error_m']=max(row['maximum_contact_source_vertex_error_m'],float(np.linalg.norm(world-sourcepoint,axis=1).min()))
for row in rows.values():row['support_distance_error_m']=abs(row['min_contact_distance_m']-row['minimum_raw_vertex_z_m'])
r=dict(model_sha256=p['model_sha256'],nonfloor_mask_changes=nonfloor_changes,all_floor_contacts_source_mesh=True,no_sdf_floor_duplicates=True,floor_contacts=rows)
(out/'v5-floor-native-proof.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r,indent=2));assert not nonfloor_changes
assert all(row['support_distance_error_m']<1e-8 for row in rows.values())
assert all(row['maximum_contact_source_vertex_error_m']<1e-8 for row in rows.values())
