"""Find first native contact boundaries for each actual internal hinge pair."""
import importlib.util,json,sys,time,xml.etree.ElementTree as ET
from pathlib import Path
import mujoco,numpy as np
root,out=map(Path,sys.argv[1:]);out.mkdir(parents=True,exist_ok=True)
canon=root/'robotics-assembly-3.0'
spec=importlib.util.spec_from_file_location('backend',canon/'assembly_policy/so101/wowrobo_contact_backend.py');b=importlib.util.module_from_spec(spec);spec.loader.exec_module(b)
meta=json.loads((canon/'assets_local/wowrobo/training-accepted-v6/manifest.json').read_text());profile=json.loads(Path(meta['profile_path']).read_text());b.load_backend(profile['grid_manifest'],out/'native');b.set_surface_mode(2)
r=ET.parse(profile['model_path']).getroot()
for g in r.iter('geom'):g.set('contype','0');g.set('conaffinity','0')
contact=r.find('contact')
for p in list(contact):
 if p.tag!='pair' or not p.get('name','').startswith('passive_motor_stop_'):contact.remove(p)
m=mujoco.MjModel.from_xml_string(ET.tostring(r,encoding='unicode'));d=mujoco.MjData(m)
result={};t=time.perf_counter()
for joint in ['target_servo_2_shaft','target_servo_3_shaft']:
 adr=int(m.joint(joint).qposadr[0]);pivot=int(m.jnt_bodyid[m.joint(joint).id]);parent=int(m.body_parentid[pivot]);pairs=[{int(m.pair_geom1[i]),int(m.pair_geom2[i])} for i in range(m.npair) if {int(m.geom_bodyid[m.pair_geom1[i]]),int(m.geom_bodyid[m.pair_geom2[i]])}=={pivot,parent}]
 def probe(q):
  d.qpos[:]=m.qpos0;d.qpos[adr]=q;mujoco.mj_forward(m,d)
  distances=[float(c.dist) for c in d.contact if set(map(int,c.geom)) in pairs]
  return min(distances,default=0.) < -1e-9
 assert not probe(0.),joint+' nominal already contacting'
 rows=[];bounds=[]
 for sign in [-1,1]:
  previous=0.;bracket=None
  for q in sign*np.linspace(0,2*np.pi,721)[1:]:
   hit=probe(q);rows.append([float(q),hit])
   if hit:bracket=[previous,float(q)];break
   previous=float(q)
  if bracket is None:bounds.append(None);continue
  clear,hit=bracket
  for _ in range(25):
   mid=(clear+hit)/2
   if probe(mid):hit=mid
   else:clear=mid
  bounds.append((clear+hit)/2)
 result[joint]=dict(existing_range_rad=m.joint(joint).range.tolist(),effective_range_rad=[max(m.joint(joint).range[0],bounds[0]) if bounds[0] is not None else float(m.joint(joint).range[0]),min(m.joint(joint).range[1],bounds[1]) if bounds[1] is not None else float(m.joint(joint).range[1])],range_rad=bounds,range_deg=[None if v is None else float(np.degrees(v)) for v in bounds],contact_threshold_m=1e-9,angular_bracket_width_rad=None if bracket is None else abs(clear-hit),coarse=rows,pair_geoms=[[m.geom(g).name for g in sorted(p)] for p in pairs])
 print(joint,result[joint]['range_deg'],flush=True)
(out/'angles.json').write_text(json.dumps(dict(joints=result,elapsed_s=time.perf_counter()-t,scope='Native original grid/SDF+exhaustive original surface contacts; first contacting boundaries around zero, not exact analytic CAD'),indent=2)+'\n')
