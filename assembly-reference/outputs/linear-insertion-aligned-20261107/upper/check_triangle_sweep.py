from pathlib import Path
import json,mujoco,numpy as np
from assembly_policy.so101.wowrobo_boundaries import world_mesh,penetration
p=Path(__file__).resolve().parent;root=p.parents[2];r=json.load(open(p/'alignment.json'));m=mujoco.MjModel.from_xml_path(str(root/'outputs/linear-insertion-20261107/source-snapshot/scene.xml'));d=mujoco.MjData(m);free=m.joint('upper_arm_module_free').qposadr[0];qstart=np.array(r['start_qpos']);qend=np.array(r['reference_qpos']);rows=[]
for t in np.linspace(0,1,17):
 d.qpos[:]=qstart;d.qpos[free:free+3]=(1-t)*qstart[free:free+3]+t*qend[free:free+3];mujoco.mj_forward(m,d)
 a='shoulder_connector_printed_1';b='servo_2_case';ma=world_mesh(m,d,m.geom(a).id);mb=world_mesh(m,d,m.geom(b).id);sa=np.vstack([ma.vertices,ma.triangles_center]);sb=np.vstack([mb.vertices,mb.triangles_center]);pen=max(penetration(ma,sb,2048),penetration(mb,sa,2048));rows.append({'fraction':float(t),'remaining_depth_m':.025*(1-t),'reciprocal_sampled_triangle_penetration_m':pen})
record={'sampling':'17whole-shape axial poses;2048AABB-filtered source vertices+trianglecenters per direction;finite verification only','pair':['shoulder_connector_printed_1','servo_2_case'],'maximum_sampled_penetration_m':max(x['reciprocal_sampled_triangle_penetration_m'] for x in rows),'rows':rows};(p/'triangle-sweep.json').write_text(json.dumps(record,indent=2)+'\n');print(json.dumps(record,indent=2))
