from pathlib import Path
import json,mujoco,numpy as np
from assembly_policy.so101.wowrobo_boundaries import world_mesh,penetration
p=Path(__file__).resolve().parent;root=p.parents[2]
r=json.load(open(p/'alignment.json'));m=mujoco.MjModel.from_xml_path(str(root/'outputs/linear-insertion-20261107/source-snapshot/scene.xml'));d=mujoco.MjData(m)
rows=[]
for label,key in [('start','start_qpos'),('seated_reference','reference_qpos')]:
 d.qpos[:]=r[key];mujoco.mj_forward(m,d)
 for a,b in [('base_side_cover_printed_0','upper_arm_module_printed_0'),('shoulder_connector_printed_1','upper_arm_module_printed_0'),('shoulder_connector_printed_1','servo_2_case')]:
  ma=world_mesh(m,d,m.geom(a).id);mb=world_mesh(m,d,m.geom(b).id)
  sa=np.vstack([ma.vertices,ma.triangles_center]);sb=np.vstack([mb.vertices,mb.triangles_center])
  rows.append({'pose':label,'pair':[a,b],'reciprocal_sampled_triangle_penetration_m':max(penetration(ma,sb,2048),penetration(mb,sa,2048))})
(p/'triangle-fit.json').write_text(json.dumps({'sampling':'2048AABB-filtered source vertices+triangle centers in each direction; finite diagnostic, source nonwatertight sign ambiguous','rows':rows},indent=2)+'\n');print(json.dumps(rows,indent=2))
