from pathlib import Path
import json,mujoco,numpy as np
from assembly_policy.so101.wowrobo_boundaries import NativeBoundaryAudit
out=Path('outputs/insertion-fix-20261107/final');a=json.loads(Path('outputs/linear-insertion-aligned-20261107/final/alignment.json').read_text());m=mujoco.MjModel.from_xml_path('outputs/linear-insertion-20261107/source-snapshot/scene.xml');d=mujoco.MjData(m);d.qpos[:]=a['start_qpos'];mujoco.mj_forward(m,d);RR=d.body('upper_arm_module').xmat.reshape(3,3);px=np.array(a['feature_frame']['receiver_local_pocket_wall_normal']);pz=np.array(a['feature_frame']['receiver_local_pocket_up_normal']);py=np.cross(pz,px);q=np.array(a['start_qpos']);qa=int(m.joint('forearm_module_free').qposadr[0]);axis=np.array(a['axis_world']);au=NativeBoundaryAudit(m,max_samples=1024);gc=m.geom('servo_3_case').id;gu=m.geom('upper_arm_module_printed_0').id;rows=[]
for dx in [-.0001,0,.0001]:
 for dy in [-.0004,-.0002,0,.0002,.0004]:
  depths=[]
  for mm in [8,13,20,25]:
   d.qpos[:]=q;d.qpos[qa:qa+3]+=mm*.001*axis+RR@(dx*px+dy*py);mujoco.mj_forward(m,d);bounds=au._bounds(d,[gc,gu]);pairs=au._pairs(d,[(gc,gu)],bounds);depths.append(max([x['penetration_m'] for x in pairs],default=0))
  row={'extra_pocket_X_mm':dx*1000,'extra_pocket_Y_mm':dy*1000,'max_mm':max(depths)*1000,'depths_mm':np.array(depths).tolist()};rows.append(row)
rows.sort(key=lambda x:x['max_mm']);(out/'pose-clearance-scan.json').write_text(json.dumps(rows,indent=2)+'\n');print(rows[:8])
