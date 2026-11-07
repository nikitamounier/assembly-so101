"""Finite untouched-source guard along the actually reached adapted path."""
from pathlib import Path
import json,mujoco,numpy as np
from assembly_policy.so101.wowrobo_boundaries import world_mesh,penetration
root=Path(__file__).resolve().parents[3];out=Path(__file__).resolve().parent;alignment=json.load(open(root/'outputs/linear-insertion-aligned-20261107/upper/alignment.json'));axis=np.array(alignment['axis_world']);cases=[]
for force in [.5,5,20]:
 p=out/f'legacy_mapped_{force:g}N.json'
 if p.exists():cases.append(json.load(open(p)))
if len(cases)!=3:raise RuntimeError('Wait for all3adapteddrivecases before moving-pathaudit')
lo=min(c['initial_axial_mm'] for c in cases)/1000;hi=max(c['all_substep_maximum_axial_mm'] for c in cases)/1000
m=mujoco.MjModel.from_xml_path(str(root/'outputs/linear-insertion-20261107/source-snapshot/scene.xml'));d=mujoco.MjData(m);start=np.array(alignment['start_qpos']);free=m.joint('upper_arm_module_free').qposadr[0];rows=[]
for travel in np.linspace(lo,hi,17):
 d.qpos[:]=start;d.qpos[free:free+3]+=axis*travel;mujoco.mj_forward(m,d)
 for receiver in ['shoulder_connector_printed_0','shoulder_connector_printed_1']:
  a=world_mesh(m,d,m.geom(receiver).id);b=world_mesh(m,d,m.geom('servo_2_case').id);sa=np.vstack([a.vertices,a.triangles_center]);sb=np.vstack([b.vertices,b.triangles_center]);pen=max(penetration(a,sb,2048),penetration(b,sa,2048));rows.append({'axial_mm':float(travel*1000),'pair':[receiver,'servo_2_case'],'reciprocal_sampled_source_penetration_m':pen})
record={'actual_adapted_axial_range_mm':[lo*1000,hi*1000],'sampling':'17actualprefixposes,2048AABB-filtered originalsourcevertices+ALLcentroids candidates per direction;finite sourceguard,notcontinuouscertificate','maximum_sampled_source_penetration_m':max(x['reciprocal_sampled_source_penetration_m'] for x in rows),'under_0p5mm_finite_source_guard':max(x['reciprocal_sampled_source_penetration_m'] for x in rows)<=.0005,'rows':rows,'geometry_changed':False};(out/'moving-path-source-audit.json').write_text(json.dumps(record,indent=2)+'\n');print(json.dumps({k:v for k,v in record.items() if k!='rows'},indent=2))
