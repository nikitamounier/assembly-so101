import json,copy
from pathlib import Path
import numpy as np,mujoco
out=Path(__file__).resolve().parent
rows=json.loads((out/'precontact-family-refined.json').read_text());row=next(r for r in rows if r['shift_m']==.00013 and r['jaw_delta_rad']==-.0039)
best=json.loads((out/'best.json').read_text());best.update(row);best['qualification']='Native geometric precontact calibration only; loaded retention unqualified pending physical probe';best['precontact_note']='Local five-DOF solve translates hand 130 micrometres toward fixed cover side while preserving measured native orientation; jaw closes additional 0.0039 radians. No target/right changes. Source-mesh sampled overlap up to 21 micrometres intentionally permitted for gentle precontact.'
m=mujoco.MjModel.from_xml_path('assets_local/wowrobo/photo-resets-repaired-v1/scene.xml');d=mujoco.MjData(m)
with np.load(out/'corrected-18_upper_floor_grasp.npz') as z:d.qpos[:]=z['qpos'];d.ctrl[:]=z['controls']
ids=[int(m.joint('left_'+n).qposadr[0]) for n in ['shoulder_pan','shoulder_lift','elbow_flex','wrist_flex','wrist_roll','gripper']];d.qpos[ids]=row['q_left'];mujoco.mj_forward(m,d)
s=m.site('left_gripperframe').id;R=d.site_xmat[s].reshape(3,3);b=m.body('base_side_cover').id;BR=d.xmat[b].reshape(3,3);local=[((np.array(p['target_point_world_m'])-d.xpos[b])@BR).tolist() for p in row['pads']];center=np.mean([p['target_point_world_m'] for p in row['pads']],axis=0)
spec=copy.deepcopy(best['support_grasp']['grasp_calibration']);spec['opening_radians']=row['q_left'][-1];spec['center_offset_site']=((center-d.site_xpos[s])@R).tolist();spec['qualification']='Measured native bilateral precontact centroid registration, not zeroed IK residual; loaded retention unqualified'
support=best['support_grasp'];support['grasp_calibration']=spec;support['contact_points_piece_local_m']=local;support['center_piece_local_m']=np.mean(local,axis=0).tolist();support['grasp_span_piece_local']=((np.array(row['pads'][1]['target_point_world_m'])-np.array(row['pads'][0]['target_point_world_m']))@BR).tolist();support['grasp_span_piece_local']=(np.array(support['grasp_span_piece_local'])/np.linalg.norm(support['grasp_span_piece_local'])).tolist();support['grasp_approach_piece_local']=(R[:,0]@BR).tolist();support['qualification']=best['qualification'];best['grasp_calibration']=spec;best['spec']=spec
(out/'best-precontact.json').write_text(json.dumps(best,indent=2))
s=(out/'verify_best.py').read_text().replace("best-pending.json","best-precontact.json").replace("corrected-{stage}.npz","precontact-{stage}.npz").replace("verification.json","verification-precontact.json")
(out/'verify_precontact.py').write_text(s)
print(json.dumps(row,indent=2))
