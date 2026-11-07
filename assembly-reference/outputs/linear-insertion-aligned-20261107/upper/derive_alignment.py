"""Geometric common-CAD case alignment; sweep is diagnostic, not dynamics."""
from pathlib import Path
import hashlib,json
import mujoco,numpy as np
from scipy.spatial.transform import Rotation
from assembly_policy.so101.wowrobo_contact_backend import load_backend

ROOT=Path(__file__).resolve().parents[3]
OUT=Path(__file__).resolve().parent
SOURCE=ROOT/'outputs/linear-insertion-20261107/source-snapshot'
NEW=ROOT/'outputs/linear-insertion-updated-20261107'
load_backend(NEW/'grids/manifest.json')
m=mujoco.MjModel.from_xml_path(str(NEW/'scene.xml'))
d=mujoco.MjData(m)
for gid in range(m.ngeom):
    m.geom_contype[gid]=0;m.geom_conaffinity[gid]=0
m.body_contype[:]=0;m.body_conaffinity[:]=0
active=['upper_arm_module_printed_0','servo_2_case','servo_2_horns']
receivers=['base_servo_printed_0','servo_1_case','servo_1_horns','base_side_cover_printed_0','shoulder_connector_printed_0','shoulder_connector_printed_1','pin']
for names,a,b in [(active,1,2),(receivers,2,1)]:
    for name in names:
        gid=m.geom(name+'__contact').id;bid=m.geom_bodyid[gid]
        m.geom_contype[gid]=a;m.geom_conaffinity[gid]=b;m.body_contype[bid]|=a;m.body_conaffinity[bid]|=b

def state(stage):
    mujoco.mj_setState(m,d,np.load(SOURCE/'stages'/f'{stage}_train.npz')['states'][0],mujoco.mjtState.mjSTATE_FULLPHYSICS)
    mujoco.mj_forward(m,d)
    return d.qpos.copy()

original_start=state('21_upper_far_side');original_end=state('24_upper_seated')
root=m.body('upper_arm_module').id;case=m.body('target_servo_2_pivot').id;base=m.body('base_servo').id
free=m.joint('upper_arm_module_free').qposadr[0];shaft=m.joint('target_servo_2_shaft').qposadr[0]
d.qpos[:]=original_start;mujoco.mj_forward(m,d)
base_p=d.xpos[base].copy();base_r=d.xmat[base].reshape(3,3).copy()


def aligned(qpos,depth,shaft_q):
    d.qpos[:]=qpos;d.qpos[shaft]=shaft_q;mujoco.mj_forward(m,d)
    old_root_p=d.xpos[root].copy();old_root_r=d.xmat[root].reshape(3,3).copy()
    case_p=d.xpos[case].copy();case_r=d.xmat[case].reshape(3,3).copy()
    delta=base_r@case_r.T
    new_root_r=delta@old_root_r
    new_root_p=base_p+[0,0,depth]+delta@(old_root_p-case_p)
    d.qpos[free:free+3]=new_root_p
    quat=np.empty(4);mujoco.mju_mat2Quat(quat,new_root_r.ravel());d.qpos[free+3:free+7]=quat
    mujoco.mj_forward(m,d)
    target_p=base_p+[0,0,depth]
    return d.qpos.copy(),float(np.linalg.norm(d.xpos[case]-target_p)),float(np.linalg.norm(Rotation.from_matrix(d.xmat[case].reshape(3,3)@base_r.T).as_rotvec()))

rows=[];starts=None;ends=None
for i in range(51):
    t=i/50;depth=.025*(1-t);shaft_q=(1-t)*original_start[shaft]+t*original_end[shaft]
    q,pe,re=aligned(original_start,depth,shaft_q)
    if i==0:starts=q.copy()
    if i==50:ends=q.copy()
    row={'fraction':t,'depth_remaining_m':depth,'case_position_error_m':pe,'case_orientation_error_rad':re,'contacts':int(d.ncon),'minimum_contact_distance_m':float(np.min(d.contact.dist[:d.ncon])) if d.ncon else None}
    if d.ncon:
        worst=d.contact[int(np.argmin(d.contact.dist[:d.ncon]))]
        row['deepest_pair']=[m.geom(worst.geom1).name,m.geom(worst.geom2).name]
    rows.append(row)

record={'step':'upper','source':'Original common-CAD body registration of servo2 case and shoulder connector; both constituent meshes share the positioned source-CAD coordinate system. Desired held case body pose equals base_servo frame plus axial stand-off. No CAD edit or constraint.','source_scene_sha256':hashlib.sha256((SOURCE/'scene.xml').read_bytes()).hexdigest(),'simulated_scene_sha256':hashlib.sha256((NEW/'scene.xml').read_bytes()).hexdigest(),'mating_body':'target_servo_2_pivot','mating_geom':'servo_2_case','receiver_body':'shoulder_connector','registration_frame_body':'base_servo','axis_world':[0,0,-1],'start_qpos':starts.tolist(),'reference_qpos':ends.tolist(),'start_stage':'21_upper_far_side','reference_stage':'24_upper_seated','authored_axial_travel_m':.025,'target_case_position_world_m':base_p.tolist(),'target_case_rotation_world':base_r.tolist(),'start_correction_qpos_l2':float(np.linalg.norm(starts[free:free+7]-original_start[free:free+7])),'end_correction_qpos_l2':float(np.linalg.norm(ends[free:free+7]-original_end[free:free+7])),'max_case_position_error_m':max(r['case_position_error_m'] for r in rows),'max_case_orientation_error_rad':max(r['case_orientation_error_rad'] for r in rows),'sweep_deepest_contact_m':min(r['minimum_contact_distance_m'] or 0 for r in rows),'sweep':rows,'qualification':'Geometric alignment and SDF diagnostic only. Sweep uses changing servo hinge and link root to maintain the case path; force-only integration is free to depart from it. No force/contact retention or insertion qualification.'}
(OUT/'alignment.json').write_text(json.dumps(record,indent=2)+'\n')
np.savez_compressed(OUT/'alignment-sweep.npz',start_qpos=starts,reference_qpos=ends,original_start_qpos=original_start,original_reference_qpos=original_end)
print(json.dumps({k:v for k,v in record.items() if k not in ['start_qpos','reference_qpos','sweep','target_case_rotation_world']},indent=2))
