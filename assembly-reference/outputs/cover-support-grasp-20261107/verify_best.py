import json
from pathlib import Path
import mujoco,numpy as np
from assembly_policy.so101.wowrobo_boundaries import NativeBoundaryAudit
out=Path(__file__).resolve().parent;bank=Path('assets_local/wowrobo/photo-resets-repaired-v1')
row=json.loads((out/'best-pending.json').read_text());m=mujoco.MjModel.from_xml_path(str(bank/'scene.xml'));A=NativeBoundaryAudit(m)
ids=[int(m.joint('left_'+n).qposadr[0]) for n in ['shoulder_pan','shoulder_lift','elbow_flex','wrist_flex','wrist_roll','gripper']]
rows=[]
for stage in ['17_left_support','18_upper_floor_grasp','20_upper_transport','21_upper_far_side','22_upper_half','23_upper_near','24_upper_seated','25_right_support']:
 d=mujoco.MjData(m);folder='completed' if stage.startswith('24') else 'stages'
 with np.load(bank/folder/f'{stage}_train.npz') as z:
  mujoco.mj_setState(m,d,z['states'][0],mujoco.mjtState.mjSTATE_FULLPHYSICS);d.ctrl[:]=z['controls'][0]
 before=d.qpos.copy();oldctrl=d.ctrl.copy();d.qpos[ids]=row['q_left']
 for a in range(m.nu):
  adr=int(m.jnt_qposadr[m.actuator_trnid[a,0]])
  if adr in ids:d.ctrl[a]=d.qpos[adr]
 mujoco.mj_forward(m,d)
 assert np.array_equal(np.delete(d.qpos,ids),np.delete(before,ids))
 au=A.audit(d);left=[p for p in au['helper_target'] if p['a']['helper_side']=='left'];hands=au['hand_hand'];table=[r for r in au['moving_helper_table'] if r['helper_side']=='left']
 worst=max(left,key=lambda p:p['penetration_m'],default=None)
 result={'stage':stage,'left_max_penetration_m':max((p['penetration_m'] for p in left),default=0),'worst_left':worst,'left_rejected_pairs':[p for p in left if p['rejected']],'hand_hand_max_penetration_m':max((p['penetration_m'] for p in hands),default=0),'left_min_table_m':min((p['clearance_m'] for p in table),default=0),'all_native_audit':au}
 rows.append(result)
 np.savez_compressed(out/f'corrected-{stage}.npz',qpos=d.qpos,controls=d.ctrl)
 (out/'verification.json').write_text(json.dumps(rows,indent=2))
 print(stage,result['left_max_penetration_m'],result['hand_hand_max_penetration_m'],result['left_min_table_m'],flush=True)
