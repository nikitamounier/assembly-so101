"""Native-source boundary and source-feature calibration for captured holds."""
import copy,json
from pathlib import Path
import mujoco,numpy as np
from assembly_policy.so101.wowrobo_boundaries import NativeBoundaryAudit
ROOT=Path(__file__).resolve().parents[2];OUT=Path(__file__).parent
m=mujoco.MjModel.from_xml_path(str(ROOT/'assets_local/wowrobo/photo-resets-repaired-cover-support-v2/scene.xml'));audit=NativeBoundaryAudit(m)
receipts={r['id']:r for r in json.loads((ROOT/'assets_local/wowrobo/photo-resets-repaired-cover-support-v2/stage-receipts.json').read_text())}
paths={'07_connector_pickup':OUT/'preload','08_connector_aligned':OUT/'connector-closer','12_pin_pickup':OUT/'pin-deeper','13_pin_aligned':OUT/'pin-retarget'}
results={};spec=mujoco.mjtState.mjSTATE_FULLPHYSICS;n=mujoco.mj_stateSize(m,spec)
for stage,folder in paths.items():
 r=json.loads((folder/(stage+'.json')).read_text());z=np.load(folder/(stage+'.npz'));side=r['side'];part=r['part'];site=m.site(side+'_gripperframe').id;body=m.body(part).id
 old=mujoco.MjData(m)
 with np.load(ROOT/f'assets_local/wowrobo/photo-resets-repaired-cover-support-v2/stages/{stage}_train.npz')as original:mujoco.mj_setState(m,old,original['states'][0],spec)
 mujoco.mj_kinematics(m,old);requested=np.array(receipts[stage]['arms'][side]['requested_grip_center']);feature_local=old.xmat[body].reshape(3,3).T@(requested-old.xpos[body])
 d=mujoco.MjData(m);mujoco.mj_setState(m,d,z['hold_state'][:n],spec);mujoco.mj_kinematics(m,d)
 center=d.xpos[body]+d.xmat[body].reshape(3,3)@feature_local;frame=d.site_xmat[site].reshape(3,3);calibration=copy.deepcopy(receipts[stage]['arms'][side]['grasp_calibration']);calibration['center_offset_site']=(frame.T@(center-d.site_xpos[site])).tolist();calibration['opening_radians']=float(d.joint(side+'_gripper').qpos[0]);calibration['qualification']='Actual native loaded hold feature registration; local grasp witness only, whole-scene admission separate'
 for key in ['pad_inner_faces_site_z_m','pad_gap_m']:calibration.pop(key,None)
 target_span_local=d.xmat[body].reshape(3,3).T@frame[:,2];approach_local=d.xmat[body].reshape(3,3).T@frame[:,0]
 qcandidate=z['qpos_candidate'];cd=mujoco.MjData(m);cd.qpos[:]=qcandidate;mujoco.mj_kinematics(m,cd)
 initial_axis=old.xmat[body].reshape(3,3)[:,2];actual_axis=d.xmat[body].reshape(3,3)[:,2];tilt=float(np.rad2deg(np.arccos(np.clip(np.dot(initial_axis,actual_axis),-1,1))))
 result={'stage':stage,'side':side,'part':part,'recipe_archive':str(folder/(stage+'.npz')),'local_hold_pass':r['local_hold_pass'],'calibration':calibration,'grasp_feature_piece_local_m':feature_local.tolist(),'grasp_span_piece_local':target_span_local.tolist(),'grasp_approach_piece_local':approach_local.tolist(),'settled_upper_axis_tilt_from_original_degrees':tilt,'candidate_boundary':audit.audit(cd),'hold_boundary':audit.audit(d),'qpos_candidate':qcandidate.tolist(),'ctrl_candidate':z['ctrl'].tolist(),'hold_min_loads_N':r['hold_min_loads_N'],'hold_max_motion_m':r['hold_max_motion_m'],'hold_max_rotation_deg':r['hold_max_rotation_deg'],'training_eligible':False}
 results[stage]=result;print(stage,'localpass',r['local_hold_pass'],'axis tilt',tilt,'candidate',result['candidate_boundary']['candidate_rejected'],'hold',result['hold_boundary']['candidate_rejected'],flush=True)
(OUT/'recipes.json').write_text(json.dumps(results,indent=2)+'\n')
