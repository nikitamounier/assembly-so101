import json,numpy as np,mujoco
from pathlib import Path
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation
from assembly_policy.so101.wowrobo_contact_backend import load_backend,set_surface_mode
ROOT=Path(__file__).resolve().parents[2];OUT=Path(__file__).parent/'from22-yaw';OUT.mkdir(exist_ok=True)
p=json.loads((ROOT/'outputs/equilibrium-model-20261107/profile-centered-v6/profile.json').read_text());load_backend(p['grid_manifest']);set_surface_mode(1);m=mujoco.MjModel.from_xml_path(p['model_path']);z=np.load(ROOT/'outputs/equilibrium-reset-20261107/minimum-v6-source-fingers/22_upper_half.npz');t=mujoco.MjData(m);mujoco.mj_setState(m,t,z['states'][0],mujoco.mjtState.mjSTATE_INTEGRATION);mujoco.mj_kinematics(m,t);bid=m.body('target_servo_2_pivot').id;sid=m.site('right_gripperframe').id;T=t.xmat[bid].reshape(3,3);relp=T.T@(t.site_xpos[sid]-t.xpos[bid]);relr=T.T@t.site_xmat[sid].reshape(3,3);source=json.loads((ROOT/'outputs/equilibrium-engaged-repair-20261107/coherent-recipes.json').read_text())['21_upper_far_side'];d=mujoco.MjData(m);d.qpos[:]=source['qpos'];mujoco.mj_kinematics(m,d);q0=d.qpos.copy();R=d.xmat[bid].reshape(3,3);hp=d.xpos[bid]+R@relp;hr=R@relr;shaft=m.joint('target_servo_1_shaft');anchor=d.xanchor[shaft.id].copy();axis=d.xaxis[shaft.id].copy();roots={n:(d.xpos[m.body(n).id].copy(),d.xmat[m.body(n).id].reshape(3,3).copy())for n in ['shoulder_connector','assembly_pin','upper_arm_module']};ids=[m.joint('right_'+n).id for n in ['shoulder_pan','shoulder_lift','elbow_flex','wrist_flex','wrist_roll']];qids=m.jnt_qposadr[ids];jaw=m.joint('right_gripper').qposadr[0];lo=np.r_[m.jnt_range[ids,0]+1e-6,-1.];hi=np.r_[m.jnt_range[ids,1]-1e-6,1.]
def residual(x):
 rot=Rotation.from_rotvec(axis*x[5]).as_matrix();target=anchor+rot@(hp-anchor);targetr=rot@hr;d.qpos[:]=q0;d.qpos[qids]=x[:5];mujoco.mj_kinematics(m,d);rr=d.site_xmat[sid].reshape(3,3);return np.r_[1000*(d.site_xpos[sid]-target),20*Rotation.from_matrix(rr@targetr.T).as_rotvec()]
fits=[]
for yaw in [0,.3,-.3,.6,-.6]:
 fit=least_squares(residual,np.clip(np.r_[t.qpos[qids],yaw],lo,hi),bounds=(lo,hi),max_nfev=250,ftol=1e-11,xtol=1e-11,gtol=1e-11);fits.append(fit);print('fit',np.linalg.norm(fit.fun),fit.x.tolist(),flush=True)
f=min(fits,key=lambda f:np.linalg.norm(f.fun));residual(f.x);rot=Rotation.from_rotvec(axis*f.x[5]).as_matrix()
for n,(pos,rr)in roots.items():
 qa=m.joint(n+'_free').qposadr[0];d.qpos[qa:qa+3]=anchor+rot@(pos-anchor);mujoco.mju_mat2Quat(d.qpos[qa+3:qa+7],(rot@rr).ravel())
d.qpos[shaft.qposadr[0]]+=f.x[5];d.qpos[jaw]=t.qpos[jaw];d.ctrl[:]=source['ctrl'];mujoco.mj_forward(m,d)
for j in ids+[m.joint('right_gripper').id]:
 ac=mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_ACTUATOR,m.joint(j).name);d.ctrl[ac]=np.clip(d.qpos[m.jnt_qposadr[j]]+((t.ctrl[ac]-t.qpos[m.jnt_qposadr[j]])if j==m.joint('right_gripper').id else d.qfrc_bias[m.jnt_dofadr[j]]/m.actuator_gainprm[ac,0]),*m.actuator_ctrlrange[ac])
row={'stage':'21_upper_far_side','qpos':d.qpos.tolist(),'ctrl':d.ctrl.tolist(),'yaw_rad':f.x[5],'residual':f.fun.tolist(),'registered_model':True};(OUT/'candidate.json').write_text(json.dumps(row,indent=2));print('saved',row['yaw_rad'],row['residual'],flush=True)
