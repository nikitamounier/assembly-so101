"""Fit unchanged source case inside real SDF socket; no simulation constraints."""
from pathlib import Path
import json,time
import mujoco,numpy as np
from scipy.spatial.transform import Rotation
from scipy.optimize import differential_evolution, minimize
from assembly_policy.so101.wowrobo_contact_backend import load_backend
ROOT=Path(__file__).resolve().parents[3];OUT=Path(__file__).resolve().parent;NEW=ROOT/'outputs/linear-insertion-updated-20261107';SOURCE=ROOT/'outputs/linear-insertion-20261107/source-snapshot'
load_backend(NEW/'grids/manifest.json');m=mujoco.MjModel.from_xml_path(str(NEW/'scene.xml'));d=mujoco.MjData(m)
m.geom_contype[:]=0;m.geom_conaffinity[:]=0;m.body_contype[:]=0;m.body_conaffinity[:]=0
active=['upper_arm_module_printed_0','servo_2_case','servo_2_horns'];receivers=['base_servo_printed_0','servo_1_case','servo_1_horns','base_side_cover_printed_0','shoulder_connector_printed_0','shoulder_connector_printed_1','pin']
for names,a,b in [(active,1,2),(receivers,2,1)]:
 for name in names:
  gid=m.geom(name+'__contact').id;bid=m.geom_bodyid[gid];m.geom_contype[gid]=a;m.geom_conaffinity[gid]=b;m.body_contype[bid]|=a;m.body_conaffinity[bid]|=b
mujoco.mj_setState(m,d,np.load(SOURCE/'stages/21_upper_far_side_train.npz')['states'][0],mujoco.mjtState.mjSTATE_FULLPHYSICS);mujoco.mj_forward(m,d);q0=d.qpos.copy();basep=d.body('base_servo').xpos.copy();baser=d.body('base_servo').xmat.reshape(3,3).copy();free=m.joint('upper_arm_module_free').qposadr[0];shaft=m.joint('target_servo_2_shaft').qposadr[0]
manifest=json.load(open(ROOT/'assets_local/wowrobo/components-v2/manifest.json'));axis=np.array(manifest['servo_axes']['2']['axis']);pivot=np.array(manifest['servo_axes']['2']['position_m'])

def pose(params,depth):
 dx,dy,q=params;r=Rotation.from_rotvec(axis*q).as_matrix();rootr=baser@r.T;rootp=basep+[dx,dy,depth]-rootr@(pivot-r@pivot)
 d.qpos[:]=q0;d.qpos[free:free+3]=rootp;quat=np.empty(4);mujoco.mju_mat2Quat(quat,rootr.ravel());d.qpos[free+3:free+7]=quat;d.qpos[shaft]=q;mujoco.mj_forward(m,d)
 return max(0.,-float(np.min(d.contact.dist[:d.ncon]))) if d.ncon else 0.

count=0;best=[1e9,None]
def score(x):
 global count,best
 if np.linalg.norm(x[:2])>.003: return 10+1000*(np.linalg.norm(x[:2])-.003)
 worst=max(pose(x,z) for z in [0,.00625,.0125,.01875,.025])
 value=worst*1000+np.linalg.norm(x[:2])*.005+abs(x[2])*.0001
 count+=1
 if value<best[0]:best[:]=[value,list(x)]
 return value
candidates=[]
for q in [0,-.2,.2,-.4,-.6]:
 x=[0,0,q];candidates.append({'params':x,'max_overlap_m':max(pose(x,z) for z in [0,.00625,.0125,.01875,.025])})
print('nominal',json.dumps(candidates),flush=True)
result=differential_evolution(score,[(-.003,.003),(-.003,.003),(-.6,.2)],seed=8,popsize=8,maxiter=35,tol=.0005,polish=True,workers=1)
x=result.x;rows=[]
for z in np.linspace(.025,0,101):
 pen=pose(x,z);worst=d.contact[int(np.argmin(d.contact.dist[:d.ncon]))] if d.ncon else None
 rows.append({'remaining_depth_m':float(z),'max_overlap_m':pen,'pair':[m.geom(worst.geom1).name,m.geom(worst.geom2).name] if worst is not None else None})
record={'params_world_dx_dy_m_and_fixed_hinge_rad':x.tolist(),'max_sweep_overlap_m':max(r['max_overlap_m'] for r in rows),'shaft_angle_rad':float(x[2]),'case_offset_world_m':[float(x[0]),float(x[1]),0],'case_rotation_world':baser.tolist(),'axis_world':[0,0,-1],'nominal_hinge_candidates':candidates,'evaluations':count,'optimizer_success':bool(result.success),'optimizer_message':str(result.message),'sweep':rows,'qualification':'Finite native SDF contact diagnostic, not continuous collision certificate or physical insertion. Original case/receiver meshes retained. No dynamic constraint.'}
(OUT/'pocket-search.json').write_text(json.dumps(record,indent=2)+'\n');print(json.dumps({k:v for k,v in record.items() if k!='sweep'},indent=2))
