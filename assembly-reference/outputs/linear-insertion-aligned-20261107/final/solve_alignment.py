from pathlib import Path
import json,numpy as np,mujoco
from scipy.spatial.transform import Rotation
from assembly_policy.so101.wowrobo_boundaries import NativeBoundaryAudit
ROOT=Path.cwd();src=ROOT/'outputs/linear-insertion-20261107/source-snapshot';out=ROOT/'outputs/linear-insertion-aligned-20261107/final'
m=mujoco.MjModel.from_xml_path(str(src/'scene.xml'));d=mujoco.MjData(m)
a=np.load(src/'stages/29_final_far_side_train.npz');mujoco.mj_setState(m,d,a['states'][0],mujoco.mjtState.mjSTATE_FULLPHYSICS);mujoco.mj_forward(m,d);qoriginal=d.qpos.copy()
receiver=m.body('upper_arm_module').id;case=m.body('target_servo_3_pivot').id;root=m.body('forearm_module').id
rr=d.xmat[receiver].reshape(3,3).copy();rp=d.xpos[receiver].copy();axis=-rr[:,2]
j=m.joint('forearm_module_free');qa=int(j.qposadr[0]);hinge=int(m.joint('target_servo_3_shaft').qposadr[0]);pivot=np.array(m.jnt_pos[m.joint('target_servo_3_shaft').id]);local_axis=np.array(m.jnt_axis[m.joint('target_servo_3_shaft').id]);audit=NativeBoundaryAudit(m,max_samples=512)
active=[g for g in audit.targets if audit.info[g]['piece']=='forearm_module'];other=[g for g in audit.targets if audit.info[g]['piece']!='forearm_module']
def state(h,stand=0):
 d.qpos[:]=qoriginal;rel=Rotation.from_rotvec(-local_axis*h).as_matrix();r=rr@rel;p=rp+rr@pivot-r@pivot-stand*axis
 xyzw=Rotation.from_matrix(r).as_quat();d.qpos[qa:qa+3]=p;d.qpos[qa+3:qa+7]=xyzw[[3,0,1,2]];d.qpos[hinge]=h;mujoco.mj_forward(m,d)
 return d.qpos.copy()
def check():
 bounds=audit._bounds(d,audit.info);pairs=audit._pairs(d,[(a,b) for a in active for b in other],bounds)
 return {'max_m':max([x['penetration_m'] for x in pairs],default=0),'pairs':[{'a':x['a']['geom_name'],'b':x['b']['geom_name'],'m':x['penetration_m']} for x in pairs if x['penetration_m']>.0002]}
results=[]
for h in [0,.02078384088578589,.23734175712701053,-.05,.05,-.10,.10]:
 state(h);r=check();row={'hinge':h,'seat':r,'case_rot_error':float(np.linalg.norm(d.xmat[case].reshape(3,3)-rr)),'case_origin_error':float(np.linalg.norm(d.xpos[case]-rp))};results.append(row);print(row,flush=True)
(out/'candidates.json').write_text(json.dumps(results,indent=2)+'\n')
translations=[]
for dx in [-.001,-.002,-.0025,-.003,-.0035]:
 state(0);d.qpos[qa:qa+3]+=rr@np.array([dx,0,0]);mujoco.mj_forward(m,d);res=check();row={'receiver_translation_m':[dx,0,0],'seat':res};translations.append(row);print(row,flush=True)
(out/'translation-candidates.json').write_text(json.dumps(translations,indent=2)+'\n')
