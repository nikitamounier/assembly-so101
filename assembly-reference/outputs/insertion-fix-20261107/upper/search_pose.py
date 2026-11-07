"""Source-preserving bounded pose-clearance correction, no contact-law change."""
from pathlib import Path
import json,time
import mujoco,numpy as np
from scipy.optimize import differential_evolution
from scipy.spatial.transform import Rotation
from assembly_policy.so101.wowrobo_contact_backend import load_backend
ROOT=Path(__file__).resolve().parents[3];OUT=Path(__file__).resolve().parent;BASE=ROOT/'outputs/linear-insertion-aligned-20261107/upper';NEW=ROOT/'outputs/linear-insertion-updated-20261107';a=json.load(open(BASE/'alignment.json'));load_backend(NEW/'grids/manifest.json');m=mujoco.MjModel.from_xml_path(str(NEW/'scene.xml'));d=mujoco.MjData(m)
m.geom_contype[:]=0;m.geom_conaffinity[:]=0;m.body_contype[:]=0;m.body_conaffinity[:]=0;m.geom_margin[:]=.0001
active=['upper_arm_module_printed_0','servo_2_case','servo_2_horns'];receivers=['base_servo_printed_0','servo_1_case','servo_1_horns','base_side_cover_printed_0','shoulder_connector_printed_0','shoulder_connector_printed_1','pin']
for names,ctype,aff in [(active,1,2),(receivers,2,1)]:
 for name in names:
  g=m.geom(name+'__contact').id;b=m.geom_bodyid[g];m.geom_contype[g]=ctype;m.geom_conaffinity[g]=aff;m.body_contype[b]|=ctype;m.body_conaffinity[b]|=aff
q0=np.array(a['reference_qpos']);d.qpos[:]=q0;mujoco.mj_forward(m,d);case=m.body('target_servo_2_pivot').id;free=m.joint('upper_arm_module_free').qposadr[0];shaft=m.joint('target_servo_2_shaft').qposadr[0];caseR=d.xmat[case].reshape(3,3).copy();gref=d.geom('servo_2_case').xpos.copy();gc=caseR.T@(gref-d.xpos[case]);socket=np.array(a['feature_frame']['socket_frame_world']);ex,ey,ez=socket.T
manifest=json.load(open(ROOT/'assets_local/wowrobo/components-v2/manifest.json'));axis=np.array(manifest['servo_axes']['2']['axis']);pivot=np.array(manifest['servo_axes']['2']['position_m']);count=0;best=[1e9,None]

def pose(x,depth):
 dx,dy,rx,ry,rz,h=x;delta=socket@Rotation.from_rotvec([rx,ry,rz]).as_matrix()@socket.T;cr=delta@caseR;cp=gref+dx*ex+dy*ey+depth*ez-cr@gc;r=Rotation.from_rotvec(axis*h).as_matrix();rr=cr@r.T;rp=cp-rr@(pivot-r@pivot);d.qpos[:]=q0;d.qpos[free:free+3]=rp;quat=np.empty(4);mujoco.mju_mat2Quat(quat,rr.ravel());d.qpos[free+3:free+7]=quat;d.qpos[shaft]=h;mujoco.mj_forward(m,d);dist=float(np.min(d.contact.dist[:d.ncon])) if d.ncon else .0001
 return dist

def score(x):
 global count,best
 if np.linalg.norm(x[:2])>.003:return 10+1e3*(np.linalg.norm(x[:2])-.003)
 worst=max(max(0,.00005-pose(x,z)) for z in [0,.005,.01,.015,.02,.025]);v=worst*1000+np.linalg.norm(x[:2])*.001+np.linalg.norm(x[2:5])*.00001;count+=1
 if v<best[0]:best[:]=[v,list(x)]
 return v
bounds=[(-.003,.003),(-.003,.003)]+[(-np.deg2rad(3),np.deg2rad(3))]*3+[(-.18,.18)]
t=time.perf_counter();res=differential_evolution(score,bounds,seed=32,popsize=10,maxiter=60,tol=.0001,polish=True,workers=1);x=res.x;rows=[]
for z in np.linspace(.025,0,101):
 dist=pose(x,z);rows.append({'depth_remaining_m':float(z),'minimum_native_signed_distance_m':dist})
pose(x,.025);start=d.qpos.copy();pose(x,0);end=d.qpos.copy();result={'params_dx_dy_rx_ry_rz_hinge':x.tolist(),'score':float(res.fun),'max_native_penetration_m':max(0,-min(r['minimum_native_signed_distance_m'] for r in rows)),'min_native_clearance_m':min(r['minimum_native_signed_distance_m'] for r in rows),'evaluations':count,'wall_s':time.perf_counter()-t,'optimizer_success':bool(res.success),'optimizer_message':str(res.message),'axis_world':a['axis_world'],'start_qpos':start.tolist(),'reference_qpos':end.tolist(),'authored_axial_travel_m':.025,'sweep':rows,'scope':'OnlyboundedcasegeomcenterXY/rotation/fixedhinge posevaried;originalCAD/grids/firmcontactlawuntouched. Native0.4mmfieldobjective requiresindependentsourceaudit.'};(OUT/'pose-search.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k not in ['start_qpos','reference_qpos','sweep']},indent=2))
