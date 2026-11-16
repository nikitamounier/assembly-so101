import json
from pathlib import Path
import mujoco,numpy as np
from scipy.optimize import least_squares
ROOT=Path(__file__).resolve().parents[2];OUT=Path(__file__).parent;m=mujoco.MjModel.from_xml_path(str(OUT/'centered-review-candidate.xml'));d=mujoco.MjData(m);r=json.loads((OUT/'coherent-recipes.json').read_text());row=r['25_right_support'];d.qpos[:]=row['qpos'];side='right';site=m.site(side+'_gripperframe').id;names=['shoulder_pan','shoulder_lift','elbow_flex','wrist_flex','wrist_roll'];jids=[m.joint(side+'_'+n).id for n in names];adr=m.jnt_qposadr[jids];bounds=m.jnt_range[jids];goal=row['hand_targets']['right'];pg=np.array(goal['target_pos']);rg=np.array(goal['target_rot']);oldq=d.qpos[adr].copy();rng=np.random.default_rng(4)
def fun(q):
 d.qpos[adr]=q;mujoco.mj_kinematics(m,d);rot=d.site_xmat[site].reshape(3,3);return np.r_[100*(d.site_xpos[site]-pg),10*(rot[:,2]-rg[:,2]),.0001*(rot[:,0]-rg[:,0])]
rows=[]
for start in [oldq, np.array([oldq[0],-.4,-1.2,1.3,oldq[-1]]),np.array([oldq[0],.5,-1.2,1.3,oldq[-1]])]+[rng.uniform(bounds[:,0]+.01,bounds[:,1]-.01)for _ in range(7)]:
 sol=least_squares(fun,np.clip(start,bounds[:,0]+1e-7,bounds[:,1]-1e-7),bounds=(bounds[:,0]+1e-7,bounds[:,1]-1e-7),max_nfev=300,ftol=1e-10,xtol=1e-10,gtol=1e-10);fun(sol.x);err=float(np.linalg.norm(d.site_xpos[site]-pg));span=float(np.linalg.norm(d.site_xmat[site].reshape(3,3)[:,2]-rg[:,2]));q=d.qpos.copy();rows.append({'error_m':err,'span_error':span,'qpos':q.tolist(),'q_right':sol.x.tolist()});print(err,span,sol.x.tolist(),flush=True)
best=min(rows,key=lambda x:x['error_m']+x['span_error']*.02);(OUT/'right-support-refit.json').write_text(json.dumps({'best':best,'rows':rows},indent=2)+'\n')
