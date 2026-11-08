import json,time,importlib.util,hashlib
from pathlib import Path
import numpy as np,mujoco,igl
from scipy.spatial import cKDTree
from assembly_policy.viewer.components import layout_collection
from assembly_policy.viewer.banks import ResetCatalog
from assembly_policy.so101.wowrobo_obs import PointTables,WOW_TASKS
folder=Path('robotics-assembly-3.0/assets_local/wowrobo/randomization-bank-repaired-cover-support-v2');out=Path('robotics-assembly-3.0/outputs/point-cloud-refinement-20261107')
m,s,b=layout_collection(folder);start=time.monotonic();t=PointTables(m);elapsed=time.monotonic()-start
config=json.loads(Path('assembly-so101/configs/wowrobo_point_features.json').read_text());d=mujoco.MjData(m)
result={'model_signature':s,'observation_signature':t.signature,'build_seconds':elapsed,'budgets':t.receiver_budget,'coverage':t.coverage,'body_counts':t.mating_counts,'tasks':{}}
for k,task in enumerate(WOW_TASKS):
 d.qpos[:]=config['tasks'][task]['reference_qpos'];mujoco.mj_kinematics(m,d);maxerror=0
 world=np.array([d.body(t.names[i]).xpos+d.body(t.names[i]).xmat.reshape(3,3)@p for p,i in zip(t.points[k].numpy(),t.which[k].numpy())])
 for gid in set(t.provenance[k]):
  mask=np.asarray(t.provenance[k])==gid;points=(world[mask]-d.geom_xpos[gid])@d.geom_xmat[gid].reshape(3,3);mid=m.geom_dataid[gid];va,vn=m.mesh_vertadr[mid],m.mesh_vertnum[mid];fa,fn=m.mesh_faceadr[mid],m.mesh_facenum[mid]
  dist=np.sqrt(igl.point_mesh_squared_distance(np.ascontiguousarray(points),np.asarray(m.mesh_vert[va:va+vn],float),np.asarray(m.mesh_face[fa:fa+fn],np.int64))[0]);maxerror=max(maxerror,float(dist.max()))
 assert maxerror<1e-6
 assert len(np.unique(np.c_[t.which[k].numpy(),t.points[k].numpy()],axis=0))==384
 result['tasks'][task]={'max_source_surface_error_m':maxerror,'ordered_surface_samples':384,'unique_surface_samples':384}
 if task=='assembly_pin':
  mouth=np.array(t.coverage[task]['mouth_world_m']);nf=t.receiver_budget[task]['feature'];delta=world[128:128+16]-mouth;axis=d.xaxis[m.joint('target_servo_1_shaft').id];u=np.array([1.,0,0]);u-=u@axis*axis;u/=np.linalg.norm(u);v=np.cross(axis,u);theta=np.arctan2(delta@v,delta@u)
  result['tasks'][task].update(rim_quadrants_covered=len(set(np.floor((theta+np.pi)/(np.pi/2)).astype(int)%4)),receiver_points_within_2mm_of_mouth=int((np.linalg.norm(world[128:]-mouth,axis=1)<.002).sum()),nearest_receiver_point_to_mouth_mm=float(np.linalg.norm(world[128:]-mouth,axis=1).min()*1000))
  assert result['tasks'][task]['rim_quadrants_covered']==4
spec=mujoco.mjtState.mjSTATE_FULLPHYSICS;c=ResetCatalog([b],s,mujoco.mj_stateSize(m,spec),state_spec=spec);before=t.points.clone();count=0;setup=0
import torch
for e in c.entries:
 task=(e.get('stage') or {}).get('task');_,state=c.state(e['id']);mujoco.mj_setState(m,d,state,spec);saved=d.qpos.copy();mujoco.mj_kinematics(m,d)
 if task not in WOW_TASKS:setup+=1;continue
 grips=[d.body(n) for n in __import__('assembly_policy.so101.wowrobo_obs',fromlist=['GRIPS']).GRIPS]
 inputs=dict(task=torch.tensor([WOW_TASKS.index(task)]),tcp_p=torch.tensor(np.asarray([[x.xpos for x in grips]],np.float32)),tcp_R=torch.tensor(np.asarray([[x.xmat.reshape(3,3) for x in grips]],np.float32)),part_p=torch.tensor(np.asarray([[d.body(n).xpos for n in t.names]],np.float32)),part_q=torch.tensor(np.asarray([[d.body(n).xquat for n in t.names]],np.float32)),finger=torch.zeros((1,2)))
 a=t(**inputs);assert a.shape==(1,1344) and torch.isfinite(a).all();assert torch.equal(a,t(**inputs));np.testing.assert_array_equal(saved,d.qpos);count+=1
assert torch.equal(before,t.points)
result['state_checks']={'active':count,'setup':setup,'finite_exact_repeat':True,'no_state_or_table_mutation':True}
(out/'final-audit.json').write_text(json.dumps(result,indent=2)+'\n');np.savez_compressed(out/'selected-point-tables.npz',points=t.points.numpy(),which=t.which.numpy(),grip=t.grip.numpy(),provenance=np.asarray(t.provenance));print(json.dumps(result,indent=2),flush=True)
