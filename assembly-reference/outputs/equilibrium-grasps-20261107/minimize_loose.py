"""Gravity potential minimum candidate, independently replayed with native dynamics."""
import json,time
from pathlib import Path
import mujoco,numpy as np
from scipy.optimize import differential_evolution,minimize
from scipy.spatial.transform import Rotation
from assembly_policy.so101.wowrobo_contact_backend import load_backend,set_surface_mode
from assembly_policy.so101.wowrobo_equilibrium import rotation_degrees
from assembly_policy.so101.wowrobo_physical_resets import physical_split_step
ROOT=Path(__file__).resolve().parents[2];BASE=Path(__file__).parent/'loose-forearm-v1';OUT=Path(__file__).parent/'loose-forearm-energy-v1';OUT.mkdir(exist_ok=True)
p=json.load(open(ROOT/'outputs/equilibrium-model-20261107/profile-bearing-v4/profile.json'));load_backend(p['grid_manifest']);set_surface_mode(1);m=mujoco.MjModel.from_xml_path(str(BASE/'scene.xml'));d=mujoco.MjData(m);z=np.load(BASE/'latest.npz');d.qpos[:]=z['qpos'];d.qvel[:]=z['qvel'];mujoco.mj_kinematics(m,d);q0=d.qpos.copy();r0=Rotation.from_quat(q0[[4,5,6,3]])
meshes=[]
for g in range(m.ngeom):
 if m.geom_type[g]==mujoco.mjtGeom.mjGEOM_MESH:
  mid=m.geom_dataid[g];i,n=m.mesh_vertadr[mid],m.mesh_vertnum[mid];meshes.append((g,m.mesh_vert[i:i+n].copy()))
assert meshes
mass=m.body_mass[1:];total=mass.sum()
def evaluate(x,store=False):
 d.qpos[:]=q0;rot=Rotation.from_rotvec([x[0],x[1],0])*r0;q=rot.as_quat();d.qpos[3:7]=q[[3,0,1,2]];d.qpos[2]=0.;d.qpos[7]=x[2];d.qpos[8]=0.;mujoco.mj_kinematics(m,d)
 low=min(float((v@d.geom_xmat[g].reshape(3,3)[2]+d.geom_xpos[g,2]).min())for g,v in meshes);height=-low
 com=float(sum(m.body_mass[i]*d.xipos[i,2]for i in range(1,m.nbody))/total)
 value=com+height
 if store:d.qpos[2]=height;mujoco.mj_kinematics(m,d)
 return value
b=[(-np.deg2rad(20),np.deg2rad(20)),(-np.deg2rad(20),np.deg2rad(20)),(0.,np.deg2rad(10.))]
start=time.monotonic();initialenergy=evaluate([0,0,q0[7]]);res=differential_evolution(evaluate,b,tol=1e-10,popsize=10,maxiter=100,seed=7,polish=True);evaluate(res.x,True);candidate=d.qpos.copy()
# Mathematical initialization only. The real dynamic settle/hold below is the certificate.
d.qvel[:]=0.;mujoco.mj_forward(m,d);records=[];names=['forearm_module','target_servo_3_pivot','target_servo_4_pivot'];initial={n:(d.xpos[m.body(n).id].copy(),d.xmat[m.body(n).id].reshape(3,3).copy())for n in names};maximum={n:0. for n in names};rotation={n:0. for n in names};maxvel=0.;fric=dict(finger_part=1.,part_table=1.,part_part=1.)
for i in range(round(2./m.opt.timestep)):
 physical_split_step(m,d,fric)
 for n in names:
  bid=m.body(n).id;maximum[n]=max(maximum[n],float(np.linalg.norm(d.xpos[bid]-initial[n][0])));rotation[n]=max(rotation[n],rotation_degrees(d.xmat[bid].reshape(3,3),initial[n][1]))
 maxvel=max(maxvel,float(max(abs(d.qvel))))
 if(i+1)%round(.5/m.opt.timestep)==0:
  row={'time':float(d.time),'translation':maximum.copy(),'rotation_degrees':rotation.copy(),'peak_qvel':maxvel,'end_qvel':d.qvel.tolist(),'ncon':int(d.ncon),'min_contact':float(min(d.contact.dist[:d.ncon]))if d.ncon else None};records.append(row);print(row,flush=True)
  initial={n:(d.xpos[m.body(n).id].copy(),d.xmat[m.body(n).id].reshape(3,3).copy())for n in names};maximum={n:0. for n in names};rotation={n:0. for n in names};maxvel=0.
r={'candidate_method':'gravity potential on exact visual source vertices','initial_com_height_m':initialenergy,'candidate_com_height_m':float(res.fun),'candidate_delta_rotvec_rad':res.x.tolist(),'candidate_qpos':candidate.tolist(),'records':records,'scene':str(BASE/'scene.xml'),'no_velocity_reset_after_dynamic_start':True,'wall_seconds':time.monotonic()-start,'training_eligible':False};(OUT/'result.json').write_text(json.dumps(r,indent=2)+'\n');np.savez_compressed(OUT/'result.npz',candidate_qpos=candidate,qpos=d.qpos,qvel=d.qvel)
