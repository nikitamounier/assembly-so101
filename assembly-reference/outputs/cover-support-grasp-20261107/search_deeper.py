import copy,json,time
from pathlib import Path
import mujoco,numpy as np,trimesh
from scipy.spatial.transform import Rotation
from assembly_policy.so101.wowrobo_randomization import WowRoboRandomizer
from assembly_policy.so101.wowrobo_grasps import native_jaw_grasp,outer_wall_contacts
from assembly_policy.so101.wowrobo_boundaries import NativeBoundaryAudit,world_mesh

OUT=Path(__file__).resolve().parent
R=WowRoboRandomizer(config='configs/wowrobo_randomization_repaired.json');m=R.model;seq=R.sequence
old=R.restore('18_upper_floor_grasp');d=R.restore('18_upper_floor_grasp');A=NativeBoundaryAudit(m)
leftq=[int(m.joint('left_'+name).qposadr[0]) for name in ['shoulder_pan','shoulder_lift','elbow_flex','wrist_flex','wrist_roll','gripper']]
cover=m.body('base_side_cover').id;cr=old.xmat[cover].reshape(3,3);cp=old.xpos[cover]
site=m.site('left_gripperframe').id
rows=[]
# Begin with the photo-facing cover section, then safe vertical/longitudinal alternatives.
for i,(y,z,pitch,depth) in enumerate([(y,z,-.2,depth) for z in [.059,.064] for y in [.041,.048] for depth in [.006,.010]]):
 try:
  center,contacts=outer_wall_contacts(seq.layout,'base_side_cover',[-.00009943,y,z])
  spec=native_jaw_grasp(m,float(np.ptp(contacts[:,0])),depth,half_length=.008)
 except ValueError:continue
 span=cr@np.array([-1.,0,0]);approach=Rotation.from_rotvec(span*pitch).as_matrix()@(cr@np.array([0.,-1,0]))
 d.qpos[:]=old.qpos;seq.previous['left']=old.qpos[leftq[:5]].copy()
 info=seq.solve_arm(d,'left',cp+cr@center,span,approach,spec['opening_radians'],np.random.default_rng(700+i),grasp=spec)
 audit=A.audit(d,target_target=False)
 maxleft=max((p['penetration_m'] for p in audit['helper_target'] if p['a']['helper_side']=='left'),default=0)
 pads=[]
 gid=next(g for g in A.targets if A.info[g]['piece']=='base_side_cover');target=world_mesh(m,d,gid)
 frame=d.site_xmat[site].reshape(3,3)
 for body in ['left_gripper','left_moving_jaw_so101_v1']:
  hg=[g for g in A.helper if m.body(m.geom_bodyid[g]).name==body][-1]
  hand=world_mesh(m,d,hg)
  points=np.vstack([hand.vertices,hand.triangles_center])
  normals=np.vstack([hand.vertex_normals,hand.face_normals])
  local=(points-d.site_xpos[site])@frame;ln=normals@frame
  mask=(local[:,0]>-.025)&(local[:,0]<.008)&(abs(local[:,1])<.010)&(ln[:,2]>(.5) if body=='left_gripper' else ln[:,2]<(-.5))
  pts=points[mask];ns=normals[mask]
  nearest,dist,faces=trimesh.proximity.closest_point(target,pts)
  k=int(np.argmin(dist))
  pads.append({'gap_m':float(dist[k]),'point':pts[k].tolist(),'target_point':nearest[k].tolist(),'target_normal':target.face_normals[faces[k]].tolist(),'pad_normal':ns[k].tolist(),'normal_dot':float(ns[k]@target.face_normals[faces[k]]),'point_site':((pts[k]-d.site_xpos[site])@frame).tolist()})
 q=d.qpos[leftq].copy()
 row={'i':i,'depth':depth,'center_piece_local_m':center.tolist(),'pitch':pitch,'spec':spec,'q_left':q.tolist(),'info':info,'max_left_penetration_m':maxleft,'pads':pads,'rejected':audit['candidate_rejected']}
 rows.append(row);(OUT/'deeper-candidates.json').write_text(json.dumps(rows,indent=2))
 print(i,'yz',y,z,'pitch',pitch,'errmm',round(info['grasp_position_error_m']*1000,3),'span',round(info['span_vector_error'],4),'penmm',round(maxleft*1000,3),'padsmm',[round(p['gap_m']*1000,3) for p in pads],flush=True)
 if maxleft<=.0005 and max(p['gap_m'] for p in pads)<.001 and all(p['normal_dot']<-.8 for p in pads):
  np.savez_compressed(OUT/f'deeper-candidate-{i}.npz',qpos=d.qpos,controls=d.ctrl)
  print('CANDIDATE',i,flush=True)
