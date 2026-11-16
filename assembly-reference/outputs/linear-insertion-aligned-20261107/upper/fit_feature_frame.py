from pathlib import Path
import json,shutil
import mujoco,numpy as np
from assembly_policy.so101.wowrobo_boundaries import world_mesh
p=Path(__file__).resolve().parent;root=p.parents[2];archive=p/'bodyframe-prior';archive.mkdir(exist_ok=True)
for path in list(p.iterdir()):
 if path.is_file() and not (archive/path.name).exists():shutil.copyfile(path,archive/path.name)
r=json.load(open(archive/'alignment.json'));m=mujoco.MjModel.from_xml_path(str(root/'outputs/linear-insertion-20261107/source-snapshot/scene.xml'));d=mujoco.MjData(m);d.qpos[:]=r['reference_qpos'];mujoco.mj_forward(m,d)

def plane(name,guide):
 mesh=world_mesh(m,d,m.geom(name).id);guide=np.array(guide,dtype=float);guide/=np.linalg.norm(guide);ns=mesh.face_normals.copy();ns[(ns@guide)<0]*=-1;mask=np.linalg.norm(ns-guide,axis=1)<.0001;weights=mesh.area_faces[mask];normal=(ns[mask]*weights[:,None]).sum(0);normal/=np.linalg.norm(normal)
 return normal,{'normal_world':normal.tolist(),'area_m2':float(weights.sum()),'triangles':int(mask.sum())}

sx,rx=plane('shoulder_connector_printed_1',[.999961923,0,-.008726535]);sy,ry=plane('shoulder_connector_printed_1',[0,1,0]);sy-=sx*(sy@sx);sy/=np.linalg.norm(sy);sz=np.cross(sx,sy);sz/=np.linalg.norm(sz)
cy,rcy=plane('servo_2_case',[-.0087262,.9999619,.00007615]);cz,rcz=plane('servo_2_case',[.0087265,0,.9999619]);cz-=cy*(cz@cy);cz/=np.linalg.norm(cz);cx=np.cross(cy,cz);cx/=np.linalg.norm(cx)
socket=np.column_stack([sx,sy,sz]);case=np.column_stack([cx,cy,cz]);delta=socket@case.T;br=d.body('base_servo').xmat.reshape(3,3).copy();target=delta@br
result={'socket_frame_world':socket.tolist(),'source_case_feature_frame_world':case.tolist(),'case_body_rotation_world':target.tolist(),'axis_world':(-sz).tolist(),'bodyframe_axis_error_deg':float(np.degrees(np.arccos(np.clip(sz@[0,0,1],-1,1)))),'dominant_wall_before_error_deg':float(np.degrees(np.arccos(np.clip(cy@sy,-1,1)))),'fitted_feature_frame_rotation_residual':float(np.max(abs(delta@case-socket))),'socket_x_fit':rx,'socket_y_fit':ry,'case_y_fit':rcy,'case_z_fit':rcz,'meaning':'Area-weighted source triangle plane normals fitted to actual socket/case. Orthonormal feature frames remove nominal sourcebodyframe yaw/entrytilt ambiguity; CAD unchanged.'}
(p/'feature-frame.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
