"""Export exact tested boxes relative to their original CAD visual frames."""
import hashlib
import json
import sys
from pathlib import Path
import mujoco
import numpy as np

folder=Path(sys.argv[1]);model=mujoco.MjModel.from_binary_path(str(folder/'model.mjb'));boxes=[]
for i in range(model.ngeom):
 name=model.geom(i).name
 if model.geom_type[i] != mujoco.mjtGeom.mjGEOM_BOX or not name.endswith('__contact'): continue
 visual=name.removesuffix('__contact');v=model.geom(visual).id
 assert model.geom_bodyid[i] == model.geom_bodyid[v]
 a,b=np.empty(9),np.empty(9)
 mujoco.mju_quat2Mat(a,model.geom_quat[v]);mujoco.mju_quat2Mat(b,model.geom_quat[i]);a=a.reshape(3,3);b=b.reshape(3,3)
 quat=np.empty(4);mujoco.mju_mat2Quat(quat,(a.T@b).ravel())
 role='finger' if model.geom_contype[i]==4 else 'robot' if name.startswith('physical_helper_') else 'part'
 boxes.append(dict(name=name,anchor=visual,body=model.body(int(model.geom_bodyid[i])).name,half_size=model.geom_size[i].tolist(),position=(a.T@(model.geom_pos[i]-model.geom_pos[v])).tolist(),quaternion=quat.tolist(),role=role,contype=int(model.geom_contype[i]),conaffinity=int(model.geom_conaffinity[i])))
receipt=json.loads((folder/'scene.json').read_text());output=dict(schema=1,physics_signature=receipt['scene_sha256'],label='Bare MuJoCo collision boxes',boxes=boxes,source_model_sha256=hashlib.sha256((folder/'model.mjb').read_bytes()).hexdigest())
(folder/'collision-boxes.json').write_text(json.dumps(output,indent=2)+'\n');print(len(boxes),'boxes exported')
