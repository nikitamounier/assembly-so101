import importlib.util,json,sys,time,xml.etree.ElementTree as ET
from pathlib import Path
import mujoco,numpy as np
root,out=map(Path,sys.argv[1:]);canon=root/'robotics-assembly-3.0';exp=root/'native-contact-optimization-20261107/swept'
# Use the same selected native contact kernels on both sides.
spec=importlib.util.spec_from_file_location('backend',exp/'backend-exact.py');b=importlib.util.module_from_spec(spec);spec.loader.exec_module(b);b.SOURCE=exp/'candidate'
meta=json.loads((canon/'assets_local/wowrobo/training-accepted-v6/manifest.json').read_text());profile=json.loads(Path(meta['profile_path']).read_text());b.load_backend(profile['grid_manifest'],out/'native-selected')
spec=importlib.util.spec_from_file_location('limits',out/'wowrobo_hinge_limits.py');limit=importlib.util.module_from_spec(spec);spec.loader.exec_module(limit)
fit=json.loads((out/'angles.json').read_text());xml=Path(profile['model_path']).read_text();r=ET.fromstring(xml);r.find('option').set('timestep','.0005');r.find('size').set('memory','256M');r.find('size').set('nuserdata','7');xml=ET.tostring(r,encoding='unicode')
changed,removed=limit.configure_xml(xml,fit);(out/'candidate.xml').write_text(changed)
a=mujoco.MjModel.from_xml_string(xml);m=mujoco.MjModel.from_xml_string(changed);jids=[m.joint(n).id for n in limit.JOINTS];adr=[int(m.jnt_qposadr[j]) for j in jids];bounds=m.jnt_range[jids].copy();bank=canon/'assets_local/wowrobo/training-accepted-v6';bad=[];count=0
mins=np.full(2,np.inf);maxs=-mins
for task in meta['task_names']:
 for split in ['train','eval']:
  path=bank/(task+'_'+split+'.npz')
  if not path.exists():continue
  with np.load(path) as z:
   values=z['states'][:,np.array(adr)+1];count+=len(values);mins=np.minimum(mins,values.min(axis=0));maxs=np.maximum(maxs,values.max(axis=0))
   for i in np.flatnonzero(np.any((values<bounds[:,0])|(values>bounds[:,1]),axis=1)):bad.append(dict(task=task,split=split,row=int(i),angles_rad=values[i].tolist()))
with np.load(bank/'base_side_cover_train.npz') as z:reset=z['states'][0].copy()
rows=[]
for label,model in [('baseline',a),('limits',m)]:
 d=mujoco.MjData(model);d.ctrl[:]=d.qpos[model.jnt_qposadr[model.actuator_trnid[:,0]]]
 times=[]
 for repeat in range(3):
  mujoco.mj_resetData(model,d);full=mujoco.mj_stateSize(model,mujoco.mjtState.mjSTATE_FULLPHYSICS);end=full+model.neq
  mujoco.mj_setState(model,d,reset[:full],mujoco.mjtState.mjSTATE_FULLPHYSICS);d.eq_active[:]=reset[full:end];d.ctrl[:]=reset[end:end+model.nu];d.qacc_warmstart[:]=reset[end+model.nu:end+model.nu+model.nv]
  start=time.perf_counter()
  for _ in range(80):mujoco.mj_step(model,d)
  times.append(time.perf_counter()-start)
 assert not any(w.number for w in d.warning),'Native warning invalidates probe'
 assert np.isfinite(d.qpos).all() and np.isfinite(d.qvel).all()
 rows.append(dict(variant=label,times_s=times,ncon=d.ncon))
result=dict(joint_ranges_rad=bounds.tolist(),joint_ranges_deg=np.degrees(bounds).tolist(),removed_pairs=removed,bank_rows=count,outside_rows=bad,bank_angle_min_rad=mins.tolist(),bank_angle_max_rad=maxs.tolist(),timing=rows,npair_before=a.npair,npair_after=m.npair,scope='Nominal original-bank range admission plus matched0.5ms loose40ms physics, no friction/limit variation hook in this geometric timing')
(out/'qualification.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k!='outside_rows'}),flush=True);print('outside_count',len(bad),flush=True)
