from pathlib import Path
import sys,os,json,time,importlib.util,subprocess,hashlib,shutil
import numpy as np
import mujoco
from patch import apply
root,out=map(Path,sys.argv[1:3]);tag=sys.argv[3];variant=tag;out=out/tag;out.mkdir(parents=True,exist_ok=True)
rel=root/'native-v13';sys.path.insert(0,str(rel/'assembly-so101'));os.environ['WOWROBO_CANONICAL_ROOT']=str(root/'robotics-assembly-3.0')
s=importlib.util.spec_from_file_location('_wowrobo_training_native_backend',rel/'robotics-assembly-3.0/assembly_policy/so101/wowrobo_contact_backend.py');b=importlib.util.module_from_spec(s);sys.modules[s.name]=b;s.loader.exec_module(b)
native=out.parent/'weld/native' if tag=='calibrated' else out/'native'
if len(sys.argv)<5 or sys.argv[4] not in ['ppo','reuse']:
 shutil.copytree(root/'native-contact-optimization-20261107/contact-separation-20261107/both/native',native,dirs_exist_ok=True)
 if variant!='baseline':
  apply(native,variant)
  for script in (['build.sh','build_contact_search.sh'] if variant in ['fused','winner','combined'] else ['build_contact_search.sh']):
   subprocess.run(['bash',str(native/script)],env=dict(os.environ,PYTHON=sys.executable),check=True)
original=b.load_backend;b.build_native=lambda directory:Path(directory);b.load_backend=lambda manifest:original(manifest,native)
from assembly_policy.wowrobo.env import BatchedAssemblyEnv,EnvConfig
bank=rel/'training-accepted-v13';meta=json.loads((bank/'manifest.json').read_text());e=BatchedAssemblyEnv(meta['task_names'],np.zeros(2,np.int64),EnvConfig(profile_path=meta['profile_path'],review_model=meta['review_model'],nthread=2))
if tag=='calibrated':
 ti=meta['task_names'].index('forearm_module')
 with np.load(bank/'forearm_module_train.npz') as z:
  ix=int(np.flatnonzero(z['stage']=='31_final_near')[0]);state=np.repeat(z['states'][ix:ix+1],2,axis=0)
 e.set_states(np.arange(2),state,np.full(2,ti,np.int64));d=e.data[0]
 parent=e.m.body('shoulder_connector').id;child=e.m.body('target_servo_2_pivot').id
 r=d.xmat[parent].reshape(3,3);pos=r.T@(d.xpos[child]-d.xpos[parent]);rot=r.T@d.xmat[child].reshape(3,3);quat=np.zeros(4);mujoco.mju_mat2Quat(quat,rot.ravel())
 eq=e.m.equality('installed_servo2_case_to_connector').id
 import xml.etree.ElementTree as ET
 profile=json.loads(Path(meta['profile_path']).read_text());tree=ET.parse(profile['model_path']).getroot();node=next(n for n in tree.find('equality') if n.get('name')=='installed_servo2_case_to_connector')
 np.testing.assert_allclose(e.m.eq_data[eq,3:10],np.array(list(map(float,node.get('relpose').split()))),atol=1e-12)
 old=e.m.eq_data[eq,3:10].copy();e.m.eq_data[eq,3:10]=np.r_[pos,quat]
 (out/'attachment-fit.json').write_text(json.dumps(dict(joint='installed_servo2_case_to_connector',relpose=np.r_[pos,quat].tolist(),old_relpose=old.tolist(),source='nativev13forearm_final_near_first_saved_row',bank_index=ix),indent=2)+'\n')
rows=[]
links=[('installed_cover_to_base','base_servo','base_side_cover'),('installed_connector_to_output_horns','target_servo_1_pivot','shoulder_connector'),('installed_servo2_case_to_connector','shoulder_connector','target_servo_2_pivot')]
try:
 for task,stage in [('upper_arm_module','21_upper_far_side'),('forearm_module','31_final_near')]:
  ti=meta['task_names'].index(task)
  with np.load(bank/(task+'_train.npz')) as z:
   ids=np.flatnonzero(z['stage']==stage);state=z['states'][[ids[0],ids[-1]]]
  for case in ['hold','force','hinge']:
   e.set_states(np.arange(2),state,np.full(2,ti,np.int64))
   for world,d in enumerate(e.data):
    def relative(a,b):
     a,b=e.m.body(a).id,e.m.body(b).id;r=d.xmat[a].reshape(3,3);return r.T@(d.xpos[b]-d.xpos[a]),r.T@d.xmat[b].reshape(3,3)
    active=[(n,a,b,*relative(a,b)) for n,a,b in links if d.eq_active[e.m.equality(n).id]]
    initial={j:float(d.qpos[int(e.m.joint(j).qposadr[0])]) for j in ['target_servo_2_shaft','target_servo_3_shaft']}
    if case=='force':d.xfrc_applied[e.m.body(task).id,:3]=[1.,-.5,.5]
    if case=='hinge':d.qfrc_applied[int(e.m.joint('target_servo_2_shaft').dofadr[0])]=.002
    maxmm=maxdeg=0.
    for k in range(128):
     mujoco.mj_step(e.m,d)
     assert np.isfinite(d.qpos).all() and not any(w.number for w in d.warning)
     for n,a,b,position,rotation in active:
      pos,rot=relative(a,b);maxmm=max(maxmm,float(np.linalg.norm(pos-position)*1000));maxdeg=max(maxdeg,float(np.degrees(np.arccos(np.clip((np.trace(rotation.T@rot)-1)/2,-1,1)))))
    row=dict(task=task,stage=stage,case=case,world=world,max_attachment_translation_mm=maxmm,max_attachment_rotation_deg=maxdeg,hinge_change={j:float(d.qpos[int(e.m.joint(j).qposadr[0])])-v for j,v in initial.items()},body_positions=d.xpos.copy().tolist(),warnings=[int(w.number) for w in d.warning]);rows.append(row);print(json.dumps({k:v for k,v in row.items() if k!='body_positions'}),flush=True)
 (out/'load.json').write_text(json.dumps(dict(rows=rows,steps_per_case=128,seconds_per_case=.064,force_N=[1.,-.5,.5],hinge_torque_Nm=.002),indent=2)+'\n')
finally:e.close()
