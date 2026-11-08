from pathlib import Path
import sys,os,json,time,importlib.util,subprocess,hashlib,shutil
import numpy as np
import mujoco
from patch import apply
root,out=map(Path,sys.argv[1:3]);tag=sys.argv[3];variant=tag;out=out/tag;out.mkdir(parents=True,exist_ok=True)
rel=root/'native-v13';sys.path.insert(0,str(rel/'assembly-so101'));os.environ['WOWROBO_CANONICAL_ROOT']=str(root/'robotics-assembly-3.0')
s=importlib.util.spec_from_file_location('_wowrobo_training_native_backend',rel/'robotics-assembly-3.0/assembly_policy/so101/wowrobo_contact_backend.py');b=importlib.util.module_from_spec(s);sys.modules[s.name]=b;s.loader.exec_module(b)
native=out/'native';shutil.copytree(root/'native-contact-optimization-20261107/contact-separation-20261107/both/native',native,dirs_exist_ok=True)
c=native/'so101_contact_search.c';text=c.read_text();header=(out.parent/'wowrobo_installed_contacts.h').read_text()
anchor='static int contact_search(const mjModel* m, mjData* d, mjPreContact* con,'
text=text.replace(anchor,header+'\n'+anchor,1)
anchor='  mjfCollision collider = original[m->geom_type[g1]][m->geom_type[g2]];'
text=text.replace(anchor,'  if(wowrobo_skip_installed_contacts(m,d,g1,g2))return 0;\n'+anchor,1);c.write_text(text)
subprocess.run(['bash',str(native/'build_contact_search.sh')],env=dict(os.environ,PYTHON=sys.executable),check=True)
original=b.load_backend;b.build_native=lambda directory:Path(directory);b.load_backend=lambda manifest:original(manifest,native)
from assembly_policy.wowrobo.env import BatchedAssemblyEnv,EnvConfig
bank=rel/'training-accepted-v13';meta=json.loads((bank/'manifest.json').read_text())
profile=json.loads(Path(meta['profile_path']).read_text())
if tag=='enabled':profile.update(training_installed_contacts='native-active-weld-v1',installed_attachment_fit=str(out.parent/'attachment-fit.json'),installed_attachment_fit_sha256=hashlib.sha256((out.parent/'attachment-fit.json').read_bytes()).hexdigest())
profile_path=out/'profile.json';profile_path.write_text(json.dumps(profile))
factory=(rel/'wowrobo_training_runtime.py').read_text();needle='        tree = ET.parse(source).getroot()';assert needle in factory
factory=factory.replace(needle,needle+(out.parent/'factory-feature-block.txt').read_text())
factory_path=out/'factory.py';factory_path.write_text(factory)
spec=importlib.util.spec_from_file_location('_canonical_wowrobo_training_runtime',factory_path);factory_mod=importlib.util.module_from_spec(spec);sys.modules[spec.name]=factory_mod;spec.loader.exec_module(factory_mod)
e=BatchedAssemblyEnv(meta['task_names'],np.zeros(2,np.int64),EnvConfig(profile_path=str(profile_path),review_model=meta['review_model'],nthread=2))
rows=[]
links=[('installed_cover_to_base','base_servo','base_side_cover'),('installed_connector_to_output_horns','target_servo_1_pivot','shoulder_connector'),('installed_servo2_case_to_connector','shoulder_connector','target_servo_2_pivot')]
try:
 for task,stage in [('upper_arm_module','21_upper_far_side'),('forearm_module','31_final_near')]:
  ti=meta['task_names'].index(task)
  with np.load(bank/(task+'_train.npz')) as z:
   ids=np.flatnonzero(z['stage']==stage);state=z['states'][[ids[0],ids[-1]]]
  for case in ['force']:
   e.set_states(np.arange(2),state,np.full(2,ti,np.int64))
   for world,d in enumerate(e.data):
    def relative(a,b):
     a,b=e.m.body(a).id,e.m.body(b).id;r=d.xmat[a].reshape(3,3);return r.T@(d.xpos[b]-d.xpos[a]),r.T@d.xmat[b].reshape(3,3)
    active=[(n,a,b,*relative(a,b)) for n,a,b in links if d.eq_active[e.m.equality(n).id]]
    initial={j:float(d.qpos[int(e.m.joint(j).qposadr[0])]) for j in ['target_servo_2_shaft','target_servo_3_shaft']}
    d.xfrc_applied[:]=0;d.qfrc_applied[:]=0
    if case=='force':d.xfrc_applied[e.m.body(task).id,:3]=[1.,-.5,.5]
    if case=='hinge':d.qfrc_applied[int(e.m.joint('target_servo_2_shaft').dofadr[0])]=.002
    maxmm=maxdeg=0.
    for k in range(1000):
     mujoco.mj_step(e.m,d)
     assert np.isfinite(d.qpos).all() and not any(w.number for w in d.warning)
     for n,a,b,position,rotation in active:
      pos,rot=relative(a,b);maxmm=max(maxmm,float(np.linalg.norm(pos-position)*1000));maxdeg=max(maxdeg,float(np.degrees(np.arccos(np.clip((np.trace(rotation.T@rot)-1)/2,-1,1)))))
    per_link={n:dict(translation_mm=float(np.linalg.norm(relative(a,b)[0]-position)*1000),rotation_deg=float(np.degrees(np.arccos(np.clip((np.trace(rotation.T@relative(a,b)[1])-1)/2,-1,1))))) for n,a,b,position,rotation in active}
    row=dict(per_link_final=per_link,task=task,stage=stage,case=case,world=world,max_attachment_translation_mm=maxmm,max_attachment_rotation_deg=maxdeg,hinge_change={j:float(d.qpos[int(e.m.joint(j).qposadr[0])])-v for j,v in initial.items()},body_positions=d.xpos.copy().tolist(),warnings=[int(w.number) for w in d.warning]);rows.append(row);print(json.dumps({k:v for k,v in row.items() if k!='body_positions'}),flush=True)
 (out/'long-load.json').write_text(json.dumps(dict(rows=rows,steps_per_case=1000,seconds_per_case=.5,force_N=[1.,-.5,.5],hinge_torque_Nm=.002),indent=2)+'\n')
finally:e.close()
