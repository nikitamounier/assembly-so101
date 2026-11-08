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
if len(sys.argv)>4 and sys.argv[4]=='ppo':
 import torch
 from assembly_policy import train
 torch.set_num_threads(1)
 args=train.parse(['--suite','wowrobo','--wowrobo-debug-resets','--bank-per-family','0','--bank-dir',str(bank),'--run-dir',str(out/'ppo'),'--envs','2','--steps','32','--nthread','2','--device','cpu','--iters','1','--micro-batch','1024','--no-eval','--cost-coef','.00002','--action-rate-coef','0','--eps-q','.01','--seed','20261107'])
 e.close();tr=train.Trainer(args)
 cp=root/'run-v10/checkpoint.before-state-build-repause.pt';d=torch.load(cp,map_location='cpu',weights_only=False);tr.model.load_state_dict(d['model']);tr.opt.load_state_dict(d['opt']);tr.lr=d['lr']
 try:
  t=time.perf_counter();cpu=time.process_time();tr.train();wall=time.perf_counter()-t;cpu=time.process_time()-cpu
 finally:tr.env.close()
 (out/'ppo-result.json').write_text(json.dumps(dict(seconds=wall,cpu_seconds=cpu,record=json.loads((out/'ppo/log.jsonl').read_text().splitlines()[-1]),affinity=sorted(os.sched_getaffinity(0))),indent=2)+'\n')
 sys.exit(0)
rows=[];arrays={}
try:
 for task,stage in [('base_side_cover','00_loose'),('base_side_cover','05_cover_near'),('upper_arm_module','21_upper_far_side'),('forearm_module','31_final_near')]:
  ti=meta['task_names'].index(task)
  with np.load(bank/(task+'_train.npz')) as z:
   ids=np.flatnonzero(z['stage']==stage);fr=z['states'][ids,159:162];nom=np.flatnonzero(np.all(fr==1,axis=1));var=np.flatnonzero(np.any(fr!=1,axis=1));ix=[int(ids[nom[0] if len(nom) else 0]),int(ids[var[0] if len(var) else 0])];state=z['states'][ix]
  times=[]
  for rep in range(3):
   e.set_states(np.arange(2),state,np.full(2,ti,np.int64));t=time.perf_counter();r,done,info=e.step(.005*np.sin(np.arange(24).reshape(2,12)));times.append(time.perf_counter()-t)
   assert np.isfinite(e.state).all();assert not any(sum(int(w.number) for w in d.warning) for d in e.data)
   arrays[stage+f'_{rep}_state']=e.state.copy();arrays[stage+f'_{rep}_reward']=r;arrays[stage+f'_{rep}_done']=done
   for key,value in info.items():arrays[stage+f'_{rep}_info_'+key]=value.copy()
  row=dict(task=task,stage=stage,times_s=times);rows.append(row);print(json.dumps(row),flush=True)
 if False:
  import ctypes
  lib=b.surface_library();lib.probe_surface_counts.argtypes=[ctypes.POINTER(ctypes.c_ulonglong)]
  counts=(ctypes.c_ulonglong*3)();lib.probe_surface_counts(counts)
  (out/'cache-counts.json').write_text(json.dumps(dict(hits=counts[0],rebuilds=counts[1],retained_on_rebuild=counts[2])))
 np.savez_compressed(out/'outputs.npz',**arrays)
 (out/'result.json').write_text(json.dumps(dict(variant=variant,rows=rows,affinity=sorted(os.sched_getaffinity(0)),hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in native.iterdir() if p.suffix in ['.h','.c','.cc']}),indent=2)+'\n')
finally:e.close()
