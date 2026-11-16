from pathlib import Path
import sys,os,json,time,importlib.util,subprocess,hashlib,shutil
import numpy as np
import mujoco
from patch import apply
root,out=map(Path,sys.argv[1:3]);tag=sys.argv[3];variant=tag;out=out/tag;out.mkdir(parents=True,exist_ok=True)
rel=root/'native-v13';sys.path.insert(0,str(rel/'assembly-so101'));os.environ['WOWROBO_CANONICAL_ROOT']=str(root/'robotics-assembly-3.0')
s=importlib.util.spec_from_file_location('_wowrobo_training_native_backend',rel/'robotics-assembly-3.0/assembly_policy/so101/wowrobo_contact_backend.py');b=importlib.util.module_from_spec(s);sys.modules[s.name]=b;s.loader.exec_module(b)
native=out.parent/'weld/native' if tag=='calibratedstiff' else out/'native'
if len(sys.argv)<5 or sys.argv[4] not in ['ppo','reuse']:
 shutil.copytree(root/'native-contact-optimization-20261107/contact-separation-20261107/both/native',native,dirs_exist_ok=True)
 if variant!='baseline':
  apply(native,variant)
  for script in (['build.sh','build_contact_search.sh'] if variant in ['fused','winner','combined'] else ['build_contact_search.sh']):
   subprocess.run(['bash',str(native/script)],env=dict(os.environ,PYTHON=sys.executable),check=True)
original=b.load_backend;b.build_native=lambda directory:Path(directory);b.load_backend=lambda manifest:original(manifest,native)
from assembly_policy.wowrobo.env import BatchedAssemblyEnv,EnvConfig
bank=rel/'training-accepted-v13';meta=json.loads((bank/'manifest.json').read_text());e=BatchedAssemblyEnv(meta['task_names'],np.zeros(2,np.int64),EnvConfig(profile_path=meta['profile_path'],review_model=meta['review_model'],nthread=2))
if tag=='calibratedstiff':
 ti=meta['task_names'].index('forearm_module')
 with np.load(bank/'forearm_module_train.npz') as z:
  ix=int(np.flatnonzero(z['stage']=='31_final_near')[0]);state=np.repeat(z['states'][ix:ix+1],2,axis=0)
 e.set_states(np.arange(2),state,np.full(2,ti,np.int64));d=e.data[0]
 import xml.etree.ElementTree as ET
 profile=json.loads(Path(meta['profile_path']).read_text());tree=ET.parse(profile['model_path']).getroot();fits={}
 for name,parent_name,child_name in [('installed_cover_to_base','base_servo','base_side_cover'),('installed_connector_to_output_horns','target_servo_1_pivot','shoulder_connector'),('installed_servo2_case_to_connector','shoulder_connector','target_servo_2_pivot')]:
  parent=e.m.body(parent_name).id;child=e.m.body(child_name).id;r=d.xmat[parent].reshape(3,3);pos=r.T@(d.xpos[child]-d.xpos[parent]);rot=r.T@d.xmat[child].reshape(3,3);quat=np.zeros(4);mujoco.mju_mat2Quat(quat,rot.ravel());eq=e.m.equality(name).id
  node=next(n for n in tree.find('equality') if n.get('name')==name);np.testing.assert_allclose(e.m.eq_data[eq,3:10],np.array(list(map(float,node.get('relpose').split()))),atol=1e-12)
  fits[name]=dict(relpose=np.r_[pos,quat].tolist(),old_relpose=e.m.eq_data[eq,3:10].copy().tolist());e.m.eq_data[eq,3:10]=np.r_[pos,quat];e.m.eq_data[eq,10]=.2
 (out/'attachment-fit.json').write_text(json.dumps(dict(fits=fits,source='nativev13forearm_final_near_first_saved_row',bank_index=ix),indent=2)+'\n')
if tag=='calibratedstiff':
 signature=hashlib.sha256(e.model_signature.encode()+e.m.eq_data.tobytes()+(native/'so101_contact_search.c').read_bytes()).hexdigest()
 for module in list(sys.modules.values()):
  cache=getattr(module,'_CACHE',None)
  if isinstance(cache,dict):
   for key,value in list(cache.items()):
    if isinstance(value,tuple) and len(value)==3 and value[0] is e.m:
     profile=dict(value[1],diagnostic_attachment_torquescale=.2,diagnostic_attachment_fit=str(out/'attachment-fit.json'));cache[key]=(value[0],profile,signature)
 e.model_signature=signature
 original_bank=bank;bank=out/'diagnostic-bank';bank.mkdir(exist_ok=True)
 for source in original_bank.glob('*.npz'):
  dest=bank/source.name
  if not dest.exists():dest.symlink_to(source)
 diagnostic_meta=dict(meta,model_signature=signature,training_qualified=False,qualification_status='DIAGNOSTIC_UNQUALIFIED_RESETS',diagnostic_source_bank=str(original_bank),diagnostic_source_manifest_sha256=hashlib.sha256((original_bank/'manifest.json').read_bytes()).hexdigest())
 (bank/'manifest.json').write_text(json.dumps(diagnostic_meta,indent=2)+'\n')
if len(sys.argv)>4 and sys.argv[4]=='ppo':
 import torch
 from assembly_policy import train
 torch.set_num_threads(1)
 args=train.parse(['--suite','wowrobo','--wowrobo-debug-resets','--bank-per-family','0','--bank-dir',str(bank),'--run-dir',str(out/'ppo'),'--envs','2','--steps','32','--nthread','2','--device','cpu','--iters','1','--micro-batch','1024','--no-eval','--cost-coef','.00002','--action-rate-coef','0','--eps-q','.01','--seed','20261107'])
 e.close();tr=train.Trainer(args)
 if tag=='calibratedstiff':
  assert np.all(tr.env.env.m.eq_data[:,10]==.2), 'Candidate calibration was not carried into Trainer'
  assert tr.env.env.model_signature==signature
 cp=root/'run-v10/checkpoint.before-state-build-repause.pt';d=torch.load(cp,map_location='cpu',weights_only=False);tr.model.load_state_dict(d['model']);tr.opt.load_state_dict(d['opt']);tr.lr=d['lr']
 try:
  t=time.perf_counter();cpu=time.process_time();tr.train();wall=time.perf_counter()-t;cpu=time.process_time()-cpu
 finally:tr.env.close()
 (out/'ppo-result.json').write_text(json.dumps(dict(seconds=wall,cpu_seconds=cpu,record=json.loads((out/'ppo/log.jsonl').read_text().splitlines()[-1]),affinity=sorted(os.sched_getaffinity(0))),indent=2)+'\n')
 sys.exit(0)
