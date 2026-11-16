import importlib.util,json,sys,hashlib
from pathlib import Path
import numpy as np,mujoco
root,out=map(Path,sys.argv[1:]);exp=root/'native-contact-optimization-20261107/swept';canon=root/'robotics-assembly-3.0'
def load(name,path):
 s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);sys.modules[name]=m;s.loader.exec_module(m);return m
b=load('_wowrobo_training_native_backend',exp/'backend-exact.py');b.SOURCE=exp/'candidate';old=b.load_backend;b.load_backend=lambda manifest:old(manifest,out/'native-selected')
meta=json.loads((canon/'assets_local/wowrobo/training-accepted-v6/manifest.json').read_text());profile=json.loads(Path(meta['profile_path']).read_text());profile.update(hinge_limit_fit=str(out/'angles.json'),hinge_limit_fit_sha256=hashlib.sha256((out/'angles.json').read_bytes()).hexdigest(),training_hinge_limits='native-userdata-v1',hinge_limit_noise_rad=float(np.deg2rad(.05)))
path=out/'profile.json';path.write_text(json.dumps(profile,indent=2)+'\n')
factory=load('_canonical_wowrobo_training_runtime',out/'factory.py')
envmod=load('assembly_policy.wowrobo.hinge_env',out/'env-with-limits.py')
helper=load('limittest',out/'wowrobo_hinge_limits.py')
e=envmod.BatchedAssemblyEnv(meta['task_names'],np.zeros(2,np.int64),envmod.EnvConfig(profile_path=str(path),review_model=meta['review_model'],nthread=2))
try:
 with np.load(canon/'assets_local/wowrobo/training-accepted-v6/base_side_cover_train.npz') as z:states=np.repeat(z['states'][0:1],2,axis=0)
 states[:,-3:]=[[.9,1.1,.95],[1.1,.9,1.05]]
 states=helper.augment_reset_states(states,e.m,np.random.default_rng(4125));e.set_states(np.arange(2),states,np.zeros(2,np.int64));base=e.m.jnt_range.copy();noise=states[:,-4:].copy()
 for _ in range(5):
  reward,done,info=e.step(np.zeros((2,12)))
  assert np.isfinite(reward).all() and not np.any(info['unstable'])
  assert not np.any(info['safety'])
  np.testing.assert_array_equal(e.state[:,-4:],noise)
  for i,d in enumerate(e.data):np.testing.assert_array_equal(d.userdata[3:7],noise[i])
 np.testing.assert_array_equal(e.m.jnt_range,base)
 proof=dict(status='PASS',nuserdata=e.m.nuserdata,nstate=e.nstate,npair=e.m.npair,worlds=2,threads=2,controls=5,noise_offsets=noise.tolist(),safety=info['safety'].tolist(),friction=states[:,-7:-4].tolist(),noise_persisted=True,shared_ranges_unchanged=True,model_signature=e.model_signature)
 (out/'env-smoke.json').write_text(json.dumps(proof,indent=2)+'\n');print(json.dumps(proof))
finally:e.close()
