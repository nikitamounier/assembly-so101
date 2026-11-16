"""Freeze measured hinge stops as a separate native version, retaining v12 kernels."""
from pathlib import Path
import hashlib,importlib.util,json,shutil,sys
import numpy as np
ROOT=Path(__file__).resolve().parent.parent;REL=ROOT/'native-v13'
OLD=ROOT/'native-v12';AD=REL/'assembly-so101';C=REL/'robotics-assembly-3.0'
HAND=ROOT/'native-contact-optimization-20261107/hinge-limits-full'
assert not (REL/'prepared.json').exists()
assert json.loads((OLD/'deployment-ready.json').read_text())['source_valid']
hashes={'angles.json':'b05b4242f365dc290b6f50acede8678ad9a3dfea104b863ab88aea4b9d2eb007',
        'wowrobo_hinge_limits.py':'737823fbb12457e40889de8cd5eac9246e99c63f00658a2238215d2d7faceabd',
        'wowrobo_hinge_limits.cc':'7760bb8e9adb94606a72c6cdfe749aca98e3391b550c29fb91082540ad24b54b',
        'factory.py':'17fe64300cea624f5bca43fd2c59722eb3e1cbc3bcdc12529aeafe9561dc6a65'}
for name,sha in hashes.items():
    assert hashlib.sha256((HAND/name).read_bytes()).hexdigest()==sha,name
    shutil.copy2(HAND/name,REL/('factory-handoff.py' if name=='factory.py' else name))
hp=REL/'wowrobo_hinge_limits.py';hs=hp.read_text();hs=hs.replace('    root = ET.fromstring(xml)', '    root = ET.fromstring(xml)\n    if root.find("compiler") is None or root.find("compiler").get("angle") != "radian":\n        raise ValueError("Measured hinge limits require radian XML")');hp.write_text(hs)
for name in ('assembly_policy','configs','plugin'):
    shutil.copytree(OLD/'assembly-so101'/name,AD/name,ignore=shutil.ignore_patterns('__pycache__'),dirs_exist_ok=True)
shutil.copytree(OLD/'robotics-assembly-3.0/assembly_policy',C/'assembly_policy',ignore=shutil.ignore_patterns('__pycache__'),dirs_exist_ok=True)
s=(OLD/'wowrobo_training_runtime.py').read_text().replace(str(OLD/'robotics-assembly-3.0'),str(C))
handoff=(HAND/'factory.py').read_text()
start=handoff.index('        limit_helper = None')
end=handoff.index('        xml = ET.tostring',start)
needle='        xml = ET.tostring';assert s.count(needle)==1;s=s.replace(needle,handoff[start:end]+needle)
start=handoff.index('        if limit_helper is not None:')
end=handoff.index('        if m.opt.gravity',start)
needle='        if m.opt.gravity';assert s.count(needle)==1;s=s.replace(needle,handoff[start:end]+needle)
s=s.replace('        if profile.get("hinge_limit_fit"):', '        if profile.get("hinge_limit_fit"):\n            if profile.get("training_episode_friction") != "native-userdata-v1":\n                raise ValueError("Hinge userdata requires the versioned friction prefix")')
s=s.replace('        xml = ET.tostring(tree, encoding="unicode")', '        limit_source += json.dumps(dict(noise_amplitude_rad=profile.get("hinge_limit_noise_rad", float(0.05*3.141592653589793/180)), userdata_offset=3), sort_keys=True).encode()\n        xml = ET.tostring(tree, encoding="unicode")')
s=s.replace('            + kernel_source\n','            + kernel_source\n            + limit_source\n')
(REL/'wowrobo_training_runtime.py').write_text(s)
p=AD/'assembly_policy/wowrobo/runtime.py';s=p.read_text().replace('_canonical_wowrobo_training_runtime_v12','_canonical_wowrobo_training_runtime_v13').replace('native-v12/wowrobo_training_runtime.py','native-v13/wowrobo_training_runtime.py');p.write_text(s)
# Only port the payload changes into the actual frozen current environment.
p=AD/'assembly_policy/wowrobo/env.py';before=p.read_text();s=before
needle='        self.nstate = self.eq_end + (self.m.nu + self.m.nv if self.extra_reset else 0) + (3 if self.episode_friction else 0)'
assert s.count(needle)==1
s=s.replace(needle,'''        self.hinge_limits = self.profile.get("training_hinge_limits") == "native-userdata-v1"
        self.friction_start = self.eq_end + (self.m.nu + self.m.nv if self.extra_reset else 0)
        self.hinge_start = self.friction_start + (3 if self.episode_friction else 0)
        self.nstate = self.hinge_start + (4 if self.hinge_limits else 0)''')
needle='            if self.episode_friction: d.userdata[:3] = self.state[i,-3:]';assert s.count(needle)==1
s=s.replace(needle,'''            if self.episode_friction: d.userdata[:3] = self.state[i,self.friction_start:self.hinge_start]
            if self.hinge_limits: d.userdata[3:7] = self.state[i,self.hinge_start:self.hinge_start+4]''')
assert s.count('self.state[:,-3:]')==1;s=s.replace('self.state[:,-3:]','self.state[:,self.friction_start:]')
s=s.replace('if self.episode_friction else []','if self.episode_friction or self.hinge_limits else []')
s=s.replace('if self.episode_friction else 0','if self.episode_friction or self.hinge_limits else 0')
# The preceding replacement must not add phantom friction slots in hinge-only profiles.
s=s.replace('self.hinge_start = self.friction_start + (3 if self.episode_friction or self.hinge_limits else 0)','self.hinge_start = self.friction_start + (3 if self.episode_friction else 0)')
compile(s,str(p),'exec');p.write_text(s)
import difflib
(REL/'env.diff').write_text(''.join(difflib.unified_diff(before.splitlines(True),s.splitlines(True),fromfile='frozen-v12/env.py',tofile='native-v13/env.py')))
oldbank=OLD/'training-accepted-v12';bank=REL/'training-accepted-v13';bank.mkdir(exist_ok=True)
meta=json.loads((oldbank/'manifest.json').read_text());profile=json.loads(Path(meta['profile_path']).read_text())
profile.update(hinge_limit_fit=str(REL/'angles.json'),hinge_limit_fit_sha256=hashes['angles.json'],training_hinge_limits='native-userdata-v1',hinge_limit_noise_rad=float(np.deg2rad(.05)))
(REL/'profile.json').write_text(json.dumps(profile,indent=2)+'\n')
def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path);mod=importlib.util.module_from_spec(spec);sys.modules[name]=mod;spec.loader.exec_module(mod);return mod
factory=load('_native13_prepare',REL/'wowrobo_training_runtime.py');m,_,identity=factory.load_native(REL/'profile.json')
assert m.opt.timestep==.0005 and m.nuserdata==7
assert not any((m.pair(i).name or '').startswith('passive_motor_stop_') for i in range(m.npair))
helper=load('_native13_augment',REL/'wowrobo_hinge_limits.py');rng=np.random.default_rng(2026110732);rows={}
for source in sorted(oldbank.glob('*.npz')):
    with np.load(source,allow_pickle=False) as z:fields={k:z[k].copy() for k in z.files}
    states=fields['states'];assert states.shape[1]==162
    fields['states']=helper.augment_reset_states(states,m,rng,amplitude=profile['hinge_limit_noise_rad'])
    assert fields['states'].shape[1]==166 and np.array_equal(fields['states'][:,:162],states)
    assert np.max(np.abs(fields['states'][:,162:]))<=profile['hinge_limit_noise_rad']
    np.savez_compressed(bank/source.name,**fields);rows[source.name]=len(states)
meta.update(model_signature=identity,profile_path=str(REL/'profile.json'),source_native_runtime_signature=json.loads((OLD/'prepared.json').read_text())['physics_signature'],source_bank_directory=str(oldbank),hinge_limit_qualification='pending full-bank/contact/dynamics/distributed-PPO qualification',state_payload_size=166,hinge_limit_noise_rad=profile['hinge_limit_noise_rad'],hinge_noise_seed=2026110732,hinge_noise_sampling='one saved independent four-endpoint draw per bank row; retained unchanged within each episode')
(bank/'manifest.json').write_text(json.dumps(meta,indent=2)+'\n')
paths=[p for folder in (AD,C/'assembly_policy',bank) for p in folder.rglob('*') if p.is_file() and '__pycache__' not in p.parts]+[REL/n for n in ('wowrobo_training_runtime.py','wowrobo_hinge_limits.py','wowrobo_hinge_limits.cc','angles.json','profile.json')]
x=dict(status='PREPARED_NOT_ACTIVE',physics_signature=identity,physics_backend='wowrobo-native-source-sdf',timestep=.0005,substeps=80,control_dt=.04,nuserdata=7,state_payload_size=166,source_prefix162_unchanged=True,noise_rad=profile['hinge_limit_noise_rad'],bank_rows=rows,files={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(set(paths))})
(REL/'prepared.json').write_text(json.dumps(x,indent=2)+'\n');print(json.dumps({k:v for k,v in x.items() if k!='files'}),flush=True)
