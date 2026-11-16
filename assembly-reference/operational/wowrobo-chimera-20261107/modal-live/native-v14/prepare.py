from pathlib import Path
import json,hashlib,shutil,importlib.util,sys
ROOT=Path(__file__).resolve().parent.parent;OLD=ROOT/'native-v13';REL=ROOT/'native-v14'
assert not (REL/'prepared.json').exists()
old=json.loads((OLD/'prepared.json').read_text())
for n,h in old['files'].items():assert hashlib.sha256((ROOT/n).read_bytes()).hexdigest()==h,n
for n in ('assembly-so101/assembly_policy','assembly-so101/configs','assembly-so101/plugin','robotics-assembly-3.0/assembly_policy'):
 shutil.copytree(OLD/n,REL/n,ignore=shutil.ignore_patterns('__pycache__'),dirs_exist_ok=True)
for n in ('wowrobo_training_runtime.py','wowrobo_hinge_limits.py','wowrobo_hinge_limits.cc','angles.json'):
 s=(OLD/n).read_text().replace(str(OLD),str(REL));(REL/n).write_text(s)
p=REL/'assembly-so101/assembly_policy/wowrobo/runtime.py';p.write_text(p.read_text().replace('native-v13','native-v14').replace('_runtime_v13','_runtime_v14'))
for n,h in json.loads((REL/'source-hashes.json').read_text()).items():
 b=(REL/'source'/n).read_bytes();assert hashlib.sha256(b).hexdigest()==h;(REL/n).write_bytes(b)
profile=json.loads((OLD/'profile.json').read_text());profile['hinge_limit_fit']=str(REL/'angles.json');(REL/'profile.json').write_text(json.dumps(profile,indent=2)+'\n')
spec=importlib.util.spec_from_file_location('_native14_prepare',REL/'wowrobo_training_runtime.py');m=importlib.util.module_from_spec(spec);sys.modules[spec.name]=m;spec.loader.exec_module(m);model,_,identity=m.load_native(REL/'profile.json')
assert model.opt.timestep==.0005 and model.nuserdata==7
bank=REL/'training-accepted-v14';shutil.copytree(OLD/'training-accepted-v13',bank)
p=bank/'manifest.json';meta=json.loads(p.read_text());meta.update(model_signature=identity,profile_path=str(REL/'profile.json'),source_native_runtime_signature=old['physics_signature'],source_bank_directory=str(OLD/'training-accepted-v13'),state_size=166,state_payload_size=166,collision_qualification='Pending main-run integration of independently qualified general separation/four-way exact cache');p.write_text(json.dumps(meta,indent=2)+'\n')
paths=[p for folder in (REL/'assembly-so101',REL/'robotics-assembly-3.0',bank) for p in folder.rglob('*') if p.is_file() and '__pycache__' not in p.parts]+[REL/n for n in ('wowrobo_training_runtime.py','wowrobo_hinge_limits.py','wowrobo_hinge_limits.cc','angles.json','profile.json')]
x=dict(status='PREPARED_NOT_ACTIVE',physics_signature=identity,physics_backend='wowrobo-native-source-sdf',timestep=.0005,substeps=80,control_dt=.04,nuserdata=7,state_payload_size=166,bank_npz_identical=True,source_physics_signature=old['physics_signature'],files={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(set(paths))});(REL/'prepared.json').write_text(json.dumps(x,indent=2)+'\n');print(json.dumps({k:v for k,v in x.items() if k!='files'}),flush=True)
