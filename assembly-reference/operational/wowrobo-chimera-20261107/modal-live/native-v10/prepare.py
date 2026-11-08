"""Isolate a native timestep-only candidate; old frozen sources stay intact."""
from pathlib import Path
import hashlib,json,shutil,importlib.util,difflib
ROOT=Path(__file__).resolve().parent.parent;REL=ROOT/'native-v10';AD=REL/'assembly-so101';C=ROOT/'robotics-assembly-3.0'
assert not (REL/'prepared.json').exists()
for name in ('assembly_policy','configs'):
 shutil.copytree(ROOT/'assembly-so101'/name,AD/name,ignore=shutil.ignore_patterns('__pycache__'),dirs_exist_ok=True)
source=C/'assembly_policy/so101/wowrobo_training_runtime.py';original=source.read_text();assert 'tree.find("option").set("timestep"' not in original
s=original.replace('ROOT = Path(__file__).resolve().parents[2]','ROOT = Path('+repr(str(C))+')');needle='        size = tree.find("size")';assert s.count(needle)==1;s=s.replace(needle,'        tree.find("option").set("timestep", ".0005")\n'+needle)
p=REL/'wowrobo_training_runtime.py';p.write_text(s);(REL/'factory.diff').write_text(''.join(difflib.unified_diff(original.splitlines(True),s.splitlines(True),fromfile=str(source),tofile=str(p))))
p=AD/'assembly_policy/wowrobo/runtime.py';s=p.read_text().replace('name = "_canonical_wowrobo_training_runtime"','name = "_canonical_wowrobo_training_runtime_v10"').replace('ROOT / "assembly_policy/so101/wowrobo_training_runtime.py"','ROOT.parent / "native-v10/wowrobo_training_runtime.py"');p.write_text(s)
oldbank=C/'assets_local/wowrobo/training-accepted-v6';newbank=REL/'training-accepted-v10';newbank.mkdir(exist_ok=True);meta=json.loads((oldbank/'manifest.json').read_text());profile=meta['profile_path']
spec=importlib.util.spec_from_file_location('_native_v10_prepare_factory',REL/'wowrobo_training_runtime.py');mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod);m,_,identity=mod.load_native(profile);assert m.opt.timestep==.0005
meta.update(model_signature=identity,source_admission_model_signature='8c4783bc443e8e9efad8f19890b6910a79299d434dbb3111af9f24733d850879',source_bank_directory=str(oldbank),native_timestep_s=.0005,control_dt_s=.04,timestep_qualification='pending; bounded native validation required')
for p in oldbank.glob('*.npz'):shutil.copy2(p,newbank/p.name)
(newbank/'manifest.json').write_text(json.dumps(meta,indent=2)+'\n')
files=[p for p in AD.rglob('*') if p.is_file() and '__pycache__' not in p.parts]+[REL/'wowrobo_training_runtime.py']+[p for p in newbank.iterdir() if p.is_file()]
x=dict(status='PREPARED',physics_backend='wowrobo-native-source-sdf',physics_signature=identity,timestep=.0005,substeps=80,control_dt=.04,original_factory_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),files={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(files)},original_source_unchanged=True,bank=str(newbank));(REL/'prepared.json').write_text(json.dumps(x,indent=2)+'\n');print(json.dumps({k:v for k,v in x.items() if k!='files'}),flush=True)
