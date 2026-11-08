from pathlib import Path
import json,hashlib,tarfile,xml.etree.ElementTree as ET
HERE=Path(__file__).parent;ROOT=HERE.parents[3];C=ROOT/'robotics-assembly-3.0';selected={}
for name in ['minimum-v6','minimum-v6-source-fingers','minimum-v6-source-fingers-refined']:
 for p in (C/'outputs/equilibrium-reset-20261107'/name).glob('*.json'):
  d=json.loads(p.read_text())
  if d.get('minimum_validity_pass'):selected[p.stem]=p
assert len(selected)==24 and '21_upper_far_side' not in selected, 'Source set changed; create a new version instead of rebuilding this frozen24-state release'
files=set()
def add(p):
 p=Path(p)
 if p.is_dir():
  for f in p.rglob('*'):
   if f.is_file():files.add(f)
 else:files.add(p)
profile=C/'outputs/equilibrium-model-20261107/profile-centered-v6';add(profile)
for x in list(files):
 if x.suffix=='.xml':
  for e in ET.parse(x).iter():
   f=e.get('file')
   if f:add(Path(f) if Path(f).is_absolute() else x.parent/f)
for name,p in selected.items():
 d=json.loads(p.read_text());z=p.with_suffix('.npz');add(p);add(z)
 assert hashlib.sha256(z.read_bytes()).hexdigest()
add(C/'assets_local/wowrobo/minimum-validity-viewer-20261107-v1/stages/manifest.json')
add(C/'assets_local/wowrobo/minimum-validity-viewer-20261107-v1/completed')
# New kinematic registration helpers, not in the old snapshot.
add(C/'assembly_policy/so101/wowrobo_housing_registration.py')
(HERE/'selected.json').write_text(json.dumps({k:str(v.relative_to(ROOT)) for k,v in selected.items()},indent=2)+'\n')
for f in HERE.glob('*.py'):files.add(f)
files.add(HERE/'selected.json')
manifest={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(files)}
(HERE/'inputs.json').write_text(json.dumps(manifest,indent=2)+'\n');files.add(HERE/'inputs.json')
with tarfile.open(HERE/'overlay.tar.gz','w:gz',dereference=True,compresslevel=1) as archive:
 for p in sorted(files):archive.add(p,arcname=str(p.relative_to(ROOT)),recursive=False)
print(json.dumps(dict(selected=len(selected),files=len(files),bytes=(HERE/'overlay.tar.gz').stat().st_size)))
