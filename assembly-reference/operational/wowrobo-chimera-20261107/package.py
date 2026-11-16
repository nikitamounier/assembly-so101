from pathlib import Path
import tarfile,hashlib,json,xml.etree.ElementTree as ET
ROOT=Path(__file__).resolve().parents[3]
OUT=Path(__file__).parent
files=set()
def add(p):
 p=p.resolve()
 if p.is_dir():
  for f in p.rglob('*'):
   if f.is_file() and '__pycache__' not in f.parts and f.suffix not in ('.so','.dylib','.pyc'):files.add(f)
 else:files.add(p)
for base in ['assembly-so101/assembly_policy','robotics-assembly-3.0/assembly_policy/so101']:
 for p in (ROOT/base).rglob('*.py'):add(p)
for base in ['assembly-so101/configs','robotics-assembly-3.0/configs']:
 for p in (ROOT/base).glob('*.json'):add(p)
for n in ['assemblenist_sdf.cc','so101_grid_bounds.h','so101_surface_tree.h','so101_contact_search.c','build.sh','build_contact_search.sh']:add(ROOT/'assembly-so101/plugin'/n)
canon=ROOT/'robotics-assembly-3.0'
for n in ['physical-resets-repaired-cover-support-v2','training-debug-v1']:add(canon/'assets_local/wowrobo'/n)
add(canon/'assets_local/wowrobo/randomization-bank-repaired-cover-support-v2/scene.xml')
for p in list(files):
 if p.suffix=='.xml':
  for e in ET.parse(p).iter():
   f=e.get('file')
   if f:
    path=Path(f) if Path(f).is_absolute() else p.parent/f
    add(path)
for p in OUT.glob('*.py'):add(p)
add(OUT/'cluster.json')
manifest={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(files)}
(OUT/'source-manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
with tarfile.open(OUT/'source.tar.gz','w:gz',dereference=True) as t:
 for p in sorted(files):t.add(p,arcname=str(p.relative_to(ROOT)),recursive=False)
 t.add(OUT/'source-manifest.json',arcname='source-manifest.json')
print(json.dumps(dict(files=len(files),bytes=(OUT/'source.tar.gz').stat().st_size)))
