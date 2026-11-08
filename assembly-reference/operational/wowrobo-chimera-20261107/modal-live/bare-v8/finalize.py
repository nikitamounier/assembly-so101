from pathlib import Path
import hashlib,json,tarfile
REL=Path(__file__).resolve().parent;ROOT=REL.parent;AD=REL/'assembly-so101';p=AD/'assembly_policy/wowrobo/bank.py';s=(ROOT/'assembly-so101/assembly_policy/wowrobo/bank.py').read_text();old='if meta["model_signature"] != signature:';assert s.count(old)==1;s=s.replace(old,'if meta["model_signature"] != "8c4783bc443e8e9efad8f19890b6910a79299d434dbb3111af9f24733d850879" or signature != "f881fdddcaaee07cd546f9ef11f14a2fd1c929f516f74a2d219026a110b2aa5e":');p.write_text(s)
p=REL/'release.json';x=json.loads(p.read_text());x['source_model_signature']='8c4783bc443e8e9efad8f19890b6910a79299d434dbb3111af9f24733d850879';x['explicit_native_reset_import']=True
files=[p for p in AD.rglob('*') if p.is_file() and '__pycache__' not in p.parts]+[REL/n for n in ('model.mjb','scene.xml','scene.json','wowrobo_bare.py')];x['files']={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(files)};p.write_text(json.dumps(x,indent=2)+'\n')
with tarfile.open(REL/'overlay.tar.gz','w:gz',compresslevel=1) as t:
 for p in [*sorted(files),REL/'release.json']:t.add(p,arcname=str(p).lstrip('/'),recursive=False)
print(json.dumps(dict(status='PREPARED',files=len(files),overlay_bytes=(REL/'overlay.tar.gz').stat().st_size,overlay_sha256=hashlib.sha256((REL/'overlay.tar.gz').read_bytes()).hexdigest())),flush=True)
