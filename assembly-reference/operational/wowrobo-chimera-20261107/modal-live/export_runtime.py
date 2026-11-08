"""Export the running frozen runtime, preserving absolute paths and model hashes."""
from pathlib import Path
import hashlib, json, subprocess, shlex, sys

HERE=Path(__file__).resolve().parent
TARGET=json.loads((Path(sys.argv[1]) if len(sys.argv)>1 else HERE.parent/'v3/target.json').read_text())
R=TARGET['remote']; DEST=R+'/modal-runtime.tar.gz'
CODE=r'''
from pathlib import Path
import hashlib,json,tarfile,xml.etree.ElementTree as ET
root=Path(ROOT); files=set(); pending=[]
def add(p):
 p=Path(p)
 if p.is_dir():
  for f in p.rglob('*'):
   if f.is_file() and '__pycache__' not in f.parts and f.suffix not in ('.so','.dylib','.pyc'): add(f)
 elif p.is_file() and p not in files: files.add(p);pending.append(p)
for folder in ['assembly-so101/assembly_policy','assembly-so101/plugin','assembly-so101/configs','assembly-so101/assets_local','robotics-assembly-3.0/assembly_policy/so101','robotics-assembly-3.0/outputs/equilibrium-model-20261107/profile-centered-v6']:add(root/folder)
add(root/'assembly-so101/requirements-worker.txt')
def refs(x):
 if isinstance(x,dict):
  for v in x.values():refs(v)
 elif isinstance(x,list):
  for v in x:refs(v)
 elif isinstance(x,str) and x.startswith(str(root)+'/'):add(Path(x))
while pending:
 p=pending.pop()
 if p.suffix=='.json':
  try:refs(json.loads(p.read_text()))
  except (UnicodeError,json.JSONDecodeError):pass
 if p.suffix=='.xml':
  for e in ET.parse(p).iter():
   if e.get('file'):add(Path(e.get('file')) if Path(e.get('file')).is_absolute() else p.parent/e.get('file'))
manifest={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(files)}
with tarfile.open(DEST,'w:gz',dereference=True,compresslevel=1) as a:
 for p in sorted(files):a.add(p,arcname=str(p).lstrip('/'),recursive=False)
Path(DEST+'.manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
print(json.dumps(dict(files=len(files),bytes=Path(DEST).stat().st_size,sha256=hashlib.sha256(Path(DEST).read_bytes()).hexdigest())),flush=True)
'''
if __name__=='__main__':
 code='ROOT='+repr(R)+';DEST='+repr(DEST)+'\n'+CODE
 command='srun --jobid=3598717 --overlap --exact --ntasks=1 --cpus-per-task=2 --mem=12G --cpu-bind=none '+TARGET['python']+' -c '+shlex.quote(code)
 p=subprocess.run(['ssh','-o','ConnectTimeout=15','chimera-login',command],capture_output=True,text=True)
 (HERE/'export-receipt.json').write_text(json.dumps(dict(returncode=p.returncode,stdout=p.stdout,stderr=p.stderr),indent=2)+'\n')
 print(p.stdout,p.stderr);p.check_returncode()
 subprocess.run(['scp','-o','ConnectTimeout=15','chimera-login:'+DEST,str(HERE/'runtime.tar.gz')],check=True)
 subprocess.run(['scp','-o','ConnectTimeout=15','chimera-login:'+DEST+'.manifest.json',str(HERE/'runtime-manifest.json')],check=True)
