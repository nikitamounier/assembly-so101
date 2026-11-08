from pathlib import Path
import json,tarfile,subprocess,shlex
HERE=Path(__file__).resolve().parent;C=HERE.parents[2];BASE=C.parent
old=json.loads((HERE.parent/'v3/target.json').read_text());R=old['remote']+'-accepted-v4';PY=old['python']
snap=sorted(HERE.glob('states-*'))[-1]
files=[C/'assembly_policy/so101'/n for n in ['wowrobo_training_runtime.py','wowrobo_rollout_friction.py','wowrobo_rollout_friction.cc','wowrobo_friction.py']]
files += [HERE/'prepare_release.py',HERE/'validate_release.py']
archive=HERE/'release-overlay.tar.gz'
with tarfile.open(archive,'w:gz') as a:
 for p in files:a.add(p,arcname=str(p.relative_to(BASE)))
 a.add(snap,arcname=str(HERE.relative_to(BASE)/'accepted-states'))
subprocess.run(['scp','-o','ConnectTimeout=15',str(archive),'chimera-login:'+R+'-overlay.tar.gz'],check=True)
code=f'''from pathlib import Path
import shutil,tarfile
old=Path({old['remote']!r});root=Path({R!r});root.mkdir(exist_ok=False)
for name in ['assembly-so101','robotics-assembly-3.0']:shutil.copytree(old/name,root/name,symlinks=False)
for base in ['assembly-so101','robotics-assembly-3.0']:
 for p in (root/base).rglob('*'):
  if p.is_file() and p.suffix in ('.py','.json','.xml','.sh','.sbatch'):
   try:s=p.read_text()
   except UnicodeError:continue
   p.write_text(s.replace(str(old),str(root)))
tarfile.open({R+'-overlay.tar.gz'!r}).extractall(root,filter='data')
print('Isolated v4 clone staged; live run unchanged',flush=True)
'''
env='env PYTHONPATH='+R+'/assembly-so101 WOWROBO_CANONICAL_ROOT='+R+'/robotics-assembly-3.0 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 SO101_SURFACE_MODE=1 '
def run(command):
 p=subprocess.run(['ssh','-o','ConnectTimeout=15','chimera-login','srun --jobid=3598717 --overlap --exact --ntasks=1 --cpus-per-task=2 --mem=12G --cpu-bind=none '+command],capture_output=True,text=True)
 print(p.stdout,p.stderr,flush=True);p.check_returncode()
run(PY+' -c '+shlex.quote(code))
(HERE/'release-target.json').write_text(json.dumps(dict(remote=R,python=PY,env=env,script=R+'/robotics-assembly-3.0/operational/wowrobo-chimera-20261107/modal-live/'),indent=2)+'\n')
run(env+PY+' '+R+'/robotics-assembly-3.0/operational/wowrobo-chimera-20261107/modal-live/prepare_release.py')
run(env+PY+' '+R+'/robotics-assembly-3.0/operational/wowrobo-chimera-20261107/modal-live/validate_release.py')
