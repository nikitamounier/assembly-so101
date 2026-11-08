from pathlib import Path
import json,subprocess,shlex,datetime
HERE=Path(__file__).parent;OLD=json.loads((HERE.parent/'target.json').read_text());R=OLD['remote']+'-minimum24';C=R+'/robotics-assembly-3.0';S=R+'/assembly-so101';ENV='env PYTHONPATH='+S+' WOWROBO_CANONICAL_ROOT='+C+' OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 SO101_SURFACE_MODE=1 '
(HERE/'target.json').write_text(json.dumps(dict(remote=R,python=OLD['python'],env=ENV,script=C+'/operational/wowrobo-chimera-20261107/',parent=OLD['remote']),indent=2)+'\n')
subprocess.run(['scp','-o','ConnectTimeout=15',str(HERE/'overlay.tar.gz'),'chimera-login:'+R+'-overlay.tar.gz'],check=True)
code=f'''from pathlib import Path
import shutil,tarfile
old=Path({OLD['remote']!r});new=Path({R!r});new.mkdir(exist_ok=False)
for name in ['assembly-so101','robotics-assembly-3.0']:shutil.copytree(old/name,new/name,symlinks=False)
# Rebind the copied frozen snapshot; active run and all its files remain untouched.
for base in ['assembly-so101','robotics-assembly-3.0']:
 for p in (new/base).rglob('*'):
  if p.is_file() and p.suffix in ('.json','.xml','.py','.sh','.sbatch'):
   try:s=p.read_text()
   except UnicodeDecodeError:continue
   p.write_text(s.replace(str(old),str(new)))
tarfile.open({R+'-overlay.tar.gz'!r}).extractall(new,filter='data')
print('Independent matched-model release staged')
'''
subprocess.run(['ssh','-o','ConnectTimeout=15','chimera-login','srun --jobid=3598717 --overlap --exact --ntasks=1 --cpus-per-task=2 --mem=12G --cpu-bind=none '+OLD['python']+' -c '+shlex.quote(code)],check=True)
subprocess.run(['ssh','-o','ConnectTimeout=15','chimera-login','srun --jobid=3598717 --overlap --exact --ntasks=1 --cpus-per-task=2 --mem=12G --cpu-bind=none '+ENV+OLD['python']+' '+C+'/operational/wowrobo-chimera-20261107/v3/prepare.py'],check=True)
