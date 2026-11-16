import json,subprocess,shlex
from pathlib import Path
here=Path(__file__).parent;c=json.loads((here/'target.json').read_text());r=c['remote'];py=c['python'];s=c['script']
subprocess.run(['scp',str(here/'rebind.py'),'chimera-login:'+r+'-rebind.py'],check=True)
code=f"import tarfile,pathlib,hashlib,json; root=pathlib.Path({r!r}); tarfile.open({r+'.tar.gz'!r}).extractall(root,filter='data'); p=root/'robotics-assembly-3.0/operational/wowrobo-chimera-20261107/rebind.py'; p.write_bytes(pathlib.Path({r+'-rebind.py'!r}).read_bytes()); m=root/'source-manifest.json'; x=json.loads(m.read_text()); x[str(p.relative_to(root))]=hashlib.sha256(p.read_bytes()).hexdigest(); m.write_text(json.dumps(x,indent=2))"
subprocess.run(['ssh','chimera-login','srun --jobid=3077705 --overlap --exact --ntasks=1 --cpus-per-task=1 '+py+' -c '+shlex.quote(code)],check=True)
subprocess.run(['ssh','chimera-login','srun --jobid=3077705 --overlap --exact --ntasks=1 --cpus-per-task=1 '+c['env']+py+' '+s+'rebind.py'],check=True)
