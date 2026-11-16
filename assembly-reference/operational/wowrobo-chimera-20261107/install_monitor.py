from pathlib import Path
import json,subprocess,shlex
h=Path(__file__).parent;c=json.loads((h/'target.json').read_text());r=c['remote']
subprocess.run(['scp',str(h/'monitor.py'),'chimera-login:'+c['script']+'monitor.py'],check=True)
code=f'''from pathlib import Path
import hashlib,json
root=Path({r!r});d=json.loads((root/'source-manifest.json').read_text());files={{f:hashlib.sha256((root/f).read_bytes()).hexdigest() for f in d if '/operational/' not in f and Path(f).suffix in ('.py','.xml','.json','.npz')}};(root/'active-inputs.json').write_text(json.dumps(files,indent=2)+'\\n');print('Frozen active inputs',len(files))
'''
subprocess.run(['ssh','chimera-login','srun --jobid=3077705 --overlap --exact --ntasks=1 --cpus-per-task=1 '+c['python']+' -c '+shlex.quote(code)],check=True)
