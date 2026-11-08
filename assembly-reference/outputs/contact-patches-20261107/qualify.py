from pathlib import Path
import subprocess,shlex,json
r='/large_storage/goodarzilab/nikitamounier/assembly/wowrobo-diagnostic-20261107T205745Z-minimum24-accepted-v4';d=r+'/native-contact-optimization-20261107/contact-patches-20261107';py='/large_storage/goodarzilab/nikitamounier/assembly/cube-trial-20261107-v6/.venv/bin/python';out=Path(__file__).parent
for v in ['baseline','weld']:
 cmd=['srun','--jobid=3598717','--overlap','--exact','-n1','-c2','--mem=12G','env','OMP_NUM_THREADS=1','OPENBLAS_NUM_THREADS=1',py,d+'/load.py',r,d,v,'reuse']
 with (out/(v+'-load.log')).open('w') as f:p=subprocess.run(['ssh','chimera-login',shlex.join(cmd)],stdout=f,stderr=subprocess.STDOUT)
 print(v,'load',p.returncode,flush=True);p.check_returncode()
 (out/v).mkdir(exist_ok=True);subprocess.run(['scp','chimera-login:'+d+'/'+v+'/load.json',str(out/v/'load.json')],check=True)
a=json.loads((out/'baseline/load.json').read_text());b=json.loads((out/'weld/load.json').read_text());bad=[]
for x,y in zip(a['rows'],b['rows']):
 if y['max_attachment_translation_mm']>x['max_attachment_translation_mm']+.2 or y['max_attachment_rotation_deg']>x['max_attachment_rotation_deg']+.5:bad.append((x['task'],x['case'],x['world']))
(out/'load-admission.json').write_text(json.dumps(dict(passed=not bad,failures=bad,tolerance_extra_mm=.2,tolerance_extra_deg=.5),indent=2)+'\n')
print('load_admission',not bad,bad,flush=True)
if bad:raise SystemExit(0)
for v in ['weld','baseline']:
 cmd=['srun','--jobid=3598717','--overlap','--exact','-n1','-c2','--mem=12G','env','OMP_NUM_THREADS=1','OPENBLAS_NUM_THREADS=1',py,d+'/experiment.py',r,d,v,'ppo']
 with (out/(v+'-ppo.log')).open('w') as f:p=subprocess.run(['ssh','chimera-login',shlex.join(cmd)],stdout=f,stderr=subprocess.STDOUT)
 print(v,'ppo',p.returncode,flush=True);p.check_returncode()
