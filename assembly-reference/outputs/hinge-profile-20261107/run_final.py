from pathlib import Path
import json,subprocess,shlex
h=Path(__file__).resolve().parent;cfg=json.loads((h.parents[1]/'operational/wowrobo-chimera-20261107/modal-live/target-v9.json').read_text());root=cfg['remote'];out=root+'/native-contact-optimization-20261107/hinge-profile-20261107';base=['srun','--jobid=3598717','--overlap','--exact','-n1','-c2','--mem=12G','--cpu-bind=cores','env','OMP_NUM_THREADS=1','OPENBLAS_NUM_THREADS=1','MKL_NUM_THREADS=1','SO101_SURFACE_MODE=1']
subprocess.run(['scp',str(h/'pairs.py'),'chimera-login:'+out+'/pairs.py'],check=True)
for tag,argv in [('v12-ppo',[out+'/ppo.py',root,out,'v12','ppo']),('pairs-plain',[out+'/pairs.py',root,out,'--plain']),('pairs-instrumented',[out+'/pairs.py',root,out])]:
 with (h/(tag+'.log')).open('w') as f:r=subprocess.run(['ssh','chimera-login',shlex.join(base+[cfg['python']]+argv)],stdout=f,stderr=subprocess.STDOUT)
 print(tag,r.returncode,flush=True)
 if r.returncode:print((h/(tag+'.log')).read_text()[-4500:]);raise SystemExit(r.returncode)
