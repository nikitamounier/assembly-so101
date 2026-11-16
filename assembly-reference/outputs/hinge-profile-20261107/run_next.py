from pathlib import Path
import json,subprocess,shlex
h=Path(__file__).resolve().parent
cfg=json.loads((h.parents[1]/'operational/wowrobo-chimera-20261107/modal-live/target-v9.json').read_text());root=cfg['remote'];out=root+'/native-contact-optimization-20261107/hinge-profile-20261107'
base=['srun','--jobid=3598717','--overlap','--exact','-n1','-c2','--mem=12G','--cpu-bind=cores']
for name in ['ppo.py','timers.py','clock.c']:subprocess.run(['scp',str(h/name),'chimera-login:'+out+'/'+name],check=True)
subprocess.run(['ssh','chimera-login',shlex.join(base+['cc','-O2','-shared','-fPIC',out+'/clock.c','-o',out+'/clock.so'])],check=True)
for script,mode,version in [('ppo.py','ppo','v10'),('ppo.py','ppo','v13'),('timers.py','panel','v10'),('timers.py','panel','v13')]:
 tag=version+'-'+('ppo' if mode=='ppo' else 'timed')
 argv=base+['env','OMP_NUM_THREADS=1','OPENBLAS_NUM_THREADS=1','MKL_NUM_THREADS=1','SO101_SURFACE_MODE=1',cfg['python'],out+'/'+script,root,out,version,mode]
 with (h/(tag+'.log')).open('w') as f:r=subprocess.run(['ssh','chimera-login',shlex.join(argv)],stdout=f,stderr=subprocess.STDOUT)
 print(tag,r.returncode,flush=True)
 if r.returncode:print((h/(tag+'.log')).read_text()[-4500:]);raise SystemExit(r.returncode)
