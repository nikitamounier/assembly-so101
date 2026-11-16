from pathlib import Path
import subprocess,shlex
p=Path(__file__).parent;r='/large_storage/goodarzilab/nikitamounier/assembly/wowrobo-diagnostic-20261107T205745Z-minimum24-accepted-v4';d=r+'/native-contact-optimization-20261107/contact-patches-20261107';py='/large_storage/goodarzilab/nikitamounier/assembly/cube-trial-20261107-v6/.venv/bin/python'
for script,v,mode in [('calibrated_ppo.py','calibratedstiff','ppo'),('calibrated_ppo.py','baseline','ppo'),('long_load.py','baseline','reuse'),('long_load.py','calibratedstiff','reuse')]:
 args=[py,d+'/'+script,r,d,v]+([mode] if mode else [])
 with (p/(v+'-'+script+'.log')).open('w') as f:q=subprocess.run(['ssh','chimera-login',shlex.join(['srun','--jobid=3598717','--overlap','--exact','-n1','-c2','--mem=12G','env','OMP_NUM_THREADS=1','OPENBLAS_NUM_THREADS=1',*args])],stdout=f,stderr=subprocess.STDOUT)
 print(script,v,q.returncode,flush=True);q.check_returncode()
