from pathlib import Path
import subprocess,shlex
r='/large_storage/goodarzilab/nikitamounier/assembly/wowrobo-diagnostic-20261107T205745Z-minimum24-accepted-v4';d=r+'/native-contact-optimization-20261107/contact-separation-20261107';py='/large_storage/goodarzilab/nikitamounier/assembly/cube-trial-20261107-v6/.venv/bin/python'
for script,tag,cpus in [('oracle.py','both',2),('packed.py','baseline',10),('packed.py','both',10)]:
 cmd=['srun','--jobid=3598717','--overlap','--exact','-n1','-c'+str(cpus),'--mem=12G','env','OMP_NUM_THREADS=1','OPENBLAS_NUM_THREADS=1',py,d+'/'+script,r,d,tag,'reuse']
 with open(Path(__file__).parent/(tag+'-'+script+'.log'),'w') as f:p=subprocess.run(['ssh','chimera-login',shlex.join(cmd)],stdout=f,stderr=subprocess.STDOUT)
 print(script,tag,p.returncode,flush=True)
 if p.returncode:break
