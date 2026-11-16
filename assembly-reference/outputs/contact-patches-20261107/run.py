from pathlib import Path
import subprocess,shlex
r='/large_storage/goodarzilab/nikitamounier/assembly/wowrobo-diagnostic-20261107T205745Z-minimum24-accepted-v4';d=r+'/native-contact-optimization-20261107/contact-patches-20261107';py='/large_storage/goodarzilab/nikitamounier/assembly/cube-trial-20261107-v6/.venv/bin/python'
for variant in ['baseline','fused','winner','weld','combined']:
 cmd=['srun','--jobid=3598717','--overlap','--exact','-n1','-c2','--mem=12G','env','OMP_NUM_THREADS=1','OPENBLAS_NUM_THREADS=1',py,d+'/experiment.py',r,d,variant]
 with open(Path(__file__).parent/(variant+'.log'),'w') as f:p=subprocess.run(['ssh','chimera-login',shlex.join(cmd)],stdout=f,stderr=subprocess.STDOUT)
 print(variant,p.returncode,flush=True)
 if p.returncode:break
