from pathlib import Path
import subprocess,shlex
r='/large_storage/goodarzilab/nikitamounier/assembly/wowrobo-diagnostic-20261107T205745Z-minimum24-accepted-v4';d=r+'/native-contact-optimization-20261107/contact-separation-20261107';py='/large_storage/goodarzilab/nikitamounier/assembly/cube-trial-20261107-v6/.venv/bin/python'
for tag,mode in [('baseline-repeat','reuse'),('both','ppo'),('baseline','ppo'),('baseline-stats','panel'),('cache-stats','panel')]:
 if tag=='baseline-repeat':
  code='from pathlib import Path;import shutil;p=Path('+repr(d)+');shutil.copytree(p/"baseline/native",p/"baseline-repeat/native",dirs_exist_ok=True)'
  subprocess.run(['ssh','chimera-login',shlex.join(['srun','--jobid=3598717','--overlap','--exact','-n1','-c1',py,'-c',code])],check=True)
 cmd=['srun','--jobid=3598717','--overlap','--exact','-n1','-c2','--mem=12G','env','OMP_NUM_THREADS=1','OPENBLAS_NUM_THREADS=1',py,d+'/experiment.py',r,d,tag,mode]
 with open(Path(__file__).parent/(tag+'-'+mode+'.log'),'w') as f:p=subprocess.run(['ssh','chimera-login',shlex.join(cmd)],stdout=f,stderr=subprocess.STDOUT)
 print(tag,mode,p.returncode,flush=True)
 if p.returncode:break
