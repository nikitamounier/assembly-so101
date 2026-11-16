import subprocess,sys
from pathlib import Path
r=Path('/large_storage/goodarzilab/nikitamounier/assembly/wowrobo-diagnostic-20261107T205745Z-minimum24-accepted-v4')
b=r/'native-contact-optimization-20261107';o=b/'contact-continuation-20261107'
for tag,mode in [('baseline','panel'),('candidate','ppo'),('baseline','ppo')]:
 subprocess.run([sys.executable,str(b/'experiment.py'),str(r),str(o),tag,mode],check=True)
