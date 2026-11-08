from pathlib import Path
import json,shlex,subprocess,sys
HERE=Path(__file__).resolve().parent
cfg=json.loads((HERE.parents[1]/'operational/wowrobo-chimera-20261107/modal-live/target-v9.json').read_text());root=cfg['remote'];exp=root+'/native-contact-optimization-20261107/swept'
for variant in sys.argv[2:] or ['baseline','candidate']:
 cmd=['srun','--jobid=3598717','--overlap','--exact','-n1','-c2','--mem=12G','--cpu-bind=cores']+shlex.split(cfg['env'])+[cfg['python'],exp+'/probe.py',root,exp,variant,'--panel',sys.argv[1],'--no-profile']
 with (HERE/(variant+'-'+sys.argv[1]+'.log')).open('w') as f:
  r=subprocess.run(['ssh','chimera-login',shlex.join(cmd)],stdout=f,stderr=subprocess.STDOUT)
 print(variant,sys.argv[1],r.returncode,flush=True)
 if r.returncode:print((HERE/(variant+'-'+sys.argv[1]+'.log')).read_text()[-5000:]);sys.exit(r.returncode)
