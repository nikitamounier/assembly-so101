from common import *
import numpy as np
import torch
import subprocess

backend('baseline')
mode=sys.argv[1]
if mode=='worker':
    from assembly_policy.rl.remote import _serve
    from assembly_policy.rl.transport import connect
    run=OUT/'ppo'
    for _ in range(600):
        if (run/'server.json').exists():break
        time.sleep(.1)
    info=json.loads((run/'server.json').read_text())
    wall=Sampler(OUT/f'worker-{sys.argv[2]}.speedscope.json')
    # Capture exactly the served lifetime, including initialization and idle barriers.
    wall.start()
    try:_serve(connect(info),1)
    finally:wall.finish();write(OUT/f'worker-{sys.argv[2]}-hardware.json',hardware())
else:
    from assembly_policy import train
    torch.set_num_threads(1)
    bank=CANON/'assets_local/wowrobo/training-accepted-v6'
    a=train.parse(['--suite','wowrobo','--wowrobo-debug-resets','--bank-per-family','0','--bank-dir',str(bank),'--run-dir',str(OUT/'ppo'),'--envs','2','--remote-workers','2','--steps','32','--nthread','1','--device','cuda','--iters','26','--micro-batch','1024','--no-eval','--cost-coef','.00002','--action-rate-coef','0','--eps-q','.01','--seed','20261107','--accept-timeout','180','--min-conn-frac','1','--worker-init-wait','180'])
    # Resume original policy, Adam, all SGS/RNG/counters through the unmodified loader.
    tr=train.Trainer(a)
    assert tr.it==25 and tr.transitions==966743
    assert tr.args['model_signature']==SIGNATURE and tr.pts.n_pts==448
    spans=[]
    for name in ['collect','update','save']:
        original=getattr(tr,name)
        def wrapped(*args,_fn=original,_name=name,**kwargs):
            t=time.perf_counter();result=_fn(*args,**kwargs);spans.append(dict(name=_name,start=t,duration=time.perf_counter()-t));return result
        setattr(tr,name,wrapped)
    steps=[];original_step=tr.env.step
    def step(action):
        t=time.perf_counter();result=original_step(action);steps.append(dict(seconds=time.perf_counter()-t,valid=int(result[2]['valid'].sum())));return result
    tr.env.step=step
    wall=Sampler(OUT/'learner.speedscope.json');wall.start();t=time.perf_counter()
    try:tr.train()
    finally:
        elapsed=time.perf_counter()-t;wall.finish();tr.env.close()
    rec=json.loads((OUT/'ppo/log.jsonl').read_text().splitlines()[-1])
    write(OUT/'ppo-result.json',dict(elapsed_s=elapsed,record=rec,spans=spans,steps=steps,hardware=hardware(),points=tr.pts.n_pts,qualified=tr.it==26 and rec['valid_transitions']==64 and len(steps)==32 and all(s['valid']==2 for s in steps),scope='Bounded two-world/two-real-worker full32-step PPO update; original frozen policy/Adam/SGS continuation. Small batch is diagnostic, not production throughput.'))
