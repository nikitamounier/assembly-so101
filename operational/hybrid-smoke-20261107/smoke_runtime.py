"""Bounded real-worker TLS smoke on a frozen stock-MuJoCo SO101 fixture."""
from __future__ import annotations
import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import threading
import time

ROOT = Path(__file__).resolve().parent


def identity():
    import mujoco
    import numpy
    from assembly_policy.cube_trial.resets import signature
    return dict(host=socket.gethostname(), mujoco=mujoco.__version__, numpy=numpy.__version__,
                model_signature=signature(), bundle_sha256=hashlib.sha256((ROOT/'smoke-manifest.json').read_bytes()).hexdigest())


def worker(ticket, provider):
    from assembly_policy.rl.transport import connect
    from assembly_policy.cube_trial.remote import _serve
    started = time.monotonic()
    conn = connect(ticket)
    try:
        _serve(conn, 1)
    finally:
        conn.close()
    return dict(provider=provider, elapsed_s=time.monotonic()-started, **identity())


def learner(queue):
    import modal
    import numpy as np
    import torch
    from assembly_policy.cube_trial.remote import RemoteVecEnv
    from assembly_policy.cube_trial.vecenv import LocalVecEnv
    from assembly_policy.cube_trial.env import EnvConfig
    from assembly_policy.cube_trial.resets import signature
    from assembly_policy.cube_trial.train import Trainer, parse
    torch.set_num_threads(4)
    os.chdir(ROOT)
    run = Path('/tmp/hybrid-smoke')
    run.mkdir(mode=0o700)
    subprocess.run(['openssl','req','-x509','-newkey','rsa:2048','-nodes','-days','1',
                    '-subj','/CN=assembly-so101','-keyout',str(run/'key.pem'),'-out',str(run/'cert.pem')],
                   check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    (run/'key.pem').chmod(0o600)
    N = 8
    bankdir=ROOT/'data/cube_trial/banks_handoff_bridge_v8'
    with np.load(bankdir/'cube_handoff_train.npz') as z:
        state=z['states'][0].copy()
    def provider(n, *_):
        return np.zeros(n,np.int64), np.repeat(state[None],n,axis=0),np.zeros(n,np.int64)
    cfg=EnvConfig(nthread=1)
    initial_t=np.full(N,374,np.int64)
    remote=None
    local=None
    rows=[]
    with modal.forward(8701, unencrypted=True) as tunnel:
        # The raw forwarding socket carries our own certificate-pinned TLS and
        # mutual per-run HMAC; no plaintext worker protocol is exposed.
        host,port=tunnel.tcp_socket
        publishing_errors=[]
        def publish():
            try:
                deadline=time.monotonic()+90
                while not (run/'server.json').exists():
                    if time.monotonic()>deadline: raise TimeoutError('No learner ticket')
                    time.sleep(.05)
                ticket=json.loads((run/'server.json').read_text())
                ticket.update(host=host,port=port)
                queue.put(ticket)
            except BaseException as exc:
                publishing_errors.append(repr(exc))
        publishing=threading.Thread(target=publish,daemon=True)
        publishing.start()
        try:
            remote=RemoteVecEnv(run,['cube_handoff'],N,cfg,provider,2,t0=initial_t,
                                port=8701,bind_host='0.0.0.0',accept_timeout=120,min_frac=1,
                                init_wait=120,tls_cert=run/'cert.pem',tls_key=run/'key.pem',
                                model_signature=signature())
            publishing.join(1)
            assert not publishing_errors,publishing_errors
            deadline=time.monotonic()+120
            while not remote.alive_mask().all():
                if time.monotonic()>deadline: raise TimeoutError('Both workers not ready')
                remote.observe()
                time.sleep(.05)
            hosts=list(remote.hosts)
            assert len(hosts)==2 and len(set(hosts))==2,hosts
            local=LocalVecEnv(['cube_handoff'],N,cfg,provider,initial_t)
            max_errors={}
            resets=0
            for _ in range(4):
                for name,value in local.observe().items():
                    actual=remote.observe()[name]
                    max_errors[name]=max(max_errors.get(name,0),float(np.max(np.abs(actual-value))))
                    np.testing.assert_allclose(actual,value,atol=2e-5,rtol=1e-5)
                actions=np.zeros((N,12),np.float32)
                a,b=remote.step(actions),local.step(actions)
                np.testing.assert_allclose(a[0],b[0],atol=1e-6,rtol=1e-5)
                np.testing.assert_array_equal(a[1],b[1])
                for key in ('valid','success','safety','timeout','ep_len'):
                    np.testing.assert_array_equal(a[2][key],b[2][key])
                assert a[2]['valid'].all()
                resets+=int(a[1].sum())
            assert resets>=N,'Reset/history path was not exercised'
            for name,value in local.observe().items():
                np.testing.assert_allclose(remote.observe()[name],value,atol=2e-5,rtol=1e-5)
            local.close();local=None
            args=parse(['--run-dir',str(run/'ppo'),'--bank-dir',str(bankdir),'--envs',str(N),
                        '--steps','32','--iters','2','--epochs','1','--minibatches','2',
                        '--nthread','1','--device','cpu','--no-eval','--seed','1047'])
            trainer=Trainer(args)
            trainer.env.close()
            trainer.env=remote
            remote.provider=trainer._provider
            parameters_before={k:v.detach().clone() for k,v in trainer.model.named_parameters()}
            trainer.train()
            rows=[json.loads(x) for x in (run/'ppo/log.jsonl').read_text().splitlines()]
            assert len(rows)==2 and [x['valid_transitions'] for x in rows]==[256,256],rows
            assert all(np.isfinite(x[k]) for x in rows for k in ('pi','vf','ent','kl','fps'))
            changed=any(not torch.equal(v,dict(trainer.model.named_parameters())[k]) for k,v in parameters_before.items())
            assert changed,'No parameter update'
            assert remote.alive_mask().all(),'Lost a provider during PPO'
            profiles=remote.profile()
            checkpoint=run/'ppo/checkpoint.pt'
            ck=torch.load(checkpoint,map_location='cpu',weights_only=False)
            class RestoreProbe(Trainer):
                def _maybe_resume(self):
                    super()._maybe_resume()
                    import copy
                    # Capture before new environment creation legitimately draws resets.
                    self.sampler_at_restore=copy.deepcopy(self.sampler.state_dict())
            restored=RestoreProbe(args)
            try:
                assert restored.it==2 and restored.transitions==512
                assert all(torch.equal(v,restored.model.state_dict()[k]) for k,v in ck['model'].items())
                assert restored.opt.state_dict()['param_groups']==ck['opt']['param_groups']
                def equal(a,b):
                    if isinstance(a,torch.Tensor):return torch.equal(a,b)
                    if isinstance(a,np.ndarray):return np.array_equal(a,b)
                    if isinstance(a,dict):return a.keys()==b.keys() and all(equal(a[k],b[k]) for k in a)
                    if isinstance(a,(tuple,list)):return len(a)==len(b) and all(equal(x,y) for x,y in zip(a,b))
                    return a==b
                assert equal(restored.opt.state_dict(),ck['opt'])
                assert equal(restored.sampler_at_restore,ck['sampler'])
            finally:
                restored.env.close()
            return dict(status='PASS',scope='Worker interoperability on frozen cube fixture, CPU PPO learner; not new-CAD or GPU qualification',
                        **identity(),hosts=hosts,active_envs=int(remote.alive_mask().sum()),
                        parity_steps=4,parity_valid_transitions=32,parity_resets=resets,
                        parity_max_observation_errors=max_errors,rollout_profile=profiles,
                        ppo_updates=rows,parameters_changed=changed,checkpoint_resume_exact=True,
                        checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                        checkpoint_bytes=checkpoint.stat().st_size)
        finally:
            if remote is not None:remote.close()
            if local is not None:local.close()
