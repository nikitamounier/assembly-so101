"""One temporary Modal learner + Modal worker + allocated Chimera worker."""
from __future__ import annotations
import datetime
import hashlib
import io
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import tarfile
import time

import modal

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
STAMP=datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ')
OUT=HERE/STAMP
OUT.mkdir(mode=0o700)
REMOTE_BASE='/large_storage/goodarzilab/nikitamounier/assembly/hybrid-smoke-20261107/'+STAMP
REMOTE_ROOT=REMOTE_BASE+'/runtime'
PYTHON='/large_storage/goodarzilab/nikitamounier/assembly/cube-trial-20261107-v6/.venv/bin/python'
REMOTE_HOST_ROOT='/opt/assembly'  # Runtime APIs resolve relative to the snapshot package, not this constant.


def make_bundle():
    paths=[ROOT/'assembly_policy/__init__.py']
    for package in ('cube_trial','so101','rl'):
        paths += list(sorted((ROOT/'assembly_policy'/package).rglob('*.py')))
    paths += [p for p in sorted((ROOT/'assets_local/so101').rglob('*')) if p.is_file() and not p.is_symlink()]
    paths += [ROOT/'configs/so101_scene.json']
    bank=ROOT/'data/cube_trial/banks_handoff_bridge_v8'
    paths += [bank/name for name in ('manifest.json','cube_handoff_train.npz','cube_handoff_eval.npz','reset-context.npz')]
    mapping={str(p.relative_to(ROOT)):p.read_bytes() for p in paths}
    # Same runtime entry bytes on each provider; ROOT follows package location.
    mapping['smoke_runtime.py']=(HERE/'smoke_runtime.py').read_bytes()
    mapping['requirements-worker.txt']=(ROOT/'requirements-worker.txt').read_bytes()
    hashes={name:hashlib.sha256(data).hexdigest() for name,data in mapping.items()}
    mapping['smoke-manifest.json']=json.dumps(hashes,sort_keys=True).encode()
    bundle=OUT/'runtime.tar.gz'
    with tarfile.open(bundle,'w:gz') as archive:
        for name,data in mapping.items():
            info=tarfile.TarInfo(name);info.size=len(data);info.mode=0o644
            archive.addfile(info,io.BytesIO(data))
    return bundle,hashes


def modal_learner(queue):
    from smoke_runtime import learner
    return learner(queue)


def modal_worker(ticket):
    from smoke_runtime import worker
    return worker(ticket,'modal')


def allocated_command(code):
    return ['ssh','-o','BatchMode=yes','-o','ConnectTimeout=15','chimera-login',
            'srun --jobid=3077705 --overlap --exact --ntasks=1 --cpus-per-task=1 '+shlex.quote(PYTHON)+' -c '+shlex.quote(code)]


def main():
    bundle,hashes=make_bundle()
    result=dict(started_utc=STAMP,scope='Isolated infrastructure smoke; no production changes',
                modal_cpu_limit=5,modal_container_limit=2,modal_timeout_s=600,
                chimera_allocation='3077705',chimera_cpus=1,
                bundle_sha256=hashlib.sha256(bundle.read_bytes()).hexdigest(),files=len(hashes))
    (OUT/'bundle-manifest.json').write_text(json.dumps(hashes,indent=2)+'\n')
    # Transfer contains source/robot meshes and fixture banks, never cloud credentials.
    r=subprocess.run(['scp',str(bundle),'chimera-login:'+REMOTE_BASE.replace('/'+STAMP,'')+'-upload-'+STAMP+'.tar.gz'],capture_output=True,text=True,timeout=60)
    r.check_returncode()
    uploaded=REMOTE_BASE.replace('/'+STAMP,'')+'-upload-'+STAMP+'.tar.gz'
    # Heavy extraction/source verification stays inside the allocated CPU step.
    setup=f"import tarfile,pathlib,json,hashlib; root=pathlib.Path({REMOTE_ROOT!r}); root.mkdir(parents=True,exist_ok=False); tarfile.open({uploaded!r}).extractall(root,filter='data'); manifest=json.loads((root/'smoke-manifest.json').read_text()); assert all(hashlib.sha256((root/k).read_bytes()).hexdigest()==v for k,v in manifest.items()); print('Frozen smoke runtime verified')"
    r=subprocess.run(allocated_command(setup),capture_output=True,text=True,timeout=60)
    (OUT/'chimera-setup.json').write_text(json.dumps(dict(returncode=r.returncode,stdout=r.stdout,stderr=r.stderr),indent=2)+'\n')
    r.check_returncode()
    image=(modal.Image.debian_slim(python_version='3.12')
           .apt_install('libgl1','openssl')
           .add_local_file(str(bundle),'/tmp/runtime.tar.gz',copy=True)
           .run_commands('mkdir -p /opt/assembly && tar -xzf /tmp/runtime.tar.gz -C /opt/assembly')
           .run_commands('python -m pip install -r /opt/assembly/requirements-worker.txt')
           .pip_install('torch==2.11.0',index_url='https://download.pytorch.org/whl/cpu')
           .env({'PYTHONPATH':'/opt/assembly','OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','MKL_NUM_THREADS':'1'}))
    app=modal.App('assembly-hybrid-smoke')
    learner_fn=app.function(image=image,cpu=(4,4),memory=(8192,8192),timeout=600,max_containers=1,
                            serialized=True,include_source=False)(modal_learner)
    worker_fn=app.function(image=image,cpu=(1,1),memory=(2048,2048),timeout=300,max_containers=1,
                           serialized=True,include_source=False)(modal_worker)
    calls=[]
    remote_proc=None
    remote_log=None
    try:
        with modal.enable_output(), app.run(), modal.Queue.ephemeral() as queue:
            result['modal_app_id']=app.app_id
            print(json.dumps(dict(event='app_started',app_id=app.app_id)),flush=True)
            call=learner_fn.spawn(queue);calls.append(call)
            ticket=queue.get(timeout=180)
            # Ticket stays in process memory/SSH stdin; never emitted into logs/argv.
            result['transport']='Certificate-pinned TLS + mutual per-run HMAC over temporary Modal TCP tunnel'
            wcall=worker_fn.spawn(ticket);calls.append(wcall)
            bootstrap=f"import pathlib,sys,json; root=pathlib.Path({REMOTE_ROOT!r});sys.path.insert(0,str(root));from smoke_runtime import worker; t=json.loads(sys.stdin.read()); print(json.dumps(worker(t,'chimera')),flush=True)"
            remote_log=(OUT/'chimera-worker.log').open('w')
            remote_proc=subprocess.Popen(allocated_command(bootstrap),stdin=subprocess.PIPE,stdout=remote_log,stderr=subprocess.STDOUT,text=True)
            remote_proc.stdin.write(json.dumps(ticket));remote_proc.stdin.close()
            started=time.monotonic()
            result['learner']=call.get(timeout=240)
            result['modal_worker']=wcall.get(timeout=30)
            remote_code=remote_proc.wait(timeout=30)
            remote_log.flush()
            result['chimera_worker_exit']=remote_code
            assert remote_code==0,(OUT/'chimera-worker.log').read_text()
            lines=(OUT/'chimera-worker.log').read_text().splitlines()
            result['chimera_worker']=json.loads(lines[-1])
            for name in ('modal_worker','chimera_worker'):
                assert result[name]['model_signature']==result['learner']['model_signature']
                assert result[name]['bundle_sha256']==result['learner']['bundle_sha256']
            assert set(result['learner']['hosts'])=={result['modal_worker']['host'],result['chimera_worker']['host']}
            result['joint_execution_elapsed_s']=time.monotonic()-started
            result['status']='PASS'
    except BaseException as exc:
        result['status']='FAIL'
        result['error']=repr(exc)
        raise
    finally:
        for call in calls:
            try:call.cancel()
            except Exception:pass
        if remote_proc is not None and remote_proc.poll() is None:
            remote_proc.terminate()
            try:remote_proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                remote_proc.kill();remote_proc.wait()
        if remote_log:remote_log.close()
        result['finished_utc']=datetime.datetime.now(datetime.timezone.utc).isoformat()
        (OUT/'result.json').write_text(json.dumps(result,indent=2)+'\n')
        print(json.dumps(dict(status=result.get('status'),result=str(OUT/'result.json'),app_id=result.get('modal_app_id'))),flush=True)


if __name__=='__main__':main()
