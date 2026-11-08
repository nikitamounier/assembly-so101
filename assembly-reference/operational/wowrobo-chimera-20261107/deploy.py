from pathlib import Path
import datetime,json,subprocess,shlex
HERE=Path(__file__).parent.resolve()
stamp=datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ')
remote='/large_storage/goodarzilab/nikitamounier/assembly/wowrobo-diagnostic-'+stamp
py='/large_storage/goodarzilab/nikitamounier/assembly/cube-trial-20261107-v6/.venv/bin/python'
subprocess.run(['scp',str(HERE/'source.tar.gz'),'chimera-login:'+remote+'.tar.gz'],check=True)
code=f"import tarfile,pathlib; root=pathlib.Path({remote!r}); root.mkdir(exist_ok=False); tarfile.open({remote+'.tar.gz'!r}).extractall(root,filter='data')"
cmd='srun --jobid=3077705 --overlap --exact --ntasks=1 --cpus-per-task=1 '+shlex.quote(py)+' -c '+shlex.quote(code)
subprocess.run(['ssh','chimera-login',cmd],check=True)
env='env PYTHONPATH='+remote+'/assembly-so101 WOWROBO_CANONICAL_ROOT='+remote+'/robotics-assembly-3.0 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 SO101_SURFACE_MODE=1 '
script=remote+'/robotics-assembly-3.0/operational/wowrobo-chimera-20261107/'
cmd='srun --jobid=3077705 --overlap --exact --ntasks=1 --cpus-per-task=1 '+env+py+' '+script+'rebind.py'
(HERE/'target.json').write_text(json.dumps(dict(remote=remote,python=py,env=env,script=script),indent=2)+'\n')
subprocess.run(['ssh','chimera-login',cmd],check=True)
print('DEPLOYED '+remote,flush=True)
