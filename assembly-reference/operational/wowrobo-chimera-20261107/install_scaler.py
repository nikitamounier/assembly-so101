from pathlib import Path
import json,hashlib,subprocess,shlex
h=Path(__file__).parent;c=json.loads((h/'target.json').read_text());r=c['remote'];dest=c['script']
source=h.parents[2]/'assembly-so101/assembly_policy/compute/autoscale.py'
# h.parents[2] is the shared workspace.
for local,name in [(source,'autoscale.py'),(h/'worker.sbatch','worker.sbatch')]:subprocess.run(['scp','-o','ConnectTimeout=15',str(local),'chimera-login:'+dest+name],check=True)
cfg=dict(enabled=True,run_dir=r+'/run-v2',root=r+'/assembly-so101',python=c['python'],worker_script=dest+'worker.sbatch',worker_script_sha256=hashlib.sha256((h/'worker.sbatch').read_bytes()).hexdigest(),base_job_ids=json.loads((h/'cluster.json').read_text())['cpu_jobs'],max_chimera_cpus=1000,worker_slots=100,cloud_workers=0,cpus_per_worker=10,max_jobs_per_cycle=10,scope='DIAGNOSTIC_UNQUALIFIED_RESETS',progress_fresh_s=1200)
code=f"from pathlib import Path; import json; p=Path({r+'/autoscale-v2'!r}); p.mkdir(exist_ok=True); (p/'active-run.json').write_text({json.dumps(cfg,indent=2)!r})"
subprocess.run(['ssh','-o','ConnectTimeout=15','-o','ConnectionAttempts=1','chimera-login','srun --jobid=3077705 --overlap --exact --ntasks=1 --cpus-per-task=1 '+c['python']+' -c '+shlex.quote(code)],check=True)
(h/'autoscale-config.json').write_text(json.dumps(cfg,indent=2)+'\n')
print('Installed distinct run-bound scaler; no cycle executed yet')
