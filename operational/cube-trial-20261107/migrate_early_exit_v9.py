"""Human-authorized termination change, exact latest optimizer/policy/sampler continuation."""
import copy,hashlib,json,os,runpy,shutil,sys
from pathlib import Path
import torch
ROOT=Path('/large_storage/goodarzilab/nikitamounier/assembly/cube-trial-20261107-v6');sys.path.insert(0,str(ROOT));assert os.environ['SLURM_JOB_ID']=='3077705'
ops=ROOT/'operational/cube-trial-20261107';art=ROOT/'artifacts_local/chimera';cfg=json.loads((ops/'cluster.json').read_text());r=json.loads((art/'controller-adoption.json').read_text());assert not r['workers'] and r['learner'] is None
assert (Path('/proc')/str(r['old_controller_pid'])/'stat').read_text().split()[2]=='T'
hp=art/'source-hashes.json';assert hashlib.sha256(hp.read_bytes()).hexdigest()=='0369e63e2d5bc64aef32edb82cb1c82ce0ada9961fedfa1f5f00b8370c6ef980';h=json.loads(hp.read_text());assert all(hashlib.sha256((ROOT/k).read_bytes()).hexdigest()==v for k,v in h.items())
run=ROOT/cfg['run'];path=run/'checkpoint.pt';archive=run/'checkpoint.before-early-exit-v9.pt';assert not archive.exists();c=torch.load(path,map_location='cpu',weights_only=False);assert c['it']==r['graceful_checkpoint_iteration']==9940
assert c['args']['env']['horizon_s']==15.;assert c['args']['reward_contract_version']=='placement_v4_table_only';assert cfg['controller_lock']=='controller-generation11.lock'
shutil.copy2(path,archive);shutil.copy2(ROOT/'assembly_policy/cube_trial/env.py',art/'env-before-early-exit-v9.py');shutil.copy2(ops/'cluster.json',ops/'cluster-before-early-exit-v9.json');shutil.copy2(hp,art/'source-hashes-before-early-exit-v9.json')
shutil.copy2(art/'env-early-exit-v9.py',ROOT/'assembly_policy/cube_trial/env.py');shutil.copy2(art/'test_cube_episode_termination.py',ROOT/'tests/test_cube_episode_termination.py')
cfg['controller_lock']='controller-generation12.lock';(ops/'cluster.json').write_text(json.dumps(cfg,indent=2)+'\n')
from assembly_policy.cube_trial.env import EnvConfig
from assembly_policy.cube_trial.resets import signature,directory_signature
assert EnvConfig().horizon_s==10. and signature()==c['args']['physics_signature'];assert directory_signature(ROOT/cfg['bank'])==c['args']['bank_directory_signature']
n=copy.deepcopy(c);n['args']['env']['horizon_s']=10.;n['args']['environment_signature']=hashlib.sha256((ROOT/'assembly_policy/cube_trial/env.py').read_bytes()).hexdigest();n['episode_migration']=dict(reason='Human requested immediate terminal failure for off-target table contact after pickup,10s timeout',resume_iteration=c['it'],old_horizon_s=15.,horizon_s=10.,failure_is_terminal_not_timeout=True,reward_formula_unchanged=True)
tmp=run/'checkpoint.early-exit-v9.tmp';torch.save(n,tmp);v=torch.load(tmp,map_location='cpu',weights_only=False);equal=runpy.run_path(str(ops/'migrate_continuity_v3.py'))['equal']
for k in ['model','opt','sampler','lr','it','transitions']:assert equal(c[k],v[k]),k
os.replace(tmp,path)
for name in ['assembly_policy/cube_trial/env.py','operational/cube-trial-20261107/cluster.json','tests/test_cube_episode_termination.py']:h[name]=hashlib.sha256((ROOT/name).read_bytes()).hexdigest()
hp.write_text(json.dumps(h,indent=2)+'\n');audit=dict(iteration=c['it'],transitions=c['transitions'],model_optimizer_sampler_lr_counters_exact=True,checkpoint_archive_sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),new_checkpoint_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),source_manifest_sha256=hashlib.sha256(hp.read_bytes()).hexdigest(),environment_sha256=n['args']['environment_signature'],cpu_jobs=cfg['cpu_jobs'],gpu_job=cfg['gpu_job'],**n['episode_migration']);(art/'early-exit-v9-migration.json').write_text(json.dumps(audit,indent=2)+'\n');print(json.dumps(audit))
