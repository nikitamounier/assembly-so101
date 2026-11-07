import copy,hashlib,json,os,shutil,runpy,sys
from pathlib import Path
import torch
ROOT=Path('/large_storage/goodarzilab/nikitamounier/assembly/cube-trial-20261107-v6')
assert os.environ.get('SLURM_JOB_ID')=='3077705'
sys.path.insert(0,str(ROOT))
cfg=json.loads((ROOT/'operational/cube-trial-20261107/cluster.json').read_text());r=json.loads((ROOT/'artifacts_local/chimera/controller-adoption.json').read_text())
assert r['learner'] is None and not r['workers']
assert (Path('/proc')/str(r['old_controller_pid'])/'stat').read_text().split()[2]=='T'
run=ROOT/cfg['run'];archive=run/'superseded-continuity-v3';assert not archive.exists() or not any(archive.iterdir());archive.mkdir(exist_ok=True)
source=run/'checkpoint.before-continuity-v3.pt';assert hashlib.sha256(source.read_bytes()).hexdigest()=='e7b6cee7615134d25a3906fb6b5d3ed68d40a90541d04537bf5467d102a4a732'
c=torch.load(source,map_location='cpu',weights_only=False);assert c['it']==1578 and c['transitions']==51291430
from assembly_policy.cube_trial.resets import signature,directory_signature
assert c['args']['environment_signature']=='224c2fd7070045c82a017313e14bbbc221ad29f41aa838ae001e973772d970e8'
assert c['args']['model_signature']==signature()==c['args']['physics_signature']
assert c['args']['bank_directory_signature']==directory_signature(ROOT/cfg['bank'])
s=hashlib.sha256((ROOT/'assembly_policy/cube_trial/env.py').read_bytes()).hexdigest()
n=copy.deepcopy(c);n['args']['environment_signature']=s;n['args']['reward_contract_version']='placement_v4_table_only';n['reward_migration']={'reason':'Human requested restore before v3; remove no-gripper-contact rule, retain off-target table contact after lift; initial drag allowed','from_environment_sha256':c['args']['environment_signature'],'to_environment_sha256':s}
tmp=run/'checkpoint.table-v4.tmp';torch.save(n,tmp)
equal=runpy.run_path(str(ROOT/'operational/cube-trial-20261107/migrate_continuity_v3.py'))['equal'];loaded=torch.load(tmp,map_location='cpu',weights_only=False)
for k in ('model','opt','sampler','lr','it','transitions'):assert equal(c[k],loaded[k]),k
# Preserve superseded branch and trim visible history to the checkpoint being restored.
for name in ['checkpoint.pt','log.jsonl','dashboard-summary.json']:
 p=run/name
 if p.exists():shutil.copy2(p,archive/name)
for p in run.glob('latest-success*.json'):p.rename(archive/p.name)
p=run/'log.jsonl';rows=[line for line in p.read_text().splitlines() if json.loads(line)['it']<=1578];p.write_text('\n'.join(rows)+'\n')
p=run/'dashboard-summary.json'
if p.exists():p.unlink()
os.replace(tmp,run/'checkpoint.pt')
audit=dict(restored_iteration=c['it'],restored_transitions=c['transitions'],training_state_exactly_equal_to_pre_rules_checkpoint=True,source_checkpoint_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),new_checkpoint_sha256=hashlib.sha256((run/'checkpoint.pt').read_bytes()).hexdigest(),superseded_branch_archive=str(archive),**n['reward_migration'])
(ROOT/'artifacts_local/chimera/reward-table-v4-migration.json').write_text(json.dumps(audit,indent=2)+'\n');print(json.dumps(audit))
