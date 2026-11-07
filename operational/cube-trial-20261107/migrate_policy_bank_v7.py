"""Audited reset-bank-only checkpoint continuation; new rows get fresh SGS history."""
import copy,hashlib,json,os,sys,shutil,runpy
from pathlib import Path
import numpy as np,torch
ROOT=Path('/large_storage/goodarzilab/nikitamounier/assembly/cube-trial-20261107-v6');sys.path.insert(0,str(ROOT));assert os.environ.get('SLURM_JOB_ID')=='3077705'
from assembly_policy.cube_trial.resets import check_bank,directory_signature,pooled_signature,signature
from assembly_policy.cube_trial.train import load_banks
from assembly_policy.cube_trial.core import FAMILIES
cfg_path=ROOT/'operational/cube-trial-20261107/cluster.json';cfg=json.loads(cfg_path.read_text());oldbank=ROOT/cfg['bank'];newbank=ROOT/'data/cube_trial/banks_policy_transport_v7';assert oldbank.name=='banks_pre_handoff_clear_v6';check_bank(newbank)
receipt=json.loads((ROOT/'artifacts_local/chimera/controller-adoption.json').read_text());assert receipt['learner'] is None and not receipt['workers'];assert (Path('/proc')/str(receipt['old_controller_pid'])/'stat').read_text().split()[2]=='T'
run=ROOT/cfg['run'];path=run/'checkpoint.pt';archive=run/'checkpoint.before-policy-bank-v7.pt';assert not archive.exists();c=torch.load(path,map_location='cpu',weights_only=False);a=c['args'];assert c['it']==receipt['graceful_checkpoint_iteration'];assert a['bank_directory_signature']==directory_signature(oldbank);assert a['physics_signature']==a['model_signature']==signature();assert a['reward_contract_version']=='placement_v4_table_only'
old=load_banks(['cube_handoff'],oldbank,a['bank_per_family']);new=load_banks(['cube_handoff'],newbank,a['bank_per_family']);mask=new[2]==FAMILIES.index('right_transport');assert np.array_equal(old[1],new[1]) and np.array_equal(old[2],new[2]);assert np.array_equal(old[0][~mask],new[0][~mask]);assert int(mask.sum())==128
changed=copy.deepcopy(c);changed['args']['env']['reset_context_path']=str(newbank/'reset-context.npz');changed['args']['environment_signature']=hashlib.sha256((ROOT/'assembly_policy/cube_trial/env.py').read_bytes()).hexdigest();changed['args'].update(bank_dir=str(newbank),bank_signature=pooled_signature(new),bank_directory_signature=directory_signature(newbank));changed['bank_migration']=dict(reason='Human requests exact right-transport states from successful dual/post rollouts',old_bank=str(oldbank),new_bank=str(newbank),replaced_entries=int(mask.sum()),other_families_unchanged=True)
for k in ['buf','ptr','cnt','sum']:changed['sampler'][k][mask]=0
shutil.copy2(path,archive);tmp=path.with_suffix('.pt.banktmp');torch.save(changed,tmp);v=torch.load(tmp,map_location='cpu',weights_only=False);equal=runpy.run_path(str(ROOT/'operational/cube-trial-20261107/migrate_continuity_v3.py'))['equal']
for k in ['model','opt','lr','it','transitions']:assert equal(c[k],v[k]),k
for k in ['buf','ptr','cnt','sum']:assert equal(c['sampler'][k][~mask],v['sampler'][k][~mask]);assert not v['sampler'][k][mask].any()
assert equal(c['sampler']['rng'],v['sampler']['rng']);os.replace(tmp,path)
cliparchive=run/'right-transport-before-policy-bank-v7';cliparchive.mkdir(exist_ok=False)
for p in [run/'latest-success-right_transport.json',run/'latest-success.json']:
 if p.exists() and json.loads(p.read_text()).get('family')=='right_transport':p.rename(cliparchive/p.name)
cfg_path.with_name('cluster-before-policy-bank-v7.json').write_bytes(cfg_path.read_bytes());cfg['bank']=str(newbank.relative_to(ROOT));cfg['controller_lock']='controller-generation10.lock';cfg_path.write_text(json.dumps(cfg,indent=2)+'\n')
# Only reviewed bank/config additions may change the source-integrity set.
p=ROOT/'artifacts_local/chimera/source-hashes.json';hashes=json.loads(p.read_text());unexpected=[k for k,v in hashes.items() if hashlib.sha256((ROOT/k).read_bytes()).hexdigest()!=v and k not in {'operational/cube-trial-20261107/cluster.json','assembly_policy/cube_trial/env.py','assembly_policy/cube_trial/train.py','assembly_policy/cube_trial/remote.py','assembly_policy/cube_trial/resets.py'}];assert not unexpected,unexpected;p.with_name('source-hashes-before-policy-bank-v7.json').write_bytes(p.read_bytes());
for name in ['operational/cube-trial-20261107/cluster.json','assembly_policy/cube_trial/env.py','assembly_policy/cube_trial/train.py','assembly_policy/cube_trial/remote.py','assembly_policy/cube_trial/resets.py','tests/test_cube_reset_context.py']:hashes[name]=hashlib.sha256((ROOT/name).read_bytes()).hexdigest()
for name in ['manifest.json','cube_handoff_train.npz','cube_handoff_eval.npz','right-transport-provenance.json','reset-context.npz']:hashes[str((newbank/name).relative_to(ROOT))]=hashlib.sha256((newbank/name).read_bytes()).hexdigest()
p.write_text(json.dumps(hashes,indent=2)+'\n')
audit=dict(iteration=c['it'],transitions=c['transitions'],policy_optimizer_exactly_preserved=True,unchanged_sampler_entries=int((~mask).sum()),new_right_entries_history_cleared=128,sampler_rng_preserved=True,archive_sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),new_checkpoint_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),source_manifest_sha256=hashlib.sha256(p.read_bytes()).hexdigest(),bank_directory_signature=directory_signature(newbank),**changed['bank_migration']);(ROOT/'artifacts_local/chimera/policy-bank-v7-migration.json').write_text(json.dumps(audit,indent=2)+'\n');print(json.dumps(audit),flush=True)
