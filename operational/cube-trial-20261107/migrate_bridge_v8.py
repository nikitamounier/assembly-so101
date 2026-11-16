"""Append acquisition families while preserving all existing training state."""
import copy,hashlib,json,os,sys,shutil,runpy
from pathlib import Path
import numpy as np,torch
ROOT=Path('/large_storage/goodarzilab/nikitamounier/assembly/cube-trial-20261107-v6');sys.path.insert(0,str(ROOT));assert os.environ['SLURM_JOB_ID']=='3077705'
from assembly_policy.cube_trial.resets import check_bank,directory_signature,pooled_signature,signature
from assembly_policy.cube_trial.train import load_banks
from assembly_policy.cube_trial.core import FAMILIES,REWARD_CONTRACT_VERSION
cp=ROOT/'operational/cube-trial-20261107/cluster.json';cfg=json.loads(cp.read_text());oldbank=ROOT/cfg['bank'];newbank=ROOT/'data/cube_trial/banks_handoff_bridge_v8';assert oldbank.name=='banks_policy_transport_v7';check_bank(newbank)
r=json.loads((ROOT/'artifacts_local/chimera/controller-adoption.json').read_text());assert r['learner'] is None and not r['workers'];assert (Path('/proc')/str(r['old_controller_pid'])/'stat').read_text().split()[2]=='T'
run=ROOT/cfg['run'];path=run/'checkpoint.pt';archive=run/'checkpoint.before-bridge-v8.pt';assert not archive.exists();c=torch.load(path,map_location='cpu',weights_only=False);assert c['it']==r['graceful_checkpoint_iteration'];a=c['args'];assert a['bank_directory_signature']==directory_signature(oldbank);assert a['physics_signature']==a['model_signature']==signature();assert a['reward_contract_version']==REWARD_CONTRACT_VERSION=='placement_v4_table_only'
old=load_banks(['cube_handoff'],oldbank,a['bank_per_family']);new=load_banks(['cube_handoff'],newbank,a['bank_per_family']);n=len(old[0]);assert n==1152 and len(new[0])==1408 and len(FAMILIES)==11
for x,y in zip(old,new):assert np.array_equal(x,y[:n])
changed=copy.deepcopy(c);changed['args'].update(bank_dir=str(newbank),bank_signature=pooled_signature(new),bank_directory_signature=directory_signature(newbank));changed['args']['env']['reset_context_path']=str(newbank/'reset-context.npz')
for key in ['buf','cnt','ptr','sum']:
 v=changed['sampler'][key];changed['sampler'][key]=np.concatenate([v,np.zeros((256,*v.shape[1:]),dtype=v.dtype)])
changed['bank_migration']=dict(reason='Human adds partial_right_contact and light_right_grasp, same reward/retained CPUs',old_entries=n,new_entries=1408,existing_entry_ids_preserved=True)
shutil.copy2(path,archive);tmp=path.with_suffix('.pt.bridgetmp');torch.save(changed,tmp);v=torch.load(tmp,map_location='cpu',weights_only=False);equal=runpy.run_path(str(ROOT/'operational/cube-trial-20261107/migrate_continuity_v3.py'))['equal']
for key in ['model','opt','lr','it','transitions']:assert equal(c[key],v[key]),key
for key in ['buf','cnt','ptr','sum']:assert equal(c['sampler'][key],v['sampler'][key][:n]);assert not v['sampler'][key][n:].any()
assert equal(c['sampler']['rng'],v['sampler']['rng']);os.replace(tmp,path)
cp.with_name('cluster-before-bridge-v8.json').write_bytes(cp.read_bytes());cfg['bank']=str(newbank.relative_to(ROOT));cfg['controller_lock']='controller-generation11.lock';cp.write_text(json.dumps(cfg,indent=2)+'\n')
p=ROOT/'artifacts_local/chimera/source-hashes.json';h=json.loads(p.read_text());changed_sources=['assembly_policy/cube_trial/core.py','operational/cube-trial-20261107/cluster.json'];bad=[k for k,v in h.items() if hashlib.sha256((ROOT/k).read_bytes()).hexdigest()!=v and k not in changed_sources];assert not bad,bad;p.with_name('source-hashes-before-bridge-v8.json').write_bytes(p.read_bytes())
for name in changed_sources+['assembly_policy/cube_trial/families/partial_right_contact.py','assembly_policy/cube_trial/families/light_right_grasp.py','tests/test_cube_bridge_bank.py']:h[name]=hashlib.sha256((ROOT/name).read_bytes()).hexdigest()
for name in ['manifest.json','cube_handoff_train.npz','cube_handoff_eval.npz','reset-context.npz','bridge-provenance.json']:h[str((newbank/name).relative_to(ROOT))]=hashlib.sha256((newbank/name).read_bytes()).hexdigest()
p.write_text(json.dumps(h,indent=2)+'\n');audit=dict(iteration=c['it'],transitions=c['transitions'],policy_optimizer_exactly_preserved=True,existing_sampler_histories_exact=True,sampler_rng_exact=True,new_sampler_entries=256,new_families=FAMILIES[-2:],reward=REWARD_CONTRACT_VERSION,checkpoint_archive_sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),source_manifest_sha256=hashlib.sha256(p.read_bytes()).hexdigest(),bank_signature=directory_signature(newbank));(ROOT/'artifacts_local/chimera/bridge-v8-migration.json').write_text(json.dumps(audit,indent=2)+'\n');print(json.dumps(audit),flush=True)
