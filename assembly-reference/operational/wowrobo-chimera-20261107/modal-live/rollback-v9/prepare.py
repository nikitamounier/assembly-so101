"""Native rollback preflight and isolated operators; no source physics changes."""
from pathlib import Path
import json,hashlib
ROOT=Path(__file__).resolve().parent.parent;REL=ROOT/'rollback-v9';HERE=ROOT/'ops-v9';OLD=ROOT/'ops-v7';HERE.mkdir(exist_ok=True)
manifest=json.loads((ROOT/'active-inputs-v7.json').read_text());bad=[n for n,h in manifest.items() if hashlib.sha256((ROOT/n).read_bytes()).hexdigest()!=h];assert not bad,bad
origin=ROOT/'run-v6/checkpoint.before-bare-v8.pt';sha=hashlib.sha256(origin.read_bytes()).hexdigest();assert sha=='df907ee1f1ff8bca4cb1ae640b45c3dd4350ad00f267ba4228a412b1ef3d409b'
for n in ['cluster.py','monitor.py','failover.py','autoscale.py','worker.sbatch','learner_bootstrap.py']:
 s=(OLD/n).read_text().replace("ROOT/'run-v6'","ROOT/'run-v9'").replace("ROOT/'active-inputs-v7.json'","ROOT/'active-inputs-v9.json'")
 (HERE/n).write_text(s)
cfg=json.loads((ROOT/'ops-v8/cluster.json').read_text());cfg.update(scope='NATIVE_ROLLBACK_FROM_PRE_BARE_CHECKPOINT_25');(HERE/'cluster.json').write_text(json.dumps(cfg,indent=2)+'\n')
# A bootstrap verifies exact initial state after the ordinary native contract checks.
p=HERE/'learner_bootstrap.py';s=p.read_text();needle="    runpy.run_path(str(Path(__file__).resolve().parent.parent/'replay-observer/record_training.py'),run_name='__main__')";assert s.count(needle)==1
hook='''    from assembly_policy import train
    import hashlib,json,copy,torch,numpy as np
    original_resume=train.Trainer._maybe_resume
    def equal(a,b):
        if isinstance(a,torch.Tensor):return isinstance(b,torch.Tensor) and torch.equal(a,b)
        if isinstance(a,np.ndarray):return np.array_equal(a,b)
        if isinstance(a,dict):return a.keys()==b.keys() and all(equal(a[k],b[k]) for k in a)
        if isinstance(a,(tuple,list)):return len(a)==len(b) and all(equal(x,y) for x,y in zip(a,b))
        return a==b
    def verified_resume(self):
        p=self.run/'checkpoint.pt';raw=p.read_bytes();d=torch.load(p,map_location=self.dev,weights_only=False)
        original_resume(self)
        sha=hashlib.sha256(raw).hexdigest()
        origin=dict(path=str(Path(__file__).resolve().parent.parent/'run-v6/checkpoint.before-bare-v8.pt'),sha256='df907ee1f1ff8bca4cb1ae640b45c3dd4350ad00f267ba4228a412b1ef3d409b',it=25,transitions=966743,physics_backend='wowrobo-native-source-sdf',reason='Latest human rejected simplified transitions; restore pre-bare native state. No bare descendant imported.')
        if sha==origin['sha256']:
            assert d['it']==25 and d['transitions']==966743
            self.model.load_state_dict(d['model'],strict=True);self.opt.load_state_dict(d['opt']);self.sampler.load_state_dict(copy.deepcopy(d['sampler']));self.lr=d['lr'];self.it=d['it'];self.transitions=d['transitions']
            assert equal(self.model.state_dict(),d['model']) and equal(self.opt.state_dict(),d['opt']) and equal(self.sampler.state_dict(),d['sampler'])
            proof=dict(status='PASS',origin=origin,policy_optimizer_sampler_RNG_exact=True,physics_signature=self.args['physics_signature'],observation_signature=self.args['observation_signature'])
            (self.run/'initial-restored.json').write_text(json.dumps(proof,indent=2)+'\\n')
        else:
            assert d['it']>=25 and d['args'].get('rollback_origin',{}).get('sha256')==origin['sha256'],'Never import a bare or unrelated descendant'
        assert self.args['physics_backend']=='wowrobo-native-source-sdf' and self.args['physics_signature']=='8c4783bc443e8e9efad8f19890b6910a79299d434dbb3111af9f24733d850879'
        self.args['rollback_origin']=origin
    train.Trainer._maybe_resume=verified_resume
'''
s=s.replace(needle,hook+needle);p.write_text(s)
manifest.pop('ops-v7/learner_bootstrap.py',None);manifest['ops-v9/learner_bootstrap.py']=hashlib.sha256(p.read_bytes()).hexdigest();(ROOT/'active-inputs-v9.json').write_text(json.dumps(manifest,indent=2)+'\n')
scale=ROOT/'autoscale-v9';scale.mkdir(exist_ok=True);x=json.loads((ROOT/'autoscale-v7/active-run.json').read_text());x.update(enabled=False,run_dir=str(ROOT/'run-v9'),worker_script=str(HERE/'worker.sbatch'),worker_script_sha256=hashlib.sha256((HERE/'worker.sbatch').read_bytes()).hexdigest(),root=str(ROOT/'assembly-so101'),cloud_workers=0,modal_reserved_worker_processes=0,sail_reserved_worker_processes=0,capacity_note='Native rollback first on retained400CPUs; no paidcloud dependency; parent validates newPPO before scaling.');(scale/'active-run.json').write_text(json.dumps(x,indent=2)+'\n')
result=dict(root=str(ROOT),run=str(ROOT/'run-v9'),config=cfg,pins={n:hashlib.sha256((HERE/n).read_bytes()).hexdigest() for n in ['cluster.py','monitor.py','failover.py','autoscale.py','worker.sbatch','learner_bootstrap.py']},physics_backend='wowrobo-native-source-sdf',physics_signature='8c4783bc443e8e9efad8f19890b6910a79299d434dbb3111af9f24733d850879',restored_checkpoint_sha256=sha,parent_it=25,parent_transitions=966743)
(ROOT/'ops-v9.json').write_text(json.dumps(result,indent=2)+'\n');proof=dict(status='PASS',source_files_verified=len(manifest),native_source_unchanged=True,origin_checkpoint_sha256=sha);(REL/'preflight.json').write_text(json.dumps(proof,indent=2)+'\n');print(json.dumps(result),flush=True)
