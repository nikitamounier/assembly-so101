"""Prepare timestep-only continuation after qualification and distributed gates."""
from pathlib import Path
import json,hashlib
ROOT=Path(__file__).resolve().parent.parent;REL=ROOT/'native-v10';HERE=ROOT/'ops-v10';OLD=ROOT/'ops-v9';HERE.mkdir(exist_ok=True)
qa=json.loads((REL/'qualification.json').read_text());smoke=json.loads((REL/'distributed-smoke.json').read_text());assert qa['status']==smoke['status']=='PASS';assert json.loads((REL/'insertion-dynamics.json').read_text())['status']=='PASS'
p=REL/'training-accepted-v10/manifest.json';x=json.loads(p.read_text());x['timestep_qualification']='bounded paired native held/near dynamics +125insertioncontrols +distributedPPO PASS; originalresetadmissionprovenance retained';p.write_text(json.dumps(x,indent=2)+'\n')
prepared=json.loads((REL/'prepared.json').read_text());files=prepared['files'];files[str(p.relative_to(ROOT))]=hashlib.sha256(p.read_bytes()).hexdigest();prepared['files']=files;prepared['status']='QUALIFIED';(REL/'prepared.json').write_text(json.dumps(prepared,indent=2)+'\n')
for n in ['cluster.py','monitor.py','failover.py','autoscale.py','worker.sbatch']:
 s=(OLD/n).read_text().replace("ROOT/'run-v9'","ROOT/'run-v10'").replace("ROOT/'active-inputs-v9.json'","ROOT/'active-inputs-v10.json'")
 if n=='cluster.py':s=s.replace("ROOT/'robotics-assembly-3.0/assets_local/wowrobo/training-accepted-v6'","ROOT/'native-v10/training-accepted-v10'")
 (HERE/n).write_text(s)
cfg=json.loads((OLD/'cluster.json').read_text());cfg['scope']='NATIVE_TIMESTEP_0_5MS_CONTINUATION';(HERE/'cluster.json').write_text(json.dumps(cfg,indent=2)+'\n')
s=(ROOT/'ops-v7/learner_bootstrap.py').read_text();needle="    runpy.run_path(str(Path(__file__).resolve().parent.parent/'replay-observer/record_training.py'),run_name='__main__')";assert s.count(needle)==1
hook='''    from assembly_policy import train
    import json,hashlib,torch,copy,numpy as np
    resume=train.Trainer._maybe_resume
    def equal(a,b):
        if isinstance(a,torch.Tensor):return isinstance(b,torch.Tensor) and torch.equal(a,b)
        if isinstance(a,np.ndarray):return np.array_equal(a,b)
        if isinstance(a,dict):return a.keys()==b.keys() and all(equal(a[k],b[k]) for k in a)
        if isinstance(a,(tuple,list)):return len(a)==len(b) and all(equal(x,y) for x,y in zip(a,b))
        return a==b
    def exact_resume(self):
        p=self.run/'checkpoint.pt';raw=p.read_bytes();d=torch.load(p,map_location=self.dev,weights_only=False);resume(self)
        receipt=json.loads((Path(__file__).resolve().parent.parent/'native-v10/migration.json').read_text())
        if hashlib.sha256(raw).hexdigest()==receipt['checkpoint_sha256']:
            self.model.load_state_dict(d['model'],strict=True);self.opt.load_state_dict(d['opt']);self.sampler.load_state_dict(copy.deepcopy(d['sampler']));self.lr=d['lr'];self.it=d['it'];self.transitions=d['transitions']
            assert equal(self.model.state_dict(),d['model']) and equal(self.opt.state_dict(),d['opt']) and equal(self.sampler.state_dict(),d['sampler'])
            (self.run/'initial-restored.json').write_text(json.dumps(dict(status='PASS',it=d['it'],transitions=d['transitions'],policy_optimizer_SGS_reset_rng_exact=True,physics_signature=self.args['physics_signature']))+'\\n')
        for k in ('continuation_origin','native_timestep_s','substeps_per_action','control_dt_s','source_model_signature'):
            self.args[k]=d['args'][k]
    train.Trainer._maybe_resume=exact_resume
'''
(HERE/'learner_bootstrap.py').write_text(s.replace(needle,hook+needle))
manifest=json.loads((ROOT/'active-inputs-v9.json').read_text());manifest.pop('ops-v9/learner_bootstrap.py',None);manifest.update(files);manifest['native-v10/prepared.json']=hashlib.sha256((REL/'prepared.json').read_bytes()).hexdigest();manifest['ops-v10/learner_bootstrap.py']=hashlib.sha256((HERE/'learner_bootstrap.py').read_bytes()).hexdigest();(ROOT/'active-inputs-v10.json').write_text(json.dumps(manifest,indent=2)+'\n')
scale=ROOT/'autoscale-v10';scale.mkdir(exist_ok=True);x=json.loads((ROOT/'autoscale-v9/active-run.json').read_text());x.update(enabled=False,run_dir=str(ROOT/'run-v10'),root=str(REL/'assembly-so101'),worker_script=str(HERE/'worker.sbatch'),worker_script_sha256=hashlib.sha256((HERE/'worker.sbatch').read_bytes()).hexdigest(),capacity_note='Native0.5ms80substeps; protectedscale onlyafteractualnewPPO proof; no paidcloud.');(scale/'active-run.json').write_text(json.dumps(x,indent=2)+'\n')
x=dict(root=str(ROOT),run=str(ROOT/'run-v10'),config=cfg,pins={n:hashlib.sha256((HERE/n).read_bytes()).hexdigest() for n in ['cluster.py','monitor.py','failover.py','autoscale.py','worker.sbatch','learner_bootstrap.py']},physics_signature=prepared['physics_signature'],physics_backend='wowrobo-native-source-sdf',timestep=.0005,substeps=80,control_dt=.04);(ROOT/'ops-v10.json').write_text(json.dumps(x,indent=2)+'\n');print(json.dumps(x),flush=True)
