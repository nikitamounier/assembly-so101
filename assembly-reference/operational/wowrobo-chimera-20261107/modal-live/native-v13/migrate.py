"""Explicit native checkpoint migration; only the versioned reset suffix changes."""
from pathlib import Path
import copy,hashlib,io,json,sys
import numpy as np,torch
from assembly_policy.train import Trainer,parse
from assembly_policy.so101.resets import pooled_signature
ROOT=Path(__file__).resolve().parent.parent;REL=ROOT/'native-v13';PARENT=ROOT/'run-v10'
preview=sys.argv[1:]==['--preview'];assert sys.argv[1:] in ([],['--preview'])
RUN=REL/'migration-preview' if preview else ROOT/'run-v13'
BANK=REL/'training-accepted-v13'
for name in ('bank-qualification.json','qualification.json','insertion-dynamics.json','stop-response.json','production-worker-proof.json','distributed-smoke.json'):
    assert json.loads((REL/name).read_text())['status']=='PASS',name
if not preview:assert json.loads((PARENT/'handoff-closed-v13.json').read_text())['status']=='PASS'
archive=PARENT/('checkpoint.before-state-build-repause.pt' if preview else 'checkpoint.before-native-v13.pt')
raw=archive.read_bytes();old=torch.load(io.BytesIO(raw),map_location='cpu',weights_only=False)
assert old['args']['physics_backend']=='wowrobo-native-source-sdf'
assert old['args']['physics_signature']=='34d224d4edff266369b3f1322baf80013c0daf09599a9eeb8751c6270a88ff92'
assert old['it']>=32 and not (RUN/'checkpoint.pt').exists()
oldAD=ROOT/'native-v10/assembly-so101';AD=REL/'assembly-so101'
for path in ('assembly_policy/so101/wowrobo_reward.py','configs/wowrobo_reward_regions.json'):
    assert (oldAD/path).read_bytes()==(AD/path).read_bytes(),path
# Prove that the complete environment-source delta is confined to userdata wiring.
s=(AD/'assembly_policy/wowrobo/env.py').read_text()
new='''        self.hinge_limits = self.profile.get("training_hinge_limits") == "native-userdata-v1"
        self.friction_start = self.eq_end + (self.m.nu + self.m.nv if self.extra_reset else 0)
        self.hinge_start = self.friction_start + (3 if self.episode_friction else 0)
        self.nstate = self.hinge_start + (4 if self.hinge_limits else 0)'''
legacy='        self.nstate = self.eq_end + (self.m.nu + self.m.nv if self.extra_reset else 0) + (3 if self.episode_friction else 0)'
assert s.count(new)==1;s=s.replace(new,legacy)
new='''            if self.episode_friction: d.userdata[:3] = self.state[i,self.friction_start:self.hinge_start]
            if self.hinge_limits: d.userdata[3:7] = self.state[i,self.hinge_start:self.hinge_start+4]'''
assert s.count(new)==1;s=s.replace(new,'            if self.episode_friction: d.userdata[:3] = self.state[i,-3:]')
s=s.replace('self.state[:,self.friction_start:]','self.state[:,-3:]')
s=s.replace('if self.episode_friction or self.hinge_limits else []','if self.episode_friction else []')
s=s.replace('if self.episode_friction or self.hinge_limits else 0','if self.episode_friction else 0')
assert s==(oldAD/'assembly_policy/wowrobo/env.py').read_text(),'Unauthorized environment/reward/diagnostics delta'
cmd=json.loads((PARENT/'command.json').read_text());cmd=cmd[cmd.index('--suite'):]
for key,value in {'--run-dir':RUN,'--device':'cpu','--bank-dir':BANK,'--envs':9600,'--remote-workers':960,'--min-conn-frac':.025}.items():
    if key in cmd:cmd[cmd.index(key)+1]=str(value)
    else:cmd.extend([key,str(value)])
a=parse(cmd);assert a.steps==32 and a.epochs==5 and a.minibatches==4 and a.micro_batch==1024
def same(a,b):
    if isinstance(a,torch.Tensor):return isinstance(b,torch.Tensor) and torch.equal(a,b)
    if isinstance(a,np.ndarray):return np.array_equal(a,b)
    if isinstance(a,dict):return a.keys()==b.keys() and all(same(a[k],b[k]) for k in a)
    if isinstance(a,(list,tuple)):return len(a)==len(b) and all(same(x,y) for x,y in zip(a,b))
    return a==b
class Done(Exception):pass
class Migration(Trainer):
    def _maybe_resume(self):
        assert self.tasks==self.backend.resolve_tasks(old['args']['tasks'])
        assert self.args['observation_signature']==old['args']['observation_signature']
        child_env=copy.deepcopy(self.args['env']);parent_env=copy.deepcopy(old['args']['env'])
        child_env.pop('profile_path');parent_env.pop('profile_path');assert child_env==parent_env
        assert self.bank_S.shape==(1143,166)
        assert pooled_signature((np.ascontiguousarray(self.bank_S[:,:162]),self.bank_T,self.bank_F))==old['args']['bank_signature']
        assert self.args['model_signature']==json.loads((REL/'prepared.json').read_text())['physics_signature']
        self.model.load_state_dict(old['model'],strict=True);self.opt.load_state_dict(old['opt']);self.sampler.load_state_dict(copy.deepcopy(old['sampler']))
        self.lr=old['lr'];self.it=old['it'];self.transitions=old['transitions']
        assert same(self.model.state_dict(),old['model']) and same(self.opt.state_dict(),old['opt']) and same(self.sampler.state_dict(),old['sampler'])
        provenance=dict(path=str(archive),sha256=hashlib.sha256(raw).hexdigest(),it=old['it'],transitions=old['transitions'],physics_backend=old['args']['physics_backend'],physics_signature=old['args']['physics_signature'],reason='Human requests measured target-servo2/3 stops with tiny endpoint noise; all external contacts, guards, observations and reward definition retained; no bare descendant.')
        self.args.update(parent_checkpoint=provenance,continuation_origin=provenance,native_timestep_s=.0005,substeps_per_action=80,control_dt_s=.04,source_model_signature=old['args']['model_signature'],source_reward_signature=old['args']['reward_signature'],reward_definition_unchanged=True,reset_schema='fullphysics-eq-ctrl-warmstart-3friction-4hingelimit-v1')
        self.save()
        x=dict(status='PASS',preview=preview,run=str(RUN),parent_it=old['it'],parent_transitions=old['transitions'],parent_checkpoint=provenance,physics_signature=self.args['physics_signature'],policy_optimizer_SGS_reset_rng_exact=True,legacy_reset_prefix162_row_order_exact=True,reset_payload_size=166,reward_definition_unchanged=True,source_reward_signature=old['args']['reward_signature'],reward_signature=self.args['reward_signature'],checkpoint_sha256=hashlib.sha256((RUN/'checkpoint.pt').read_bytes()).hexdigest(),bank_signature=self.args['bank_signature'],bank_directory_signature=self.args['bank_directory_signature'])
        (REL/('migration-preview.json' if preview else 'migration.json')).write_text(json.dumps(x,indent=2)+'\n')
        print(json.dumps(x),flush=True);raise Done
try:Migration(a)
except Done:pass
