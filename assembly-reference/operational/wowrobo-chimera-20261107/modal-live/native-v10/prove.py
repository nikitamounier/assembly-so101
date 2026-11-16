from pathlib import Path
import io,json,hashlib,math,torch
ROOT=Path(__file__).resolve().parent.parent;REL=ROOT/'native-v10';RUN=ROOT/'run-v10';migration=json.loads((REL/'migration.json').read_text())
assert json.loads((RUN/'initial-restored.json').read_text())['status']=='PASS'
raw=(RUN/'checkpoint.pt').read_bytes();ck=torch.load(io.BytesIO(raw),map_location='cpu',weights_only=False);assert ck['it']>migration['parent_it'] and ck['transitions']>migration['parent_transitions'];rows=[json.loads(x) for x in (RUN/'log.jsonl').read_text().splitlines()];row=next(x for x in rows if x['it']==ck['it']);assert row['transitions']==ck['transitions'] and row['valid_transitions']>0
assert ck['args']['physics_signature']==migration['physics_signature'] and ck['args']['physics_backend']=='wowrobo-native-source-sdf';assert ck['args']['native_timestep_s']==.0005 and ck['args']['substeps_per_action']==80 and ck['args']['control_dt_s']==.04
assert ck['args']['continuation_origin']['sha256']==migration['parent_checkpoint']['sha256'];assert ck['sampler']['buf'].shape==(1143,50)
assert all(torch.isfinite(v).all() for v in ck['model'].values())
for s in ck['opt']['state'].values():
 for v in s.values():
  if isinstance(v,torch.Tensor):assert torch.isfinite(v).all()
assert all(math.isfinite(row[k]) for k in ('pi','vf','ent','kl','lr','std','fps','safety','success'));manifest=json.loads((ROOT/'active-inputs-v10.json').read_text());bad=[n for n,h in manifest.items() if hashlib.sha256((ROOT/n).read_bytes()).hexdigest()!=h];assert not bad,bad
x=dict(status='PASS',run=str(RUN),parent_it=migration['parent_it'],parent_transitions=migration['parent_transitions'],origin_sha256=migration['parent_checkpoint']['sha256'],physics_signature=ck['args']['physics_signature'],timestep=.0005,substeps=80,control_dt=.04,actual_new_updates=ck['it']-migration['parent_it'],actual_new_transitions=ck['transitions']-migration['parent_transitions'],checkpoint_sha256=hashlib.sha256(raw).hexdigest(),policy_optimizer_finite=True,source_valid=True,latest_matching=row);(REL/'live-proof.json').write_text(json.dumps(x,indent=2)+'\n');print(json.dumps(x),flush=True)
