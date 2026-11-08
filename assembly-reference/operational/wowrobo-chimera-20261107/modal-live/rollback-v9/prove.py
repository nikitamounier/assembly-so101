from pathlib import Path
import json,hashlib,io,torch,math,time
ROOT=Path(__file__).resolve().parent.parent;RUN=ROOT/'run-v9';REL=ROOT/'rollback-v9';origin=ROOT/'run-v6/checkpoint.before-bare-v8.pt'
assert hashlib.sha256(origin.read_bytes()).hexdigest()=='df907ee1f1ff8bca4cb1ae640b45c3dd4350ad00f267ba4228a412b1ef3d409b'
assert json.loads((RUN/'initial-restored.json').read_text())['status']=='PASS'
rows=[json.loads(x) for x in (RUN/'log.jsonl').read_text().splitlines()];raw=(RUN/'checkpoint.pt').read_bytes();ck=torch.load(io.BytesIO(raw),map_location='cpu',weights_only=False)
assert ck['it']>=26 and ck['transitions']>966743
row=next(x for x in rows if x['it']==ck['it']);assert row['transitions']==ck['transitions'];assert row['valid_transitions']>0
assert all(math.isfinite(row[k]) for k in ('pi','vf','ent','kl','lr','std','fps','safety','success'))
assert ck['args']['physics_backend']=='wowrobo-native-source-sdf' and ck['args']['physics_signature']=='8c4783bc443e8e9efad8f19890b6910a79299d434dbb3111af9f24733d850879'
assert ck['args']['rollback_origin']['sha256']=='df907ee1f1ff8bca4cb1ae640b45c3dd4350ad00f267ba4228a412b1ef3d409b'
assert ck['sampler']['buf'].shape==(1143,50)
for v in ck['model'].values():assert torch.isfinite(v).all()
for state in ck['opt']['state'].values():
 for v in state.values():
  if isinstance(v,torch.Tensor):assert torch.isfinite(v).all()
manifest=json.loads((ROOT/'active-inputs-v9.json').read_text());bad=[n for n,h in manifest.items() if hashlib.sha256((ROOT/n).read_bytes()).hexdigest()!=h];assert not bad,bad
proof=dict(status='PASS',rollback_complete=True,run=str(RUN),origin_checkpoint=str(origin),origin_sha256=ck['args']['rollback_origin']['sha256'],origin_it=25,origin_transitions=966743,initial_state_exact=True,source_valid=True,physics_signature=ck['args']['physics_signature'],physics_backend=ck['args']['physics_backend'],actual_new_native_updates=ck['it']-25,actual_new_native_transitions=ck['transitions']-966743,checkpoint_sha256=hashlib.sha256(raw).hexdigest(),latest_matching=row,policy_optimizer_finite=True,bare_descendant_imported=False,time_unix=time.time())
(REL/'live-proof.json').write_text(json.dumps(proof,indent=2)+'\n');print(json.dumps(proof),flush=True)
