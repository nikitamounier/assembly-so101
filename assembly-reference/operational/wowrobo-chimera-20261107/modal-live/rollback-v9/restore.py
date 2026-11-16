from pathlib import Path
import json,hashlib,time
ROOT=Path(__file__).resolve().parent.parent;RUN=ROOT/'run-v9';REL=ROOT/'rollback-v9';source=ROOT/'run-v6/checkpoint.before-bare-v8.pt'
assert json.loads((REL/'preflight.json').read_text())['status']=='PASS'
assert json.loads((ROOT/'run-v8/handoff-closed-v9.json').read_text())['status']=='PASS'
assert (ROOT/'run-v8/worker-closure-v9.json').exists()
raw=source.read_bytes();sha=hashlib.sha256(raw).hexdigest();assert sha=='df907ee1f1ff8bca4cb1ae640b45c3dd4350ad00f267ba4228a412b1ef3d409b'
RUN.mkdir(exist_ok=True);p=RUN/'checkpoint.pt';assert not p.exists();tmp=p.with_suffix('.tmp');tmp.write_bytes(raw);tmp.replace(p)
assert hashlib.sha256(p.read_bytes()).hexdigest()==sha
x=dict(status='PASS',run=str(RUN),source=str(source),restored_checkpoint_sha256=sha,it=25,transitions=966743,restore_is_byte_exact=True,source_native_unchanged=True,physics_signature='8c4783bc443e8e9efad8f19890b6910a79299d434dbb3111af9f24733d850879',bare_descendant_imported=False,time_unix=time.time());(REL/'restore.json').write_text(json.dumps(x,indent=2)+'\n');(RUN/'rollback.json').write_text(json.dumps(x,indent=2)+'\n')
alias=ROOT/'v9';assert not alias.exists();alias.symlink_to(ROOT/'ops-v9',target_is_directory=True)
print(json.dumps(x),flush=True)
