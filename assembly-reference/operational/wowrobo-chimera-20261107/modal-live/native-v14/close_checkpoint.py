"""Close only the old capacity-phase learner after verifying its durable checkpoint."""
from pathlib import Path
import hashlib,io,json,os,signal,time,torch
ROOT=Path(__file__).resolve().parent.parent;run=ROOT/'run-v13'
rows=[]
for candidate in Path('/proc').iterdir():
 if not candidate.name.isdigit():continue
 try:
  args=(candidate/'cmdline').read_bytes().split(b'\0')
  if candidate.stat().st_uid==os.getuid() and str(ROOT/'ops-v13/learner_bootstrap.py').encode() in args and str(run).encode() in args:rows.append(dict(pid=int(candidate.name),start=(candidate/'stat').read_text().split()[21]))
 except (FileNotFoundError,PermissionError,ProcessLookupError):pass
assert len(rows)==1,rows
row=rows[0];p=Path('/proc')/str(row['pid']);paused=False;killed=False
def alive():
    try:return p.exists() and (p/'stat').read_text().split()[21]==row['start'] and (p/'stat').read_text().split()[2]!='Z'
    except FileNotFoundError:return False
try:
    if alive():
        assert p.stat().st_uid==os.getuid()
        args=(p/'cmdline').read_bytes().split(b'\0')
        assert str(ROOT/'ops-v13/learner_bootstrap.py').encode() in args and str(run).encode() in args
        os.kill(row['pid'],signal.SIGSTOP);paused=True
    raw=(run/'checkpoint.pt').read_bytes();ck=torch.load(io.BytesIO(raw),map_location='cpu',weights_only=False)
    latest=next(json.loads(line) for line in (run/'log.jsonl').read_text().splitlines() if json.loads(line)['it']==ck['it'])
    assert ck['it']==latest['it'] and ck['transitions']==latest['transitions'],'Wait for final atomic checkpoint after log flush'
    assert ck['sampler']['buf'].shape==(1143,50)
    assert ck['it']>=35 and ck['args']['physics_signature']=='13ac7a8d07d4d63e7de21bd426319fd32e60acfd33e9fd681a43a2d1f8cd959e'
    for v in ck['model'].values():assert torch.isfinite(v).all()
    for state in ck['opt']['state'].values():
        for v in state.values():
            if isinstance(v,torch.Tensor):assert torch.isfinite(v).all()
    assert (run/'checkpoint.pt').read_bytes()==raw
    archive=run/'checkpoint.before-native-v14.pt'
    if archive.exists():assert archive.read_bytes()==raw
    else:archive.write_bytes(raw)
    for name in ('config.json','command.json','controller.json'):
        old=run/name;dest=run/(old.stem+'.before-native-v14.json')
        if old.exists() and not dest.exists():dest.write_bytes(old.read_bytes())
    if alive():
        os.kill(row['pid'],signal.SIGKILL);killed=True
        for _ in range(50):
            if not alive():break
            time.sleep(.1)
    assert not alive()
    proof=dict(status='PASS',learner_alive=False,pid=row['pid'],start=row['start'],parent_it=ck['it'],transitions=ck['transitions'],checkpoint_bytes=len(raw),checkpoint_sha256=hashlib.sha256(raw).hexdigest(),archive=str(archive),time=time.time(),method='Verified committed checkpoint; only unfinished rollout discarded',allocations_retained=True)
    (run/'handoff-closed-v14.json').write_text(json.dumps(proof,indent=2)+'\n');print(json.dumps(proof),flush=True)
finally:
    pass  # Original32 is intentionally paused; a failed handoff must never resume it.
