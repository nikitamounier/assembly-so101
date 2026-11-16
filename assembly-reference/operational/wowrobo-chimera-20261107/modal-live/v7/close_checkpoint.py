"""Close only the old capacity-phase learner after verifying its durable checkpoint."""
from pathlib import Path
import hashlib,io,json,os,signal,time,torch
ROOT=Path(__file__).resolve().parent.parent;run=ROOT/'run-v6'
row=json.loads((run/'handoff-request-v7.json').read_text());p=Path('/proc')/str(row['pid']);paused=False;killed=False
def alive():
    try:return p.exists() and (p/'stat').read_text().split()[21]==row['start'] and (p/'stat').read_text().split()[2]!='Z'
    except FileNotFoundError:return False
try:
    if alive():
        assert p.stat().st_uid==os.getuid()
        args=(p/'cmdline').read_bytes().split(b'\0')
        assert str(ROOT/'ops-v6/learner_bootstrap.py').encode() in args and str(run).encode() in args
        os.kill(row['pid'],signal.SIGSTOP);paused=True
    raw=(run/'checkpoint.pt').read_bytes();ck=torch.load(io.BytesIO(raw),map_location='cpu',weights_only=False)
    latest=json.loads((run/'log.jsonl').read_text().splitlines()[-1])
    assert ck['it']==latest['it'] and ck['transitions']==latest['transitions'],'Wait for final atomic checkpoint after log flush'
    assert ck['sampler']['buf'].shape==(1143,50)
    for v in ck['model'].values():assert torch.isfinite(v).all()
    for state in ck['opt']['state'].values():
        for v in state.values():
            if isinstance(v,torch.Tensor):assert torch.isfinite(v).all()
    assert (run/'checkpoint.pt').read_bytes()==raw
    archive=run/'checkpoint.before-capacity-v7.pt'
    if archive.exists():assert archive.read_bytes()==raw
    else:archive.write_bytes(raw)
    for name in ('config.json','command.json','controller.json'):
        old=run/name;dest=run/(old.stem+'.before-capacity-v7.json')
        if old.exists() and not dest.exists():dest.write_bytes(old.read_bytes())
    if alive():
        os.kill(row['pid'],signal.SIGKILL);killed=True
        for _ in range(50):
            if not alive():break
            time.sleep(.1)
    assert not alive()
    proof=dict(status='PASS',learner_alive=False,pid=row['pid'],start=row['start'],parent_it=ck['it'],transitions=ck['transitions'],checkpoint_bytes=len(raw),checkpoint_sha256=hashlib.sha256(raw).hexdigest(),archive=str(archive),time=time.time(),method='Verified committed checkpoint; only unfinished rollout discarded',allocations_retained=True)
    (run/'handoff-closed-v7.json').write_text(json.dumps(proof,indent=2)+'\n');print(json.dumps(proof),flush=True)
finally:
    if paused and not killed and alive():os.kill(row['pid'],signal.SIGCONT)
