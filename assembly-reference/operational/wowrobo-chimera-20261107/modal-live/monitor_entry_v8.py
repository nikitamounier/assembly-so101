"""One pinned allocated observation/scaling invocation; use approved tool networking."""
from pathlib import Path
import hashlib,json,shlex,subprocess,sys,datetime
HERE=Path(__file__).parent
TARGET=json.loads((HERE/'target-v8.json').read_text())
PINS={'read': ('monitor.py', 'cf0c5b09b995aad5bc31f8fe3d78b53556909b879e5adf80395e94820b29b8d3'), 'scale': ('autoscale.py', '182977dbbc217ff79858ef8b68d0488ab763b8efa10c2c21cba5a8357fa6b9de'), 'failover': ('failover.py', 'ebca1df001b2c2247db5a28bf191b0b602f7d1b103e97ade8771bef2e27f33db')}
mode=sys.argv[1]
name,sha=PINS[mode];remote=TARGET['script']+name
args=[] if mode in ('read','failover') else ['--state-dir',TARGET['remote']+'/autoscale-v8']
code=f'import hashlib,runpy,sys; p={remote!r}; assert hashlib.sha256(open(p,"rb").read()).hexdigest()=={sha!r}; sys.argv={[remote,*args]!r}; runpy.run_path(p,run_name="__main__")'
command='srun --jobid=3077705 --overlap --exact --ntasks=1 --cpus-per-task=1 '+TARGET['env']+TARGET['python']+' -c '+shlex.quote(code)
try:
 p=subprocess.run(['ssh','-o','ConnectTimeout=15','-o','ConnectionAttempts=1','chimera-login',command],text=True,capture_output=True,timeout=120)
 record=dict(mode=mode,returncode=p.returncode,stdout=p.stdout,stderr=p.stderr)
except subprocess.TimeoutExpired as e:
 record=dict(mode=mode,returncode=124,unknown_remote_execution=True,stdout=str(e.stdout),stderr=str(e.stderr))
folder=HERE/'monitor-receipts';folder.mkdir(exist_ok=True);stamp=datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S.%fZ');path=folder/(stamp+'-'+mode+'.json');path.write_text(json.dumps(record,indent=2)+'\n')
print(json.dumps(dict(receipt=str(path),receipt_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),**record)),flush=True)
sys.exit(record['returncode'])
