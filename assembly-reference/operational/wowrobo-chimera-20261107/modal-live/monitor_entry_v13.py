"""One pinned allocated observation/scaling invocation; use approved tool networking."""
from pathlib import Path
import hashlib,json,shlex,subprocess,sys,datetime
HERE=Path(__file__).parent
TARGET=json.loads((HERE/'target-v13.json').read_text())
PINS={'read': ('monitor.py', '24786238a3b29549df44045d1860df0f00061057896778d8f232a0b2f39ca197'), 'scale': ('autoscale.py', '182977dbbc217ff79858ef8b68d0488ab763b8efa10c2c21cba5a8357fa6b9de'), 'failover': ('failover.py', '8aa483fc2c6c6f4e1185a7b2f10d67726db64879899fed5caf005616b71c4b34')}
mode=sys.argv[1]
name,sha=PINS[mode];remote=TARGET['script']+name
args=[] if mode in ('read','failover') else ['--state-dir',TARGET['remote']+'/autoscale-v13']
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
