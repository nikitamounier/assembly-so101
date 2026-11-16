"""One hash-bound allocated CPU projection cycle; no arguments."""
import datetime, hashlib, json, pathlib, shlex, subprocess, sys
if len(sys.argv) != 1: raise SystemExit('No arguments accepted')
root=pathlib.Path(__file__).resolve().parent
source=root/'autoscale.py'
expected='00c0bb42afccae9053a23f967cde3bd1b48aad2c807ba2359d6bd9f3a315b851'
if hashlib.sha256(source.read_bytes()).hexdigest()!=expected: raise SystemExit('Local scaler identity changed')
script="import hashlib,runpy,sys; p='/large_storage/goodarzilab/nikitamounier/assembly/hybrid-autoscale-20261107.py'; assert hashlib.sha256(open(p,'rb').read()).hexdigest()=={expected!r}; sys.argv=[p,'--state-dir','/large_storage/goodarzilab/nikitamounier/assembly/hybrid-autoscale-20261107']; runpy.run_path(p,run_name='__main__')".format(expected=expected)
cmd=['ssh','-o','BatchMode=yes','-o','ConnectTimeout=15','chimera-login', 'srun --jobid=3077705 --overlap --exact --ntasks=1 --cpus-per-task=1 /large_storage/goodarzilab/nikitamounier/assembly/source-85c006a/.venv/bin/python -c '+shlex.quote(script)]
stamp=datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S.%fZ')
try:
 p=subprocess.run(cmd,capture_output=True,text=True,timeout=180)
 receipt=dict(command=cmd,returncode=p.returncode,stdout=p.stdout,stderr=p.stderr)
except subprocess.TimeoutExpired:
 receipt=dict(command=cmd,returncode=None,error='Timeout: unknown remote completion; do not retry this cycle')
out=root/'receipts';out.mkdir(exist_ok=True)
path=out/(stamp+'.json');path.write_text(json.dumps(receipt,indent=2)+'\n')
print(json.dumps(dict(receipt=str(path),receipt_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),**receipt)))
