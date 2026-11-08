from pathlib import Path
import hashlib,json,sys,tarfile
import numpy as np,torch
root,out=map(Path,sys.argv[1:]);torch.set_num_threads(1)
def equal(a,b):
 if isinstance(a,torch.Tensor):return isinstance(b,torch.Tensor) and torch.equal(a,b)
 if isinstance(a,np.ndarray):return isinstance(b,np.ndarray) and np.array_equal(a,b)
 if isinstance(a,dict):return isinstance(b,dict) and a.keys()==b.keys() and all(equal(a[k],b[k]) for k in a)
 if isinstance(a,(list,tuple)):return type(a)==type(b) and len(a)==len(b) and all(equal(x,y) for x,y in zip(a,b))
 return type(a)==type(b) and a==b
cp={v:torch.load(out/v/'ppo/checkpoint.pt',map_location='cpu',weights_only=False) for v in ['v10','v12','v13']}
fields=['model','opt','lr','it','transitions','sampler'];checks={v:{k:equal(cp['v10'][k],cp[v][k]) for k in fields} for v in ['v12','v13']}
assert all(all(d.values()) for d in checks.values()),checks
with np.load(out/'pairs-plain/outputs.npz') as a,np.load(out/'pairs-instrumented/outputs.npz') as b:
 paircheck={k:np.array_equal(a[k],b[k]) for k in a.files};assert set(a.files)==set(b.files) and all(paircheck.values())
banks={};nom=root/'native-v10/training-accepted-v10'
for v in ['v12','v13']:
 d=root/('native-'+v)/('training-accepted-'+v);rows=0;offsetmax=0
 for p in nom.glob('*.npz'):
  with np.load(p) as a,np.load(d/p.name) as b:
   assert np.array_equal(a['states'],b['states'][:,:162]);rows+=len(a['states'])
   if v=='v13':offsetmax=max(offsetmax,float(np.max(np.abs(b['states'][:,162:]))))
 banks[v]=dict(rows=rows,prefix162_identical=True,max_endpoint_offset_rad=offsetmax)
source_checks={}
for v in ['v10','v12','v13']:
 identity=json.loads((out/v/'panel-identity.json').read_text());rel=root/('native-'+v);changed=[]
 for name,old in identity['sources'].items():
  if Path(name).suffix not in ['.py','.cc','.c','.h']:continue
  p=rel/name
  if hashlib.sha256(p.read_bytes()).hexdigest()!=old:changed.append(name)
 source_checks[v]=dict(changed_source_files=changed,source_unchanged=not changed)
 assert not changed,changed
receipt=dict(status='PASS',checkpoint_sha256={v:hashlib.sha256((out/v/'ppo/checkpoint.pt').read_bytes()).hexdigest() for v in cp},checkpoint_fields_equal=checks,pair_instrumentation_output_equal=paircheck,bank_prefixes=banks,source_checks=source_checks)
(out/'audit.json').write_text(json.dumps(receipt,indent=2)+'\n')
files=[p for p in out.rglob('*') if p.is_file() and (p.suffix in ['.json','.npz'] or p.name=='so101_contact_search.c') and ('native' not in p.relative_to(out).parts or p.name=='build-receipt.json')]
with tarfile.open(out/'profile-receipts.tgz','w:gz') as t:
 for p in files:t.add(p,arcname=str(p.relative_to(out)))
print(json.dumps(receipt,indent=2))
