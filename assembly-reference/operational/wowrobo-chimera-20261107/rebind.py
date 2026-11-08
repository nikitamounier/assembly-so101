from pathlib import Path
import hashlib,json,os
ROOT=Path(__file__).resolve().parents[3]
OLD='/Users/nikitamounier/Documents/ChatGPT/robot assembly 2'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
original=json.loads((ROOT/'source-manifest.json').read_text())
for f,h in original.items():
 assert sha(ROOT/f)==h,f
changes={}
for f in original:
 p=ROOT/f
 if p.suffix in ('.xml','.json'):
  old=sha(p);s=p.read_text();s=s.replace(OLD,str(ROOT));p.write_text(s);changes[old]=sha(p)
# Rebind XML identity references, without changing geometry or validation status.
canon=ROOT/'robotics-assembly-3.0';profile=canon/'assets_local/wowrobo/physical-resets-repaired-cover-support-v2/profile.json'
x=json.loads(profile.read_text());x['model_sha256']=sha(Path(x['model_path']));review=canon/'assets_local/wowrobo/randomization-bank-repaired-cover-support-v2/scene.xml';x['source_model_sha256']=sha(review);profile.write_text(json.dumps(x,indent=2)+'\n')
for base in ['assembly-so101','robotics-assembly-3.0']:
 for name in ['wowrobo_reward_regions.json','wowrobo_point_features.json']:
  p=ROOT/base/'configs'/name
  if not p.exists():continue
  x=json.loads(p.read_text());x['model_signature']=sha(review);p.write_text(json.dumps(x,indent=2)+'\n')
from assembly_policy.wowrobo.runtime import load_runtime
m,_,signature=load_runtime(profile)
bank=canon/'assets_local/wowrobo/training-debug-v1/manifest.json';x=json.loads(bank.read_text());assert x['training_qualified'] is False;x['model_signature']=signature;x['source_model_signature']=sha(review);bank.write_text(json.dumps(x,indent=2)+'\n')
(ROOT/'deployment.json').write_text(json.dumps(dict(scope='DIAGNOSTIC_UNQUALIFIED_RESETS',root=str(ROOT),model_signature=signature,physical_sha256=sha(Path(json.loads(profile.read_text())['model_path'])),review_sha256=sha(review),nq=m.nq,nv=m.nv,neq=m.neq,surface_mode=1,source_manifest=original,rebased={f:sha(ROOT/f) for f in original}),indent=2)+'\n')
print(json.dumps(dict(model_signature=signature,nq=m.nq,nv=m.nv,neq=m.neq,surface_mode=1)),flush=True)
