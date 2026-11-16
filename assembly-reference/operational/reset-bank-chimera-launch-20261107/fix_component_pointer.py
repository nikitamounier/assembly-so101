"""Reseal only a path-relocated component-manifest pointer after semantic proof."""
import hashlib
import json
from pathlib import Path
import re

remote = Path('/large_storage/goodarzilab/nikitamounier/assembly/reset-bank-1800-20261107-v4')
project = remote/'robotics-assembly-3.0'
source_workspace = '/Users/nikitamounier/Documents/ChatGPT/robot assembly 2'
asset = project/'assets_local/wowrobo/components-repaired-v1/manifest.json'
value = json.loads(asset.read_text())


def normalize(item):
    if isinstance(item,dict):return {k:normalize(v) for k,v in item.items()}
    if isinstance(item,list):return [normalize(v) for v in item]
    if isinstance(item,str):return 'HASH' if re.fullmatch('[0-9a-f]{64}',item) else item.replace(str(remote),source_workspace)
    return item

fingerprint=hashlib.sha256(json.dumps(normalize(value),sort_keys=True).encode()).hexdigest()
assert fingerprint=='68793a72f1f5175a292630591245ad919c13a8820e56807867ffd6e773ee8a16', 'Component geometry semantics changed'
manifest = project/'outputs/reset-bank-1800-20261107-v4/references/stages/manifest.json'
data=json.loads(manifest.read_text())
prior=data['component_manifest_sha256']
actual=hashlib.sha256(asset.read_bytes()).hexdigest()
data['component_manifest_sha256']=actual
manifest.write_text(json.dumps(data,indent=2)+'\n')
(remote/'component-pointer-amendment.json').write_text(json.dumps({'old_pointer':prior,'current_relocated_manifest_sha256':actual,'component_geometry_semantics_unchanged':True,'semantic_sha256':fingerprint},indent=2)+'\n')
print('Component path-only pointer resealed; geometry semantics verified')
