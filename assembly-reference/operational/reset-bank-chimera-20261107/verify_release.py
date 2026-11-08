"""Verify a freshly relocated reset workspace before allocating any bank identity."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import time


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True)
    a=p.parse_args();root=a.root.resolve();transport=json.loads((root/'transport-manifest.json').read_text())
    remote=Path(transport['remote_workspace'])
    def disk(value):
        path=Path(value)
        return root/path.relative_to(remote) if path.is_absolute() else root/'robotics-assembly-3.0'/path
    bad=[]
    for rel,record in transport['files'].items():
        path=root/rel
        if not path.is_file() or sha(path)!=record['relocated_sha256']:bad.append(rel)
    if bad:raise RuntimeError(f'Relocated payload bytes changed: {bad[:10]}')
    plan_dir=root/'robotics-assembly-3.0/outputs/reset-bank-1800-20261107-v4'
    plan=json.loads((plan_dir/'plan.json').read_text());config=json.loads((plan_dir/'config.json').read_text())
    profile=json.loads((plan_dir/'physical/profile.json').read_text())
    checks={
        'config':plan['config_sha256']==sha(plan_dir/'config.json'),
        'grasp_library':plan['grasp_library_sha256']==sha(plan_dir/'grasps.json'),
        'physical_model':profile['model_sha256']==sha(disk(profile['model_path'])),
        'nominal_model':profile['source_model_sha256']==sha(disk(config['nominal_bank'])/'scene.xml'),
        'pin_registration':profile['pin_registration_sha256']==sha(disk(profile['pin_registration_path'])),
        'input_grid_manifest':profile['input_grid_manifest_sha256']==sha(disk(config['grid_manifest'])),
    }
    for name,value in profile['completed_anchor_sha256'].items():
        checks['anchor:'+name]=value==sha(disk(profile['completed_anchor_bank'])/(name+'_train.npz'))
    for name,record in json.loads(disk(profile['grid_manifest']).read_text())['meshes'].items():
        checks['source_mesh:'+name]=record['source_sha256']==sha(disk(record['source_file']))
        checks['grid:'+name]=record['grid_sha256']==sha(disk(record['grid_file']))
    library=json.loads((plan_dir/'grasps.json').read_text())
    checks['library_model']=library['model_signature']==sha(disk(config['nominal_bank'])/'scene.xml')
    if not all(checks.values()):raise RuntimeError(f'Relocation digest pointer checks failed: {[k for k,v in checks.items() if not v]}')
    result=dict(status='verified',verified_at_unix=time.time(),files=len(transport['files']),checks=checks,
                original_source_plan=transport['source_plan'],fresh_remote_identity=True,
                local_attempts_imported=False,local_witnesses_preserved='transport-provenance/original-reference-witnesses',
                relocated_physical_model_sha256=profile['model_sha256'])
    (root/'relocation-complete.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='checks'}))


if __name__=='__main__':main()
