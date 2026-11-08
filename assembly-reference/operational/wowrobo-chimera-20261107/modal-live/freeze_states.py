"""Freeze only physically accepted, hash-verified generator output; hold EVAL apart."""
from pathlib import Path
import datetime,hashlib,json,shutil
import numpy as np

ROOT=Path(__file__).resolve().parents[3]
SOURCE=ROOT/'outputs/reset-bank-1800-20261107-v4'
OUT=Path(__file__).parent/('states-'+datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ'))
OUT.mkdir()
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def array_sha(a):return hashlib.sha256(a.tobytes()).hexdigest()
records=[]
for p in sorted((SOURCE/'shards').glob('*/attempts/*.json')):
 try:r=json.loads(p.read_text())
 except json.JSONDecodeError:continue
 if not r.get('training_eligible') or r.get('candidate_rejected') or r.get('rejection_reasons'):continue
 assert r['physical']['minimum_validity']['minimum_validity_pass']
 assert r['physical']['physics_profile']['model_sha256']=='909e034485939b77037f9568e59d7d027e37238372e95995c1a8222803ee96ef'
 zpath=p.parent.parent/r['qualified_archive']
 with np.load(zpath) as z:
  assert array_sha(z['states'][0])==r['qualified_state_sha256']
  assert array_sha(z['controls'][0])==r['qualified_control_sha256']
 dest=OUT/r['split']/r['stage'];dest.mkdir(parents=True,exist_ok=True)
 shutil.copy2(zpath,dest/zpath.name);shutil.copy2(p,dest/p.name)
 records.append(dict(stage=r['stage'],split=r['split'],receipt=str(p.relative_to(ROOT)),receipt_sha256=sha(p),archive=str((dest/zpath.name).relative_to(OUT)),archive_sha256=sha(dest/zpath.name),state_sha256=r['qualified_state_sha256'],friction_multipliers=r['metadata']['friction_multipliers']))
nominal=ROOT/'assets_local/wowrobo/initial-valid-resets-v1-physical'
shutil.copytree(nominal,OUT/'nominals')
result=dict(source=str(SOURCE),config_sha256=sha(SOURCE/'config.json'),prototype_count=25,accepted_train=sum(r['split']=='train' for r in records),accepted_eval=sum(r['split']=='eval' for r in records),records=records,friction_required=True,live_training_imported=False,reason='Frozen accepted rows; random friction needs verified native rollout integration before import. Evaluation rows remain held out.')
(OUT/'manifest.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({k:v for k,v in result.items() if k!='records'}));print(OUT)
