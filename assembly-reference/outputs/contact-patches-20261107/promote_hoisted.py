"""Promote only the qualified exact grid optimization; attachment trials stay isolated."""
from pathlib import Path
import tempfile,shutil,json,hashlib
from hoist import apply
here=Path(__file__).resolve().parent;workspace=here.parents[2];plugin=workspace/'assembly-so101/plugin'
names=['assemblenist_sdf.cc','so101_contact_search.c','so101_surface_tree.h']
original_identity=json.loads((workspace/'robotics-assembly-3.0/outputs/hinge-profile-20261107/v13/panel-identity.json').read_text())['sources']
previous_promoted=json.loads((workspace/'robotics-assembly-3.0/outputs/contact-separation-20261107/promoted-source-hashes.json').read_text())
for name in names:
 path=plugin/name;expected=(previous_promoted if name=='so101_surface_tree.h' else original_identity)[str(path.relative_to(workspace))]
 assert hashlib.sha256(path.read_bytes()).hexdigest()==expected,'Concurrent source change: '+str(path)
backup=here/'source-before-hoisted';backup.mkdir(exist_ok=True)
with tempfile.TemporaryDirectory(dir=here) as tmp:
 stage=Path(tmp)
 for name in names:shutil.copy2(plugin/name,stage/name)
 apply(stage)
 expected=json.loads((here/'hoisted/result.json').read_text())['hashes']['assemblenist_sdf.cc']
 assert hashlib.sha256((stage/'assemblenist_sdf.cc').read_bytes()).hexdigest()==expected,'Staged evaluator differs from qualified candidate'
 for name in names:
  shutil.copy2(plugin/name,backup/name);shutil.copy2(stage/name,plugin/name)
(here/'promoted-source-hashes.json').write_text(json.dumps({str((plugin/name).relative_to(workspace)):hashlib.sha256((plugin/name).read_bytes()).hexdigest() for name in names},indent=2)+'\n')
print('Promoted exact grid evaluation in three canonical plugin files; physical attachment trials not promoted')
