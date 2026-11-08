"""Promote the qualified exact source delta; does not deploy any live process."""
from pathlib import Path
import hashlib,json
here=Path(__file__).resolve().parent
workspace=here.parents[2]
files=[workspace/'assembly-so101/plugin/so101_surface_tree.h',workspace/'robotics-assembly-3.0/assembly_policy/so101/wowrobo_contact_cache.h',workspace/'robotics-assembly-3.0/assembly_policy/so101/wowrobo_contact_backend.py']
identity=json.loads((workspace/'robotics-assembly-3.0/outputs/hinge-profile-20261107/v13/panel-identity.json').read_text())['sources']
for p in files[1:]:
 assert hashlib.sha256(p.read_bytes()).hexdigest()==identity[str(p.relative_to(workspace))], 'Concurrent source change: '+str(p)
assert hashlib.sha256(files[0].read_bytes()).hexdigest()==json.loads((here/'baseline/result.json').read_text())['hashes'][files[0].name]
backup=here/'source-before';backup.mkdir(exist_ok=True)
for p in files:
 target=backup/p.name
 if not target.exists():target.write_bytes(p.read_bytes())
p=files[0];s=(here/'so101_surface_tree.candidate.h').read_text();s=s.replace('std::array<std::array<SweptSurfaceCandidates,4>,1021> swept;', '// Keep multiple relative-pose regions per pair so alternating worlds do\n  // not immediately evict one another. Every hit rechecks conservative bounds.\n  std::array<std::array<SweptSurfaceCandidates,4>,1021> swept;');p.write_text(s)
p=files[1];s=p.read_text();old='if(!e.internal)return false;';assert s.count(old)==1;s=s.replace(old,'// The same conservative grid proof applies to external pairs too.\n  if(!e.internal) { if(!assemblenist_grid_domain_radius(m,d,first,e.radius))return false; }');p.write_text(s)
p=files[2];s=p.read_text();old='surface_mode!=2 && wow_sdf_pair_clear(m,d,g1,g2)';assert s.count(old)==1;s=s.replace(old,'surface_mode!=2 && margin<=0 && wow_sdf_pair_clear(m,d,g1,g2)');p.write_text(s)
(here/'promoted-source-hashes.json').write_text(json.dumps({str(p.relative_to(workspace)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files},indent=2)+'\n')
print('Promoted three canonical source files; live deployment unchanged')
