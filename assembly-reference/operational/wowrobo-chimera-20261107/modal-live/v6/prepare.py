"""Freeze eligible accepted witnesses, deduplicate logical slots, retain split provenance."""
from pathlib import Path
import hashlib,json,shutil,tarfile
import numpy as np,mujoco
ROOT=Path(__file__).resolve().parent.parent;C=ROOT/'robotics-assembly-3.0';HERE=Path(__file__).parent
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
local=HERE/'local';local.mkdir(exist_ok=True)
with tarfile.open(ROOT/'states-v6.tar.gz') as t:t.extractall(local,filter='data')
source=Path('/large_storage/goodarzilab/nikitamounier/assembly/reset-bank-1800-20261107-v4/robotics-assembly-3.0/outputs/reset-bank-1800-20261107-v4')
# Both are Linux models generated from the same source; require exact numerical identity.
from assembly_policy.so101.wowrobo_contact_backend import load_backend,set_surface_mode
profile=C/'outputs/equilibrium-model-20261107/profile-centered-v6/profile.json'
p=json.loads(profile.read_text());q=json.loads((source/'physical/profile.json').read_text())
load_backend(p['grid_manifest']);set_surface_mode(1)
a=mujoco.MjModel.from_xml_string(Path(p['model_path']).read_text());b=mujoco.MjModel.from_xml_string(Path(q['model_path']).read_text())
assert sha(q['model_path'])==q['model_sha256']=='1b071d9b92dc5e3e3e9ba33234a25f24bfc3b3049081a52ac673b1df423b952d'
fields='body_parentid body_jntadr body_jntnum body_dofadr body_dofnum body_pos body_quat body_ipos body_iquat body_mass body_inertia body_gravcomp jnt_type jnt_bodyid jnt_qposadr jnt_dofadr jnt_pos jnt_axis jnt_range jnt_limited jnt_stiffness jnt_margin dof_bodyid dof_jntid dof_armature dof_damping dof_frictionloss geom_type geom_bodyid geom_dataid geom_pos geom_quat geom_size geom_contype geom_conaffinity geom_condim geom_friction geom_solref geom_solimp geom_margin geom_gap geom_priority mesh_vert mesh_face mesh_vertadr mesh_vertnum mesh_faceadr mesh_facenum actuator_trntype actuator_trnid actuator_gainprm actuator_biasprm actuator_ctrlrange actuator_forcerange actuator_gear actuator_ctrllimited actuator_forcelimited eq_type eq_obj1id eq_obj2id eq_active0 eq_data eq_solref eq_solimp pair_geom1 pair_geom2 pair_dim pair_friction pair_solref pair_solimp pair_margin pair_gap exclude_signature qpos0'.split()
for k in fields:np.testing.assert_allclose(getattr(a,k),getattr(b,k),rtol=0,atol=1e-14,err_msg=k)
for k in 'nq nv nu na nbody njnt ngeom nmesh neq npair nexclude nplugin nuserdata'.split():assert getattr(a,k)==getattr(b,k),k
for k in 'timestep gravity integrator solver cone jacobian iterations tolerance noslip_iterations noslip_tolerance disableflags enableflags'.split():np.testing.assert_array_equal(getattr(a.opt,k),getattr(b.opt,k),err_msg=k)
ga=json.loads(Path(p['grid_manifest']).read_text())['meshes'];gb=json.loads(Path(q['grid_manifest']).read_text())['meshes'];assert ga.keys()==gb.keys()
for name in ga:
 for k in ('grid_sha256','source_sha256','spacing','origin','shape','native_mesh_pos','native_mesh_quat'):
  assert ga[name].get(k)==gb[name].get(k),(name,k)
 for x in (ga[name],gb[name]):
  for key,h in [('source_file','source_sha256'),('grid_file','grid_sha256')]:
   path=Path(x[key]);path=path if path.exists() else Path(gb[name][key]);assert sha(path)==x[h]
(HERE/'linux-physics-parity.json').write_text(json.dumps(dict(status='PASS',fields=len(fields),source_model=q['model_sha256'],tolerance=1e-14,source_cad_and_contact_grid_bytes_exact=True))+'\n')
print('Linux physics and original CAD/contact fields exact',flush=True)
out=HERE/'accepted-states-v6';out.mkdir(exist_ok=False);shutil.copytree(local/'nominals',out/'nominals')
chosen={}
def consider(receipt,archive,platform):
 r=json.loads(receipt.read_text())
 if not r.get('training_eligible') or r.get('candidate_rejected') or r.get('rejection_reasons'):return
 assert r['physical']['minimum_validity']['minimum_validity_pass']
 expected={'mac':'909e034485939b77037f9568e59d7d027e37238372e95995c1a8222803ee96ef','linux':q['model_sha256']}[platform]
 assert r['physical']['physics_profile']['model_sha256']==expected
 with np.load(archive) as z:
  for key,h in [('states','qualified_state_sha256'),('controls','qualified_control_sha256')]:assert hashlib.sha256(z[key][0].tobytes()).hexdigest()==r[h]
 key=(r['stage'],r['split'],r['slot']);assert r['split'] in ('train','eval')
 assert 0<=r['slot']<(60 if r['split']=='train' else 12)
 # Linux platform duplicates of the same frozen logical seed are not extra states.
 if key not in chosen:chosen[key]=(receipt,archive,r,platform)
for receipt in sorted((source/'shards').glob('*/attempts/*.json')):
 try:r=json.loads(receipt.read_text())
 except json.JSONDecodeError:continue
 if r.get('qualified_archive'):consider(receipt,receipt.parent.parent/r['qualified_archive'],'linux')
frozen=json.loads((local/'manifest.json').read_text())
for r in frozen['records']:
 archive=local/r['archive'];assert sha(archive)==r['archive_sha256'];receipt=archive.with_suffix('.json');assert sha(receipt)==r['receipt_sha256'];consider(receipt,archive,'mac')
records=[];hashes={}
for key,(receipt,archive,r,platform) in sorted(chosen.items()):
 fingerprint=(r['qualified_state_sha256'],json.dumps(r['metadata']['friction_multipliers'],sort_keys=True))
 assert fingerprint not in hashes,('duplicate state or split leakage',key,hashes.get(fingerprint));hashes[fingerprint]=key
 dest=out/r['split']/r['stage'];dest.mkdir(parents=True,exist_ok=True)
 for f in (archive,receipt):shutil.copy2(f,dest/f.name)
 records.append(dict(stage=r['stage'],split=r['split'],slot=r['slot'],effective_seed=r['metadata'].get('effective_seed'),platform=platform,receipt=str((dest/receipt.name).relative_to(out)),receipt_sha256=sha(receipt),archive=str((dest/archive.name).relative_to(out)),archive_sha256=sha(archive),state_sha256=r['qualified_state_sha256'],friction_multipliers=r['metadata']['friction_multipliers']))
manifest=dict(prototype_count=25,accepted_train=sum(r['split']=='train' for r in records),accepted_eval=sum(r['split']=='eval' for r in records),records=records,logical_slot_union=True,eval_heldout=True)
(out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n');print(json.dumps({k:v for k,v in manifest.items() if k!='records'}),flush=True)
