"""Isolated approximate-physics release; original native source untouched."""
from pathlib import Path
import hashlib,json,shutil,tarfile
ROOT=Path(__file__).resolve().parent;REL=ROOT/'bare-v8';AD=REL/'assembly-so101';C=ROOT/'robotics-assembly-3.0';PILOT=ROOT.parent/'bare-assembly-20261107'
assert not (REL/'release.json').exists()
shutil.copytree(ROOT/'assembly-so101/assembly_policy',AD/'assembly_policy',ignore=shutil.ignore_patterns('__pycache__'),dirs_exist_ok=True)
for name in ['configs','assets_local']:
 if (ROOT/'assembly-so101'/name).exists():shutil.copytree(ROOT/'assembly-so101'/name,AD/name,dirs_exist_ok=True)
# Worker dependencies remain pinned by the existing image.
shutil.copy2(PILOT/'model.mjb',REL/'model.mjb');shutil.copy2(PILOT/'scene.xml',REL/'scene.xml');shutil.copy2(PILOT/'scene.json',REL/'scene.json');shutil.copy2(PILOT/'wowrobo_bare.py',REL/'wowrobo_bare.py')
scene=json.loads((REL/'scene.json').read_text());assert hashlib.sha256((REL/'scene.xml').read_bytes()).hexdigest()==scene['scene_sha256']
s=(AD/'assembly_policy/wowrobo/env.py').read_text();a=s.index('        safety = unstable | (penetration >');b=s.index('        success = np.zeros',a);s=s[:a]+'        safety = unstable.copy()\n'+s[b:];a=s.index('            safety[i] |= bool(');b=s.index('            result = self.reward.snapshot',a);s=s[:a]+s[b:];(AD/'assembly_policy/wowrobo/env.py').write_text(s)
assert s[s.index('class BatchedAssemblyEnv:'):].strip()==(PILOT/'original-task-bare-env.py').read_text().strip(),'Executed environment must match qualified same-task pilot'
(AD/'assembly_policy/wowrobo/runtime.py').write_text('''"""Frozen, explicitly approximate MuJoCo runtime. No SDF loader/callbacks."""\nimport hashlib,json,os\nfrom pathlib import Path\nimport mujoco\nROOT=Path(os.environ['WOWROBO_CANONICAL_ROOT']);DEFAULT_PROFILE=ROOT/'assets_local/wowrobo/physical-resets-repaired-cover-support-v2/profile.json'\nDEFAULT_REVIEW=ROOT/'assets_local/wowrobo/randomization-bank-repaired-cover-support-v2/scene.xml'\nDEFAULT_BANK=ROOT/'assets_local/wowrobo/training-pilot-v1'\nREL=ROOT.parent/'bare-v8'\n_MODEL=None\ndef load_runtime(profile=DEFAULT_PROFILE):\n global _MODEL\n receipt=json.loads((REL/'scene.json').read_text());release=json.loads((REL/'release.json').read_text())\n assert hashlib.sha256((REL/'model.mjb').read_bytes()).hexdigest()==release['model_mjb_sha256']\n if _MODEL is None:\n  _MODEL=mujoco.MjModel.from_binary_path(str(REL/'model.mjb'))\n  assert _MODEL.nplugin==0 and _MODEL.npair==0 and abs(_MODEL.opt.timestep-.002)<1e-12\n return _MODEL,json.loads(Path(profile).read_text()),receipt['scene_sha256']\n''')
p=AD/'assembly_policy/train.py';s=p.read_text();s=s.replace('self.args["model_signature"] = bank_meta["model_signature"]','self.args["source_model_signature"] = bank_meta["model_signature"]\n            self.args["model_signature"] = __import__("assembly_policy.wowrobo.runtime",fromlist=["load_runtime"]).load_runtime(bank_meta["profile_path"])[2]');s=s.replace('"wowrobo-native-source-sdf"','"bare-mujoco-boxes-v8"').replace('"DIAGNOSTIC_UNQUALIFIED_RESETS" if a.wowrobo_debug_resets else "CURRENT_ASSEMBLY_TRAINING"','"APPROXIMATE_ASSEMBLY_TRAINING"');p.write_text(s)
files={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in REL.rglob('*') if p.is_file() and p.name not in ('prepare.py','release.json')}
receipt=dict(status='PREPARED',physics_backend='bare-mujoco-boxes-v8',physics_signature=scene['scene_sha256'],model_mjb_sha256=hashlib.sha256((REL/'model.mjb').read_bytes()).hexdigest(),qualified_pilot_environment_exact=True,task_reward_actions_resets_CAD_inputs_unchanged=True,penetration_velocity_terminations=False,inter_part_collision=False,robot_self_collision=False,source_root=str(ROOT),files=files)
(REL/'release.json').write_text(json.dumps(receipt,indent=2)+'\n')
with tarfile.open(REL/'overlay.tar.gz','w:gz',compresslevel=1) as t:
 for p in sorted(REL.rglob('*')):
  if p.is_file() and p.name not in ('overlay.tar.gz','prepare.py') and '__pycache__' not in p.parts:t.add(p,arcname=str(p).lstrip('/'),recursive=False)
print(json.dumps({k:v for k,v in receipt.items() if k!='files'}),flush=True)
