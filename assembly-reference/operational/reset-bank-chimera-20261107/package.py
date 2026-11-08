"""Freeze a standalone, path-relocated reset generation workspace, no local edits."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import tarfile
import xml.etree.ElementTree as ET

WORKSPACE = Path(__file__).resolve().parents[3]
CANON = WORKSPACE / 'robotics-assembly-3.0'
PLAN = CANON / 'outputs/reset-bank-1800-20261107-v4'
DEFAULT_REMOTE = '/large_storage/goodarzilab/nikitamounier/assembly/reset-bank-1800-20261107-v4'
EXCLUDE = {'__pycache__', '.git', 'node_modules', 'build', 'static'}
BINARY_EXT = {'.so', '.dylib', '.pyc', '.png', '.jpg', '.jpeg', '.mp4', '.mov', '.heic', '.3dm'}


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def build(out, remote):
    stage = out / 'workspace'
    if stage.exists():
        raise ValueError('Immutable package: choose a new output directory')
    files = set()
    def add(path, recursive=True):
        p = Path(path).absolute()
        if not p.exists():
            raise FileNotFoundError(p)
        if not p.is_relative_to(WORKSPACE):
            raise ValueError(f'Asset outside assembly workspace: {p}')
        if p.is_dir():
            for f in p.rglob('*') if recursive else p.iterdir():
                if f.is_file() and not any(x in EXCLUDE for x in f.relative_to(p).parts) and f.suffix.lower() not in BINARY_EXT:
                    files.add(f.absolute())
        else:
            files.add(p)
    for project in [CANON, WORKSPACE/'assembly-so101']:
        for folder in ['assembly_policy', 'scripts']:
            for f in (project/folder).rglob('*.py'):
                if '__pycache__' not in f.parts:
                    add(f)
        add(project/'configs')
    add(CANON/'cad/wowrobo')
    for name in ['components-repaired-v1', 'components-v2', 'helper', 'camera-reference', 'grasp-sdf-cache-v1']:
        folder = CANON/'assets_local/wowrobo'/name
        if folder.exists():add(folder)
    for n in ['assemblenist_sdf.cc','so101_grid_bounds.h','so101_surface_tree.h','so101_contact_search.c','build.sh','build_contact_search.sh']:
        add(WORKSPACE/'assembly-so101/plugin'/n)
    for name in ['config.json','plan.json','grasps.json','support-recalibration.json']:
        add(PLAN/name)
    add(PLAN/'references')
    config = json.loads((PLAN/'config.json').read_text())
    grid_seen, profile_seen = set(), set()
    def grids(path):
        path=Path(path)
        if path in grid_seen:return
        grid_seen.add(path);add(path)
        data=json.loads(path.read_text())
        for r in data['meshes'].values():
            for key in ['grid_file','source_file']:
                add(Path(r[key]))
        for m in path.parent.glob('grid_*.json'):
            add(m)
            r=json.loads(m.read_text())
            for key in ['grid_file','source_file']:
                if key in r:add(Path(r[key]))
    def profile(data):
        key=data['model_path']
        if key in profile_seen:return
        profile_seen.add(key)
        for k in ['model_path','audit_model_path','pin_mesh_path','pin_registration_path']:
            if data.get(k):add(Path(data[k]))
        if data.get('completed_anchor_bank'):add(Path(data['completed_anchor_bank']))
        if data.get('grid_manifest'):grids(Path(data['grid_manifest']))
        if data.get('mating_profile',{}).get('config_path'):
            add(Path(data['mating_profile']['config_path']))
    physical = PLAN/'physical/profile.json';add(physical)
    profile(json.loads(physical.read_text()))
    for f in (PLAN/'references/physical-replays').glob('*.json'):
        profile(json.loads(f.read_text())['physics_profile'])
    for key in ['pin_mesh','pin_registration_path','frozen_cad_ready']:
        if config.get(key):add(CANON/config[key] if not Path(config[key]).is_absolute() else Path(config[key]))
    grids(CANON/config['grid_manifest'])
    # Each imported helper uses a generated XML before loading the nominal model.
    # XML asset paths are therefore closed for EVERY included model, not only
    # the final physical one.
    while True:
        before=len(files)
        for p in list(files):
            if p.suffix=='.xml' and p.name in {'scene.xml','physical.xml','source.xml','source-scene.xml','physical-unframed.xml'}:
                root=ET.parse(p).getroot()
                compiler=root.find('compiler')
                meshdir=compiler.get('meshdir','') if compiler is not None else ''
                texturedir=compiler.get('texturedir','') if compiler is not None else ''
                for e in root.iter():
                    f=e.get('file')
                    if not f:continue
                    path=Path(f)
                    if not path.is_absolute():
                        sub=meshdir if e.tag=='mesh' else texturedir if e.tag=='texture' else ''
                        path=p.parent/sub/path
                    add(path)
        if len(files)==before:break
    for p in Path(__file__).parent.glob('*.py'):add(p)
    # Keep original evidence outside the relocated runtime tree.
    provenance=out/'provenance';provenance.mkdir(parents=True)
    shutil.copy2(PLAN/'plan.json',provenance/'original-plan.json')
    shutil.copytree(PLAN/'references/physical-replays',provenance/'original-reference-witnesses')
    manifest={}
    copied={}
    original_digests={}
    for p in sorted(files):
        rel=p.relative_to(WORKSPACE)
        target=stage/rel;target.parent.mkdir(parents=True,exist_ok=True)
        resolved=p.resolve()
        # Repeated physical profiles refer to the same immutable grid bytes.
        # Hardlinks deduplicate transfer while preserving every logical filename.
        if p.suffix in {'.bin','.obj','.stl','.STL','.npz'} and resolved in copied:
            os.link(copied[resolved],target)
        else:
            shutil.copy2(p,target)
            if p.suffix in {'.bin','.obj','.stl','.STL','.npz'}:copied[resolved]=target
        original_digests[str(rel)]=sha(target)
        manifest[str(rel)]={'source_sha256':original_digests[str(rel)],'source':str(p),'bytes':target.stat().st_size}
    # Replace paths only in structured runtime text. Source-code bytes stay exact.
    texts=[]
    for target in stage.rglob('*'):
        if target.is_file() and target.suffix in {'.json','.xml'}:
            old=target.read_text()
            new=old.replace(str(WORKSPACE),remote)
            if new!=old:target.write_text(new)
            texts.append(target)
    # Reseal digest pointers after path-only transformations. Binary grid/mesh/
    # snapshot hashes remain exact, and originals remain in provenance.
    for _ in range(12):
        mapping={v:sha(stage/k) for k,v in original_digests.items() if v!=sha(stage/k)}
        changed=0
        for path in texts:
            if path.suffix!='.json':continue
            text=path.read_text();new=text
            for old,newhash in mapping.items():new=new.replace('"'+old+'"','"'+newhash+'"')
            if new!=text:path.write_text(new);changed+=1
        if not changed:break
    else:raise RuntimeError('Digest pointer relocation did not converge')
    for rel,r in manifest.items():r['relocated_sha256']=sha(stage/rel)
    transport={'schema_version':1,'source_workspace':str(WORKSPACE),'remote_workspace':remote,
               'runtime_root':remote+'/robotics-assembly-3.0','source_plan':str(PLAN),
               'fresh_remote_identity_required':True,'local_attempts_imported':False,
               'paths_only_relocated':True,'files':manifest}
    (out/'transport-manifest.json').write_text(json.dumps(transport,indent=2)+'\n')
    # Main's convenience record; archive expands directly under the isolated root.
    with tarfile.open(out/'reset-workspace.tar.gz','w:gz',compresslevel=1,dereference=False) as archive:
        for f in sorted(stage.rglob('*')):
            if f.is_file():archive.add(f,arcname=str(f.relative_to(stage)),recursive=False)
        archive.add(out/'transport-manifest.json',arcname='transport-manifest.json')
        archive.add(provenance,arcname='transport-provenance')
    summary={'files':len(files),'archive':str(out/'reset-workspace.tar.gz'),
             'archive_bytes':(out/'reset-workspace.tar.gz').stat().st_size,
             'archive_sha256':sha(out/'reset-workspace.tar.gz'),
             'runtime_root':transport['runtime_root']}
    (out/'package-summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,default=CANON/'outputs/reset-bank-chimera-transfer-20261107')
    p.add_argument('--remote-root',default=DEFAULT_REMOTE)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True);build(a.output,a.remote_root)
