"""Create or install the hash-bound native SO101 runtime; original CAD is excluded."""
import argparse
import gzip
import hashlib
import io
import json
from pathlib import Path,PurePosixPath
import shutil
import subprocess
import tarfile

ROOT=Path(__file__).resolve().parents[2]


def digest(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def pack(source,output,config,release):
    source=source.resolve();files={}
    scene=json.loads((source/'configs/so101_scene.json').read_text())
    part_metadata=json.loads((source/'assets_local/parts/manifest.json').read_text())['parts']
    for directory in ('assets_local/parts','assets_local/so101','assets_local/so101_collision'):
        for path in (source/directory).rglob('*'):
            if directory=='assets_local/parts' and path.stem not in {*part_metadata,'manifest'}:continue
            if path.is_file():files[str(path.relative_to(source))]=path
    grids={int(p['sdf']['id']) for p in part_metadata.values()}|{201,202}
    for grid in sorted(grids):
        key=f'assets_local/sdf/grid_{grid}.bin';files[key]=source/key
    for path in (source/scene['bank_directory']).iterdir():
        if path.is_file():files[str(path.relative_to(source))]=path
    for suffix in ('.npz','.json'):
        path=(source/scene['reference_prefix']).with_suffix(suffix)
        files[str(path.relative_to(source))]=path
    manifest={key:dict(sha256=digest(path),bytes=path.stat().st_size) for key,path in sorted(files.items())}
    output.parent.mkdir(parents=True,exist_ok=True)
    with output.open('xb') as raw,gzip.GzipFile(fileobj=raw,mode='wb',mtime=0) as compressed,tarfile.open(fileobj=compressed,mode='w|') as tar:
        for key,path in sorted(files.items()):
            info=tarfile.TarInfo(key);info.size=path.stat().st_size;info.mode=0o644
            with path.open('rb') as stream:tar.addfile(info,stream)
        payload=json.dumps(manifest,sort_keys=True).encode();info=tarfile.TarInfo('runtime-manifest.json');info.size=len(payload);info.mode=0o644;tar.addfile(info,io.BytesIO(payload))
    bank=json.loads((source/scene['bank_directory']/'manifest.json').read_text())
    config.write_text(json.dumps(dict(repository='Alessandro-Maniscalco/robotics-assembly-3.0',release=release,asset=output.name,sha256=digest(output),bytes=output.stat().st_size,
        source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=source,text=True).strip(),
        scene_configuration_sha256=digest(source/'configs/so101_scene.json'),model_signature=bank['model_signature'],bank_directory=scene['bank_directory'],
        scope='Quest physical scene, all-loose randomized starts and independent compiled-model reference; original CAD, optional GPU-physics grids and private runs excluded',files=manifest),indent=2)+'\n')


def install(archive,destination,config):
    expected=json.loads(config.read_text())
    if digest(archive)!=expected['sha256']:raise ValueError('Runtime archive SHA-256 mismatch')
    destination=destination.resolve()
    with tarfile.open(archive,'r:gz') as tar:
        members=tar.getmembers();names=[m.name for m in members]
        if len(names)!=len(set(names)) or set(names)!=set(expected['files'])|{'runtime-manifest.json'}:
            raise ValueError('Unexpected runtime archive members')
        manifest=json.load(tar.extractfile('runtime-manifest.json'))
        if manifest!=expected['files']:raise ValueError('Runtime archive manifest mismatch')
        for m in members:
            path=PurePosixPath(m.name)
            if not m.isfile() or path.is_absolute() or '..' in path.parts:raise ValueError('Unsafe runtime member')
            if m.name=='runtime-manifest.json':continue
            target=destination/m.name
            if not target.resolve().is_relative_to(destination):raise ValueError('Runtime path escapes checkout')
            if target.exists():
                if digest(target)!=manifest[m.name]['sha256']:raise FileExistsError(f'Existing runtime file differs: {target}')
                continue
            target.parent.mkdir(parents=True,exist_ok=True)
            temporary=target.with_suffix(target.suffix+'.installing')
            try:
                with temporary.open('xb') as out,tar.extractfile(m) as source:shutil.copyfileobj(source,out)
                if temporary.stat().st_size!=manifest[m.name]['bytes'] or digest(temporary)!=manifest[m.name]['sha256']:
                    raise ValueError(f'Runtime member failed integrity: {m.name}')
                temporary.replace(target)
            finally:temporary.unlink(missing_ok=True)
    print('Native runtime installed and verified',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='command',required=True)
    q=sub.add_parser('pack');q.add_argument('--source',type=Path,default=ROOT);q.add_argument('--output',type=Path,required=True);q.add_argument('--release',default='native-runtime-v2');q.add_argument('--config',type=Path,default=ROOT/'configs/runtime_bundle.json')
    q=sub.add_parser('install');q.add_argument('archive',type=Path);q.add_argument('--destination',type=Path,default=ROOT);q.add_argument('--config',type=Path,default=ROOT/'configs/runtime_bundle.json')
    a=p.parse_args()
    if a.command=='pack':pack(a.source,a.output,a.config,a.release)
    else:install(a.archive,a.destination,a.config)
