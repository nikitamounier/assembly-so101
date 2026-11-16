"""Reproduce the added arm-link runtime from the hash-pinned original vendor CAD."""
import argparse
import hashlib
import json
from pathlib import Path
import struct
import urllib.request
import numpy as np
import igl
from .so101_assets import object_mesh, SHA
import rhino3dm as rh

ROOT = Path(__file__).resolve().parents[1]
SOURCE = 'https://drive.usercontent.google.com/download?id=13ffIl5rWNmyrwV8Q4zCmyH30RVABJGAM&export=download&confirm=t'
INDEX = 139
OBJECT_ID = '73c7aeca-8917-4afe-aa11-3778a4b0992f'
SPACING = .00035


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def ensure(cad=None):
    directory = ROOT/'assets_local/parts'
    manifest = directory/'arm_link.json'
    outputs = [directory/'arm_link.obj', directory/'arm_link.npz', ROOT/'assets_local/sdf/grid_107.bin']
    if manifest.exists():
        saved = json.loads(manifest.read_text())
        if (saved.get('cad_sha256') == SHA and saved.get('object_ids') == [OBJECT_ID]
                and all(p.is_file() and digest(p) == saved['files'][str(p.relative_to(ROOT))] for p in outputs)):
            print('Arm-link runtime verified', flush=True)
            return
        raise ValueError('Existing arm-link runtime differs; preserve it before regenerating')
    cad = Path(cad) if cad else ROOT/'assets_local/hiwonder/follower.3dm'
    if not cad.exists():
        cad.parent.mkdir(parents=True, exist_ok=True)
        temporary = cad.with_suffix('.downloading')
        try:
            print('Downloading pinned Hiwonder CAD to extract the arm link', flush=True)
            urllib.request.urlretrieve(SOURCE, temporary)
            if digest(temporary) != SHA:
                raise ValueError('Vendor CAD checksum mismatch; no geometry installed')
            temporary.replace(cad)
        finally:
            temporary.unlink(missing_ok=True)
    if digest(cad) != SHA:
        raise ValueError('Unreviewed vendor CAD revision')
    document = rh.File3dm.Read(str(cad))
    if document.Settings.ModelUnitSystem != rh.UnitSystem.Millimeters:
        raise ValueError('Expected millimetres in vendor CAD')
    obj = document.Objects[INDEX]
    if str(obj.Attributes.Id) != OBJECT_ID:
        raise ValueError('Unexpected vendor arm-link object')
    mesh = object_mesh(obj)
    origin = mesh.bounds.mean(0)
    mesh.vertices -= origin
    directory.mkdir(parents=True, exist_ok=True)
    mesh.export(outputs[0])
    np.savez_compressed(outputs[1], vertices=mesh.vertices, faces=mesh.faces)
    vertices, faces = np.asarray(mesh.vertices, float), np.asarray(mesh.faces, np.int64)
    tree = igl.AABB();tree.init(vertices, faces)
    winding = igl.FastWindingNumberBVH();winding.init(vertices, faces)
    grid_origin = np.floor((vertices.min(0)-4*SPACING)/SPACING)*SPACING
    shape = np.ceil((vertices.max(0)+4*SPACING-grid_origin)/SPACING).astype(int)+1
    x, y = [grid_origin[k]+np.arange(shape[k])*SPACING for k in range(2)]
    xy = np.stack(np.meshgrid(x, y, indexing='xy'), -1).reshape(-1, 2)
    grid = outputs[2];grid.parent.mkdir(parents=True, exist_ok=True)
    temporary = grid.with_suffix('.building')
    try:
        with temporary.open('wb') as stream:
            stream.write(struct.pack('<3i4d', *shape, *grid_origin, SPACING))
            for k in range(shape[2]):
                points = np.column_stack((xy, np.full(len(xy), grid_origin[2]+k*SPACING)))
                values = np.sqrt(tree.squared_distance(vertices, faces, points)[0])
                values *= np.where(winding.winding_number(points) > .5, -1., 1.)
                if not np.isfinite(values).all():
                    raise ValueError('Nonfinite arm-link SDF')
                stream.write(values.astype('<f2').tobytes())
                if k % 50 == 0:
                    print(f'Arm-link SDF slice {k}/{shape[2]}', flush=True)
        temporary.replace(grid)
    finally:
        temporary.unlink(missing_ok=True)
    record = dict(cad_sha256=SHA, indices=[INDEX], object_ids=[OBJECT_ID], origin=origin.tolist(),
                  bounds=mesh.bounds.tolist(), faces=len(mesh.faces), watertight=bool(mesh.is_watertight),
                  volume=float(mesh.volume), units='metres',
                  scope='Unmodified vendor arm link after shoulder yoke; seam merge only; non-watertight source retained',
                  sdf=dict(id=107, spacing=SPACING, origin=grid_origin.tolist(), shape=shape.tolist()),
                  files={str(p.relative_to(ROOT)):digest(p) for p in outputs})
    manifest.write_text(json.dumps(record, indent=2)+'\n')
    print('Arm-link runtime generated and verified', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cad', type=Path)
    ensure(parser.parse_args().cad)
