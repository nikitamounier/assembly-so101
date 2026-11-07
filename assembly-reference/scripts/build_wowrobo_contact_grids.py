"""Build true triangle-distance SDFs in untouched source OBJ coordinates.

Binary format matches assemblenist.grid_sdf: <3i4d header and float16 Z/Y/X.
The winding sign preserves open cavities; no occupancy filling or convex hulls.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import struct
import time
import xml.etree.ElementTree as ET
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import igl
import mujoco
import numpy as np
import trimesh

ROOT = Path(__file__).resolve().parents[1]


def build_one(spec):
    started = time.monotonic()
    path = Path(spec["source_file"])
    target = Path(spec["grid_file"])
    mesh = trimesh.load_mesh(path, process=False)
    v = np.asarray(mesh.vertices, dtype=np.float64)
    f = np.asarray(mesh.faces, dtype=np.int64)
    h = spec["spacing"]
    origin = np.floor((v.min(0) - 4 * h) / h) * h
    shape = np.ceil((v.max(0) + 4 * h - origin) / h).astype(int) + 1
    tree = igl.AABB()
    tree.init(v, f)
    nx, ny, nz = map(int, shape)
    xy = np.stack(
        np.meshgrid(
            origin[0] + np.arange(nx) * h, origin[1] + np.arange(ny) * h, indexing="xy"
        ),
        -1,
    ).reshape(-1, 2)
    pending = target.with_suffix(".partial")
    with pending.open("wb") as stream:
        stream.write(struct.pack("<3i4d", *shape, *origin, h))
        for k in range(nz):
            q = np.column_stack((xy, np.full(len(xy), origin[2] + k * h)))
            sq, _, _ = tree.squared_distance(v, f, q)
            winding = igl.fast_winding_number(v, f, q)
            d = np.sqrt(sq) * np.where(winding > 0.5, -1.0, 1.0)
            if not np.isfinite(d).all():
                raise ValueError(spec["name"])
            stream.write(d.astype("<f2").tobytes())
    pending.replace(target)
    spec.update(
        origin=origin.tolist(),
        shape=shape.tolist(),
        bounds=np.array([v.min(0), v.max(0)]).tolist(),
        grid_sha256=hashlib.sha256(target.read_bytes()).hexdigest(),
        source_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        elapsed_s=time.monotonic() - started,
        vertices=len(v),
        triangles=len(f),
        grid_bytes=target.stat().st_size,
        source_watertight=bool(mesh.is_watertight),
    )
    # Independent interpolation error vs triangle-distance/winding at fixed samples.
    rng = np.random.default_rng(1481)
    surface, _ = trimesh.sample.sample_surface(mesh, 1000, seed=1481)
    q = np.vstack((surface, rng.uniform(v.min(0), v.max(0), (1000, 3))))
    sq, _, _ = tree.squared_distance(v, f, q)
    w = igl.fast_winding_number(v, f, q)
    truth = np.sqrt(sq) * np.where(w > 0.5, -1.0, 1.0)
    grid = np.memmap(target, dtype="<f2", mode="r", offset=44, shape=tuple(shape[::-1]))
    coord = (q - origin) / h
    base = np.floor(coord).astype(int)
    frac = coord - base
    value = np.zeros(len(q))
    for z in (0, 1):
        for y in (0, 1):
            for x in (0, 1):
                ijk = base + [x, y, z]
                weight = np.prod(np.where(np.array([x, y, z]), frac, 1 - frac), axis=1)
                value += weight * grid[ijk[:, 2], ijk[:, 1], ijk[:, 0]].astype(float)
    err = np.abs(value - truth)
    spec["sampled_error_max_m"] = float(err.max())
    spec["sampled_error_p99_m"] = float(np.quantile(err, 0.99))
    spec["validation_sample_count"] = len(q)
    return spec


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--source",
        type=Path,
        default=ROOT / "outputs/linear-insertion-20261107/source-snapshot/scene.xml",
    )
    p.add_argument(
        "--output",
        type=Path,
        default=ROOT / "outputs/linear-insertion-updated-20261107/grids",
    )
    p.add_argument("--spacing", type=float, default=0.0004)
    p.add_argument("--workers", type=int, default=4)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    root = ET.parse(args.source).getroot()
    model = mujoco.MjModel.from_xml_path(str(args.source))
    specs = []
    for asset in root.find("asset"):
        if asset.tag != "mesh" or asset.get("name", "").startswith(("left", "right")):
            continue
        name = asset.get("name")
        mid = model.mesh(name).id
        gid = 1001 + len(specs)
        specs.append(
            {
                "name": name,
                "id": gid,
                "grid_id": gid,
                "spacing": 0.0002 if name == "pin" else args.spacing,
                "source_file": str(Path(asset.get("file")).resolve()),
                "grid_file": str((args.output / f"grid_{gid}.bin").resolve()),
                "native_mesh_pos": model.mesh_pos[mid].tolist(),
                "native_mesh_quat_wxyz": model.mesh_quat[mid].tolist(),
            }
        )
    manifest = {
        "schema_version": 1,
        "grid_directory": str(args.output.resolve()),
        "coordinate_frame": "original source OBJ meters, prior to MuJoCo principal recentering",
        "binary_format": "<3i4d header [nx,ny,nz,ox,oy,oz,h]; float16 values flattened z,y,x",
        "sign": "negative when libigl fast winding number>0.5; positive otherwise; no fill/hull operation",
        "source_scene": str(args.source.resolve()),
        "source_scene_sha256": hashlib.sha256(args.source.read_bytes()).hexdigest(),
        "meshes": {s["name"]: s for s in specs},
        "status": "building",
    }
    target = args.output / "manifest.json"
    target.write_text(json.dumps(manifest, indent=2) + "\n")
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for future in as_completed([pool.submit(build_one, s) for s in specs]):
            row = future.result()
            manifest["meshes"][row["name"]] = row
            target.write_text(json.dumps(manifest, indent=2) + "\n")
            print(json.dumps(row), flush=True)
    manifest["status"] = "complete"
    target.write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
