"""Compare compound occupancy with native SDF at sampled void/material points."""

import ctypes
import importlib.util
import json
import sys
from pathlib import Path

import mujoco
import numpy as np
from scipy.spatial import ConvexHull

root, output = map(Path, sys.argv[1:3])
meta = json.loads(
    (
        root
        / "robotics-assembly-3.0/assets_local/wowrobo/training-accepted-v5/manifest.json"
    ).read_text()
)
profile = json.loads(Path(meta["profile_path"]).read_text())
name = "_wowrobo_training_native_backend"
spec = importlib.util.spec_from_file_location(
    name,
    root / "robotics-assembly-3.0/assembly_policy/so101/wowrobo_contact_backend.py",
)
backend = importlib.util.module_from_spec(spec)
sys.modules[name] = backend
spec.loader.exec_module(backend)
backend.load_backend(profile["grid_manifest"])
m = mujoco.MjModel.from_binary_path(str(output / "candidate/source-model.mjb"))
d = mujoco.MjData(m)
mujoco.mj_forward(m, d)
receipt = json.loads((output / "candidate/receipt.json").read_text())
native = sys.modules["_wowrobo_training_native_backend"].surface_library()
query = native.wowrobo_sdf_distance
query.argtypes = [
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_int,
    np.ctypeslib.ndpointer(dtype=np.float64, flags="C_CONTIGUOUS"),
]
query.restype = ctypes.c_double
batch = ctypes.CDLL(str(output / "distance-batch.so")).proxy_distance_batch
batch.argtypes = [
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_int,
    np.ctypeslib.ndpointer(dtype=np.float64, flags="C_CONTIGUOUS"),
    ctypes.c_size_t,
    np.ctypeslib.ndpointer(dtype=np.float64, flags="C_CONTIGUOUS"),
]
batch.restype = None
rng = np.random.default_rng(119013)
rows = []
seen = set()
try:
    for record in receipt["records"]:
        if not record["fine"] or record["key"] in seen:
            continue
        seen.add(record["key"])
        visual = m.geom(record["visual"]).id
        target = m.geom(record["name"]).id
        mid = int(m.geom_dataid[visual])
        va, vn = m.mesh_vertadr[mid], m.mesh_vertnum[mid]
        vertices = np.asarray(m.mesh_vert[va : va + vn], float)
        fa, fn = m.mesh_faceadr[mid], m.mesh_facenum[mid]
        np.savez_compressed(
            output / ("source-" + record["key"][:12] + ".npz"),
            vertices=vertices,
            faces=m.mesh_face[fa : fa + fn],
        )
        points = rng.uniform(
            vertices.min(0) - 0.001, vertices.max(0) + 0.001, (30000, 3)
        )
        chosen = vertices[
            rng.choice(len(vertices), min(5000, len(vertices)), replace=False)
        ]
        points = np.concatenate([points, chosen + rng.normal(0, 0.0004, chosen.shape)])
        inside = np.zeros(len(points), bool)
        hull_centers = []
        for path in record["paths"]:
            v = np.array(
                [
                    [float(x) for x in line.split()[1:4]]
                    for line in Path(path).read_text().splitlines()
                    if line.startswith("v ")
                ]
            )
            equations = ConvexHull(v).equations
            inside |= np.all(
                points @ equations[:, :3].T + equations[:, 3] <= 1e-10, axis=1
            )
            hull_centers.append(v.mean(0))
        world = points @ d.geom_xmat[visual].reshape(3, 3).T + d.geom_xpos[visual]
        world = np.ascontiguousarray(world)
        distances = np.empty(len(world))
        batch(
            ctypes.cast(query, ctypes.c_void_p),
            m._address,
            d._address,
            target,
            world,
            len(world),
            distances,
        )
        far_void = distances > 0.0005
        deep_material = distances < -0.0005
        extra = inside & far_void
        missing = ~inside & deep_material
        row = {
            "name": record["name"],
            "samples": len(points),
            "hulls": record["count"],
            "false_fill_over_0_5mm": int(extra.sum()),
            "missing_material_over_0_5mm": int(missing.sum()),
            "void_samples": int(far_void.sum()),
            "material_samples": int(deep_material.sum()),
            "maximum_sdf_inside_proxy_m": float(distances[inside].max())
            if inside.any()
            else None,
        }
        rows.append(row)
        np.savez_compressed(
            output / ("occupancy-" + record["key"][:12] + ".npz"),
            points=points,
            source_distance=distances,
            inside_proxy=inside,
        )
        print(json.dumps(row), flush=True)
    result = {
        "rows": rows,
        "scope": "Seeded native-SDF occupancy panel, not a global error bound or physical insertion certification",
        "false_fill_over_0_5mm": sum(r["false_fill_over_0_5mm"] for r in rows),
        "missing_material_over_0_5mm": sum(
            r["missing_material_over_0_5mm"] for r in rows
        ),
    }
    (output / "geometry-check.json").write_text(json.dumps(result, indent=2) + "\n")
finally:
    pass
