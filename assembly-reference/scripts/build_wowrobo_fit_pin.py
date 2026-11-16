"""Build an isolated printable pin and native mesh without changing frozen assets."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import trimesh

ROOT = Path(__file__).resolve().parents[1]


def pin_mesh(p):
    n = p["facets"]
    theta = np.arange(n) * 2 * np.pi / n
    rings = [
        (p["grip_diameter"] / 2, 0),
        (p["grip_diameter"] / 2, p["grip_length"]),
        (p["shaft_diameter"] / 2, p["grip_length"]),
        (p["shaft_diameter"] / 2, p["total_length"]),
    ]
    vertices = np.array(
        [[r * np.cos(t), r * np.sin(t), z] for r, z in rings for t in theta]
    )
    faces = []
    for ring in range(3):
        for i in range(n):
            a, b = ring * n + i, ring * n + (i + 1) % n
            faces.extend([(a, b, b + n), (a, b + n, a + n)])
    vertices = np.vstack([vertices, [0, 0, 0], [0, 0, p["total_length"]]])
    for i in range(n):
        faces.extend(
            [(4 * n, (i + 1) % n, i), (4 * n + 1, 3 * n + i, 3 * n + (i + 1) % n)]
        )
    return trimesh.Trimesh(vertices, faces, process=True)


def main():
    import cadquery as cq

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=ROOT / "configs/wowrobo_components.json"
    )
    parser.add_argument(
        "--output", type=Path, default=ROOT / "assets_local/wowrobo/fit-pin-v2"
    )
    a = parser.parse_args()
    cfg = json.loads(a.config.read_text())
    p = cfg["pin_mm"]
    out = a.output
    out.mkdir(parents=True, exist_ok=True)
    mesh = pin_mesh(p)
    assert mesh.is_watertight
    mesh.export(out / "assembly_pin_print_mm.stl")
    native = mesh.copy()
    native.apply_scale(0.001)
    native.export(out / "pin.obj")
    shape = (
        cq.Workplane("XY")
        .circle(p["grip_diameter"] / 2)
        .extrude(p["grip_length"])
        .union(
            cq.Workplane("XY")
            .workplane(offset=p["grip_length"] - p["overlap"])
            .circle(p["shaft_diameter"] / 2)
            .extrude(p["shaft_length"] + p["overlap"])
        )
    )
    cq.exporters.export(shape, str(out / "assembly_pin.step"))
    (out / "assembly_pin.scad").write_text(
        (ROOT / "cad/wowrobo/assembly_pin.scad").read_text()
    )
    files = {
        name: {
            "sha256": hashlib.sha256((out / name).read_bytes()).hexdigest(),
            "units": "m" if name.endswith(".obj") else "mm",
        }
        for name in [
            "pin.obj",
            "assembly_pin_print_mm.stl",
            "assembly_pin.step",
            "assembly_pin.scad",
        ]
    }
    manifest = {
        "profile": cfg["pin_fit"]["profile"],
        "pin_mm": p,
        "fit": cfg["pin_fit"],
        "files": files,
        "watertight": mesh.is_watertight,
        "native_mesh_bounds_m": native.bounds.tolist(),
        "native_volume_m3": float(native.volume),
        "center_mass_m": native.center_mass.tolist(),
        "inertia_at_density_1_kg_m3": native.moment_inertia.tolist(),
        "training_eligible": False,
        "frozen_assets_modified": False,
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
