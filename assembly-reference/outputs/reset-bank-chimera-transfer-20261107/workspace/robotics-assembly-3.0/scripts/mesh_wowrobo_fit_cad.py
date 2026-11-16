"""Mesh unchanged STEP solids with shared Gmsh surface nodes.

Run in the optional CAD environment with gmsh, numpy and trimesh installed.
Only derived OBJ/JSON files are written. No tetrahedra or CAD healing/cuts are
performed. Deflection checks sample every triangle centroid and edge midpoint;
they are not a formal trimmed-surface Hausdorff certificate.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import trimesh


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def mesh_statistics(mesh):
    _, counts = np.unique(mesh.edges_sorted, axis=0, return_counts=True)
    return {
        "vertices": len(mesh.vertices),
        "faces": len(mesh.faces),
        "watertight": bool(mesh.is_watertight),
        "winding_consistent": bool(mesh.is_winding_consistent),
        "boundary_edges": int((counts == 1).sum()),
        "nonmanifold_edges": int((counts > 2).sum()),
        "signed_volume_mm3": float(mesh.volume * 1e9),
        "zero_area_faces": int((mesh.area_faces < 1e-24).sum()),
        "euler_number": int(mesh.euler_number),
    }


def mesh_step(source, output, options):
    import gmsh

    source, output = Path(source).resolve(), Path(output).resolve()
    before = sha256(source)
    started = time.monotonic()
    gmsh.clear()
    gmsh.model.add(source.stem)
    gmsh.model.occ.importShapes(str(source))
    gmsh.model.occ.synchronize()
    volumes = gmsh.model.getEntities(3)
    if len(volumes) != 1:
        raise ValueError(f"Expected one solid in {source}, got {len(volumes)}")
    cad_volume = gmsh.model.occ.getMass(*volumes[0])
    gmsh.model.mesh.generate(2)
    tags, coordinates, _ = gmsh.model.mesh.getNodes()
    vertices_mm = np.asarray(coordinates).reshape(-1, 3)
    ordering = np.argsort(tags)
    sorted_tags = np.asarray(tags)[ordering]

    def triangles(surface=-1):
        _, nodes = gmsh.model.mesh.getElementsByType(2, surface)
        return ordering[np.searchsorted(sorted_tags, np.asarray(nodes).reshape(-1, 3))]

    mesh = trimesh.Trimesh(vertices_mm / 1000, triangles(), process=False)
    raw = mesh_statistics(mesh)
    orientation = "native Gmsh orientation"
    if not mesh.is_winding_consistent:
        boundary = gmsh.model.getBoundary(volumes, combined=False, oriented=True)
        gmsh.model.mesh.reverse([(2, abs(tag)) for dim, tag in boundary if tag < 0])
        mesh = trimesh.Trimesh(vertices_mm / 1000, triangles(), process=False)
        orientation = "signed volume-boundary orientation"
    if mesh.volume < 0:
        mesh.invert()
        orientation += "; globally reversed outward"
    stats = mesh_statistics(mesh)
    if (
        not mesh.is_watertight
        or not mesh.is_winding_consistent
        or stats["zero_area_faces"]
    ):
        raise RuntimeError(f"Invalid conforming mesh: {stats}")
    maximum = 0.0
    records, worst = [], []
    for _, tag in gmsh.model.getEntities(2):
        indices = triangles(tag)
        if not len(indices):
            continue
        v = vertices_mm[indices]
        queries = np.vstack(
            [
                v.mean(1),
                (v[:, 0] + v[:, 1]) / 2,
                (v[:, 1] + v[:, 2]) / 2,
                (v[:, 2] + v[:, 0]) / 2,
            ]
        )
        closest, _ = gmsh.model.getClosestPoint(2, tag, queries.ravel())
        distance = np.linalg.norm(np.asarray(closest).reshape(-1, 3) - queries, axis=1)
        part_maximum = float(distance.max())
        maximum = max(maximum, part_maximum)
        records.append(
            {"surface": tag, "triangles": len(indices), "max_distance_mm": part_maximum}
        )
        for index in np.argsort(distance)[-3:]:
            worst.append(
                {
                    "surface": tag,
                    "point_mm": queries[index].tolist(),
                    "surface_distance_mm": float(distance[index]),
                }
            )
    if maximum > options["max_support_distance_mm"]:
        raise RuntimeError(f"CAD support deflection {maximum} mm exceeds limit")
    output.parent.mkdir(parents=True, exist_ok=True)
    mesh.export(output, digits=12)
    roundtrip = mesh_statistics(trimesh.load_mesh(output, process=False))
    if (
        not roundtrip["watertight"]
        or not roundtrip["winding_consistent"]
        or roundtrip["zero_area_faces"]
    ):
        raise RuntimeError(f"OBJ roundtrip failed: {roundtrip}")
    if sha256(source) != before:
        raise RuntimeError("Input STEP changed while meshing")
    result = {
        "name": source.stem,
        "source_STEP": str(source),
        "source_sha256": before,
        "source_unchanged": True,
        "output_OBJ": str(output),
        "output_sha256": sha256(output),
        "gmsh_version": gmsh.__version__,
        "options": options,
        "OBJ_decimal_digits_m": 12,
        "mesh_stats": stats,
        "raw_stats": raw,
        "OBJ_roundtrip_stats": roundtrip,
        "orientation": orientation,
        "OCC_volume_mm3": cad_volume,
        "relative_volume_difference": (mesh.volume * 1e9 - cad_volume) / cad_volume,
        "max_centroid_edge_midpoint_CAD_support_distance_mm": maximum,
        "surface_records": records,
        "worst_support_projection_queries": sorted(
            worst, key=lambda row: row["surface_distance_mm"], reverse=True
        )[:50],
        "elapsed_s": time.monotonic() - started,
        "scope": "Global shared-node surface mesh; source CAD unchanged. CAD support projections can fall outside trimmed faces; supplement with independent trimmed-surface checks.",
    }
    output.with_suffix(".json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def main():
    import gmsh

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sources", type=Path, nargs="+")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--min-mm", type=float, default=0.08)
    parser.add_argument("--max-mm", type=float, default=0.7)
    parser.add_argument("--curvature-segments", type=int, default=48)
    parser.add_argument("--max-support-distance-mm", type=float, default=0.04)
    args = parser.parse_args()
    options = {
        "minimum_size_mm": args.min_mm,
        "maximum_size_mm": args.max_mm,
        "curvature_segments_per_circle": args.curvature_segments,
        "max_support_distance_mm": args.max_support_distance_mm,
        "threads": 4,
        "algorithm": 6,
        "extend_boundary_sizes": False,
    }
    gmsh.initialize()
    try:
        for key, value in {
            "General.NumThreads": 4,
            "Mesh.MeshSizeMin": args.min_mm,
            "Mesh.MeshSizeMax": args.max_mm,
            "Mesh.MeshSizeFromCurvature": args.curvature_segments,
            "Mesh.MeshSizeExtendFromBoundary": 0,
            "Mesh.Algorithm": 6,
            "Mesh.AlgorithmSwitchOnFailure": 1,
            "Mesh.ElementOrder": 1,
        }.items():
            gmsh.option.setNumber(key, value)
        parts = [
            mesh_step(source, args.output_dir / f"{source.stem}.obj", options)
            for source in args.sources
        ]
    finally:
        gmsh.finalize()
    manifest = {
        "gmsh_version": gmsh.__version__,
        "builder_sha256": sha256(__file__),
        "options": options,
        "parts": parts,
        "scope": "Only derived OBJ/JSON outputs. No CAD cuts/healing, source overwrites or tetrahedra.",
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
