"""Read-only per-mesh cross-platform compiled-vertex diagnostic; no gate changes."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
HERE = Path(__file__).resolve().parent


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--profile",
        type=Path,
        default=ROOT / "outputs/reset-bank-1800-20261107-v4/physical/profile.json",
    )
    p.add_argument("--reference", type=Path, default=HERE / "source-model-arrays.npz")
    p.add_argument(
        "--output", type=Path, default=ROOT.parent / "mesh-portability-diagnostic.json"
    )
    a = p.parse_args()
    from assembly_policy.so101.wowrobo_contact_backend import (
        load_backend,
        set_surface_mode,
    )

    profile = json.loads(a.profile.read_text())
    load_backend(profile["grid_manifest"])
    set_surface_mode(1)
    m = mujoco.MjModel.from_xml_string(Path(profile["model_path"]).read_text())
    with np.load(a.reference) as expected:
        reference = expected["mesh_vert"].astype(np.float64)
        current = np.asarray(m.mesh_vert, dtype=np.float64)
        if reference.shape != current.shape:
            raise ValueError(f"Shape mismatch {reference.shape} {current.shape}")
        topo = {
            k: bool(np.array_equal(expected[k], getattr(m, k)))
            for k in [
                "mesh_face",
                "mesh_vertadr",
                "mesh_vertnum",
                "mesh_faceadr",
                "mesh_facenum",
                "geom_dataid",
            ]
        }
        rows = []
        for mid in range(m.nmesh):
            start, count = int(m.mesh_vertadr[mid]), int(m.mesh_vertnum[mid])
            old, new = reference[start : start + count], current[start : start + count]
            difference = new - old
            norm = np.linalg.norm(difference, axis=1)
            changed = np.any(difference != 0, axis=1)
            if not np.any(changed):
                continue
            indices = np.argsort(norm)[-5:][::-1]
            # Global affine consistency distinguishes a frame/compiler transform
            # from arbitrary geometry changes. Report residual; never rewrite data.
            affine = None
            sample = np.linspace(0, count - 1, min(count, 10000), dtype=int)
            x = np.column_stack((old[sample], np.ones(len(sample))))
            coef, _, _, _ = np.linalg.lstsq(x, new[sample], rcond=None)
            residual = new - (np.column_stack((old, np.ones(count))) @ coef)
            affine = {
                "matrix_4x3": coef.tolist(),
                "max_vertex_residual_norm_m": float(
                    np.linalg.norm(residual, axis=1).max(initial=0)
                ),
            }
            rows.append(
                {
                    "id": mid,
                    "name": m.mesh(mid).name,
                    "vertices": count,
                    "changed_vertices": int(changed.sum()),
                    "max_component_abs_m": float(np.abs(difference).max(initial=0)),
                    "max_vertex_norm_m": float(norm.max(initial=0)),
                    "reference_bounds_m": [
                        old.min(axis=0).tolist(),
                        old.max(axis=0).tolist(),
                    ],
                    "linux_bounds_m": [
                        new.min(axis=0).tolist(),
                        new.max(axis=0).tolist(),
                    ],
                    "best_affine_fit": affine,
                    "worst5": [
                        {
                            "local_vertex": int(i),
                            "global_vertex": int(start + i),
                            "reference": old[i].tolist(),
                            "linux": new[i].tolist(),
                            "difference": difference[i].tolist(),
                            "norm_m": float(norm[i]),
                        }
                        for i in indices
                    ],
                }
            )
        relevant = {
            k: float(np.max(np.abs(np.asarray(getattr(m, k)) - expected[k]), initial=0))
            for k in [
                "geom_pos",
                "geom_quat",
                "body_pos",
                "body_quat",
                "body_mass",
                "body_inertia",
            ]
        }
    rows.sort(key=lambda r: r["max_vertex_norm_m"], reverse=True)
    grid = json.loads(Path(profile["grid_manifest"]).read_text())
    checks = {}
    for name, r in grid["meshes"].items():
        checks[name] = {
            "source_file": r["source_file"],
            "source_sha_match": sha(r["source_file"]) == r["source_sha256"],
            "grid_file": r["grid_file"],
            "grid_sha_match": sha(r["grid_file"]) == r["grid_sha256"],
        }
    result = {
        "mujoco": mujoco.__version__,
        "mesh_vertex_dtype": str(m.mesh_vert.dtype),
        "topology_exact": topo,
        "other_frame_dynamics_max_abs": relevant,
        "changed_mesh_count": len(rows),
        "changed_meshes": rows,
        "source_grid_byte_checks": checks,
        "all_source_and_grid_sha_match": all(
            all((r["source_sha_match"], r["grid_sha_match"])) for r in checks.values()
        ),
        "qualification": "diagnostic only; no parity bound or data changed",
    }
    a.output.write_text(json.dumps(result, indent=2) + "\n")
    print(
        json.dumps(
            {
                "output": str(a.output),
                "changed_mesh_count": len(rows),
                "worst_meshes": [
                    {
                        k: r[k]
                        for k in ["id", "name", "changed_vertices", "max_vertex_norm_m"]
                    }
                    for r in rows[:10]
                ],
                "all_source_and_grid_sha_match": result[
                    "all_source_and_grid_sha_match"
                ],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
