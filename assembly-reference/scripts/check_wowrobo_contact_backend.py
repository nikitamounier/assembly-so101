"""Independent exhaustive-vs-tree contact and source-triangle SDF validation.

Finite panels establish exact acceleration parity on sampled trajectories, not
continuous collision coverage or physically qualified assembly insertion.
"""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import struct
from pathlib import Path

import mujoco
import numpy as np
import trimesh

ROOT = Path(__file__).resolve().parents[1]


def grid_values(path, points):
    """Independent reproduction of native trilinear f16 grid_eval distance."""
    header = struct.unpack("<3i4d", Path(path).read_bytes()[:44])
    n = np.array(header[:3])
    origin = np.array(header[3:6])
    h = header[6]
    grid = np.memmap(path, dtype="<f2", mode="r", offset=44, shape=tuple(n[::-1]))
    f = (np.asarray(points) - origin) / h
    c = np.clip(f, 0, n - 1.000001)
    i = np.floor(c).astype(int)
    t = c - i

    def at(x, y, z):
        return grid[i[:, 2] + z, i[:, 1] + y, i[:, 0] + x].astype(float)

    c00 = at(0, 0, 0) + (at(1, 0, 0) - at(0, 0, 0)) * t[:, 0]
    c10 = at(0, 1, 0) + (at(1, 1, 0) - at(0, 1, 0)) * t[:, 0]
    c01 = at(0, 0, 1) + (at(1, 0, 1) - at(0, 0, 1)) * t[:, 0]
    c11 = at(0, 1, 1) + (at(1, 1, 1) - at(0, 1, 1)) * t[:, 0]
    c0 = c00 + (c10 - c00) * t[:, 1]
    c1 = c01 + (c11 - c01) * t[:, 1]
    return c0 + (c1 - c0) * t[:, 2] + np.linalg.norm((f - c) * h, axis=1), h


def stats(native, reset=True):
    values = np.zeros(7, dtype=np.float64)
    native.so101_surface_stats.argtypes = [
        ctypes.POINTER(ctypes.c_double),
        ctypes.c_int,
    ]
    native.so101_surface_stats(
        values.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), int(reset)
    )
    return values


def contacts(data):
    return np.array(
        [np.r_[c.dist, c.pos, c.frame.ravel(), c.geom] for c in data.contact],
        dtype=float,
    ).reshape(-1, 15)


def replay(model, qpos, mode, native, steps=12):
    previous = native.so101_surface_mode(mode)
    stats(native)
    try:
        d = mujoco.MjData(model)
        d.qpos[:] = qpos
        result = []
        for _ in range(steps):
            mujoco.mj_step(model, d)
            result.append((d.qpos.copy(), d.qvel.copy(), contacts(d)))
        return result, stats(native)
    finally:
        native.so101_surface_mode(previous)


def parity(left, right):
    failures = []
    for i, (a, b) in enumerate(zip(left, right, strict=True)):
        for name, x, y in zip(("qpos", "qvel", "contacts"), a, b, strict=True):
            if not np.array_equal(x, y):
                failures.append(
                    {
                        "step": i,
                        "array": name,
                        "shape_a": x.shape,
                        "shape_b": y.shape,
                        "max_absolute_difference": float(np.max(np.abs(x - y)))
                        if x.shape == y.shape and x.size
                        else None,
                    }
                )
    return failures


def mesh_grid_sanity(mesh, grid_path, seed=4511):
    """Exact winding + triangle-distance oracle, including concave empty space.

    The source servo casings contain overlapping shells (winding reaches three).
    Ray-parity sign is unsuitable for that input: the grid explicitly uses
    positive winding >0.5 as material. Compare exact solid-angle winding against
    the generator's approximate fast winding, preserving the source sign rule.
    """
    import igl

    rng = np.random.default_rng(seed)
    points = rng.uniform(*mesh.bounds, size=(512, 3))
    surface = np.asarray(mesh.vertices)
    if len(surface) > 128:
        surface = surface[np.linspace(0, len(surface) - 1, 128, dtype=int)]
    points = np.vstack((points, surface))
    vertices = np.ascontiguousarray(mesh.vertices, dtype=np.float64)
    faces = np.ascontiguousarray(mesh.faces, dtype=np.int64)
    winding = igl.winding_number(vertices, faces, np.ascontiguousarray(points))
    unsigned = trimesh.proximity.closest_point(mesh, points)[1]
    exact = np.where(winding > 0.5, -unsigned, unsigned)
    values, h = grid_values(grid_path, points)
    reliable = np.abs(exact) > 2 * h
    sign_bad = reliable & (np.sign(exact) != np.sign(values))
    hull_inside = mesh.convex_hull.contains(points)
    concavity = hull_inside & (exact > 2 * h)
    return {
        "samples": len(points),
        "grid_spacing_m": h,
        "watertight": bool(mesh.is_watertight),
        "sign_oracle": "exact solid-angle winding >0.5; preserves published source-grid sign contract",
        "winding_min": float(winding.min()),
        "winding_max": float(winding.max()),
        "max_absolute_error_m": float(np.max(np.abs(exact - values))),
        "p95_absolute_error_m": float(np.quantile(np.abs(exact - values), 0.95)),
        "sign_test_samples": int(reliable.sum()),
        "sign_mismatches_away_from_surface": int(sign_bad.sum()),
        "concavity_empty_samples": int(concavity.sum()),
        "concavity_grid_positive_samples": int(np.count_nonzero(values[concavity] > 0)),
        "passed": bool(not sign_bad.any() and np.all(values[concavity] > 0)),
    }


def configure_step(model, piece, receivers):
    def descendants(name):
        ids = {model.body(name).id}
        for i in range(model.nbody):
            if int(model.body_parentid[i]) in ids:
                ids.add(i)
        return ids

    active = descendants(piece)
    fixtures = set().union(*(descendants(x) for x in receivers))
    model.geom_contype[:] = 0
    model.geom_conaffinity[:] = 0
    for i in range(model.ngeom):
        if model.geom_type[i] != mujoco.mjtGeom.mjGEOM_SDF:
            continue
        b = int(model.geom_bodyid[i])
        if b in active:
            model.geom_contype[i] = 1
            model.geom_conaffinity[i] = 2
        elif b in fixtures:
            model.geom_contype[i] = 2
            model.geom_conaffinity[i] = 1
    model.body_contype[:] = 0
    model.body_conaffinity[:] = 0
    for i in range(model.ngeom):
        b = int(model.geom_bodyid[i])
        model.body_contype[b] |= model.geom_contype[i]
        model.body_conaffinity[b] |= model.geom_conaffinity[i]
    return active


def fixture_replay(model, qpos, mode, native, active, force_body, steps):
    previous = native.so101_surface_mode(mode)
    stats(native)
    try:
        d = mujoco.MjData(model)
        d.qpos[:] = qpos
        fq = np.ones(model.nq, dtype=bool)
        fv = np.ones(model.nv, dtype=bool)
        for j in range(model.njnt):
            if int(model.jnt_bodyid[j]) not in active:
                continue
            qa = int(model.jnt_qposadr[j])
            va = int(model.jnt_dofadr[j])
            free = model.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE
            fq[qa : qa + (7 if free else 1)] = False
            fv[va : va + (6 if free else 1)] = False
        result = []
        for _ in range(steps):
            d.qpos[fq] = qpos[fq]
            d.qvel[fv] = 0
            mujoco.mj_forward(model, d)
            d.qfrc_applied[:] = 0
            mujoco.mj_applyFT(
                model,
                d,
                np.array([-0.05, 0, -0.5]),
                np.zeros(3),
                d.xipos[force_body],
                force_body,
                d.qfrc_applied,
            )
            mujoco.mj_step(model, d)
            d.qpos[fq] = qpos[fq]
            d.qvel[fv] = 0
            mujoco.mj_forward(model, d)
            result.append((d.qpos.copy(), d.qvel.copy(), contacts(d)))
        return result, stats(native)
    finally:
        native.so101_surface_mode(previous)


def run(args):
    from assembly_policy.so101.wowrobo_contact_backend import (
        load_backend,
        prepare_scene,
        surface_library,
    )

    out = args.output
    out.mkdir(parents=True, exist_ok=True)
    manifest = json.loads(args.grids.read_text())
    report = {
        "source_manifest_sha256": hashlib.sha256(args.grids.read_bytes()).hexdigest(),
        "finite_panel_only": True,
        "grids": {},
        "parity": [],
    }
    for name, record in manifest["meshes"].items():
        path = Path(record["grid_file"])
        if not path.exists():
            raise FileNotFoundError(path)
        mesh = trimesh.load_mesh(record["source_file"], process=False)
        report["grids"][name] = mesh_grid_sanity(mesh, path)
        print("grid", name, json.dumps(report["grids"][name]), flush=True)
        (out / "validation.json").write_text(json.dumps(report, indent=2) + "\n")
    backend_info = load_backend(args.grids)
    native = surface_library()
    report["backend"] = backend_info
    scene = out / "validation-scene.xml"
    prepare_scene(args.source / "scene.xml", args.grids, scene)
    model = mujoco.MjModel.from_xml_path(str(scene))
    # Independent source-coordinate -> compiled visual -> world -> native SDF.
    query = native.wowrobo_sdf_distance
    ptr = ctypes.POINTER(ctypes.c_double)
    query.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ptr]
    query.restype = ctypes.c_double
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    report["native_frame_checks"] = {}
    from scipy.spatial.transform import Rotation

    for name, record in manifest["meshes"].items():
        visual = model.geom(name).id
        contact = model.geom(name + "__contact").id
        mid = int(model.geom_dataid[visual])
        mesh = trimesh.load_mesh(record["source_file"], process=False)
        points = np.vstack(
            (
                mesh.bounds.mean(0),
                mesh.bounds,
                mesh.vertices[np.linspace(0, len(mesh.vertices) - 1, 32, dtype=int)],
            )
        )
        rotation = Rotation.from_quat(
            model.mesh_quat[mid], scalar_first=True
        ).as_matrix()
        principal = (points - model.mesh_pos[mid]) @ rotation
        world = (
            principal @ data.geom_xmat[visual].reshape(3, 3).T + data.geom_xpos[visual]
        )
        expected, _ = grid_values(record["grid_file"], points)
        actual = np.array(
            [
                query(
                    model._address,
                    data._address,
                    contact,
                    np.ascontiguousarray(p).ctypes.data_as(ptr),
                )
                for p in world
            ]
        )
        error = float(np.max(np.abs(expected - actual)))
        report["native_frame_checks"][name] = {
            "samples": len(points),
            "max_abs_raw_grid_native_distance_difference_m": error,
            "passed": bool(error < 1e-9),
        }
        print("frame", name, error, flush=True)
    (out / "validation.json").write_text(json.dumps(report, indent=2) + "\n")
    report["solver"] = {
        "jacobian": int(model.opt.jacobian),
        "sparse_expected": int(mujoco.mjtJacobian.mjJAC_SPARSE),
        "timestep": float(model.opt.timestep),
    }
    assert model.opt.jacobian == mujoco.mjtJacobian.mjJAC_SPARSE
    specs = {
        "cover": (
            "base_side_cover",
            ["base_servo"],
            ["04_cover_entry", "05_cover_near", "06_cover_seated"],
        ),
        "connector": (
            "shoulder_connector",
            ["base_servo", "base_side_cover"],
            ["09_connector_entry", "10_connector_near", "11_connector_seated"],
        ),
        "pin": (
            "assembly_pin",
            ["base_servo", "base_side_cover", "shoulder_connector"],
            ["14_pin_half", "15_pin_near", "16_pin_seated"],
        ),
        "upper": (
            "upper_arm_module",
            ["base_servo", "base_side_cover", "shoulder_connector", "assembly_pin"],
            ["21_upper_far_side", "23_upper_near", "24_upper_seated"],
        ),
        "final": (
            "forearm_module",
            [
                "base_servo",
                "base_side_cover",
                "shoulder_connector",
                "assembly_pin",
                "upper_arm_module",
            ],
            ["29_final_far_side", "31_final_near", "32_final_seated"],
        ),
    }
    rng = np.random.default_rng(7503)
    for name, (piece, receivers, stages) in specs.items():
        active = configure_step(model, piece, receivers)
        for stage in stages:
            d = mujoco.MjData(model)
            mujoco.mj_setState(
                model,
                d,
                np.load(args.source / "stages" / f"{stage}_train.npz")["states"][0],
                mujoco.mjtState.mjSTATE_FULLPHYSICS,
            )
            for perturb in range(args.perturbations + 1):
                q = d.qpos.copy()
                if perturb:
                    address = int(model.joint(piece + "_free").qposadr[0])
                    q[address : address + 3] += rng.uniform(-0.0003, 0.0003, 3)
                a, sa = fixture_replay(
                    model, q, 0, native, active, model.body(piece).id, args.steps
                )
                b, sb = fixture_replay(
                    model, q, 1, native, active, model.body(piece).id, args.steps
                )
                failures = parity(a, b)
                row = {
                    "step": name,
                    "stage": stage,
                    "perturbation": perturb,
                    "dynamic_steps": args.steps,
                    "exact_equal": not failures,
                    "failures": failures,
                    "mode0_stats": sa.tolist(),
                    "mode1_stats": sb.tolist(),
                    "mode1_peak_contacts": max(len(x[2]) for x in b),
                    "mode1_finite": all(
                        np.isfinite(x).all() for frame in b for x in frame
                    ),
                    "activated_new_route": bool(sb[0] > 0 and sb[3] > 0),
                    "contact_geom_names": sorted(
                        {
                            model.geom(int(g)).name
                            for frame in b
                            for c in frame[2]
                            for g in c[-2:]
                        }
                    ),
                }
                report["parity"].append(row)
                print(
                    "parity",
                    name,
                    stage,
                    perturb,
                    "equal",
                    not failures,
                    "contacts",
                    row["mode1_peak_contacts"],
                    "stats",
                    sb.tolist(),
                    flush=True,
                )
                (out / "validation.json").write_text(
                    json.dumps(report, indent=2) + "\n"
                )
    report["exact_acceleration_parity_passed"] = all(
        x["exact_equal"] and x["mode1_finite"] for x in report["parity"]
    )
    report["callback_activated_for_each_step"] = {
        s: any(x["activated_new_route"] for x in report["parity"] if x["step"] == s)
        for s in specs
    }
    report["source_grid_sanity_passed"] = all(
        x["passed"] for x in report["grids"].values()
    )
    report["passed"] = bool(
        report["exact_acceleration_parity_passed"]
        and all(report["callback_activated_for_each_step"].values())
        and report["source_grid_sanity_passed"]
        and all(x["passed"] for x in report["native_frame_checks"].values())
    )
    (out / "validation.json").write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps(
            {k: v for k, v in report.items() if k not in ["parity", "grids"]}, indent=2
        )
    )
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--source",
        type=Path,
        default=ROOT / "outputs/linear-insertion-20261107/source-snapshot",
    )
    p.add_argument(
        "--grids",
        type=Path,
        default=ROOT / "outputs/linear-insertion-updated-20261107/grids/manifest.json",
    )
    p.add_argument(
        "--output",
        type=Path,
        default=ROOT / "outputs/linear-insertion-updated-20261107/validation",
    )
    p.add_argument("--steps", type=int, default=6)
    p.add_argument("--perturbations", type=int, default=1)
    raise SystemExit(run(p.parse_args()))
