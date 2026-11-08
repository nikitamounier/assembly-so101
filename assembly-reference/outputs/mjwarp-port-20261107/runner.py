"""Isolated H100 query/contact/assembly experiments. Never attaches to training."""

import argparse
import ctypes
import hashlib
import importlib.util
import json
import os
import subprocess
import sys
import threading
import time
import traceback
from pathlib import Path

P = Path("/large_storage/goodarzilab/nikitamounier/assembly/mjwarp-port-20261107")
R = Path(
    "/large_storage/goodarzilab/nikitamounier/assembly/wowrobo-diagnostic-20261107T205745Z-minimum24-accepted-v4"
)
parser = argparse.ArgumentParser()
parser.add_argument(
    "--mode", choices=["query", "assembly", "benchmark"], default="query"
)
parser.add_argument("--worlds", type=int, default=1)
parser.add_argument("--steps", type=int, default=160)
parser.add_argument("--case", default="00_loose")
parser.add_argument("--variation", action="store_true")
parser.add_argument("--bank-version", type=int, default=5)
parser.add_argument("--arm-action", type=float, default=0.0)
parser.add_argument("--jacobian", choices=["native", "dense"], default="native")
args = parser.parse_args()
assert abs(args.arm_action) <= 1.0
OUT = P / (
    args.mode
    + "-"
    + str(args.worlds)
    + "-"
    + args.case
    + ("-variation" if args.variation else "")
    + "-steps"
    + str(args.steps)
    + "-bank"
    + str(args.bank_version)
    + ("-action" + str(args.arm_action) if args.arm_action else "")
    + ("-dense" if args.jacobian == "dense" else "")
)
OUT.mkdir(exist_ok=True)
result = {
    "host": os.uname().nodename,
    "bank_version": args.bank_version,
    "arm_action": args.arm_action,
    "jacobian": args.jacobian,
    "backend_sha256": hashlib.sha256((P / "wowrobo_warp.py").read_bytes()).hexdigest(),
    "runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    "mode": args.mode,
    "worlds": args.worlds,
    "case": args.case,
    "steps": args.steps,
    "time": time.time(),
}


def save():
    (OUT / "result.json").write_text(json.dumps(result, indent=2, default=str) + "\n")


def report(**kw):
    result.update(kw)
    save()
    print(json.dumps(kw, default=str), flush=True)


def watch():
    while True:
        time.sleep(1)
        try:
            app = subprocess.check_output(
                ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"],
                text=True,
            )
            other = [
                s.strip()
                for s in app.splitlines()
                if s.strip() and s.strip() != str(os.getpid())
            ]
            if other:
                print("YIELD main/other GPU workload " + str(other), flush=True)
                os._exit(75)
            for f in R.glob("run-v*/controller.json"):
                c = json.loads(f.read_text())
                if time.time() - f.stat().st_mtime < 120 and str(
                    c.get("gpu_job")
                ) == os.environ.get("SLURM_JOB_ID"):
                    print("YIELD main controller selected this allocation", flush=True)
                    os._exit(75)
        except (OSError, ValueError):
            pass


try:
    app = subprocess.check_output(
        ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"], text=True
    )
    if app.strip():
        raise RuntimeError("Retained GPU is busy; refusing probe")
    threading.Thread(target=watch, daemon=True).start()
    os.environ["WOWROBO_CANONICAL_ROOT"] = str(P / "canonical")
    os.environ["WOWROBO_WARP_GRID_SOURCE"] = str(
        P / "assembly-so101/plugin/assemblenist_sdf.cc"
    )
    os.environ["WOWROBO_WARP_FRICTION_SOURCE"] = str(
        P / "canonical/assembly_policy/so101/wowrobo_friction.py"
    )
    sys.path.insert(0, str(R / "assembly-so101"))
    import mujoco
    import mujoco_warp as mjw
    import numpy as np
    import warp as wp

    sys.path.append(
        "/large_storage/goodarzilab/nikitamounier/assembly/cube-trial-20261107-v6/.venv/lib/python3.12/site-packages"
    )
    wp.config.kernel_cache_dir = "/large_storage/goodarzilab/nikitamounier/assembly/mjwarp-h100-20261107/warp-cache"
    wp.init()
    wp.set_device("cuda:0")
    if args.steps == 4:
        wp.config.verify_cuda = True
    import importlib.metadata as md

    report(
        versions={n: md.version(n) for n in ["mujoco", "mujoco-warp", "warp-lang"]},
        gpu=subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"],
            text=True,
        ),
    )
    bank = (
        R
        / f"robotics-assembly-3.0/assets_local/wowrobo/training-accepted-v{args.bank_version}"
    )
    meta = json.loads((bank / "manifest.json").read_text())
    from assembly_policy.wowrobo.runtime import load_runtime

    m, profile, signature = load_runtime(meta["profile_path"])
    assert signature == meta["model_signature"], (
        "Native model signature differs from bank"
    )
    report(
        native_signature=signature,
        expected_signature=meta["model_signature"],
        native_ngeom=m.ngeom,
        native_npair=m.npair,
    )
    # Wrapper exports exist only in the isolated reference library.
    backend = sys.modules["_wowrobo_training_native_backend"]
    lib = backend.surface_library()
    refgrid = lib.warp_reference_grid
    refgrid.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_int,
        np.ctypeslib.ndpointer(np.float64, flags="C_CONTIGUOUS"),
        np.ctypeslib.ndpointer(np.float64, flags="C_CONTIGUOUS"),
    ]
    refgrid.restype = ctypes.c_int
    refsupp = lib.warp_reference_supplement
    refsupp.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_double,
        np.ctypeslib.ndpointer(np.float64, flags="C_CONTIGUOUS"),
    ]
    refsupp.restype = ctypes.c_int
    spec = importlib.util.spec_from_file_location(
        "wowrobo_warp_port", P / "wowrobo_warp.py"
    )
    port = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = port
    spec.loader.exec_module(port)
    manifest = Path(profile["grid_manifest"])
    gridmeta = json.loads(manifest.read_text())
    griddir = Path(
        gridmeta.get("grid_dir", gridmeta.get("grid_directory", manifest.parent))
    )
    if not griddir.is_absolute():
        griddir = manifest.parent / griddir
    # Choose actual accepted task start; full reset payload preserves control/warmstart/friction.
    match = []
    for task in meta["task_names"]:
        z = np.load(bank / (task + "_train.npz"))
        ids = np.flatnonzero(z["stage"] == args.case)
        ids = [
            i for i in ids if bool(np.any(z["states"][i, -3:] != 1)) == args.variation
        ]
        if len(ids):
            match.append((task, z["states"][ids[0]].copy()))
    if not match:
        raise ValueError("Unknown stage " + args.case)
    task, state = match[0]
    d = mujoco.MjData(m)

    # Frozen trainer encodes FULLPHYSICS+eq+ctrl+warmstart+episode friction.
    def reset_native():
        d = mujoco.MjData(m)
        full = mujoco.mj_stateSize(m, mujoco.mjtState.mjSTATE_FULLPHYSICS)
        eq_end = full + m.neq
        assert (
            profile.get("training_reset_payload") == "fullphysics-eq-ctrl-warmstart-v1"
        )
        assert profile.get("training_episode_friction") == "native-userdata-v1"
        assert len(state) == eq_end + m.nu + m.nv + 3
        value = state[:full].copy()
        value[0] = 0
        mujoco.mj_setState(m, d, value, mujoco.mjtState.mjSTATE_FULLPHYSICS)
        d.eq_active[:] = state[full:eq_end]
        d.ctrl[:] = state[eq_end : eq_end + m.nu]
        d.userdata[:3] = state[-3:]
        mujoco.mj_forward(m, d)
        d.qacc_warmstart[:] = state[eq_end + m.nu : eq_end + m.nu + m.nv]
        return d

    d = reset_native()
    report(
        task=task,
        stage=args.case,
        native_start_ncon=int(d.ncon),
        native_friction=state[-3:].tolist(),
    )
    gpu = port.WowRoboWarp(
        m,
        d,
        griddir,
        OUT,
        nworld=args.worlds,
        nconmax=512,
        njmax=2048,
        jacobian=args.jacobian,
    )
    gpu.install_context_capture()
    # FULLPHYSICS/friction fields read by the actual environment, not guessed slicing.
    if np.all(np.isfinite(d.userdata[:3])) and np.all(d.userdata[:3] > 0):
        wp.copy(
            gpu.draws,
            wp.array(
                np.tile(d.userdata[:3], (args.worlds, 1)).astype(np.float32),
                dtype=float,
            ),
        )
    report(
        table_audit=gpu.table_audit,
        solver_tolerance_native=float(m.opt.tolerance),
        solver_tolerance_gpu=float(gpu.model.opt.tolerance.numpy()[0]),
    )
    if args.mode == "query":
        rng = np.random.default_rng(719)
        points = []
        ids = []
        geoms = []
        for g in np.flatnonzero(m.geom_plugin >= 0):
            # Interior, original boundaries, just-outside and exact grid corners.
            idx = int(gpu.model.geom_plugin_index.numpy()[g])
            a = gpu.model.plugin_attr.numpy()[idx]
            shape, head, _ = gpu.grids[int(a[0])]
            lo = head[:3]
            hi = lo + (shape - 1) * head[3]
            values = np.vstack(
                [
                    rng.uniform(lo, hi, size=(64, 3)),
                    lo,
                    hi,
                    lo - np.ones(3) * 1.0e-7,
                    hi + np.ones(3) * 1.0e-7,
                    (lo + hi) / 2,
                ]
            )
            points.extend(values)
            ids.extend([idx] * len(values))
            geoms.extend([g] * len(values))
        points = np.asarray(points, np.float64)
        ids = np.asarray(ids, np.int32)
        answer = wp.empty(len(points), dtype=wp.vec4d)
        t = time.perf_counter()
        wp.launch(
            port.query_grid,
            dim=len(points),
            inputs=[
                wp.array(points, dtype=wp.vec3d),
                gpu.model.plugin_attr,
                wp.array(ids, dtype=int),
                answer,
            ],
        )
        wp.synchronize()
        elapsed = time.perf_counter() - t
        native = np.zeros((len(points), 4), np.float64)
        for i, (g, p) in enumerate(zip(geoms, points)):
            assert refgrid(
                m._address, d._address, int(g), np.ascontiguousarray(p), native[i]
            )
        actual = answer.numpy()
        error = np.abs(actual - native)
        np.savez(
            OUT / "queries.npz", points=points, geom=geoms, native=native, gpu=actual
        )
        report(
            query={
                "count": len(points),
                "max_distance_error_m": float(error[:, 0].max()),
                "max_gradient_error": float(error[:, 1:].max()),
                "finite": bool(np.isfinite(actual).all()),
                "first_compile_and_query_s": elapsed,
            }
        )
    else:
        t = time.perf_counter()
        mjw.forward(gpu.model, gpu.data)
        wp.synchronize()
        report(first_forward_compile_s=time.perf_counter() - t)
        gpu.check()
        # One uncaptured, synchronized phase probe after compilation. Restore all
        # callbacks before timing the assembly trajectory below.
        from mujoco_warp._src import collision_driver as driver
        from mujoco_warp._src import forward as forward_module

        phases = {}

        def phase(name, fn):
            def wrapped(*a, **kw):
                wp.synchronize()
                begin = time.perf_counter()
                value = fn(*a, **kw)
                wp.synchronize()
                phases[name] = phases.get(name, 0.0) + time.perf_counter() - begin
                return value

            return wrapped

        original_sdf = driver.sdf_narrowphase
        original_contacts = gpu.model.callback.contactfilter
        original_solve = forward_module.solver.solve
        try:
            driver.sdf_narrowphase = phase("ordinary_sdf", original_sdf)
            gpu.model.callback.contactfilter = phase(
                "supplement_friction", original_contacts
            )
            forward_module.solver.solve = phase("solver", original_solve)
            begin = time.perf_counter()
            mjw.forward(gpu.model, gpu.data)
            wp.synchronize()
            phases["forward_total"] = time.perf_counter() - begin
        finally:
            driver.sdf_narrowphase = original_sdf
            gpu.model.callback.contactfilter = original_contacts
            forward_module.solver.solve = original_solve
        report(phase_seconds=phases)
        nc = int(gpu.data.nacon.numpy()[0])
        cg = gpu.data.contact.geom.numpy()[:nc]
        cf = gpu.data.contact.friction.numpy()[:nc]
        cw = gpu.data.contact.worldid.numpy()[:nc]
        roles = gpu.roles.numpy()
        draws = gpu.draws.numpy()
        base = gpu.fr.numpy()
        err = []
        for i, (a, b) in enumerate(cg):
            if a < 0 or b < 0:
                continue
            expected = base[a, b].copy()
            pair = {int(roles[a]), int(roles[b])}
            k = (
                0
                if pair == {1, 2}
                else 1
                if pair == {2, 3}
                else 2
                if roles[a] == 2 and roles[b] == 2
                else -1
            )
            if k >= 0:
                expected[:2] *= draws[cw[i], k]
            err.append(np.max(np.abs(expected - cf[i])))
        report(
            initial_constraint_force_relative_error=float(
                np.linalg.norm(gpu.data.qfrc_constraint.numpy()[0] - d.qfrc_constraint)
                / max(np.linalg.norm(d.qfrc_constraint), 1.0e-12)
            )
        )
        report(
            friction_max_error=float(max(err, default=0.0)),
            sensor_min=float(gpu.data.sensordata.numpy()[0, 0]),
            contact_min=float(gpu.data.contact.dist.numpy()[:nc].min(initial=0)),
        )
        np.savez(
            OUT / "initial-contacts.npz",
            gpu_geom=cg,
            gpu_dist=gpu.data.contact.dist.numpy()[:nc],
            gpu_pos=gpu.data.contact.pos.numpy()[:nc],
            gpu_frame=gpu.data.contact.frame.numpy()[:nc],
            gpu_friction=cf,
            native_geom=d.contact.geom.copy(),
            native_dist=d.contact.dist.copy(),
            native_pos=d.contact.pos.copy(),
            native_frame=d.contact.frame.copy(),
            native_friction=d.contact.friction.copy(),
        )
        # Isolate supplement selection using matched float32 transform inputs.
        pos = gpu.data.geom_xpos.numpy()[0]
        rot = gpu.data.geom_xmat.numpy()[0]
        for g in range(m.ngeom):
            d.geom_xpos[g] = pos[g]
            d.geom_xmat[g] = rot[g].reshape(-1)
        ctx = gpu.context
        count = int(gpu.data.ncollision.numpy()[0])
        pairs = ctx.collision_pair.numpy()[:count]
        winners = gpu.winners.numpy()
        distances = gpu.distances.numpy()
        eligible = gpu.eligible.numpy()
        margins = gpu.margin.numpy()
        worlds = ctx.collision_worldid.numpy()
        checks = []
        collision_ids = gpu.data.contact.geomcollisionid.numpy()[:nc]
        contact_positions = gpu.data.contact.pos.numpy()[:nc]
        contact_frames = gpu.data.contact.frame.numpy()[:nc]
        for cid, (a, b) in enumerate(pairs):
            if worlds[cid] != 0 or not eligible[a, b]:
                continue
            reference = np.zeros((50, 7), np.float64)
            n = refsupp(
                m._address, d._address, int(a), int(b), float(margins[a, b]), reference
            )
            selected = np.asarray(
                [
                    distances[cid, k]
                    for k in range(12)
                    if distances[cid, k] < margins[a, b]
                ]
            )
            delta = (
                float(np.max(np.abs(selected - reference[:n, 0])))
                if len(selected) == n and n
                else (0.0 if n == 0 and len(selected) == 0 else None)
            )
            selected_bins = [k for k in range(12) if distances[cid, k] < margins[a, b]]
            p_error = []
            n_error = []
            for j, k in enumerate(selected_bins):
                found = np.flatnonzero(
                    (cw == 0)
                    & (cg[:, 0] == a)
                    & (cg[:, 1] == b)
                    & (collision_ids == 40 + k)
                )
                if len(found) == 1 and j < n:
                    p_error.append(
                        float(
                            np.linalg.norm(
                                contact_positions[found[0]] - reference[j, 1:4]
                            )
                        )
                    )
                    n_error.append(
                        float(
                            np.linalg.norm(
                                contact_frames[found[0], 0] - reference[j, 4:7]
                            )
                        )
                    )
            checks.append(
                {
                    "a": int(a),
                    "b": int(b),
                    "native": n,
                    "gpu": len(selected),
                    "max_dist_error": delta,
                    "matched_emitted": len(p_error),
                    "max_position_error": max(p_error, default=0.0),
                    "max_normal_error": max(n_error, default=0.0),
                }
            )
        report(
            supplement_panel=checks,
            supplement_count=gpu.supplement_count.numpy().tolist(),
            gpu_ncon=int(gpu.data.nacon.numpy()[0]),
        )
        if any(
            item["native"] != item["gpu"]
            or item["matched_emitted"] != item["native"]
            or item["max_dist_error"] is None
            or item["max_dist_error"] > 1.0e-9
            or item["max_position_error"] > 1.0e-6
            or item["max_normal_error"] > 1.0e-5
            for item in checks
        ):
            raise RuntimeError(
                "Supplement contact parity failed; trajectory benchmark refused"
            )
        # Restore original native state after transform-only contact check.
        d = reset_native()
        # Reset GPU state from native to avoid timing a pre-warmed trajectory.
        fields = [
            "qpos",
            "qvel",
            "qacc_warmstart",
            "ctrl",
            "act",
            "qfrc_applied",
            "xfrc_applied",
            "mocap_pos",
            "mocap_quat",
            "eq_active",
        ]
        for key in fields:
            if hasattr(gpu.data, key) and np.asarray(getattr(d, key)).size:
                dst = getattr(gpu.data, key)
                arr = dst.numpy()
                arr[:] = np.asarray(getattr(d, key))
                wp.copy(dst, wp.array(arr, dtype=dst.dtype))
        gpu.data.time.zero_()
        # Warm kernels with1step, then reset time/state again before timed graph replay.
        mjw.step(gpu.model, gpu.data)
        wp.synchronize()
        gpu.check()
        for key in fields:
            if hasattr(gpu.data, key) and np.asarray(getattr(d, key)).size:
                dst = getattr(gpu.data, key)
                arr = dst.numpy()
                arr[:] = np.asarray(getattr(d, key))
                wp.copy(dst, wp.array(arr, dtype=dst.dtype))
        gpu.data.time.zero_()
        peak = wp.zeros(args.worlds, dtype=float)

        def step_and_monitor():
            mjw.step(gpu.model, gpu.data)
            wp.launch(
                port.monitor_contacts,
                dim=gpu.data.naconmax,
                inputs=[
                    gpu.data.contact.geom,
                    gpu.data.contact.worldid,
                    gpu.data.nacon,
                    gpu.data.naconmax,
                    gpu.data.contact.dist,
                    peak,
                ],
            )

        wp.config.verify_cuda = False
        # Compile the monitor before capture; reset its accumulated value.
        wp.launch(
            port.monitor_contacts,
            dim=gpu.data.naconmax,
            inputs=[
                gpu.data.contact.geom,
                gpu.data.contact.worldid,
                gpu.data.nacon,
                gpu.data.naconmax,
                gpu.data.contact.dist,
                peak,
            ],
        )
        wp.synchronize()
        peak.zero_()
        with wp.ScopedCapture() as capture:
            step_and_monitor()
        wp.synchronize()
        initial_ctrl = d.ctrl.copy()
        gpu_ctrl = initial_ctrl.copy()
        t = time.perf_counter()
        for i in range(args.steps):
            if args.arm_action and i % 160 == 0:
                gpu_ctrl[1] = np.clip(
                    gpu_ctrl[1] + 0.035 * args.arm_action,
                    m.actuator_ctrlrange[1, 0],
                    m.actuator_ctrlrange[1, 1],
                )
                wp.copy(
                    gpu.data.ctrl,
                    wp.array(
                        np.tile(gpu_ctrl, (args.worlds, 1)).astype(np.float32),
                        dtype=float,
                    ),
                )
            wp.capture_launch(capture.graph)
        wp.synchronize()
        elapsed = time.perf_counter() - t
        gpu.check()
        q = gpu.data.qpos.numpy()
        v = gpu.data.qvel.numpy()
        t = time.perf_counter()
        native_peak = 0.0
        for i in range(args.steps):
            if args.arm_action and i % 160 == 0:
                d.ctrl[1] = np.clip(
                    d.ctrl[1] + 0.035 * args.arm_action,
                    m.actuator_ctrlrange[1, 0],
                    m.actuator_ctrlrange[1, 1],
                )
            mujoco.mj_step(m, d)
            native_peak = min(native_peak, float(d.contact.dist.min(initial=0)))
        native_s = time.perf_counter() - t
        np.savez(
            OUT / "trajectory-final.npz",
            gpu_qpos=q,
            gpu_qvel=v,
            native_qpos=d.qpos,
            native_qvel=d.qvel,
        )
        translation = []
        rotation = []
        passive = []
        actuated = {int(x) for x in m.actuator_trnid[:, 0]}
        for j in range(m.njnt):
            adr = int(m.jnt_qposadr[j])
            if m.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE:
                translation.append(
                    np.linalg.norm(
                        q[:, adr : adr + 3] - d.qpos[adr : adr + 3], axis=1
                    ).max()
                )
                native_quat = d.qpos[adr + 3 : adr + 7] / np.linalg.norm(
                    d.qpos[adr + 3 : adr + 7]
                )
                quat = q[:, adr + 3 : adr + 7].astype(np.float64)
                quat = quat / np.linalg.norm(quat, axis=1)[:, None]
                rotation.append(
                    np.degrees(
                        2 * np.arccos(np.clip(np.abs(quat @ native_quat), 0, 1))
                    ).max()
                )
            elif j not in actuated:
                passive.append(float(np.abs(q[:, adr] - d.qpos[adr]).max()))
        report(
            free_translation_error_m=float(max(translation, default=0.0)),
            free_rotation_error_degrees=float(max(rotation, default=0.0)),
            passive_joint_error_rad=max(passive, default=0.0),
        )

        report(
            assembly={
                "gpu_seconds": elapsed,
                "native_one_world_seconds": native_s,
                "world_substeps_per_s": args.worlds * args.steps / elapsed,
                "native_substeps_per_s": args.steps / native_s,
                "max_qpos_error": float(np.max(np.abs(q - d.qpos))),
                "max_qvel_error": float(np.max(np.abs(v - d.qvel))),
                "finite": bool(np.isfinite(q).all() and np.isfinite(v).all()),
                "gpu_peak_penetration_m": (-peak.numpy()).tolist(),
                "native_peak_penetration_m": -native_peak,
                "native_warnings": [int(w.number) for w in d.warning],
                "gpu_ncon": int(gpu.data.nacon.numpy()[0]),
                "native_ncon": int(d.ncon),
                "gpu_min_distance": float(
                    gpu.data.contact.dist.numpy()[: int(gpu.data.nacon.numpy()[0])].min(
                        initial=0
                    )
                ),
                "native_min_distance": float(d.contact.dist.min(initial=0)),
            }
        )
    report(complete=True)
except BaseException as e:
    report(error=repr(e), traceback=traceback.format_exc())
    raise
