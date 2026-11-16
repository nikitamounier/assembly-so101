"""Pinned direct-runtime GPU profiling/scaling without one CPU MjData per world."""

import argparse
import collections
import cProfile
import csv
import hashlib
import importlib.util
import json
import os
import resource
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--release", type=Path, required=True)
parser.add_argument("--bank", type=Path, required=True)
parser.add_argument("--out", type=Path, required=True)
parser.add_argument("--source", type=Path, required=True)
parser.add_argument("--worlds", type=int, default=16)
parser.add_argument("--steps", type=int, default=80)
parser.add_argument("--repeats", type=int, default=3)
parser.add_argument("--profile", action="store_true")
parser.add_argument("--nsys", action="store_true")
parser.add_argument("--tolerance", type=float)
parser.add_argument("--ls-tolerance", type=float)
parser.add_argument("--contact-cap", type=int, default=512)
parser.add_argument("--constraint-cap", type=int, default=2048)
parser.add_argument("--noslip-block", type=int)
parser.add_argument("--audit-noslip", action="store_true")
parser.add_argument("--audit-contacts", action="store_true")
parser.add_argument("--all-stages", action="store_true")
parser.add_argument("--diagnostics", action="store_true")
a = parser.parse_args()
a.out.mkdir(parents=True, exist_ok=True)
os.environ["WOWROBO_CANONICAL_ROOT"] = str(a.release.parent / "robotics-assembly-3.0")
sys.path.insert(0, str(a.release / "assembly-so101"))
import mujoco
import warp as wp
from assembly_policy.wowrobo.runtime import load_runtime

if a.noslip_block is not None:
    from mujoco_warp._src import solver

    original_noslip = solver.noslip
    original_launch = wp.launch
    audited = []

    def tuned_noslip(*args, **kwargs):
        def launch(kernel, *values, **options):
            name = getattr(kernel, "key", "") + getattr(kernel.func, "__qualname__", "")
            if "_noslip_solve" in name:
                options["block_dim"] = a.noslip_block
                if a.audit_noslip and not audited:
                    outputs = options["outputs"]
                    before = [x.numpy().copy() for x in outputs]
                    stock = dict(options)
                    stock.pop("block_dim")
                    original_launch(kernel, *values, **stock)
                    expected = [x.numpy().copy() for x in outputs]
                    for x, y in zip(outputs, before):
                        x.assign(y)
                    result = original_launch(kernel, *values, **options)
                    actual = [x.numpy().copy() for x in outputs]
                    proof = [
                        {
                            "shape": list(x.shape),
                            "equal": bool(np.array_equal(x, y)),
                            "max_error": float(abs(x - y).max(initial=0)),
                        }
                        for x, y in zip(expected, actual)
                    ]
                    (a.out / "noslip-launch-parity.json").write_text(
                        json.dumps(proof, indent=2) + "\n"
                    )
                    assert all(x["equal"] for x in proof)
                    audited.append(True)
                    return result
            return original_launch(kernel, *values, **options)

        wp.launch = launch
        try:
            return original_noslip(*args, **kwargs)
        finally:
            wp.launch = original_launch

    solver.noslip = tuned_noslip

wp.init()
wp.set_device("cuda:0")
meta = json.loads((a.bank / "manifest.json").read_text())
m, profile, signature = load_runtime(meta["profile_path"])
assert signature == meta["model_signature"]
stages = (
    "05_cover_near",
    "10_connector_near",
    "15_pin_near",
    "21_upper_far_side",
    "29_final_far_side",
)
rows = []
all_rows = {}
for i, task in enumerate(meta["task_names"]):
    with np.load(a.bank / (task + "_train.npz")) as bank:
        selected = np.flatnonzero(bank["stage"] == stages[i])
        rows.append(bank["states"][selected].copy())
        if a.all_stages:
            for stage in np.unique(bank["stage"]):
                if str(stage) not in all_rows:
                    ids = np.flatnonzero(bank["stage"] == stage)[:2]
                    all_rows[str(stage)] = bank["states"][ids].copy()
packed = np.array([rows[i % 5][(i // 5) % len(rows[i % 5])] for i in range(a.worlds)])
if a.all_stages:
    pool = np.concatenate([all_rows[k] for k in sorted(all_rows)])
    packed = pool[np.arange(a.worlds) % len(pool)].copy()
full = mujoco.mj_stateSize(m, mujoco.mjtState.mjSTATE_FULLPHYSICS)
eqend = full + m.neq
warmend = eqend + m.nu + m.nv
assert packed.shape[1] == warmend + m.nuserdata
initial = packed[:, :full].copy()
initial[:, 0] = 0
warm = packed[:, eqend + m.nu : warmend].copy()
ctrl = packed[:, eqend : eqend + m.nu].copy()
ctrl[:, 1] += 0.00035
control = np.repeat(
    np.concatenate([ctrl, packed[:, full:eqend], packed[:, warmend:]], axis=1)[
        :, None, :
    ],
    a.steps,
    axis=1,
)
manifest = Path(profile["grid_manifest"])
gridmeta = json.loads(manifest.read_text())
grid_dir = Path(
    gridmeta.get("grid_dir", gridmeta.get("grid_directory", manifest.parent))
)
if not grid_dir.is_absolute():
    grid_dir = manifest.parent / grid_dir
for name in ("wowrobo_warp", "wowrobo_warp_runtime"):
    spec = importlib.util.spec_from_file_location(
        "assembly_policy.so101." + name, a.source / (name + ".py")
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
runtime = sys.modules["assembly_policy.so101.wowrobo_warp_runtime"]
out = {
    "status": "INITIALIZING",
    "worlds": a.worlds,
    "steps": a.steps,
    "repeats": a.repeats,
    "native_signature": signature,
    "source": {
        n: hashlib.sha256((a.source / n).read_bytes()).hexdigest()
        for n in ("wowrobo_warp.py", "wowrobo_warp_runtime.py")
    },
    "records": [],
    "all_stages": sorted(all_rows),
    "unique_initial_states": len(np.unique(packed, axis=0)),
    "phases": [],
    "kernels": [],
}


def save(**updates):
    out.update(updates)
    (a.out / "result.json").write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps(updates), flush=True)


engine = runtime.WowRoboWarpRollout(
    m,
    grid_dir,
    a.out / "scratch",
    worlds=a.worlds,
    nconmax=a.contact_cap,
    njmax=a.constraint_cap,
)
p = engine.physics
if a.diagnostics:

    @wp.kernel
    def diagnostic(
        overflow: wp.array[int],
        iterations: wp.array[int],
        constraints: wp.array[int],
        flags: wp.array2d[int],
        counts: wp.array2d[int],
        nefc: wp.array2d[int],
        index: int,
    ):
        world = wp.tid()
        flags[world, index] = overflow[world]
        counts[world, index] = iterations[world]
        nefc[world, index] = constraints[world]

    traces = [wp.zeros((a.worlds, a.steps), dtype=int) for _ in range(3)]
    original_step = engine._step

    def traced_step(inputs, states, sensors, index):
        original_step(inputs, states, sensors, index)
        wp.launch(
            diagnostic,
            dim=a.worlds,
            inputs=[p.data.overflow, p.data.solver_niter, p.data.nefc, *traces, index],
        )

    engine._step = traced_step
if a.audit_contacts:
    from contact_audit import install

    install(p, Path(__file__).parent / "baseline", a.out / "contact-parity.json")
if a.tolerance is not None:
    p.model.opt.tolerance.fill_(a.tolerance)
if a.ls_tolerance is not None:
    p.model.opt.ls_tolerance.fill_(a.ls_tolerance)
options = {
    name: float(getattr(m.opt, name))
    for name in (
        "timestep",
        "tolerance",
        "iterations",
        "ls_tolerance",
        "ls_iterations",
        "noslip_iterations",
        "noslip_tolerance",
        "solver",
        "cone",
        "integrator",
    )
}
save(
    status="WARMUP",
    options=options,
    gpu_tolerance=p.model.opt.tolerance.numpy().tolist(),
    gpu_ls_tolerance=p.model.opt.ls_tolerance.numpy().tolist(),
    identity=engine.identity,
    dimensions={
        "nq": m.nq,
        "nv": m.nv,
        "ngeom": m.ngeom,
        "neq": m.neq,
        "nu": m.nu,
        "free_dof": [
            int(m.jnt_dofadr[j])
            for j in range(m.njnt)
            if m.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE
        ],
        "free_qpos": [
            int(m.jnt_qposadr[j])
            for j in range(m.njnt)
            if m.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE
        ],
        "naconmax": p.data.naconmax,
        "njmax": p.data.njmax,
    },
)
try:
    start = time.perf_counter()
    reference, reference_sensors = engine.rollout(
        m, None, initial, control, nstep=a.steps, initial_warmstart=warm
    )
    save(
        status="MEASURING",
        warmup_s=time.perf_counter() - start,
        device_memory=subprocess.check_output(
            [
                "nvidia-smi",
                "--query-compute-apps=pid,used_memory",
                "--format=csv,noheader",
            ],
            text=True,
        ).strip(),
    )
    for repeat in range(a.repeats):
        profiler = cProfile.Profile()
        profiler.enable()
        start = time.perf_counter()
        states, sensors = engine.rollout(
            m, None, initial, control, nstep=a.steps, initial_warmstart=warm
        )
        elapsed = time.perf_counter() - start
        profiler.disable()
        if repeat == 0:
            profiler.dump_stats(str(a.out / "host.prof"))
        record = {
            "repeat": repeat,
            "wall_s": elapsed,
            "controls_per_s": a.worlds / elapsed,
            "max_repeat_state_error": float(abs(states - reference).max()),
            "min_contact_distance": float(sensors.min()),
            "ncon": int(p.data.nacon.numpy()[0]),
            "ncollision": int(p.data.ncollision.numpy()[0]),
            "nefc": p.data.nefc.numpy().tolist(),
            "solver_niter": p.data.solver_niter.numpy().tolist(),
            "overflow": p.data.overflow.numpy().tolist(),
        }
        out["records"].append(record)
        save(last_record=record)
    np.savez_compressed(
        a.out / "trajectory.npz",
        initial=initial,
        control=control,
        warmstart=warm,
        states=states,
        sensors=sensors,
    )
    # Isolate device graph duration from reset, upload, download and CPU packing.
    inputs, outputs, readings, graph = engine.buffers[a.steps]
    graph_ms = []
    for repeat in range(1):
        engine._restore(initial, warm)
        inputs.assign(control.astype(np.float32))
        before = wp.Event(enable_timing=True)
        after = wp.Event(enable_timing=True)
        wp.record_event(before)
        wp.capture_launch(graph)
        wp.record_event(after)
        graph_ms.append(wp.get_event_elapsed_time(before, after))
        if engine.flags.numpy().any():
            raise FloatingPointError('Overflow/convergence failure during isolated graph timing')
    save(
        graph_ms=graph_ms,
        host_peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
    )
    if a.profile:
        from mujoco_warp._src import collision_driver, forward, sensor, solver

        originals = []
        events = []

        def wrap(obj, name, label):
            original = getattr(obj, name)

            def call(*args, **kwargs):
                begin = wp.Event(enable_timing=True)
                end = wp.Event(enable_timing=True)
                wp.record_event(begin)
                value = original(*args, **kwargs)
                wp.record_event(end)
                events.append((label, begin, end))
                return value

            originals.append((obj, name, original))
            setattr(obj, name, call)

        for obj, name, label in (
            (forward, "fwd_position", "position"),
            (forward, "fwd_velocity", "velocity"),
            (forward, "fwd_actuation", "actuation"),
            (forward, "fwd_acceleration", "acceleration"),
            (solver, "solve", "solver"),
            (solver, "noslip", "noslip"),
            (collision_driver, "collision", "collision"),
            (collision_driver, "sdf_narrowphase", "ordinary_sdf"),
            (p.model.callback, "contactfilter", "supplement_friction"),
            (sensor, "sensor_pos", "sensor_pos"),
            (sensor, "sensor_vel", "sensor_vel"),
            (sensor, "sensor_acc", "sensor_acc"),
        ):
            wrap(obj, name, label)
        engine._restore(initial, warm)
        wp.timing_begin()
        try:
            for step in range(3):
                engine._step(inputs, outputs, readings, step)
            activities = wp.timing_end()
            wp.synchronize()
        finally:
            for obj, name, original in reversed(originals):
                setattr(obj, name, original)
        out["phases"] = [
            {"name": n, "ms": wp.get_event_elapsed_time(b, e)} for n, b, e in events
        ]
        agg = collections.defaultdict(lambda: [0, 0.0])
        for item in activities:
            agg[item.name][0] += 1
            agg[item.name][1] += item.elapsed
        out["kernels"] = [
            {"name": n, "count": v[0], "ms": v[1]}
            for n, v in sorted(agg.items(), key=lambda x: -x[1][1])
        ]
        with (a.out / "kernels.csv").open("w") as f:
            writer = csv.DictWriter(f, fieldnames=["name", "count", "ms"])
            writer.writeheader()
            writer.writerows(out["kernels"])
        save(profile_steps=3, profile_solver_niter=p.data.solver_niter.numpy().tolist())
    if a.nsys:
        engine._restore(initial, warm)
        wp.cuda_profiler_start()
        wp.capture_launch(graph)
        wp.synchronize()
        wp.cuda_profiler_stop()
    save(
        status="PASS",
        min_contact_distance=float(sensors.min()),
        finite=bool(np.isfinite(states).all() and np.isfinite(sensors).all()),
    )
except Exception as error:
    detail = {
        "error": repr(error),
        "last_overflow": p.data.overflow.numpy().tolist(),
        "latched": engine.flags.numpy().tolist(),
    }
    if a.diagnostics:
        from mujoco_warp._src.types import OverflowType

        flags, iterations, constraints = [x.numpy() for x in traces]
        np.savez_compressed(
            a.out / "failure-diagnostics.npz",
            overflow=flags,
            iterations=iterations,
            constraints=constraints,
        )
        positions = np.argwhere(flags != 0)
        detail.update(
            failed_world_steps=positions[:30].tolist(),
            failure_count=len(positions),
            max_iterations=int(iterations.max()),
            max_constraints=int(constraints.max()),
            overflow_codes={
                str(int(x)): [
                    name
                    for name, v in OverflowType.__members__.items()
                    if int(v) > 0 and (int(v) & (int(v) - 1)) == 0 and int(x) & int(v)
                ]
                for x in np.unique(flags)
                if x
            },
        )
    save(status="FAIL", failure=detail)
    raise
finally:
    engine.close()
