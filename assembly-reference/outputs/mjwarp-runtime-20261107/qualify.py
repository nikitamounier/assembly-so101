"""Bounded native/current-environment integration test. Run on an idle H100."""

import argparse
import importlib.util
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--release", type=Path, required=True)
parser.add_argument("--bank", type=Path, required=True)
parser.add_argument("--out", type=Path, required=True)
parser.add_argument("--worlds", type=int, default=2)
parser.add_argument("--steps", type=int, default=4)
parser.add_argument("--controls", type=int, default=2)
args = parser.parse_args()
args.out.mkdir(parents=True, exist_ok=True)
os.environ["WOWROBO_CANONICAL_ROOT"] = str(
    args.release.parent / "robotics-assembly-3.0"
)
sys.path.insert(0, str(args.release / "assembly-so101"))
# Reuse installed Torch only; the isolated environment supplies MuJoCo/Warp.
sys.path.append(
    str(args.release.parents[1] / "source-85c006a/.venv/lib/python3.12/site-packages")
)
import warp as wp

wp.init()
wp.set_device("cuda:0")
from assembly_policy.wowrobo.env import BatchedAssemblyEnv, EnvConfig

meta = json.loads((args.bank / "manifest.json").read_text())
tasks = meta["task_names"]
cfg = EnvConfig(
    profile_path=meta["profile_path"], review_model=meta["review_model"], nthread=1
)
native = BatchedAssemblyEnv(tasks, np.arange(args.worlds) % len(tasks), cfg)
cfg.control_dt = args.steps * native.m.opt.timestep
native.substeps = args.steps
native.ecfg.control_dt = cfg.control_dt
states = []
stages = (
    "05_cover_near",
    "10_connector_near",
    "15_pin_near",
    "21_upper_far_side",
    "29_final_far_side",
)
for i in range(args.worlds):
    with np.load(args.bank / (tasks[i % len(tasks)] + "_train.npz")) as bank:
        rows = np.flatnonzero(bank["stage"] == stages[i % len(stages)])
        if not len(rows):
            raise ValueError("Missing representative stage")
        states.append(bank["states"][rows[min(i, len(rows) - 1)]])
states = np.asarray(states)
native.set_states(np.arange(args.worlds), states)

# Load only the new modules into the frozen trainer's package namespace.
for name in ("wowrobo_warp", "wowrobo_warp_runtime"):
    spec = importlib.util.spec_from_file_location(
        "assembly_policy.so101." + name, Path(__file__).parent / (name + ".py")
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
runtime = sys.modules["assembly_policy.so101.wowrobo_warp_runtime"]
gpu = BatchedAssemblyEnv(tasks, np.arange(args.worlds) % len(tasks), cfg)
gpu.substeps = args.steps
runtime.install_on_environment(gpu, args.out / "scratch")
gpu.set_states(np.arange(args.worlds), states)
out = {
    "status": "RUNNING",
    "worlds": args.worlds,
    "steps": args.steps,
    "identity": gpu.warp_identity,
    "bank": str(args.bank),
    "records": [],
}


def save():
    (args.out / "result.json").write_text(json.dumps(out, indent=2) + "\n")


save()
try:
    for k in range(args.controls):
        action = np.zeros((args.worlds, 12))
        action[:, 1] = 0.01 * (k + 1)
        begin = time.perf_counter()
        nr = native.step(action)
        native_s = time.perf_counter() - begin
        begin = time.perf_counter()
        gr = gpu.step(action)
        gpu_s = time.perf_counter() - begin
        assert all(np.isfinite(v).all() for v in gpu.observe().values())
        assert np.isfinite(gpu.state).all()
        record = {
            "index": k,
            "native_s": native_s,
            "gpu_s": gpu_s,
            "max_state_error": float(abs(native.state - gpu.state).max()),
            "native_reward": np.asarray(nr[0]).tolist(),
            "gpu_reward": np.asarray(gr[0]).tolist(),
            "native_safety": np.asarray(nr[2]["safety"]).tolist(),
            "gpu_safety": np.asarray(gr[2]["safety"]).tolist(),
            "gpu_unstable": np.asarray(gr[2]["unstable"]).tolist(),
            "max_body_position_error_m": float(
                max(
                    np.linalg.norm(a.xpos - b.xpos, axis=1).max()
                    for a, b in zip(native.data, gpu.data)
                )
            ),
            "max_qpos_error": float(
                abs(
                    native.state[:, 1 : 1 + gpu.m.nq] - gpu.state[:, 1 : 1 + gpu.m.nq]
                ).max()
            ),
            "observation_shapes": {
                key: list(value.shape) for key, value in gpu.observe().items()
            },
        }
        out["records"].append(record)
        np.savez(
            args.out / f"control-{k}.npz",
            native=native.state,
            gpu=gpu.state,
            native_reward=nr[0],
            gpu_reward=gr[0],
        )
        print(json.dumps(record), flush=True)
        save()
    # Reject invalid physics inputs before upload, with no contaminated graph state.
    engine = gpu.engine
    bad_control = np.zeros((args.worlds, args.steps, engine.ncontrol))
    rejected = 0
    for bad in (np.full_like(bad_control, np.nan), bad_control):
        try:
            engine.rollout(
                gpu.m, gpu.work, gpu.state[:, : gpu.full_size], bad, nstep=args.steps
            )
        except ValueError:
            rejected += 1
    assert rejected == 2
    out["invalid_inputs_rejected"] = rejected
    # Reset only one world; exercise distinct equality/userdata and reset warmstart.
    gpu.set_states(np.array([0]), states[:1])
    gpu.step(np.zeros((args.worlds, 12)))
    out["partial_reset_finite"] = bool(np.isfinite(gpu.state).all())
    out["status"] = "PASS_BOUNDED_INTEGRATION"
    save()
finally:
    native.close()
    gpu.close()
    from mujoco_warp._src import collision_driver

    out["hooks_restored"] = (
        collision_driver.sdf_narrowphase.__module__ != "_wowrobo_warp_collision_sdf"
    )
    save()
