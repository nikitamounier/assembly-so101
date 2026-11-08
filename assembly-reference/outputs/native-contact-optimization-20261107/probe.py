"""Frozen-runtime A/B probe. Isolated libraries; no learner attachment or mutation."""

import argparse
import hashlib
import importlib.util
import json
import sys
import time
from pathlib import Path

import mujoco
import numpy as np

p = argparse.ArgumentParser()
p.add_argument("root", type=Path)
p.add_argument("experiment", type=Path)
p.add_argument("variant", choices=["baseline", "candidate", "exact"])
p.add_argument("--panel", choices=["quick", "all"], default="quick")
p.add_argument("--mode", type=int, default=1)
p.add_argument("--no-profile", action="store_true")
args = p.parse_args()
canonical = args.root / "robotics-assembly-3.0"
name = "_wowrobo_training_native_backend"
spec = importlib.util.spec_from_file_location(
    name, args.experiment / ("backend-before.py" if args.variant=="baseline" else ("backend-exact.py" if args.variant=="exact" else "backend-candidate.py"))
)
backend = importlib.util.module_from_spec(spec)
sys.modules[name] = backend
spec.loader.exec_module(backend)
backend.SOURCE = args.experiment / args.variant
original_load = backend.load_backend
backend.load_backend = lambda manifest: original_load(
    manifest, args.experiment / args.variant / "native"
)

from assembly_policy.wowrobo.env import BatchedAssemblyEnv, EnvConfig

bank = canonical / "assets_local/wowrobo/training-accepted-v5"
meta = json.loads((bank / "manifest.json").read_text())
env = BatchedAssemblyEnv(
    meta["task_names"],
    np.zeros(1, np.int64),
    EnvConfig(
        profile_path=meta["profile_path"], review_model=meta["review_model"], nthread=1
    ),
)
backend.set_surface_mode(args.mode)
original_engine = env.engine
arrays, rows, timing = {}, [], {}


class Capture:
    def rollout(self, *a, **kw):
        if not args.no_profile:
            backend.surface_stats(reset=True)
        started = time.perf_counter()
        trajectory, sensors = original_engine.rollout(*a, **kw)
        timing["rollout_s"] = time.perf_counter() - started
        timing["stats"] = (
            [] if args.no_profile else list(backend.surface_stats(reset=True))
        )
        arrays["trajectory"] = trajectory.copy()
        arrays["sensors"] = sensors.copy()
        return trajectory, sensors

    def close(self):
        original_engine.close()


env.engine = Capture()
cases = []
for ti, task in enumerate(meta["task_names"]):
    with np.load(bank / (task + "_train.npz")) as z:
        for stage in np.unique(z["stage"]):
            ids = np.flatnonzero(z["stage"] == stage)
            prototypes = [i for i in ids if np.all(z["states"][i, -3:] == 1)]
            if prototypes:
                cases.append(
                    (ti, task, str(stage), False, z["states"][prototypes[0]].copy())
                )
            if stage == "21_upper_far_side":
                variations = [i for i in ids if np.any(z["states"][i, -3:] != 1)]
                if variations:
                    cases.append(
                        (ti, task, str(stage), True, z["states"][variations[0]].copy())
                    )
if args.panel == "quick":
    cases = [c for c in cases if c[2] in ("05_cover_near", "21_upper_far_side") or (c[2]=="00_loose" and c[0]==0)]
output = args.experiment / (
    f"{args.variant}-{args.panel}-mode{args.mode}"
    + ("-unprofiled" if args.no_profile else "")
)
saved = {}
try:
    for ci, (ti, task, stage, variation, state) in enumerate(cases):
        env.set_states(np.array([0]), state[None], np.array([ti]))
        for step in range(3 if args.panel == "quick" else 1):
            # Zero-action first interval preserves earlier benchmark; subsequent
            # deterministic controls exercise both arms and grippers.
            action = (
                np.zeros((1, 12))
                if step == 0
                else np.random.default_rng(701 + step).uniform(-0.15, 0.15, (1, 12))
            )
            cpu_start = time.process_time()
            start = time.perf_counter()
            reward, done, info = env.step(action)
            wall = time.perf_counter() - start
            d = env.data[0]
            forces = np.zeros((d.ncon, 6))
            for i in range(d.ncon):
                mujoco.mj_contactForce(env.m, d, i, forces[i])
            contact = np.array(
                [np.r_[c.dist, c.pos, c.frame, c.geom, c.friction] for c in d.contact]
            )
            values = dict(
                arrays,
                state=env.state.copy(),
                reward=reward,
                done=done,
                contacts=contact,
                forces=forces,
                qpos=d.qpos.copy(),
                qvel=d.qvel.copy(),
                warmstart=d.qacc_warmstart.copy(),
                observation=env.hist.copy(),
            )
            values.update({"info_" + k: v for k, v in info.items()})
            for k, v in values.items():
                saved[f"case{ci}_step{step}_{k}"] = v
            row = dict(
                case=ci,
                task=task,
                stage=stage,
                variation=variation,
                step=step,
                wall_s=wall,
                cpu_s=time.process_time()-cpu_start,
                **timing,
                ncon=d.ncon,
            )
            rows.append(row)
            print(json.dumps(row), flush=True)
    np.savez_compressed(str(output) + ".npz", **saved)
    result = {
        "variant": args.variant,
        "panel": args.panel,
        "mode": args.mode,
        "model_signature": meta["model_signature"],
        "rows": rows,
        "source_sha256": hashlib.sha256(
            (backend.SOURCE / "plugin/so101_surface_tree.h").read_bytes()
        ).hexdigest(),
    }
    output.with_suffix(".json").write_text(json.dumps(result, indent=2) + "\n")
finally:
    env.close()
