"""Matched frozen-state benchmark; candidate physics deliberately differs."""

import argparse
import hashlib
import importlib.util
import json
import os
import socket
import time
from pathlib import Path

import mujoco
import numpy as np
import torch
from assembly_policy.wowrobo import env as env_module
from assembly_policy.wowrobo.obs import PointTables

p = argparse.ArgumentParser()
p.add_argument("root", type=Path)
p.add_argument("output", type=Path)
p.add_argument("variant", choices=["baseline", "candidate"])
p.add_argument("--steps", type=int, default=1)
p.add_argument("--envs", type=int, default=1)
p.add_argument("--nthread", type=int, default=1)
p.add_argument("--variation-only", action="store_true")
p.add_argument("--stage-filter", default="")
p.add_argument("--case-indices", default="")
p.add_argument("--tag", default="")
p.add_argument("--cached-baseline", action="store_true")
p.add_argument("--repeat-resets", action="store_true")
p.add_argument("--candidate-scene", default="candidate/scene.xml")
p.add_argument("--candidate-native", action="store_true")
a = p.parse_args()
bank = a.root / "robotics-assembly-3.0/assets_local/wowrobo/training-accepted-v5"
meta = json.loads((bank / "manifest.json").read_text())
if a.variant == "candidate" or a.cached_baseline:
    profile = json.loads(Path(meta["profile_path"]).read_text())
    hook_path = (
        a.root
        / "robotics-assembly-3.0/assembly_policy/so101/wowrobo_rollout_friction.py"
    )
    spec = importlib.util.spec_from_file_location("_proxy_friction", hook_path)
    hook = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(hook)
    if a.variant != "candidate" or a.candidate_native:
        backend_path = (
            a.root
            / "robotics-assembly-3.0/assembly_policy/so101/wowrobo_contact_backend.py"
        )
        bs = importlib.util.spec_from_file_location("_repeat_backend", backend_path)
        backend = importlib.util.module_from_spec(bs)
        bs.loader.exec_module(backend)
        backend.load_backend(profile["grid_manifest"])
        model = mujoco.MjModel.from_binary_path(
            str(a.output / "candidate/source-model.mjb")
        )
    if a.variant == "candidate":
        scene_path = a.output / a.candidate_scene
        model = (
            mujoco.MjModel.from_binary_path(str(scene_path))
            if scene_path.suffix == ".mjb"
            else mujoco.MjModel.from_xml_path(str(scene_path))
        )
    hook.install(model, a.output / "candidate/friction")
    signature = (
        hashlib.sha256((a.output / a.candidate_scene).read_bytes()).hexdigest()
        if a.variant == "candidate"
        else meta["model_signature"]
    )
    variant_profile = (
        dict(profile, collision_proxy=signature)
        if a.variant == "candidate"
        else profile
    )
    env_module.load_runtime = lambda _: (
        model,
        variant_profile,
        signature,
    )
config = env_module.EnvConfig(
    profile_path=meta["profile_path"],
    review_model=meta["review_model"],
    nthread=a.nthread,
)
env = env_module.BatchedAssemblyEnv(
    meta["task_names"], np.zeros(a.envs, np.int64), config
)
tables = PointTables(meta["task_names"], review_model=meta["review_model"])
point_hash = hashlib.sha256(
    tables.points.cpu().numpy().tobytes()
    + tables.which.cpu().numpy().tobytes()
    + tables.grip.cpu().numpy().tobytes()
).hexdigest()
arrays, rows = {}, []


def contacts(data):
    result = []
    for c in data.contact:
        g1, g2 = map(int, c.geom)
        b1, b2 = map(int, env.m.geom_bodyid[[g1, g2]])
        result.append(
            {
                "geoms": [env.m.geom(g1).name, env.m.geom(g2).name],
                "bodies": [env.m.body(b1).name, env.m.body(b2).name],
                "dist": float(c.dist),
            }
        )
    return result


try:
    ci = 0
    for ti, task in enumerate(meta["task_names"]):
        with np.load(bank / (task + "_train.npz")) as z:
            for stage in np.unique(z["stage"]):
                ids = np.flatnonzero(z["stage"] == stage)
                matches = [
                    i
                    for i in ids
                    if bool(np.all(z["states"][i, -3:] == 1)) != a.variation_only
                ]
                if not matches:
                    continue
                if a.stage_filter and str(stage) not in a.stage_filter.split(","):
                    ci += 1
                    continue
                if a.case_indices and ci not in set(
                    map(int, a.case_indices.split(","))
                ):
                    ci += 1
                    continue
                env.set_states(
                    np.arange(a.envs),
                    np.repeat(z["states"][matches[0]][None], a.envs, axis=0),
                    np.full(a.envs, ti),
                )
                observation = env.observe()
                kwargs = {
                    k: torch.from_numpy(v)
                    for k, v in observation.items()
                    if k not in ["proprio"]
                }
                encoded = tables(**kwargs)
                if isinstance(encoded, tuple):
                    for k, v in enumerate(encoded):
                        arrays[f"case{ci}_initial_points_{k}"] = (
                            v.detach().cpu().numpy()
                        )
                else:
                    arrays[f"case{ci}_initial_points"] = encoded.detach().cpu().numpy()
                initial = contacts(env.data[0])
                arrays[f"case{ci}_initial_qpos"] = env.data[0].qpos.copy()
                arrays[f"case{ci}_initial_bodypos"] = env.data[0].xpos.copy()
                arrays[f"case{ci}_initial_bodyquat"] = env.data[0].xquat.copy()
                for step in range(a.steps):
                    if a.repeat_resets and step:
                        env.set_states(
                            np.arange(a.envs),
                            np.repeat(z["states"][matches[0]][None], a.envs, axis=0),
                            np.full(a.envs, ti),
                        )
                    start = time.perf_counter()
                    reward, done, info = env.step(np.zeros((a.envs, 12)))
                    elapsed = time.perf_counter() - start
                    row = {
                        "case": ci,
                        "task": task,
                        "stage": str(stage),
                        "step": step,
                        "wall_s": elapsed,
                        "ncon": env.data[0].ncon,
                        "contacts": contacts(env.data[0]),
                        "initial_contacts": initial if step == 0 else [],
                        "reward": reward.tolist(),
                        "done": done.tolist(),
                        "info": {k: v.tolist() for k, v in info.items()},
                    }
                    rows.append(row)
                    arrays[f"case{ci}_step{step}_state"] = env.state.copy()
                    arrays[f"case{ci}_step{step}_bodypos"] = env.data[0].xpos.copy()
                    arrays[f"case{ci}_step{step}_bodyquat"] = env.data[0].xquat.copy()
                    print(
                        json.dumps(
                            {
                                k: row[k]
                                for k in [
                                    "case",
                                    "stage",
                                    "step",
                                    "wall_s",
                                    "ncon",
                                    "info",
                                ]
                            }
                        ),
                        flush=True,
                    )
                ci += 1
    np.savez_compressed(a.output / (a.variant + a.tag + "-replay.npz"), **arrays)
    result = {
        "hostname": socket.gethostname(),
        "cpu_affinity": sorted(os.sched_getaffinity(0)),
        "variant": a.variant,
        "source_model_signature": meta["model_signature"],
        "point_signature": tables.signature,
        "point_array_sha256": point_hash,
        "review_model": meta["review_model"],
        "rows": rows,
        "ngeom": env.m.ngeom,
        "npair": env.m.npair,
        "nbody": env.m.nbody,
        "body_names": [env.m.body(i).name for i in range(env.m.nbody)],
        "nq": env.m.nq,
        "nv": env.m.nv,
        "substeps": env.substeps,
        "timestep": float(env.m.opt.timestep),
        "steps": a.steps,
        "envs": a.envs,
        "nthread": a.nthread,
        "variation_only": a.variation_only,
        "repeat_resets": a.repeat_resets,
    }
    (a.output / (a.variant + a.tag + "-benchmark.json")).write_text(
        json.dumps(result, indent=2) + "\n"
    )
finally:
    env.close()
