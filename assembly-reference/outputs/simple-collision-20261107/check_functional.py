"""Grasp-contact and deliberately blocked/clear insertion controls at fixed poses."""

import importlib.util
import json
import sys
from pathlib import Path

import mujoco
import numpy as np
from assembly_policy.wowrobo import env as env_module

root, output = map(Path, sys.argv[1:3])
canonical = root / "robotics-assembly-3.0"
bank = (
    canonical
    / "assets_local/wowrobo"
    / (sys.argv[3] if len(sys.argv) > 3 else "training-accepted-v6")
)
meta = json.loads((bank / "manifest.json").read_text())
profile = json.loads(Path(meta["profile_path"]).read_text())
spec = importlib.util.spec_from_file_location(
    "native_backend", canonical / "assembly_policy/so101/wowrobo_contact_backend.py"
)
backend = importlib.util.module_from_spec(spec)
spec.loader.exec_module(backend)
backend.load_backend(profile["grid_manifest"])
m = mujoco.MjModel.from_binary_path(str(output / "candidate/source-model.mjb"))
scene = output / (sys.argv[4] if len(sys.argv) > 4 else "candidate/scene.xml")
tag = sys.argv[5] if len(sys.argv) > 5 else ""
c = (
    mujoco.MjModel.from_binary_path(str(scene))
    if scene.suffix == ".mjb"
    else mujoco.MjModel.from_xml_path(str(scene))
)
receipt = json.loads((output / "candidate/receipt.json").read_text())
reverse = {p: old for old, ps in receipt["mappings"].items() for p in ps}
models = [m, c]
envs = []
for i, model in enumerate(models):
    env_module.load_runtime = lambda _, model=model, i=i: (
        model,
        profile,
        "functional-" + str(i),
    )
    envs.append(
        env_module.BatchedAssemblyEnv(
            meta["task_names"],
            np.zeros(1, np.int64),
            env_module.EnvConfig(
                profile_path=meta["profile_path"],
                review_model=meta["review_model"],
                nthread=1,
            ),
        )
    )
config = json.loads((canonical / "configs/wowrobo_mating_profile.json").read_text())
steps = ["cover", "connector", "pin", "upper", "final"]
near = [
    "05_cover_near",
    "10_connector_near",
    "15_pin_near",
    "23_upper_near",
    "31_final_near",
]
rows = []
grasp_rows = []


def entries(model, data):
    result = []
    for con in data.contact:
        names = [model.geom(int(g)).name for g in con.geom]
        roots = [reverse.get(name, name) for name in names]
        bodies = [model.body(int(model.geom_bodyid[int(g)])).name for g in con.geom]
        result.append(
            {
                "geoms": roots,
                "bodies": bodies,
                "dist": float(con.dist),
                "pos": con.pos.tolist(),
                "normal": con.frame[:3].tolist(),
            }
        )
    return result


try:
    for ti, task in enumerate(meta["task_names"]):
        z = np.load(bank / (task + "_train.npz"))
        for stage in np.unique(z["stage"]):
            ids = [
                i
                for i in np.flatnonzero(z["stage"] == stage)
                if np.all(z["states"][i, -3:] == 1)
            ]
            if not ids:
                continue
            state = z["states"][ids[0]].copy()
            for env in envs:
                env.set_states(np.array([0]), state[None], np.array([ti]))
            sets = []
            for model, env in zip(models, envs, strict=True):
                pairs = set()
                for con in entries(model, env.data[0]):
                    if con["dist"] > 0.0001:
                        continue
                    if any(
                        "gripper" in body or "moving_jaw" in body
                        for body in con["bodies"]
                    ):
                        pairs.add(tuple(sorted(con["bodies"])))
                sets.append(pairs)
            grasp_rows.append(
                {
                    "task": task,
                    "stage": str(stage),
                    "lost": sorted(sets[0] - sets[1]),
                    "added": sorted(sets[1] - sets[0]),
                }
            )
        ids = [
            i
            for i in np.flatnonzero(z["stage"] == near[ti])
            if np.all(z["states"][i, -3:] == 1)
        ]
        state = z["states"][ids[0]].copy()
        pairs = {
            frozenset([p["geom1"], p["geom2"]]) for p in config["steps"][steps[ti]]
        }
        body = m.body(task).id
        root_joint = int(m.body_jntadr[body])
        assert m.jnt_type[root_joint] == mujoco.mjtJoint.mjJNT_FREE
        address = int(m.jnt_qposadr[root_joint])
        offsets = [np.zeros(3)] + [
            np.eye(3)[k] * sign * distance
            for distance in [0.00025, 0.0005, 0.001, 0.003]
            for k in range(3)
            for sign in [-1, 1]
        ]
        for offset in offsets:
            shifted = state.copy()
            shifted[1 + address : 1 + address + 3] += offset
            captures = []
            for model, env in zip(models, envs, strict=True):
                env.set_states(np.array([0]), shifted[None], np.array([ti]))
                captures.append(
                    [
                        x
                        for x in entries(model, env.data[0])
                        if frozenset(x["geoms"]) in pairs
                    ]
                )
            depths = [min([x["dist"] for x in contacts] + [0]) for contacts in captures]
            row = {
                "task": task,
                "offset_world_m": offset.tolist(),
                "baseline_deepest_m": depths[0],
                "candidate_deepest_m": depths[1],
                "missed_block": depths[0] < -0.0005 and depths[1] > -0.0001,
                "new_obstruction": depths[0] > -0.0001 and depths[1] < -0.0005,
                "contacts": captures,
            }
            rows.append(row)
            print(
                json.dumps({k: v for k, v in row.items() if k != "contacts"}),
                flush=True,
            )
    result = {
        "bank": str(bank),
        "scope": "Fixed-pose prototype grasp contacts and near-insertion 0.25/0.5/1/3mm Cartesian perturbations; no dynamic success certification",
        "insertion_controls": rows,
        "grasp_controls": grasp_rows,
        "missed_blocks": sum(r["missed_block"] for r in rows),
        "new_obstructions": sum(r["new_obstruction"] for r in rows),
        "starts_with_lost_grasp_pairs": [r for r in grasp_rows if r["lost"]],
    }
    (output / ("functional" + tag + "-check.json")).write_text(
        json.dumps(result, indent=2) + "\n"
    )
finally:
    for env in envs:
        env.close()
