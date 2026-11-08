"""Build and smoke-test original assembly tasks with fast approximate physics."""

import importlib.util
import json
import socket
import sys
import time
from pathlib import Path

import mujoco
import numpy as np
import torch
from assembly_policy.so101.wowrobo_reward import RewardRegions
from assembly_policy.wowrobo import env as native
from assembly_policy.wowrobo.obs import PointTables

root, output = map(Path, sys.argv[1:3])
meta = json.loads(
    (
        root
        / "robotics-assembly-3.0/assets_local/wowrobo/training-accepted-v6/manifest.json"
    ).read_text()
)
profile = json.loads(Path(meta["profile_path"]).read_text())
output.mkdir(parents=True, exist_ok=True)
spec = importlib.util.spec_from_file_location("bare", output / "wowrobo_bare.py")
bare = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bare)
review = mujoco.MjModel.from_xml_path(meta["review_model"])
old_output = output.parent / "simple-collision-20261107"
torch.set_num_threads(1)
tables = PointTables(meta["task_names"], review_model=meta["review_model"])
from assembly_policy.so101.wowrobo_obs import GRIPS

finger_points = {name: tables.grip[i].numpy() for i, name in enumerate(GRIPS)}
model, receipt = bare.build(
    (old_output / "baseline-runtime.xml").read_text(),
    review,
    output / "scene.xml",
    finger_points=finger_points,
)
mujoco.mj_saveModel(model, str(output / "model.mjb"))
cls = bare.environment_class(native, model, profile, receipt["scene_sha256"])
(output / "original-task-bare-env.py").write_text(cls.modified_source)
env = cls(
    meta["task_names"],
    np.zeros(1, np.int64),
    native.EnvConfig(
        profile_path=meta["profile_path"], review_model=meta["review_model"], nthread=1
    ),
)
torch.set_num_threads(1)
reference = np.load(old_output / "baseline-hardware-replay.npz")
reference_reward = RewardRegions(review, profile["source_model_sha256"])
rd = mujoco.MjData(review)
rows = []
selected = []
arrays = {}
ci = 0
for ti, task in enumerate(meta["task_names"]):
    z = np.load(
        root
        / "robotics-assembly-3.0/assets_local/wowrobo/training-accepted-v6"
        / (task + "_train.npz")
    )
    for stage in np.unique(z["stage"]):
        ids = [
            i
            for i in np.flatnonzero(z["stage"] == stage)
            if np.all(z["states"][i, -3:] == 1)
        ]
        if not ids:
            continue
        state = z["states"][ids[0]]
        env.set_states(np.array([0]), state[None], np.array([ti]))
        o = env.observe()
        points = tables(
            **{k: torch.from_numpy(v) for k, v in o.items() if k != "proprio"}
        )
        assert np.array_equal(points.numpy(), reference[f"case{ci}_initial_points"]), (
            ci,
            "points",
        )
        rd.qpos[:] = env.data[0].qpos
        mujoco.mj_kinematics(review, rd)
        assert reference_reward.snapshot(rd, {"task": task}) == env.reward.snapshot(
            env.data[0], {"task": task}
        ), (ci, "reward")
        times = []
        infos = []
        for repeat in range(5):
            env.set_states(np.array([0]), state[None], np.array([ti]))
            start = time.perf_counter()
            reward, done, info = env.step(np.zeros((1, 12)))
            times.append(time.perf_counter() - start)
            infos.append({k: v.tolist() for k, v in info.items()})
        row = {
            "case": ci,
            "task": task,
            "stage": str(stage),
            "wall_s": float(np.median(times)),
            "times": times,
            "info": infos[-1],
            "reward": reward.tolist(),
        }
        rows.append(row)
        print(json.dumps(row), flush=True)
        selected.append((ti, str(stage), state.copy()))
        ci += 1
long = []
for ti, stage, state in selected:
    if stage not in (
        "05_cover_near",
        "10_connector_near",
        "15_pin_near",
        "23_upper_near",
        "31_final_near",
    ):
        continue
    env.set_states(np.array([0]), state[None], np.array([ti]))
    captures = []
    errors = []
    safety = 0
    unstable = 0
    reward_sum = 0
    for t in range(64):
        # Small deterministic real arm commands, then hold; original action scaling.
        action = np.zeros((1, 12))
        action[:, [0, 6]] = 0.1 * np.sin(t * 0.15) if t < 32 else 0
        reward, done, info = env.step(action)
        captures.append(env.data[0].qpos.copy())
        errors.append(float(info["d_asm"][0]))
        reward_sum += float(reward[0])
        safety += int(info["safety"][0])
        unstable += int(info["unstable"][0])
        if done.any():
            break
    arrays[stage + "_qpos"] = np.array(captures)
    long.append(
        {
            "task": meta["task_names"][ti],
            "stage": stage,
            "intervals": len(captures),
            "reward_sum": reward_sum,
            "safety": safety,
            "unstable": unstable,
            "initial_d_asm": errors[0],
            "final_d_asm": errors[-1],
        }
    )
np.savez_compressed(output / "rollout.npz", **arrays)
result = {
    "hostname": socket.gethostname(),
    "receipt": receipt,
    "source_model_signature": meta["model_signature"],
    "point_signature": tables.signature,
    "initial_point_cases_exact": ci,
    "initial_seating_reward_snapshots_exact": ci,
    "reset_payload_size": env.nstate,
    "substeps": env.substeps,
    "rows": rows,
    "total_median_step_s": sum(r["wall_s"] for r in rows),
    "long": long,
    "guard_source_sha256": cls.modified_environment_sha256,
}
(output / "probe.json").write_text(json.dumps(result, indent=2) + "\n")
print(
    json.dumps({k: v for k, v in result.items() if k not in ("rows", "receipt")}),
    flush=True,
)
env.close()
