"""Bounded original-task PPO pilot; no production source or run mutation."""

import hashlib
import importlib.util
import json
import sys
import time
from pathlib import Path

import mujoco
import numpy as np
import torch
from assembly_policy import train
from assembly_policy.wowrobo import env as native

root, output = map(Path, sys.argv[1:3])
meta = json.loads(
    (
        root
        / "robotics-assembly-3.0/assets_local/wowrobo/training-accepted-v6/manifest.json"
    ).read_text()
)
profile = json.loads(Path(meta["profile_path"]).read_text())
spec = importlib.util.spec_from_file_location("bare", output / "wowrobo_bare.py")
bare = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bare)
model = mujoco.MjModel.from_binary_path(str(output / "model.mjb"))
receipt = json.loads((output / "scene.json").read_text())
native.BatchedAssemblyEnv = bare.environment_class(
    native, model, profile, receipt["scene_sha256"]
)
torch.set_num_threads(2)
run = output / "pilot-100"
assert not (run / "checkpoint.pt").exists(), (
    "Do not silently resume a pilot with altered physics"
)
args = train.parse(
    [
        "--run-dir",
        str(run),
        "--bank-dir",
        str(root / "robotics-assembly-3.0/assets_local/wowrobo/training-accepted-v6"),
        "--envs",
        "32",
        "--steps",
        "16",
        "--iters",
        "100",
        "--epochs",
        "3",
        "--nthread",
        "4",
        "--device",
        "cuda",
        "--save-every",
        "10",
        "--no-eval",
        "--wowrobo-debug-resets",
        "--bank-per-family",
        "0",
        "--action-scale-joints",
        ".035+.035+.035+.035+.035",
        "--max-hours",
        ".12",
    ]
)
tr = train.Trainer(args)
# Frozen copy of a compatible current actor, never its optimizer or SGS state.
checkpoint = root / "run-v6/checkpoint.pt"
if checkpoint.exists():
    payload = checkpoint.read_bytes()
    (output / "source-policy.pt").write_bytes(payload)
    ck = torch.load(
        output / "source-policy.pt", map_location=tr.dev, weights_only=False
    )
    assert ck["args"]["observation_signature"] == tr.pts.signature
    assert ck["args"]["model_signature"] == meta["model_signature"]
    tr.model.load_state_dict(ck["model"])
    source = {
        "path": str(checkpoint),
        "sha256": hashlib.sha256(payload).hexdigest(),
        "iteration": ck.get("it"),
        "transitions": ck.get("transitions"),
    }
else:
    source = {"initialization": "fresh policy"}
tr.args.update(
    run_status="APPROXIMATE_ASSEMBLY_PILOT",
    physics_backend="bare-mujoco-boxes",
    physics_signature=receipt["scene_sha256"],
    model_signature=receipt["scene_sha256"],
    source_model_signature=meta["model_signature"],
    source_policy=source,
    penetration_termination=False,
    velocity_termination=False,
    inter_part_collision=False,
    robot_self_collision=False,
)
(run / "config.json").write_text(json.dumps(tr.args, indent=2) + "\n")


def evaluate(label):
    # Same eval rows, actions and reward before/after; held-out from PPO reset bank.
    states = []
    tasks = []
    snapshot = root / "v6/accepted-states-v6"
    frozen = json.loads((snapshot / "manifest.json").read_text())
    stages = json.loads((snapshot / "nominals/stages/manifest.json").read_text())[
        "stage_by_task"
    ]
    data = mujoco.MjData(model)
    fullsize = mujoco.mj_stateSize(model, mujoco.mjtState.mjSTATE_FULLPHYSICS)
    selected = []
    for ti, task in enumerate(tr.tasks):
        eligible = [
            r
            for r in frozen["records"]
            if r["split"] == "eval"
            and stages[r["stage"]]["task"] in (task, "initial_scene")
        ]
        ids = np.random.default_rng(7100 + ti).choice(len(eligible), 8, replace=False)
        for index in ids:
            record = eligible[index]
            path = snapshot / record["archive"]
            assert (
                hashlib.sha256(path.read_bytes()).hexdigest()
                == record["archive_sha256"]
            )
            z = np.load(path)
            mujoco.mj_resetData(model, data)
            source_spec = int(mujoco.mjtState.mjSTATE_INTEGRATION) & ~int(
                mujoco.mjtState.mjSTATE_USERDATA
            )
            mujoco.mj_setState(model, data, z["states"][0], source_spec)
            np.testing.assert_array_equal(data.ctrl, z["controls"][0])
            full = np.empty(fullsize)
            mujoco.mj_getState(model, data, full, mujoco.mjtState.mjSTATE_FULLPHYSICS)
            friction = [
                record["friction_multipliers"][k]
                for k in ("finger_part", "part_table", "part_part")
            ]
            states.append(
                np.r_[full, data.eq_active, data.ctrl, data.qacc_warmstart, friction]
            )
            tasks.append(ti)
            selected.append(
                {"task": task, "archive": str(path), "sha256": record["archive_sha256"]}
            )
    (output / "evaluation-split.json").write_text(json.dumps(selected, indent=2) + "\n")
    env = native.BatchedAssemblyEnv(tr.tasks, np.array(tasks), tr.ecfg)
    env.set_states(np.arange(len(states)), np.array(states), np.array(tasks))
    done_total = np.zeros(len(states), bool)
    success = np.zeros(len(states), bool)
    failed = np.zeros(len(states), bool)
    frames = []
    tr.model.eval()
    for step in range(env.horizon):
        with torch.no_grad():
            ob = tr._obs(env.observe())
            dist = tr.model.dist(tr._points(ob), ob["prop"])
            act = dist.mean.cpu().numpy()
        _rew, done, info = env.step(act)
        first = ~done_total
        success |= first & info["success"]
        failed |= first & info["unstable"]
        done_total |= done
        if step < 100:
            frames.append(np.array([d.qpos for d in env.data]))
        if done_total.all():
            break
    result = {
        "episodes": len(states),
        "successes": int(success.sum()),
        "success_rate": float(success.mean()),
        "numerical_failures": int(failed.sum()),
        "by_task": {
            task: {
                "n": int(np.sum(np.array(tasks) == ti)),
                "success": int(success[np.array(tasks) == ti].sum()),
            }
            for ti, task in enumerate(tr.tasks)
        },
        "scope": "same held-out rows under approximate physics, not native-contact assembly certification",
    }
    (output / (label + "-eval.json")).write_text(json.dumps(result, indent=2) + "\n")
    np.savez_compressed(
        output / (label + "-eval.npz"),
        qpos=np.array(frames),
        tasks=tasks,
        initial_states=states,
        success=success,
    )
    env.close()
    print(label, json.dumps(result), flush=True)
    return result


before = evaluate("before")
start = time.perf_counter()
try:
    tr.train()
finally:
    tr.env.close()
elapsed = time.perf_counter() - start
after = evaluate("after")
summary = {
    "iterations": tr.it,
    "transitions": tr.transitions,
    "training_s": elapsed,
    "transitions_per_s": tr.transitions / elapsed,
    "source_policy": source,
    "before": before,
    "after": after,
    "physics_signature": receipt["scene_sha256"],
}
(output / "pilot-summary.json").write_text(json.dumps(summary, indent=2) + "\n")
print(json.dumps(summary), flush=True)
