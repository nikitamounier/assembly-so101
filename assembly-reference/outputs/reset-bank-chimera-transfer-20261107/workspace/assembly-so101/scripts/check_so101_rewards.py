"""Audit every native subtask's reward and reject misplaced diagnostic fixtures.

Completed poses are explicit initialization, not learned or executed assembly.
Negative cases fork a genuinely rewarded context to test streak invalidation.
"""

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import mujoco
import numpy as np

from assembly_policy.so101 import scene
from assembly_policy.so101.env import STATE_SPEC, TASK_NAMES, BatchedAssemblyEnv, EnvConfig
from assembly_policy.so101.goals import placement_errors
from assembly_policy.so101.limits import PENETRATION_LIMIT_M
from assembly_policy.so101.resets import check_bank
from assembly_policy.so101.task_scenes import task_scene
from assembly_policy.so101.workcell import default_bank_directory, environment_identity


def _states(env, prepared):
    states = []
    for task in TASK_NAMES:
        d = mujoco.MjData(env.m)
        p = prepared[task]
        d.qpos[:] = p["qpos"]
        d.qvel[:] = p["qvel"]
        d.ctrl[:] = p["ctrl"]
        state = np.empty(env.nstate)
        mujoco.mj_getState(env.m, d, state, STATE_SPEC)
        states.append(state)
    return np.asarray(states)


def _sample(env, reward, info, i):
    return dict(
        time_s=float(env.state[i, 0]),
        reward=float(reward[i]),
        cost=float(info["cost"][i]),
        success=bool(info["success"][i]),
        safety=bool(info["safety"][i]),
        placement_error_m=float(info["d_asm"][i]),
        stability_samples=int(env.stable[i]),
    )


def audit_rewards(*, ticks=30, bank=None):
    if ticks < 11:
        raise ValueError("At least eleven pose samples are required")
    bank = Path(bank or default_bank_directory())
    check_bank(bank)
    model = scene.model(contact_monitor=True)
    prepared = {task: task_scene(model, task) for task in TASK_NAMES}
    report = dict(
        environment=environment_identity(),
        scope="Reward/target diagnostics; not executed assembly or learned policy performance",
        cases=[],
    )
    ids = np.arange(len(TASK_NAMES))
    action = np.zeros((len(ids), 12))
    for label, cfg in (
        ("default", EnvConfig(nthread=1)),
        ("guarded", EnvConfig(nthread=1, stability_metric="pose", max_contact_depth=PENETRATION_LIMIT_M)),
    ):
        env = BatchedAssemblyEnv(TASK_NAMES, ids, cfg)
        try:
            env.set_states(ids, _states(env, prepared))
            traces = [[] for _ in ids]
            terminal = np.zeros(len(ids), bool)
            snapshot = {
                k: np.empty_like(getattr(env, k))
                for k in (
                    "state",
                    "prev_action",
                    "last_ctrl",
                    "hist",
                    "t",
                    "stable",
                    "stable_position",
                    "stable_quaternion",
                )
            }
            for _ in range(ticks):
                reward, done, info = env.step(action)
                for i in ids:
                    if terminal[i]:
                        continue
                    traces[i].append(_sample(env, reward, info, i))
                    if done[i]:
                        terminal[i] = True
                        for key, values in snapshot.items():
                            values[i] = getattr(env, key)[i]
                if terminal.all():
                    break
            for i, task in enumerate(TASK_NAMES):
                final = traces[i][-1]
                report["cases"].append(
                    dict(
                        case="completed",
                        profile=label,
                        task=task,
                        config=asdict(cfg),
                        history=traces[i],
                        passed=bool(
                            final["success"]
                            and final["reward"] > 0
                            and not final["safety"]
                            and final["time_s"] >= 0.4
                            and final["placement_error_m"] < 0.001
                        ),
                    )
                )
            if label != "guarded":
                continue
            if not all(t[-1]["success"] for t in traces):
                raise ValueError("Cannot construct negative contexts without genuine preceding success")

            def restore():
                for key, values in snapshot.items():
                    getattr(env, key)[:] = values
                env._refresh()

            for case in ("misplaced_part", "misplaced_predecessor", "loose_reset"):
                restore()
                selected = ids if case != "misplaced_predecessor" else ids[1:]
                if case == "loose_reset":
                    states = []
                    for task in TASK_NAMES:
                        with np.load(bank / f"{task}_eval.npz") as archive:
                            states.append(archive["states"][0].copy())
                    env.set_states(ids, np.asarray(states))
                    if env.stable.any():
                        raise AssertionError("Reset retained a previous success streak")
                else:
                    for i in selected:
                        part = TASK_NAMES[i] if case == "misplaced_part" else TASK_NAMES[i - 1]
                        q = int(env.m.joint(part + "_free").qposadr[0])
                        env.state[i, 1 + q] += 0.18
                    env._refresh()
                before = env.assembly_distance()
                reward, done, info = env.step(action)
                for i in selected:
                    sample = _sample(env, reward, info, i)
                    report["cases"].append(
                        dict(
                            case=case,
                            profile=label,
                            task=TASK_NAMES[i],
                            initial_error_m=float(before[i]),
                            result=sample,
                            passed=bool(
                                before[i] > 0.001
                                and sample["placement_error_m"] > 0.001
                                and not sample["success"]
                                and sample["reward"] <= 0
                                and env.stable[i] == 0
                            ),
                        )
                    )
            # These two cases audit geometry only. Wrong rotations can intersect
            # the receiving part, so they are never advanced as physical starts.
            for case in ("wrong_orientation_geometry", "translated_geometry"):
                restore()
                for i, task in enumerate(TASK_NAMES):
                    d = env.data[i]
                    expected = env.assembly_distance()[i]
                    if case == "translated_geometry":
                        for name in env.names:
                            q = int(env.m.joint(name + "_free").qposadr[0])
                            d.qpos[q] -= 0.03
                    else:
                        q = int(env.m.joint(task + "_free").qposadr[0])
                        rotation = np.empty(4)
                        quat = np.empty(4)
                        mujoco.mju_axisAngle2Quat(rotation, np.array([1.0, 0, 0]), 0.6)
                        mujoco.mju_mulQuat(quat, rotation, d.qpos[q + 3 : q + 7])
                        d.qpos[q + 3 : q + 7] = quat
                    mujoco.mj_kinematics(env.m, d)
                    error = max(placement_errors(d, task, env.meta, env.origins).values())
                    passed = (
                        error > 0.001
                        if case == "wrong_orientation_geometry"
                        else abs(error - expected) < 1e-12 and error < 0.001
                    )
                    report["cases"].append(
                        dict(
                            case=case,
                            profile=label,
                            task=task,
                            scope="Geometry only; no physics rollout",
                            placement_error_m=error,
                            passed=bool(passed),
                        )
                    )
        finally:
            env.close()
    report["summary"] = dict(
        cases=len(report["cases"]),
        passed=sum(x["passed"] for x in report["cases"]),
        failed=sum(not x["passed"] for x in report["cases"]),
    )
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, default=scene.ROOT / "artifacts_local/subtask-rewards/report.json")
    p.add_argument("--ticks", type=int, default=30)
    p.add_argument("--bank-dir", type=Path, default=default_bank_directory())
    a = p.parse_args()
    report = audit_rewards(ticks=a.ticks, bank=a.bank_dir)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(report, indent=2) + "\n")
    for row in report["cases"]:
        print(json.dumps({k: v for k, v in row.items() if k not in ("history", "config")}), flush=True)
    print(json.dumps(report["summary"]), flush=True)
    if report["summary"]["failed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
