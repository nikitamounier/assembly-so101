"""Fixed-state contact and trajectory capture for native acceleration qualification."""

import argparse
import ctypes
import hashlib
import json
import time
from pathlib import Path

import mujoco
import numpy as np

from assembly_policy.so101.env import STATE_SPEC
from assembly_policy.so101.scene import ASSEMBLY_TASKS, model
from assembly_policy.so101.task_scenes import geometric_scene
from assembly_policy.so101.workcell import ROOT, default_bank_directory, load_contact_search


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mode", type=int, choices=[0, 1, 2], default=1)
    parser.add_argument("--steps", type=int, default=16)
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--inputs", type=Path)
    a = parser.parse_args()
    a.output.mkdir(parents=True, exist_ok=False)
    m = model(contact_monitor=True)
    native = load_contact_search()
    if hasattr(native, "so101_surface_mode"):
        native.so101_surface_mode(a.mode)
    stats = (ctypes.c_double * 7)()
    if a.profile:
        native.so101_surface_stats(stats, 1)
    d = mujoco.MjData(m)
    states, labels = [], []
    for p in sorted(default_bank_directory().glob("*.npz")):
        with np.load(p) as bank:
            for i, state in enumerate(bank["states"]):
                states.append(state.copy())
                labels.append(f"{p.stem}/{i}")
    for task in ASSEMBLY_TASKS:
        g = geometric_scene(m, task)
        state = np.empty(mujoco.mj_stateSize(m, STATE_SPEC))
        mujoco.mj_getState(m, g, state, STATE_SPEC)
        states.append(state)
        labels.append(f"geometric/{task}")
    for name in ["motor_seated_pose", "motor_side_pose"]:
        fixture = json.loads((ROOT / f"tests/{name}.json").read_text())
        mujoco.mj_resetData(m, d)
        d.qpos[:] = fixture["qpos"]
        d.qvel[:] = fixture["qvel"]
        state = np.empty(mujoco.mj_stateSize(m, STATE_SPEC))
        mujoco.mj_getState(m, d, state, STATE_SPEC)
        states.append(state)
        labels.append(name)
    if a.inputs:
        with np.load(a.inputs) as fixed:
            states, labels = fixed["states"], fixed["labels"].tolist()
    # Immutable input payload is also retained to make repeatability inspectable.
    np.savez(a.output / "inputs.npz", states=states, labels=labels)
    contacts, counts, endpoints, times = [], [], [], []
    timer_totals = np.zeros((2, int(mujoco.mjtTimer.mjNTIMER)))
    begin = time.perf_counter()
    for state_index, state in enumerate(states):
        mujoco.mj_resetData(m, d)
        mujoco.mj_setState(m, d, state, STATE_SPEC)
        t = time.perf_counter()
        for _ in range(a.steps):
            mujoco.mj_step(m, d)
            counts.append(d.ncon)
            contacts.extend(np.concatenate(([c.dist], c.pos, c.frame.ravel(), c.geom)) for c in d.contact)
        mujoco.mj_forward(m, d)
        times.append(time.perf_counter() - t)
        timer_totals[int(state_index >= 320)] += d.timer.duration
        final = np.empty(mujoco.mj_stateSize(m, STATE_SPEC))
        mujoco.mj_getState(m, d, final, STATE_SPEC)
        endpoints.append(final)
    np.savez(a.output / "results.npz", counts=counts, contacts=np.array(contacts), endpoints=endpoints)
    report = dict(
        native_phase_seconds={
            group: {
                name: float(timer_totals[i, int(value)])
                for name, value in mujoco.mjtTimer.__members__.items()
                if name != "mjNTIMER"
            }
            for i, group in enumerate(["loose", "engaged"])
        },
        mode=a.mode,
        states=len(states),
        steps_per_state=a.steps,
        total_s=time.perf_counter() - begin,
        loose_s=sum(times[:320]),
        engaged_s=sum(times[320:]),
        times_s=times,
        labels=labels,
        mujoco=mujoco.__version__,
        plugin_sha256={
            p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in (ROOT / "plugin/build").glob("*.so")
        },
    )
    if a.profile:
        native.so101_surface_stats(stats, 0)
        report["native_metrics"] = dict(
            zip(
                [
                    "calls",
                    "native_s",
                    "supplement_s",
                    "nodes",
                    "sample_tests",
                    "aabb_rejects",
                    "grid_rejects",
                ],
                stats,
            )
        )
    (a.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps({k: v for k, v in report.items() if k not in ("labels", "times_s", "plugin_sha256")}),
        flush=True,
    )


if __name__ == "__main__":
    main()
