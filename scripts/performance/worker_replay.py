"""Fixed-action production-sized worker benchmark, including observations/reset handling."""

import argparse
import cProfile
import hashlib
import json
import time
from pathlib import Path

import numpy as np

from assembly_policy.so101.env import BatchedAssemblyEnv, EnvConfig
from assembly_policy.so101.scene import ASSEMBLY_TASKS, model
from assembly_policy.so101.workcell import default_bank_directory


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--profile", action="store_true")
    p.add_argument("--bank-dir", type=Path, default=default_bank_directory())
    p.add_argument(
        "--jacobian", type=int, choices=[0, 1, 2], help="Override configured Jacobian:0dense,1sparse,2auto"
    )
    a = p.parse_args()
    a.output.mkdir(exist_ok=False, parents=True)
    tasks = list(ASSEMBLY_TASKS)
    task_ids = np.arange(40) % len(tasks)
    bank = {task: np.load(a.bank_dir / f"{task}_train.npz")["states"] for task in tasks}
    states = np.stack([bank[tasks[t]][i // 5] for i, t in enumerate(task_ids)])

    def factory(**kwargs):
        m = model(**kwargs)
        if a.jacobian is not None:
            m.opt.jacobian = a.jacobian
        return m

    env = BatchedAssemblyEnv(
        tasks,
        task_ids,
        EnvConfig(nthread=2, max_contact_depth=0.0004, safety_penalty=1),
        model_factory=factory,
    )
    profiler = cProfile.Profile()
    results = []
    iterations = []
    try:
        env.set_states(np.arange(40), states, task_ids)
        action = np.random.default_rng(20261107).uniform(-0.12, 0.12, (32, 40, 12))
        if a.profile:
            profiler.enable()
        start = time.perf_counter()
        for controls in action:
            reward, done, info = env.step(controls)
            iterations.extend(int(d.solver_niter.max()) for d in env.work)
            results.append(
                np.r_[env.state.ravel(), reward, done, info["safety"], env.observe()["proprio"].ravel()]
            )
            ids = np.flatnonzero(done)
            if len(ids):
                env.set_states(ids, states[ids], task_ids[ids])
        elapsed = time.perf_counter() - start
        if a.profile:
            profiler.disable()
            profiler.dump_stats(a.output / "profile.pstats")
        np.save(a.output / "results.npy", np.stack(results))
        report = dict(
            bank_directory=str(a.bank_dir),
            solver_iterations=iterations,
            jacobian=a.jacobian,
            effective_jacobian=int(env.m.opt.jacobian),
            seconds=elapsed,
            envs=40,
            control_steps=32,
            physics_steps=40 * 32 * 80,
            sha256=hashlib.sha256(np.stack(results).tobytes()).hexdigest(),
        )
        (a.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report), flush=True)
    finally:
        env.close()


if __name__ == "__main__":
    main()
