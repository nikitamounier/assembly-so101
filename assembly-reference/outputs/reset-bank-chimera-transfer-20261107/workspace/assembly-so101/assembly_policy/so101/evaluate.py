"""Evaluate SO101 checkpoints on the independent loose reset bank."""

import argparse
import json
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from queue import Queue

import numpy as np
import torch

from ..rl.nets import ActorCritic
from .bank_paths import default_training_bank_directory
from .curriculum import FAMILIES
from .env import HISTORY, PROPRIO_DIM, BatchedAssemblyEnv, EnvConfig, resolve_tasks
from .obs import PointTables, require_observation_signature
from .resets import check_bank, directory_signature, pooled_signature, signature
from .workcell import configuration


def evaluation_indices(families, requested, trials):
    if trials < 1 or not requested or len(set(requested)) != len(requested) or set(requested) - set(FAMILIES):
        raise ValueError("Invalid evaluation family selection")
    indices = []
    for family in requested:
        selected = np.flatnonzero(families == FAMILIES.index(family))[:trials]
        if not len(selected):
            raise ValueError(f"Missing evaluation family: {family}")
        indices.extend(selected.tolist())
    return indices


def episode_action(dist, mode, generator):
    """Match Normal.sample with an episode-private CPU generator."""
    return dist.mean if mode == "mean" else torch.normal(dist.mean, dist.stddev, generator=generator)


def evaluate_episode(env, net, pts, state, task_index, index, family, seed, action_mode):
    generator = torch.Generator(device="cpu").manual_seed(seed + 10000 * task_index + index)
    env.set_states([0], state[None], [task_index])
    initial = float(env.assembly_distance()[0])
    best = initial
    for _ in range(env.horizon):
        o = {k: torch.as_tensor(v) for k, v in env.observe().items()}
        with torch.no_grad():
            pc = pts(*[o[k] for k in ("task", "tcp_p", "tcp_R", "part_p", "part_q", "finger")])
            action = episode_action(net.dist(pc, o["proprio"]), action_mode, generator).numpy()
        _, done, info = env.step(action)
        best = min(best, float(info["d_asm"][0]))
        if done[0]:
            break
    return dict(
        {k: bool(info[k][0]) for k in ("success", "safety", "timeout")},
        entry=index,
        family=family,
        initial_error_m=initial,
        best_error_m=best,
        final_error_m=float(info["d_asm"][0]),
    )


def evaluate_jobs(jobs, envs, net, pts, seed, action_mode):
    """Lease private simulation data and preserve submitted task/entry order."""
    available = Queue()
    for env in envs:
        available.put(env)
    started = time.monotonic()

    def run(job):
        task, task_index, index, family, state = job
        env = available.get()
        episode_started = time.monotonic()
        try:
            record = evaluate_episode(env, net, pts, state, task_index, index, family, seed, action_mode)
            print(
                f"[evaluate] task={task} entry={index} family={family} "
                f"success={record['success']} elapsed_s={time.monotonic() - episode_started:.3f} "
                f"total_elapsed_s={time.monotonic() - started:.3f}",
                flush=True,
            )
            return task, record
        finally:
            available.put(env)

    with ThreadPoolExecutor(max_workers=len(envs)) as executor:
        return list(executor.map(run, jobs))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", required=True)
    ap.add_argument("--tasks", default="all")
    ap.add_argument("--bank-dir", default=str(default_training_bank_directory()))
    ap.add_argument("--out", required=True)
    ap.add_argument(
        "--nthread", type=int, default=1, help="Compatibility option; each episode uses one physics thread"
    )
    ap.add_argument("--trials", type=int, default=8)
    ap.add_argument(
        "--families", default="loose", help="loose, all, or + separated families; trials per family"
    )
    ap.add_argument("--action-mode", choices=["mean", "sample"], default="sample")
    ap.add_argument("--seed", type=int, default=501)
    ap.add_argument(
        "--workers", type=int, default=1, help="Independent episode workers sharing one read-only model"
    )
    a = ap.parse_args()
    selected_families = list(FAMILIES) if a.families == "all" else a.families.split("+")
    if not selected_families or set(selected_families) - set(FAMILIES):
        raise ValueError("Unknown evaluation family")
    if a.trials < 1 or a.nthread < 1 or a.workers < 1:
        raise ValueError("Positive trials, threads and workers required")
    check_bank(a.bank_dir)
    ck = torch.load(a.policy, map_location="cpu", weights_only=False)
    cfg = ck["args"]
    if cfg.get("model_signature") != signature():
        raise ValueError("Checkpoint and evaluation model differ")
    trained = resolve_tasks(cfg["tasks"])
    tasks = resolve_tasks(a.tasks)
    from ..train import load_banks

    if cfg.get("bank_signature") != pooled_signature(load_banks(trained, a.bank_dir, cfg["bank_per_family"])):
        raise ValueError("Checkpoint training bank differs; use its recorded bank directory")
    if cfg.get("bank_directory_signature") != directory_signature(a.bank_dir):
        raise ValueError(
            "Checkpoint evaluation bank differs or lacks a frozen evaluation identity; use a new qualified run"
        )
    if not set(tasks) <= set(trained):
        raise ValueError("Evaluation task was not trained")
    pts = PointTables(trained, cfg["n_part"], cfg["n_fix"], cfg["n_grip"])
    require_observation_signature(cfg, pts)
    net = ActorCritic(pts.n_pts, HISTORY * PROPRIO_DIM, 12)
    net.load_state_dict(ck["model"])
    net.eval()
    result = {
        "policy": str(Path(a.policy).resolve()),
        "distribution": json.loads((Path(a.bank_dir) / "manifest.json").read_text()).get(
            "evaluation_distribution", "independent full-sequence loose table"
        ),
        "action_mode": a.action_mode,
        "seed": a.seed,
        "families": selected_families,
        "evaluation_backend": f"native {configuration()['profile']} geometry",
        "training_backend": cfg.get("physics_backend", "native"),
        "model_signature": signature(),
        "bank_directory_signature": directory_signature(a.bank_dir),
        "tasks": {},
    }
    envs = []
    try:
        # Install callbacks and construct shared model metadata serially.
        ecfg = EnvConfig(**{**cfg["env"], "nthread": 1})
        envs.append(BatchedAssemblyEnv(trained, [0], ecfg))
        shared_model = envs[0].m
        for _ in range(a.workers - 1):
            envs.append(BatchedAssemblyEnv(trained, [0], ecfg, model_factory=lambda **_: shared_model))
        jobs = []
        for task in tasks:
            directory = (
                Path(a.bank_dir) if selected_families == ["loose"] else Path(a.bank_dir) / "family_eval"
            )
            with np.load(directory / f"{task}_eval.npz") as bank:
                for index in evaluation_indices(bank["family"], selected_families, a.trials):
                    jobs.append(
                        (
                            task,
                            trained.index(task),
                            index,
                            FAMILIES[int(bank["family"][index])],
                            bank["states"][index].copy(),
                        )
                    )
            result["tasks"][task] = {"trials": 0, "successes": 0, "episodes": []}
        for task, record in evaluate_jobs(jobs, envs, net, pts, a.seed, a.action_mode):
            result["tasks"][task]["episodes"].append(record)
        for task in tasks:
            records = result["tasks"][task]["episodes"]
            result["tasks"][task].update(trials=len(records), successes=sum(x["success"] for x in records))
    finally:
        for env in envs:
            env.close()
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
