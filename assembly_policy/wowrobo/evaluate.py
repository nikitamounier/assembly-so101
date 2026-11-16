"""Evaluate a current-assembly checkpoint on independent task evaluation resets."""

import argparse
import dataclasses
import json
from pathlib import Path

import numpy as np
import torch

from ..rl.nets import ActorCritic
from ..rl.vecenv import LocalVecEnv
from .bank import check_bank, directory_signature
from .env import ACT_DIM, HISTORY, PROPRIO_DIM, EnvConfig, resolve_tasks
from .obs import PointTables, require_observation_signature


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--policy", required=True)
    parser.add_argument("--tasks", default="all")
    parser.add_argument("--bank-dir", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--nthread", type=int, default=1)
    args = parser.parse_args()
    checkpoint = torch.load(args.policy, map_location="cpu", weights_only=False)
    config = checkpoint["args"]
    if config["suite"] != "wowrobo":
        raise ValueError("Expected current-assembly checkpoint")
    metadata = check_bank(args.bank_dir, allow_debug=config.get("wowrobo_debug_resets", False))
    if (
        metadata["model_signature"] != config["model_signature"]
        or directory_signature(args.bank_dir) != config["bank_directory_signature"]
    ):
        raise ValueError("Evaluation model/bank differs from trained identity")
    tasks = resolve_tasks(args.tasks)
    if tasks != resolve_tasks(config["tasks"]):
        raise ValueError("Evaluation task order differs from policy")
    env_config = EnvConfig(**{**config["env"], "nthread": args.nthread})
    tables = PointTables(tasks, review_model=metadata["review_model"])
    require_observation_signature(config, tables)
    policy = ActorCritic(
        tables.n_pts,
        proprio_dim=HISTORY * PROPRIO_DIM,
        act_dim=ACT_DIM,
        init_std=config["init_std"],
        max_std=config["max_std"],
        min_std=config["min_std"],
    )
    policy.load_state_dict(checkpoint["model"])
    policy.eval()
    states = []
    task_ids = []
    for index, task in enumerate(tasks):
        path = Path(args.bank_dir) / f"{task}_eval.npz"
        if not path.exists():
            raise ValueError(f"Independent evaluation reset missing for {task}")
        with np.load(path, allow_pickle=False) as archive:
            states.extend(archive["states"])
            task_ids.extend([index] * len(archive["states"]))
    states = np.asarray(states)
    task_ids = np.asarray(task_ids)
    n = len(states)
    cursor = 0

    def provider(k, *_):
        nonlocal cursor
        indices = (np.arange(k) + cursor) % n
        cursor += k
        return indices, states[indices], task_ids[indices]

    env = LocalVecEnv(
        tasks,
        n,
        env_config,
        provider,
        env_type=__import__(
            "assembly_policy.wowrobo.env", fromlist=["BatchedAssemblyEnv"]
        ).BatchedAssemblyEnv,
    )
    finished = np.zeros(n, bool)
    results = []
    try:
        for step in range(env.horizon):
            observation = env.observe()
            compact = {
                key: torch.as_tensor(observation[key], dtype=torch.long if key == "task" else torch.float32)
                for key in ("task", "tcp_p", "tcp_R", "part_p", "part_q", "finger")
            }
            with torch.no_grad():
                points = tables(**compact)
                actions = policy.dist(points, torch.as_tensor(observation["proprio"])).mean.numpy()
            _, done, info, _, _ = env.step(actions)
            for index in np.flatnonzero(done & ~finished):
                results.append(
                    dict(
                        task=tasks[task_ids[index]],
                        success=bool(info["success"][index]),
                        safety=bool(info["safety"][index]),
                        timeout=bool(info["timeout"][index]),
                        length=step + 1,
                    )
                )
            finished |= done
            if finished.all():
                break
    finally:
        env.close()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(
            dict(
                status=config["run_status"],
                model_signature=metadata["model_signature"],
                episodes=results,
                env=dataclasses.asdict(env_config),
            ),
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
