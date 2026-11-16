"""Observational training rewards and bounded, actual successful episode traces."""

import json
import time
from pathlib import Path

import numpy as np

from .core import REWARD_CONTRACT_VERSION


class EpisodeRecorder:
    def __init__(self, run, n_envs, families, horizon=375, dt=0.04):
        self.run = Path(run)
        self.families = tuple(families)
        self.dt = dt
        self.frames = np.zeros((n_envs, horizon + 2, 26), np.float32)
        self.length = np.zeros(n_envs, int)
        self.returns = np.zeros(n_envs, float)
        self.complete = np.zeros(n_envs, bool)
        self.entry = np.full(n_envs, -1, int)
        self.totals = np.zeros((len(families), 3), float)
        self.reward_sum = np.zeros(len(families))
        self.reward_n = np.zeros(len(families), int)
        self.completed = [[] for _ in families]
        self.started = time.time()

    @staticmethod
    def frame(obs):
        p = obs["proprio"].reshape(len(obs["proprio"]), 5, 50)[:, -1, 26:38]
        return np.concatenate(
            (p, obs["part_p"][:, 0], obs["part_q"][:, 0], obs["part_p"][:, 1], obs["part_q"][:, 1]), axis=1
        )

    def begin(self, obs, entries):
        changed = (self.entry != entries) & (self.entry != -1)
        self.length[changed] = 0
        self.returns[changed] = 0
        self.complete[changed] = False
        self.entry[:] = entries
        ids = np.arange(len(entries))
        self.frames[ids, np.minimum(self.length, self.frames.shape[1] - 1)] = self.frame(obs)

    def step(self, reward, done, info, final_obs, family_ids, iteration):
        valid = np.asarray(info["valid"], bool)
        np.add.at(self.reward_sum, family_ids[valid], reward[valid])
        np.add.at(self.reward_n, family_ids[valid], 1)
        self.returns[valid] += reward[valid]
        self.length[valid] += 1
        lost = ~valid
        self.length[lost] = 0
        self.returns[lost] = 0
        self.complete[lost] = False
        ids = np.flatnonzero(done & valid)
        latest = None
        latest_by_family = {}
        if len(ids):
            final = self.frame(final_obs)
            for row, i in enumerate(ids):
                f = int(family_ids[i])
                full = bool(self.complete[i])
                if full:
                    value = float(self.returns[i])
                    success = bool(info["success"][i])
                    self.completed[f].append(value)
                    self.totals[f] += [1, int(success), value]
                    if success:
                        count = min(self.length[i], self.frames.shape[1] - 1)
                        self.frames[i, count] = final[row]
                        latest = dict(
                            id=f"{time.time_ns()}-{i}",
                            recorded_at=time.time(),
                            iteration=iteration,
                            family=self.families[f],
                            bank_entry=int(self.entry[i]),
                            episode_return=value,
                            duration_s=float(self.length[i] * self.dt),
                            dt=self.dt,
                            source="actual_training_episode",
                            reward_contract_version=REWARD_CONTRACT_VERSION,
                            complete_episode=True,
                            frames=self.frames[i, : count + 1].tolist(),
                        )
                        latest_by_family[self.families[f]] = latest
                self.length[i] = 0
                self.returns[i] = 0
                self.complete[i] = True
                # The vector environment has already selected the next entry.
                self.entry[i] = -1
            # begin() recognizes -1 as the expected next reset, preserving complete=True.
        for family, clip in latest_by_family.items():
            self.atomic(self.run / f"latest-success-{family}.json", clip)
        if latest is not None:
            self.atomic(self.run / "latest-success.json", latest)

    @staticmethod
    def atomic(path, value):
        temp = path.with_suffix(".json.tmp")
        temp.write_text(json.dumps(value, separators=(",", ":")) + "\n")
        temp.replace(path)

    def flush(self, iteration, transitions):
        metrics = dict(
            timestamp=time.time(),
            fam_reward={
                f: float(self.reward_sum[i] / self.reward_n[i]) if self.reward_n[i] else None
                for i, f in enumerate(self.families)
            },
            fam_return={
                f: float(np.mean(self.completed[i])) if self.completed[i] else None
                for i, f in enumerate(self.families)
            },
            fam_completed={f: len(self.completed[i]) for i, f in enumerate(self.families)},
        )
        self.atomic(
            self.run / "dashboard-summary.json",
            dict(
                iteration=iteration,
                transitions=transitions,
                recording_started=self.started,
                **metrics,
                families={
                    f: dict(episodes=int(t[0]), successes=int(t[1]), return_sum=float(t[2]))
                    for f, t in zip(self.families, self.totals)
                },
            ),
        )
        self.reward_sum[:] = 0
        self.reward_n[:] = 0
        self.completed = [[] for _ in self.families]
        return metrics
