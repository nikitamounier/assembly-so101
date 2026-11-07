"""Lightweight evaluator parity checks; no native simulation is constructed."""

import time

import numpy as np
import pytest
import torch

from assembly_policy.so101.evaluate import episode_action, evaluate_jobs


@pytest.mark.parametrize("task,index", [(0, 0), (2, 3), (4, 19)])
def test_private_generator_matches_original_normal_sample(task, index):
    seed = 501 + 10000 * task + index
    generator = torch.Generator().manual_seed(seed)
    original_rng = torch.random.get_rng_state()
    try:
        torch.manual_seed(seed)
        for step in range(7):
            mean = torch.arange(12, dtype=torch.float32).reshape(1, 12) * 0.01 + step
            dist = torch.distributions.Normal(mean, torch.full_like(mean, 0.37))
            assert torch.equal(dist.sample(), episode_action(dist, "sample", generator))
    finally:
        torch.random.set_rng_state(original_rng)


class FakeEnv:
    horizon = 4

    def set_states(self, ids, states, tasks):
        self.position = float(states[0, 0])
        self.delay = float(states[0, 1])
        self.steps = 0
        self.task = tasks[0]

    def assembly_distance(self):
        return np.array([abs(self.position)])

    def observe(self):
        return {
            key: np.array([[self.position]], dtype=np.float32)
            for key in ("task", "tcp_p", "tcp_R", "part_p", "part_q", "finger", "proprio")
        }

    def step(self, action):
        time.sleep(self.delay)
        self.position += float(action[0, 0]) * 0.01
        self.steps += 1
        done = self.steps == self.horizon
        info = dict(
            success=np.array([self.position < 0]),
            safety=np.array([False]),
            timeout=np.array([done]),
            d_asm=self.assembly_distance(),
        )
        return None, np.array([done]), info


class FakeNet:
    def dist(self, pc, proprio):
        return torch.distributions.Normal(proprio * 0.1, torch.ones_like(proprio) * 0.3)


def fake_points(*args):
    return args[0]


@pytest.mark.parametrize("mode", ["sample", "mean"])
def test_parallel_schedule_preserves_episode_outputs_order_and_global_rng(mode, capsys):
    # First episode finishes last in parallel; jobs also vary seed/task identity.
    jobs = [
        ("motor", 0, 3, "loose", np.array([0.3, 0.005])),
        ("motor", 0, 1, "partial", np.array([0.1, 0.0])),
        ("horn", 1, 2, "loose", np.array([-0.2, 0.0])),
    ]
    rng_before = torch.random.get_rng_state().clone()
    serial = evaluate_jobs(jobs, [FakeEnv()], FakeNet(), fake_points, 501, mode)
    parallel = evaluate_jobs(jobs, [FakeEnv() for _ in range(3)], FakeNet(), fake_points, 501, mode)
    assert serial == parallel
    assert [(task, row["entry"]) for task, row in parallel] == [("motor", 3), ("motor", 1), ("horn", 2)]
    assert torch.equal(rng_before, torch.random.get_rng_state())
    assert all(row["timeout"] for _, row in parallel)
    assert all(
        set(row)
        == {
            "success",
            "safety",
            "timeout",
            "entry",
            "family",
            "initial_error_m",
            "best_error_m",
            "final_error_m",
        }
        for _, row in parallel
    )
    output = capsys.readouterr().out.splitlines()
    assert len(output) == 6
    assert all("elapsed_s=" in line and "total_elapsed_s=" in line for line in output)
    assert "entry=3" not in output[3]  # Slow first submission did not finish first.
