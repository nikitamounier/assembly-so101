"""Allocated stock-physics and real CUDA PPO/checkpoint qualification."""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import mujoco
import numpy as np
import torch

from assembly_policy.cube_trial.core import model
from assembly_policy.cube_trial.resets import check_bank, signature

ROOT = Path(__file__).resolve().parents[2]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", action="store_true")
    args = parser.parse_args()
    assert os.environ.get("SLURM_JOB_ID"), "Remote qualification must be allocated"
    m = model()
    assert mujoco.__version__ == "4.0.0" and m.nplugin == 0 and m.neq == 0
    assert not np.any(m.geom_type == mujoco.mjtGeom.mjGEOM_SDF)
    assert mujoco.get_mjcb_contactfilter() is None
    maps = Path("/proc/self/maps").read_text()
    assert not any(name in maps for name in ("libassemblenist", "libso101_contact", "libso101_screw"))
    assert not (ROOT / "plugin").exists() and not (ROOT / "assets_local/sdf").exists()
    bank = ROOT / "data/cube_trial/banks_pre_handoff_clear_v6"
    check_bank(bank)
    report = dict(
        mujoco=mujoco.__version__,
        torch=torch.__version__,
        model_signature=signature(),
        plugins=m.nplugin,
        equalities=m.neq,
        sdf_geoms=0,
        custom_collision_libraries=False,
        contact_filter=False,
        allocated_job=os.environ["SLURM_JOB_ID"],
    )
    if args.gpu:
        assert torch.cuda.is_available()
        run = ROOT / "artifacts_local/chimera/cuda-smoke-touchdown-cost-p256-c128"
        command = [
            sys.executable,
            "-m",
            "assembly_policy.cube_trial.train",
            "--run-dir",
            str(run),
            "--bank-dir",
            str(bank),
            "--envs",
            "8",
            "--n-part",
            "256",
            "--n-fix",
            "128",
            "--steps",
            "8",
            "--iters",
            "2",
            "--epochs",
            "1",
            "--minibatches",
            "1",
            "--device",
            "cuda",
            "--nthread",
            "2",
            "--no-eval",
            "--seed",
            "20261107",
        ]
        subprocess.run(command, check=True, cwd=ROOT)
        first = torch.load(run / "checkpoint.pt", map_location="cpu", weights_only=False)
        command[command.index("--iters") + 1] = "3"
        subprocess.run(command, check=True, cwd=ROOT)
        final = torch.load(run / "checkpoint.pt", map_location="cpu", weights_only=False)
        assert first["it"] == 2 and final["it"] == 3 and final["transitions"] > first["transitions"]
        assert all(torch.isfinite(t).all() for t in final["model"].values() if torch.is_tensor(t))
        assert any(not torch.equal(t, final["model"][k]) for k, t in first["model"].items() if "actor" in k)
        report.update(
            cuda_device=torch.cuda.get_device_name(0),
            actual_cuda_updates=3,
            checkpoint_resume=True,
            transitions=final["transitions"],
            sparse_success_with_control_cost=True,
            cost_coef=2e-5,
            touchdown_requires_right_grasp=True,
        )
    path = ROOT / "artifacts_local/chimera" / ("READY.json" if args.gpu else "cpu-preflight.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
