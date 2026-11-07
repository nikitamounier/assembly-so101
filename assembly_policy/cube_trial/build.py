"""Prepare reproducible cube-handoff reset banks for the shared reset viewer."""

import argparse
import importlib
import json
from pathlib import Path

import mujoco
import numpy as np

FAMILIES = (
    "loose",
    "left_pickup",
    "converging_transport",
    "pre_handoff",
    "dual_grasp",
    "post_handoff",
    "right_transport",
    "near_target",
    "placed_held",
)


def main():
    from .core import generate, model, signature

    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=int, default=128)
    parser.add_argument("--eval", type=int, default=32)
    parser.add_argument("--out", type=Path, default=Path("data/cube_trial/banks_v1"))
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    m = model()
    spec = mujoco.mjtState.mjSTATE_INTEGRATION
    labels = [importlib.import_module(f".families.{f}", __package__).CONFIG["label"] for f in FAMILIES]
    receipts = []
    for split, count, base in (("train", args.train, 100000), ("eval", args.eval, 900000)):
        states, seeds, families = [], [], []
        for index, family in enumerate(FAMILIES):
            for row in range(count):
                seed = base + index * 10000 + row
                data, metadata = generate(family, seed, model=m)
                state = np.empty(mujoco.mj_stateSize(m, spec))
                mujoco.mj_getState(m, data, state, spec)
                if not np.isfinite(state).all():
                    raise ValueError(f"Nonfinite reset {family}/{seed}")
                states.append(state)
                seeds.append(seed)
                families.append(index)
                receipts.append(dict(split=split, row=len(states) - 1, **metadata))
            print(f"{split}: {family}: {count} states", flush=True)
        np.savez_compressed(
            args.out / f"cube_handoff_{split}.npz",
            states=np.asarray(states),
            seed=np.asarray(seeds),
            family=np.asarray(families),
        )
    manifest = dict(
        model_signature=signature(),
        task_names=["cube_handoff"],
        family_names=labels,
        state_spec="mjSTATE_INTEGRATION",
        reset_distribution="cube-handoff-nine-families",
        train_per_family=args.train,
        eval_per_family=args.eval,
        mujoco_version=mujoco.__version__,
        custom_collision=False,
    )
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    (args.out / "qualification.json").write_text(json.dumps(receipts, indent=2))


if __name__ == "__main__":
    main()
