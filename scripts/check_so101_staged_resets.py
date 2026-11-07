"""Audit saved staged states, prerequisite geometry, and native replay receipts."""

import argparse
import json
from pathlib import Path

import mujoco
import numpy as np

from assembly_policy.so101.env import STATE_SPEC
from assembly_policy.so101.fastener import origins
from assembly_policy.so101.goals import placement_errors
from assembly_policy.so101.scene import ASSEMBLY_TASKS, metadata, model
from assembly_policy.so101.subtask_curriculum import validate_row
from assembly_policy.so101.workcell import environment_identity


def audit(directory):
    manifest = json.loads((directory / "manifest.json").read_text())
    if manifest["reset_distribution"] != "subtask_loose":
        raise ValueError("Expected a staged loose collection")
    plan = manifest["generation_plan"]
    if plan["environment"] != environment_identity():
        raise ValueError("Staged collection belongs to a different environment")
    m = model()
    d = mujoco.MjData(m)
    meta = metadata()
    targets = origins(meta)
    count = 0
    maximum = 0.0
    groups = {}
    for task in ASSEMBLY_TASKS:
        records = json.loads((directory / f"{task}_provenance.json").read_text())
        for split in ("train", "eval"):
            rows = [row for row in records if row["split"] == split]
            with np.load(directory / f"{task}_{split}.npz", allow_pickle=False) as archive:
                if len(rows) != len(archive["states"]) or len(rows) != plan[split + "_count"]:
                    raise ValueError("Incomplete staged cohort")
                for i, row in enumerate(rows):
                    validate_row(row, task, "loose", plan)
                    np.testing.assert_array_equal(archive["states"][i], row["state"])
                    np.testing.assert_array_equal(archive["controls"][i], row["controls"])
                    assert archive["seed"][i] == row["seed"] and archive["family"][i] == 0
                    mujoco.mj_resetData(m, d)
                    mujoco.mj_setState(m, d, archive["states"][i], STATE_SPEC)
                    mujoco.mj_kinematics(m, d)
                    errors = placement_errors(d, task, meta, targets)
                    error = max((errors[name] for name in row["predecessors"]), default=0.0)
                    if error >= 0.001:
                        raise ValueError(f"{task}/{split}/{i}: prerequisite geometry differs by {error:g} m")
                    maximum = max(maximum, float(error))
                    count += 1
            groups[f"{task}/{split}"] = len(rows)
    return dict(
        passed=True,
        states=count,
        groups=groups,
        max_initial_predecessor_error_m=maximum,
        model_signature=manifest["model_signature"],
        mujoco_version=mujoco.__version__,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    result = audit(args.directory)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
