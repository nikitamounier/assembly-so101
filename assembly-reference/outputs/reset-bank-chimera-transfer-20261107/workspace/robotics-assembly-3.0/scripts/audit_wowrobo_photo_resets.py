"""Audit frozen review snapshots; write separate sampled-boundary receipts."""

import argparse
import hashlib
import json
import time
from pathlib import Path

import mujoco
import numpy as np

from assembly_policy.so101.wowrobo_boundaries import NativeBoundaryAudit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--folder", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--states", nargs="*")
    parser.add_argument("--max-samples", type=int, default=512)
    args = parser.parse_args()
    source = args.folder.resolve()
    output = args.output.resolve()
    if output == source or source in output.parents:
        parser.error("Write audit receipts outside the frozen source bank")
    xml = source / "scene.xml"
    model = mujoco.MjModel.from_xml_path(str(xml))
    data = mujoco.MjData(model)
    auditor = NativeBoundaryAudit(model, max_samples=args.max_samples)
    bank = source / "stages" if (source / "stages").is_dir() else source
    manifest_path = bank / "manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    spec = manifest.get("state_spec", int(mujoco.mjtState.mjSTATE_FULLPHYSICS))
    files = sorted(bank.glob("*.npz"))
    output.mkdir(parents=True, exist_ok=True)
    summary = {
        "source": str(source),
        "scene_sha256": hashlib.sha256(xml.read_bytes()).hexdigest(),
        "state_spec": spec,
        "training_eligible": False,
        "stages": [],
        "limitations": "Sampled triangle checks only; these receipts never qualify a training reset.",
    }
    for path in files:
        stage = path.stem.removesuffix("_train")
        if args.states and stage not in args.states:
            continue
        with np.load(path, allow_pickle=False) as saved:
            states = saved["states"]
            for index, state in enumerate(states):
                started = time.monotonic()
                mujoco.mj_resetData(model, data)
                mujoco.mj_setState(model, data, state, spec)
                if "controls" in saved:
                    data.ctrl[:] = saved["controls"][index]
                mujoco.mj_forward(model, data)
                result = auditor.audit(data)
                result.update(
                    stage=stage,
                    row=index,
                    state_sha256=hashlib.sha256(state.tobytes()).hexdigest(),
                    scene_sha256=summary["scene_sha256"],
                    elapsed_seconds=time.monotonic() - started,
                )
                if "controls" in saved:
                    result["control_sha256"] = hashlib.sha256(
                        saved["controls"][index].tobytes()
                    ).hexdigest()
                filename = f"{stage}-{index:03d}.json"
                (output / filename).write_text(json.dumps(result, indent=2) + "\n")
                compact = {
                    k: result[k]
                    for k in (
                        "stage",
                        "row",
                        "candidate_rejected",
                        "max_sampled_penetration_m",
                        "table_minimum",
                        "elapsed_seconds",
                        "state_sha256",
                    )
                }
                compact["receipt"] = filename
                summary["stages"].append(compact)
                summary["candidate_rejected"] = any(
                    s["candidate_rejected"] for s in summary["stages"]
                )
                summary["count"] = len(summary["stages"])
                summary["rejected_count"] = sum(
                    s["candidate_rejected"] for s in summary["stages"]
                )
                (output / "summary.json").write_text(
                    json.dumps(summary, indent=2) + "\n"
                )
                print(
                    stage,
                    index,
                    "REJECT" if result["candidate_rejected"] else "sampled checks pass",
                    f"max={result['max_sampled_penetration_m'] * 1000:.3f}mm",
                    f"{result['elapsed_seconds']:.1f}s",
                    flush=True,
                )


if __name__ == "__main__":
    main()
