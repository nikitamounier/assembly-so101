"""Attach state-matched sampled audit summaries to a review viewer manifest."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--folder", required=True, type=Path)
    parser.add_argument("--audit", required=True, type=Path)
    args = parser.parse_args()
    manifest_path = args.folder / "stages/manifest.json"
    manifest = json.loads(manifest_path.read_text())
    scene_hash = hashlib.sha256((args.folder / "scene.xml").read_bytes()).hexdigest()
    for task, definition in manifest["stage_by_task"].items():
        definition["boundary_review"] = "Sampled boundary audit pending"
        receipt_path = args.audit / (task + ".json")
        if not receipt_path.exists():
            receipt_path = args.audit / (task + "-000.json")
        if not receipt_path.exists():
            continue
        receipt = json.loads(receipt_path.read_text())
        with np.load(args.folder / "stages" / (task + "_train.npz")) as saved:
            state_hash = hashlib.sha256(saved["states"][0].tobytes()).hexdigest()
            control_hash = hashlib.sha256(saved["controls"][0].tobytes()).hexdigest()
        if (
            receipt.get("scene_sha256") != scene_hash
            or receipt.get("state_sha256") != state_hash
            or receipt.get("control_sha256") != control_hash
        ):
            definition["boundary_review"] = "Pose changed · sampled audit pending"
            continue
        if receipt["candidate_rejected"]:
            depth_mm = receipt["max_sampled_penetration_m"] * 1000
            if receipt.get("candidate_rejection_reasons") == ["target_target"]:
                definition["boundary_review"] = (
                    f"Helper boundaries pass · part fit rejected ({depth_mm:.2f} mm overlap)"
                )
            else:
                definition["boundary_review"] = (
                    f"Rejected by sampled boundary audit · max overlap {depth_mm:.2f} mm"
                )
        else:
            definition["boundary_review"] = (
                "Sampled boundaries pass · grasp forces and dynamics remain unqualified"
            )
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
