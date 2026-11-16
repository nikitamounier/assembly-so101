"""Combine selected stage-shard draws with the latest approved reference viewer."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from assembly_policy.so101.wowrobo_bank import export_review, write_json


def publish(source, shard_root, output, stages):
    if output.exists():
        raise ValueError("Choose a new versioned preview directory")
    manifest = json.loads((source / "stages/manifest.json").read_text())
    unknown = set(stages) - set(manifest["task_names"])
    if unknown:
        raise ValueError(f"Unknown stages: {sorted(unknown)}")
    rows = []
    for stage in stages:
        run = json.loads((shard_root / stage / "run.json").read_text())
        if run["status"] not in ("pilot_complete", "complete", "incomplete_rejections"):
            raise ValueError(f"Stage still generating: {stage}")
        for path in sorted((shard_root / stage / "attempts").glob("*.json")):
            row = json.loads(path.read_text())
            rows.append(row)
    if not rows:
        raise ValueError("No generated variations")
    shutil.copytree(source, output)
    for row in rows:
        stage = row["stage"]
        old_archive = row["archive"]
        old_receipt = row["receipt"]
        row["archive"] = f"variation-inputs/{stage}/{Path(old_archive).name}"
        row["receipt"] = f"variation-inputs/{stage}/{Path(old_receipt).name}"
        destination = output / row["archive"]
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(shard_root / stage / old_archive, destination)
        write_json(output / row["receipt"], row)
    by_stage = {}
    for stage in manifest["task_names"]:
        selected = [row for row in rows if row["stage"] == stage]
        by_stage[stage] = {
            "generated_count": len(selected),
            "accepted_train": sum(
                r["training_eligible"] and r["split"] == "train" for r in selected
            ),
            "accepted_eval": sum(
                r["training_eligible"] and r["split"] == "eval" for r in selected
            ),
            "rejected_count": sum(r["candidate_rejected"] for r in selected),
            "pending_count": sum(
                not r["training_eligible"] and not r["candidate_rejected"]
                for r in selected
            ),
        }
    progress = {
        "status": "cover_preview_ready",
        "by_stage": by_stage,
        "preview_only": True,
        "complete": False,
        "requested_component": "Base side cover · Left arm",
    }
    for key in (
        "generated_count",
        "accepted_train",
        "accepted_eval",
        "rejected_count",
        "pending_count",
    ):
        progress[key] = sum(value[key] for value in by_stage.values())
    # Rejected draws remain in archived receipts/progress for diagnosis, but
    # the user's review selector contains physically admitted variations only.
    admitted = [row for row in rows if row["training_eligible"]]
    missing = set(stages) - {row["stage"] for row in admitted}
    if missing:
        raise ValueError(
            f"No admitted variation for requested start: {sorted(missing)}"
        )
    export_review(output, source, manifest["task_names"], admitted, progress)
    index = json.loads((output / "stages/manifest.json").read_text())
    index["display_name"] = "Base side cover · variation preview"
    write_json(output / "stages/manifest.json", index)
    return progress


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--shards", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--stages", nargs="+", required=True)
    a = p.parse_args()
    print(json.dumps(publish(a.source, a.shards, a.output, a.stages), indent=2))


if __name__ == "__main__":
    main()
