"""Materialize the checksum-pinned helper assets and portable cube reset snapshot."""

import argparse
import hashlib
import json
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SNAPSHOT = ROOT / "runtime/cube-trial"


def checked_copy(source, target, expected):
    raw = source.read_bytes()
    if hashlib.sha256(raw).hexdigest() != expected:
        raise ValueError(f"Checksum mismatch: {source.name}")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(raw)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--asset-source", type=Path, help="Use a matching local assets_local/so101 cache")
    args = parser.parse_args()
    lock = json.loads((SNAPSHOT / "snapshot.json").read_text())
    assets = ROOT / "assets_local/so101"
    for name, record in lock["robot_files"].items():
        target = assets / name
        if target.exists() and hashlib.sha256(target.read_bytes()).hexdigest() == record["sha256"]:
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        if args.asset_source:
            checked_copy(args.asset_source / name, target, record["sha256"])
        else:
            # Pinned XML/STL data only; no downloaded source code is executed.
            with urllib.request.urlopen(lock["robot_url"] + name, timeout=60) as response:
                raw = response.read()
            if hashlib.sha256(raw).hexdigest() != record["sha256"]:
                raise ValueError(f"Upstream asset checksum mismatch: {name}")
            target.write_bytes(raw)
    (assets / "revision.txt").write_text(lock["robot_revision"] + "\n")
    from assembly_policy.cube_trial.core import signature as local_signature
    from assembly_policy.cube_trial.resets import check_bank, signature

    if signature() != lock["portable_physics_signature"]:
        raise ValueError("Pinned cube snapshot needs MuJoCo 4.0.0 and the matching native model")
    bank = ROOT / "data/cube_trial" / lock["bank"]
    bank.mkdir(parents=True, exist_ok=True)
    for name, digest in lock["bank_files"].items():
        checked_copy(SNAPSHOT / lock["bank"] / name, bank / name, digest)
    # Viewer signatures contain absolute local mesh paths. Only this ignored
    # manifest is rebound; checked-in state arrays/context remain byte exact.
    manifest = json.loads((bank / "manifest.json").read_text())
    manifest["model_signature"] = local_signature()
    (bank / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    check_bank(bank)
    print(f"Prepared {lock['train_states']} TRAIN / {lock['eval_states']} EVAL states: {bank}")


if __name__ == "__main__":
    main()
