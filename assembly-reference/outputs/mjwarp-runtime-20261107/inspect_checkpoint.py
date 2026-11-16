"""Verify saved experimental PPO state without executing physics."""

import hashlib
import json
import sys
from pathlib import Path

import torch

run = Path(sys.argv[1])
checkpoint = run / "checkpoint.pt"
data = torch.load(checkpoint, map_location="cpu", weights_only=False)


def finite(value):
    if isinstance(value, torch.Tensor):
        return bool(torch.isfinite(value).all())
    if isinstance(value, dict):
        return all(finite(v) for v in value.values())
    if isinstance(value, (tuple, list)):
        return all(finite(v) for v in value)
    return True


proof = {
    "iteration": data["it"],
    "transitions": data["transitions"],
    "physics_backend": data["args"]["physics_backend"],
    "physics_signature": data["args"]["physics_signature"],
    "status": data["args"]["run_status"],
    "model_finite": finite(data["model"]),
    "adam_finite": finite(data["opt"]),
    "adam_states": len(data["opt"]["state"]),
    "sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
    "warp_runtime": data["args"]["warp_runtime"],
}
assert proof["iteration"] == 1 and proof["transitions"] > 0
assert proof["model_finite"] and proof["adam_finite"] and proof["adam_states"] > 0
assert proof["physics_backend"] == "wowrobo-mjwarp-runtime-v1"
assert proof["status"] == "EXPERIMENTAL_WARP_UNQUALIFIED"
(run / "checkpoint-proof.json").write_text(json.dumps(proof, indent=2) + "\n")
print(json.dumps(proof, indent=2))
