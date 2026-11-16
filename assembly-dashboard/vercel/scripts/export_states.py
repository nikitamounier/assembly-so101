"""Export read-only viewer snapshots; never step physics or touch training."""
from pathlib import Path
import collections
import gzip
import hashlib
import json
import os
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
WORKSPACE = ROOT.parent
sys.path.insert(0, str(WORKSPACE / "assembly-so101"))
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
import torch
torch.set_num_threads(1)
from assembly_policy.viewer.components import layout_collection
from assembly_policy.viewer.server import ResetApplication

BANK = WORKSPACE / "robotics-assembly-3.0/assets_local/wowrobo/randomization-bank-19200-chimera-display-v1"
OUT = ROOT / "vercel/public/data/states"
OUT.mkdir(parents=True, exist_ok=True)


def write(name, value):
    raw = json.dumps(value, separators=(",", ":"), allow_nan=False).encode()
    compressed = gzip.compress(raw, compresslevel=5, mtime=0)
    (OUT / name).write_bytes(compressed)
    return dict(file=name, bytes=len(compressed), sha256=hashlib.sha256(compressed).hexdigest())


started = time.time()
model, signature, bank = layout_collection(BANK)
options = {}
for key, name in [("point_config", "wowrobo_point_features.json"), ("reward_config", "wowrobo_reward_regions.json")]:
    options[key] = json.loads((BANK / name).read_text())
app = ResetApplication([bank], Path(sys.executable), 0, scene="wowrobo-layout", model=model, model_signature=signature, **options)
catalog = dict(banks=app.catalog.banks, entries=app.catalog.entries, studio_installed=False, scene=app.scene_name, default_bank=app.catalog.default_bank, hosted=True)
proof = dict(model_signature=signature, states=len(catalog["entries"]), physics_steps=0, files=[])
proof["files"].append(write("catalog.json.gz", catalog))
proof["files"].append(write("scene.json.gz", app.scene))
groups = collections.defaultdict(list)
for entry in catalog["entries"]:
    groups[entry["task"]].append(entry)
index = {}
for task, entries in groups.items():
    filename = hashlib.sha256(task.encode()).hexdigest()[:16] + ".json.gz"
    destination = OUT / filename
    if destination.exists():
        saved = json.loads(gzip.decompress(destination.read_bytes()))
        if set(saved) == {e["id"] for e in entries}:
            index.update({e["id"]: filename for e in entries})
            proof["files"].append(dict(file=filename, bytes=destination.stat().st_size, reused=True))
            continue
    snapshots = {}
    for entry in entries:
        snapshot = app.select(entry["id"])
        # This duplicate is reconstructed exactly by the hosted adapter.
        if snapshot.get("point_cloud", {}).get("available"):
            snapshot["point_cloud"].pop("flat", None)
        snapshots[entry["id"]] = snapshot
        index[entry["id"]] = filename
    proof["files"].append(write(filename, snapshots))
    print(json.dumps(dict(task=task, states=len(entries), elapsed=round(time.time()-started, 1), bytes=proof["files"][-1]["bytes"])), flush=True)
proof["files"].append(write("index.json.gz", index))
proof.update(elapsed_s=time.time()-started, complete=True)
(OUT / "export-proof.json").write_text(json.dumps(proof, indent=2)+"\n")
print(json.dumps(proof), flush=True)
app.temp.cleanup()
