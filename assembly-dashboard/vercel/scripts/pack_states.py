"""Pack sixteen exact snapshots per gzip block for fast HTTP Range reads."""
from pathlib import Path
import gzip
import hashlib
import json

base = Path(__file__).resolve().parents[1] / "public/data/states"
index = json.loads(gzip.decompress((base / "index.json.gz").read_bytes()))
packed_index = {}
for name in sorted(set(index.values())):
    snapshots = json.loads(gzip.decompress((base / name).read_bytes()))
    packed_name = name.replace(".json.gz", ".pack")
    entries = list(snapshots.items())
    with (base / packed_name).open("wb") as out:
        for offset in range(0, len(entries), 16):
            group = dict(entries[offset:offset+16])
            data = gzip.compress(json.dumps(group, separators=(",", ":"), allow_nan=False).encode(), compresslevel=5, mtime=0)
            start = out.tell()
            out.write(data)
            for key in group:
                packed_index[key] = [packed_name, start, len(data)]
            assert json.loads(gzip.decompress(data)) == group
    print(packed_name, len(entries), flush=True)
assert set(packed_index) == set(index)
(base / "packed-index.json.gz").write_bytes(gzip.compress(json.dumps(packed_index, separators=(",", ":")).encode(), mtime=0))
(base / "packed-proof.json").write_text(json.dumps(dict(states=len(packed_index), complete=True, lossless=True, files=len(set(index.values()))), indent=2)+"\n")
