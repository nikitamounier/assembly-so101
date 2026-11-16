# Isolated Chimera reset-bank release

This directory owns the portable snapshot tooling only. Live local generation,
local v4 receipts, active training, shared cluster environments and job resources
are not modified by these tools.

Frozen artifacts:

- `outputs/reset-bank-chimera-transfer-20261107/reset-workspace.tar.gz`:304,792,522bytes; SHA256 `8244db21faccdd7a61b530603a9154ed4c3b028056bda2f11c98d71bf023f008`.
- `outputs/reset-bank-chimera-transfer-20261107/reset-preflight-overlay.tar.gz`:43,446,458bytes; SHA256 `f41e15298096a6ab4fcf281cae1374b6bd41a73738770ba7da81fc87a3deed13`.

Extract both directly into:

```
/large_storage/goodarzilab/nikitamounier/assembly/reset-bank-1800-20261107-v4
```

The archive preserves the workspace structure: canonical `robotics-assembly-3.0`
and sibling `assembly-so101`. It is already relocated to that exact root. Do not
apply another blind path rewrite. `release.json` supplies the exact verification
and array-preflight commands for the allocated driver. Run all verification,
extraction and numerical work inside the CPU allocation.

`transport-manifest.json` records original and relocated digests for675files.
Sources are byte-identical; only JSON/XML path strings and their digest pointers
were rewritten. Grid, mesh and snapshot binary bytes are unchanged. Main archive
hardlinks deduplicate repeated references to the same immutable grid/mesh data;
logical extracted size is902MB. Local staged verification passed all675payload
hashes plus model/config/grasp/pin/grid/anchor pointer checks. The relocated grasp
library still validates40records and25stage bindings.

`transport-provenance` retains the untouched original local plan and25reference
witness reports/NPZ. Existing local variation attempts are deliberately not
imported or silently relabeled. This is a fresh remote generation identity with
the same sampled distributions and native admission, to be independently checked
by every generated candidate. Local accepted rows remain in their original plan.

The source physical XML SHA was
`909e034485939b77037f9568e59d7d027e37238372e95995c1a8222803ee96ef`.
Path relocation gives
`1b071d9b92dc5e3e3e9ba33234a25f24bfc3b3049081a52ac673b1df423b952d`.
That hash change is not itself numerical equivalence evidence: `parity.py verify`
must pass on Linux before generation. The included49MBsource-array snapshot
covers body/joint/DOF/mesh topology and frames, dynamics, contacts, actuators,
equalities, limits and solver options. Integer fields must match exactly; floating
fields use absolute1e-14 tolerance. MuJoCo4.0.0 is required by the native adapter.
Plugin libraries are compiled on Linux from the included frozen C/C++ sources;
no Mac libraries are shipped.

Exported runtime asset closure includes repaired and original component meshes,
pinned helper/camera assets and lockfiles, reference and physical model files,
raw native grids and metadata, Python/configuration sources and native plugin
build inputs. Historical authoring-proof links that are not read by this frozen
runtime are provenance, not a claim that every historical experiment artifact
was copied. No credentials, `.git`, vendor Rhino files or unrelated image/video
files are included.

Important driver requirement: export `PYTHONPATH` to the remote canonical project
before spawning preflight or stage workers; a parent's `sys.path.insert` is not
inherited by subprocesses. Fresh remote plan relative path is
`outputs/reset-bank-1800-20261107-v4`.
