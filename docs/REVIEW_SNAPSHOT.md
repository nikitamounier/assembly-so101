# Collaboration snapshot — November 7, 2026

Current assembly/training/runtime source and the complete portable viewer runtime are published to `nikitamounier/assembly-so101` on `nikita/so101-acceleration-cleanup`. Start with [viewer launch instructions](VIEWER.md).

The repository root contains the SO101 viewer, cube trial and compute implementation. `assembly-reference/` preserves the current companion assembly source and review reports without overwriting root modules. It is a source snapshot, not an installed dependency or a second active checkout. Its historical upstream/repository statements describe its origin; this snapshot's publication destination is this repository. Source modules in the two trees have not been merged into a single implementation.

Start with `assembly-reference/User Intent.md` and `assembly-reference/docs/user-intent/README.md`. The latest editable and printable stabilizer pin is in `cad/wowrobo/` (2.3 mm diameter, 14 mm shaft, 12 × 28 mm grip). This supersedes the older 14.5 mm source retained inside the companion snapshot.

Frontend checks: `cd web && npm ci && npm test`. Python setup is defined in `pyproject.toml`; provider dependencies are in `requirements-providers.txt`. CLI entry points: `python -m assembly_policy.viewer --help` and `python -m assembly_policy.compute --help`.

The exact viewer meshes, saved-state bank and frozen overlay contracts are included under `runtime/wowrobo-viewer/`, with checksums and upstream licenses. Credentials, original vendor CAD, unrelated generated meshes/contact grids/reset banks, checkpoints, compute-provider receipts and compiled plugins remain excluded. The complete viewer needs no sibling checkout or private asset download. Other training workflows and historical report links can still require their separate runtime data.

This is work in progress. Geometric seating rewards and viewer poses do not establish physical training admission, settling, support retention or hardware calibration. No experiments are launched or modified by this publication.

The current native/Warp implementation, reset-bank tooling, tests, research reports and operational source are refreshed under `assembly-reference/`. [Exact published source hashes](ASSEMBLY_SOURCE_SNAPSHOT.json) distinguish copied source from generated local experiment data. Publication is exclusively to `nikitamounier/assembly-so101`; no shared-upstream push is part of this update.
