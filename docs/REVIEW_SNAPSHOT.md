# Collaboration snapshot — November 7, 2026

All current collaboration source is published to `nikitamounier/assembly-so101` on `nikita/so101-acceleration-cleanup`.

The repository root contains the SO101 viewer, cube trial and compute implementation. `assembly-reference/` preserves the current companion assembly source and review reports without overwriting root modules. It is a source snapshot, not an installed dependency or a second active checkout. Its historical upstream/repository statements describe its origin; this snapshot's publication destination is this repository. Source modules in the two trees have not been merged into a single implementation.

Start with `assembly-reference/User Intent.md` and `assembly-reference/docs/user-intent/README.md`. The latest editable and printable stabilizer pin is in `cad/wowrobo/` (2.3 mm diameter, 14 mm shaft, 12 × 28 mm grip). This supersedes the older 14.5 mm source retained inside the companion snapshot.

Frontend checks: `cd web && npm ci && npm test`. Python setup is defined in `pyproject.toml`; provider dependencies are in `requirements-providers.txt`. CLI entry points: `python -m assembly_policy.viewer --help` and `python -m assembly_policy.compute --help`.

Credentials, original vendor CAD, generated meshes/contact grids/reset banks, checkpoints, run receipts and compiled plugins are excluded. CAD-backed execution requires matching local runtime assets; historical report links to excluded evidence and local paths will need those assets. Source provenance and build instructions remain in the companion notices, lock files and documentation.

This is work in progress. Geometric seating rewards and viewer poses do not establish physical training admission, settling, support retention or hardware calibration. No experiments are launched or modified by this publication.
