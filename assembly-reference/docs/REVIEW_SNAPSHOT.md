# Collaboration snapshot — November 7, 2026

This snapshot publishes the current WowRobo CAD/contact, reset-review and compute preparation source for review. The companion viewer/compute repository is https://github.com/nikitamounier/assembly-so101 .

Start with `User Intent.md`, `docs/user-intent/README.md`, and `docs/HYBRID_COMPUTE_PREPARATION.md`. The reports under `outputs/*/README.md` describe the measured results and outstanding failures; their large local evidence files are not part of this source snapshot.

The repaired review collection has 25 nominal states: 16 pass the sampled boundary screen and 9 have remaining material conflicts. No reset has passed full physical training admission. Settling, support retention and full dynamic assembly qualification remain open. Guided insertion results are fixture diagnostics, not learned assembly results.

Vendor CAD, generated contact grids, meshes, reset banks, checkpoints and run receipts remain local. `cad/wowrobo/*-lock.json`, the notices, build scripts and `docs/WOWROBO_*.md` describe source provenance and generation. A fresh clone requires those runtime assets before the CAD-backed viewer or physics checks can run. Historical reports contain local paths that must be adapted to your clone. This push does not launch or modify experiments.
