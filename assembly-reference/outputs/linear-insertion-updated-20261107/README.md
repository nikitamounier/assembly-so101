# Updated WowRobo insertion force probes

These five videos replace the earlier convex/contact-off diagnostics for the user's updated-backend request. They retain the frozen current poses and use MuJoCo4.0.0 with16grid-SDF components, exact surface-tree/grid-bound acceleration and sparse constraint solving adapted from the initial-run implementation.

The isolated native adapter changes pair routing for the new component names. It retains the original sampling, contact equations, original sample order and culling headers; neither existing training libraries nor running experiments were changed. This does not transfer the old task's14.25× measured throughput result to this new geometry.

Grid pitch is0.4mm for structural/servo meshes and0.2mm for the pin. Source triangles and winding-number sign define the grids; cavities are not convex-filled. Random sampled interpolation error and targeted mating/corridor checks have finite coverage. Several source meshes are not watertight, so this remains a provisional physical contact profile.

Force ramps to0.5N. Incoming components have dynamic free roots and internal servo hinges. Helpers and receiver coordinates are fixed each0.5ms substep, gravity is off, and force acts directly on the part. There is no physically maintained robot grasp and no pose-interpolated insertion. Authored-depth and motion safety stops hold the last simulated frame with force removed. No clip is claimed as qualified seating or learned assembly success.

Solver: implicitfast/elliptic;100iterations,tolerance1e-12;20line-search iterations;20noslip iterations,tolerance1e-12;SPARSE; sleepdisabled;40×40mating searches with50contact allocator capacity. Per-run receipts contain actual options and native search counters.

Validation evidence:

- `pose-preservation.json`: all33input states preserve original visual poses exactly, with unchanged qpos and body mass/inertia.
- `validation/`: independent exhaustive/accelerated contact/dynamic parity, source-grid cavity checks and native frame registration.
- `grids/generation-validation.json`: binary format, finite fields and sampled grid approximation.
- `pin/`: targeted shaft/corridor checks against source geometry.
- `final/`: exact source identity, solver provenance and known original overlap diagnostics.

Each step folder contains `insertion.mp4`, sampled frames, `final-state.npz` and `receipt.json`. `all-insertions.mp4` concatenates cover, connector, pin, upper and final. The old convex/OFF outputs remain separately preserved at `../linear-insertion-20261107/`.

Frozen source scene SHA256: `3da3dc40dc58e72d194e438dfc314964fd1a93f34ec722c31d38b8966ca3485a`.
Updated scene SHA256: `cc60e4c511540a7f3dd90720088bc863e14afe6175b89f6e5f6841f7b64d5262`.
Grid manifest SHA256: `059630544992bb2c5a1d2d732074b8e34611d3d205295240a5dddfe4e24215d5`.

## Recorded response

| Step | Axial travel / reference (mm) | Lateral drift (mm) | Orientation error (degrees) |
|---|---:|---:|---:|
| cover | 21.64 / 55.00 | 6.54 | 6.97 |
| connector | 42.27 / 65.00 | 8.35 | 25.30 |
| pin | 14.56 / 32.00 | 0.03 | 0.00 |
| upper | 7.12 / 25.00 | 7.32 | 10.48 |
| final | 5.20 / 25.00 | 11.95 | 11.38 |

All five run4seconds and remain short of the authored reference. No insertion success is established. The pin stalls within0.020mm axial travel of an independently detected source-triangle obstruction. Final-module starts retain approximately1.5mm material overlap from the accepted imperfect poses; this profile is not training-admitted.
