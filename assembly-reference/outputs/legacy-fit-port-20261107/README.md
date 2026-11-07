# Legacy mating-contact candidate

The legacy laws are implemented and bounded-tested in an isolated candidate. **Full insertion and retention remain unqualified.** The production/default scene and existing banks were not switched.

`configs/wowrobo_mating_profile.json` maps 14 constituent pairs by interface role. Unlisted contacts remain firm. The canonical builder accepts `--mating-step` and `--mating-profile legacy_mapped`. No CAD, grid, mass or actual-feature alignment was changed for this comparison.

No extra rail/bearing boxes were added: independent audits show the new bidirectional source-vertex/centroid sampler already covers the relevant surfaces. Old Hiwonder patch coordinates would duplicate contacts on different geometry.

## Results

We ran 35 v1 screening cases plus 7 canonical v2 pin cases. Baseline and mapped cases use identical saved stalled states, 0.5/5/20 N loads, speed-bounded implicit damping, 2.5 seconds of drive and 0.5 seconds of guided unloading. A contacts-off case is only an actuation control. The guide holds lateral position, rotation and incoming internal shape; it does not prescribe axial travel.

| Step | Baseline at 20 N (mm) | Mapped at 20 N (mm) | Reference (mm) |
|---|---:|---:|---:|
| cover | 33.112 | 33.606 | 55 |
| connector | 44.589 | 44.589 | 65 |
| pin | 17.170 | 17.148 | 27 |
| upper | 6.921 | 7.099 | 25 |
| final | 7.294 | 12.541 | 25 |

At 0.5 N, the cover improves from 25.38 to 33.09 mm. The final module gains about 5.25 mm at 20 N. None fully seats. Source-path and deliberately blocked-wall checks pass within their stated finite coverage. Partial endpoints rebound when unloaded or move during gravity release; those releases do not establish seated retention or robot grasp qualification.

The old model omitted the lower horn; the new servo has both horns. A coherent horn/fork index scan at −6°, −3°, 0°, +3° and +6° does not improve their relative fit. Copying the old 0.6 mm upward correction creates 1.025 mm of lower-horn interference and was rejected.

## Original pin is a negative control

The pin video visibly labels the **original Ø2.7 × 22 mm shaft**. Main run separately approved an unprinted **Ø2.3 × 14.5 mm shaft**, 42.5 mm overall. That new design is preserved and is not tested or rejected by this frozen-pin comparison.

The v1 firm pin law passed tests initialized at an existing stall but failed fresh 20 N approaches at both 20 and 2 mm/s. The independent source guard caught crossing a thin modeled thread crest. Failed controls remain in `pin/video/` and `pin/video-slow/`.

V2 restores only the exact old fastener normal law on pin/case: `solref=[.001,1]`, `solimp=[.99,.999,.00005]`. Geometry, friction and the other five pin pairs are unchanged. It passes stored-stall and both fresh-speed source guards while keeping the original pin blocked. No analytic thread geometry or virtual hole was added.

## Evidence and reproduction

`legacy-mapped-insertions.mp4` is the reviewed 20-second reel. `summary.json` records versions, measured results and limits. Per-step folders contain matrices, traces, source audits, wall tests, gravity releases and videos. V1 failures and final sources are preserved under `source-snapshot/`. Final hash checks confirm all 16 original source meshes still match the frozen SDF grid sources.

```sh
PYTHONPATH=. ../assembly-so101/.venv/bin/python scripts/validate_wowrobo_fit_profile.py --step upper
PYTHONPATH=. ../assembly-so101/.venv/bin/python scripts/probe_wowrobo_fit_force.py --step upper --force 20
```

Pin v2 results were exported separately with `--output outputs/legacy-fit-port-20261107/pin/v2`. Rendering requires native graphics access. These benchmark scripts use frozen comparison geometry and guard thresholds; they are not the separate fit-sized-pin validation pipeline.

The contact and material assumptions are not hardware-calibrated. Finite overlap checks below 0.5 mm do not establish positive clearance or continuous fit. This candidate remains outside training.
