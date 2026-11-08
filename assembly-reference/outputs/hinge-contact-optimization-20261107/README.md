# Swept native collision certificates — November7,2026

## Selected implementation

User authorized implementing profile-driven large acceleration while preserving original native task/CAD/contacts/guards. This is independent of the bare-MuJoCo branch. Runtime geometry and per-pair40x40search budgets remain unchanged.

- Cached surface candidate sets cover a target-local swept region:5um for direct hinge parent/child pairs,50um otherwise. This is lazy configuration-space coverage rather than an angular sampling heuristic. Common world translation/rotation is allowed when the relative-motion bound passes.
- A point may be omitted only if its entire swept box is outside the original acceptance AABB or has a conservative positive distance-field bound. Fine bounds split at voxel faces and use extrema of trilinear interpolation on the clipped cell boxes. The nonnegative outside-grid extension is omitted from the lower bound. Every uncertain point remains, in original order; depth/gradient/normal/world position are recomputed with original arithmetic.
- Native SDF search has a separate optional volume-separation certificate restricted to named explicit passive-stop pairs. Source boundary slabs must be positive; adaptive grid-volume subdivision then proves every possible source zero-sublevel region maps into positive target field over the swept envelope. Unknown, touching, non-grid, precision-guard failure or traversal-budget exhaustion runs the original collider. Source samples alone never justify skipping ordinary SDF search.
- Failed volume certificates use a63-call retry cooldown; they always use native search during cooldown. Successful certificates are checked against relative motion each call. Tree/surface and hinge certificates are thread-local and model-bound. Immutable grid/boundary data are shared under a construction lock.

Coverage inequality for each target-local axis r:

```text
abs(t_current[r]-t_anchor[r])
+ sum_c(abs(R_current[r,c]-R_anchor[r,c]) * source_radius[c])
+ floating_point_guard < sweep_radius
```

This bounds every source point, not just previous contacts. Failed coverage rebuilds the candidate set. No assumption that an interpolated SDF is1-Lipschitz is used. All model/grid arrays must remain immutable, as in the existing native cache contract; versioned model changes require new processes.

Ordinary contact acceptance was checked against the actual [MuJoCo4.0.0 source](https://github.com/google-deepmind/mujoco/blob/4.0.0/src/engine/engine_collision_sdf.c): the SDF pair collider filters with the intersection field before accepting nonpositive contact distances. The older3.17snapshot was not used as authority for the4.0 implementation.

## Qualification completed

- **Original frozen native versus selected candidate:**12actual40mscontrol intervals,160substeps each; zero commands first, deterministic nonzero commands next. **216arrays byte-identical**, including all substep trajectories, contacts/forces, state/warmstart, reward/done/guards and observations. `original-comparison.json` and both NPZ captures.
- **247pose exhaustive oracle:**1235byte-identical arrays, max243contacts. Original five-task prototypes, free-body pose perturbations, both hinge ranges[-1.7,1.7], tiny angular changes and common rigid world translation; serial exhaustive mode2 versus two simultaneous tree threads. Both candidate caching and native skipping are bypassed in exhaustive mode. `oracle-parity.json`.
- **Independent compiled-field oracle:**1200random boxes against an independent NumPy trilinear evaluation with128sample points per accepted box;1182certified clear boxes. Separated synthetic grid volumes accepted; overlapping volumes refused.
- **9existing/extended regressions PASS**, including fine/coarse bounds and the isolated synthetic-grid probe. Ruff passes. Permanent tests are `assembly-so101/tests/{test_surface_acceleration.py,grid_certificate_probe.py}`.

## Timing

All paired replay timings use retained3598717/GPUCB7C, one world/one native thread in a2CPU/12G step. No live controller/checkpoint, training step or allocation was changed. CPU process times were also recorded and closely track wall time. These are sequential paired panels, not confidence intervals or production-fleet measurements.

| Reference | Reference s | Selected s | Aggregate gain |
|---|---:|---:|---:|
| Frozen original native tree3df103be |63.825986|31.109912|2.051629x|
| Previous qualified4voxel/8leaf/exact-pose memo |53.067005|31.109912|1.705791x|

Against frozen original: loose8.07–8.64x; covernear2.86–3.09x; upperheld nominal/variation1.66–2.16x. This benchmark retains original0.25ms/160steps; no claimed combined80step speedup or recovery of1.72xNISTefficiency.

Selected panel telemetry:73305candidate-region reuse hits/7611builds,52561empty-region reuse hits. Fine refinement retained3707416of19568254candidate IDs across builds. Ordinary-volume certificates had2successful builds/62failed attempts and1934reuse hits. Most ordinary searches therefore still fall back; do not claim that the two expensive internal SDF searches have been eliminated. The large loose gain principally comes from removing repeated supplemental sampling.

One region per hashed directional geometry pair is retained per native thread (1021slots). Worker packing/thread scheduling may reduce locality; production10world/10thread throughput and RSS must be measured before claiming fleet gains. No collision samples were removed without a conservative field/AABB certificate.

## Reproduction and release

Exact native files: `candidate/plugin/{so101_grid_bounds.h,so101_surface_tree.h}`, `backend-exact.py`, `wowrobo_contact_cache.h`. These four coupled files have been promoted locally to their canonical plugin/native-adapter paths. Build receipts hash all embedded header/source contents. Frozen baseline inputs remain distinct; phase1 and phase2 trials are preserved in `supplement-only/` and `volume-v2/`.

Remote isolated directory is frozen ROOT/native-contact-optimization-20261107/swept. `run.py quick original` and `run.py quick candidate` execute paired probes; `oracle.py` runs the exhaustive/motion panel. Use the pinned original virtualenv and ROOT environment shown in target-v9.json, selecting currently authorized retained resources. Avoid the live learner directories. New native libraries require fresh worker processes; no hot loading.

Two-real-worker,32step PPO comparisons with a CPU-only learner are underway in `ppo-original/` and `ppo-candidate/`. Both start from the exact archived native checkpoint25/966743; there is no CUDA dependency and no live policy update. This will measure actual policy actions rather than only the small replay commands. Main-run chat retains sole production worker/RSS qualification, versioned deployment and monitor ownership.

## Completed actual-policy PPO comparison

Two real rollout workers,32control rounds,64/64valid transitions, CPU-only learner; original and candidate each resume archived25/966743 and save26/966807. Same hostGPUCB7C, same physical-core affinities for bothworkersandlearner, same191sourcepins before/after. Original214.001662s→candidate116.735940s=**1.833211xfull-update throughput**. Scientific metrics match exactly. Savedpolicy,Adam,allSGShistory/sampler,countersandlr match exactly. Bothownedchildsetsclosed; no retained job released and no livecheckpoint changed. Results are one tiny diagnostic update pervariant, notproductionfleet orlearning-quality qualification.

Selectedfourfilecoupledsource and9passingregressions+Ruff are bound by qualification.json. Main-run owner received final receipts for production10world/10threadRSS/performance check and separatelyversioned deployment afterits independentlyowned80steprelease. Livebenefit remains unverified bythischat.
