# Native narrow-phase collision profile — November 7, 2026

[Open the self-contained interactive flame graph](flamegraph.html). Click frames or use the pair selector to zoom; Reset returns to the full view. No iframe, network requests or external JavaScript dependencies. [Whole-phase SVG](whole-narrowphase.speedscope.svg), [raw native-call SVG](native-detail.speedscope.svg), [measurements](summary.json), [function counts](functions.csv).

## Result

**The custom base/motor contact supplement dominates this workload: 4.389 s, or 75.76% of narrow-phase time.** This is a native C loop over visual-mesh vertices and triangle centroids, in both directions. It is not the 40 × 40 SDF search.

Across 3,168 base/motor collider calls, the supplement tested **241,677,216 points — 76,287 per invocation**. **Every point failed its target-local AABB check.** The supplement performed **zero SDF distance queries, zero gradient queries and zero contact-bin updates**. The entire base/motor pair returned zero contacts in this replay.

The original native base/motor SDF collider took only **0.001046 s total**, about **330 ns per invocation**. It takes the cheap bounding-box exit. The costly work happens afterward in `supplement_motor_base`, which still walks the visual meshes. Source: [executed supplement source](source-so101_contact_search.c), [coarse timing and counter copy](stage_contact.c), [exact counter values](surface-events.txt).

This resolves the earlier pair-level ambiguity: a “base/motor collider call” included both MuJoCo's collider and our supplement. The prior auxiliary probe's 82.93% pair-time result did not isolate SDF optimization. Do not infer that reducing SDF starts or gradient iterations would remove this cost.

The next largest cost is screw `male_contact` ↔ `driver_case_front`: **1.257762 s / 21.71%**. It follows `mjc_MeshSDF → traverseBVH → meshFaceCallback → processOneFace → processSdfCorners → stepFrankWolfe → mjc_gradient → geomGradient → bolt_eval → thread2d / helix / segdist`. It made 72 calls and returned 216 contacts. This is a mesh/SDF path, distinct from the SDF/SDF gradient-descent routine.

## Measured whole-pipeline stages

| Stage | Seconds | Fraction of narrow phase |
|---|---:|---:|
| Base/motor custom supplement | 4.388815 | 75.76% |
| Screw/driver native MeshSDF | 1.257762 | 21.71% |
| Other native colliders | 0.109893 | 1.90% |
| Search-budget selection / adapter setup | 0.007468 | 0.13% |
| Other supplement invocations | 0.003089 | 0.05% |
| Outer MuJoCo / stock collider / insertion / timer residual | 0.026332 | 0.45% |
| **Whole narrow phase** | **5.793360** | **100%** |

The tiny outer residual includes MuJoCo pair setup/filtering, stock colliders outside the custom SDF adapter, contact insertion and timer/observer bookkeeping. It is included in the graph but remains **unexpanded**, rather than being attributed to invented native functions. Plane/SDF contacts retain MuJoCo's original `mjc_PlaneConvex` implementation; its internals are opaque in this capture. Broad-phase work is excluded by the MuJoCo narrow-phase timer boundary. Search-budget selection and all executed supplement invocations are included.

## Workload and correctness

- Separate CPU-only Slurm job **3604284**, host **CPUDFDE34**, four allocated CPU slots. Replay executes on one thread; observed step affinity `[85, 181]`. No GPU requested; no training process restarted or modified. All remote source reads, builds and simulation execute in allocated steps. Login handles transport / scheduler / step launch only.
- Active Chimera **v13** physical model, signature `abc9a9fcaec52d0db9d1c43b0238df66447d9915b7dc618eba9b3fe63cdb24c4`, MuJoCo **3.17.0**, motor/base material-contact guard **enabled**. This is not a benchmark of the newer local v14 environment.
- CPU checkpoint-policy collect: 44 environments × 8 control steps, 352 valid transitions across the five tasks. States and applied controls saved before/after each control step into [frozen replay](replay.npz). No optimizer update.
- For each of those 352 snapshots, replay the saved control for **eight native physics substeps**, plus initial `mj_forward`. Thus 2,816 native steps and 352 initial forward passes; **not** a full control-step rollout or full PPO iteration. This samples point-in-trajectory workloads without changing the saved positions or controls.
- All four observer variants match the original replay **bit for bit**: final integration state for every snapshot, contact count after every substep, and SHA256 over every step's contact pair IDs, distances, positions and frames. Contact geometry hash: `525b0d87539552da8a23b50643bdac69341d25ad2f0e33a447558bd860603a83`. All 2,816 substep contact counts are preserved; sum is 74,965 contact records. See `*-states.npz` and [parity assertions](summary.json).
- First diagnostic dispatch attempt incorrectly routed plane/SDF to the generic SDF collider. It failed parity and was rejected. Corrected dispatch preserves the stock plane path. Rejected outputs remain remote under `invalid-dispatch-attempt/` and are excluded from all delivered graphs and findings.

## Measurement method and limits

Three complementary observers avoid asynchronous signals, ptrace and native stack unwinding:

1. **Original-library pair timers:** a delegating `mjCOLLISIONFUNC` wrapper records every callback's duration, contact count and pair. Original adapter and plugin binaries remain in use. Replay wall **6.674 s**, versus stock **6.662 s**; see [pair timings](timed-pairs.csv). Base/motor pair 4.162 s, screw/driver pair 1.274 s.
2. **Coarse adapter stage timers:** diagnostic copy of the exact adapter, compiled at the original `-O2`, with timestamps around budget selection, original native collider and supplement. Original MuJoCo / SDF plugin / driver binaries remain in use. Plain counters classify surface samples; no per-point clocks. Wall **6.888 s**, about **3.4%** above this single stock replay. These stage timers supply the main graph's measured widths. The run-to-run difference includes timing noise as well as observer work.
3. **Native compiler function tracing:** isolated copies of MuJoCo's exact 3.17.0 `engine_collision_sdf.c`, the custom adapter and both plugin sources, with `-finstrument-functions`. One in eight calls **per pair** is selected; all nested instrumented function calls inside the selected callback are recorded synchronously. **6,540 of 52,317** callback invocations traced, including 396 base/motor and 9 screw/driver calls. No asynchronous sampling. Raw function names come from the compiled library's symbol table. The wheel hides `mjp_getPluginAtSlotUnsafe`, so the diagnostic copy delegates that accessor to the public checked API; it returns the same registered plugin. Parity passes. Wall **10.030 s**, about **50.6%** above stock. Rebuilt-but-not-actively-traced replay was **7.181 s**.

**Interpret the two flame views separately.** “Whole narrow phase” uses measured coarse stage durations; deeper widths are scaled from the corresponding stage's traced-call proportions. Those deeper proportions are estimates and can be biased by instrumentation, particularly for tiny leaf functions. “Raw traced calls” shows unscaled measured instrumented durations of selected calls only. Compiler inline hooks, clock reads and trace bookkeeping distort fine timing; function-call counts and paths are the stronger evidence at that granularity. Do not sum inclusive durations or compare raw selected-call seconds directly with full-replay seconds. Native functions in uninstrumented shared libraries are charged to their nearest instrumented caller. The raw trace includes 30,209,652 `sample_surface` calls; the full-workload counter establishes 241,677,216 point attempts.

These snapshots cover this all-loose checkpoint-policy workload. They do not establish the distribution for successful insertions, near-mating contacts, other reset banks or a whole long-running training iteration. No speedup or contact-suppression change has been applied. A future optimization should conservatively reject the **supplement's mesh traversal** when it cannot add contacts and be validated on contact-positive / near-mating states too; this profile is not permission to disable the supplement globally.

## Reproduction and artifacts

Executed source copies, exact source/library hashes and replay hashes are included in [build metadata](build.json) / [provenance](provenance.json). Upstream source: [MuJoCo 3.17.0 SDF collision routines](https://github.com/google-deepmind/mujoco/blob/3.17.0/src/engine/engine_collision_sdf.c), [stock collision dispatch table](https://github.com/google-deepmind/mujoco/blob/3.17.0/src/engine/engine_collision_driver.c). Our original native source copies retain their source names with `source-` prefixes. Diagnostic driver scripts are also under `operational/narrowphase-20261107` in the local repository.

```sh
# On allocated CPU compute only; scripts guard the diagnostic allocation.
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
../source-85c006a/.venv/bin/python build.py
../source-85c006a/.venv/bin/python capture.py prepare
../source-85c006a/.venv/bin/python capture.py stock
../source-85c006a/.venv/bin/python capture.py diagnostic
../source-85c006a/.venv/bin/python capture.py timed
../source-85c006a/.venv/bin/python capture.py traced
../source-85c006a/.venv/bin/python capture.py stages
../source-85c006a/.venv/bin/python analyze.py .
```

For a future allocation, deliberately update the diagnostic allocation guards. Do not run those commands on the login node. The existing training GPU/CPU pool, source hash guards and monitor contract are unchanged. No commit or push.

Diagnostic allocation3604284 was positively confirmed CANCELLED after artifact retrieval (17m21s elapsed,4CPUs). Retained trainingGPU3598717 remained RUNNING onGPU70A0. Viewer verified with80 rendered SVG rectangles, zeroiframes, working pair zoom/raw trace/reset controls.
