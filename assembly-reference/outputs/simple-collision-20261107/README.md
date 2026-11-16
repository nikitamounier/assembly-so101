# Simplified collision experiment — November 7

The detailed CAD points supplied to the policy remain unchanged. Two separate, opt-in collision models were constructed; neither has been deployed to the current run.

## Full compound model: rejected

A matched single-world panel on GPU1112 covers 29 task-conditioned starts (25 unique prototypes), one 40 ms zero-action interval each, at the original 0.25 ms physics timestep / 160 substeps. Full `env.step` is timed, including observation refresh; model loading, resets and point encoding are excluded. Both variants use one simulation thread, affinity `[6, 102]`, in separate processes.

| Measurement | Original | Full compound candidate |
|---|---:|---:|
| Total stepping time | 224.424 s | 206.579 s |
| Relative throughput | 1.000× | 1.086× |
| Safety-triggered starts | 0 / 29 | 29 / 29 |
| Instability-triggered starts | 0 / 29 | 2 / 29 |
| Collision geoms | 130 | 1,993 |
| Explicit collision pairs | 17 | 87,604 |

This raw 1.086× ratio is **not a usable training speedup**: no matched candidate case passes the unchanged safety checks. Some early stages step 2–6× faster, but larger assemblies generate enough compound contacts and constraints to erase most of the gain. Constraint-arena warnings appear in late stages. No safety threshold, solver, timestep or contact law was relaxed.

Actual point-table hashes match, and all 87 initial arrays present in the original capture are byte-identical, including encoded detailed-CAD policy inputs. The builder and independent contract check retain all 54 visual meshes, mechanical fields, integration options and pair/material laws. Thus the failed physical qualification comes from the approximation, despite the retained visual/input contract.

Examples of newly overlapping shapes at valid baseline starts:

- Forearm print against servo 3 case: 0.578 mm penetration even at loose starts.
- Base against side cover near insertion: 0.735 mm.
- Servo 1 horn against connector: roughly 1.62 mm.
- Connector against pin at final-near: roughly 4.47 mm initial contact depth.

The sampled material/void panel uses 762,602 deterministic samples across 22 critical geometry instances: 595 samples have added material more than 0.5 mm into the native void, and five miss native material deeper than 0.5 mm. These are **sampled diagnostics, not a global geometric bound**. Approximate compaction can bridge small functional clearances despite its construction-time guard. Generic decomposition settings are insufficient to certify an insertion opening.

Evidence: [paired results](comparison.json), [baseline timing/contacts](baseline-benchmark.json), [candidate timing/contacts](candidate-benchmark.json), [geometry panel](geometry-check.json), [contract check](contract-check.json), [asset/source receipt](candidate-receipt.json).

## Conservative hardware-only model

The second model replaces 32 non-grasp robot helper collision surfaces with single convex envelopes, while retaining all 22 original SDF surfaces for fingers and assembly parts, including the pin, holes, servo/horn interfaces, grasp surfaces and support geometry. Collision names/indices, explicit pairs and native hooks remain intact. No decomposition or compaction touches these functional surfaces.

The separate build verifies exact mechanical fields, all original explicit-pair laws, and original functional geometry types, poses, mesh vertices/faces, masks and material parameters. Compiled model retains 130 geoms and 17 pairs. Detailed policy inputs still come from the original review model.

The fresh matched panel on GPU105E measures **158.863 → 154.843 s: 1.026× throughput, or 2.53% less stepping time**, with **0/29 safety failures and 0/29 instability flags**. This is one sequential trial per model; no confidence interval, worker scaling or full-PPO improvement is established. It is a small gain and does not justify changing the live run on its own.

All 116 initial capture arrays are byte-identical, including the original detailed point inputs. In total 200/203 replay arrays match exactly: 28/29 trajectories are byte-identical. Only pin-pickup differs, with maximum body translation difference **1.770 micrometres** and rotation difference **0.000501 degrees** after 40 ms. Its helper13/helper19 self-contact changes from one to two near-identical ~2.55-micrometre-deep contacts, affecting subsequent finger/pin forces slightly. This is a measured approximation difference, not exact physics parity.

On accepted-v6 prototypes, **all 29 grasp-contact controls retain the same body pairs, with none added or lost**. **All 125 insertion controls pass**, using Cartesian shifts of0.25/0.5/1/3mm around the five near-insertion states: zero missed blocks, zero new obstructions, zero depth difference; 125/125 full mating-contact captures (positions/normals/depths) are identical. These are fixed-pose checks and do not establish long-episode insertion success. See [functional controls](functional-hardware-check.json). Timing uses the verified idle retained hold 3598717/GPU105E, affinity `[2, 98]`, with a step-ownership check before each launch. Allocation 3606264 now serves the learner and is excluded from further probes. Timings from the two hosts must not be mixed.

Evidence: [conservative paired comparison](hardware-comparison.json), [baseline](baseline-hardware-benchmark.json), [candidate](candidate-hardware-benchmark.json), [preserved geometry/build contract](hardware-receipt.json).

**Disposition:** reject the full compound model; keep the conservative hardware-only version as an opt-in experimental candidate. Leave the current run unchanged. The expensive functional SDF/supplement path remains in the conservative model, so this has not delivered a large acceleration. A future proxy model needs explicit preservation of narrow openings and a much smaller contact manifold, rather than simply decomposing the entire CAD into many pieces.

## Implementation and reproducibility

Canonical opt-in implementation:

- `assembly_policy/so101/wowrobo_collision_proxy.py`: full compound builder and conservative hardware-only builder. Default runtime/factory are untouched.
- `assembly_policy/so101/wowrobo_collision_compaction.py`: native-field voxel boxes and sampled guarded convex compaction.
- `tests/test_wowrobo_collision_compaction.py`: four passing regressions for solid unions, open bores, native-field bounds and voxel material extraction.

Full compound build uses CoACD1.0.14: standard search (20 nodes, 150 iterations, depth3), deterministic seed20261107, no mesh decimation, no blind preprocessing; welded CAD must be closed and oriented. Target concavity is 0.5 mm for prints/pin and 1 mm for hardware/fingers, **not an error certificate**. Four open vendor cases and one open connector use existing native material fields instead of inventing caps (0.5 mm boxes for cases, 0.27 mm connector). Native case frame registrations are applied. Compaction targets 64 pieces softly, with a sampled 0.5 mm added-material guard; difficult parts retain more. The full compound runtime has zero SDF collision geoms; the conservative model intentionally preserves 22.

Frozen source root:

```text
/large_storage/goodarzilab/nikitamounier/assembly/wowrobo-diagnostic-20261107T205745Z-minimum24-accepted-v4
```

Experiment/cache root:

```text
/large_storage/goodarzilab/nikitamounier/assembly/simple-collision-20261107
```

Original model signature: `8c4783bc443e8e9efad8f19890b6910a79299d434dbb3111af9f24733d850879`.
Point signature: `faece0010eb71bc4d2ab7351d67d4e23935d0545aa7d1cc89e308fe50baaa70d`.
Point-array SHA: `062b507a3662bfbea407a6e6da7c1285ea287dd518a2e2278a87e6462bbcb56a`.

The timing panel freezes accepted-v5 prototypes, whose physical and observation signatures also match the current run-v6 configuration inspected earlier. It does not certify the entire expanded randomized v6 bank. A pre-requeue baseline is saved as `baseline-before-requeue-*` and excluded from paired comparisons. Use [source identities](source-identities.json) and both variant receipts for exact versions; `wowrobo_collision_proxy-v1.py` preserves the first builder with its receipt-matched SHA.

CoACD is installed in a separate `simple-collision-20261107-deps` directory, not the live Python environment. The offline native-distance batch helper exactly matches 64 scalar reference queries; it is used for constructing/diagnosing proxies, not for changing the live backend.

All substantive work runs inside retained Slurm CPU steps; no GPUs, Modal or SAIL compute are launched. Generated meshes, frozen model binaries and raw replay arrays remain at the isolated remote root; local JSON/NPZ captures and scripts accompany this report. No live model, reset bank, learner, monitor, cloud application or allocation lifecycle was changed by this experiment.

To reproduce the conservative paired benchmark while the retained hold is actually idle:

```sh
python3 outputs/simple-collision-20261107/run_benchmark.py baseline --cached-baseline --tag=-hardware
python3 outputs/simple-collision-20261107/run_benchmark.py candidate --candidate-scene hardware/model.mjb --candidate-native --tag=-hardware
```

Run the launcher from the canonical checkout; it reads the frozen target settings. Revalidate allocation ownership and hostname before reusing an old lease ID. Both processes must finish on the same host with the same CPU affinity. These are small zero-action simulation probes, not complete PPO, worker-scaling, learned-policy or real-world transfer tests.
