# Deep MuJoCo Warp profiling and scaling — November 7, 2026

The initial H100 result was dominated by a launch-layout problem. The selected changes make the 16-world GPU backend **3.06× faster**, and a fresh full-environment comparison reaches **7.41× versus one frozen native-v13 simulation thread** (previous result2.70×). Large batches reach **62.0control intervals/s at2048worlds**, with diminishing gains beyond1024. Live native-v14 training is unchanged.

![Measured kernel bottleneck and batch scaling](performance.png)

## What was actually slow

Four complementary measurements were collected: guarded whole-rollout wall timing, CUDA-event timing around the captured graph, synchronized engine-phase events, and per-kernel CUDA events. Nsight Systems separately captured actual four-substep CUDA graph executions. Compilation/model initialization are excluded from warm throughput numbers. Raw event profiles and captured-graph timings are separate runs/poses, not additive performance totals.

At16worlds, the serial **no-slip friction solver consumes91.18% of summed measured GPU kernel time**. It runs one thread per world, but the original launch uses one256-thread block for all16worlds. One block cannot be distributed across SMs. Nsight confirms the same kernel uses128registers/thread in both versions.

The selected launch uses16blocks of1thread at16worlds. Per-world arithmetic, the20no-slip iterations, Newton100/line-search20maximum iterations,1e-12solver/no-slip tolerances, contacts, geometry, margins, timestep and safety guards are retained. The four captured no-slip calls fall392.31→101.34ms; total captured kernel time420.98→129.57ms. [Nsight launch/timing receipt](nsight-summary.json).

A cProfile read initially appears to blame NumPy copies:4.506s in copies and3.883s in capture_launch within8.409s. These include waiting for outstanding GPU work. The full adapter averages8.428s while CUDA-event graph timing averages8.398s. Actual reset/upload/download/packing overhead is therefore small in this workload; optimizing the apparent copy stack would miss the dominant work. Native endpoint observation/reward refresh is measured separately by the full-environment test.

## Selected implementation

1. `wowrobo_warp_runtime.py` privately recompiles only the upstream `noslip` launch function with `block_dim=1`; it invokes the unchanged cached upstream kernels. Installed dependency files are not edited. Version/source-pattern guards fail closed, and `close()` restores the original global function. `noslip_block_dim=None` selects upstream scheduling.
2. `wowrobo_warp.py` skips sample scratch writes and reductions for inactive or ineligible contact slots. `_samples`, `_reduce` and `_emit` share the same eligibility predicate. Valid samples retain their original IDs, arithmetic, tie breaking and contact material laws. At64worlds,2080broadphase pairs occupy capacity32768; most capacity slots were wasted work. The measured extra improvement beyond launch scheduling is small at64worlds (~2.6%), larger at1024 (~10.6%).
3. Compatibility only: the full-gradient source generator defines the native evaluator's infinity-default threshold when using the newer optional-threshold native function. The frozen old source used for these measurements generates the same CUDA evaluation body.

Selected sources: [GPU contacts](../../assembly_policy/so101/wowrobo_warp.py), [runtime](../../assembly_policy/so101/wowrobo_warp_runtime.py). Frozen copies are in `selected-source/`; the original baseline is in `baseline/`. Checkpoint/backend source signatures distinguish these implementations.

## Batch scaling

Each control interval is80physics substeps ×0.5ms =40ms. Values below are aggregate physics control intervals/second, including the runtime's reset/upload/download/fine-state/sensor packing and guards, excluding policy, CPU endpoint observations/rewards and reset admission. Each warm measured call restores identical saved inputs, including warmstarts, equality activation, friction and hinge noise. Two measured repeats per point. These are short matched replay panels, not production-weighted confidence intervals.

| Worlds | Original Warp intervals/s | Selected intervals/s | Same-batch gain |
|---:|---:|---:|---:|
| 16 | 1.90 | 5.81 | 3.06× |
| 64 | 5.49 | 18.69 | 3.40× |
| 256 | 15.61 | 39.96 | 2.56× |
| 1024 | 31.15 | 56.74 | 1.82× |
| 2048 | — | 62.00 | — |

The launch-only intermediate series is preserved in [machine-readable summary](summary.json). The selected2048case uses23,580MiB of device memory;1024uses12,920MiB.1024→2048 doubles per-call latency18.05→33.03s for only9.3% more throughput. A1024-world batch is a reasonable next qualification size;2048demonstrates the throughput ceiling on this replay mix rather than an automatic training recommendation. No allocation or batch setting was changed for live training.

Worlds span five actual assembly stages and saved variations. At1024and2048the pool contains213unique initial payloads and cycles them to fill the batch; these are independently simulated worlds, not1024/2048unique episodes. The16-world sample has14unique payloads,256has184. The separate all-stage contact audit covers all25saved stages. No homogeneous cover-only replication is substituted for the mixed batch series.

**End-to-end scaling caveat:** the large-batch harness constructs one native model/data reference and a batched GPU runtime; it deliberately avoids duplicate CPU environments. The current full environment still maintains per-world native endpoint data and serial native forwards for observations/rewards/independent checks. The2048physics result does not establish2048-world full-environment memory use or PPO throughput. Profile/parallelize that endpoint path or port its required quantities to GPU, preserving the guards, before a large training rollout. The directly measured full-environment result here is16worlds.

The original16-world setting delivers1.90intervals/s; selected2048delivers62.0 (~32.7× greater aggregate throughput). This combines algorithm scheduling and much larger batching. It is not a same-batch32.7× speedup, a32.7× full-PPO speedup or a CPU-fleet equivalence claim.

## Where time goes after scaling

A separate2048-world, three-substep profile of the selected runtime shifts the bottleneck:

| Kernel | Fraction of measured kernel time | ms/substep |
|---|---:|---:|
| Surface sample evaluation |60.04%|247.47|
| Ordinary SDF narrowphase |22.37%|92.23|
| Serial no-slip solve |9.93%|40.94|
| Newton constraint Hessian update |4.71%|19.40|

The original tiny-batch bottleneck and the large-batch bottleneck differ. Even eliminating no-slip completely now has only~1.11× ideal kernel-time headroom at2048worlds. Surface sampling and ordinary SDF together account for82.4%; they are the next major target.

The strongest next implementation is a GPU grid-range hierarchy: use conservative SDF brick minima to reject whole transformed source regions, including empty regions inside overlapping bounding boxes. Keep uncertain leaves and their exact current arithmetic. For positive contact patches, use the previous twelve normal-bin winners as seeds and conservative distance/normal bounds to avoid evaluating candidates that cannot win. The current port still scans vertices/triangle centroids and uses the ordinary full search budgets; it does not inherit the CPU runtime's conservative separation/candidate-tree accelerations merely by importing the same physical model.

For smaller batches, a cooperative warp/block per world in no-slip remains worthwhile: parallelize independent46-DOF acceleration updates and improve contiguous row access while preserving sequential friction-coordinate order. At large batches this is secondary to SDF work. Tightening focus on transfers or relaxing Newton tolerance cannot remove the dominant82% collision cost.

## Rejected/deferred trials

- **Spatial Morton chunks + conservative AABB rejection:** contact parity passed, but64-world wall time3.425→3.484s and sample kernel29.21→31.31ms/three-substeps regressed. Preserved in `spatial-rejected/`, not promoted. This bounds-only trial does not invalidate stronger SDF-value-range rejection.
- **Conditional surface gradients:** exact contact parity passed;64-world wall time3.425→3.370s,1024-case17.45s. One2048trial hit the strict **line-search iteration limit20** during warmup (`LS_ITERATIONS`, preserved in the raw warning). A subsequent instrumented2048replay passed, so the failure is intermittent and is not causally attributed to gradient arithmetic from one event. The small timing gain does not justify default promotion without diagnosis and broader qualification. Source and failed receipt are preserved in `lazy-rejected/`; canonical runtime retains unconditional original gradients. No overflow flags, iteration limits or safety checks were disabled to obtain timings. Diagnostic support now records per-world/per-substep overflow bits, iteration counts and constraint counts on failure.
- **Initial Nsight attempts:** the GPU ownership guard stopped Nsight's target because Nsight created a separate process session. The corrected guard accepts only the same UID and exact experiment script/output arguments; the repeated traces pass. These stopped attempts are excluded from timing tables.

## Correctness and complete integration

- Same-input no-slip launch audit at64worlds: solver counters `[64]`, accelerations `[64,46]`, constraint forces `[64,2048]` byte-identical, zero error.
- Same ordinary manifold and transforms, old/new contact callbacks: all semantic contact fields (distance, position, frame, margin, friction, solver parameters, dimensionality, geometry/world/type IDs) exactly equal after canonical row sorting. Tests include64mixed worlds and64worlds covering all25saved stages. Inactive scratch skipping changes no accepted contact selection.
- All-stage short replay stays finite, zero overflow, minimum contact distance-0.23296mm against0.5mmguard. Larger selected batches remain finite, recorded overflow0, peak penetration~0.201mm in the measured calls. Full per-substep trajectories/sensors remain available to the original guards. An independent pass reapplies the original arm/free-body velocity and penetration guards to the saved scaling trajectories: [trajectory safety receipt](trajectory-safety.json).
- Fresh full environment,16worlds ×3controls:48reward rows match; safety/unstable0; maximum endpoint body-position difference37.4µm over120ms. Warm native47.9493s vs selectedGPU6.47164s =7.409×. Includes original native endpoint refresh, observations, rewards and existing safety screening. Native comparator is frozenv13; it excludes later native-v14's conservative separation/four-way-cache optimization. This is one short paired sequence, not16parallel CPU workers or a fleet forecast.
- Fresh actual PPO update:4worlds ×2controls,8valid transitions, finite policy and45Adam states. Checkpoint SHA256 `7f1bcb5733d14bb8d7ba73cb3c52dbedae4f8660c202ce6a7b194664ee51e84c`; collection6.1s, update0.2s. [PPO proof](ppo-proof.json). This is integration qualification, not a paired full-PPO speedup or learned-assembly result.

GPU state/ordinary manifolds remain float32 and differ from native MuJoCo. Full trajectories can differ between repeated GPU calls due to ordinary atomic contact ordering; fixed-input component parity does not imply bit-identical whole trajectories. No long episode, held-out/full-bank admission, successful learned assembly or production activation is claimed. Live native-v14 jobs/checkpoints/controllers/monitors remain untouched.

## Reproduce and inspect

Remote root: `/large_storage/goodarzilab/nikitamounier/assembly/mjwarp-profile-20261107`. Dependencies: preserved isolated MuJoCo/MJWarp4.0.0, Warp1.19.0+cu12. Physical model signature `13ac7a8d07d4d63e7de21bd426319fd32e60acfd33e9fd681a43a2d1f8cd959e`, original1143-row diagnostic bank. Source, bank, model, precision and capacities are explicit in each receipt.

The launchers require an actually idle retained GPU, refuse other GPU consumers and write only their private experiment root. The job ID3606264 is historical, not a perpetual reservation. Source transfer uses stdin to avoid shell argument-size limits. Primary timing loops have host cProfile enabled, but no per-kernel/phase event or Nsight instrumentation. Host overhead is small relative to device time; independent CUDA-graph event measurements cross-check the wall timings. Event/phase/Nsight attribution uses separate runs.

```sh
python outputs/mjwarp-profile-20261107/launch.py --job ID --worlds 16 --profile
python outputs/mjwarp-profile-20261107/launch.py --job ID --worlds 64 --noslip-block 1 --audit-noslip --profile
python outputs/mjwarp-profile-20261107/launch.py --job ID --variant candidate --worlds 1024
python outputs/mjwarp-profile-20261107/launch.py --job ID --variant candidate --worlds 16 --environment
```

`analyze.py` regenerates `summary.json` and `performance.png` from saved receipts. GPU-event timing follows the [Warp profiling API](https://nvidia.github.io/warp/v1.19/api_reference/warp.html); the installed source/API was inspected before instrumentation. Hardware launch facts come from our actual Nsight traces.
