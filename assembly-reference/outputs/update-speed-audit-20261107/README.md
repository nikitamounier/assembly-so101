# Live update-speed audit — November 7, 2026

Read-only audit of the active `run-v5`. No training parameters, CPU/GPU leases, provider pool, monitor, geometry, contacts or running source were changed. Two isolated four-case CPU probes used the retained spare allocation3598717; live observations used3077705. Probe scripts/results are in this directory.

## What is happening

Latest existing monitor observation: iteration11,87,260 valid transitions,616.900s simulation/collection,2.068s PPO update,140.91 valid transitions/s. The preceding update10 collected42,620 valid transitions in630.714s and optimized in1.055s. Average active environments increased1331.875→2726.875: samples per second doubled while update cadence stayed approximately ten minutes. Configured5090×32 gives162,880 target transitions; inactive/joining workers are correctly masked out.

A fixed32-step rollout must complete32 sequential control rounds. Each worker has10 environments and10 native simulation threads. Adding workers activates more parallel worlds and increases samples/update; it does not divide those32 sequential rounds. Increasing the configured environment count further is not a solution to update latency.

The global barrier waits for every live worker on every control round. Update11 mean worker compute was9.087s/round; mean per-round maximum was16.417s.32×16.417=525.36s of critical-path compute, versus616.90s collection. The remaining91.54s includes dispatch, sampling, membership admission, network and Python overhead; it is NOT a measured network-only duration. The profile clock excludes the late-join loop. Authentication and initialization are currently admitted synchronously between steps. Ideal removal of imbalance alone gives a roughly1.8× compute-bound opportunity, not a guaranteed speedup. The latest slowest host was Modal; earlier updates9/10 repeatedly identifiedGPU1112.

A read-only affinity observation of job3602580 onGPU1112 found10 allowed logical CPUs on5 physical cores. Do not equate its ten Slurm CPU units with ten Modal physical CPU units. The observed process had one thread at that instant, which is insufficient to infer its active native pool size or diagnose serialization.

## Dashboard defect

At the audit, `http://localhost:8881/api/live` returned `version=v3`, path ending `minimum24/run-v3`, and `connection=live`. `assembly-dashboard/server.py::targets` enumerates onlyv3 andv4, excluding the activev5. Its exporter also hardcodes `migration-v4.json` and the older `accepted-states` collection. Thus a healthy connection can continuously display a stopped historical experiment.

Recommendation: discover the activated run from a single release/active-run manifest, preserving the existing requirement for both `command.json` and `learner-process.json`; resolve migration/bank/split counts from that run's metadata. Show collection progress/step number and time since actual SGD separately from exporter freshness. This audit does not modify the separately owned dashboard fork.

## Fresh native measurements

Exact model identity `8c4783bc443e8e9efad8f19890b6910a79299d434dbb3111af9f24733d850879`, MuJoCo4 native CPU, one world/one thread, zero actions, saved controls/warmstarts/friction. Each control interval is40ms simulated time and160 physics substeps at0.25ms.29 bodies,46 velocities,130 geoms,17 explicit pairs. Native C monotonic-clock callback avoids Python timer callbacks. Native phase timers and existing contact split counters are separate evidence; inclusive timer categories are not summed together.

| Start | First probe wall s | Native narrow-phase share | Second probe original SDF s | Second probe surface supplement s |
|---|---:|---:|---:|---:|
| Loose prototype |1.338|98.74%|0.159|1.168|
| Cover near prototype |3.107|98.70%|0.881|2.178|
| Upper far-side prototype |8.295|98.82%|3.350|4.792|
| Upper far-side variation |8.355|98.72%|3.299|4.883|

The second instrumented probe reproduced comparable total times. These are four isolated starts, not production-wide percentages or a learned-policy benchmark. Observation refresh was0.010–0.051s in the first probe; constraint solve was below1% of native stepping in the contact-rich cases. Friction/solver micro-optimization is not the first target.

The surface supplement remains58.8% of the held-variation rollout; original SDF is39.8%. Even eliminating the supplement entirely cannot yield10× on this case (ceiling approximately2.4×). Both stages need attention for a large speedup.

Existing BVH acceleration is ALREADY active (`SO101_SURFACE_MODE=1`). Do not recommend merely turning it on. Relevant remaining costs in `so101_surface_tree.h`: eight-corner/two-transform bound reconstruction at every tree node, repeated grid-block range queries, per-query candidate allocation and sorting. A useful fidelity-preserving first experiment is precomputed pair-relative transforms plus conservative center/extent AABB propagation, reusable traversal buffers and original-order candidate evaluation. Prove identical contacts and replay before a versioned deployment. Original part/part collision search still uses40×40 budgets; blindly reducing those budgets can miss contacts.

## AssembleNIST comparison

Reviewed upstream commit `eb5d15e004591d380c0d521da67e025ebee37703` is available in this checkout and was checked against its public source.

- [Training defaults](https://github.com/suniique/assembly-nist/blob/eb5d15e004591d380c0d521da67e025ebee37703/assembly_policy/train.py):8192 environments,32 rollout steps,5 epochs,4 minibatches. We already inherit the PPO pattern. Its batch is262,144 transitions.
- [Reference setup](https://github.com/suniique/assembly-nist/blob/eb5d15e004591d380c0d521da67e025ebee37703/README.md#training-and-evaluation): one GPU,960 single-thread CPU worker processes across roughly30nodes;10k iterations in roughly26h, approximately9.36s/iteration averaged over the run. This is a reported workload, not an equivalent-hardware comparison to today's bimanual task.
- [Environment](https://github.com/suniique/assembly-nist/blob/eb5d15e004591d380c0d521da67e025ebee37703/assembly_policy/sim/env.py) and [physics](https://github.com/suniique/assembly-nist/blob/eb5d15e004591d380c0d521da67e025ebee37703/assembly_policy/sim/scene.py): task-specific models, primitive rods/pegs, convex gear teeth, specialized helical SDFs and mesh connectors. Most tasks use2.5ms/16substeps; gear/nut tasks use0.5ms/80substeps with20×5 fine SDF search. Our entire full-workcell bimanual model uses0.25ms/160substeps for every task. That is2× their fine-task and10× their coarse-task substep count, before geometry/contact complexity.
- Upstream constructs equal-sized task/family banks. Our partial bank currently has65/65/65/187/131 rows across the five tasks. With fresh all-failure SGS history, the larger upper/final pools receive more initial sampling mass. Latest batch success0.46%/safety95.6% is a separate learning-health concern; faster updates alone do not establish useful learning.

The earlier same-allocation v13 SO101/upstream audit in `outputs/ppo-profile-20261106/README.md` measured301.5s versus7.56s at identical8192×32 batch size, with narrow phase dominant. It is historical corroboration, NOT a current-model speed ratio.

## Recommended order

1. Fix dashboard run discovery so the user sees actualv5 progress.
2. For faster optimizer feedback, run a versioned16-step pilot:81,440 target samples at5090envs, approximately4–5min/update at current steady per-step cost. An8-step pilot is roughly2–3min but has a smaller40,720target batch and shorter GAE horizon. Withgamma=.995/lambda=.95, the tail factor is0.406 at16steps versus0.165 at32 and0.637 at8; shorter rollouts rely more on bootstrap values. No claim of equal sample efficiency.
3. Remove authentication from the training thread and make admission bounded; use per-stage cost keys and measured physical-core capacity for assignment. Retain hard starts and correct SGS probabilities rather than dropping expensive samples to inflate throughput. Benchmark1/2/4worlds per physical core before selecting packing.
4. Optimize the measured collision paths while preserving source geometry and accepted contact behavior. The current exact-sample tree is still expensive; original SDF search is also material. Native timing shows why more GPU/PPO work is not the priority.
5. Only then test0.5ms or phase-specific timesteps and task simplification, with the full overlap/retention/insertion/passive-hinge panel. Those are physics changes, not free speed improvements. Balance task/start admission coverage and audit the high safety-termination rate when judging learning.

No live scientific changes were made. Smaller rollouts, new scheduling, collision implementations and timestep changes remain proposed experiments.


## Implemented follow-up

The user authorized the first surface-tree optimization. Source, exact frozen A/B replay, native identities and qualification results are documented in [CANDIDATE.md](CANDIDATE.md). The baseline live run remains unchanged.
