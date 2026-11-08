# Original native WowRobo flame profile — November7,2026

[Open the interactive flame graph](flamegraph.html). Select learner, either real worker, or the native pair-timer hierarchy. Click a frame to zoom, reset to return, search names, and hover for exact seconds. No external scripts or network are required.

## Actual capture and isolation

This is a new run, not a reuse of the older SO101 profile. Frozen accepted-v4 native source/model `8c4783bc443e8e9efad8f19890b6910a79299d434dbb3111af9f24733d850879`; original native checkpoint25/966743 SHA256 `df907ee1f1ff8bca4cb1ae640b45c3dd4350ad00f267ba4228a412b1ef3d409b`. All191 active-inputs-v9 pins verified before and after. Original unoptimized surface-tree SHA `3df103be0e3b8cfde509a00da787ad753fc40cd4686db9a4b45163be70e88d8f`; no candidate bounds optimization or bare approximation. Native SDF, generalized supplements, per-episode friction, guards,0.25ms/160substeps/40ms action, sparse solver, tree mode1, original five tasks/rewards,448observation points.

Isolated output under frozen ROOT/native-flame-profile-20261107, using spare retained3606264 on GPU122C. Live run-v9, its workers/checkpoint, all retained allocations and monitors were not changed. One bounded complete32-step PPO update from25→26, with two original worker protocol processes and two worlds; original policy/Adam/SGS/RNG/counters restored by the original checkpoint loader. Bank is exact training-accepted-v6,1143task-conditioned rows; no held-out rows. PPO5epochs/4minibatches/1024gradient chunks and original reward settings. This small diagnostic batch is not a production-fleet throughput benchmark.

| Full PPO metric | Measured |
|---|---:|
| Qualification | PASS |
| Valid transitions | 64 /64 |
| Control rounds | 32 |
| Full collection→checkpoint wall | 194.897814s |
| Collection incl inference/GAE | 194.573059s |
| Wrapped remote steps | 194.183757s |
| PPO update | 0.256178s |
| Checkpoint save | 0.066470s |
| Sum per-round worker maximum | 194.156422s |
| Sum per-round worker mean | 137.494148s |
| Ideal compute imbalance gap | 56.662273s |
| Remote-step residual above worker maximum | 0.027335s |
| Numerical instability flags | 0 |

Residual includes dispatch, serialization, result placement, sampling and scheduling; it is not pure network latency. Worker maximum and mean derive from all32actual worker reply compute timers. Localhost transport removes wide-area provider variation. The two-worker mixture cannot characterize400+worker tail behavior. The ideal balancing gap is a bound, not a promised improvement.

## Physical cores and SMT

Reserved8Slurm logical CPU slots map to4physical cores. Original step affinity: `[2, 5, 73, 75, 98, 101, 169, 171]`. Each simulation worker is pinned to one logical CPU on its own distinct physical core, nthread1; the learner uses a third physical core and the spare H100. See worker hardware JSON and learner hardware in ppo-result.json. Do not call8scheduler slots8physical cores. GPU allocation is reused, not newly submitted. Initialization/model build is outside the learner iteration window; worker wall graphs include their served initialization and idle barriers.

## Native collision attribution

Separate auxiliary panel: four single-world zero-action control intervals, each160substeps. These are measured timer widths, not production-weighted proportions or native stack-unwind samples.

| Start | Original rollout s | Instrumented rollout s | Original SDF s | Supplement s | Calls | Tree nodes | Leaf samples |
|---|---:|---:|---:|---:|---:|---:|---:|
| 00_loose | 1.292117 | 1.286614 | 0.156198 | 1.109434 | 480 | 828800 | 8994240 |
| 05_cover_near | 2.962670 | 2.973809 | 0.848854 | 2.079861 | 1760 | 1640320 | 16538400 |
| 21_upper_far_side | 8.125736 | 7.880301 | 3.220449 | 4.556970 | 5600 | 3923594 | 33698830 |
| 21_upper_far_side variation | 8.595470 | 7.708358 | 3.046560 | 4.550828 | 5600 | 3976156 | 34118834 |

The loose start's three calls/substep are now identified. Both expensive internal pairs return zero contacts throughout this control interval:

| Loose pair | Original SDF s | Supplement s | Leaf samples | Returned contacts |
|---|---:|---:|---:|---:|
| forearm_module_printed_0__contact ↔ servo_3_case__contact | 0.078707 | 0.624560 | 5068320 | 0 |
| upper_arm_module_printed_0__contact ↔ servo_2_case__contact | 0.077409 | 0.484676 | 3925920 | 0 |
| forearm_module_printed_1__contact ↔ servo_3_case__contact | 0.000082 | 0.000199 | 0 | 0 |

The first two are printed-module/own-servo-pivot relationships (authoritative body names: forearm_module↔target_servo_3_pivot, upper_arm_module↔target_servo_2_pivot). They are persistent internal articulated assembly geometry, rather than the active cover insertion. Zero contacts at these poses does not authorize deleting a passive stop interaction over hinge motion.

Upper far-side variation's largest combined pair costs:

| Pair | Original SDF s | Supplement s | Calls | Returned contacts | Zero-contact calls |
|---|---:|---:|---:|---:|---:|
| shoulder_connector_printed_0__contact ↔ servo_2_case__contact | 0.075382 | 0.686890 | 160 | 2073 | 0 |
| forearm_module_printed_0__contact ↔ servo_3_case__contact | 0.077231 | 0.626286 | 160 | 0 | 160 |
| shoulder_connector_printed_1__contact ↔ servo_2_case__contact | 0.078396 | 0.548960 | 160 | 1280 | 0 |
| upper_arm_module_printed_0__contact ↔ servo_2_case__contact | 0.085869 | 0.534609 | 160 | 0 | 160 |
| base_servo_printed_0__contact ↔ base_side_cover_printed_0__contact | 0.075079 | 0.532216 | 160 | 1760 | 0 |
| servo_1_horns__contact ↔ shoulder_connector_printed_0__contact | 0.074172 | 0.349473 | 160 | 152 | 8 |
| servo_1_case__contact ↔ pin__contact | 0.097158 | 0.247018 | 160 | 0 | 160 |
| servo_1_case__contact ↔ shoulder_connector_printed_0__contact | 0.084357 | 0.203608 | 160 | 0 | 160 |

00_loose: pairs returning zero contacts on every invocation account for 1.2656s/1.2656s (100.00%) of pair collision time.
05_cover_near: pairs returning zero contacts on every invocation account for 1.6548s/2.9287s (56.50%) of pair collision time.
21_upper_far_side: pairs returning zero contacts on every invocation account for 3.9290s/7.7774s (50.52%) of pair collision time.
21_upper_far_side variation: pairs returning zero contacts on every invocation account for 3.7788s/7.5974s (49.74%) of pair collision time.

Every invoked source-SDF pair in this panel, including physical_helper grasps, uses40initpoints×40iterations. These are invoked budgets, not proof1600iterations executed. Tree pruning is active: nodes, AABB/grid rejected nodes and surviving leaf samples are recorded per pair. Contact counts are sums across substeps, not simultaneous manifold size. Supplement net-added contacts exclude replacements that leave the manifold size unchanged. The table in the viewer contains every executed pair, its exact bodies, calls, budgets, node/leaf/rejection counts, original and final returned contacts. Lexical role labels are hints; exact geometry/body names should guide qualification.

Prioritize conservative rejection/caching or a qualified restricted surface region for the persistent internal pairs, then the true shoulder_connector↔servo_2 insertion interfaces. Ordinary SDF is distributed across many upper-stage pairs; its largest variation pair is base_side_cover_printed_0↔upper_arm_module_printed_0. Restricting supplements alone cannot remove that search cost. Role-specific budgets need replay/contact/hinge/insertion qualification. No optimization was deployed here.

## Parity, measurement limits, and files

All72arrays across four complete160-substep trajectories are byte-identical: trajectories, sensors, state/qpos/qvel/warmstart, contacts/forces, observations, reward, done and every info field. Both compressed NPZ files also have the same SHA256. See byte-parity.json. Instrumentation delegates original collision math unchanged; only private source copies contain timers/counters. Original build inputs, compiled libraries and instrumented C hash are retained in instrumentation.json and identity.json.

The single sequential original/instrumented timing pass is an overhead check with host noise, not a precise overhead estimate or speedup result. Different probe durations do not establish an optimization. Main PPO uses uninstrumented original native libraries. Worker/learner wall sampling targets20ms and can be biased by GIL scheduling. Native C++ rollout threads have no Python frames. Cluster perf_event_paranoid=4 restricts hardware sampling, so the native view is explicitly a timer hierarchy down to geometry-pair SDF/supplement phases, not invented C function stacks. Observation refresh is directly timed in the auxiliary panel; its small contribution can also be inspected at _refresh/observe frames in worker wall captures. No production-wide observation percentage is inferred from four starts.

Reproducible scripts: common.py, instrument.py, probe.py, capture.py, run.py. Run on the original frozen ROOT with the pinned virtualenv, PYTHONPATH=ROOT/assembly-so101, WOWROBO_CANONICAL_ROOT=ROOT/robotics-assembly-3.0, SO101_SURFACE_MODE=1 and OMP/OPENBLAS/MKL_NUM_THREADS=1. PROFILE_ROOT=ROOT, PROFILE_OUT=ROOT/native-flame-profile-20261107. Full exact launch command is in launch.sh. run.py requires the approved spare allocation3606264, checks every source pin and checkpoint, executes original/instrumented probes, then a bounded two-worker PPO. Review allocation availability before rerunning. Never point PROFILE_OUT at a live run or use mutable local environment code.

Local export excludes the per-run authentication key/server ticket and37MBcheckpoint. Frozen remote artifacts retain the isolated checkpoint. complete.json records closure of every owned child and post-run source verification. Viewer: flamegraph.html; raw sampled stacks: learner.speedscope.json, worker-0.speedscope.json, worker-1.speedscope.json. Rebuild with `python3 build_report.py`; write the report with `python3 write_report.py`.


## Fresh native phase hierarchy and sampled worker detail

The additional phase-clock pass resolves full native stepping through forward/position/collision/narrow phase and the same-pass pair SDF/supplement branches. Child widths use exclusive residuals, so inclusive timers are not double counted. This pass independently passes all72byte-identical arrays against the original replay. Native-phase-validation.json binds the small C monotonic clock source/library.

| Start | Native step s | Narrow phase s | Narrow share | Constraint solve s | Refresh s |
|---|---:|---:|---:|---:|---:|
| 00_loose | 1.292437 | 1.276520 | 98.768% | 0.005588 | 0.008372 |
| 05_cover_near | 2.953653 | 2.915199 | 98.698% | 0.022932 | 0.018832 |
| 21_upper_far_side | 7.604565 | 7.513265 | 98.799% | 0.063468 | 0.047501 |
| 21_upper_far_side variation | 7.702711 | 7.602099 | 98.694% | 0.055374 | 0.048238 |

Sampled leaf-frame times across the real PPO worker lifetime (includes initialization, outside learner capture):

| Process | Wall capture s | Samples | Native rollout call site s | Observation _refresh s | Worker receive / learner selector s |
|---|---:|---:|---:|---:|---:|
| learner | 194.897975 | 9707 | 0.000000 | 0.000000 | 194.221900 |
| worker-0 | 217.895257 | 10829 | 192.401463 | 2.388878 | 1.277546 |
| worker-1 | 217.895092 | 10821 | 80.693689 | 1.445353 | 113.797387 |

These sampled observation/receive times overlap other processes and have20ms sampling resolution; they are not summed with the native auxiliary panel or substituted for explicit worker reply timers. Worker0spends about192s in native stepping while worker1spends about81s and waits about114s, making the barrier imbalance visible directly. The learner spends about194s in selectors.select; that is worker waiting, not evidence of slow networking.
