# Full PPO iteration profiles: SO101 and original AssembleNIST

**The measured gap is simulation cost.** SO101 took **301.53 s** for a full iteration; original AssembleNIST took **7.56 s** on the same allocations. PPO updates took **1.34 s** and **0.75 s**. In two sampled SO101 workers, **93.0% of native stepping time was narrow-phase collision detection**.

Open [the interactive comparison](comparison.html) for zoomable learner and worker flame graphs, a native MuJoCo phase hierarchy, and the timing figures. Click a frame to zoom, hover for full paths, and search for a function. The NIST worker selector also exposes its Python compute-thread stacks.

![Timing and collision breakdown](overview.png)

## What was captured

| | SO101 | Original AssembleNIST |
|---|---:|---:|
| Captured update | 30 | 2, after one warmup update |
| Valid transitions | 262,144 | 262,144 |
| Environments × control steps | 8,192 × 32 | 8,192 × 32 |
| Worker processes / native threads each | 187 / 2 | 187 / 2 |
| Simulator CPU slots | 374 Slurm CPUs | Same |
| GPU | Same H100 80 GB, same lease and host | Same |
| Total iteration, including checkpoint | **301.534 s** | **7.562 s** |
| Collect, including policy inference and GAE | 300.062 s | 6.680 s |
| Remote environment steps | **299.523 s** | **6.113 s** |
| PPO optimizer update | 1.339 s | 0.751 s |
| Checkpoint write | 0.097 s | 0.092 s |
| Actual CUDA kernel interval union | 1.384 s | 0.711 s |
| Kernel-active fraction of iteration | 0.46% | 9.41% |

This is a **39.9× total-iteration difference**, and a **49.0× remote-rollout difference**. Kernel-active fraction measures actual kernel intervals, excluding memory copies; it is not MFU or a general GPU-utilization measurement. CUDA user annotations can span long CPU waits and must not be counted as GPU computation.

Both qualification records confirm 32 steps, every environment active, every transition valid, stable worker connections, complete 32-step worker profiles, and unchanged allocation start times, nodes and restart counts. [Hardware/lifetime comparison](hardware-match.json) is exactly equal between the two captures. These windows contain no requeue, preemption or membership change.

Thirty-seven ten-CPU worker leases supplied 185 workers; two diagnostic workers used four spare CPU slots in the retained GPU allocation. This avoided queueing another GPU. The sampled CPU affinities were `0,96` and `1,97` on an Intel Xeon Platinum 8468: each pair consists of SMT siblings on one physical core. Two Slurm CPUs here should not be read as two independent physical cores.

## Read the flame graphs correctly

The **learner graph** measures main-thread Python wall stacks. SO101 spends approximately **297.6 s sampled inside `selectors.select`**, because it is waiting for worker replies. This does not establish a network bottleneck.

The **worker graphs** also measure wall time, including native call sites and idle waits. SO101's native C++ rollout threads do not have Python frames. Their internal work is covered by the separate **MuJoCo phase-timer hierarchy**, rather than invented native stack samples. NIST's Python thread pool exposes compute-thread call sites in its additional thread views. Worker and native-thread durations overlap and must not be added to iteration wall time.

The native SO101 phase hierarchy represents **222,720 actual MuJoCo substeps** across the two sampled workers, covering all their 32-step physics trajectories. Its widths sum native timer durations across concurrent threads, not elapsed iteration seconds. It resolves the simulation pipeline through narrow phase, rather than individual C functions inside the collider.

## The critical path: physics first, then stragglers

For SO101, the sum of the slowest worker's reported compute phase over the 32 steps was **299.227 s**. The learner's remote-step total was **299.523 s**. The residual was only **0.296 s** across the entire iteration. That residual includes dispatch, serialization, reply placement and scheduling; it is not a pure wire-latency measurement.

The mean worker's compute phase summed to **197.622 s**. The **101.604 s** difference to the per-step maximum is an idealized balancing opportunity. Perfectly removing that gap would reduce the iteration toward **200 s**, approximately **1.5× faster**, with the existing physics. It is a bound, not a promised speedup: state complexity, CPU contention and synchronization cannot all necessarily be removed.

A single worker on **GPU71FM** was last on **all 32 SO101 steps**. A different worker on the **same host** was last on **all 32 NIST steps**. That is strong reason to investigate host/core placement, CPU frequency, memory pressure and contention as well as task allocation. It does not by itself prove which host mechanism caused the slowdown.

![Worker barriers](worker-barriers.png)

All 187 workers contributed step timing. The two reference workers were at approximately the **0.5th and 89.8th percentiles** of SO101 mean compute time, and the **1.1st and 29.9th percentiles** for NIST. Higher means slower. Thus the NIST stack samples do not represent its slowest worker; the all-worker telemetry, not those two samples, determines the reported critical path.

## Inside SO101 physics

Across the two sampled workers' integrated trajectories:

| Native phase | Fraction of stepping time |
|---|---:|
| Narrow-phase collision detection | **93.0%** |
| Constraint solve | 2.86% |
| Inertia projection | 2.18% |
| Broad-phase collision detection | 0.37% |
| Contact construction and remaining physics | 1.59% |

The native rollout occupied **98.0% of their worker compute-phase wall time**. Observation refresh, placement distances, success checks and Python bookkeeping were small. Loosening the constraint solver cannot recover most of the time: it occupies about 3%, whereas narrow phase occupies about 93%.

The original NIST backend keeps active MuJoCo data in thread-local rollout buffers. This capture's internal timer snapshot looked at unused model buffers, so its zero internal counters are **not evidence of zero collision work**. Its full-iteration measurements, outer native-rollout spans and worker stack samples remain valid. No NIST internal-phase percentages are claimed here.

## Which collision pairs are expensive

A **separate auxiliary CPU probe** ran 44 SO101 environments for eight control steps, using checkpoint weights and the same model signature. It does not replace the full PPO capture. A delegating native collider timer recorded geometry-pair time, call count and returned contact count. A before/after trajectory check was **bitwise identical**. No optimizer step or change to the live policy occurred.

| Pair | Collision time in this probe | Calls | Returned contacts |
|---|---:|---:|---:|
| `base_contact` ↔ `motor_contact` | **82.93%** | 28,600 | **0** |
| `male_contact` (screw) ↔ `driver_case_front` | **15.63%** | 650 | 1,950 |
| Everything else | 1.44% | — | — |

The base/motor pair averaged **2.65 ms per collider call**, despite returning no contacts. The screw/driver case averaged **21.93 ms per call**. These are thread-summed collider timings from the auxiliary workload, not percentages extrapolated to the whole training run. [Complete pair results and parity evidence](pair-probe/result.json).

The current SO101 collider uses these pair-specific search budgets:

| Search class | Starting points × search iterations |
|---|---:|
| General | 4 × 10 |
| Motor/base | **40 × 40** |
| Horn/motor | **40 × 40** |
| Free-object assembly | **40 × 40** |
| Grasp | 16 × 20 |

MuJoCo uses multiple gradient-descent starts to find contacts for nonconvex SDFs. See its [SDF collision explanation](https://github.com/google-deepmind/mujoco/blob/main/doc/programming/extension.rst) and [search-option definitions](https://mujoco.readthedocs.io/en/latest/XMLreference.html). SO101's global `sdf_initpoints=50` also reserves contact capacity; the custom collider selects the per-pair budgets above. Simply reducing the global capacity is not an equivalent performance change.

The first physics optimization to investigate is a **conservative early rejection for genuinely separated base/motor geometry**, followed by the screw/driver case. Keep required contact detection and high-detail mating behavior. Validate against frozen states, contact/safety checks and physical task outcomes before changing the live model. No such optimization was applied in this profiling task.

## Why the comparison is not a single-variable ablation

SO101 uses the frozen v13 source at `85c006a`, dual arms, all workcell parts and the driver, 448 points, 265 proprioceptive history values and 12 actions. Its bank contains 160 all-loose training states. Original AssembleNIST is isolated at reviewed upstream commit `eb5d15e004591d380c0d521da67e025ebee37703`: 20 NIST part/fixture tasks, a Franka arm, 320 points, 150 history values and eight actions. Its separate bank was generated with the inherited **640 states per task/family**, 64,000 training states across five families. Source/assets and training state were not mixed.

SO101 takes 80 physics substeps per 40 ms action. NIST takes 16 for coarse tasks and 80 for gear/nut tasks. An equal-task mixture would average 38.4; this rough factor of two cannot alone explain the observed 49× rollout gap. Exact task exposure and policy/start-state differences prevent interpreting that arithmetic as an isolated per-substep benchmark.

Both use MuJoCo 3.17.0, Torch 2.11.0+cu128, 8,192 × 32 transitions, five PPO epochs and four minibatches. Policy age, adaptive learning rate, task geometry and reset distributions differ. These are measured workloads, not a claim that one implementation alone causes every part of the gap or that either profile establishes learning success.

## Integrity, limitations and disposition

- The first asynchronous native-unwinding attempt stalled two sampled workers. Its interrupted iteration and failed export were rejected. The retry used safe wall-stack sampling and existing native phase timers. Checkpointed training progress was preserved.
- Raw `python.pstats` accounting is unreliable when combined with the Torch Python stack profiler; inconsistent self/cumulative times were detected. It is retained for diagnosis and **not used** in the findings. Attribution uses independent wall samples, explicit spans, worker telemetry, native phase counters and actual CUDA kernel intervals.
- These are instrumented captures, not a calibrated profiler-overhead experiment. Nearby unprofiled SO101 iterations were generally roughly 260–275 s with slightly different worker counts and states; that is context, not an exact overhead subtraction.
- The native phase graph is an actual-timer hierarchy, not unwound C stack samples. Resolving individual functions inside narrow phase requires another suitable profiler or targeted instrumentation. The separate pair probe provides targeted attribution without asynchronous unwinding.
- All remote computation, installation, builds, monitoring and analysis ran in Slurm allocations. No computation ran on the login node. The same GPU job was retained, all profiling holds were released, original 200-worker training configuration was restored, and baseline-only CPU leases and temporary worker steps were retired. Normal Slurm preemptions may still temporarily reduce live worker count.

Raw qualification records: [SO101](so101/result.json), [NIST](assembly-nist/result.json). Raw CPU/CUDA traces: [SO101 gzip](so101/torch-trace.json.gz), [NIST gzip](assembly-nist/torch-trace.json.gz). Open them in a local-compatible trace viewer; no upload is required by these artifacts. Speedscope files and standalone HTML views are included alongside them.

After profiling, a later GPU preemption exposed a coordinator recovery gap. The coordinator now waits for positively observed requeues of the same GPU job; unrelated same-lifetime failures still surface for inspection. The actual failed receipt and two failure guards were checked in [recovery validation](gpu-recovery-validation.json), and training resumed from verified checkpoint 31.

The viewer is now self-contained for sandboxed file previews: all flame trees and figures are embedded in the parent page. If an older preview remains cached, reopen [comparison-standalone.html](comparison-standalone.html).
