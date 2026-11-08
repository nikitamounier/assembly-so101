# Fresh native hinge-limit profiling — November 7, 2026

Matched saved-state panel on Chimera GPU720E, existing allocation3598717, two Slurm logical CPUs (sibling threads2 and98 on socket0/core27; one physical core), two worlds/two native threads. Every variant uses0.5ms physics and80substeps per40ms action. Versions: v10 original native collision kernel after timestep correction; v12 prior conservative collision optimizations; v13 measured servo2/3 hinge stops with saved ±0.05degree endpoint variation and native friction hook. This is isolated benchmarking; no live checkpoint/process/CPU role was changed.

| Stage | v10 s | v12 s | v13 s | v10/v13 | v12/v13 |
|---|---:|---:|---:|---:|---:|
| 00_loose | 0.9923 | 0.1207 | 0.0127 | 78.04× | 9.49× |
| 01_cover_pickup | 1.2342 | 0.2610 | 0.1156 | 10.68× | 2.26× |
| 03_cover_aligned | 1.3783 | 0.2824 | 0.2113 | 6.52× | 1.34× |
| 04_cover_entry | 2.8867 | 0.7122 | 0.6308 | 4.58× | 1.13× |
| 05_cover_near | 2.3354 | 0.6729 | 0.5991 | 3.90× | 1.12× |
| 10_connector_near | 3.0608 | 1.0512 | 0.9590 | 3.19× | 1.10× |
| 15_pin_near | 4.1828 | 1.6428 | 1.5214 | 2.75× | 1.08× |
| 21_upper_far_side | 6.1267 | 3.2370 | 2.9747 | 2.06× | 1.09× |
| 31_final_near | 9.3786 | 4.4751 | 4.1818 | 2.24× | 1.07× |

Sum of stage medians: v1031.5757s, v1212.4554s, v1311.2064s. Combined gain2.818× vs v10; hinge replacement incremental gain1.111× vs v12. This weights the nine stages equally and is not a production SGS-weighted throughput estimate.

The new limits accelerate loose states strongly (9.49× beyond v12), but expensive contact stages improve only7–13% beyond v12. Forearm remains about329× slower than loose in this panel, and upper-arm about234×. Thus the earlier14.6× loose-state geometry probe does not imply a14.6× training gain.

## Panel protocol

- Five base-cover stages, plus connector-near, pin-near, upper-far-side and forearm-final-near. Two saved worlds per stage: nominal and first friction variation where available; assembly-pin has no distinct friction variation in this bank and duplicates nominal. v13 worlds carry independently saved hinge noise. Original162-column state prefixes remain unchanged.
- Four reset-and-step repeats per stage. First treated as warmup; report median of the other three (two deterministic-small-action runs and one zero-action run). Raw distributions, reset/rollout/refresh spans, contacts, friction and safety flags are in panel.json per version.
-36calls/72world-intervals/5,760physics substeps per version; all finite with zero native warnings, safety events or unstable flags. Reported warm timing uses4,320physics substeps per version. Startup/native compilation excluded.
- Separate fresh processes and private native compilation directories; same CPU binding[2,98] and thread limits. Identities and source hashes captured beside each panel. Changed physical stops mean v10/v13 trajectory equality is neither assumed nor required.

## Actual-policy PPO and native phases

| Version | First update s | Reverse-order repeat s | Mean s |
|---|---:|---:|---:|
| v10 baseline | 205.0363 | not repeated | 205.0363 |
| v12 collision optimizations | 128.7287 | 127.3365 | 128.0326 |
| v13 hinge limits + optimizations | 140.0481 | 118.8145 | 129.4313 |

One actual32-step PPO update per run,64/64valid transitions. CPU learner, two local native worlds/two threads, same archived native32 policy+Adam. Checkpoint counters retained; identical fresh seeded diagnostic SGS/reset initialization. This is not production-fleet throughput or exact training continuation. Startup/native compilation excluded; first-run initialization took about34seconds. The two optimized versions were run in both orders to investigate the initial apparent regression.

- Against v10, mean v13 throughput is1.584×. Against v12, two-run means differ by1.1% (v13 slower), while v13 itself spans118.8–140.0seconds (~18%). **These runs do not demonstrate an incremental full-PPO throughput gain from hinge replacement.** The nine-stage fixed-reset panel does show an11.1% incremental gain. Two replicates are insufficient for a robust production estimate.
- The reverse-order pair favored v13:127.336→118.814seconds,1.072×. Process CPU time206.404→192.736seconds,6.6%less. Per-thread cache locality, worker scheduling and host timing variation are plausible explanations for unstable wall timing; this profile does not isolate their individual contribution.
- v10 collection204.618seconds vs optimizer0.340seconds; v13 first collection139.641seconds vs optimizer0.333seconds. Python policy/optimizer optimization cannot explain the dominant cost here.
- All scientific metrics agree exactly across all five runs, including losses, std, task/episode counts and the same1safety termination among3episodes; numerical instability0. Initial three saved checkpoints have exact model, Adam, SGS, lr and counter equality. Repeat scientific-metric equality is separately recorded. No learning-quality gain is inferred.
- Native phase timers put v13 collision at97.37% of upper-arm and97.90% of forearm native step time; constraint solver1.75%/1.30%. Loose collision19.67%, solver30.68%, with the whole loose interval already cheap. Phase-clock instrumentation is a separate pass and not used for primary throughput numbers.

## Remaining native pair costs

Fresh v13 two-world replay, separate private timing wrapper (includes native rollout and final refresh):

| Stage | Distinct SDF pairs | Ordinary SDF fraction | Supplement fraction | Time in always-empty final-result pairs |
|---|---:|---:|---:|---:|
| Loose | 0 | — | — | — |
| Cover near | 8 | 66.6% | 33.4% | 57.2% |
| Upper far side | 32 | 61.6% | 38.4% | 52.9% |
| Forearm final near | 40 | 57.9% | 42.1% | 38.1% |

Every invoked SDF pair still has40initial points ×40iteration budget. Always-empty means every observed final result was empty during this replay; it is not a proof that future contacts are impossible.

Forearm's top two pairs are shoulder_connector_printed_0/1 ↔ servo_2_case, together26.5% of pair time, principally surface supplements. These are assembly mating contacts and are distinct from the removed same-arm servo/printed-part internal stops. Other cost is spread across many helper/assembly pairs. The next substantial acceleration target is those remaining empty-pair searches plus positive assembly-contact supplements; removing the three internal pairs has already made loose states cheap. Native solver, PPO optimization and policy-point count are not the dominant remaining target in the heavy stages.

The private instrumentation leaves40output arrays exactly equal to an independently run plain v13 wrapper. All1,143bank rows have identical original162-column state prefixes across releases; v13 adds four small endpoint offsets. Source files remained unchanged throughout measurements. See audit.json and per-version identities/build receipts.

## Flame graphs

[Open the zoomable flame graph](http://127.0.0.1:8896/flamegraph.html), or open flamegraph.html locally. It reuses the original profiler's self-contained viewer, with9views: sampled Python PPO/learner stacks for all three versions; measured native collision-pair hierarchy; native engine-phase timer hierarchies; and actual PPO wall-span hierarchies. Local native workers execute within the learner process and are opaque in Python samples. Native graphs are **timer-based attribution, not unwound C/C++ stack samples**. Pair timers sum native threads and include end-step refresh; they are a separate pass and must not be added to phase/PPO times.

HTTP200 and JavaScript syntax verified. Browser visual verification was unavailable because the desktop browser bridge timed out; no visual verification claim is made. Final optimized repeats are included in raw receipts; sampled dropdown captures are the initial runs, explicitly distinguished from the mean cards.

Compared with the prior original profile (~1,002seconds), this run took about31minutes because it added three source versions, nine-stage/four-repeat panels, three actual PPO updates plus two repeats to resolve timing ambiguity, separate native phase panels and plain/instrumented pair attribution. Two early harness setup failures and about60seconds of unavailable browser-bridge checks also contributed. No further profiling scope is pending.

## Reproduction and failures

`profile.py ROOT OUT v10|v12|v13 panel` runs the panel; use a fresh isolated OUT directory for PPO and set PYTHONSAFEPATH=1 to prevent the panel filename from shadowing stdlib profile; `ppo.py ... ppo` runs the training diagnostic; `timers.py ... panel` enables native MuJoCo phase timers in a separate process, keeping instrumentation out of main panel times. `run_next.py` records exact srun/environment arguments. Sources and receipts remain under ROOT/native-contact-optimization-20261107/hinge-profile-20261107.

- Baseline initial harness import assumed every release had a private backend source; v10 instead references the frozen canonical backend. Corrected before any measurements.
- First PPO attempt failed before optimizer creation because the harness profile.py shadowed Python stdlib profile through Torch/cProfile. Fixed by excluding the harness directory from sys.path; failed source and log retained.
- One SCP receipt transfer had an explicit connection-reset failure. Retrieved only missing read-only receipts; no benchmark was restarted for that transfer failure.

The historical1.72× CPU-slot ratio used a different all-loose SO101 workload and cross-capture core counts. This run does not establish a new NIST ratio; a same-host, same-core, matched workload NIST benchmark would be needed.
