# Deep Warp profiling and scaling — November7

The user requests profiling and substantially faster Warp execution, including larger mixed batches. Implemented exact launch scheduling for the serial per-world no-slip kernel plus inactive contact-slot pruning.16-world backend3.06× faster; complete16-world environment7.41× versus one frozen native-v13 simulation thread.1024/2048batches reach56.7/62.0physics controls/s; doubling adds only9.3%throughput. GPU-event and actual Nsight traces identify the bottleneck shift from91.2%no-slip at16worlds to82.4%surface/SDF work at2048. Original tolerances, solver budgets, geometry and guards retained. Fixed-input component parity,25-stage contact audit, independent saved-trajectory safety and8valid-transition PPO pass. Live native-v14 unchanged; long-episode/full-bank/production qualification remains separate. [Full evidence, rejected trials and next priorities](../../../outputs/mjwarp-profile-20261107/README.md).

# Current Warp runtime request — November7

Build an optional MuJoCo Warp version of the current runtime, informed by the slowdown investigation. Reuse the qualified custom-contact prototype and integrate current reset payloads, per-world equality/friction/noisy hinge limits, unchanged observation/reward logic and substep safety screening. CUDA graphs reduce launch overhead; batched model arrays carry world-specific hinge ranges. Keep a distinct physics/checkpoint identity and validate through PPO in an isolated run. Live native training remains separate. [Implementation and measured limits](../../../outputs/mjwarp-runtime-20261107/README.md).

# Latest performance decision — November7

**Latest request and deployment:** Continue the accepted exact acceleration from `investigate slow-down`. Main has rebuilt both native libraries and launched isolated native-v15 from CURRENT native-v14 checkpoint64, preserving policy/Adam/SGS/reset-RNG/counters and all1143 reset rows. The accepted change hoists immutable grid lookup per tree query and shares interpolation for distance and a qualifying gradient, retaining clamping/arithmetic/fallback/contact order and prior separation/four-way cache.

Source matched two-world PPO99.6285→86.5508s is1.1511×,CPU160.705→137.447s; model/Adam/sampler/LR/counters and scientific metrics exact,120replayNPYarrays exact. Independent162000queries/54grids/inside+outside/thresholds bitwise equal. Main current-bank1143obs/reward/contact and compiled numeric arrays exact; actual10-env/10-thread current-policy61arrays exact,12.2110→12.0409s=**1.0141×**, peak8457.3MiB/lower than baseline. This production-sized panel shows only a small gain; do not claim15%fleet improvement. Main2worker/4envPPO2updates/16valid saved finite checkpoint with unstable0. [Release evidence](../../../operational/wowrobo-chimera-20261107/modal-live/native-v15/README.md), [source report](../../../outputs/contact-patches-20261107/README.md).

Rejected attachment-contact omission and calibrated/stiffer welds remain disabled: short load success was followed by2unstable policy transitions/5of5safety episodes versus baseline0/1of3. Both pruning prototypes failed performance selection. The accepted kernel changes no physical parameter/state schema; explicit profile/factory guards plus exact equality arrays exclude those prototypes. Preserve .5ms/80substeps/40msactions, measured target-hinge stops/noise, CAD/SDF/external/supplement/friction/search/solver/guards/actions/points/rewards. Original.25ms profiling source frozen; bare-v8 never imported. Native-v14's actual live51proof and earlier modest speedup remain historical; larger-bank admission remains separate, not silently imported.

## Prior experiments and decisions

# Simulation speed and engine comparison

**Goal:** Aim for 10–20× faster complete PPO iterations at fixed valid-transition count and physical acceptance. Preserve material exclusion, grasp/support, passive hinges, friction, insertion and jamming behavior. Compare equivalent geometry, initial states, reward conditions and action/observation contracts.

## Historical approximate-physics investigations (not selected for live training)

The user now authorizes an **open-channel collision revision**: preserve the same assembly tasks and actual CAD inputs, replace solid envelopes around mating geometry with simpler shapes that retain openings/pads, restore inter-part collision, and test grasp retention/insertion and speed. Build a separate candidate; preserve the main-run owner’s native rollback and live resources. The existing bare-v1 task objective alone does not establish physical insertion behavior.

The latest request changes the acceptance target: keep the **same five assembly tasks**, original reward/settling, actions, reset bank and detailed CAD point inputs, but test a separate bare MuJoCo simulator with deliberately approximate contacts. Do not substitute reaching, posture-following, dense movement rewards or a different task. The user explicitly accepts rough movements and removing penetration/velocity episode termination in this simulation-only experiment. Keep numerical failure diagnostics. Use few primitive collision shapes, stock MuJoCo without custom SDF/supplement hooks, fewer substeps and cheaper solver settings; precise contact behavior can be restored later. Leave the existing live run unchanged while this separate experiment is tested. Report which contact interactions are omitted and distinguish approximate seating success from physically valid assembly.

The earlier full compound approximation failed strict contact qualification and the conservative variant gained only1.026×; those results remain in [the collision report](../../../outputs/simple-collision-20261107/README.md). They do not prevent this newly authorized relaxed-physics experiment.

The bare assembly pilot now preserves all29tested initial point/reward snapshots and measures about2,946×less single-world stepping time (different physics), with a100-update/51,200-transition original-task PPO smoke test completed in50.6s. Held-out success remains0/40before andafter: speed/integration is demonstrated, learning improvement is not. Rough grasp drift remains. [Implementation, omissions, real clip and results](../../../outputs/bare-assembly-20261107/README.md). The bounded pilot has ended; the native live run remains unchanged.

The earlier exact surface-tree implementation remains available as a separately qualified modest optimization. User authorization includes isolated CPU probes and uploading necessary code/data. This performance fork does not own the parent run's monitor, cloud scaling, learner, or dashboard deployment.

## Original native measurements and earlier implementations

MuJoCo4 native CPU physics feeds one CUDA learner. The active frozen WowRobo v5 audit found updates10/11 collecting for~10minutes and optimizing for1–2seconds. More CPU worlds increase valid samples/update;32 sequential control rounds still determine update latency. Four isolated current-model probes attribute98.7–98.8% of native stepping to narrow-phase collision. A held-state interval costs8.30seconds:3.30s ordinary SDF search and4.88s surface supplement. Existing BVH mode1 was already enabled.

The first native optimization is implemented in `assembly-so101/plugin/so101_surface_tree.h`. It changes bounds/traversal storage only. The frozen uninstrumented30-case panel measures1.05256× speedup (about5% less simulation time), with every measured case faster. Across54 control intervals/8640 native substeps,972 arrays are byte-identical, including full trajectories and final contacts, forces, observations and reward/episode results. The exhaustive oracle passes87 original/perturbed poses against two-thread traversal (435 byte-identical arrays, up to243 contacts). All7 existing surface-acceleration regressions pass. These are isolated qualification results; production worker/whole-iteration validation and a versioned release remain outstanding. No live release has been deployed by this fork.

Evidence: [candidate implementation and qualification](../../../outputs/update-speed-audit-20261107/CANDIDATE.md), [current-run timing audit](../../../outputs/update-speed-audit-20261107/README.md).

## Next decisions and limits

The measured bounds optimization is a modest improvement. Larger gains require reducing surviving supplement work and ordinary SDF search with equally conservative contact/replay checks. For the exact-physics track, retain physical qualification. The separate relaxed-physics experiment above explicitly permits simpler contacts, fewer substeps and cheaper solver settings. The separate0.5ms/80substep version has since passed bounded grasp/insertion and distributed-PPO gates and reached native-v10 checkpoint31; its combined fleet performance remains unmeasured. A16-step rollout pilot remains a separate PPO batch/bootstrap tradeoff.

AssembleNIST's reported training throughput uses simpler task-specific scenes and16/80 native substeps versus the current full bimanual model's160. It is a useful architectural reference, not a controlled speed ratio. Measure actual valid transitions/sec, critical-path latency, memory and parallel scaling before claiming a full-iteration improvement.

AGX, Isaac and GPU physics remain unqualified alternatives. Check APIs, hardware, storage and licensing, then evaluate the same correctness cases where feasible. No backend switch or paid license is authorized by this native optimization request.

Historical screwdriver-removal and v13/v14 profiling decisions remain in `MEMORY.md` and their evidence reports; they are not current-model throughput measurements. The audit also found the dashboard selecting stoppedv3 whilev5 trained; dashboard correction belongs to its owner.


## Chimera H100 MuJoCo Warp probe — November7

The user authorized testing Warp. The isolated H100/CUDA12 probe and synthetic contact benchmark pass, and frozen WowRobo accepted-v5 imports with matching identity. Actual assembly GPU physics remains unqualified: its54custom SDFs, supplemental contacts and per-world friction are not ported. No assembly throughput or10–20× gain is established. Preserve physical acceptance and current training. See [measured results](../../../outputs/mjwarp-h100-20261107/README.md).


## Actual assembly Warp port — November7

Port the original custom SDFs, supplemental contacts and per-world friction to Warp, then test actual assembly starts on a spare Chimera H100 without changing the main run. Completed optional backend and12bounded accepted-v6 checks: zero sampled SDF/gradient and supplemental-depth error; peak penetration0.17355mm against0.5mm guard.32-world cover physics reaches8.60× one native CPU thread, or10.24× with private dense Jacobian storage. These are physics-only single-trial replicated-start results, not complete-PPO or fleet speedups. Ordinary engine contact manifolds differ; full-episode/PPO qualification remains pending. Main native learner continued through iteration19. [Implementation, evidence, limitations and reproduction](../../../outputs/mjwarp-port-20261107/README.md).


The user also requests viewing the actual simplified collision shapes at http://localhost:8878. Implemented54tested-model boxes as an independent, initially visible **Collisions** layer with CAD/points initially hidden (CAD can be toggled back on) (bluehardware/greenpads/orangeparts). Shapes follow state/variation poses; existing CAD/points and training data remain unchanged. This display does not requalify native snapshots under approximate physics.

The collision-only view uses the actual tested model geoms, currently boxes. Part-part contacts are disabled in that bare prototype, so a visible solid box does not automatically block receiver insertion; it can allow pass-through. The user asks whether slide-ins will all fail. Do not claim impossibility or successful insertion learning: measured pilot0/40, rough grasp drift, and no physical insertion/jamming interaction. A proposed next version should preserve channels/bores and pads using a few primitives and restore inter-part collision while retaining bare MuJoCo; this is not yet implemented or measured.


## Original-native flame graph — November7

User requests actual original-native profiling before new speed changes, with an interactive flame graph. Completed isolated32-round/two-worker PPO from frozen25/966743:64valid transitions in194.90s,0.256sPPO, zero instability. Two worker sampled wall graphs and full native engine/pair timer hierarchy; source/model8c4783.../.25ms160substeps preserved. All72arrays byte-identical with instrumentation. No live/default physics mutation. [Viewer, measurements, source identities and reproduction](../../../outputs/native-flame-profile-20261107/README.md).
