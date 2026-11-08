# Bare MuJoCo assembly pilot — November 7

This separate experiment keeps the **original five assembly tasks**, original sparse seating reward and settling requirement, original 12 actions, reset poses and original detailed CAD point inputs. It does not substitute reaching or introduce a dense movement reward. The current native training run is unchanged.

The user explicitly accepts rough simulation and removing penetration/velocity episode terminations for this experiment. The earlier strict proxy-geometry qualification is not its acceptance gate.

## What is simpler

- Stock MuJoCo only: zero SDF plugins, no custom surface supplements, no native friction callback.
- One primitive box per former collision mesh; four distal finger-pad boxes use the actual CAD finger samples rather than full gripper housing bounds. Visual meshes and reward reference frames remain intact.
- Table/object and finger/object collisions remain. Robot self-collisions and assembly-part versus assembly-part collisions are disabled. Objects can therefore pass through other assembly parts; precise insertion resistance/jamming is deliberately absent.
- Timestep2ms,20substeps per unchanged40ms control interval, versus0.25ms/160substeps. Solver10iterations, pyramidal friction and no no-slip iterations. Approximate contact friction is constant0.8; stored episode friction multipliers are not applied by a custom callback.
- Penetration and velocity limits do not end episodes. Numerical nonfiniteness, time-reset/solver warning diagnostics remain. Original task success, action cost, episode horizon and stable-seating logic remain.

The opt-in builder/adapter is `assembly_policy/so101/wowrobo_bare.py`. It copies the installed environment class into an isolated namespace and makes two explicit edits removing the physical guard blocks; missing source anchors raise rather than silently changing an incompatible trainer. The executed environment source is saved as `original-task-bare-env.py` remotely. No installed trainer files or global production hooks are edited.

## Speed and contract probe

On GPU105E,29task-conditioned starts, five reset repeats per start, median full `env.step` time per start:

| Panel | Sum over29starts |
|---|---:|
| Prior exact-native baseline |158.863s|
| Bare MuJoCo |0.05393s|
| Ratio |about2,946×|

Individual bare steps take roughly1.4–2.3ms. These are deliberately **different physics**, and this is simulation/control-step throughput, not an accuracy-preserving or whole-PPO speedup. The native panel was measured earlier on the same host; neither model loading nor initial point encoding is timed. See [probe](probe.json).

All29initial encoded point inputs are byte-identical to the original capture. All29initial seating-reward snapshots match the original review model exactly. Same point signature `faece0010eb71bc4d2ab7351d67d4e23935d0545aa7d1cc89e308fe50baaa70d`; reset payload remains162values. Physical scene has its own new SHA in [scene receipt](scene.json); it must never be labeled the original native model.

Five near-insertion starts each run64control intervals with small real arm commands and then holding,320intervals/12.8s aggregate simulated time. No numerical failures or guard terminations. This does **not** establish retained grasps or task learning: pieces drift and the controls do not achieve seated success. The initial coarse housing-box version was replaced with distal pad boxes; the recorded final probe is the pad-box version.

## Same-task training test

A bounded32world/16step PPO pilot uses the original SimBa actor/critic, original sparse assembly reward and the accepted-v6 TRAIN bank (1143task-conditioned rows). It starts from a frozen copy of the compatible current policy weights; optimizer and SGS state start fresh. No live checkpoint is modified.

Before/after evaluation uses the same40frozen EVAL witnesses (eight per task), selected deterministically with source archive hashes verified. They are packed with the original integration-to-reset conversion; TRAIN/EVAL are not mixed. The TRAIN bank has no `_eval.npz` files, so EVAL comes from the existing `v6/accepted-states-v6` frozen witness snapshot. The trainer's experimental reset flag is explicit and run status is `APPROXIMATE_ASSEMBLY_PILOT`.

The pilot completed **100 PPO updates /51,200transitions in50.617seconds**, averaging**1,012transitions/s** across32worlds. All updates have finite logged metrics, no numerical failures and no physical safety terminations. This is a small512-transition update budget, not the live fleet batch size; do not infer a whole-PPO speedup from the2,946×single-world physics ratio.

The compatible source policy was main-run iteration25/966743transitions; frozen checkpoint SHA`364f5fb94996103bece71bd1b13c9c2df95beefc6faf6a0c4551225e32ed7687`. The original held-out task reward is unchanged: **0/40success before and0/40after**. Training also produced no positive assembly success. This demonstrates fast finite simulation and PPO integration, **not learned assembly or a learning improvement**. Measured grasp/part drift is a limitation of these crude contacts and is the next geometry issue to improve while retaining the same task.

Evidence: [pilot summary](pilot-summary.json), [100 update records](pilot-log.jsonl), [before](before-eval.json), [after](after-eval.json), [held-out archive identities](evaluation-split.json), [executed pilot source](pilot-executed.py). [Actual saved policy rollout video](same-task-pilot.mp4) shows original visual CAD, backed by recorded `qpos`; it is not a successful assembly demonstration. Saved after-evaluation states are [here](after-eval.npz). Checkpoint remains isolated under remote `bare-assembly-20261107/pilot-100/checkpoint.pt`. The bounded pilot is finished; no background training is left by this experiment.

## Reproduction and ownership

Remote root: `/large_storage/goodarzilab/nikitamounier/assembly/bare-assembly-20261107`.
Frozen source root: `/large_storage/goodarzilab/nikitamounier/assembly/wowrobo-diagnostic-20261107T205745Z-minimum24-accepted-v4`.

`probe.py ROOT OUTPUT` constructs the model from the frozen native runtime XML and original review/point tables, verifies input/reward contracts and runs the speed/finite-motion probes. `pilot.py ROOT OUTPUT` runs the bounded original-task pilot and held-out checks. `render.py OUTPUT` renders stored actual policy qpos when offscreen GL is available. The model and assets remain remote; local JSON/NPZ metrics accompany this report.

All tests use the verified idle retained3598717/GPU105E with4CPUs and16GiB, while the native learner was observed on3606264before these tests. The final step inventory no longer lists its earlier3606264.13step; this experiment did not operate that learner/controller or alter either allocation lifecycle. Before each launch, the held allocation must show only batch/extern steps. Do not reuse these allocation IDs blindly. No jobs are submitted/cancelled/requeued, no cloud pool changes, no reset-bank or main-run changes.


## Collision boxes in the live local viewer

User-requested display at http://localhost:8878 now has an independent **Collisions** toggle (on by default, CAD and policy points initially hidden). All54boxes come from the tested modelMJB via `export-boxes.py`, expressed relative to their original CAD visual frames. Blue=robot hardware, green=distal finger pads, orange=assembly parts. They follow every saved nominal/variation pose independently of CAD and policy points. This is a display overlay; native snapshot qualification labels still describe the original admitted reset geometry, not the bare-box contact approximation.

Frontend implementation: `assembly-so101/web/src/collision-boxes.js`, with small integration edits inviewer.js/app.js/index.html; generated payload `web/public/collision-boxes.json`. Static frontend rebuilt and served by the existing8878process without restart or bank changes. All15existingfrontendtests pass;54anchor/layer/pose/disposalchecks pass;liveHTTPpayloadexact. Browser verified loose→05covernear, CAD/points/box toggles, no console warnings/errors. [Actual collision-geometry view](collision-geometry-viewer.jpg).


The collision-only view shows actual exported geoms, not separately fitted viewer bounds. The present model genuinely uses boxes. **Part-to-part collision masks are disabled**: apparent overlap between orange assembly pieces does not create an insertion-blocking contact. Thus slides can geometrically pass through receivers; this simulator cannot yet teach physical insertion resistance or jamming. It is not established that every slide must fail, but the pilot yielded0/40success and measured grasp drift. The next useful same-task simplification is a few primitives arranged around actual channels/bores and grasp pads, with inter-part contacts restored, rather than solid envelopes filling openings. No such follow-up has been implemented or benchmarked here.
