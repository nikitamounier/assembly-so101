# Publish the current training handoff

**Goal:** Push the current five-task shared environment to canonical main so a collaborator can start a GPU run.

## Approach

Review pending source, verify runtime availability and native/learner startup, commit project changes, push main and verify the remote head. Keep original CAD, generated data/runs and credentials ignored.

## Progress

- Remote main matches the local base; pinned private geometry release is available.
- Current v8 environment preserves five validated A tasks and removes the unused link. Prior 38 native cases pass; fresh publication adds 18 learner/source cases, 6 workcell cases and 21 browser cases.
- Generated all 320 independent TRAIN/EVAL states and reference. Native preflight passes; CPU learning/optimizer/checkpoint/restore smoke passes. CUDA qualification remains for the collaborator’s GPU.
- Fixed preflight to accept only the exact Mac half-conversion/header shim while continuing to reject SDF/conversion equation changes. Added docs/START_RUN.md.
- Completed: source commit eea45cd8c709ea9058c62889e36de0d2544dbcd2 is pushed to origin/main and the remote head is verified. Current collaborator commands: docs/START_RUN.md. Existing Quest host and training jobs remain untouched.
- Follow-up: collaborator with private repo/release access runs bootstrap and GPU smoke, then a bounded run. Local CUDA hardware is unavailable.

## Policy-input explanation

- User requests an understandable visualization of the policy's points.
- Exported actual 320-point production tensors from a frozen live state for all six tasks; exact reconstruction checks pass. No service or scene mutation.
- Completed interactive point/group/frame inspection. All six task coordinate checks and browser interactions pass; light/dark layouts pass at 736/360/320 px without horizontal overflow or JavaScript exceptions. See `docs/POLICY_INPUT_VIEW.md` and ignored `artifacts_local/observation-viz/`.

## Paper-based observation review

**Goal:** Compare AlphaNIST with the current five-task adapter, verify point placement and propose denser, more useful sampling plus a clear runtime task-switch contract.

- Read the primary AlphaNIST paper and independent AssembleNIST sampler. Current v8 base receives 13 context points; the historical visualization has 12.
- Independent point-to-native-surface checks pass on frozen live state and asymmetric arm/jaw poses. Palm and TCP origins differ by 98.445 mm; no transform defect found in these checks.
- Completed the primary-paper/independent-code comparison, native audit, proposed 512-point interface-aware budget, fixed offline coverage sampling, controlled comparisons and state-preserving task-switch contract. Production sampler, live Quest and learner are unchanged. Evidence and per-file inventory: `docs/OBSERVATION_SAMPLING_REVIEW.md`.
- Follow-up if selected: version/implement and visualize candidates, verify region coverage, then compare native learning outcomes. Continuous sequencing remains proposed.
- Gripper follow-up complete: upstream uses 64 finger-only samples for one gripper. Withdraw the 128-gripper proposal; compare better-placed 32/64-total samples on our two grippers. Revised reference budget and inferred contact-geometry rationale are recorded in the review; production inputs remain unchanged.

## Standard NIST/Franka visualization

- Completed the requested original setup with one Franka, the NIST board, all 20 tasks and unchanged 128/128/64 samples from original seeded loose resets.
- Native geometry/FK checks and all task/view/point controls pass. Light/dark layouts pass at 736/360/320 px; screenshots inspected.
- No production sampler, shared SO101 scene, live Quest or training change. Source/receipts/file inventory: `docs/NIST_POINT_VIEW.md` and ignored `artifacts_local/nist-point-viz/`. No follow-up required for the view.

## Implement part/mating observations

- User authorizes replacing all-object context with task-local mating geometry and improving finger locations, then showing SO101.
- Completed production sampler, receiver crops, coverage sampling, distal fingers, checkpoint guards and updated five-task view. Counts remain128/128/64; old checkpoints are rejected.
- Validation:27 tests pass,1 optional skip; native two-pose surface audit and all five browser task/coordinate checks pass. Pair/finger images inspected; no physics/Quest restart or push.
- Follow-up: fresh training/performance assessment; tool-aware observation limitations documented in `docs/SO101_MATING_INPUTS.md`.

## Full mating and screwdriver follow-up

- Completed latest correction:256 samples across full receiving bodies, no local crop;64 screwdriver samples for screw only. Distal finger coverage unchanged.
- Network uses512 slots, with zero-padded tool block outside screw; updated SO101 training default and view. Standard NIST default remains128 mating.
- Suite28 passed/1 skipped, plus new full-base-coverage regression; five observation tests pass. Native screw/tool two-pose audit and browser checks pass. Changes local; fresh training remains necessary.

## Assembly-order correction

- Mating now includes the base and **all preceding parts** in canonical motor→horn→screw→retainer→yoke order. The total256 mating samples is shared across the full predecessor assembly.
- View uses native-qualified predecessor prepared states with active part loose, instead of reusing one frozen scene for every task. No curriculum/live-scene change.
- Canonical role/order regression and full suite pass:29passed,1optional skip. Fresh observation signature rejects old role layouts; no push/restart.

## Small-fastener mating allocation

- Completed latest correction: replace equal per-body quotas with area-weighted allocation and four-sample presence floor. Screw receives5 mating points rather than51/64; total remains256.
- Updated prepared-stage SO101 view displays per-body counts.31tests pass,1optional skip; production signature changes guard older policies.
- Files/evidence: docs/SO101_MATING_INPUTS.md and ignored artifacts_local/so101-points-v5/. Fresh training remains necessary; no live service or physics change.

## Combined mating budget correction

- User clarified assembly+driver must total256, not256each. Implemented256(default) and128 presets; screw splits50/50, other tasks use full budget for assembly. Active128/fingers64 unchanged; totals448/320, no padding.
- Current view has preset selector, same predecessor poses and full coverage. Tests and paired browser checks recorded in artifacts_local/so101-points-v6/. Old role/budget proposals are superseded. Fresh training remains necessary.

## Publish selected256 preset and visualization

- User selects256combined mating for use;128remains a comparison. Standalone HTML and download/open instructions added under docs/visualizations/.
- Preserved and merged newer remote v9 tool/reward work; approved figure is labeled a frozen v8 inspection snapshot.
- Final merged validation:47tests pass,1optional skip;320v9starts/reference generated without rejection;13CPU learner/restore checks pass. Export works via file:// with both presets/all tasks.
- Completed: source push through fb8f697 verified against origin/main; GitHub visualization blob matches local committed HTML (523068bytes). No live service or running training job changed. Follow-up: fresh training/CUDA smoke on the chosen host.
