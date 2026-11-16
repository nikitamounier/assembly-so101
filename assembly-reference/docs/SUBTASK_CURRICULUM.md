# Balanced subtask training

**Goal:** Learn each placement with its prerequisites already assembled, then evaluate full assembly separately.

## Reset meaning

For each of motor, horn, screw, retainer and yoke, `loose` means the active part rests on the table while all preceding parts are correctly assembled. The receiving assembly remains dynamic, supported by native contacts. It is never welded. The other four families are pickup, in-hand, partial and near.

The shared native completion factory prepares prerequisites. The obsolete retainer-height workaround is removed. All-loose full-sequence banks and Quest's initial scene remain separate; they are not silently relabeled as single-subtask tests.

## Observations and sampling

The selected default uses 448 points: 128 active, 256 combined predecessor assembly/tool, and 64 distal fingers. The earlier weighted-context proposal is superseded. [Exact allocation and checkpoint compatibility](SO101_MATING_INPUTS.md).

SGS still samples individual entries by success history. Equal counts per task/family give equal initial probability mass; subsequent probabilities adapt. No equal-family quota or reward change is added.

The first production target is **256 TRAIN + 64 independent EVAL per task/family**: 6,400 TRAIN and 1,600 EVAL states. Expansion to 640 TRAIN per group is conditional on the first pilot's learning/coverage results, not an automatic claim that more data is sufficient.

## Generate

After native setup, from the repository root (use `.venv/bin/python` instead if installed through cloud bootstrap):

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MUJOCO_GL=egl \
  .venv-teleop/bin/python -m assembly_policy.so101.subtask_curriculum \
  --train-count 256 --eval-count 64 --workers 2
```

Optional `--grasp-source` paths supply fixed TRAIN grasp calibrations and productive variants. Missing calibrations can be discovered through native preparation; screw/yoke grasps use an actual close-and-lift. Calibration parameters may be shared across TRAIN/EVAL, but scenes/seeds are independent. Required runtime assets remain private; no legacy v9 bank is required by the generator.

The output defaults to `<physical-bank>_subtask_256` from the current shared configuration. A pilot or custom count requires an explicit, different `--out`. Its hidden `.building` sibling stores immutable plan identities, per-group accepted JSONL, failures and progress. Repeating the same command resumes completed work; a single-writer lock rejects concurrent writers to the same build. Changes to physical/validation/generation provenance require a new build. `--adopt-build` can transfer already-qualified rows only when physical and validator identities match. Partial or imbalanced data is never promoted to the final directory. Native identity changes require fresh validation; old v10 receipts cannot certify a new physical profile.

For coordination, run the generator with `--out PATH --stop`, then wait for its workers to exit and `build_status.json` to say `stopped` before updating physical source. `--resume-after-stop` resumes only when the original plan still matches. Low disk space stops at a candidate boundary and preserves progress. A recognized interrupted publication can be retried without losing its qualified rows.

Generation uses actual production actions, native fine-step guards, independent sampled original-CAD overlap checks, nonterminal starts and prerequisite retention. Held states require loaded bilateral grip and limited slip. Every partial state additionally replays at 0.25 ms. Invalid samples are rejected; successful poses do not establish a reachable insertion trajectory or hardware calibration.

Loose XY/yaw varies up to ±8 mm / ±8° for the assembly and independently loose parts. Measured in-hand means add up to ±0.02 rad arm-joint noise with fresh scene noise. Pickup and mating proposals retain their tighter grasp/alignment bounds. Nominal coupled arm/assembly placement is recorded separately from noise. Means, seeds, controls and qualification receipts accompany every state.

## Use and evaluate

SO101 training and the cloud launcher default to the dedicated `<physical-bank>_subtask_256` directory. They fail if it is absent or stale; an incomplete build is not a fallback training set. Original physical/full-sequence banks remain selectable with explicit `--bank-dir`.

Root `*_eval.npz` contains independent **subtask-loose** starts. `family_eval/*_eval.npz` contains all five holdout families. Evaluation defaults to subtask-loose; use `--families all` (or a `+`-separated selection) for separate family records, with `--trials` applied per family. Compare policies on held-out starts and report full-sequence outcomes separately.

Generation does not launch policy training. Current execution status and evidence are linked from [run readiness](RUN_READINESS.md). Historical v10 receipts remain separately preserved.

## Current execution status

The user accepts the v14 fall review and explicitly selects 0.5 mm penetration/material acceptance. The physical v14 model is unchanged; new validator identities include `limits.py`. Historical v9/v10/v11 proposals remain historical until replayed. The current v14 all-loose bank is separate from the staged bank.

The approved nearby-arm distribution in [plan.md](../plan.md) is implemented: open, noncontact unused arm near the assembly; both arms policy-controlled; both nearby for subtask-loose starts. The production generator now prepares nearby arms and rechecks full geometry at saved-state and every control-step replay pose. State/control digests bind both production and finer-step receipts. Fixed-layout grasp IK no longer translates or rotates the receiver to solve reach. Pilot v2 now contains all 50 qualified states, with 60 production/fine-step receipts and a complete native gallery. Strict preflight passes 31 checks and CUDA update/restore passes 13. A bounded training experiment is active on that pilot before scaling to all 6,400 TRAIN / 1,600 EVAL states. The full production bank is not yet built.

The first baseline uses tested nominal friction; the requested friction variation remains subsequent validation work. [Run readiness](RUN_READINESS.md) and [sectional intent](user-intent/sgs/USER_INTENT.md) are current. Old v10 coordination receipts remain under `artifacts_local/subtask-curriculum-v10/`.
