# Current training handoff publication

**Goal:** Publish the current five-task shared environment with enough verified setup to let a collaborator start a run.

Published source is `Alessandro-Maniscalco/robotics-assembly-3.0`, branch `main`, implementation commit `eea45cd8c709ea9058c62889e36de0d2544dbcd2`. Push and remote-head equality are verified. The profile remains `shared-side-v8-five-tasks`: motor, horn, screw, retainer and yoke. The unused arm link is absent. Mac/Quest setup, A completion/reset controls, minimal HUD, owner-input recovery and shared native reward/contact fixes are included in this publication. See [collaborator commands](START_RUN.md).

## Verification

- The pinned private `native-runtime-v1` release asset is present, 97,127,742 bytes. Bootstrap installs its hash-bound derived geometry and generates the new versioned banks/reference; the collaborator needs authenticated access to the private repo/release, not original CAD.
- All 320 v8 starts are generated: 32 TRAIN and 32 disjoint EVAL states per task, zero rejected states. Frozen native reference matches the current shared source.
- Actual native preflight passes 29 checks, using the existing learner-debug allowance for the known screw-pitch task-mechanics failure. Safety, provenance, source-preservation, timeout, point/control and driver checks remain required.
- The preflight formerly compared the full half-conversion source literally, rejecting the Mac portability shim. It now normalizes only the two exact approved header/conversion additions. Mutation tests confirm changed thread equations, grid evaluation and half conversion still fail.
- 18 learner/source tests pass, covering the guard, PPO edge cases, sampler, checkpoint/resume, launcher signals and assessment. Six workcell tests pass; one optional absent old-bank fixture is skipped. All 21 affected browser cases pass. The preceding physical update's 38 native cases also pass; see [removal evidence](UNUSED_LINK_REMOVAL.md).
- A real CPU smoke passes finite losses, actor/critic gradients and weight changes, normalization, optimizer state, SGS RNG, atomic checkpoint, exact saved learning-state restore and resumed counters. It uses the v8 bank, with a separate Mac CPU PyTorch 2.14.1 environment. This is not CUDA verification of the pinned Linux wheel. The collaborator must run the documented GPU smoke before training.
- The one-hour example launch command is verified with the real launcher's dry run. No GPU run or remote job is started during publication; the existing Quest session is not restarted.

Local evidence remains ignored under `artifacts_local/publication/`: `prepare.log`, `preflight.json`, `preflight.log`, `learner-tests.log`, `workcell-tests.log`, `browser-tests.log`, `cpu-smoke/report.json` and `run-command.txt`. Generated banks/reference, private CAD, binaries, logs and runs do not enter Git. The existing geometry release is reused; no new release is needed for this source change.

## Files changed

The preceding [Mac session](MAC_QUEST_SESSION.md), [controls recovery](QUEST_CONTROLS_RECOVERY.md), [reward-state change](REWARD_STATE_UPDATE.md) and [unused-link removal](UNUSED_LINK_REMOVAL.md) reports provide per-file inventories for the accumulated implementation. Historical link extraction tooling is optional and is not called by current setup.

This publication additionally changes:

| File | Modification |
| --- | --- |
| `scripts/so101_source_checks.py` | Preserve upstream source checks while allowing the exact portable header/half-conversion shim. |
| `scripts/so101_preflight.py` | Use the source guard; derive task counts from the canonical list. |
| `tests/test_source_preservation.py` | Check the actual upstream/current source and reject physics/conversion mutations. |
| `docs/START_RUN.md` | Give fresh-clone, existing-clone, setup, GPU smoke, bounded training and evaluation commands. |
| `README.md`, `docs/ORACLE_HANDOFF.md` | Point to current v8 profile and collaborator commands. |
| `docs/CURRENT_ENVIRONMENT.md`, `User Intent.md` | Record current data/native/CPU verification and requested publication. |
| `MEMORY.md`, `TASK.md`, this report | Record handoff decisions, verification and publication status. |

Full loose-part assembly remains unlearned. Motor press-fit, actual screw driving and physical-kit calibration remain unresolved; CPU/software checks do not establish task learning or real-hardware readiness.
