# Five-subtask reward validation

**Goal:** Verify that each correctly placed part earns native policy reward and incorrect placements do not.

Pulled canonical main through `076c5ca` on November 6, 2026, then rebuilt the native plugins on Linux. The physical profile remains `shared-side-v8-five-tasks`, signature `bd1478216658d1061a078051b3561332b8211bfea005ac0b1f90332e2c5c4cd4`. No geometry, physical settings or running simulation/training instance was changed.

## What earns reward

The active task earns **+1 minus a small control cost**, then terminates successfully. Before completion there is no positive task bonus; approaching the target alone earns none. Optional safety penalties remain separate.

All active and predecessor parts must match their CAD-relative targets within **1 mm**, including orientation through placement keypoints. Targets move with the base, not a fixed world coordinate. The active part must be released, avoid deep contact and remain within **0.25 mm / 0.01 rad** of its stability reference across eleven samples spanning **0.4 s**. The screw additionally needs native thread and head/horn bearing contacts. Its axial keypoints intentionally permit rotation about the screw axis.

Motor completion retains the side-resting base. Horn completion turns the assembly upright; screw, retainer and yoke remain upright. Later rewards require the earlier parts correctly placed. Training resets remain all loose: these completion fixtures validate rewards, not a new staged curriculum or executed assembly.

## Result for every task

The actual `BatchedAssemblyEnv.step` earns positive terminal reward in both the default environment and the guarded cloud configuration (`max_contact_depth=0.0004`). Each starts its stability history from zero. Measured on the current Linux build:

| Task | Maximum cumulative placement error | First positive reward | Reward |
| --- | ---: | ---: | ---: |
| Motor |0.045mm |0.44s |0.9999997 |
| Horn |0.468mm |0.44s |0.9999997 |
| Screw |0.489mm |0.44s |0.9999997 |
| Retainer |0.503mm |0.44s |0.9999997 |
| Yoke |0.724mm |0.44s |0.9999997 |

The **34-case audit passes**: ten positive native episodes; five displaced-active-part, four displaced-predecessor and five loose-reset native cases; plus ten geometry-only orientation/translation checks. Negative native cases fork a genuinely rewarded diagnostic context, so stale success cannot hide a bad placement. Wrong orientations are checked without advancing potentially overlapping fixtures. Uniform translation preserves the base-relative target. These checks do not qualify held-object manipulation, robot reachability or real-kit fidelity.

## Default mismatch corrected

The cloud launcher and Quest completion validator already use pose stability. Direct environment/trainer/preflight/overnight defaults still selected legacy velocity stability. In the before-change replay, the screw was correctly positioned and had thread/bearing contacts, but contact velocities repeatedly reset its streak. Pose mode rewarded at0.44s; velocity mode first rewarded at1.60s, with none during the initial1.2s. The screw was delayed, not permanently unrewardable.

All four direct defaults now select the existing pose mode. No reward formula, placement tolerance, stability threshold, physical equation or PPO update changes. Explicit `--so101-stability velocity` / `--stability velocity` and `EnvConfig(stability_metric='velocity')` remain supported. Evaluation reconstructs the saved environment; training's existing environment-equality guard rejects incompatible resumes. Resume an older velocity experiment with its explicit original settings, or start a new run directory. Already-running jobs retain their settings.

## Verification and reproduction

- **63 Python cases pass:** root regressions plus task-scene qualification/replay and compiled training/Quest parity.
- **320 v8 reset states** and the native reference regenerate on Linux with zero rejected states.
- **13 native/CUDA learner checks pass** on RTX5080 / PyTorch2.11.0+cu128, including192transitions and saved learning-state restoration. This does not establish learned task success.
- **30 mandatory preflight checks pass** under the existing explicit task-mechanics exception. Strict screw rotation/pitch qualification still fails: the current bench turns only about0.000024revolutions, below its greater-than-one-turn requirement. Initialized screw reward is not successful screw driving. Motor insertion and hardware calibration remain unresolved.

```bash
PYTHON="$PWD/.venv/bin/python" bash scripts/so101_build_plugins.sh
.venv/bin/python -m scripts.prepare_so101_workcell --count 32 --workers 2
MUJOCO_GL=egl OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  .venv/bin/python -m scripts.check_so101_rewards
```

The auditor writes `artifacts_local/subtask-rewards/report.json` and exits nonzero on a failed case. For regressions:

```bash
MUJOCO_GL=egl PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
  .venv/bin/python -m pytest -q tests teleop/tests/test_task_scenes.py \
  teleop/tests/test_shared_environment.py
```

Ignored local receipts include `probe.json` (before-change two-mode replay), `screw-trace.json`, `report.json` (final audit), `tests.log`, `prepare.log`, `preflight.json` and `learner-smoke/report.json`. Generated targets, banks, native binaries and original CAD remain excluded from Git.
