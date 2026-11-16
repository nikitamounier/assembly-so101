# Shared workcell verification

**Goal:** Verify that policy training and Quest use the same physical environment, while retaining failed mechanics checks.

This table records the earlier **v3** qualification. The current **v4** gripper/contact changes, regression results and broader failed interaction cases are in [the interaction audit](INTERACTION_AUDIT.md); current settings are in [current environment](CURRENT_ENVIRONMENT.md). The v3 signature/banks below remain historical.

The canonical builder/configuration lives in `assembly_policy/so101/scene.py`, `workcell.py` and `configs/so101_scene.json`. The standalone Quest app lives in `teleop/`. The earlier Feather-only adapter has been superseded; no external Feather code is required. See [teleop handoff](TELEOP_HANDOFF.md) for current setup/publication checks and [current environment](CURRENT_ENVIRONMENT.md) for physical settings.

| Check | Evidence and limits |
| --- | --- |
| Training versus Quest | Full serialized native models and initialized positions/controls match byte for byte through the actual entrypoints. |
| Model identity | `b271084fd7234e71aebd9c02e93f6eeb0ca5b004e4efecec0e10d4d33002256d`, profile `shared-side-v3`; training records `model_signature`, Quest exposes the same signature in `/api/config`. |
| Reset bank |320new all-loose starts:32TRAIN/32EVAL per task,±8mmXY/±8-degree yaw, zero rejections under the existing finite/clock/0.4mmnative-contact checks. Yaw composes with the sideways orientation. Old incompatible banks/checkpoints are rejected. |
| Learner smoke |13CUDA/native health/restore checks pass with192real transitions; this does not establish task learning. |
| Preflight |30mandatory checks pass with the existing explicit learner-debug exception for guided screw pitch. Strict mechanics qualification fails. |
| Layout | Base rests sideways, opening upward; motor is aligned for downward insertion. Loose settling has zero sampled motor/base overlap and no numerical warnings. |
| Approach reach |Three of four position-only50mm-above-object checks pass. Right-arm motor approach misses by5.1mm; full grasp/reach remains unqualified. |
| Downward insertion |All5/20Ntests at0.5/0.25mssteps fail seating, ending12.8–14.9mmshort. Peak sampled overlap0.104–0.289mm; halving the step shifts the endpoint by about2mm, so contact-rich insertion convergence is unqualified. |
| Compliance candidates |2/4mscontacts also fail seating and can exceed0.4mmoverlap (up to0.847mm). Keep1mscanonical contacts; retain failures rather than loosening acceptance. |
| Released rest |A passive2scheck bounds loose motor position ranges below5.7micrometres, with zero sampled overlap and no held grips/faults. This is not seated insertion. |
| Hardware calibration |Real dimensions, force, friction, compliance, inertia and servo response are unmeasured. Software checks do not establish realism or sim-to-real transfer. |

The shared native adapter separates capacity from search: the model reserves50contacts; per-model budgets select4/10search or40/40for motor/base volume. Local model views preserve shared options. The untouched motor triangle surface supplements volume contact. Sleeping, welds, live pose projection and automatic mating remain absent. Restart consumers after native-source changes; running jobs keep frozen models/identities.

## Reproduce

```bash
PYTHON=.venv/bin/python bash scripts/so101_build_plugins.sh
.venv/bin/python -m scripts.prepare_so101_workcell --count 32 --workers 2
MUJOCO_GL=egl PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv/bin/python -m pytest -q tests
MUJOCO_GL=egl .venv/bin/python -m scripts.check_so101_workcell
.venv/bin/python -m scripts.cloud.healthcheck --output runs/new-smoke
```

For teleop, use `bash teleop/setup.sh` then `bash teleop/run.sh --quest`; no training bank or CUDA stack is required. Training bootstrap installs the pinned geometry release and generates current reset/reference data. Existing incompatible datasets/references are never overwritten.

Original generated reports remain local and ignored under `artifacts_local/`: `shared_workcell/report.json`, `contact-compliance-candidates.json`, `shared-learner-smoke/report.json`, `shared-preflight.json`, and their test/setup logs. The larger prepublication cross-project inventory is retained locally as `shared-prepublication-notes.md`; it does not define the portable app. No original CAD, private recordings or generated runs are published as source.
