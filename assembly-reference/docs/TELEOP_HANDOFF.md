# Quest teleop handoff

**Goal:** Open this repository, run the shared simulator on the computer and teleoperate through Meta Quest.

Current source includes the [v9 bare HOTO S-A01 reconstruction](HOTO_S_A01_MODEL.md). Update and restart consumers to load it; prior startup/publication evidence below retains its original model identity.

The publication checks below describe the original v3 handoff. The live host uses the [current shared environment](CURRENT_ENVIRONMENT.md), now v5 with the loose parts centered between the mounts. Both fingers close to native contact; the [v4 interaction audit](INTERACTION_AUDIT.md) records earlier checks and unresolved mechanics. Restart consumers after updating; original setup and controls still apply.

Apple Silicon Mac setup and a live USB-connected Quest 3S browser launch are now verified locally. See [Mac session evidence and per-file changes](MAC_QUEST_SESSION.md). These changes have not been pushed; the historical publication results below remain Linux results.

The complete simulation application is in [`teleop/`](../teleop/README.md). It contains host/control code, the stereo browser client, USB connection, setup/launch and tests. No Feather checkout is required. Physical settings remain in the same canonical builder/configuration used by policy training.

From the repository root:

```bash
bash teleop/setup.sh
bash teleop/run.sh --quest
```

Use a Linux x86-64host with Python3.12/uv, C/C++compilers, OpenGL drivers and authorized Quest USB/adb. Setup uses a separate CPU environment; CUDA/PyTorch and a training bank are unnecessary. The pinned private geometry release is downloaded with authenticated `gh` and verified before extraction. Inside Quest choose Enter Quest VR, release Y/grips and press Y; grip moves an arm, trigger controls its jaw, X recenters and B Stops. See the app README for custom adb/ports, connect-later and graphics troubleshooting.

## Verification

| Check | Result |
| --- | --- |
| Packaged app regressions |48distinct Python cases verified:45in the full CPU suite, plus the absent-USB HTTP and two SDK-discovery cases in focused reruns. |
| Canonical environment/learner regressions |21Python cases pass; combined69Python cases. |
| Browser client |16Node browser/stereo/lifecycle cases pass. |
| CUDA learner smoke |13actual native/CUDA checks pass again, including gradients, state updates and restored continuation. |
| Fresh source copy |A new folder with spaces and no assets/venv downloads the real private release, installs the13-package CPU runtime, builds native libraries and launches three views with the matching shared signature. No external checkout is needed. |
| Missing USB |The real launcher stays running and exits cleanly after a bounded run when `--quest` cannot find adb/headset. The original path/USB failures are retained as local diagnostics. |
| Software graphics |Mesa llvmpipe startup renders all three views with `torch_loaded:false`; NVIDIA EGL startup also passes. This is startup qualification, not a software-rendering frame-rate guarantee. |
| Live host |`robotics-assembly-quest.service` runs this repository's app at localhost18883 with the current shared identity. The former Feather service is stopped. |
| Actual headset |The Quest is disconnected during final packaging; physical-controller use after reconnect is pending. The unchanged client previously worked in the wearer-accepted stereo scene. No XR entry/controller input is fabricated. |

The original v3 environment signature was `b271084fd7234e71aebd9c02e93f6eeb0ca5b004e4efecec0e10d4d33002256d`. The packaged training/Quest regression compares complete serialized native models and initialized positions/controls byte for byte. ADB discovery also finds the installed SDK outside PATH without modifying shell settings. The setup fixes interpreter paths with spaces and builds native libraries atomically so active processes keep their mapped binaries.

**Physical limits remain:** nominal motor insertion jams13–15mmshort, guided screw pitch fails strict tolerance, some approach poses are unreachable, and real fit/friction/compliance are uncalibrated. Sleeping, grasp welds and automatic mating remain absent. Software checks and a working teleop connection do not establish realistic assembly or a trained policy. See [current environment](CURRENT_ENVIRONMENT.md) and [shared workcell verification](SHARED_WORKCELL_VALIDATION.md).

Generated local receipts are ignored under `artifacts_local/teleop-*` and `teleop-publication/`: setup logs, full/focused Python tests, Node tests, software graphics, fresh-copy launch/missing-USB checks and CUDA learner report. Original vendor CAD, credentials, recorded sessions, generated banks and native binaries are excluded from source. Runtime geometry stays in the existing private release.

## Files included or modified

This publication includes the previously prepared shared-environment changes as well as the complete application. Existing upstream history and attribution are preserved.

| File | Modification |
| --- | --- |
| [.gitattributes](../.gitattributes) | Preserve the pinned vendor JavaScript's original whitespace while checking project source normally. |
| [.gitignore](../.gitignore) | Exclude the separate CPU teleop environment and generated/private runtime files. |
| [AGENTS.md](../AGENTS.md) | Keep all assembly chats on the canonical shared environment and maintained intent. |
| [MEMORY.md](../MEMORY.md) | Preserve shared-environment and publication decisions with their evidence limits. |
| [README.md](../README.md) | Add clone/setup/Quest launch before the separate CUDA training instructions. |
| [THIRD_PARTY.md](../THIRD_PARTY.md) | Attribute the simulation adaptation and pinned Three.js MIT files. |
| [User Intent.md](<../User Intent.md>) | Record the complete repository-owned Quest app, publication request and remaining mechanics goals. |
| [assembly_policy/so101/contact_audit.py](../assembly_policy/so101/contact_audit.py) | Keep original-CAD sampled material-overlap checks in the canonical environment. |
| [assembly_policy/so101/evaluate.py](../assembly_policy/so101/evaluate.py) | Use the configured bank and report the actual environment profile. |
| [assembly_policy/so101/resets.py](../assembly_policy/so101/resets.py) | Version model identity and retain side orientation in randomized all-loose resets. |
| [assembly_policy/so101/robot.py](../assembly_policy/so101/robot.py) | Initialize arm positions/controls from the shared configuration. |
| [assembly_policy/so101/scene.py](../assembly_policy/so101/scene.py) | Apply the canonical workcell and load the shared native contact runtime. |
| [assembly_policy/so101/workcell.py](../assembly_policy/so101/workcell.py) | Centralize layout, contacts, solver/search, cameras, damping, identity and reference checks. |
| [assembly_policy/train.py](../assembly_policy/train.py) | Select the current configured SO101 bank without changing PPO objectives. |
| [configs/cloud_stress.json](../configs/cloud_stress.json) | Explicitly retain the known screw-mechanics exception in learner-debug mode. |
| [configs/so101_scene.json](../configs/so101_scene.json) | Own the side-resting layout, cameras, caps, contact coverage, solver and data versions. |
| [docs/ARCHITECTURE.md](../docs/ARCHITECTURE.md) | Update shared-environment architecture, handoff, verification or historical status for the packaged app. |
| [docs/CHANGES.md](../docs/CHANGES.md) | Update shared-environment architecture, handoff, verification or historical status for the packaged app. |
| [docs/CURRENT_ENVIRONMENT.md](../docs/CURRENT_ENVIRONMENT.md) | Update shared-environment architecture, handoff, verification or historical status for the packaged app. |
| [docs/HANDOFF_SCOPE.md](../docs/HANDOFF_SCOPE.md) | Update shared-environment architecture, handoff, verification or historical status for the packaged app. |
| [docs/HANDOFF_VALIDATION.md](../docs/HANDOFF_VALIDATION.md) | Update shared-environment architecture, handoff, verification or historical status for the packaged app. |
| [docs/ORACLE_HANDOFF.md](../docs/ORACLE_HANDOFF.md) | Update shared-environment architecture, handoff, verification or historical status for the packaged app. |
| [docs/SHARED_WORKCELL_VALIDATION.md](../docs/SHARED_WORKCELL_VALIDATION.md) | Update shared-environment architecture, handoff, verification or historical status for the packaged app. |
| [docs/SO101_STATUS.md](../docs/SO101_STATUS.md) | Update shared-environment architecture, handoff, verification or historical status for the packaged app. |
| [docs/TELEOP_HANDOFF.md](../docs/TELEOP_HANDOFF.md) | Record current packaging/verification, scope and this per-file inventory. |
| [plugin/build.sh](../plugin/build.sh) | Quote interpreter paths and atomically replace the SDF library. |
| [plugin/build_contact_search.sh](../plugin/build_contact_search.sh) | Compile the version-pinned shared native adapter atomically. |
| [plugin/so101_contact_search.c](../plugin/so101_contact_search.c) | Separate50-contact capacity from per-model4/10or40/40search without preload. |
| [requirements-cloud.lock](../requirements-cloud.lock) | Pin the added dependency while retaining the CUDA wheel index. |
| [requirements-cloud.txt](../requirements-cloud.txt) | Declare libigl for shared CAD checks alongside the existing CUDA learner stack. |
| [scripts/check_so101_workcell.py](../scripts/check_so101_workcell.py) | Measure native settling, approach reach and force/timestep/compliance insertion outcomes. |
| [scripts/cloud/assess_learning.py](../scripts/cloud/assess_learning.py) | Assess policies against the configured shared environment bank. |
| [scripts/cloud/bootstrap.sh](../scripts/cloud/bootstrap.sh) | Build shared plugins and generate matching reset/reference data. |
| [scripts/cloud/healthcheck.py](../scripts/cloud/healthcheck.py) | Use the shared current bank for actual CUDA/native health and restore checks. |
| [scripts/cloud/run.py](../scripts/cloud/run.py) | Launch with the configured bank and explicit mechanics exception. |
| [scripts/cloud/runtime_bundle.py](../scripts/cloud/runtime_bundle.py) | Package and install versioned shared runtime/reference data with integrity checks. |
| [scripts/prepare_so101_workcell.py](../scripts/prepare_so101_workcell.py) | Prepare current versioned bank/reference data and reject incompatible existing data. |
| [scripts/so101_build_plugins.sh](../scripts/so101_build_plugins.sh) | Build the shared contact library and atomically replace the driver library. |
| [scripts/so101_preflight.py](../scripts/so101_preflight.py) | Check current bank/reference and retain strict versus explicit learner-debug mechanics gates. |
| [teleop/PROVENANCE.json](../teleop/PROVENANCE.json) | Bind copied source/browser assets and describe packaging adaptations. |
| [teleop/README.md](../teleop/README.md) | Give setup, USB, controls, shared-physics ownership and troubleshooting commands. |
| [teleop/__init__.py](../teleop/__init__.py) | Define the standalone simulation-only application package. |
| [teleop/backend.py](../teleop/backend.py) | Adapt the canonical model/home API without physical overrides and expose matching identity. |
| [teleop/connect_quest.py](../teleop/connect_quest.py) | Validate the simulator, select an authorized Quest and preserve USB routes. |
| [teleop/control.py](../teleop/control.py) | Preserve the tested relative clutch, attainable rotation, reference dynamics and simulation lifecycle. |
| [teleop/doctor.py](../teleop/doctor.py) | Check native physics, three camera views, graphics renderer and absence of PyTorch loading. |
| [teleop/requirements.lock](../teleop/requirements.lock) | Resolve all13CPU runtime packages. |
| [teleop/requirements.txt](../teleop/requirements.txt) | Declare the minimal pinned CPU simulation dependencies. |
| [teleop/run.sh](../teleop/run.sh) | Launch from any working directory using repository-relative paths. |
| [teleop/runtime.py](../teleop/runtime.py) | Package the single owner, HTTP/session transport, coherent camera capture and diagnostics. |
| [teleop/serve.py](../teleop/serve.py) | Expose the standalone host CLI with bounded runs and nonfatal missing-USB startup. |
| [teleop/setup.sh](../teleop/setup.sh) | Install the isolated CPU runtime, verify/download assets and build/check native plugins. |
| [teleop/tests/contact_pose.json](../teleop/tests/contact_pose.json) | Preserve sanitized native simulation failure poses for reproducible regressions. |
| [teleop/tests/immersive.test.mjs](../teleop/tests/immersive.test.mjs) | Verify metric stereo geometry, head-dependent perspective, interpolation and owner fencing. |
| [teleop/tests/motor_seated_pose.json](../teleop/tests/motor_seated_pose.json) | Preserve sanitized native simulation failure poses for reproducible regressions. |
| [teleop/tests/motor_side_pose.json](../teleop/tests/motor_side_pose.json) | Preserve sanitized native simulation failure poses for reproducible regressions. |
| [teleop/tests/session.test.mjs](../teleop/tests/session.test.mjs) | Regress production browser owner recovery, Enable/grip behavior and marker handling. |
| [teleop/tests/test_assembly.py](../teleop/tests/test_assembly.py) | Verify native arm/jaw movement, lifecycle, receipt timing, frames and exported geometry. |
| [teleop/tests/test_connect_quest.py](../teleop/tests/test_connect_quest.py) | Discover existing SDK installs; refuse wrong services, unauthorized devices and conflicting routes; verify custom-port connection. |
| [teleop/tests/test_contact_stability.py](../teleop/tests/test_contact_stability.py) | Check safe contact capacity and clear-start motor drops. |
| [teleop/tests/test_http_launch.py](../teleop/tests/test_http_launch.py) | Exercise real CLI/HTTP/assets/views and continued operation without an attached headset. |
| [teleop/tests/test_motor_walls.py](../teleop/tests/test_motor_walls.py) | Retain force-driven wall blocking with independent sampled CAD overlap. |
| [teleop/tests/test_responsiveness.py](../teleop/tests/test_responsiveness.py) | Retain native tracking/settling, independent arm and control lifecycle regressions. |
| [teleop/tests/test_seated_motor.py](../teleop/tests/test_seated_motor.py) | Reject invalid overlap, verify awake force response and safe pose-only resume. |
| [teleop/tests/test_shared_environment.py](../teleop/tests/test_shared_environment.py) | Compare complete training/Quest models, home controls and environment identity. |
| [teleop/visual_scene.py](../teleop/visual_scene.py) | Export native visual geometry for local head-tracked stereo rendering. |
| [teleop/web/app.js](../teleop/web/app.js) | Package the tested Quest/browser UI, controller mapping and local stereo rendering assets. |
| [teleop/web/hybrid.mjs](../teleop/web/hybrid.mjs) | Package the tested Quest/browser UI, controller mapping and local stereo rendering assets. |
| [teleop/web/immersive.mjs](../teleop/web/immersive.mjs) | Package the tested Quest/browser UI, controller mapping and local stereo rendering assets. |
| [teleop/web/index.html](../teleop/web/index.html) | Package the tested Quest/browser UI, controller mapping and local stereo rendering assets. |
| [teleop/web/preview.mjs](../teleop/web/preview.mjs) | Package the tested Quest/browser UI, controller mapping and local stereo rendering assets. |
| [teleop/web/style.css](../teleop/web/style.css) | Package the tested Quest/browser UI, controller mapping and local stereo rendering assets. |
| [teleop/web/vendor/LICENSE](../teleop/web/vendor/LICENSE) | Include the pinned Three.js renderer, its MIT license and hash provenance. |
| [teleop/web/vendor/PROVENANCE.json](../teleop/web/vendor/PROVENANCE.json) | Include the pinned Three.js renderer, its MIT license and hash provenance. |
| [teleop/web/vendor/three.core.js](../teleop/web/vendor/three.core.js) | Include the pinned Three.js renderer, its MIT license and hash provenance. |
| [teleop/web/vendor/three.module.js](../teleop/web/vendor/three.module.js) | Include the pinned Three.js renderer, its MIT license and hash provenance. |
| [tests/test_workcell.py](../tests/test_workcell.py) | Regress shared contact settings, sideways reset preservation, reference and bank compatibility. |
