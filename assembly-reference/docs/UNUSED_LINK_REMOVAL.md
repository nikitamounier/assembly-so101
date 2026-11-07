# Removed unused arm link

**Goal:** Remove the unused arm link from the canonical training/Quest scene and preserve the five validated completion states.

The latest November 5 request supersedes keeping the unsupported link loose as an experimental sixth task. The link body, free joint, collision/visual geometry, policy task and observation body are removed from the shared builder. The active scene contains base, motor, round horn, screw, retainer, yoke and screwdriver. A still cycles five completed native-reward states, then the loose reset. Motor remains sideways; horn through yoke remain upright. A hold reset, control mappings, HUD, mounts and camera are unchanged.

Setup/bootstrap no longer extract the link. Runtime packing selects only the original part manifest and its referenced grids, excluding cached unused link files. Original CAD, optional extraction tooling, historical records and generated assets are preserved on disk.

## Verification

All **38 affected Python cases pass** in 129.12 s, including actual native reward qualification/replay for all five states, no-link body/task/observation assertions, training/Quest compiled-model parity, physical arm/gripper response, input expiry/Stop and two real HTTP CLI launches. The model now has **27 bodies, 65 position coordinates and 58 velocities**. It still uses 12 actions, 265 robot-history values and 320 geometry points, with eleven observed bodies. The context sampler's motor-task base allocation becomes 13 points after removing one context body. The existing visualization remains a labeled historical six-task snapshot.

The updated service is PID **80614** on port18883. Quest Browser was reopened with cache disabled at the v8 URL. Its real owner matches the healthy service; 27-body geometry is ready, the unused link is absent, synthetic mode is false, and no native faults or JavaScript exceptions were observed. The screenshot was inspected. HUD reads `DISABLED · MOTOR`. Follow-up: wearer taps **Enter Quest VR**, then fresh **Y**, and tests the scene.

Evidence is under ignored `artifacts_local/unused-link-removal/`: `tests.log`, `qualification.json`, `identity.json`, `browser.json`, `browser.png`, `live.json`, `host.log` and `host.pid`.

Profile: `shared-side-v8-five-tasks`. Model signature: `bd1478216658d1061a078051b3561332b8211bfea005ac0b1f90332e2c5c4cd4`. Configuration SHA256: `3f2ed24533c1b13aa5b80fed9d40ec0ff5295bc886847d6517e3775c2ad2ec4d`. New v8 bank/reference paths prevent incompatible data reuse. No training-data generation, GPU job change, commit, push or release was performed.

## Files changed

| File | Modification |
| --- | --- |
| `assembly_policy/so101/scene.py` | Remove link metadata, body, joint, geometry and task/observation entries. |
| `assembly_policy/so101/workcell.py` | Remove link from shared assembled-body initialization. |
| `assembly_policy/so101/fastener.py` | Remove unused link target offset. |
| `assembly_policy/so101/env.py`, `task_scenes.py` | Update body/task descriptions for the five-task model. |
| `configs/so101_scene.json` | Remove link poses/extraction/rotation and version identity/data paths. |
| `teleop/setup.sh`, `scripts/cloud/bootstrap.sh` | Remove automatic link generation/download. |
| `scripts/cloud/runtime_bundle.py` | Package only manifest-owned parts and grids. |
| `teleop/tests/test_task_scenes.py`, `test_shared_environment.py`, `test_assembly.py`, `test_http_launch.py` | Verify removal, current dimensions/observations, model parity and current HTTP scene. |
| `teleop/tests/native_fixture.py` | Require fixtures to match the current native model dimensions. |
| `README.md`, `teleop/README.md`, `docs/CURRENT_ENVIRONMENT.md` | Document current scene/task identity. |
| `docs/POLICY_INPUT_VIEW.md`, `docs/REWARD_STATE_UPDATE.md` | Label prior six-task data and v7 evidence as historical. |
| `User Intent.md`, `MEMORY.md`, `TASK.md`, this report | Record the corrected scope, verification and follow-up. |

Unrelated uncommitted changes are preserved.
