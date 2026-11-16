# HOTO SNAPBLOQ S-A01 reconstruction

**Goal:** Replace the crude screwdriver/sleeve proxy with a detailed reconstruction of Amazon B0DK4XJ9HF in the shared training and Quest scene.

![Native model and detail views](images/hoto-s-a01.png)

The default is the bare **HOTO SNAPBLOQ S-A01 / QWLSA002** with an installed C4 Phillips bit. The shell, recessed bezel, rocker ends/arrows, circular selector, three indicators, grip fluting, diffuser/collet, rear seam and USB-C opening/tongue are geometry. No vendor CAD, external textures or fonts are required. Quest receives these same meshes and the new tool's native material gloss.

## Sources and accuracy

| Source | Use |
| --- | --- |
|[User's Amazon product](https://www.amazon.com/dp/B0DK4XJ9HF) |Identify the exact S-A01. |
|[Official HOTO photographs](https://hototools.com/products/snapbloq-a01) |Shape, controls, finish, fluting, light ring and charging port. |
|[Official HOTO Japan specification](https://hototools.jp/products/hfe0025gl) |Published181.7×19.7×22.5mm,105gdriver and C4×28mmbits. |
|[Official manual](https://hoto-jp.com/document/HFE0025GL_HOTO_Electric_Precision_Screwdriver_Master_Set.pdf) |Rocker ends, separate selector and three indicators. |

This is a photo reconstruction, **not manufacturer CAD or hardware calibration**. Published length is interpreted as the bitless tool. The173.5mmbarrel/8.2mmnose split,8mmbit insertion/20mmexposure and small features are image-derived assumptions. The driver excluding its bit totals105g. Bit mass/inertia, internal mass distribution, button travel/springs and surface properties remain estimates. The PH1 tip approximates the nominal screw recess.

The selector physically moves, but torque-switching firmware is not simulated. The plugin retains its existing0.05Nmcap and250rpmno-load target, without a new electrical policy action. Three indicators show the nominal low setting; the diffuser is unlit. Native contact depresses the forward/reverse keys; release removes motor torque.

## Collision and compatibility

Three convex casing sections leave an open control well; bezel rails leave openings for the controls. Fine recess corners/fluting remain collision approximations. Internal guide/bearing exclusions prevent the inserted bit and moving controls from colliding with their own housing, including when a test bolts the tool to world. External finger, part and table contacts remain enabled.

Three explicit case/table pairs enable6Dcontact using the existing0.0001mrolling-friction coefficient, retaining previous sliding/torsional coefficients and normal response. This provisional surface assumption resolves continued rocking after a60mmdrop. No sleeping, pose correction or free-body damping was added.

Profile **shared-side-v9-hoto-s-a01** generates new v9 banks/reference; incompatible old data/checkpoints remain rejected. The model retains27bodies,65qpos,58velocities,12actions,265history values and320points. For observation API continuity, `driver_sleeve` now names the actual fixed bezel and `driver_trigger_saddle` / `trigger_saddle` name the round selector.

The old design is isolated in `tool_proxy.py`. `add_driver(..., with_sleeve=True)` selects that historical proxy for standalone experiments; it is not an accessory fitted to the new reconstruction. The three historical bench scripts explicitly import the proxy. Their old grasp/torque results do **not** qualify the new bare tool. Robot pickup, OFF carrying, reaction torque and screw driving remain unqualified.

## Reproduce

The final v9 model binds signature `99f8a29b33345a667b64a861ae38e9e4822f07ed13c37c6e92dc50605b9f8793`. All320independent reset states regenerate without rejection and match the new reference. Six dedicated tool cases pass: envelope/mass/outward mesh normals, forward/reverse and selector contact presses, release/coast-down and20/60mmdrops. Drop native depth is sampled every10ms, so this does not certify transient nonintersection. All22browser cases pass, including native gloss and legacy material support.

All34shared reward audit cases also pass with this tool installed. Motor, horn, screw, retainer and yoke each earn approximately+0.9999997after0.44s in the default and guarded pose configurations. Misplaced parts/predecessors, loose resets and wrong orientations are rejected. These are initialized target tests, not successful robot assembly. Local receipts are `qualified-prepare.log`, `qualified-tool-tests.log`, `qualified-rewards.json`, `qualified-regressions.log` and `browser-tests.log` under `artifacts_local/hoto-reference/`; failed/interrupted candidates remain there too.

Final verification passes121Python cases,22browser cases,34reward-audit cases (24native episodes/state forks and10geometry checks),13RTX5080CUDA/native learner checks and30mandatory preflight checks. The existing screw rotation/pitch mechanics exception is explicit and remains failed. The local Linux host is restarted onto this identity, healthy and Disabled with no faults. The Mac/headset consumer must update/restart separately; wearer operation of this version is not claimed.

```bash
MUJOCO_GL=egl .venv-teleop/bin/python -m scripts.render_so101_tool
MUJOCO_GL=egl PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
  .venv-teleop/bin/python -m pytest -q teleop/tests/test_hoto_tool.py
node --test teleop/tests/*.test.mjs
.venv/bin/python -m scripts.prepare_so101_workcell --count 32 --workers 2
MUJOCO_GL=egl .venv/bin/python -m scripts.check_so101_rewards \
  --out artifacts_local/hoto-reference/rewards.json
```

Restart consumers after updating; a running model retains its previous identity. Use the normal repository launcher and documented CGL settings on Mac. Reference photos/manual, generated states, binaries and logs remain ignored under `artifacts_local/hoto-reference/` and the normal asset/data directories.

## Files changed

| File | Modification |
| --- | --- |
|[tool.py](../assembly_policy/so101/tool.py) |Build the bare reconstructed tool and native controls. |
|[tool_geometry.py](../assembly_policy/so101/tool_geometry.py) |Generate casing, recess, fluting, bezel, icons and bit meshes. |
|[tool_proxy.py](../assembly_policy/so101/tool_proxy.py) |Preserve the old sleeve proxy for explicit historical experiments. |
|[scene.py](../assembly_policy/so101/scene.py) |Supply the shared mesh/material asset container. |
|[so101_scene.json](../configs/so101_scene.json) |Version v9 data/reference, set resting height and casing/table rolling contacts. |
|[check_so101_tool.py](../scripts/check_so101_tool.py) |Select the historical proxy explicitly. |
|[check_so101_tool_gripper.py](../scripts/check_so101_tool_gripper.py) |Keep old grasp experiments on their original proxy. |
|[check_so101_fixed_finger_pocket.py](../scripts/check_so101_fixed_finger_pocket.py) |Keep the old pocket experiment on its original proxy. |
|[check_so101_interactions.py](../scripts/check_so101_interactions.py) |Use the new driver's width and grasp reference. |
|[render_so101_tool.py](../scripts/render_so101_tool.py) |Produce the native detail figure from the shared asset. |
|[visual_scene.py](../teleop/visual_scene.py) |Export native gloss for HOTO materials. |
|[immersive.mjs](../teleop/web/immersive.mjs) |Render those materials in Quest. |
|[test_hoto_tool.py](../teleop/tests/test_hoto_tool.py) |Check envelope/mass/normals, physical controls and drops. |
|[immersive.test.mjs](../teleop/tests/immersive.test.mjs) |Check material transport and backwards compatibility. |
|[test_http_launch.py](../teleop/tests/test_http_launch.py) |Require the current physical profile. |
|[hoto-s-a01.png](images/hoto-s-a01.png) |Show the complete tool and native detail views. |
|[README.md](../README.md) |Point to v9 and this reconstruction. |
|[User Intent.md](../User%20Intent.md) |Record the fidelity requirement and current reset compatibility. |
|[MEMORY.md](../MEMORY.md) |Record decisions, qualification and remaining limits. |
|[CURRENT_ENVIRONMENT.md](CURRENT_ENVIRONMENT.md) |Identify current geometry, data and checks. |
|[SO101_TOOL.md](SO101_TOOL.md) |Mark earlier sleeve evidence historical. |
|[TELEOP_HANDOFF.md](TELEOP_HANDOFF.md) |Explain the model update/restart. |
|[HOTO_S_A01_MODEL.md](HOTO_S_A01_MODEL.md) |Record sources, assumptions, reproduction and inventory. |

Intentionally unchanged: original target/helper CAD, mounts, accepted camera, arm controls/efforts, motor plugin and reward/PPO/SGS equations. Follow-up: real tool measurements, fitted holder/bare-grip qualification, physical screw driving and wearer verification.
