# Upright screw presets and final arm link

**Goal:** Present the screw from above after motor insertion and add the arm/link after the shoulder yoke as the sixth task.

The user approves the existing steps and clarifies that the missing piece is the arm/link after yoke, not the bottom horn. The new task order is motor → horn → screw → retainer → yoke → arm link. A tap loads the next prepared scene; A held two seconds still resets to the loose motor stage. Control/HUD mappings and the accepted main camera remain.

## Canonical implementation

`configs/so101_scene.json` owns stage orientation and the new loose-part pose. `task_scenes.py` turns the base/motor upright for horn/screw/retainer and settles the canonical native model. Measured screw insertion axes align with world down to better than 0.001 in dot-product error. Motor insertion remains sideways; yoke/link presets return to the side because upright retainer preparation fails the existing 0.4 mm contact gate. No contact threshold was increased and no fixture/weld added.

The new arm link is vendor CAD object 139, UUID `73c7aeca-8917-4afe-aa11-3778a4b0992f`, from `follower arm black.3dm` with SHA256 `f675f6049f94206c7b74ab73c40485913916babff1a6f4fc145dc9040672a2f8`. Its extracted mesh is 73.66×111.70×112.06 mm in its original CAD orientation, 12,816 triangles. Source triangles are unchanged except coincident seam merging. The source is not watertight; that limitation is retained. A 0.35 mm signed-distance grid uses exact triangle distance and winding-sign classification, matching the existing field approach. The new free-body mass is a provisional 0.05 kg with box-based inertia, not a hardware measurement.

The additive `arm_link.json` manifest preserves the original private v1 archive's manifest and hashes. Setup can install/verify the original archive, then independently verify or reproduce the link runtime. On a fresh machine the generator downloads the pinned 176.6 MB CAD from the public vendor link, verifies its SHA, extracts the pinned object and builds grid 107. Original CAD stays ignored and is excluded from runtime packing. Cached runtime members are rehashed on setup. CPU setup adds only pinned rhino3dm/trimesh; every original lockfile version is preserved. Cloud bootstrap also prepares the added geometry.

Both training and Quest load the new part through the same scene factory and task list. Twelve actions, 53 proprio values per frame and 265 history values are unchanged; geometry context includes the added body. The model has 28 bodies, 72 qpos and 64 qvel. The added free joint is appended at qpos 65/dof 58 after the tool, preserving all older joint/body ordering. Historical diagnostic fixtures are expanded only by a tests-only helper; production old-state resumes remain incompatible. The v6 bank/reference paths are new; no bank, checkpoint, running learner or older experiment is overwritten.

Model signature: `642239a3bf93d4b1e7056d3ab1283d2817e518ac7422039c7fdb0dd141d15dc6`. Profile: `shared-side-v6-arm-link`. Physical configuration SHA256: `ebfca66ca900c636a50487fea6dbe2d6fb02851115ac1b232986a6d93a96217b`.

## Verification and limits

- All six prepared scenes are finite, warning-free and pass the unchanged contact/independent motor-base overlap gate. Maximum reported native depth is below 0.096 mm.
- Shared training/Quest binary-model and home-state parity passes. A native learner world for `arm_link` preserves observation/action shapes, steps and remains unsuccessful from a loose start.
- 41 targeted Python cases pass across the broad run and focused rerun; 21 browser cases pass. The first broad run exposes two test-duration errors inherited from the previous scheduling edit. Restore the native arm-motion loop to 2 s of simulation and the loaded fixture loop to 0.4 s; both pass without changing their motion/expiry assertions or runtime code.
- Full `bash teleop/setup.sh` passes with 15 pinned CPU packages, verified old/new geometry, atomic native builds and all three Apple M5 CGL camera views, without torch.
- Original CAD checksum/object UUID, generated mesh/grid hashes, direct download header and cached asset revalidation are checked. Rendered actual-model images are retained below.

The final prepared link stage still has about 6.02 mm retainer error and 0.56 mm yoke error. The upright retainer stage has about 1.23 mm horn position error. These are reported preparation/fit limitations, not successful predecessor assembly. Hardware friction, compliance/mass, full link mating, screw driving and learned assembly remain unqualified. Six-task banks, GPU learning and new-asset publication are pending. All source/runtime changes are local and unpushed.

![Upright screw-stage native view](../artifacts_local/upright-link/screw-view.jpg)

![Sixth arm-link task native view](../artifacts_local/upright-link/arm_link-view.jpg)

Local evidence: `artifacts_local/upright-link/`: original catalog/candidate meshes and images, asset build/readback, six-scene measurements, setup log, Python/browser logs, native view images and final host/browser receipt. The live PID is saved in `host.pid`; inspect that process and localhost 18883 health before restarting. Final live verification confirms PID 68690 healthy, the exact v6 six-task list/identity, 28-body headset geometry, a camera-attached DISABLED · MOTOR HUD and no JavaScript exceptions. The reopened page loads its initial native state/views independently of requestAnimationFrame, so Quest animation throttling outside XR does not leave an uninitialized scene. Wearer XR entry and manipulation remain pending for this revision. Receipt: `artifacts_local/upright-link/live.json`.

## Files changed and modifications per file

This lists this turn; all preceding Mac/control edits are preserved.

| File | Modification |
| --- | --- |
| `assembly_policy/so101/scene.py` | Add canonical arm-link mesh/free body, metadata and shared six-task list; preserve old joint/body ordering. |
| `assembly_policy/so101/task_scenes.py` | Use shared tasks; prepare stage-specific base orientation and report the insertion axis. |
| `assembly_policy/so101/fastener.py` | Carry the existing assembly-reference lift into the next link reference. |
| `assembly_policy/so101/workcell.py` | Include the link in assembled-reference transforms. |
| `assembly_policy/so101/env.py` | Support six task identities and observe the seventh target body. |
| `configs/so101_scene.json` | Add loose-link pose, upright-stage settings and distinct v6 identity/data paths. |
| `scripts/so101_link_assets.py` | Verify/download pinned CAD and deterministically extract/check the original link runtime. |
| `scripts/cloud/runtime_bundle.py` | Include derived addon grids when packing; exclude original CAD. |
| `scripts/cloud/bootstrap.sh` | Prepare added geometry before building/qualifying the environment. |
| `scripts/so101_preflight.py` | Derive world/observation/action batch counts from the shared task list. |
| `requirements-cloud.txt`, `requirements-cloud.lock` | Add only pinned rhino3dm while preserving every previous resolved version. |
| `teleop/setup.sh` | Verify or reproduce the added runtime before the native doctor check. |
| `teleop/requirements.txt`, `teleop/requirements.lock` | Add pinned CAD extraction dependencies without changing existing versions. |
| `teleop/web/app.js` | Display ARM LINK naturally and load the first native state/views independently of Quest animation timing. |
| `teleop/tests/native_fixture.py` | Explicitly append only the added link to historical test fixtures. |
| `teleop/tests/test_task_scenes.py` | Check all six scenes, stage cycling, original vendor identity and downward screw axes. |
| `teleop/tests/test_assembly.py` | Update model dimensions/fixture and correct simulation loop durations. |
| `teleop/tests/test_motor_walls.py`, `test_seated_motor.py` | Reuse preserved failure fixtures on the expanded model in tests only. |
| `teleop/tests/test_shared_environment.py` | Validate native six-task observation/action contract and training/Quest parity. |
| `teleop/tests/test_http_launch.py` | Check the real v6 HTTP host and 28-body payload. |
| `README.md`, `teleop/README.md` | Describe current six-stage setup, orientation, assets and compatibility. |
| `User Intent.md` | Replace the superseded all-sideways/five-stage agreement with the clarified scope. |
| `MEMORY.md`, `TASK.md` | Record decision, evidence, completion and next-session state. |
| `docs/CURRENT_ENVIRONMENT.md`, `docs/QUEST_CONTROLS_RECOVERY.md` | Distinguish active v6 behavior from retained v5 evidence. |
| `docs/UPRIGHT_LINK_UPDATE.md` | This evidence, limitations and file inventory. |

## Sources

[Hiwonder technical folder](https://drive.google.com/drive/folders/1sAwusnQrOkSJRaRBbTIlIXnjiwk_TyVz), [exact downloaded CAD](https://drive.google.com/file/d/13ffIl5rWNmyrwV8Q4zCmyH30RVABJGAM/view), and the existing [source/provenance document](SO101.md). The current online manual was consulted but does not identify the cached CAD objects; the pinned file hash and native extraction establish their identity.
