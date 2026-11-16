# Native task-completion states

**Goal:** Make A display actual policy-reward completion states, with the base sideways for motor completion and upright from flat-horn completion onward.

This report records v7. The later [v8 removal](UNUSED_LINK_REMOVAL.md) supersedes retaining the loose link and updates the model identity/body count. The five-task A behavior is preserved.

The November 5 wearer correction supersedes the v6 next-task-start presets and their later rotation back onto the side. A now cycles initial loose → completed motor → completed horn → completed screw → completed retainer → completed yoke → initial loose. Holding A for two seconds resets the entire scene. Each change starts Disabled and discards old grabs. Physics continues while Disabled. The existing top-left status-only HUD and accepted camera remain.

The user explicitly requires native reward validation and rejected paused geometric inspection. The final link has a 4.136 mm vertical gap above the yoke at the original CAD target, without a modeled shoulder connection. Native tests show it falling and failing reward. The user chose to end the validated sequence at yoke. The original loose link and experimental sixth policy task remain; its target is excluded from A.

## Implementation and evidence

`goals.py` extracts the existing cumulative placement-error calculation for shared use. The policy's reward criteria are unchanged. Completion initialization includes the current part, then settles and exercises the real `BatchedAssemblyEnv.step`: release, required screw thread/head contacts, safety, the 1 mm placement tolerance and eleven pose samples spanning 0.4 s. The existing pose profile retains 0.25 mm translation/0.01 rad stability thresholds and a 0.4 mm contact-depth guard. No pause, weld, force-limit increase, threshold relaxation or ongoing pose correction is used. Initialization is not executed robot assembly or learned success.

Sparse horn/motor contact search allowed the round horn to tilt. A native search comparison reduced tilt from 14.12 degrees at 4/10 starts/iterations to 0.25 degrees at 16/20 and 0.022 degrees at 40/40. The shared contact adapter now uses 40/40 for motor/horn, retains motor/base 40/40 and finger/object 16/20, and uses 16/20 for remaining free-body pairs. Other contacts retain 4/10 and contact capacity remains 50. Training and Quest consume these same settings.

The screw's nominal target lacked head-bearing contact after the horn settled. Preparation now seats its head at the settled horn bearing plane with a physical 270-degree thread phase. Screw yaw is unconstrained by the existing axial reward keypoints. This is an initialized threaded placement, not a claim of successful driver operation. Other tested phases failed; the retained phase passes the actual thread/head-contact and stability checks.

Native-qualified snapshots are cached only in ignored `assets_local/task_targets/`, bound to full environment identity and qualification/reward source. Fresh machines qualify independently and reject failures. First startup can take several minutes. Loaded snapshots stay fully dynamic.

| Completed task | Maximum cumulative placement error | Round-horn tilt relative to base | Native reward |
| --- | ---: | ---: | --- |
| Motor | 0.045 mm | Not placed | Pass |
| Horn | 0.467 mm | 0.024 degrees | Pass |
| Screw | 0.487 mm | 0.062 degrees | Pass |
| Retainer | 0.503 mm | 0.049 degrees | Pass |
| Yoke | 0.650 mm | 0.775 degrees | Pass |

All five also pass independent 1.2 s replay through the native reward implementation. This is bounded native validation, not indefinite retention or real-hardware qualification. Exact measurements and histories are under `artifacts_local/reward-states/qualification.json` and `task-tests.log`.

48 distinct Python cases pass across broad and focused runs, including independent reward replays, the A sequence, reset gravity, expiry/Stop behavior, native arm/gripper movement, shared compiled-model parity and two actual HTTP CLI launches. All 21 browser cases pass. The first concurrent run expired controller input in the dense contact fixture while the old live plant and validation jobs were also running. The fixture subsequently passed twice with the old plant closed; the 200 ms leases and performance assertion were preserved. Do not infer performance under arbitrary CPU contention. Logs retain both failures and reruns in the same artifact directory. Python compilation and `git diff --check` pass.

## Live session and limits

The updated Mac host is PID **77242**, listening on port **18883**. The authorized USB Quest 3S browser was reopened with cache disabled at the v7 URL. Its real native owner matches the service; the 28-body immersive geometry is ready, synthetic mode is false, no native faults or JavaScript exceptions were observed, and the head-locked HUD reads `DISABLED · MOTOR`. The page is awaiting **Enter Quest VR** and a fresh **Y** press. Browser/health receipts and screenshot are `artifacts_local/reward-states/{browser.json,browser.png,live.json,host.log,host.pid}`. Fresh wearer operation remains the follow-up.

Identity: profile `shared-side-v7-reward-targets`; model signature `f64dc4d76a4b9b6db185cac39cd4069941a13c2549d826f432c1fb705b0670f9`; configuration SHA256 `4785589233ee3cb907bdb7fe73d393309492805701de06800485c231c343e337`; MuJoCo 3.17.0. No v7 bank/reference generation, GPU training, commit, push or release was performed. Previous running training jobs are untouched. Executed motor insertion, physical screw driving, calibrated forces/fit and real hardware remain unqualified.

## Files changed in this correction

| File | Modification |
| --- | --- |
| `assembly_policy/so101/goals.py` | Share the policy's existing cumulative placement geometry. |
| `assembly_policy/so101/env.py` | Consume the shared geometry without changing success gates. |
| `assembly_policy/so101/task_scenes.py` | Build completed targets, settle, earn actual native reward, cache qualified states and exclude unsupported link from A. |
| `assembly_policy/so101/workcell.py` | Validate and serialize shared horn/free-body contact budgets. |
| `configs/so101_scene.json` | Correct completion orientations, define five validated A tasks and version physical identity/data paths. |
| `plugin/so101_contact_search.c` | Apply per-model horn and free-body contact search budgets. |
| `teleop/backend.py` | Separate initial stage from completed-task indices, reset to loose state and expose shared contact settings. |
| `teleop/tests/test_task_scenes.py` | Verify native success/replay, correct rotations/flat horn, complete A cycle, dynamic reset and existing input protections. |
| `teleop/tests/test_http_launch.py` | Expect the current v7 environment identity. |
| `README.md`, `teleop/README.md` | Document completion semantics, qualification startup, link deferral and data compatibility. |
| `docs/CURRENT_ENVIRONMENT.md`, this report | Record current identity, evidence and limitations. |
| `User Intent.md`, `MEMORY.md`, `TASK.md` | Replace superseded behavior and retain decisions/progress. |

Unrelated and earlier uncommitted work was preserved. Original CAD, generated states and native build products remain ignored.
