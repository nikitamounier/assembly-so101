# Experimental v11 training snapshot

**Goal:** Record the fixed source, checks and launch limits for training while CAD work continues.

Use tag `training-v11-20261106` in a separate checkout. This publication merges the local shared physics/subtask work with GitHub's selected 448-point observation, preserving both development histories. It does not mark the separate CAD repair/visual review complete.

## Identity and data

- Physical profile: `shared-side-v11-contact-coverage`.
- Model signature: `3b6955d8ddf4cb75da809d5d947d1a779e3d819c01017bb2110bcbc0568e6925`.
- Configuration SHA256: `a85fd5b0ad69ba127490cea487e906d72443ba44716b824b601a61fce1655079`.
- Policy input: 128 active + 256 combined assembly/tool + 64 distal-finger points; 265 history values and 12 actions. Observation signatures reject incompatible weights.
- Ready baseline bank: `data/so101/banks_shared_v11`, 160 TRAIN + 160 independent EVAL, all loose. Bootstrap regenerates it locally from the pinned private runtime geometry release.
- Historical v9/v10 SGS banks remain incompatible. The larger staged bank is unfinished; the explicit baseline launch does not silently substitute it.

## Validation

- Strict native preflight: **31/31**, without a mechanics exception, including the updated 448-point check.
- Merged CUDA learner: **13/13**, on RTX 5080 / Torch 2.11.0+cu128. Finite gradients/losses, actor/critic/normalizer updates, SGS updates, atomic checkpoint restore and resumed counters pass. This tiny smoke reports no demonstrated task learning.
- The documented one-hour/64-environment launch passes the launcher's dry run.
- Core regression suite: **81 passed**; the subsequently changed bank-manifest publication path also passes its targeted test.
- Additional teleop sweep: 25 passed and 2 failures before deliberate interruption at the next gripper test (27 of 80 completed). The missing-headless-context camera failure passes when rerun with `MUJOCO_GL=egl`. The remaining physical regression is the off-center motor drop below; the rest of this broad sweep was not completed. This is not an all-green teleop qualification.

Local receipts: ignored `artifacts_local/training-snapshot-v11/` (`preflight.json`, `cuda-report.json`, `tests.log`, `teleop-tests.log`, `camera-egl.log`, `launch-dry-run.txt`, `sgs-report.json`). Native geometry/reward evidence predating the observation merge remains in `artifacts_local/drop-repair/`.

## Representative SGS replay

Only accepted states are listed. These are prepared starts checked with production actions and native physics, not demonstrated robot insertion paths. Partial starts additionally pass 0.25 ms replay. The bounded probe is not balanced or promoted as a training bank.

| Task / family | Split | Replay | Sampled material peak | Fine step |
| --- | --- | ---: | ---: | --- |
| horn / loose | eval | 1.00 s | 0.132 mm | not required |
| horn / loose | train | 1.00 s | 0.132 mm | not required |
| motor / loose | eval | 1.00 s | 0.005 mm | not required |
| motor / loose | train | 1.00 s | 0.005 mm | not required |
| motor / partial | train | 1.00 s | 0.123 mm | pass |

The generator was asked to stop when this host fell below its 128 MiB disk reserve. One native candidate did not return promptly, so the three owned probe processes were terminated after several minutes and their exit verified; completed receipts remain intact. Accepted receipts and failures remain under `data/so101/.banks_shared_v11_snapshot_probe.building`; no incomplete bank is presented as complete. Native regression tests also exercise staged loose starts for all five tasks. See [launch instructions](START_RUN.md) for the immediately available baseline and separate staged-pilot generation.

## Limits

Force-fixture insertions do not establish reachable arm grasps or learned behavior. Yoke's finer-step path approaches the 0.4 mm material gate (about 0.3996 mm in the prior audit). An existing motor drop regression at world offset `(0, -0.025, 0.06)` m remains outside its settling criterion: final speed **0.0158466 m/s after 5 s**, versus the required 0.01 m/s. Coordinates remained finite, native warnings were absent and peak speed stayed below 2 m/s. This physical limitation is retained for the separate CAD repair; no test threshold was relaxed. Hardware forces/friction and exact kit dimensions remain uncalibrated. All-loose sparse-reward training may make little task progress. Do not resume checkpoints after changing physical or observation identity.

The local host lacks durable-run disk headroom. No long training job was started; the user will launch on a suitable host. Original vendor CAD, generated banks/runs, native binaries and credentials are intentionally excluded from this source publication. Live Quest remains untouched.

## Files changed

Each row describes the accumulated publication relative to the local starting commit `60aeb33`; remote observation work is included rather than overwritten.

| File | Modification |
| --- | --- |
| `MEMORY.md` | Record merged development history and the explicit experimental-publication override. |
| `README.md` | Link current training, subtask and observation entrypoints. |
| `TASK.md` | Preserve remotely published observation/visualization status. |
| `User Intent.md` | Separate immediate training publication from final CAD review; retain the selected observation budget. |
| `assembly_policy/so101/bank_paths.py` | Separate staged-training and all-loose bank defaults. |
| `assembly_policy/so101/contact_audit.py` | Audit sampled original-material overlap and mating contacts. |
| `assembly_policy/so101/curriculum.py` | Prepare and physically qualify task-specific grasp/mating starts. |
| `assembly_policy/so101/env.py` | Gate terminal rewards on receiving-frame placement and material checks. |
| `assembly_policy/so101/evaluate.py` | Evaluate independent families and reject incompatible observation/bank identities. |
| `assembly_policy/so101/fastener.py` | Align fastener indexing/phase and shared native thread preparation. |
| `assembly_policy/so101/fit_geometry.py` | Build source-contained support contact geometry. |
| `assembly_policy/so101/goals.py` | Express mating targets in actual receiving-part frames. |
| `assembly_policy/so101/obs.py` | Use combined predecessor/tool surface budgets and distal finger samples. |
| `assembly_policy/so101/policy_demo.py` | Reject incompatible observation checkpoints during replay. |
| `assembly_policy/so101/resets.py` | Bind prepared banks to validator and native identities. |
| `assembly_policy/so101/scene.py` | Integrate shared mating contact geometry and native monitoring. |
| `assembly_policy/so101/subtask_curriculum.py` | Generate balanced resumable staged banks with native qualification and controlled publication. |
| `assembly_policy/so101/task_scenes.py` | Prepare native-qualified completion targets while retaining predecessors. |
| `assembly_policy/so101/workcell.py` | Apply shared contact settings and include physical sources in identity. |
| `assembly_policy/train.py` | Use staged-bank defaults and the selected observation signature/default budget. |
| `configs/cloud_stress.json` | Require strict task-mechanics preflight. |
| `configs/so101_scene.json` | Adopt v11 mating transforms, native contact coverage and table-impact settings. |
| `docs/ASSEMBLY_DESIGN_AUDIT.md` | Document assembly design audit and distinguish current evidence from historical results. |
| `docs/ASSEMBLY_MATING_REPAIR.md` | Document assembly mating repair and distinguish current evidence from historical results. |
| `docs/CURRENT_ENVIRONMENT.md` | Document current environment and distinguish current evidence from historical results. |
| `docs/NIST_POINT_VIEW.md` | Document nist point view and distinguish current evidence from historical results. |
| `docs/OBSERVATION_SAMPLING_REVIEW.md` | Document observation sampling review and distinguish current evidence from historical results. |
| `docs/OVERNIGHT_READINESS.md` | Document overnight readiness and distinguish current evidence from historical results. |
| `docs/POLICY_INPUT_VIEW.md` | Document policy input view and distinguish current evidence from historical results. |
| `docs/SGS_CURRICULUM.md` | Document sgs curriculum and distinguish current evidence from historical results. |
| `docs/SO101_MATING_INPUTS.md` | Document so101 mating inputs and distinguish current evidence from historical results. |
| `docs/START_RUN.md` | Give fixed-tag setup, explicit-bank smoke/training/assessment and optional staged-pilot commands. |
| `docs/SUBTASK_CURRICULUM.md` | Document subtask curriculum and distinguish current evidence from historical results. |
| `docs/TRAINING_SNAPSHOT_V11.md` | Record source identity, verification, representative starts and the complete changed-file inventory. |
| `docs/visualizations/README.md` | Document readme and distinguish current evidence from historical results. |
| `docs/visualizations/so101-mating-budget.html` | Preserve the published interactive 128/256 combined-mating inspection view. |
| `plugin/so101_contact_search.c` | Supplement missed motor/base contacts through existing native SDF geometry. |
| `scripts/audit_so101_review.py` | Provide audit so101 review diagnostics/validation. |
| `scripts/check_so101_curriculum_bank.py` | Provide check so101 curriculum bank diagnostics/validation. |
| `scripts/check_so101_sliding.py` | Provide check so101 sliding diagnostics/validation. |
| `scripts/cloud/run.py` | Select explicit qualified banks and scale SGS floor to available entries. |
| `scripts/merge_so101_curriculum.py` | Provide merge so101 curriculum diagnostics/validation. |
| `scripts/preview_so101_motor_families.py` | Provide preview so101 motor families diagnostics/validation. |
| `scripts/review_so101_matings.py` | Provide review so101 matings diagnostics/validation. |
| `scripts/so101_preflight.py` | Validate the selected 448-point production observation. |
| `scripts/so101_thread_check.py` | Use a corrected motor-local screw fixture and matched native search. |
| `scripts/trace_so101_sliding.py` | Provide trace so101 sliding diagnostics/validation. |
| `teleop/README.md` | Document readme and distinguish current evidence from historical results. |
| `teleop/tests/test_http_launch.py` | Update shared-profile expectations for teleoperation checks. |
| `tests/test_checkpoint.py` | Verify checkpoint under the shared model/input contract. |
| `tests/test_so101_contact_coverage.py` | Verify so101 contact coverage under the shared model/input contract. |
| `tests/test_so101_curriculum.py` | Verify so101 curriculum under the shared model/input contract. |
| `tests/test_so101_fit_geometry.py` | Verify so101 fit geometry under the shared model/input contract. |
| `tests/test_so101_material_gate.py` | Verify so101 material gate under the shared model/input contract. |
| `tests/test_so101_mating_fixture.py` | Verify so101 mating fixture under the shared model/input contract. |
| `tests/test_so101_mating_targets.py` | Verify so101 mating targets under the shared model/input contract. |
| `tests/test_so101_observation_roles.py` | Verify so101 observation roles under the shared model/input contract. |
| `tests/test_so101_subtask_curriculum.py` | Verify so101 subtask curriculum under the shared model/input contract. |
| `tests/test_thread_bench_frame.py` | Verify thread bench frame under the shared model/input contract. |
| `tests/test_workcell.py` | Verify workcell under the shared model/input contract. |
