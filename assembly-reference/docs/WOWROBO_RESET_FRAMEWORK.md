# Object-relative reset bank

The framework prepares **25 starts × (60 TRAIN + 12 EVAL) = 1,800 accepted resets**. The count is configurable. TRAIN uses24 easy,24 medium and12 hard slots per start; EVAL uses12 uniformly stratified difficulty slots. Independent split/stage seeds prevent sampling leakage; retries retain their original difficulty band.

## Geometry before simulation

Upstream AssembleNIST composes object-relative grasps in `assembly_policy/sim/resets.py::_world_grasp`, obtains analytic or antipodal grasps in `any_grasp`, solves IK and rejects failed contacts/retention. Its `sim/grasps.py` builds opposing-surface candidates and tests closure and retention under gravity changes. `scripts/build_banks.py` distributes independent task/family jobs over CPU workers.

Our implementation reuses the existing source-mesh SO101 jaw calibration and5DOF IK. A library entry stores `T_object,gripper`, the actual rotational jaw opening, site-to-contact-center offset, source anchor and witness status. The runtime computes:

```
T_world,gripper = T_world,anchor * T_anchor,gripper
```

The anchor is the motor case for an active articulated module, the selected printed body for a cover/connector/support, or the pin. A freely rotating bracket is not the grasp anchor. This preserves the motor/gripper relationship when the bracket rotates. The existing stage sampler handles tangential loose placement, stage-specific insertion/transport offsets, coherent connector/receiver rotation, and ±3% friction draws. IK proposes a reachable arm configuration. A short native check then closes the actual jaws and witnesses material clearance and retained finger capture. Movement is allowed; equilibrium is not an admission gate.

A library contains multiple measured seed variants, bound explicitly to their stages and arms. Equal records deduplicate by content. Motor-grasp retries may select another witnessed grip on the same servo case, reusing a successful floor/transport grip at another stage; support and small-part grasps retain their named feature. A source pose or an IK solution is **not** a validated grasp: every sampled variation must independently pass physics. The current prepared source witnesses pass22/25 starts;21/22/23 are marked unqualified. This framework does not claim those three grips are repaired, nor invent an already validated surface-grasp library for them.

## Commands

Run from the canonical repository with its configured Python and `PYTHONPATH=.`:

```sh
PYTHONPATH=. ../assembly-so101/.venv/bin/python scripts/build_wowrobo_reset_framework.py prepare \
  --output outputs/reset-framework-1800-v1 \
  --nominal assets_local/wowrobo/photo-resets-repaired-cover-support-v2

PYTHONPATH=. ../assembly-so101/.venv/bin/python scripts/build_wowrobo_reset_framework.py run \
  --output outputs/reset-framework-1800-v1 --pilot \
  --stages 00_loose 18_upper_floor_grasp --workers 2

PYTHONPATH=. ../assembly-so101/.venv/bin/python scripts/build_wowrobo_reset_framework.py run \
  --output outputs/reset-framework-1800-v1 --workers 4 --max-attempts 20

PYTHONPATH=. ../assembly-so101/.venv/bin/python scripts/build_wowrobo_reset_framework.py status \
  --output outputs/reset-framework-1800-v1
```

Preparation freezes measured seed snapshots into a new reference collection, writes `grasps.json`, resizes the inherited sampling configuration and records identities in `plan.json`. It refuses to overwrite an existing preparation. Use a new output path for revised sources. The existing `v1` output is already prepared; do not run `prepare` again at that path.

The coordinator prepares one common native profile before launching workers. Workers are independent subprocesses, one per stage, with separate attempts, logs, review collections and qualified archives. A coordinator lock prevents two invocations racing on the same plan. Worker count controls CPU parallelism; this is not a GPU workload. No cluster/training jobs or existing running experiment identities are changed.

Rerun the same command to resume. The existing builder verifies input/config/state/control hashes and skips accepted slots. It tries at most `--max-attempts` candidates per full-bank slot; failure leaves a hole and an incomplete bank. Pilot mode draws one TRAIN and one EVAL candidate per chosen stage without retrying; it measures rejection, not quota completion. Review and physical archives remain separate. Pose noise amplitudes are inherited rather than silently reduced to fill quotas.

Only after every full-bank stage reaches60/12 does the coordinator seal `bank/manifest.json`, the common physical `scene.xml`, and flat per-stage NPZ files. It checks model/state-layout agreement, exact quotas and duplicate physical states across splits. Each NPZ includes per-row `friction_multipliers` with named columns; consumers must apply them at each fresh contact set, using `wowrobo_friction`, to reproduce the admission physics. Exporting these parameters does not modify or hot-load any running trainer.

`progress.json`, `pilot-progress.json`, stage logs, attempt receipts and qualified archives expose actual results. No failed candidate or nominal reference is counted toward1,800. The current framework starts from measured grasps; automated antipodal surface-library discovery and its own broader robustness campaign remain future extensions.

## Verified result

Twenty focused framework, resumable-bank and minimum-admission tests pass; Ruff passes for new and modified generator modules. The native pilot completed8 draws with7 accepted:

| Start | TRAIN | EVAL |
|---|---|---|
|00_loose|Accepted|Accepted|
|18_upper_floor_grasp|Accepted|Accepted|
|22_upper_half|Rejected|Accepted|
|28_final_transport|Accepted|Accepted|

The rejected22 TRAIN candidate failed retained bilateral capture and native material clearance (`held_part_escaped_grip`, `held_part_not_captured_by_both_pads`, `unintended_material_overlap`). Its bilateral-contact fractions were LEFT0.5375/RIGHT0.055; accepted22 EVAL retained1.0/1.0. This shows neither every reference nor every variation should be assumed valid. A later full build retries missing slots; this pilot deliberately kept one draw per split.

Real resume of00/18 kept the same2 attempts per start and generated no additional candidates. The accepted physical archives share model signature `f6bc5ea4fdc4e721c7dd02ac6aa5d4cc79fa971f6f167a17a3438c6bbbc3bef6`, state spec32767 and saved friction categories. Paths: `outputs/reset-framework-1800-v1/{pilot-progress.json,pilot-verification.json,pilot-run.log}` and per-stage attempt receipts. These are bounded generator results, not a full1,800-state bank or workspace-wide grasp robustness proof. Full generation was not launched.

The cross-project research-runbook append was not executed: automatic approval review reported workspace credits exhausted. Project-local intent, MEMORY and this guide retain the implementation/evidence; the external runbook still needs updating after the review service becomes available.


## Approved full build: November7, version4

The user approved the7-variation cover preview and requested the full1,800 accepted states. Current durable plan: `outputs/reset-bank-1800-20261107-v4`; source25/25 validated prototypes. The3-worker generator prioritizes21/29/00, then remaining starts. `live-progress.json` counts actual atomic accepted/rejected receipts every30seconds; `supervisor.json` records ownership and terminal status. No running trainer is modified.

Preparation refreshes actual printed far-patch/receiver witnesses in six support phases. Historical pose metadata was stale by up to millimetres, causing a6.75degree hinge snap even for tiny noise. Version2 was stopped and preserved as a diagnostic; version3 preparation rejected an extra static-contact gate inappropriate for moving prototypes. Version4 preserves the measured signed receiver-plane offset (including honest instantaneous separation), rather than forcing moving references into an old zero-gap pose. Native overlap and bilateral-retention gates, threshold, friction ranges and approved noise amplitudes are unchanged. Measured source velocities are retained and free linear velocities rotate with their frames.

Read-only continuity check for21/22/23/29/30/31: zero noise preserves qpos/qvel/controls within3.7e-16. At level1e-6, maximum hinge delta2.25e-8rad and support-plane residual1.08e-10m; all12 samples pass geometric constraints. Evidence `outputs/reset-support-receipts-20261107/continuity-v4-assertions.json`.21 focused tests pass, including actual hinge/case baseline-offset fitting.

Durable supervisor PID is saved in `supervisor-launch.json`; it adopts the active coordinator, resumes missing slots with attempt limits20/40/80/160, and never pads or relaxes the bank. On exact1,800 quota and successful duplicate/model checks it publishes `assets_local/wowrobo/randomization-bank-1800-valid-v4` and starts its viewer onlocalhost8878. Persistent failures leave explicit incomplete status rather than claiming completion. Existing cover preview8877 and reference8876 viewers remain intact.
