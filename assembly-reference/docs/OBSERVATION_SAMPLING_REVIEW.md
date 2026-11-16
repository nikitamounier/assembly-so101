# Observation sampling and task switching review

**Goal:** Ground a denser, more useful policy observation proposal in AlphaNIST and verify current base/gripper transforms before changing sampling.

Reviewed November 6, 2026. This is a source review and read-only native audit, followed by a proposed experiment. No production sampling, physics, policy weights, live Quest state or running training job is changed.

**Subsequent implementation:** the user then explicitly requested fixing the SO101 inputs to active-part/mating geometry and better finger locations. [Implemented part/mating inputs](SO101_MATING_INPUTS.md) supersede the allocation proposals below: the production total remains 320, with unrelated object/tool context removed, local receiver patches and distal-finger coverage. The earlier research remains historical evidence, not the current implementation specification.

## What the primary paper establishes

[AlphaNIST, sections III-A and IV-D](https://arxiv.org/html/2610.13354v1#S4.SS4) describes surface point clouds of the movable part, mating geometry and robot, plus proprioception and previous actions. There is no explicit discrete task identifier in the policy input. Figure 3 shows a flattened `3 × n_pts` geometry input, a 256/128 two-layer ELU encoder and separate SimBa actor/critic networks. The 128 there is an encoder width; the available methods text does not specify an optimal 128/128/64 point allocation or a contact-region sampling ablation.

The [independent AssembleNIST implementation](https://github.com/suniique/assembly-nist/blob/main/assembly_policy/rl/obs.py) defaults to 128 movable-part, 128 mating-geometry and 64 gripper points. Its `mating_points` samples the fixture and, for gears, neighboring mating gears. These numbers are implementation defaults, not a paper-proven optimum. Our SO101 adaptation retains the total but spreads the 128 fixture/context slots across every other observed body, including screwdriver sub-bodies. This meaningfully reduces receiving-geometry coverage.

### Gripper follow-up: exact upstream behavior and inferred intuition

The user questions whether even the current gripper budget is excessive. Re-reading upstream `finger_points` and `PointTables` confirms **64 total for one Franka gripper, 32 on each finger**, selected from finger collision geometry only. It samples the left finger once and constructs the right by symmetry; runtime coordinates shift with the two jaw positions in the TCP frame. The full palm and arm are absent from the gripper point set. The training CLI independently confirms `--n-grip=64` as its default. There is no explicit inner-pad-only selection and no documented gripper-count ablation establishing that 64 is necessary or optimal in the reviewed code/README.

The design intuition below is an inference from the representation, not a quoted author rationale. Points put object, mating and finger surfaces in one coordinate system, allowing the network to learn spatial relationships such as a part being between jaws or a fingertip approaching a rim. Proprioception already supplies robot/finger state, but surface points explicitly encode contact-capable shape rather than requiring the network to reconstruct it from joint angles. These are a deterministic geometric expansion of known CAD/jaw state, not 64 independent measurements. That redundancy may improve learning, but does not imply that more finger points are always useful.

Our present 64 points cover two SO101 grippers: 16 each on two palm/fixed-finger bodies and two moving-jaw bodies. This is not directly equivalent to upstream's 64 finger-only points. In particular, sampling a broad palm/fixed-finger visual body can spend some of its quota away from contact-capable finger geometry. Prioritize meaningful finger surfaces before increasing count, including tips and inner/outer contact surfaces useful for grasping, pushing and avoiding collisions. An inner-pad-only mask would remove useful geometry for other contacts.

**Revised recommendation:** withdraw the earlier proposed increase to 128 gripper points. Keep the current 64-total setting as a control and test **32 total**, eight per fixed/moving finger-related body, focused on actual finger contact geometry. Neither number is selected as proven optimal. Additional geometry budget should go first to receiving/mating geometry, with the lower gripper allocation evaluated on the same held-out manipulation tasks.

[AlphaNIST, section IV-B](https://arxiv.org/html/2610.13354v1#S4.SS2) uses SGS to allocate reset entries and thereby episode/task exposure. SGS is not a surface-point allocation or attention algorithm. Its cache includes loose, grasped, transport, near-mating and partial-assembly states. The weakness of our all-loose-only starts cannot be attributed solely to point count, and the paper does not establish that dense progress shaping is required.

For multi-pair experiments, [sections IV-C and V-D](https://arxiv.org/html/2610.13354v1#S5.SS4) describe separately trained policies with several active assembly pairs and reward only when all are assembled. The policy can change which pair it works on without a prescribed order, explicit planner or policy switching. This differs from our fixed-single-task episodes.

## Current observation and placement audit

The current source/live scene is `shared-side-v8-five-tasks`, model signature `bd1478216658d1061a078051b3561332b8211bfea005ac0b1f90332e2c5c4cd4`, configuration SHA256 `3f2ed24533c1b13aa5b80fed9d40ec0ff5295bc886847d6517e3775c2ad2ec4d`. There are five tasks, eleven observed object/tool bodies and 320 points. In the motor task, the base has 13 context points. The previous visualization is a frozen historical six-task state with 12 base samples.

The audit captures current live qpos read-only, uses the production sampler, and inverts its encoded coordinates with independently read MuJoCo transforms. It then measures distance to the actual native visual triangles/primitive surfaces. It repeats with asymmetric shoulder rotations and different left/right jaw apertures. All 640 tested points align with their corresponding surfaces: maximum distance 0.000051 mm in the live state and 0.000056 mm in the asymmetric state. Base maxima are below 0.000015 mm. These checks find no base/gripper rigid-transform error; they do not establish physical-kit calibration, ideal workcell reach or sufficient feature coverage.

Geometry uses the **left gripper body/palm frame**, with metre values scaled by ten to decimetres. Proprioception uses `left/right_gripperframe` site poses in world coordinates. Those are different origins: the site is 98.445 mm from the palm origin, with palm-local offset `[-7.9, -0.218121, -98.1274] mm`. Calling either one simply “the gripper position” is ambiguous. This is a naming/frame-design issue, not evidence of an incorrect transform. Moving-jaw points use actual separate moving-body transforms each step; the unused `finger` sampler argument does not freeze their geometry.

The current source has no quotas for inner pads, fingertip edges, cavity walls or the entry rim. Area weighting can devote samples to large exterior/structural surfaces, even when smaller surfaces matter more for the task. Sixteen samples on each palm/fixed-finger or moving-jaw body do not guarantee adequate contact-pad coverage.

Receipts: ignored `artifacts_local/observation-review/geometry-audit.json`, `audit-mm.log` and `audit_geometry.py`. The first audit used trimesh closest-point queries directly in metres and falsely reported small-CAD surface errors because its absolute geometric tolerances treated small triangles poorly. Its failed receipt is preserved as `audit-metres-first.log` and `geometry-audit-metres-first.json`; the independent distance queries were rerun with both triangles and query coordinates scaled to millimetres, then converted back. No production coordinate or sampler was modified to pass the check.

## Revised 512-point reference experiment

This is an engineering hypothesis for the motor stage, not a result established by AlphaNIST.

| Role | Current motor task | Candidate | Allocation rationale |
| --- | ---: | ---: | --- |
| Moving motor | 128 | 96 | 32 broad shape, 32 grasp surfaces, 32 insertion/leading features. |
| Receiving base | 13 | 256 | 64 broad shape, 192 cavity walls, entry rim and seating surfaces. |
| Both grippers | 64 | 64 | Keep total fixed as a control; improve actual finger-surface coverage, and compare a 32-total variant. |
| Other parts and tool | 115 | 96 | Stable per-body slots; avoid repeatedly allocating full budgets to the screwdriver's small sub-bodies. |
| Total | 320 | 512 | More receiving detail with fewer active-part points and no gripper-count increase. |

This reference revises the withdrawn 96/192/128/96 proposal. A 32-gripper variant with all other groups unchanged totals 480; a constant-512 comparison can instead transfer the freed 32 slots to the receiving region. Record which comparison is used rather than attributing a simultaneous redistribution solely to the gripper count. The total 512 and its allocation remain experimental choices, not a committed implementation.

For later tasks, receiving geometry must mean the verified assembly interface, not automatically the base. Define reviewed CAD-local regions for each part/interface and the relevant predecessor assembly. Retain enough global geometry for orientation, grasping and nearby obstacles. Tool bit/button/grip features need explicit coverage for the screw task. Keep all physical objects dynamic and represent articulated buttons/jaws with their own transforms.

Generate an area-weighted surface candidate pool offline, then apply deterministic **farthest-point sampling within each designated region**, with fixed seeds, indices and quotas. [PointNet++, section 3.2](https://arxiv.org/html/1706.02413#S3.SS2) uses FPS for more even set coverage than random sampling. This supports the coverage method, not a claim of improved assembly success. Global FPS alone cannot guarantee cavity/pad coverage; the region quotas provide that. Selecting functional regions introduces CAD/interface knowledge beyond the paper's documented generic input description and must be reported as our change.

Keep the present ordered MLP encoder for this first comparison. Cache samples in body-local frames and reuse their order as bodies move. Do not resample every frame, sort points by current distance, or implement attention-like sampling that silently changes the meaning of an input index. Dynamic sampling or variable numbers of bodies would need a suitable set/object encoder and separate evaluation. No PointNet++ network migration is proposed merely because FPS is used offline.

For clarity, show palm origins, named TCP sites, moving-jaw bodies and sampled contact regions separately in the next visualization. Keep units and transforms explicit. A future switch to a TCP-centered geometry frame is a separate observation-version change, not a needed repair of the currently verified transform.

## Comparison and compatibility

Run three controlled conditions on the same native environment, validated reset distribution, held-out starts, seeds and interaction budget:

1. Current 320 samples and allocation.
2. Smart 320: 60 active, 160 receiver, 40 grippers and 60 context. This matches the revised 512-role proportions while holding total count fixed.
3. Smart 512: 96 active, 256 receiver, 64 grippers and 96 context.

Separately compare 32 versus 64 total gripper samples with the other regions controlled, as described above. Fewer points must retain enough contact geometry for reliable grasps/pushes/releases; known joint state alone does not establish equivalent learning performance.

Measure point coverage on reviewed mating/pad regions first, then independent task success, final pose error, release/stability, safety outcomes and inference/collection cost. Control candidate pools and sampling seeds so the count comparison is interpretable. More points alone cannot repair an infeasible fit or invalid contact dynamics.

At unchanged encoder widths, increasing from 320 to 512 adds 147,456 weights to each first point-encoder layer, 294,912 across actor/critic, about 10.3% over the current parameter total. Actual timing still requires measurement. Changing even the same-sized allocation changes feature semantics: record an observation schema/version, seeds, task/body order, region definitions, frame and sampler hash in training/checkpoints/evaluation. Do not treat old 320-point weights as compatible merely because tensor dimensions match. A 512-point input also changes layer shape and requires a new model or explicit initialization experiment.

## How active parts switch at runtime

**Current implementation:** `Trainer._provider` samples an entry and returns its task ID. `LocalVecEnv` installs that task at episode initialization/reset; `PointTables.which[task]` selects the fixed active/context table. The task is fixed during that episode. On success, safety termination or timeout, vector training loads a new sampled reset state and task. It does not continue the assembled physical scene into the next task. `so101/policy_demo.py` similarly runs one requested task until termination. Quest A loads prepared completed-task previews and resets the scene; it is not a learned sequencer or evidence of executed assembly.

**Recommended first continuous sequence:** use an explicit task coordinator outside the low-level policy. It starts with active motor/receiver base. Only after the unchanged native placement, release, contact, safety and stability validator succeeds does it select active horn and its receiving interface, then screw, retainer and yoke. It changes the sampling table and task-specific reward validator while preserving qpos, qvel, simulation clock, previous actions and real observation history. Reset only stage-specific success counters and manage stage/full-episode termination explicitly. Do not use `set_states` or Quest prepared-scene loading for this handoff; both initialize/reset state/history.

The coordinator changes which fixed observation slots have active/receiver roles; it does not move physical objects or reload neural weights. Motor points move into the appropriate predecessor/context role at the next task. Train/evaluate the policy on actual predecessor handoff states before claiming continuous sequencing works. If autonomous task ordering like AlphaNIST is the goal, instead design and train a joint multi-pair observation/objective with all required geometry represented; that is a larger task formulation change, not a small active-label toggle.

## Files changed and follow-up

- `docs/OBSERVATION_SAMPLING_REVIEW.md`: paper/implementation separation, native audit, candidate sampling and task-switch contract.
- `docs/POLICY_INPUT_VIEW.md`: link the new review while preserving the historical visualization's identity.
- `User Intent.md`: record requested research/placement review and distinguish the proposal from selected implementation.
- `TASK.md`: append the review's progress without replacing other work.
- `MEMORY.md`: dated findings, rejected assumptions and session handoff.
- Ignored `artifacts_local/observation-review/`: primary-source snapshot, independent upstream sampler copy and native audit code/receipts.

Follow-up: implement and visualize versioned candidate samples if selected, verify region coverage and train the controlled comparisons. No sampler or continuous sequencer has been implemented by this review.
