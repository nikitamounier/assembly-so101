# Native contact optimization — November7,2026

Human requests proceeding immediately with native collision optimizations, independently of the pending original profiling run; excludes mujoco-speed-up/bare branches. Original CAD/grid fields, supplement samples, contact merge order and guards retained.

Implemented candidate: helper-versus-target grasp16x20, helper-versus-helper4x10, explicit mating/passive-stop and target-target40x40; bounded thread-local supplement memo keyed on exact absolute geometry poses/margin/model signature+buffer; tighter4-voxel conservative grid-bound blocks and8-sample tree leaves in isolated candidate plugin copy. Cache never skips ordinary native collision search.

Local C++8-thread memo check PASS: hit, one-bit-scale motion, margin, signed-zero, ordered-pair and model-signature invalidation. Initial native adapter compiles. Actual baseline/candidate replay is running on retained3598717,2CPU/12G, newhostGPUCB7C; same host for both phases. No training worker or allocation mutation. Remote experiment directory is ROOT/native-contact-optimization-20261107; human explicitly approved candidate upload to that exact location after approval review rejected first attempt.

Pre-optimization builder preserved as backend-before.py. Candidate build source and headers captured under candidate/plugin plus backend-candidate.py/wowrobo_contact_cache.h. Frozen remote runtime remains0.25ms/160steps so this comparison isolates collision changes; main owner separately validates0.5ms/80steps release.

STATUS: qualification/timing pending. No live deployment authorized by a qualification receipt yet; main-run owner warned to exclude unqualified local edits.

## Selected exact candidate and results

- Lower-helper-budget trial FAILED exact parity:60/216arrays differ after nonzero commands, despite matching short-panel rewards/guards. Preserved separately as backend-candidate.py and lower-budget-comparison.json; NOT selected for default. Canonical per-role defaults remain40x40, with distinct names for later independent qualification.
- Selected backend-exact.py uses exact-pose memo plus4voxel bound blocks/8sample leaves. FullCAD, grid_eval, surface samples, original-order evaluation, contact laws and guards unchanged. Exhaustive mode2 bypasses memo reads for an independent oracle. Canonical header changes now in assembly-so101/plugin; nativebuilder tracks source/header hashes.
- Paired12control-interval replay:82.1933447→61.1006466seconds,1.345212x throughput (~25.66%less wall time);216arrays byte-identical, including full160substep trajectories, contacts/forces, reward, safety, observations. Cases loose, covernear, upperheldnominal+variation;3intervals each, nonzero commands afterinitialhold.
- Exhaustive87pose oracle vs2thread tree:435arrays byte-identical, max243contacts. All5tasks/prototypes plus deterministic freebody translation/quaternion perturbations.
- Existing7surface-acceleration regressions PASS15.64s. Local C++memo invalidation8threadcheck PASS.
- Timings are one sequential paired panel on shared retainedGPUCB7C, not confidence intervals or fullproduction throughput. This comparison retains original0.25ms/160steps; does not multiply timestep gains into an observed combined speedup. Tighter block cache increases bound-table storage8x; productionworker/RSS remains owner qualification requirement.
- Main-run owner received qualification and explicit request to incorporate selected exact candidate after its independently owned0.5ms release. Livecontroller/monitor unchanged bythischat; productiondeployment/newPPO verification pending in mainrun. Do not interpret this as recoveryof1.72xNISTefficiency.
