# Gripper and environment interaction audit

**Goal:** Make both fingers physically touch, then expose failures through diverse native interactions and independent geometry checks.

This is the historical **v4** contact audit. The current **v5** part-centering update and top view are documented in [current environment](CURRENT_ENVIRONMENT.md); the101-case results below were not rerun for that layout-only update.

Verified November 5, 2026, profile **shared-side-v4-gripper-contact**, model signature `13c1d6b495ff3f43a8562b0ebb0f12534a3e63f0dfe8ec105cc33f1e966d5162`. Training and Quest consume the same physical factory. The live localhost18883 host has been restarted onto this identity, Disabled with no faults. Quest USB is disconnected; wearer verification remains pending.

**Verdict: closure is fixed; realistic full assembly remains unqualified.** A stable-looking hold or visible lift can still contain unacceptable mesh penetration. Failed grasps also reflect the tested grasp/approach strategy; they do not independently prove that the contact engine is wrong.

## Closure and contact changes

![Native render of previous and corrected closure](images/gripper-closure.png)

The previous closed command,−0.1rad, left7.77mm between the original visual finger surfaces. The old joint limit,−0.17453rad, still left1.91mm. The original CAD first touches near−0.19915rad. The shared command now closes to−0.23rad with a−0.24rad joint limit, letting native contact stop the fingers first. Both arms stop and reopen normally; independent original-triangle checks measure less than2micrometres of sampled material overlap in four native closure regressions.

An explicit distal triangle/SDF pair permits fingertip contact across the parent/child joint. The moving-jaw triangle subset comes directly from the pinned original STL. It excludes the intentionally overlapping bearing, adds zero mass and changes no visual geometry. There are no attachments, sleeping or pose correction. The actual kit's range, grip tape/pads and force response remain unmeasured; this is a simulation correction.

Finger contacts now use1msnormal response and solimp `[0.999,0.9999,0.0001,0.5,2]`. Priority1 selects finger contact parameters, including0.7sliding friction. Against the driver's0.8pads, this changes the previous equal-priority maximum to0.7; stored coefficients and mass/inertia remain unchanged. Explicit contact pairs retain their own parameters. These values improve numerical contact accuracy; they are not identified material properties.

The native version4 adapter spends16starts/20iterations only on finger SDF versus free-object trees, including tool descendants. Other pairs retain4/10; motor/base volume retains40/40; capacity remains50. This avoids increasing all collision searches and preserves the tested input deadline. NoSlip20 remains: a diagnostic NoSlip0/impratio10 candidate lost/ejected the block and motor despite zero numerical warnings. That comparison does not establish hardware fidelity.

For a12mmblock at the distal grasp, sampled finger penetration falls from0.935mm in the earlier parameter probe to0.141mm across three selected poses with the new model. A4mmsphere falls from2.008mm to0.197mm. These are sampled comparisons, not continuous nonintersection certificates. Small objects remain borderline, and a4mmblock at half timestep reaches0.273mm and fails its0.2mmlimit.

## Native interaction results

The final auditor executed **101 cases** with zero recorded numerical warnings. It keeps initialization, friction, gravity and timestep perturbations explicit. Full output includes environment identity, auditor SHA256, forces, drift, release, native depth and independent material checks. Auditor SHA256: `92566ea9682275b3b2cba8cb00f7e290328379585765ac0f000d98f3fccc23c9`.

| Interaction | Result | Meaning |
| --- | --- | --- |
|36primitive grasps |23pass,13fail |Boxes/spheres/cylinders;4/12/24mm; two insertion depths; friction0.35/0.7. Some deep grasps eject the object; small grasps exceed overlap/retention gates. |
|4half-timestep grasps |3pass,1fail |0.25mscomparison; not general convergence qualification. |
|4old-command comparisons |2pass,2fail |The old−0.1command holds the24mmobjects but loses both4mmobjects. These are controls, not four new-model failures. |
|14actual-part grasps |7pass,5fail,2invalid fixtures |Motor2/2, distal horn/screw, retainer2/2and distal yoke pass. Base, deep horn/screw and driver fail. Deep base/yoke fixtures start overlapped and are rejected without simulation. Screw uses the documented nominal M3 supplement, not exact vendor threads. |
|14drops |12pass,2fail |20/60mmdrops of seven parts. Base/floor sampled mesh penetration reaches0.428/0.661mm, exceeding0.4mm. Parts settle without numerical warnings. |
|14force pushes |Measurements only |0.5/2times nominal friction load. Curved parts can roll, so a universal displacement pass/fail gate would be misleading. |
|12position-only reach endpoints |10pass,2fail |Right motor approaches at50/80mm miss by5.1/20.5mm. This does not qualify full orientation or a collision-free path. |
|3whole-arm motor pickups |1pass,2fail |Native actuators approach, close, lift25mm, hold and release. Yaw0passes with0.166mmsampled penetration. Yaw90misses the grasp. Yaw180lifts but reaches0.632mmpenetration and is rejected. No policy is used. |

Prepared grasps fix the palm to isolate contact mechanics; they do not demonstrate tabletop pickup. They start open with nominal clearance and close without gravity, then hold under gravity and reopen. Invalid fixtures are reported separately. The driver has no independent mesh overlap qualification; both tested holds also fail retention. Original vendor visuals are checked for ordinary parts; screw checks use the nominal analytic host mesh.

Grasp gates require under1mmhold drift, under0.03radasymmetric rotation, bilateral finger forces without other support, over5mmrelease, finite states/clock and zero warnings. Native depth must stay below0.4mm. Sampled material penetration must stay below `min(0.4mm, 5% of object width)`. Sphere rotation is ignored; cylinder retention checks its axis. Geometry is sampled at the native peak, closure end and hold end; whole-arm checks sample phase ends. Brief penetration between samples can be missed. Drop floor checks use original mesh vertices sampled every10ms; driver drop checks have native contacts only.

Four renewed5/20Ndownward insertion probes at0.5/0.25msstill fail, ending12.8–14.9mmshort. The known31.8mmCAD cavity versus32mmnominal motor interference is unresolved. The guided screw-pitch check also still fails strict tolerance. These are retained failures, not passed assembly tasks.

## Online comparison and next work

- [MuJoCo contact and slip guidance](https://mujoco.readthedocs.io/en/stable/modeling.html#preventing-slip) calls for checking geometry, available contact wrench and normal force before tuning friction/solver settings. Its [collision filtering documentation](https://mujoco.readthedocs.io/en/stable/computation/index.html#collision-detection) explains why adjacent bodies need deliberate contact pairing. This informed the fingertip stop and force measurements.
- [IndustReal, section IV-E](https://arxiv.org/html/2305.17110v1#S4.SS5) checks mesh interpenetration independently and filters/weights learning updates. We use that validation principle here; its policy algorithm has not been added. Our yaw180lift shows why pose success alone is insufficient.
- [Factory](https://arxiv.org/html/2205.03532v1) distinguishes visual/collision assets, models fit tolerances and compares contact forces with physical data. Accurate source CAD alone does not validate our actual kit fit or forces.
- [Official SO101 calibration guidance](https://huggingface.co/docs/lerobot/so101) requires hardware calibration. No real robot was commanded or newly calibrated here.

Next priorities are measured motor/base clearance and insertion force; usable base/tool grasps and reachable approaches; small-part collision accuracy; then real gripper travel, normal force, slip and compliance measurements. Preserve these failed cases when changing geometry/contact parameters. Reliable assembly and sim-to-real transfer have not been demonstrated.

## Software checks and reproduction

- 74 Python regression cases pass, including the saved contact-loaded teleop deadline, both-arm closure/reopen and byte-identical training/Quest model initialization.
- 320 new versioned all-loose resets pass; old banks/checkpoints are incompatible. New reference/banks are generated locally and on setup, not committed.
- 13 CUDA/native learner smoke checks pass, covering192transitions and checkpoint restore. This is software health, not task learning.
- 30 mandatory preflight checks pass in the existing learner-debug profile. Its explicit screw-pitch exception is retained; strict task mechanics remain failed.

```bash
MUJOCO_GL=egl OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  .venv-teleop/bin/python -m scripts.check_so101_interactions
MUJOCO_GL=egl PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
  .venv/bin/python -m pytest -q tests teleop/tests
.venv/bin/python -m scripts.prepare_so101_workcell --count 32 --workers 2
MUJOCO_GL=egl .venv/bin/python -m scripts.check_so101_workcell \
  --out artifacts_local/interaction-audit/workcell
```

The auditor is diagnostic and exits successfully even when individual physical gates fail; inspect its JSON `summary` and cases. Use `--sections grasps+parts+table+reach+pickup` or a subset. Original baseline/probe reports are retained locally under `artifacts_local/interaction-audit/`; the final101-case run is `qualified/interactions.json`. Other receipts include `regressions.log`, `gripper-tests.log`, `closure-comparison.json`, `v4-prepare.log`, `preflight.json`, `learner-smoke/report.json`, `workcell/report.json` and `live-config.json`. Runtime data, vendor CAD and recordings remain ignored.

## Files changed

| File | Modification |
| --- | --- |
|[README.md](../README.md) |Point to the v4 model and audit. |
|[User Intent.md](../User%20Intent.md) |Record physical closure/audit requirement and v4 data compatibility. |
|[MEMORY.md](../MEMORY.md) |Record the selected approach, evidence and rejected alternatives. |
|[gripper.py](../assembly_policy/so101/gripper.py) |Build distal original-triangle stops and configure physical closure/contact. |
|[contact_audit.py](../assembly_policy/so101/contact_audit.py) |Independently measure original fingertip clearance and overlap. |
|[workcell.py](../assembly_policy/so101/workcell.py) |Apply the shared gripper and targeted search budgets; require native adapter4. |
|[so101_scene.json](../configs/so101_scene.json) |Version the model, gripper endpoints/contact and new reset/reference locations. |
|[so101_contact_search.c](../plugin/so101_contact_search.c) |Scope denser search to finger/free-object contacts. |
|[check_so101_interactions.py](../scripts/check_so101_interactions.py) |Add reproducible grasps, drops, pushes, reach and actuator-driven pickups with failure retention. |
|[backend.py](../teleop/backend.py) |Read Quest gripper endpoints from shared configuration. |
|[test_gripper_closure.py](../teleop/tests/test_gripper_closure.py) |Verify touch, force, reopen and policy/Quest endpoint parity. |
|[test_contact_stability.py](../teleop/tests/test_contact_stability.py) |Require the new contact adapter in existing drop regressions. |
|[test_http_launch.py](../teleop/tests/test_http_launch.py) |Require current identity in the real launcher check. |
|[test_workcell.py](../tests/test_workcell.py) |Check current capacity/search and command endpoints. |
|[CURRENT_ENVIRONMENT.md](CURRENT_ENVIRONMENT.md) |Describe selected v4 settings and physical limits. |
|[SHARED_WORKCELL_VALIDATION.md](SHARED_WORKCELL_VALIDATION.md) |Mark the v3 evidence historical and point to this audit. |
|[TELEOP_HANDOFF.md](TELEOP_HANDOFF.md) |Identify the restarted v4 host and preserve original packaging evidence. |
|[INTERACTION_AUDIT.md](INTERACTION_AUDIT.md) |Record this audit, sources, criteria, limitations and inventory. |
|[gripper-closure.png](images/gripper-closure.png) |Show native previous/corrected finger closure. |

Intentionally unchanged: original CAD/assets, layout/cameras, efforts/masses/inertias, PPO/reward/SGS equations, native thread equations and browser controls. Follow-up: USB wearer check and the physical/grasp/fit qualification above.
