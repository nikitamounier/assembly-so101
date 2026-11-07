# WowRobo linear-force insertion review

The latest review corrects incoming poses to the actual receiving features while
preserving source CAD and the canonical pose bank. Five separate probes cover
the side cover, shoulder connector, pin, upper arm and final module. This is an
aligned insertion fixture review, not training admission or a learned controller.

## Updated contact profile

The updated profile uses MuJoCo 4.0.0, the initial-run sparse solver settings,
and its accelerated grid-SDF surface-search implementation. The original
surface-tree and grid-bound headers are reused without algorithm changes. An
isolated adapter extends the old task-specific pair routing to the WowRobo
constituents. It does not modify the libraries used by existing experiments.

Distance grids are generated directly from source triangles, using closest
triangle distance and winding-number sign. Their pitch is 0.4 mm for structural
and servo meshes and 0.2 mm for the pin. No convex hull or occupancy-fill operation
defines these collision surfaces. Grid interpolation and signs on nonwatertight
source meshes remain approximations requiring explicit checks.

The native adapter supplements native SDF collision with the original visual
vertices and triangle centroids in both directions. Accelerated and exhaustive
paths must agree on the new geometry; merely loading a plugin is insufficient.
The generated scene retains the source body, joint, inertia and state layout.

## Aligned force and fixture assumptions

Force ramps to 0.5 N and is distributed over real upstream source-mesh vertices.
For each moving rigid body, parallel exterior forces have a resultant passing
through its center of mass. Loads are proportional to body mass, so the whole
subassembly receives the intended force without an applied turning moment.
The convex envelope is used only to select exterior load vertices; it does not
replace any collision surface. Cartesian viscous drag acts at each actual body
COM. The earlier free-joint translation damping acted at an offset model origin
and introduced artificial rotation; those old videos are superseded.

An explicit one-DOF slide fixture holds the corrected lateral position, rotation
and internal shape, like a virtual gripper. Helpers and receiver are static,
gravity is disabled, and axial travel integrates actual forces and SDF contact.
There is no prescribed axial trajectory or artificial seated joint stop. This
fixture temporarily locks the incoming internal pivots for the whole-shape
straight test; it does not alter the production component model or demonstrate
robot grasp retention.

The cover follows its existing source-CAD frame. Connector and upper/final
socket centers use documented small fit corrections; pin and final axes use
actual bore/triangle-plane directions. Frame alignment does not eliminate CAD
fit errors: the custom 2.7 mm shaft is wider than a modeled 2.5 mm threaded
opening, and the connector has about 0.435 mm residual sampled interference after following its true tilted rail axis.
Keep these blockers explicit rather than claiming perfect seating.

The guide builder checks original geometry poses after baking, then verifies
exact rigid translation and zero rotation over the intended travel. Every
video asserts negligible lateral and rotational drift. Separately, contact-off
free-root tests with all original internal hinges free verify that the corrected
loads and COM drag alone produce translation without artificial spin.

## Implementation and evidence

- Grid generation: `scripts/build_wowrobo_contact_grids.py`.
- Isolated backend and scene: `assembly_policy/so101/wowrobo_contact_backend.py`.
- Independent validation: `scripts/check_wowrobo_contact_backend.py`.
- Current force integration/videos: `scripts/probe_wowrobo_aligned_force.py`.
- Exterior pushes/COM drag: `assembly_policy/so101/wowrobo_balanced_push.py`.
- Explicit guide: `assembly_policy/so101/wowrobo_insertion_guide.py`.
- Free-root mechanics check: `scripts/check_wowrobo_push_mechanics.py`.
- Earlier free-body SDF probes: `scripts/probe_wowrobo_updated_force.py`.
- Frozen input: `outputs/linear-insertion-20261107/source-snapshot/`.
- Updated SDF assets: `outputs/linear-insertion-updated-20261107/`.
- Current aligned outputs: `outputs/linear-insertion-aligned-20261107/`.

The earlier `linear-insertion-20261107` contact-off/convex videos are historical
diagnostics. They are not the updated contact profile.

For a separate fresh environment, install `.[wowrobo-contact-review]`. That
profile pins the MuJoCo bindings, core and studio to 4.0.0 and includes the mesh
query dependencies. The commands below use the already verified sibling venv.

```sh
PYTHONPATH=. ../assembly-so101/.venv/bin/python scripts/build_wowrobo_contact_grids.py
PYTHONPATH=. ../assembly-so101/.venv/bin/python -m assembly_policy.so101.wowrobo_contact_backend --source outputs/linear-insertion-20261107/source-snapshot/scene.xml --grids outputs/linear-insertion-updated-20261107/grids/manifest.json --output outputs/linear-insertion-updated-20261107/scene.xml
PYTHONPATH=. ../assembly-so101/.venv/bin/python scripts/probe_wowrobo_aligned_force.py --step cover
```

Run validation before recording the final clips. Each delivered run records the
engine version, plugin count, sparse solver options, grid/native source hashes,
surface-search counters, force trajectory and actual stopping reason.

## Why the aligned probes stop

Matched continuations from the exact saved stalled states distinguish friction
from normal contact. `scripts/diagnose_wowrobo_insertion.py` runs baseline,
friction removed (zero coefficients and condim1), contacts removed, and10×drive
controls with the same guide, COM drag, distance fields and solver. Results are
in `outputs/linear-insertion-diagnosis-20261107/summary.json`.

The connector, pin, upper and final remain at essentially the same depth without
friction. Their source triangles contact horn rims, case bevels, receiver walls
or the pin's undersized thread; increased force adds only micrometres of contact
compliance. Every contacts-off control crosses its reference depth.

The cover initially sticks through tangential friction supported by preload at
a nearly zero-clearance rear plane. Friction removal advances it to33.09mm,
where the servo-case corner causes normal interference. That second barrier
persists without friction. Tiny rear relief opens one wall but closes diagonal
rails, so it is not a general fix.

These findings concern the rigid straight guide. Real rocking, elastic
compliance or a different entry path may matter. Passing a0.5mm overlap threshold
does not establish positive clearance for sliding. Source-CAD/receiver fit and
entry motion require correction or validation before training; simply reducing
friction or raising force does not repair the observed geometric barriers.

## Legacy fit-repair audit

The old v10–v14 Hiwonder model already addressed similar mating jams. Adopted
repairs preserved vendor CAD and changed indexed/axial mating references,
source-contained rail and bearing contacts, continuous surface support and
pair-specific compliant response. The private bore-relief grid experiment was
rejected. See `ASSEMBLY_MATING_REPAIR.md`, `SO101_FIT.md` and the historical
`configs/so101_scene.json`.

The generic WowRobo baseline reused acceleration and sparse solving but had no
explicit pair contacts: it assigns generic mu0.7/condim4 and relatively stiff
near-contact impedance to every constituent. The old motor/base pair instead
uses mu0.3/condim3 and a nonlinear low-near-contact-impedance response. Those
fit-specific settings and mating patches were not adapted to the new geometry.
Thus the previous ablations diagnose this generic prototype, not the complete
old validated contact setup. Before proposing new material cuts, map and
validate the relevant legacy mating repairs for the actual WowRobo interfaces.

Hiwonder follower CAD and RobotStudio/WowRobo leader CAD are different sources;
the numerical offsets and patch coordinates are not interchangeable. The new
custom pin also differs from the old analytic screw task. Full comparison:
`outputs/legacy-fit-audit-20261107/comparison.json`.

## Adapted candidate and validation

The explicit14pair candidate is `configs/wowrobo_mating_profile.json`; selected
legacy laws are mapped by actual interface role. Existing bidirectional source
sampling covers reviewed supports, so no extra collision boxes were added.
The canonical builder can export an isolated per-step profile with
`--mating-step upper --mating-profile legacy_mapped`; its default remains baseline.

42matched bench cases, source-path audits, blocked-wall tests and gravity
release diagnostics are saved in `outputs/legacy-fit-port-20261107/`. The cover
and final module improve, but none fully seats or passes retention. Thus the
candidate is not admitted for training or substituted beneath existing banks.

The original2.7mm pin is a frozennegativecontrol, separatefrom mainrun's newly
approved2.3×14.5mm fitpin. Genericfirmness failedfresh20Nsourceguards despite
passingstored-stalltests; the exact old hard-fastener normal response on the
pin/case pair fixes those bounded negative tests. The failed controls and
canonicalv2proofs remain preserved; no threadgeometry orvirtualhole was added.

Oldnumeric matingcorrections are not interchangeable: oldtaskexcludedlowerhorn,
newservohasboth; coherent±3/6degindexdoesnotchangefork/hornfit, and old+.6mm
lift worsens the newlowerhorncollision. See the per-step audits and final
`summary.json` for exact limits and the20second reviewed candidate reel.


## Consolidated derived-CAD candidate, November 7

The later frozen candidate in `outputs/insertion-fix-20261107/` supersedes the contact-only failed result above for guided insertion. All five reach full reference depth: cover55.003/55, connector65.008/65, pin32.10035/32.1, upper25.001/25, final25.003/25 mm. See that directory's README and summary for exact forces, candidate identities, controls and release evidence.

The fixes combine actual-feature registration, coherent predecessor frames, isolated measured printed-part CAD clearance and fine source-case contact fields. The candidate pin is Ø2.3×14.0 mm, reflecting the corrected connector head-stop position; older v2 assets are preserved. Final insertion retains the original mapped case/fork law without the wider compliance fallback. Original vendor CAD and cases remain preserved. All five material-obstruction controls still block.

The reel uses fixed receivers and a virtual axial gripper; it replays force-integrated native states. Upper gravity release reaches1.449° tilt and fails the strict1° retention benchmark. Contact parameters, provisional pin inertia and hardware fit remain uncalibrated; this is not training admission or complete unconstrained assembly qualification. The full 716-frame reel decodes without errors.
