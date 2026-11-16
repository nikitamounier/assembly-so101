# Initial WowRobo workcell and tangent layouts

This is the first scene/reset-layout step requested November 7. It uses the
corrected `components-v2` CAD, not the historical Hiwonder task.

## Fixed layout

- Two camera-equipped SO101 followers face one another along world X.
- The rear X edges of their **printed base footprints** are at −0.360 m and
  +0.360 m: **0.720 m apart**. Rear-mounted electronics are excluded from that
  measurement. Joint origins are different and are calculated from the native
  mesh footprint, rather than silently using 720 mm between joints.
- The target base is fixed on the table, forward toward the right arm (+X).
  Its rear edge is at −0.060 m, **0.300 m from the left helper's rear edge**.
- Both helpers begin parked above the workcell, with separated open grippers;
  wrist cameras and the four existing target servo pivots are retained.
- The table is 1.30 × 0.92 m. The five loose pieces are placed on its surface.

## Confirmed pickup assignment

| Order | Loose component | Assigned helper | Local grip reference |
|---|---|---|---|
| 1 | Separate U-shaped base side cover | Left | Short end rim of the cover |
| 2 | Empty two-part shoulder connector | Right | Lower printed spine, below the empty servo socket |
| 3 | Two-cylinder pin | Left | Center of the Ø12 mm grip section |
| 4 | Upper arm with one servo | Right | Servo 2 case |
| 5 | Final arm piece with two servos | Left | Exposed servo 3 case |

The user explicitly confirmed both this order and the tangent definition.

## Tangency is defined at the grasp center

For assigned shoulder-pan axis `a`, grip center `g`, and horizontal grip-span
unit vector `t`, the sampler enforces:

```text
r = g.xy - a.xy
t.z = 0
dot(r, t.xy) = 0
```

Each component carries an explicit CAD-local grip center and grip span. The two
servo modules use the actual servo case and shaft-axis geometry. Whole-part yaw
is solved from the sampled **grasp center**; simply rotating a bounding-box
center would not preserve the constraint for off-center servo grips. The grasp end faces the assigned helper and is the nearest end along its approach.
The long arm links extend radially **away** from that helper. The selected servo
assembly must have the minimum radial support projection of the whole module;
this is checked to1e-8m floating-point tolerance, not merely inferred from yaw. The pin lies on its side,
with its transverse grip span tangent, its thick grip nearest the arm and its
shaft extending radially away.

Blue spans show left pickups; orange spans show right pickups. Three thin gray
witness lines mark the left helper rear, assembly rear and right helper rear.

## Randomization

`configs/wowrobo_layout.json` exposes all placement defaults. The grip center can
move throughout the assigned arm's forward annular sector, rather than staying
in a fixed per-piece bin: radius 150–310 mm (240–310 mm for the two long modules),
angle ±82° about the arm's forward direction. These are configurable starting
ranges, not measured reachability limits. Rotation follows the grip constraint.

Every candidate is grounded using the transformed CAD surface minimum. Rejection
sampling retains the table boundary and a conservative 8 mm XY bounding-box
clearance between components, the fixed assembly base and helper footprints.
The opposite arm's initial gripper is parked clear; no supporting grasp is
prescribed. Saved layouts are deterministic by seed.

The viewer contains 32 initial states: seed 0 is the nominal scene, seeds 1–31
are randomized layouts. These are layout examples, **not SGS training/evaluation
starts**; the inherited viewer's archive suffix is a transport convention.

## Implementation and verification

- Factory and sampler: `assembly_policy/so101/wowrobo_layout.py`.
- Reproducible build: `scripts/build_wowrobo_layout.py`.
- Native render: `scripts/render_wowrobo_layout.py`.
- Local output: `assets_local/wowrobo/initial-layout-v2/`, including `scene.xml`,
  `layouts/`, per-seed grasp receipts, nominal and randomized renders.
- Native XML contains an `initial_scene` keyframe for the complete parked nominal
  state. Consumers should reset through the sampled state or that keyframe.
- Viewer uses the same compiled native model/poses for Three.js and MuJoCo.
- Tests check the two measured rear-edge offsets, right-facing assembly base,
  confirmed pickup order, actual grasp-frame tangency, table grounding, planar
  clearances, seed reproducibility, position diversity over 128 samples, parked
  tip separation, and exported nominal-state parity.

```sh
PYTHONPATH=. .venv/bin/python scripts/build_wowrobo_layout.py
PYTHONPATH=. .venv/bin/python -m pytest tests/test_wowrobo_layout.py -q
```

This first step is **geometric layout inspection**. It retains the CAD review's
disabled gravity/collisions and provisional inertias; no dynamics, stable resting
contact, IK reachability, successful grasp or assembly claim follows from these
checks. Those qualifications precede using the scene for training. Camera
calibration and the original servo STEP-topology limitation remain as documented
in `WOWROBO_COMPONENTS.md`. Existing training jobs were not modified.


Camera attachment was corrected using the public registered camera-equipped
follower and the new close-up photos; see [camera comparison](WOWROBO_WRIST_CAMERA.md).

The two-servo forearm is turned end-for-end per the latest correction: servo 3 is the left helper pickup end; table-up and tangency are preserved.

The side cover is reversed end-for-end, with its opposite rim facing the left helper. The shoulder connector retains its original orientation; the earlier request naming it was corrected by the user.
