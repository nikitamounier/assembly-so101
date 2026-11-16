# WowRobo leader: corrected photo-based components

The November 7 hardware instruction supersedes the previous Hiwonder assembly target.
Two **WowRobo SO101 followers with grippers and wrist cameras** manipulate five
structural pieces and one pin. The IMG_1622/IMG_1610 corrections supersede the earlier
four-modules-plus-pin interpretation.

| Piece | Included geometry | Photo evidence |
|---|---|---|
| Base + servo | Base_SO101 + servo 1 with its horns | IMG_1622 |
| Separate base side cover | Base_08p; installed after the base/servo is prepared | IMG_1622 |
| Empty shoulder connector | Rotation_Pitch_SO101 + Motor_holder_SO101_Base; no servo | IMG_1617, IMG_1620–1624 |
| Upper-arm module | Upper_arm_SO101 + servo 2 only | IMG_1610; explicit user correction |
| Final arm piece | Under_arm_SO101 + Motor_holder_SO101_Wrist + servos 3 and 4 | IMG_1610 foreground; explicit user correction |
| Assembly pin | Grip Ø12 ×28mm; shaft Ø2.3 ×14.5mm; total42.5mm | User-authorized fit-first design; [clearance evidence](WOWROBO_PIN_FIT.md) |

The small terminal wrist bracket visible in IMG_1624 is omitted, along with target
servos 5/6, the leader handle/trigger, and target gripper. The helpers retain their
grippers. The base electronics board, wiring and incidental fasteners are not
separate assembly pieces. Camera rigs belong to the helpers and do not count as
leader target pieces.

## Sources and geometry

Public upstream: [TheRobotStudio/SO-ARM100](https://github.com/TheRobotStudio/SO-ARM100),
revision `a758567c3978dfeefe282ede0500085a48fe8f78` (the project lists the exact
[WowRobo product](https://shop.wowrobo.com/products/so-arm101-diy-kit-assembled-version-1)
as a supplier). The upstream Apache-2.0 license applies; retain its attribution.
The user's photos are visual references and are not distributed in the bundle.

- Exact assembly-positioned solids come from `STEP/SO101/SO101 Assembly.step`.
  Printed parts common to leader/follower are reused; follower end-effector
  solids are excluded from the assembly target.
- The assembly names its base motor holder `Base_08p v18:1`; this is the matching
  holder in the positioned assembly, not an independently aligned guess.
- Separate source files, original transforms, source URLs and SHA-256 checksums
  are recorded in `cad/wowrobo/source-lock.json` and the generated manifest.
- The public hex-nut SO101 wrist-camera adapter is included as helper hardware.
  Its corrected registered attachment and camera module now come from a public
  camera-equipped follower, cross-checked against IMG_8569/IMG_8570. The earlier
  guessed mounting transform/adapter-face choice was wrong and is superseded.
  Actual camera dimensions and 75° FOV remain uncalibrated. See
  [camera comparison](WOWROBO_WRIST_CAMERA.md).
- `cad/wowrobo/assembly_pin.scad` preserves the two-cylinder construction with the user-authorized fit-selected dimensions (the originalØ2.7×22/50mm design is superseded), including
  the 0.01 mm overlap and 128 facets. STEP contains analytic cylinders; the mesh
  has 128 sides, one watertight boundary and no internal overlap faces.
- **STEP and STL use millimetres. OBJ and GLB use metres.** Individual component
  exports retain their common upstream assembly coordinate frame; they are not
  separately recentered. `components.glb` arranges a review tray.

The original servo contains one invalid OpenCascade solid, `XG-ZIJI_16:1`, in each
motor instance. Its original geometry is preserved and readable in the exported
STEP, rather than silently remeshed or changed to make a CAD-validity claim.
`source-servo-validity.json` records the original source check. The connector and
new pin are valid STEP solids; the three motor-containing compounds retain that
source defect. Rendered shape matching does not establish physical fit tolerance.

## Articulation and review

`review.xml` is a separate, portable MuJoCo **visual review scene** with six free
component roots, four internal target shaft hinges, twelve helper actuators and
two working wrist cameras. Target cases and horns remain separate constituent
meshes. Motors 2 and 3 have their horns attached to the printed yokes, so their
cases pivot; motors 1 and 4 have their cases attached to the base/link, so their
horns pivot. Compound STEP files capture one pose and are not rigid-body welds.

Gravity and all collisions are disabled in this review. It must not be used as a
training environment. Masses/inertias are display placeholders. No pin receiving
hole was specified, drilled or enlarged. Contact geometry, new task/reward logic,
reset banks, pin insertion paths and success checks remain to be qualified for a
new environment identity. Existing training, viewers and cluster jobs were not
modified by this component build.

## Reproduce

Run from the repository root. CAD dependencies stay in a separate environment:

```sh
uv venv .venv-cad --python 3.12
uv pip install --python .venv-cad/bin/python -r cad/wowrobo/requirements.txt
.venv-cad/bin/python scripts/build_wowrobo_components.py
.venv/bin/python scripts/review_wowrobo_components.py
.venv/bin/python -m pytest tests/test_wowrobo_components.py -q
```

The builder downloads missing pinned source files and rejects hash mismatches.
Corrected assets live under ignored `assets_local/wowrobo/components-v2/`; the previous
`components/` export is superseded. macOS may require
normal desktop graphics access for rendering. `review.xml` resolves meshes
relative to its location and can be moved with its `meshes/` and sibling
`../helper/assets/` directories. Screenshots are `components.png`, `workcell.png`,
`left_wrist.png` and `right_wrist.png`. The zip includes the six STEP/STL/GLB
exports, constituent meshes, the helper assets, portable review model, SCAD source,
manifest and this specification; it excludes original CAD downloads and photos.
