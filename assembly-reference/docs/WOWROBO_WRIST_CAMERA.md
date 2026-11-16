# Wrist-camera comparison and corrected attachment

User supplied close-ups IMG_8569.HEIC and IMG_8570.HEIC show an exposed square PCB
on a raised bracket attached to the fixed gripper, with the lens aimed forward
and down toward the jaws. The camera is not below the gripper and does not move
with the moving finger.

## Public sources checked

- [WowRobo SO101 kit](https://shop.wowrobo.com/products/so-arm101-diy-kit-assembled-version-1):
  identifies the hardware kit, but no vendor-certified complete camera-equipped
  STEP assembly was located in this search.
- [RobotStudio SO101 hex-nut camera adapter](https://github.com/TheRobotStudio/SO-ARM100/tree/a758567c3978dfeefe282ede0500085a48fe8f78/Optional/SO101_Wrist_Cam_Hex-Nut_Mount_32x32_UVC_Module):
  provides the actual adapter STEP/STL and installation instructions.
- [desmondzee/ICL-so101 camera-equipped follower model](https://github.com/desmondzee/ICL-so101/blob/01782763fb8ad2473af96a345f91bfc319300326/sim/so101/so101_new_calib_camera.xml):
  complete MJCF and matching URDF with adapter, camera geometry and optical frame.
  This is a public community model, not a WowRobo-certified CAD revision.

The third source uses exactly the same `wrist`, `gripper` and moving-jaw body
transforms as our pinned RobotStudio follower. Its camera subtree can therefore
be registered without guessing a transform between wrist conventions.

## Comparison

| Detail | Previous implementation | Photo/reference correction |
|---|---|---|
| Robot attachment | Arbitrary gripper offset (−38, −22, −25 mm) | Registered reference transform (−15, +24, −31.5 mm) and its full quaternion |
| Adapter face | Incorrect diagonal face chosen for PCB | Reference camera module attached to the correct adapter face |
| Camera geometry | Separate generic box PCB and lens cylinder | Reference 32 mm PCB/camera mesh with projecting lens |
| Position | Low, on the wrong side of the gripper | Raised above the fixed gripper, matching the photos' arrangement |
| Optical frame | Guessed direction | Reference lens center and optical axis, approximately25° down toward the jaws |
| Appearance | Generic green board | Dark exposed board with cosmetic gold perimeter as seen in the close-ups |

Both helpers now use the corrected subtree. In gripper-local coordinates the
optical center is approximately `(2.50, 61.58, −18.77)` mm and optical direction
`(−0.000009, −0.42377, −0.90577)`. It lies outside the reference lens mesh and
points toward the grasp region (direction cosine≈0.970). Gripper opening leaves
the camera fixed; wrist roll moves it rigidly.

The board form factor and lens geometry are the reference model's dimensions;
the supplied photos do not establish exact real-camera dimensions, sensor type,
focal length or distortion. We did **not** adopt that repository's OV2710 sensor
claim or focal calibration for this kit. Vertical FOV remains an explicitly
provisional75°. Hand-eye/intrinsic calibration remains necessary for image-based
sim-to-real work. No motor gains, jaw collision modifications or task settings
from the reference repository were imported.

## Reproducibility

`cad/wowrobo/camera-source-lock.json` pins the repository revision and SHA-256 of
the camera XML, adapter STL, camera STL and license. The camera model's MIT notice
is retained in `cad/wowrobo/ICL-so101.LICENSE`; the underlying RobotStudio adapter
retains Apache-2.0 attribution. Source assets remain under ignored
`assets_local/wowrobo/camera-reference/`. Missing files are downloaded from the
pinned URLs and hash checked by the review builder.

Current corrected workcell: `assets_local/wowrobo/initial-layout-v2/`. Close-ups:
`camera-closeup.png` and `camera-side.png`. `scripts/render_wowrobo_camera.py`
reproduces the photo-comparable view. Focused tests cover exact registered
attachment, raised position, optical direction/lens clearance, scale conversion,
and rigid wrist attachment independent of moving-jaw articulation.
