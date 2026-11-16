# Fit-selected assembly pin

The user authorized choosing dimensions for the unprinted pin to fit the existing CAD. The selected profile is `wowrobo-fit-pin-20261107-v2`: Ø2.3mm shaft ×14.5mm, Ø12mm grip ×28mm, total42.5mm,128facets. Current canonical config and SCAD carry these dimensions. Frozen components-v2 and photo-resets-v3 are unchanged.

## Why both diameter and length changed

The source servo bore has a nominal2.5mm minor opening. Radial source-ray samples through the thread show minimum radius about1.2495mm; Ø2.3 leaves about0.0995mm radial clearance.

The old fully seated pin anchor was inferred from the servo pivot rather than the actual printed stop.1536 rays through the head's annular footprint identify the printed support surface11.602510mm above the servo-axis reference. A ray down the bore centerline finds the first servo-case cap3.197494mm below that reference. The actual stop-to-cap span is14.800004mm. The earlier22mm shaft therefore cannot seat at the printed head stop.14.5mm leaves0.300004mm axial clearance after the head settles; a0.1mm initial head gap gives0.400004mm initial tip clearance.

This also explains the previous native stalled depth: its tip began14mm above the servo reference, and stalled after17.147645mm, leaving the tip3.147645mm below the reference—about50µm above the source cap. The diameter-only explanation was incomplete. Smaller diameter alone does not avoid a blind-bore end.

The native servo mesh is not watertight; winding-based containment is not reliable enough to infer bore depth. The registration utility uses direct exact-triangle ray intersections. No receiving hole is enlarged, removed, or masked.

## Build and application

`scripts/build_wowrobo_fit_pin.py` builds a versioned native OBJ in metres, printable STL/SCAD and analytic STEP in millimetres. It records file hashes, volume, centre of mass and inertia per unit density. The main components builder previously hardcoded a1.35mm shaft radius in its triangulation while STEP used the config. That mismatch is fixed: both meshes now derive every pin dimension from config.

`assembly_policy/so101/wowrobo_pin_registration.py` measures the true head stop and bore axis, including its0.5° tilt. Its receiver-relative anchor follows connector rotation. Apply it to13/14/15 with32/7.25/2mm stand-offs (half-depth derives from the new14.5mm shaft), and installed pin states17onward. Recompute the active pin-holding arm's IK after changing that target pose. Pickup12 keeps its placement but uses the new geometry.

Artifacts: `assets_local/wowrobo/fit-pin-v2/` includes `pin.obj`, `assembly_pin_print_mm.stl`, `assembly_pin.step`, `assembly_pin.scad`, manifest, measured registration and a native review audit. All pin-target sampled triangle penetrations are zero for the registered installed state. This is source-geometry validation, not a claim of calibrated printed tolerance, dynamic insertion, loaded retention or training admission. The connector/upper/final mating barriers remain separate.
