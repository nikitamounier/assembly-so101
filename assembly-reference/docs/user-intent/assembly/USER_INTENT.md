# Assembly scope and hardware

**Goal:** Make this section’s user intent, implementation and unfinished work easy to review.

Use two clamped **WowRobo SO101 follower arms with wrist cameras** to assemble **WowRobo SO101 leader components**, matching the November 7 hardware photos. Keep parts near the middle of the two helpers.

The corrected target contains **five structural pieces plus the pin**: (1) base already combined with servo 1, (2) the separate U-shaped base side cover held in IMG_1622, (3) the empty two-part shoulder connector shown in IMG_1617, (4) upper-arm module with servo 2 only, and (5) final arm piece with servos 3 and 4. The sixth object is the user's pin. These explicit photo corrections supersede the earlier interpretation of “five pieces total.” Assembly stops after the final arm piece; do not create a separate terminal wrist bracket, leader handle/trigger or target gripper. Both helper followers retain their functioning grippers. Internal servo pivots in preassembled modules remain articulated.

Use the official RobotStudio SO101 base geometry for the [WowRobo kit](https://shop.wowrobo.com/products/so-arm101-diy-kit-assembled-version-1), pinned by revision and source hashes. Pin dimensions in mm: total 50; shaft length 22, diameter 2.7; grip length 28, diameter 12; 128 facets. Preserve the supplied 0.01 mm cylinder overlap in the OpenSCAD source. See [component specification](../../WOWROBO_COMPONENTS.md).

**Current state:** The prior Hiwonder five-task implementation and its results describe a superseded hardware target. New component assets and a separate review scene are being built; this does not qualify new rewards, reset banks, insertion mechanics or learning. Exact as-built tolerances, camera calibration, forces, friction and compliance remain unmeasured.

The main-plan meeting notes raise helper-arm torque/accuracy concerns. Treat those as a feasibility warning to verify against the actual kit, not as measured motor specifications or permission to replace SO101 with a different robot.

The original four-structural-modules-plus-pin interpretation was incorrect and is superseded by the IMG_1622/IMG_1610 corrections.


The initial-scene step now fixes 720 mm between printed helper rear edges, target rear 300 mm from left helper rear, target facing right, and the confirmed LEFT/RIGHT/LEFT/RIGHT/LEFT pickup sequence. Position randomization preserves tangent grip spans at explicit grasp centers. See [layout contract](../../WOWROBO_INITIAL_LAYOUT.md). This step validates geometry/reset layout; contact dynamics and grasp IK remain unqualified.
