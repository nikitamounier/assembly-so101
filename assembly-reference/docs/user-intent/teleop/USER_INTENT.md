# Meta Quest teleoperation and views

**Goal:** Make this section’s user intent, implementation and unfinished work easy to review.

Keep the Quest simulation app inside this repository. It consumes the same physical scene as training; the adapter owns controller mapping and rendering, without separate physics/layout settings. A separate Feather checkout is unnecessary.

Desired experience: immersive, close stereo views of both arms; responsive motion; controller orientation tracking within the five-joint arms’ kinematic limits; and the wrist marker hidden while actively tracking. Preserve the accepted mounts/camera layout and actual finger closure/contact.

Controls remain Y to enable, grip to clutch/move an arm, trigger for its jaw, X to recenter and B to stop. Start disabled and require fresh input/session ownership. Current software collision-Hold settings follow the earlier simulation-only user choice; native part/table/gripper physics remains active.

**Current state:** The v14 teleop suite passed 84 tests before the numerical-limit update. No physical model changed in that update. The local Quest service is currently inactive/not loaded after validation maintenance. A new live wearer/controller session has not been verified by this audit.

After qualification, start a fresh loose, disabled session; do not restore incompatible old physical states. Setup/USB access depends on the computer actually connected to the Quest.

**For your review:** Views and control behavior. See [teleop setup](../../../teleop/README.md) and the [latest v14 native videos](../../../artifacts_local/mating-review-v14/media-review.json).
