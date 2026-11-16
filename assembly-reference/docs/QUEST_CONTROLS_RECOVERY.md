# Quest controls and recovery, November 5, 2026

**Goal:** Load prepared tasks with A, reset the entire simulation with a two-second A hold, show one top-left status line and keep arms responsive under contact load.

The current sequence now includes a sixth arm-link task and upright screw presets. See [upright/link update](UPRIGHT_LINK_UPDATE.md). This document preserves the original five-stage recovery evidence.

## Behavior

- Short right-A press, on release: load the next prepared task, motor → horn → screw → retainer → yoke → motor.
- Hold right A continuously for two seconds: reset all parts/arms to the initial loose motor scene. It fires once; release does not also advance. Tracking loss, a sampling gap over 200 ms or B cancels the incomplete gesture.
- Task advance/reset starts Disabled, revokes old goals/grabs and requires released Y/grips plus a fresh Y press. B retains Stop and its old one-second reset gesture.
- The headset has one head-locked status/task line at the top left, such as `DISABLED · MOTOR`, `ENABLED · HORN` or `STOPPED · SCREW`. The former instructional banner, branding/help and footer are hidden. Desktop controls remain at the bottom.
- Scene resets snap the measured rendering to the new state; interpolation does not animate predecessor assembly.

## Failure evidence and correction

The user says the initial immersive session worked, then the arms stopped moving. Before changes, the native host remained alive, finite and fault-free. The browser retained an XR-hidden/fresh-Y hold message. That focus hold is deliberate; there is insufficient pre-failure history to attribute every wearer interruption to it.

The real-wall-time loaded-contact regression also fails its 150 ms cycle budget on this Mac, reaching 169.8 ms close to the unchanged200msinput lease. Separately, stereo controls depended on optional hidden wrist JPEGs being under 500 ms old. Live verification later measures a 658 ms JPEG age while owner state is 9.1 ms old. A hidden image stream could therefore hold otherwise live stereo arm control.

The final owner still advances ten 2 ms controller steps per cycle. It checks for genuinely new pending input and publishes measured state every two steps inside that cycle. New Stop/Disable and control samples are admitted before the remaining physics steps. Old receipts never renew their own lease. Stereo freshness uses the measured native owner state and renderer health, independently of optional wrist JPEG acquisition age. Input age, focus/session checks, finite/bounds/effort checks and manual Stop remain. The native timestep, contact solver, geometry, effort caps and layout are unchanged.

A first version used two physics steps per complete cycle. It passed feature checks but reduced physics progress per input; cooperative polling supersedes it. The existing whole-cycle timing test now bounds the full ten-step cycle by 200 ms and a new regression proves that a Stop arriving after step 1 is admitted before step 3. This tests the actual changed scheduling boundary instead of implying an uninterrupted ten-step cycle is still the control admission interval.

## Prepared starts and limits

`assembly_policy/so101/task_scenes.py` initializes nominal predecessor poses, settles the canonical native model for 0.5 s, rejects warnings/nonfinite state or contact/independent motor-base overlap above 0.4 mm, and returns measured state. These are explicit initialization assumptions, not executed or learned assembly. All bodies remain free; there are no welds or continuing pose corrections. Training's default all-loose bank and checkpoints are untouched.

| Selected task | Prepared predecessors | Largest predecessor position error after settling |
| --- | --- | --- |
| Motor | None | Not applicable |
| Horn | Motor | 0.057 mm |
| Screw | Motor, horn | 0.193 mm |
| Retainer | Motor, horn, screw | 0.377 mm |
| Yoke | Motor, horn, screw, retainer |**5.132 mm retainer offset; fit unqualified**|

Prepared native contact depth stays below 0.071 mm; independently sampled motor/base overlap is about 0.145–0.150 mm where assembled. This is bounded soft contact, not literal nonintersection or qualified fit. Motor insertion, screw mechanics and real calibration retain their existing limitations. Native model signature is `108baa5d495a94c358a5bf8636eded6f031f597fb1f5b71ab014bcc66a1de026`, physical profile shared-side-v5-centered; physical JSON SHA256 remains `06315153f47da3ecaa95d85f2a90eb27b9a960ae7ad045b7058a6e7121c8e8cd`.

## Verification and current session

The intermediate full feature run passes 24 native cases, including prepared states, protocol/expiry, native motion and both real HTTP launch paths. The shared-model parity test also passes. All 21 Node/browser cases pass, including production right-A mapping, long-hold exclusivity, interrupted gesture cancellation, owner/session behavior and explicit-reset rendering. An owned torque-disable reset check passes. The final cooperative version passes 11 selected regressions after the scheduling change. Local logs distinguish these runs instead of claiming the earlier full suite executed the final scheduling code.

Intermediate live HTTP verification moves both arms by 2.57 mmduring its short clutch test, loads all five prepared scenes and resets to motor, Disabled/no faults. It verifies command delivery, not real-time tracking accuracy or assembly. The final host is started from the same main Mac clone; inspect `artifacts_local/mac-quest-controls/host.pid` and health before stopping/restarting it. The headset page must be navigated to a fresh query URL if Quest keeps its old document after reload. A cache-disabled navigation loads the new status-only page with no JavaScript exceptions.

Local ignored evidence: `artifacts_local/mac-quest-controls/`: `before-state.json`, `before-health.json`, `before-browser.json`, `loaded-before.log`, `prepared-diagnostic.json`, `prepared-scenes.json`, `native-tests-final.log`, `cooperative-tests.log`, `shared-parity.log`, `browser-tests.log`, `live-protocol.json`, browser verification JSON and final host log/PID. Source changes remain local, uncommitted and unpushed.

Normal launch after stopping the existing host:

```bash
cd /Users/ale/Documents/ChatGPT/robotics-assembly
bash teleop/run.sh --quest --adb "/Applications/Unity/Hub/Editor/6000.3.25f1/PlaybackEngines/AndroidPlayer/SDK/platform-tools/adb"
```

Prepared scenes are generated before the host becomes ready, adding roughly 20 seconds on this Mac; real launch tests allow 60 seconds. The wearer must re-enter VR and press Y after a browser reload or scene reset. The final Mac host is PID 64379. Final device inspection confirms an active XR session, the new head-locked status-only HUD, no JavaScript exceptions and no native faults. The final host has processed 15 scene transitions from the headset and is currently Disabled on motor. A further wearer check of arm tracking and the two-second hold remains useful. Receipt: `artifacts_local/mac-quest-controls/final-host-browser.json`.

## Files changed and modifications per file

This lists this follow-up; the earlier Mac portability edits are preserved.

| File | Modification |
| --- | --- |
| `assembly_policy/so101/task_scenes.py` | Canonical settled prepared task initialization with contact checks and reported predecessor errors. |
| `teleop/backend.py` | Prepared scene selection/reset, Disabled controller recreation, two-step input polling and stereo owner freshness. |
| `teleop/runtime.py` | Validate/preserve authenticated scene actions; admit new packets within native cycles; keep Stop priority and original expiry. |
| `teleop/serve.py` | Prepare task starts before interactive startup. |
| `teleop/web/preview.mjs` | A tap/hold state machine, cancellation and concise status labels. |
| `teleop/web/app.js` | Map production A actions, clear stale grabs on reset and render a small camera-attached HUD. |
| `teleop/web/immersive.mjs` | Snap measured poses on an explicit scene generation change. |
| `teleop/web/index.html` | Remove the branded header and hide instructional content/footer. |
| `teleop/web/style.css` | Place desktop controls at the bottom and keep the headset view minimal. |
| `teleop/tests/task-controls.test.mjs` | Short/long A exclusivity, cancellation, queue Stop priority and status tests. |
| `teleop/tests/session.test.mjs` | Production right-A sampling and status fixture support. |
| `teleop/tests/immersive.test.mjs` | Explicit scene changes do not interpolate fictitious assembly. |
| `teleop/tests/test_task_scenes.py` | Prepared native states, advance/reset, rejection/expiry, hidden-JPEG independence and in-cycle Stop. |
| `teleop/tests/test_assembly.py` | Verify native contact responsiveness at the new admission boundary. |
| `teleop/tests/test_http_launch.py` | Allow the pre-start preparation budget; retain real HTTP/USB-absence checks. |
| `teleop/README.md` | Document A controls, status HUD, scheduling and preparation limits. |
| `User Intent.md` | Record the clarified prepared-scene behavior, minimal HUD and arms-response requirement. |
| `MEMORY.md` | Preserve decisions, rejected approach, evidence, limitations and next session. |
| `TASK.md` | Keep the current goal, implementation, verification and wearer follow-up brief. |
| `docs/MAC_QUEST_SESSION.md` | Mark the original startup/process as historical and point to the current session. |
| `docs/CURRENT_ENVIRONMENT.md` | Link current controls and preparation limits with unchanged physical identity. |
| `docs/QUEST_CONTROLS_RECOVERY.md` | This result, evidence and per-file inventory. |
