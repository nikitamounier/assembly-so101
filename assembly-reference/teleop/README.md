# Meta Quest simulation teleop

**Goal:** Run the shared training scene on your computer and control both SO101 arms from Quest controllers.

## Setup and start

Supported hosts: **Linux x86-64** with OpenGL/EGL drivers, or **Apple Silicon macOS** with Apple CGL rendering. Both use Python 3.12 and `cc`/`g++` (Xcode Command Line Tools on macOS). Install `uv` (or Python 3.12 venv/pip) and authenticate `gh` to this private repository. For Quest, install Android platform-tools (`adb`), enable headset Developer Mode/USB debugging, connect a USB data cable and accept its authorization prompt.

From the repository root:

```bash
bash teleop/setup.sh
bash teleop/run.sh --quest
```

Setup creates `.venv-teleop`, installs 15 pinned runtime packages and verifies the private geometry release before extraction. It compiles the native plugins and tests physics plus all three camera views. **PyTorch/CUDA and a training reset bank are not required.** The training environment remains separate in `.venv`. Native plugin builds replace files atomically, preserving already-running processes.

Inside Quest Browser, select **Enter Quest VR**, release Y/grips, then **press Y** to enable. Keep USB connected. Your computer also exposes a preview at **http://localhost:18883/**. Stop the host with Ctrl+C. If port18883is already in use, stop the previous simulator or use `--port 18884`. A locally managed instance can be stopped with `systemctl --user stop robotics-assembly-quest`.

The connector finds adb on PATH, in `ANDROID_HOME`/`ANDROID_SDK_ROOT`, common Android SDK folders and the existing local teleop platform-tools install. If you need a different executable or multiple Quests are attached:

```bash
bash teleop/run.sh --quest --adb /path/to/adb --serial YOUR_QUEST_SERIAL
```

For preview first, omit `--quest`. Connect later from another terminal:

```bash
.venv-teleop/bin/python -m teleop.connect_quest
```

If startup cannot find an authorized headset, `--quest` leaves the host running and prints a connect-later command.

The connection helper checks this app's health, selects an authorized Quest and preserves existing USB routes. It opens the browser; XR entry and Enable stay deliberate. A custom `--port 18884` works in both launcher and connection helper. Bindings remain localhost-only; the USB route provides Quest's secure localhost context.

## Controls

| Input | Action |
| --- | --- |
| Left Y press | Enable/disable |
| Either grip held | Move that arm relative to the controller's acquisition pose; grip can start anywhere |
| Trigger | Operate that gripper |
| Left X | Recenter the workspace |
| Right A, short press | Show the completed task's native-validated reward state; starts Disabled |
| Right A, hold 2 seconds | Reset all arms/parts to the initial loose motor scene; starts Disabled |
| Right B | Stop; release B/grips, then hold B for 1 s to reset before Y |

The main view is head-tracked stereo geometry at the accepted close side-on anchor. Tool/goal dots hide while gripping. Optional wrist close-ups start hidden. Each arm has five positioning joints, so full arbitrary controller orientation is not always reachable. Startup, scene advance and reset are Disabled and never restore old grabs. After a scene change, release A/Y/grips before enabling. A cycles loose initial scene → completed motor → completed horn → completed screw → completed retainer → completed yoke → loose initial scene. The motor completion keeps the base sideways; horn completion turns it upright with the round horn flat. Later completions retain that orientation. A long hold fires once and does not advance on release. Tracking loss cancels incomplete A gestures. The headset shows only status and selected task at the top left.

## Shared environment and code

| Location | Responsibility |
| --- | --- |
| `assembly_policy/so101/scene.py`, `workcell.py` | The single native physical model used by training and teleop |
| `configs/so101_scene.json` | Poses, cameras, mounts, contact settings, damping and effort caps |
| `teleop/backend.py` | Canonical model/home adapter and teleoperation descriptors |
| `teleop/control.py` | Relative clutch, IK, bounded references and simulation controller |
| `teleop/runtime.py`, `serve.py` | Single-owner simulation, input/session lifecycle, cameras and HTTP |
| `teleop/web/` | Quest client, stereo geometry rendering and pinned Three.js |
| `teleop/connect_quest.py` | Authorized USB route and browser launch |
| `teleop/tests/` | Native motion/physics, training parity, HTTP/ADB and browser regressions |

Change physical settings in the shared builder/configuration and restart the consumer. The `environment_identity.model_signature` in `/api/config` matches training's `model_signature`. This package has no dependency on an external Feather checkout and no real-robot SDK.

Native part/table/gripper contacts remain enabled. The temporary user-selected simulation profile disables software collision Holds and measured-speed Stops; bounds, finite-state checks, effort caps, input/session expiry and manual Stop remain. **V13 passes 83 teleop software tests but its horn free-drop settling check remains unresolved; hardware calibration and real-time contact performance remain unqualified**; see [current environment](../docs/CURRENT_ENVIRONMENT.md). A responsive teleop session is not proof of assembly success.

## Diagnostics and development

```bash
.venv-teleop/bin/python -m teleop.doctor
# Optional development dependency, with uv:
uv pip install --python .venv-teleop/bin/python pytest
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv-teleop/bin/python -m pytest -q teleop/tests
node --test teleop/tests/*.test.mjs
```

Node is only needed for browser tests. No full-rate recording occurs unless `--record` is supplied; recordings go to ignored `runs/teleop/`. `--seconds 10` gives a bounded host check. `--resume-state FILE` initializes measured poses/velocities only, rejects large motor/base overlap and never restores authority.

For an EGL error, check host graphics/Mesa drivers. Mesa software EGL can be selected on Linux installations with GLVND's Mesa vendor file:

```bash
__EGL_VENDOR_LIBRARY_FILENAMES=/usr/share/glvnd/egl_vendor.d/50_mesa.json \
LIBGL_ALWAYS_SOFTWARE=1 bash teleop/run.sh
```

Software rendering performance depends on the computer; startup checks do not qualify real-time teleop. `MUJOCO_GL` selects the supported MuJoCo renderer. `TELEOP_PYTHON` can select another prepared interpreter; `RUNTIME_ARCHIVE` supplies an already-downloaded pinned archive to setup. Windows native builds and wireless Quest routing are unqualified. Apple Silicon setup, native physics, all three CGL views, training/Quest parity and USB browser launch were checked on the Mac; wearer XR/controller operation remains a separate check. See [Mac session](../docs/MAC_QUEST_SESSION.md) for the local launch and evidence.

On macOS, the launcher selects `MUJOCO_GL=cgl`; Linux retains `egl`. ADB bundled with Unity can be selected explicitly with `--adb "/Applications/Unity/Hub/Editor/VERSION/PlaybackEngines/AndroidPlayer/SDK/platform-tools/adb"`. Native libraries retain the established `.so` filenames and contain host-native Mach-O on macOS or ELF on Linux. The ARM64 half conversion changes the source-bound model signature, so previous reset banks/checkpoints are incompatible until regenerated for this source.

Task completion states are generated, settled and qualified by `assembly_policy/so101/task_scenes.py` using the unchanged policy reward implementation. They must earn reward with release, bearing/thread contact where applicable, safety and 0.4 s stability gates. Native contacts/free joints remain active, with no pause, weld or ongoing pose correction. They are explicit initialization, not assembly demonstrations. Local ignored caches bind physical identity and qualification source; a fresh machine qualifies independently. First startup can take several minutes. See [reward-state evidence](../docs/REWARD_STATE_UPDATE.md).

Stereo control uses fresh measured owner state independently of optional wrist JPEG age. Its native owner cycle retains ten 2 ms controller steps, admitting genuinely new input and publishing measured state between every two steps. The original 200 ms input/owner lease, focus checks and manual Stop remain. Training physics, layout and native solver settings are unchanged by these teleop changes.

The unused arm link is removed from the shared physical model, observations and task list. Setup no longer downloads or extracts that part, and future runtime bundles exclude its cached assets. Historical source CAD and extraction tooling are retained locally for future work. The shared scene has five tasks ending at yoke. Its current physical identity is recorded in [current environment](../docs/CURRENT_ENVIRONMENT.md); incompatible old data cannot be resumed. See [removal evidence](../docs/UNUSED_LINK_REMOVAL.md).
