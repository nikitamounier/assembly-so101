# Mac Quest session, November 5, 2026

**Goal:** Run the canonical v5 workcell from the Mac clone and open it on the connected Quest for wearer testing.

This records the original Mac startup. The latest live controls, recovery changes and process are documented in [Quest controls and recovery](QUEST_CONTROLS_RECOVERY.md). The simulator was started detached at http://localhost:18883/ from `/Users/ale/Documents/ChatGPT/robotics-assembly`, main, parent e063403. The connected device identifies as Quest 3S and is USB-authorized. The USB reverse route maps tcp18883 to the same host port. The helper successfully opens Quest Browser, and Android reports its activity visible. A temporary read-only DevTools connection verifies the exact session tab is fully loaded, WebXR is available, the client reports live stereo geometry and the mode is Disabled. Its temporary forwarding route was removed afterward. Startup remains Disabled. A black headset screencap does not establish XR entry; wearer/controller operation is pending.

Put on the headset, select **Enter Quest VR**, release grips/Y, then press **Y**. Hold either grip to move its arm, use its trigger for the jaw, X to recenter and B to stop. Release B/grips, then hold B for one second to reset before enabling again. Keep USB attached.

## Verification

- The full `bash teleop/setup.sh` passes on Apple Silicon using the pinned 13-package CPU runtime and verified private archive. All three plugins are Mach-O ARM64 and retain their established `.so` filenames.
- `teleop.doctor` reports Apple M5 CGL rendering, native physics and three JPEG views, without torch.
- 14 focused Python cases pass: USB routing/discovery, training/Quest byte-identical models and initialization, physical finger closure/reopening and real HTTP launch, including unavailable USB. Log: `artifacts_local/mac-quest/focused-tests.log`.
- 16 Node browser/stereo/session cases pass.
- All 65,536 binary16 patterns checked: 63,488 finite values match exact float32 bits; NaN/Inf/sign classifications match. Local harness/result: `artifacts_local/mac-quest/half_check.cc` and `half-check.json`.
- The headset browser tab is verified at the exact localhost18883 URL with readyState complete, live stereo geometry and WebXR available. Receipt: `artifacts_local/mac-quest/quest-browser.json`.
- Live health reports ready owner and renderer with no errors; state is Disabled, not stopped, no faults; main/left/right JPEGs are present. Receipts and images: `artifacts_local/mac-quest/live-verification.json`, `main.jpg`, `left.jpg`, `right.jpg`; host/setup logs are beside them.

The v5 physical JSON is unchanged (SHA256 `06315153f47da3ecaa95d85f2a90eb27b9a960ae7ad045b7058a6e7121c8e8cd`). Adding ARM64 half conversion changes the plugin source hash, producing model signature `108baa5d495a94c358a5bf8636eded6f031f597fb1f5b71ab014bcc66a1de026` for both Mac consumers. Previous reset banks/checkpoints remain incompatible. No bank generation, training, hardware calibration or physical assembly qualification was performed. All source edits are local, uncommitted and unpushed.

## Restart and stop

The current PID is saved in `artifacts_local/mac-quest/host.pid`; inspect the process before stopping it. The original process has been replaced. Read the current PID file and verify that the process still belongs to this teleop host before sending SIGTERM. A normal foreground restart is:

```bash
cd /Users/ale/Documents/ChatGPT/robotics-assembly
bash teleop/run.sh --quest --adb "/Applications/Unity/Hub/Editor/6000.3.25f1/PlaybackEngines/AndroidPlayer/SDK/platform-tools/adb"
```

Use Ctrl+C for a foreground host. To reopen the browser while this host runs:

```bash
.venv-teleop/bin/python -m teleop.connect_quest --adb "/Applications/Unity/Hub/Editor/6000.3.25f1/PlaybackEngines/AndroidPlayer/SDK/platform-tools/adb"
```

The Homebrew platform-tools download failed DNS; an already-installed Unity adb works. No global shell settings were changed. First imports incurred macOS dynamic-library validation delays; subsequent native startup and tests pass.

## Files changed and modifications per file

| File | Modification |
| --- | --- |
| `plugin/assemblenist_sdf.cc` | Keep x86 F16C conversion; use native ARM64 half conversion. |
| `plugin/build.sh` | Detect host library/linker and architecture flags; build atomically on Mac/Linux. |
| `plugin/build_contact_search.sh` | Link the pinned MuJoCo library on Mac/Linux with the host linker. |
| `scripts/so101_build_plugins.sh` | Build the driver against the native host library. |
| `teleop/setup.sh` | Accept Mac ARM64, support the shipped Bash 3.2 and select CGL for doctor. |
| `teleop/run.sh` | Select CGL on Mac, EGL on Linux; retain explicit overrides. |
| `teleop/tests/test_http_launch.py` | Exercise the real host using its platform's renderer. |
| `teleop/requirements.txt` | Correct the supported-platform comment. |
| `teleop/README.md` | Document Mac prerequisites, rendering, explicit Unity adb and signature consequences. |
| `README.md` | Include Apple Silicon in teleop host instructions. |
| `AGENTS.md` | Distinguish canonical Linux and Mac checkout paths. |
| `User Intent.md` | Record Mac launch intent, host path and remaining wearer check. |
| `MEMORY.md` | Preserve decisions, verification, source identity and next-session state. |
| `TASK.md` | Keep current goal, approach, completion and wearer follow-up brief. |
| `docs/CURRENT_ENVIRONMENT.md` | Distinguish local Mac signature from published Linux evidence. |
| `docs/TELEOP_HANDOFF.md` | Link local Mac evidence without rewriting historical Linux results. |
| `docs/MAC_QUEST_SESSION.md` | Record this session, evidence, commands, changes and limits. |

## References

[MuJoCo native extension documentation](https://mujoco.readthedocs.io/en/latest/programming/) and [official GL backend selection](https://github.com/google-deepmind/mujoco/blob/main/python/mujoco/rendering/classic/gl_context.py) informed the port; the installed 3.17.0 source and real runtime checks establish the selected CGL behavior. [Android ADB documentation](https://developer.android.com/tools/adb) describes USB debugging and device authorization. Existing physical-mechanics limitations remain in [current environment](CURRENT_ENVIRONMENT.md).
