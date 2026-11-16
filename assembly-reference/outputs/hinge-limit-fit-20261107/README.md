# Measured internal servo hinge stops

The user confirmed the targets are the assembled upper-arm/forearm servo 2 and servo 3 hinges. Each has a lower and upper stop. These replace the previous artificial symmetric ±1.7 rad travel caps.

| Joint | Lower (degrees) | Upper (degrees) |
|---|---:|---:|
| target_servo_2_shaft | -35.8179401383 | 215.8175186291 |
| target_servo_3_shaft | -215.8028853163 | 35.7951155975 |

Angles are relative joint coordinates. They are measured against the existing discretized native SDF plus exhaustive surface supplement, not analytic CAD or hardware calibration. `measure.py` scans a full revolution in each direction at 0.5° spacing, then bisects the first crossing 25 times using distance < -1e-9 m. Raw brackets/provenance are in `angles.json`; `angles-first-pass.json` is the superseded half-turn exploration.

## Implementation

- Canonical helpers: `assembly_policy/so101/wowrobo_hinge_limits.py` and `.cc`; fit: `configs/wowrobo_hinge_limits.json`; opt-in factory integration: `assembly_policy/so101/wowrobo_training_runtime.py`.
- Remove exactly three explicit passive motor stop pairs and exclude their two body combinations from implicit collision routing. Other CAD/contact paths remain.
- Independently vary each of four endpoints by up to ±0.05° at reset, conditioned on the reset remaining inside the resulting interval. Persist each draw for the episode and in replay state.
- Append four offsets to the existing 162-column reset payload (166 total). Userdata has three friction values followed by four limit offsets. Install the limit callback after the native friction callback.
- Use a thread-local copy of only joint ranges while materializing native constraint rows. Shared model arrays remain unchanged. Near-limit worlds rebuild constraints; distant worlds take the fast path.
- Factory verifies the fit SHA256 and binds helper sources, noise amplitude and userdata offset into physics identity. Hinge mode requires the versioned friction payload.

`factory.py` and `env-with-limits.py` are isolated integration reference snapshots, not replacements for a newer production environment. Port only the hinge delta and preserve current diagnostics/replay behavior. The canonical factory has subsequent validation and provenance additions; regenerate release hashes from final sources.

## Qualification

- All 1,143 original native reset rows fit nominal limits; `qualification.json` records angle extrema and zero out-of-range rows. The expanded 19,200-state bank still requires release-owner admission.
- Native hook regression: two pytest tests passed. Two simultaneous noisy worlds match separate fixed-range reference models exactly over 600 direct steps and two-thread 300-step rollouts. Shared joint ranges remain unchanged.
- Actual environment smoke: two worlds/two native threads, five controls, distinct non-unit friction and independent limit noise. PASS: no safety/unstable outcomes, saved noise retained, shared ranges unchanged; see `env-smoke.json`.
- Same 0.5 ms / 80-substep loose-state timing: baseline warmed samples 75.677 and 75.371 ms; limits 5.167 and 5.141 ms (about 14.6×). Both finish with 16 contacts and zero warnings. This is physics-only, one loose state, not full PPO/fleet performance. This geometric timing excludes the friction/limit variation hook.
- Invalid initial timing used default qpos and a 64 MiB arena, generated arena warnings and is discarded. Valid timing restores saved FULLPHYSICS/eq/ctrl/warmstart and uses production 256 MiB arena.

## Deployment ownership

Candidate uploaded and tested only under the approved isolated `native-contact-optimization-20261107/hinge-limits-full/` directory. Main-run chat received values, sources, payload schema and qualification; it is preparing native-v13 integration/full-bank validation. No live cutover by this chat. Preserve state-build CPU priority and migrate the current committed native checkpoint through the main owner.
