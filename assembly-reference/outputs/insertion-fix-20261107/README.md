# Consolidated insertion results — 2026-11-07

All five guided, force-driven insertions reach their reference depth using the updated MuJoCo 4.0.0 runtime and one frozen candidate. The reel replays saved native qpos; the renderer performs no integration. Orange arrows indicate the actual exterior source-vertex push locations and intended force direction. Each step is an isolated fixture test; receivers are held fixed and a virtual gripper constrains lateral pose, orientation and internal shape.

| Step | Travel / reference (mm) | Applied force | Drive duration (s) |
|---|---:|---:|---:|
| cover | 55.002686 / 55.000 | 20.0 N | 4.8775 |
| connector | 65.008000 / 65.000 | 20.0 N | 3.2505 |
| pin | 32.100349 / 32.100 | 1 N | 8.0000 |
| upper | 25.000999 / 25.000 | 20.0 N | 3.9905 |
| final | 25.002862 / 25.000 | 20 N | 2.6015 |

## Diagnosis and changes

The old repository had interface-specific CAD and contact repairs; its Hiwonder geometry omitted a lower horn present in this WowRobo kit. Its offsets were not interchangeable. Friction-off ablations left connector, pin, upper and final at essentially the same original stalls. Real material interference and inconsistent predecessor registration were the primary barriers; coarse SDFs added numerical interference at small clearances.

- Cover: small rigid pose correction; source CAD and legacy contact law unchanged.
- Connector: derived horn-throat and case-bottom-post clearance. Both original horns remain. The receiver is centered on the actual shaft.
- Pin: isolated Ø2.3 × 14.0 mm shaft candidate. Centering the connector moves the head stop by 0.425 mm; the earlier 14.5 mm shaft bottoms 0.125 mm early in this coherent assembly. The 14.0 mm candidate leaves about 0.375 mm axial clearance. Existing original and v2 assets remain distinct.
- Upper: measured source-case swept clearance in the derived connector pocket; original case geometry retained.
- Final: local derived inner-wall/chamfer relief (about 0.131–0.202 mm), fine source-case fields, and the actual preceding upper endpoint and physical hinge. No wider final contact-law fallback was used.

Derived printed meshes are conforming, watertight and oriented; checked tessellation errors are below 21 µm. The vendor files are preserved. All four unchanged case meshes use verified 0.1 mm fields; pin uses 0.1 mm, connector 0.2 mm and upper printed part its verified 0.4 mm field. The native field-to-source frame audit is below 2.1e-16 m; fields were reframed once. The frozen identity, asset hashes, exact proof paths and metrics are in `summary.json` and `final-common-v1/FINAL_READY.json`.

## Controls and release limits

All five obstruction controls remain blocked: cover 21.680/55 mm; connector 65.057/70 mm at the closed back wall; longer pin 31.974/32.100 mm; upper 6.624/25 mm; final 7.753/25 mm. These exercise retained material and blind stops rather than disabling contacts to show completion.

Free incoming-root gravity release tests remove root actuation and damping while retaining fixed receivers. Cover moves at most 0.00738 mm over 1 s. Connector root drift is about 0.209 mm over 1 s. Pin drift is about 0.00165 mm over 2 s. Upper case drift is at most 0.319 mm and tilt 1.449° over 2 s: it fails the strict 1° retention benchmark. Final case drift is 0.181 mm / 0.749° with internal shape locked, and 0.091 mm / 0.234° with internal joints free over 0.5 s; the fixtures remain held. These are bounded release observations, not indefinite settling or complete dynamic assembly qualification.

Insertion uses zero gravity and implicit guide damping of 1000 Ns/m (pin 200 Ns/m). Pin force ramps over 0.5 s; other steps use constant 20 N. Runtime collision masks and guide damping are recorded in scripts/receipts; profile XML alone is insufficient to reproduce every runtime override. The pin's provisional 0.1 kg mass and all contact/material responses require hardware calibration. No training default, production pose bank, or running experiment was replaced.

## Files and reproduction

The five canonical directories are listed in `summary.json`; they contain drive states, native trajectories, controls, release receipts and render specifications. The final fixture inherits `upper/canonical-common-v1/actual-endpoint-full51.npz`. Historical failed variants remain preserved and are not included in this reel.

```sh
PYTHONPATH=. ../assembly-so101/.venv/bin/python scripts/render_wowrobo_fit_trace.py outputs/insertion-fix-20261107/final/canonical/render-spec.json
ffmpeg -y -f concat -safe 0 -i outputs/insertion-fix-20261107/reel-inputs.txt -c copy outputs/insertion-fix-20261107/insertion-fixes.mp4
```

Native macOS rendering requires a graphics context. Video captions disclose fixed fixtures. Replay does not qualify learned gripper behavior or hardware assembly.
