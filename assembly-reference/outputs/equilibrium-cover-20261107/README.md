# Side-cover pickup capture repair — 2026-11-07

These are bounded local grasp repairs for states01 and03 using the corrected full native model `outputs/equilibrium-model-20261107/profile-bearing-v4`. They are not whole-scene equilibrium/training admissions: the independent future loose forearm is still moving during these tests.

The original airborne cover starts had a gap between the real pads and the cover. Closing the jaw only after integration began let the cover fall before bilateral capture. Local helper translation and initial jaw closure acquire the same opposed exterior side walls before gravity integration; target geometry and all target/RIGHT initial coordinates are bitwise unchanged. Only the six LEFT coordinates change. Native actuator gains, force limits, masses, gravity and contact laws are unchanged. No external holding force, hand weld, bracket-angle target, or post-settle velocity clearing is used.

Use `capture-recipes.json`. It contains the six LEFT initial joint positions, absolute real position controls including one gravity preload, source identity, and evidence. If reconstructing controls, set them to qpos plus gravity compensation ONCE, then subtract the stated small jaw closing preload. Do not apply gravity compensation again to the supplied absolute control array.

| State | Local helper translation toward fixed pad | Initial jaw change | Additional jaw control preload | Minimum fixed / moving pad force in unchanged .2s hold | Cover motion / rotation in hold | Total settling motion |
|---|---:|---:|---:|---:|---:|---:|
|01 cover pickup|0.30mm|−0.0194rad|−0.00025rad|5.09535 /5.36127N|1.023e−10m /0deg|0.1157mm|
|03 cover aligned|0.29mm|−0.0188rad|−0.001rad|6.68186 /6.96240N|1.047µm /0.000556deg|0.1267mm|

All physical trajectories use .5s settling and .2s unchanged controls. Global velocity remains nonzero because the separate loose forearm is unsettled; do not claim a whole-scene pass.

Native source-mesh boundary audits, initial/final:

-01: PASS, maximum sampled overlap9.74/13.59µm.
-03: PASS, maximum sampled overlap6.87/7.41µm.

The stronger03 `aligned-capture-v3` also holds, but the adopted v2 uses lower force. The initial v1 aligned candidate with only .00025rad jaw preload slipped and is retained as a negative trial. States04/05 are untouched here; the parent full-state sweep determines whether they need further changes.

`scan.py` and `gaps.py` perform bounded initial capture diagnostics. The latter temporarily changes contact margins only in its own diagnostic model to query positive pad gaps; that modified model is NEVER used by `trial.py` or published. Actual trial models load the untouched v4 XML from disk.

Reproduce the adopted candidates from the canonical repo:

```sh
PYTHONPATH=. ../assembly-so101/.venv/bin/python outputs/equilibrium-cover-20261107/trial.py --stage 01_cover_pickup --shift .0003 --jaw -.0194 --name pickup-capture-v1
PYTHONPATH=. ../assembly-so101/.venv/bin/python outputs/equilibrium-cover-20261107/trial.py --stage 03_cover_aligned --shift .00029 --jaw -.0188 --preload .001 --name aligned-capture-v2
PYTHONPATH=. ../assembly-so101/.venv/bin/python outputs/equilibrium-cover-20261107/audit.py --name pickup-capture-v1
PYTHONPATH=. ../assembly-so101/.venv/bin/python outputs/equilibrium-cover-20261107/audit.py --name aligned-capture-v2
PYTHONPATH=. ../assembly-so101/.venv/bin/python outputs/equilibrium-cover-20261107/export.py
```

The recipes intentionally describe pre-settle capture initialization. Apply them before the parent's full-scene settling/unchanged-control hold and admission checks. Preserve the real saved velocities and controls in the eventual reset.
