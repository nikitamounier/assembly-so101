# Portable viewer verification — November 7, 2026

Verified using a fresh export of the staged Git tree at a different directory,
Python 3.12.12 and a newly installed `.venv-viewer` environment. The original
viewer on port 8876 was left unchanged; the test copy ran on port 8878.

- The live `/api/scene` responses are identical: 66 visible geoms, 46 compiled meshes.
- All 31 referenced mesh files and all 50 state archives are byte-identical to
  the original repaired-cover-support-v2 collection.
- Native compilation matches on 24 model fields covering geometry, body/joint
  frames, topology, masses, inertias, actuator parameters and reference qpos.
  Both models have nq=51, nv=46, 29 bodies, 76 geoms and 12 actuators.
- Selected all 100 saved states through the test viewer API. Every qpos/qvel
  matches its saved archive. All 96 task states return 448-point observations
  with matching tensor hashes and an available reward overlay; the four loose
  setup states correctly have no active policy task.
- Browser verification passed for Three.js, MuJoCo WebViewer, reference/variation
  navigation, recorded rejection status, point clouds and seating rewards.
- Both included comparison videos load in the browser without errors: real
  assembly 14.833 seconds, physics simulation 16.033 seconds.
- 15 targeted Python tests, 14 frontend tests, Ruff, shell syntax checks and the
  production frontend build passed.

Runtime file checksums and the original model identity are recorded in
`runtime/wowrobo-viewer/snapshot.json`. Replacing absolute mesh paths changes the
XML file checksum while preserving native geometry and the saved-state contract.
This verifies saved-state viewing; it does not qualify physical training resets
or real hardware control.
