# Initial reset minimum checks — 2026-11-07

Latest user requirement permits motion. Admission requires material boundaries without unintended overlaps and retention of all held parts in actual grippers. Intended resting, mating and partly inserted contacts remain allowed.

Physical model: `../equilibrium-model-20261107/profile-centered-v6/profile.json`, SHA256 `909e034485939b77037f9568e59d7d027e37238372e95995c1a8222803ee96ef`.

Repairs retain all four unpowered target shafts, zero stiffness and original gains/force limits. Internal case/horn bearing contacts are excluded while real bracket stops remain; exact original source meshes support parts on the table. Motor housing registration is centered axially by 0.4 mm, preserving source meshes, mass, inertia and world shaft axes through state migration. No bracket-angle locks, helper welds or external holding forces.

Native hold witnesses retain measured velocities, controls and integration warmstarts. Minimum checks use a 0.2 s unchanged-control replay, material checks at start/end and every 0.05 s, relative held-part drift limits 5 mm / 10 degrees, and bilateral finger loading over at least 80% of the replay. Complete curved inner fingertip extent comes from the original finger source mesh; the previous +4 mm measurement window omitted real contacts near +5 mm. This measurement correction changes no geometry or forces. Regression tests verify real fingertip loads and unchanged contact forces.

The display collection is for rendering. Its physical sibling contains the actual native INTEGRATION states and physical scene for reset consumption. Earlier force-held diagnostic video predates housing centering; its fold trajectory is historical.

Final result: **25/25 valid prototypes** published to `assets_local/wowrobo/initial-valid-resets-v1` and its `-physical` sibling. Display/native body-position difference is exactly zero. The bank-local point contract was regenerated for the corrected frames and builds all448 points; calibrated world mating targets were preserved. Full variation quotas are separate and were not generated here. See published-verification.json and publish-final.log.

Viewer verified in both Three.js and the official MuJoCo WebViewer on localhost8876. Screenshots: viewer-valid-bank.png and viewer-mujoco.png. Bank-local feature/reward contracts are loaded automatically by the viewer CLI; source global training contracts remain unchanged. Reproduce the viewer contract regeneration from the workspace parent with `PYTHONPATH=assembly-so101 assembly-so101/.venv/bin/python robotics-assembly-3.0/outputs/equilibrium-reset-20261107/regenerate-viewer-contracts.py` after publication.
