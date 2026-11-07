# Aligned exterior-force insertion review

Five step agents registered actual mating features and reviewed the corrected clips. Source CAD, the canonical pose bank and running experiments remain unchanged.

Parallel pushes act on real upstream source-mesh vertices. Nonnegative loads distribute in proportion to body mass, with each rigid-body resultant through its actual center of mass. Their complete-component resultant is0.5N along the intended insertion axis with essentially zero applied turning moment. Orange arrows show actual exterior point loads. The convex envelope selects load vertices only; it does not replace collision geometry.

Cartesian viscous drag acts at actual body centers of mass. Earlier probes incorrectly used free-joint translation damping at an offset model origin, introducing unwanted rotation. The free-root/contact-off mechanics test verifies the corrected load model independently: all original incoming hinges remain free, there is no alignment guide, and the loads produce no artificial rotation or lateral velocity.

An explicit one-DOF virtual-gripper slide holds the corrected lateral position, rotation and incoming shape. Helpers and receiver are fixed; gravity is off. Axial displacement integrates force and accelerated SDF contacts. There is no prescribed axial trajectory or artificial seated joint stop. Incoming internal pivots are temporarily locked solely for this whole-shape straight test; production geometry and joints are unchanged.

Actual source slot/bore directions define alignment: cover centered on printed rails; connector follows the tilted horn/slot tangent; pin follows the existing tilted bore; upper/final use fitted actual socket walls and entrance planes. Historical body-frame-only passes are preserved in each applicable `bodyframe-prior/` folder. Alignment is to modeled features, not a hardware calibration or proof of perfect manufactured fit.

All five remain contact-limited under0.5N. The custom2.7mm pin meets the unchanged modeled2.5mm threaded opening; actual stopping travel agrees with independent source triangles within0.002mm. Finite source/sweep audits do not establish continuous clearance. Cover side clearance(~0.099mm) is smaller than the0.4mm SDF pitch, so perfect analytic plane alignment does not establish resolved contact clearance.

| Step | Travel / reference (mm) | Maximum lateral drift (m) | Rotation-matrix drift |
|---|---:|---:|---:|
| cover | 25.16 / 55.00 | 0 | 0 |
| connector | 44.57 / 65.00 | 8.75e-18 | 0 |
| pin | 17.15 / 27.00 | 1.76e-18 | 0 |
| upper | 6.91 / 25.00 | 6.66e-18 | 0 |
| final | 7.27 / 25.00 | 1.24e-17 | 0 |

Each step contains `alignment.json`, `guide-scene.xml`, `insertion.mp4`, sampled frames, `receipt.json`, final native state and independent checks. Receipts record force application points, resultants, moments, solver/grid identities and guide limits. `summary.json` aggregates final clips; `all-insertions.mp4` is the20second reel.

```sh
PYTHONPATH=. ../assembly-so101/.venv/bin/python scripts/check_wowrobo_push_mechanics.py
PYTHONPATH=. ../assembly-so101/.venv/bin/python scripts/probe_wowrobo_aligned_force.py --step cover
```

Replace `cover` with `connector`, `pin`, `upper` or `final`. Native rendering needs graphics access. The16component SDF/native tree assets and sparse solver are reused from `../linear-insertion-updated-20261107/`.
