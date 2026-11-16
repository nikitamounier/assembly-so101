# Passive bracket contact model — 2026-11-07

This is the full bimanual source-SDF model, with no virtual motor holding controller. Existing unpowered target shaft joints remain native hinges. Added explicit same-parent source contacts restore physical stops omitted by MuJoCo parent filtering:

- servo 2 case against upper-arm printed link;
- servo 3 case against each of the two forearm printed links.

The stop contact law is copied unchanged from the force-held passive-final diagnostic: condim4, friction .7/.005/.0001/.0001/.0001, solref .002/1, solimp .95/.99/.0002. This is still an uncalibrated contact candidate. It does not establish equilibrium, fit, loaded grasp retention or training admission.

Profile `profile/profile.json`; model SHA256 `7761dd32d77771ac5a49daffd7cb38394d517778aa25e551b8c2527b46a62139`. nq/nv/nu =51/46/12, three inactive installed-interface welds, seventeen explicit pairs (fourteen prior mating pairs plus three motor stops). All four target shafts have zero stiffness, damping .02, no angle actuators and no equality locking across articulation. Original helper controls/masses/state layout and target meshes unchanged.

API: `build_physical_profile(..., passive_motor_stops=True, helper_grid_profile=existing_profile_dir)`. Both kwargs also available on `WowRoboPhysicalValidator`. Default false preserves prior profile identities. Cached reads reject passive-option changes and modified stop laws. Helper-grid reuse checks actual raw mesh and field hashes, avoiding regeneration.

Validation:17 tests in `tests/test_wowrobo_physical_resets.py` pass; Ruff passes. Added mechanical parent-filter contact counterfactual and fail-closed actuator/spring/joint-equality/body-weld tests. Existing 12 tests preserved.

## Internal bearing defect and corrected v3 profile

Latest selected profile: `profile-bearing-v3/profile.json`, model SHA256 `dc9693f790195721fb75c969d88bf2e0bb5458d5cf99890fd0c3d76c2e80bc18`. This supersedes the initial contact model above. Stop friction is the latest corrected diagnostic's isotropic five-value tuple `.7 .7 .005 .0001 .0001`. Four explicit body exclusions remove manufactured motor/horn bearing collisions across real shaft joints; explicit printed-bracket/motor stops survive those exclusions. This is checked in native MuJoCo with a world-fixed base and two dynamic bracket stops.

The prior full model's world-fixed base exposed a MuJoCo parent-filter exception: servo1 case and its own rotating horns collided inside the manufactured bearing. At the natural-v1 loose snapshot t1.5s, shaft1 had qvel0.2070288058rad/s despite gravity bias2.4e-21Nm. Those near-degenerate contacts produced recomputed normal forces of order1e14–1e16N. New profile has zero such contacts; starting from that same integration snapshot without clearing velocity, shaft1 decays to−6.6e-20rad/s after0.1s. Original qpos, meshes, masses, helper controls, shaft damping and joint DOFs remain unchanged. New layout nq51/nv46/nu12/neq3/npair17/nexclude4.

API additionally requests `exclude_internal_servo_contacts=True`; cached mismatches fail closed. All18 physical regression tests and Ruff pass. Initial `profile` and `profile-bearing-v2` remain historical anisotropic prototypes and should not be used for qualification.

## LEFT cover contact diagnosis

`left-contact.json` and `left-mesh-gap.json` inspect actual old stage18 input/settled states. At input, fixed/moving pads carry4.151/7.618N with every classifier mask satisfied. After settling there is no fixed-pad contact at all, and its native source mesh gap is1.981mm. Only the moving pad remains (~40.37N); hand site drifts−.449/+2.305/−.005mm and shoulder pan changes−.00985rad. Thus this is actual loss of contact, not a classifier false zero.

Bounded old-profile trials moved the actual initial hand0.1/0.2mm inward using native five-joint IK; neither restored fixed-pad retention. Disabling adaptive jaw closure likewise failed. Evidence `left-preload.json`. A different acquisition/load-balancing strategy is needed; no incorrect grasp was promoted.

## Helper base bearings — final v4 profile

Final selected model is now `profile-bearing-v4/profile.json`, superseding v3. The same world-parent exception also created nine contacts per helper between fixed `sts3215_03a_v1` and rotating `rotation_pitch_so101_v1` at the shoulder-pan output axis. At the prior loose snapshot, 0–157 micrometer overlap generated recomputed17–123kN normal loads. This invalidated earlier conclusions that LEFT pan drift was caused solely by jaw reaction; it remained pinned even when the jaw was opened.

The same opt-in bearing exclusion now covers `left_base`↔`left_shoulder` and `right_base`↔`right_shoulder`, restoring normal adjacent-link filtering at those manufactured bearing interfaces. Target/helper, table, cross-arm and nonadjacent self contacts are unchanged; joint limits remain active. Six exclusions total, and seventeen explicit pairs still include all three bracket/motor stops. A native regression demonstrates unwanted target/helper base bearings disappear while both explicit bracket stops remain active. All18 physical tests and Ruff pass.

Evidence `helper-base-contacts.json`; v4 native contact smoke `v4-contact-smoke.json`. Earlier v1/v2/v3 equilibrium or grasp-search outcomes retain evidence but use known defective internal-bearing contact models and should not determine corrected-model qualification.

## Exact source-vertex floor manifolds — v5

New selected profile: `profile-bearing-v5/profile.json`; SHA256 `7558caa03889f6c395de39a9c2929b420ea5768da1063ce97e2fd06e69686674`. Additional explicit opt-in is `source_mesh_table_contacts=True` on builder/validator. Cache identity rejects changed selections.

The prior SDF/plane route emits one support per flat source mesh. A loose final module could balance on only two points with its center of mass4.4mm outside their line, causing indefinite rocking. The native mesh/plane manifold uses the existing source mesh's extremal vertices. A source mesh and its convex hull have the same support function against a plane, so this changes manifold coverage without adding collision volume against the flat table. All54 existing source mesh geoms receive floor-only bit16/affinity4. SDF geoms lose only table bit4 (affinity11); the table retains contype4 and now accepts only bit16. All source-SDF target/helper/mating interactions, friction, solver laws and internal-bearing exclusions are preserved. No new geom, box, decomposition, artificial damping or rolling friction is introduced.

The builder compiles before/after models and checks exact native arrays for body mass/inertia/frame, joint layout/limits/damping, actuator gains/force limits, equality frames, geometry positions/rotations/vertices, and friction/solver settings; all unchanged. Tests demonstrate a flat native source-mesh floor manifold has at least three contacts, all raw source vertices, and arbitrarily tilted support matches the exact raw-source lowest point. Masks permit no source-mesh↔SDF or source-mesh↔source-mesh route and disable SDF/table duplicates. All19 physical tests and Ruff pass.
