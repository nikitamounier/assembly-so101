# Historical SO101 → WowRobo slowdown investigation

## 2026-11-07 [codex] Scope and conclusion

Investigated all eight user-referenced chats and the saved profiles. Primary later evidence is the **run-v5** capture in `../update-speed-audit-20261107/`, observed at Unix1794093696.979 (November7,15:21:37 Pacific), before the Review robot assembly run chat's18:46Pacific assessment. Excludes the later bare-v8/rollback and collision revisions. No live experiment, job, monitor or physics was changed in this investigation; no new benchmark was run.

The slowdown accompanies a change from the old Hiwonder/SO101 all-loose task to the new articulated WowRobo source-CAD contact model and staged starts. The collision adapter generalizes a previously selective supplement to every eligible source-SDF pair and forces the expensive assembly-search budget on those pairs. Physics substeps double. Staged configurations invoke more pairs and expose expensive near-contact regions. The earlier BVH and sparse-solver optimizations are still enabled. Point-table refinement and bank size do not explain the measured steady-state bottleneck.

## Reconstructing the approximately1.72× result

Saved old accelerated benchmark: `assembly-so101/artifacts_local/acceleration-20261107/full-ppo-result.json`, also versioned in `assembly-so101/benchmarks/so101-acceleration-20261107.json`:

- Old no-driver SO101, MuJoCo4, **all-loose bank**,8192worlds×32controls=262144valid transitions/update.
- Same46simulation Slurm CPU slots across flat/dense, tree/dense and tree/sparse phases.
- Mean full iteration1504.905935→115.001668→105.576625seconds;13.086× tree gain and14.254× combined gain.

Earlier original-NIST capture: `../ppo-profile-20261106/summary.json` and README:7.561593764seconds,374Slurm CPU slots, same262144transition count.

```text
allocated simulation CPU-slot seconds per transition:
old accelerated SO101 = 105.576625105 × 46 / 262144 = 0.01852617
original NIST         =   7.561593764 ×374 / 262144 = 0.01078810
ratio = 1.71728 ≈ 1.72
```

This reconstructs the user's number. It is a cross-capture resource-normalized comparison, not a new matched-hardware NIST rerun after acceleration. Host/core placement, MuJoCo version, rollout packing, policy age, geometry and state distribution differ. Slurm CPU slots include SMT siblings; they are not guaranteed physical cores. Full-iteration normalization also allocates CPU time to GPU/checkpoint intervals. The earlier number is useful context, not an invariant speed ratio for later environments.

## Concrete changes in collision execution

### 1. Selective supplement became general source-SDF supplementation

Old `assembly-so101/plugin/so101_contact_search.c::supplement_mating_surfaces` handles only base↔motor and motor↔horn, with separate flags; other pairs return immediately.

The frozen pre-Review adapter at `../update-speed-audit-20261107/snapshot/robotics-assembly-3.0/assembly_policy/so101/wowrobo_contact_backend.py:51` adds a general route when **both names end in `__contact`** and the WowRobo guard is enabled. It obtains their original source visuals and runs the same bidirectional vertex/triangle-centroid supplement. It preserves the algorithm but greatly expands where it executes.

The later native model has54SDF collision geoms,130total geoms and17explicit contact pairs. Those are model counts, not54²executed pairs: broad phase, exclusions, body relationships and masks still select eligible pairs.

### 2. All these pairs take40×40search

Frozen adapter lines83–87 select `so101_assembly_search` for `__contact`↔`__contact` before old pair-role dispatch. The model's custom numeric is `40 40`. The old configuration distinguished general4×10, grasp16×20 and mating40×40. The new generic branch does not retain cheaper role-specific routing for those source-SDF pairs; the old finger-name matcher also does not describe the renamed WowRobo helpers.

Thus up to1600 nominal search iterations per call replace40 for a general route or320 for a grasp route. These budget ratios are **not measured runtime multipliers**: search may terminate early. Per-role qualification is necessary before lowering budgets.

### 3. Twice the substeps

Old versioned benchmark:0.5ms/80substeps per40ms action. Later physical builder explicitly sets0.25ms/160substeps. This doubles opportunities for collision work per policy action, holding trajectory and per-substep work fixed. It cannot alone explain the full regression.

### 4. Staged starts amplify geometry cost

The older full-PPO acceleration benchmark was all-loose. The later bank contains grasped, supported, engaged and partially assembled configurations. Same-model saved quick probes count:

| One40ms action | Loose | Upper far-side nominal | Upper far-side variation |
|---|---:|---:|---:|
| Collider invocations |480|5600|5600|
| Invocations/substep |3|35|35|
| Supplement leaf sample evaluations |8994240|33698830|34118834|
| Wall seconds, first quick repeat |1.364|8.120|8.244|

Source: `../update-speed-audit-20261107/baseline-quick-mode1.json`, stats0=pair calls, stats4=leaf sample evaluations. These are repeated **physics surface queries**, distinct from the448policy observation points. Tree pruning remains active but still leaves millions of evaluations. The nominal is already slow; the selected randomized upper state adds only~1.5% in this quick repeat. This is not a full randomization ablation or an estimate for every stage.

## Native attribution before Review

Separate saved split probe on the same frozen model:

| Start | Rollout s | Original SDF search s | Surface supplement s |
|---|---:|---:|---:|
| Loose |1.348|0.159|1.168|
| Cover near |3.104|0.881|2.178|
| Upper nominal |8.260|3.350|4.792|
| Upper variation |8.299|3.299|4.883|

Source: `../update-speed-audit-20261107/native-profile-split.json`. Four single-world zero-action diagnostic intervals; not whole-fleet percentages. Independent native phase timers put narrow phase at98.7–98.8% of stepping. On the upper variation, the supplement is~58.8%, ordinary SDF~39.8%; removing only the supplement has an ideal~2.43×ceiling on rollout time.

Saved run-v5 update11:616.900scollection/2.068sPPO,87260valid transitions. Mean per-round worker compute9.087s versus mean per-round maximum16.417s;32×maximum=525.344scritical-path compute. Remaining~91.6s includes dispatch/admission/sampling/network/Python; it is not measured network-only time. Barrier imbalance compounds expensive physics. More workers increase batch production but do not divide the32sequential control rounds.

## What this rules out

- **Lost tree/sparse settings:** frozen runtime explicitly calls `set_surface_mode(1)` and sets `mjJAC_SPARSE`. Native build receipts retain the accelerated headers. Turning them on again is not a fix.
- **448policy points:** old SO101 already used448; current refinement keeps448. Its report says physics/jobs/banks unchanged, initial table construction~6.6s then cached. Later observation refresh is~0.01–0.06s versus seconds of native collision work.
- **1800state entries by themselves:** each world simulates one selected state, not the whole bank. State *distribution* matters. Full1800bank completion/adoption must not be conflated; run-v5 used an earlier partial snapshot.
- **Mainly PPO/solver:** measured optimization~2s versus~617scollection; native constraint solve below1% in the contact-rich probes. Neither explains the dominant loss.

## Next diagnostic and repair targets

1. Pair-level counters on the frozen current model: separate helper grasps, real mating interfaces, passive bracket stops and other hardware. Include zero-supplement-contact pairs and exact tree rejection/leaf counts. The three loose calls/substep are consistent with persistent internal pairs, but their identities are **not proven by the aggregate counters**.
2. Replace blanket expensive routing with explicit pair roles and independently qualified search budgets. Avoid asserting that all40×40work is unnecessary; some interfaces depend on it.
3. Restrict or replace generic whole-source supplementation where contact/retention/insertion tests demonstrate equivalent behavior. Optimize ordinary SDF and supplement together.
4. Qualify0.5ms or phase-specific timesteps separately. Compare old/new configurations with matched stage distributions, worlds per physical core, hardware and valid-transition count.

Existing captures establish the bottleneck and the source changes. They do **not** isolate a percentage of the total regression attributable to each change; that requires controlled ablations. Report throughput per valid transition and per physical core alongside update latency.

## 2026-11-07 [codex] Profiling chat and requested timestep restoration

- User requested new actual pre-speedup profiling run: created local chat01a1b92b-7946-7310-88cd-0e24133b90a2, active and explicitly preserving frozen0.25ms native baseline.
- User requested removing doubled substeps. Local wowrobo_training_runtime.py now sets XML timestep0.0005 before serialization/model identity; policy40ms implies80substeps. AST and frozen-XML comparison PASS; only timestep changes in XML. Dynamic qualification/deployment pending.
- Automatic approval review REJECTED message to main-run owner requesting validation/versioned live deployment and monitor migration, citing unestablished authorization for existing-chat message/live side effects. No handoff delivered or live changes made. Explicit user approval requested. Do not describe deployment or validation as assigned/completed.
- Human subsequently explicitly approved main-run chat validation and live deployment. Authorization-bearing handoff sent to main-run owner; delivery/progress tool result is authoritative. Live verification remains pending; original profiling continues separately.
