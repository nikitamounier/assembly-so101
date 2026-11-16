# SO101 part and mating observations

**Goal:** Represent the active part, full predecessor assembly, fingers and screw-task screwdriver within the requested combined mating budget.

## Current presets

| Preset | Active part | Assembly during screw | Screwdriver during screw | Fingers | Total points |
| --- | ---: | ---: | ---: | ---: | ---: |
| 256 combined mating, default | 128 | 128 | 128 | 64 | 448 |
| 128 combined mating | 128 | 64 | 64 | 64 | 320 |

**Assembly plus screwdriver together equal the selected mating budget.** The screwdriver is not an additional budget. The split is 50/50 for the screw task. Other tasks give the entire 256 or 128 budget to their mating assembly. There are no extra tool slots or zero-padding slots.

SO101 training defaults to256 combined mating; `--n-fix 128` selects the lower preset. Standard NIST remains128 mating, unchanged. Both production presets are included in the conversation view's Combined mating budget selector.

The user selected **256 combined mating** for use. The [standalone visualization](visualizations/so101-mating-budget.html) is included in the repository with256 selected by default;128 remains a comparison option. See [opening and snapshot identity](visualizations/README.md).

## Assembly order and coverage

Mating contains the full base and every predecessor in canonical order:

| Active task | Mating assembly |
| --- | --- |
| Motor | Base |
| Horn | Base, motor |
| Screw | Base, motor, horn; screwdriver shares the combined budget |
| Retainer | Base, motor, horn, screw |
| Yoke | Base, motor, horn, screw, retainer |

Assembly samples are distributed by native surface area, with a four-point presence floor and deterministic largest-remainder rounding. This preserves full-body coverage without overrepresenting the tiny screw. At256, yoke receives base142/motor62/horn6/screw5/retainer41. At128 it receives67/31/5/4/21. There is no local mating crop.

The screwdriver's share covers body, saddle, bit and both buttons with relative weights40:8:8:4:4. Points follow each actual body, without double-counting welded descendants. Fixed farthest-point coverage is reused as bodies move. Fingers remain16distal points per fixed/moving finger-related body, excluding palm/servo shell. Geometry stays in the left palm frame, in decimetres.

The view initializes native-qualified predecessor completions and leaves the active part loose. These are prepared inspection scenes, not policy execution, a reset-curriculum change or continuous sequencing. Both presets use the same stage poses and reference meshes for comparison.

## Compatibility and checks

The observation signature binds sampler source, task order, counts and seed. Resume, weights-only initialization, SO101 evaluation and checkpoint rollout reject missing/mismatched signatures. The two presets have different input dimensions and need matching newly trained weights. Physical model/reset-bank identity, live Quest and running learners are unchanged.

- After merging current v9 tool/reward changes:47repository tests passed,1optional skip. All13real CPU PPO/optimizer/sampler/checkpoint/restore smoke checks pass using the selected default. All320v9 TRAIN/EVAL states and native reference were generated locally with no rejected starts. CUDA qualification remains a separate check on the training host.
- Actual production tensors exported for all five stages at each preset. The browser checks both presets, task-only tool visibility, exact total point counts, finite projections, themes and narrow layout.
- Existing native pose/surface audits remain historical checks of the unchanged rigid-body transformation path; no task-learning improvement is claimed.
- Local receipts and current exporter: ignored `artifacts_local/so101-points-v6/`.

## Files changed and follow-up

- `assembly_policy/so101/obs.py`: combined mating/tool allocation, full predecessor surfaces, area quotas and distal fingers.
- `assembly_policy/train.py`: suite-specific mating defaults and observation compatibility guard.
- `assembly_policy/so101/evaluate.py`, `policy_demo.py`: incompatible-input rejection.
- `tests/test_so101_observation_roles.py`: role, allocation, coverage and two-preset regressions.
- Conversation `so101-mating-budget-points.html`: selectable128/256 combined budgets.
- `docs/visualizations/so101-mating-budget.html`: standalone export, default256.
- `docs/visualizations/README.md`: download/open instructions and frozen snapshot identity.
- `User Intent.md`, `TASK.md`, `MEMORY.md`, `docs/POLICY_INPUT_VIEW.md`: current requirement and implementation status.

Follow-up: fresh training/evaluation with the chosen preset. Publication status and verified source commit are recorded in `MEMORY.md`; live services remain unchanged.
