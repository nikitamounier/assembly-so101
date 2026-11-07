# WowRobo stage-specific randomization

The user approved this distribution design on November7. Its executable specification is [wowrobo_randomization.json](../configs/wowrobo_randomization.json). The25 accepted nominal poses remain in `assets_local/wowrobo/photo-resets-v3`; randomization creates separate candidates. These bounds are starting envelopes, not measured manufacturing tolerances or proof that every draw is physically feasible.

The target bank has640TRAIN and128independent EVAL candidates per active stage:16,000TRAIN +3,200EVAL. Train difficulty is256easy,256medium,128hard, with one difficulty value coupling that candidate's pose offsets. Evaluation uses independently seeded stratified difficulty across the same qualified range. No duplicate padding or failed-candidate relabeling is permitted.8,192parallel environments sample the training pool independently of its size.

## Per-stage bounds

Lengths below are metres in the configuration. The table uses millimetres/degrees for review. Lateral radius is perpendicular to the current receiver insertion axis. Receiver frames inherit installed connector rotation. Engaged offsets must also fit registered physical clearances. Near states must remain outside completion tolerance.

| ID | Distribution |
|---|---|
|00 loose|Each loose piece±50mm XY; use nominal tangent orientation plus±2°tiny yaw noise, retain nearest grasp end and settle. Expansion toward±80mm requires validation.|
|01 cover pickup|Placement±30mmXY; side-wall grip slide±2mm, grip angle±2°.|
|03 cover aligned|Lateral≤4mm, angle±3°, stand-off40–65mm.|
|04 cover entry|Axial progress±2mm, lateral≤.5mm, angle±1°.|
|05 cover near|Axial progress±.4mm, lateral≤.2mm, angle±.5°.|
|07 connector pickup|Placement±30mmXY; grip slide±1.5mm; finger length horizontal.|
|08 connector aligned|Stand-off45–80mm, lateral≤3mm, angle±2°; horizontal approach constraint.|
|09 connector entry|Axial progress±2mm, lateral≤.4mm, angle±1°; horizontal approach.|
|10 connector near|Axial progress±.4mm, lateral≤.2mm, angle±.5°; horizontal approach.|
|12 pin pickup|Placement±20mmXY; grip height±2mm along large upper cylinder only.|
|13 pin aligned|Stand-off20–40mm, lateral≤.75mm, shaft tilt±1°.|
|14 pin partial|Axial progress±2mm, lateral≤.1mm, tilt±.4°, capped by hole clearance.|
|15 pin near|Axial progress±.25mm, lateral≤.05mm, tilt±.25°, capped by hole clearance.|
|17 left support acquisition|Both valid contact patches slide±1mm; recompute IK and reclose/settle; preserve support.|
|18 upper floor grasp|Placement±30mmXY; case grip slide±1.5mm, angle±2°; approved pickup orientation.|
|20 upper transport|Held case±20mmXY/±10mmZ, feasible orientation±5°, link sag5–11°.|
|21 upper far-side support|Rest point along edge±1mm, held-case stand-off±3mm; recompute resting angle.|
|22 upper partial|Axial progress±2mm, lateral≤.2mm, alignment±.5°; far-end support retained.|
|23 upper near|Axial progress±.25mm, lateral≤.1mm, alignment±.25°; far-end support retained.|
|25 right support acquisition|Valid fork contact patch slides±1mm; reclose/settle tight bilateral grip.|
|26 final floor grasp|Placement±30mmXY; servo3case grip slide±1.5mm, angle±2°; approved wrist branch.|
|28 final transport|Held case±20mmXY/±10mmZ, feasible orientation±5°, link sag5–11°.|
|29 final far-side support|Rest point along upper edge±1mm, held-case stand-off±3mm along tilted receiver; recompute resting angle.|
|30 final partial|Axial progress±1.5mm, lateral≤.2mm, alignment±.5°; far-end support retained.|
|31 final near|Axial progress±.25mm, lateral≤.1mm, alignment±.25°; far-end support retained.|

Pickup families target80%settled bilateral held grasps and20%close pickup approaches. Approach gaps are1–3mm, reduced to.5–1mm for the pin. Held transport/insertion families require a retained grasp. Waiting arms may receive±3° joint noise only inside verified clear regions. Supporting arms are repositioned through contact constraints/IK, never independent joint jitter. Sag changes the free link around its held shaft while preserving held-case/gripper coherence.

## Friction and installed assembly rotation

Sliding friction is multiplied byU(.97,1.03) once per environment episode for finger–part, part–table and part–part contact categories. Store each draw in reset metadata and keep it fixed during the episode. Torsional/rolling and joint friction remain nominal. Verify the actual effective contact coefficient after MuJoCo priority/mixing rules; randomizing an overridden geom parameter is not meaningful variation. Shared-model workers must isolate environment-specific parameters.

Starting at12, sample connector rotation initially±3° around the actual servo1swivel axis and pivot. Expansion toward±8° is validation-gated. Keep the base fixed. Propagate one shared transform through installed descendants, receiver pockets, target axes and observation points. Connector-only is installed in12–15; connector+pin in17–23; connector+pin+upper in25onward. Incoming loose/unattached pieces do not inherit the rigid assembly transform, but their mating targets do. Recompute affected grasps and robot IK while preserving the other arm's actual contact role.

## Admission and provenance

Every sample needs stage/split/seed, difficulty, noise draws, material multipliers, connector angle, native model/state/control identity, support/mating references, and explicit admission status. Candidate generation, kinematic/geometric review, physical settling, loaded retention, partial-insertion feasibility and nonterminal reward checks are separate outcomes. The current nominal CAD has known fitting conflicts; noise cannot silently repair or qualify those conflicts. Accepted physical bank counts must count only passing distinct samples. Rejections remain available with reasons; they never count toward the16,000/3,200target.

## Implemented candidate tooling

`wowrobo_randomization.py` samples kinematic candidates from the frozen v3 poses, with stable stage/split seeds, exact difficulty strata, coherent connector inheritance and affected-arm IK. Supported insertion noise is projected onto contact-preserving hinge/edge motion; the receipt distinguishes those constrained freedoms. This is not a settling or reward validator.

`wowrobo_friction.py` supplies per-data effective-contact friction and an Euler split-step constraint-rebuild hook. It leaves the shared model unchanged. A physical backend must invoke that hook for each fresh contact set; this utility is not automatically installed into the production training loop. The reset validator now invokes friction on each fresh contact set through `wowrobo_physical_resets.physical_split_step`, including a separately tested implicit/implicitfast constraint rebuild; production training integration remains separate.

`build_wowrobo_randomized_bank.py` now runs resumable per-slot rejection sampling. `--full --validation physical` requests640TRAIN/128EVAL accepted states per stage; retries retain each slot’s difficulty stratum. Only physically admitted distinct snapshots count. Model/config/source identities and authored, repaired physical-input, settled state/control hashes are checked on resume. Failed attempts remain separate. Qualified snapshots use the actual physical model/specification; viewer snapshots use the matching source visual geometry and carry the distinct validation model signature. `--max-total-attempts` bounds a diagnostic run; exhausted retries produce an explicitly incomplete bank, not padded data. The 19,200-state production bank is not claimed as built.

```sh
PYTHONPATH=. ../assembly-so101/.venv/bin/python scripts/build_wowrobo_randomized_bank.py --stages 00_loose 08_connector_aligned 20_upper_transport --difficulty 0.25 --output assets_local/wowrobo/randomization-pilot-v2
PYTHONPATH=. ../assembly-so101/.venv/bin/python scripts/build_wowrobo_randomized_bank.py --full --output assets_local/wowrobo/randomization-full-preflight-v2
```


## Current variation collections

The current review collection `assets_local/wowrobo/randomization-bank-pilot-v2` contains25 nominal references plus75 sampled variations (two TRAIN and one EVAL per stage at difficulty0.5). It has24 geometric/IK candidates pending physical checks and51 rejected draws; zero physically admitted starts. `progress.json` and the viewer manifest report actual generated/pending/rejected/accepted counts independently of the16,000/3,200targets. Nominal references never pad accepted quotas. The viewer also shows each loose piece’s actual signed tangent-yaw draw; these are independently sampled within±2°×difficulty. The saved attempts retain their original state and control hashes when viewer metadata is refreshed.

A first native physical pilot, `randomization-physical-pilot-v1`, ran gravity/SDF/sparse/friction dynamics on00/08. All four attempts were rejected: dynamic penetration exceeded0.5mm before the hold window completed;08 also lacked loaded finger-pad contacts. These are actual dynamics results, not the prior contact-off preflight. Its earlier pin geometry is historical; later physical profiles use the separately registered repaired pin.

```sh
PYTHONPATH=. ../assembly-so101/.venv/bin/python scripts/build_wowrobo_randomized_bank.py --validation geometry --train-per-stage 2 --eval-per-stage 1 --difficulty 0.5 --output assets_local/wowrobo/randomization-bank-pilot-v1
# Full physical rejection sampler, with a diagnostic total-attempt limit:
PYTHONPATH=. ../assembly-so101/.venv/bin/python scripts/build_wowrobo_randomized_bank.py --full --validation physical --physical-profile assets_local/wowrobo/physical-resets-v3 --pin-mesh assets_local/wowrobo/fit-pin-v2/pin.obj --max-total-attempts 75 --output assets_local/wowrobo/randomization-physical-bank-v1
```
