# Current assembly trainer handoff — 2026-11-07

The user asked to switch the trainer to the new assembly and connect all task rewards, then stopped testing and delegated the run launch to another chat. This implementation has **not completed an end-to-end PPO test**. No training process or cluster job is running from this work.

## Entry point and model

The trainer is `../assembly-so101/assembly_policy/train.py`. Its default suite is now `wowrobo`; `--suite so101` explicitly preserves the older historical backend. Do not resume old checkpoints into the new suite. New checkpoints bind model, reset-bank, observation, reward and termination identities and reject incompatible continuation.

The current factory is owned by the canonical fork: `assembly_policy/so101/wowrobo_training_runtime.py`. The adapter in `../assembly-so101/assembly_policy/wowrobo/` bridges to it via `WOWROBO_CANONICAL_ROOT` (defaults to the sibling canonical checkout).

It loads `assets_local/wowrobo/physical-resets-repaired-cover-support-v2/profile.json`, not the viewer XML. This has real gravity, source SDF/triangle contacts, 0.25ms native steps, sparse solving, 12 actual helper actuators, passive target shafts and explicit installed-part equalities. It adds the canonical physical-reset module's current bracket/motor stop laws and a per-substep minimum-contact-distance sensor. Target and source assets remain unchanged.

## Environments and reward

One native environment and reward mapping per task:

| Policy task | Mating feature evaluated |
|---|---|
| base_side_cover | seated cover printed part |
| shoulder_connector | seated connector printed part |
| assembly_pin | seated pin |
| upper_arm_module | servo 2 case in connector pocket |
| forearm_module | servo 3 case in upper-arm pocket |

The shared predicate is `so101/wowrobo_reward.py`, matching the viewer's reward definition. Active and all preceding insertions must be seated within ±0.5mm in target axes and 1degree orientation error. Training additionally requires 0.4s stable seating before paying +1 once and terminating. Safety termination has no success reward; timeout is separate. Existing small action cost is retained. Contact depth at every native substep, nonfinite trajectories, clock resets and excessive joint/part speeds are guarded. Pose-only membership alone is not a successful episode.

Policy inputs use the same current `so101/wowrobo_obs.py` feature-aware 448-point encoder as the viewer, on actual current-body poses, with five frames of 50 proprioceptive values and 12 relative joint/gripper actions. Default control interval is0.04s (25Hz); default horizon15s. Observation/model references are checked independently of physics-only contact geometry. PPO/SimBa/GAE/SGS equations remain the inherited implementation. Local vector auto-reset, remote-worker suite dispatch, current-model evaluation dispatch, task metrics and checkpoint signatures are wired.

## Reset bank status — do not mislabel

There is still **no physically qualified production reset bank**. Default current bank is `assets_local/wowrobo/training-pilot-v1` and will fail closed until built from qualified source rows.

For integration only, a versioned diagnostic bank was generated at `assets_local/wowrobo/training-debug-v1`:61 rows across five tasks, explicitly `training_qualified=false`. Geometry-rejected source rows were excluded. Because no acceptable staged final-module training row exists, that task uses existing all-loose review variations, with all installed-part equalities off; this is a harder full-sequence diagnostic task, not a fabricated seated prerequisite. Other staged rows carry explicit equality-activation flags alongside full physics state. No review bank was modified.

Using this bank requires `--wowrobo-debug-resets`; config.json records `DIAGNOSTIC_UNQUALIFIED_RESETS`. It is not evidence of physical reset admission, successful learning or production readiness.

## Launch from the shared workspace

A minimal diagnostic invocation for the next chat (not executed after the user stopped testing):

```sh
cd '/Users/nikitamounier/Documents/ChatGPT/robot assembly 2'
PYTHONPATH=assembly-so101 assembly-so101/.venv/bin/python -m assembly_policy.train \
  --suite wowrobo \
  --bank-dir robotics-assembly-3.0/assets_local/wowrobo/training-debug-v1 \
  --wowrobo-debug-resets \
  --run-dir robotics-assembly-3.0/outputs/wowrobo-training-integration-20261107/pilot-next \
  --envs 5 --steps 4 --iters 3 --epochs 1 --minibatches 1 \
  --nthread 1 --device cpu --no-eval
```

For a real training run, qualify reset rows first and omit the diagnostic override. Preserve the separate existing PRIMARY/CUBE jobs and their frozen sources. Remote hosts need the canonical native source/profile/meshes/grids and matching reset manifests staged before launching workers; absolute asset paths are local and must be rebuilt/bound consistently on that host. This turn did not deploy or allocate anything remotely.

## Verification boundary

Before the stop-testing instruction, one current all-loose native step with zero actions completed with gravity/contacts, no safety termination and ~0.159mm peak native penetration. An initial PPO invocation failed before construction because an API export was missing; that export has been fixed. The final trainer/evaluator/remote dispatch changes have not been executed end-to-end. Formatting was applied, but no further tests or pilot run were performed after the user's instruction. The next chat owns execution and any resulting fixes; do not claim a trained policy or positive episode yet.
