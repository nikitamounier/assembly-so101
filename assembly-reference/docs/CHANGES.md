# Handoff source changes

**Goal:** List every source file touched for the private Oracle handoff.

| File | Modification |
| --- | --- |
| [.gitignore](../.gitignore) | Exclude local caches, runtime artifacts and credential files. |
| [README.md](../README.md) | Replace the entrypoint with current structure, setup and handoff limits. |
| [MEMORY.md](../MEMORY.md) | Record the handoff scope and verified learner fixes. |
| [assembly_policy/train.py](../assembly_policy/train.py) | Keep all valid minibatch samples, guard tiny/empty batches and checkpoint SGS atomically. |
| [scripts/so101_preflight.py](../scripts/so101_preflight.py) | Check pinned runtime provenance/objective expressions and execute a fresh thread bench. |
| [configs/cloud_stress.json](../configs/cloud_stress.json) | Declare a bounded one-GPU/native-physics stress profile. |
| [configs/runtime_bundle.json](../configs/runtime_bundle.json) | Pin every member and the checksum of the private runtime release. |
| [requirements-cloud.txt](../requirements-cloud.txt) | Pin the tested core runtime versions. |
| [requirements-cloud.lock](../requirements-cloud.lock) | Resolve and pin all 51 cloud dependencies. |
| [scripts/cloud/bootstrap.sh](../scripts/cloud/bootstrap.sh) | Install dependencies/runtime and compile native plugins. |
| [scripts/cloud/runtime_bundle.py](../scripts/cloud/runtime_bundle.py) | Build/install a checked runtime archive without original CAD. |
| [scripts/cloud/healthcheck.py](../scripts/cloud/healthcheck.py) | Exercise actual CUDA PPO, gradients, state changes, SGS and resume. |
| [scripts/cloud/run.py](../scripts/cloud/run.py) | Preflight and launch bounded runs, retain diagnostics and forward TERM. |
| [scripts/cloud/train.sbatch](../scripts/cloud/train.sbatch) | Provide a single-node Slurm job entrypoint. |
| [tests/test_learner_batches.py](../tests/test_learner_batches.py) | Regress valid-sample coverage and tiny/empty rollout handling. |
| [tests/test_checkpoint.py](../tests/test_checkpoint.py) | Regress embedded SGS consistency and explicit legacy failures. |
| [tests/test_sampler.py](../tests/test_sampler.py) | Check history wrap, exploration probability and RNG restore. |
| [tests/test_launcher_signal.py](../tests/test_launcher_signal.py) | Verify trainer termination forwarding with an actual child process. |
| [docs/ARCHITECTURE.md](../docs/ARCHITECTURE.md) | Explain subsystem ownership and a staged organization proposal. |
| [docs/HANDOFF_SCOPE.md](../docs/HANDOFF_SCOPE.md) | State software-health acceptance independently of task success. |
| [docs/HANDOFF_VALIDATION.md](../docs/HANDOFF_VALIDATION.md) | Report actual local passes and unqualified Oracle/resource claims. |
| [docs/ORACLE_HANDOFF.md](../docs/ORACLE_HANDOFF.md) | Give Nikita setup, Slurm/direct launch, stop/resume and bug-report steps. |
| [docs/UPSTREAM_README.md](../docs/UPSTREAM_README.md) | Preserve the original upstream overview and results. |
| [docs/CHANGES.md](../docs/CHANGES.md) | List the complete source-file handoff inventory. |

Original `projects/assemble_so101`, the separate Feather tree, task geometry/effort settings, network architecture and SGS equations were intentionally not edited. The new private release carries runtime derivatives and reset banks only. Existing raw CAD, private plans, credentials and past run directories are excluded. Local generated checks and artifacts remain ignored.

Follow-up: Nikita accepts his GitHub invitation, runs the smoke check on the actual Oracle allocation, then submits the bounded stress run. Oracle credentials/GPU architecture were not available during local qualification.

## H100 follow-up

| File | Modification |
| --- | --- |
| [requirements-cloud.txt](../requirements-cloud.txt) | Add the missing trimesh runtime dependency. |
| [requirements-cloud.lock](../requirements-cloud.lock) | Pin the 52 resolved runtime packages. |
| [assembly_policy/train.py](../assembly_policy/train.py) | Save fitted normalization and unupdated weights before the first PPO update. |
| [assembly_policy/so101/evaluate.py](../assembly_policy/so101/evaluate.py) | Add mean actions, per-episode seeds and placement error metrics. |
| [scripts/cloud/healthcheck.py](../scripts/cloud/healthcheck.py) | Verify the saved baseline weights and normalization. |
| [scripts/cloud/test_learning_h100.sh](../scripts/cloud/test_learning_h100.sh) | Run smoke, allocation-aware training and initial/final assessment. |
| [scripts/cloud/assess_learning.py](../scripts/cloud/assess_learning.py) | Compare policies on identical independent EVAL starts. |
| [tests/test_learning_assessment.py](../tests/test_learning_assessment.py) | Regress conservative interpretation of task improvement. |
| [README.md](../README.md) | Link the one-command H100 workflow. |
| [docs/ORACLE_HANDOFF.md](../docs/ORACLE_HANDOFF.md) | Explain CPU/GPU use, settings, monitoring and result interpretation. |
| [docs/HANDOFF_SCOPE.md](../docs/HANDOFF_SCOPE.md) | Record the requested learning-assessment scope. |
| [docs/HANDOFF_VALIDATION.md](../docs/HANDOFF_VALIDATION.md) | Report the new tests and their hardware/resource limits. |
| [MEMORY.md](../MEMORY.md) | Preserve the H100 workflow decision and known dependency issue. |
| [docs/CHANGES.md](../docs/CHANGES.md) | Extend the exact source-file inventory. |

Task geometry, friction, effort caps, reset banks, network architecture and PPO update equations were intentionally unchanged. Nikita still needs to run the H100 workload on his allocation.

## Living user-intent documentation

| File | Modification |
| --- | --- |
| [User Intent.md](<../User Intent.md>) | Reconcile current goals, hardware/task scope, shared environment, measured architecture, compute and unresolved work. |
| [AGENTS.md](../AGENTS.md) | Require intent maintenance and canonical-main/shared-builder coordination across assembly chats. |
| [README.md](../README.md) | Link intent and maintenance instructions at the entrypoint. |
| [MEMORY.md](../MEMORY.md) | Record the living-document decision and source corrections. |
| [docs/CHANGES.md](../docs/CHANGES.md) | Record this documentation-only change inventory. |

Runtime source, physics assets, live Feather service, existing runs and other pending environment edits were intentionally untouched by this documentation update.

## Canonical shared workcell and side-orientation integration

Full per-file inventory, modifications and verification are in [shared workcell verification](SHARED_WORKCELL_VALIDATION.md). Training and Quest use one canonical factory; the Feather working layout/native duplicate are removed. Old media and unrelated edits are preserved. Changes are local on main and are not published remotely; full motor seating and hardware calibration remain unresolved.

## Complete Quest application publication

The shared environment changes and complete simulation-only Quest app are included in this repository. See [teleop handoff](TELEOP_HANDOFF.md) for verification and the full current source-file inventory. The earlier separate-Feather/unpublished status above is historical.
