# Handoff verification

**Goal:** Record what was actually exercised before Nikita's Oracle run.

**Current shared environment:** the local main checkout now uses `shared-side-v3` through the same factory as Quest. Bootstrap installs the pinned geometry runtime and generates current reset/reference data. Earlier snapshot/results below are historical; see [current environment](CURRENT_ENVIRONMENT.md) and [shared validation](SHARED_WORKCELL_VALIDATION.md). Full motor seating and hardware calibration still fail qualification; the complete Quest app is now included under `teleop/`; see [teleop handoff](TELEOP_HANDOFF.md).

Verified locally on Linux x86-64, Python 3.12, MuJoCo 3.17.0 and PyTorch 2.11.0+cu128 with an RTX 5080:

- **11 regression tests pass.** They cover dropped minibatch remainders, lost-worker masks, a single valid transition, an all-invalid batch, consistent SGS/model pairing, missing/mismatched SGS state, complete legacy checkpoint restore, sampler history wrapping, RNG restore, nonzero exploration probability and TERM forwarding to an owned trainer.
- The original four batch regressions all fail on the reviewed parent; those failures are retained locally. The original implementation discards remainders and produces empty/invalid statistics for tiny valid batches. The new loops use every valid transition and reject an empty rollout before optimization.
- **Actual CUDA/native learner smoke passes all eleven checks**, with two updates followed by restored iteration 3: 192 real environment transitions. Actor/critic gradients and weights, normalization and optimizer state change; losses remain finite; saved model, optimizer, SGS, iteration, transition count and learning rate restore exactly before the next reset draw. Removing the legacy sampler sidecar does not break a new atomic checkpoint.
- The private runtime archive was installed into this checkout after removing its temporary source-asset links. Every member hash passes. Original vendor CAD is absent. Both native plugins were rebuilt from source against MuJoCo.
- The bounded cloud launcher completes **30 fresh preflight checks and one real CUDA PPO update / 256 transitions**, exits zero and saves a checkpoint. The preflight checks the exact runtime derivatives/import provenance and original actor/critic objective expressions; it no longer incorrectly requires the complete PPO function to match before the recorded batch fixes. It reruns the guided thread bench instead of requiring a hidden historical JSON file.
- Shell/Python syntax checks and Git whitespace checks pass. The first checkpoint unit-fixture omission and first portable-preflight attempts are preserved in ignored local logs; the corrected passes above are the results used for this handoff.

The 448 total smoke/launcher transitions are a software execution check. Their task successes and safety stops are not a loose-assembly benchmark. The reset bundle contains 360 pooled TRAIN entries with prepared-state coverage, and independent loose EVAL states. The task, current curriculum and canonical training layout retain their known limitations.

All 51 pinned package dependencies resolve successfully; full fresh dependency installation, Oracle GPU/driver compatibility, the default 256-environment allocation, multi-GPU/multi-node scaling and exact simulator trajectory continuation have not been qualified here. The source scripts provide a reproducible smoke gate for the allocated Oracle node. Hardware throughput remains to be measured there; native simulation may be CPU-bound.

Local original receipts remain under `artifacts_local/cloud_handoff/` and `runs/local-launch-portable/`, excluded from Git. The concise source checks are reproducible via the root README; do not silently replace failed cases or infer successful learning from decreasing PPO loss.

## H100 command follow-up

The dependency lock now includes `trimesh` and resolves 52 packages. The updated native/CUDA smoke passes 13 checks, including an initial-policy snapshot with unchanged trainable weights and fitted normalization. Fourteen regression tests pass. The H100 command's shell syntax and dry-run resource construction pass locally; the H100 hardware and automatic 512–4096-world profile remain unqualified until run there. Same-cohort assessment now supports explicit mean actions and records success, safety, timeout and initial/best/final placement error. Its classification tests reject weight/error-only changes and gains accompanied by increased safety stops.

The final shell workflow also completes end to end on the local RTX5080 with8worlds,2simulation threads,one training update andone independent EVAL state per task/policy. Smoke passes13checks, preflight passes30checks, training exits zero, and allten assessment episodes complete. Initial/final task successes are0/5and0/5, with no safety stops; task improvement is correctly reported as not demonstrated. This is an execution check, not a learning claim. The earlier shell test was interrupted by editing its script while it ran; its log is preserved, and the final unchanged script completed successfully. The full H100 resource profile remains unbenchmarked.
