# Oracle handoff scope

**Goal:** Reproduce the learner, identify software defects and then run a larger bounded stress session, without requiring solved assembly.

**Current shared environment:** the local main checkout now uses `shared-side-v3` through the same factory as Quest. Bootstrap installs the pinned geometry runtime and generates current reset/reference data. Earlier snapshot/results below are historical; see [current environment](CURRENT_ENVIRONMENT.md) and [shared validation](SHARED_WORKCELL_VALIDATION.md). Full motor seating and hardware calibration still fail qualification; the complete Quest app is now included under `teleop/`; see [teleop handoff](TELEOP_HANDOFF.md).

Keep the reviewed SO101 task code and original PPO/SimBa/SGS implementation. Add a pinned Linux/x86-64/CUDA setup, hash-checked private native runtime bundle, native CPU simulation with one CUDA learner, learner smoke/checkpoint checks, small learning regressions, and a Slurm single-node entrypoint. Verify actual weight/gradient updates, finite observations/returns/losses, sampler updates, checkpoint restoration and resumed counters. Report task success/safety/timeouts separately. A falling combined PPO loss is not required and does not certify learning.

Fix defects found by meaningful regression tests in this handoff branch. Preserve prior failures. No remote Oracle access or cluster credentials are available here; Nikita submits the supplied job using his cluster's allocation. No GPU physics, multi-GPU or multi-node throughput qualification is claimed by this handoff. The existing optional GPU backend remains research code with its existing requirements.

Use the versioned Quest physical snapshot described below, retaining policy contact/stability guards. Headset input processing remains separate. Source histories and licenses stay intact; no original vendor CAD or unrelated research/personal documents enter the repository.

H100 follow-up: include the observation generator's runtime dependency `trimesh`, save an unupdated initial policy after the first real normalization rollout, and compare initial/final mean policies on the same independent EVAL starts. Provide a single-GPU command that uses the allocated CPU affinity/Slurm budget, supports explicit environment/time overrides, and reports health separately from task improvement. The H100 configuration remains a starting allocation until measured there.

## Latest Quest environment migration

The user explicitly requests copying and pushing the latest environment. Freeze the current Feather scene configuration, including reverse-assembly part poses, opposing helper mounts, effort caps, cameras, current contact-search parameters, the motor/base seating pair and free-body damping. Keep native contacts and existing policy scoring/guards; headset timing and clutch controls do not become policy actions. Use a versioned reset bank and private runtime release; preserve old banks/runs/releases and reject incompatible checkpoint resumes. Verify native model parity against the frozen source and actual learner execution before publishing. This selection supersedes the earlier canonical-layout-only handoff decision.
