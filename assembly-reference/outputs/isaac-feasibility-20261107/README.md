# GPU simulation feasibility — 2026-11-07

## Finding

GPU rollout development addresses the measured bottleneck. Current WowRobo run-v5 update11 took616.8997s collection versus2.0684s PPO; isolated current-model probes put98.7–98.8% of native stepping in narrow-phase collision. These are recorded measurements, not an Isaac speed prediction.

The inspected retained assembly allocation3598717 on GPUCB52 exposes NVIDIA H10080GB HBM3, driver560.35.05. Isaac Sim's current official requirements explicitly exclude GPUs without RT cores, including H100/A100. No compatible RTX capacity was verified. This inspection does not prove every Chimera node has the same GPU model. Rendering-free PhysX/CUDA functionality is distinct from a supported Isaac Sim installation; headless alone must not be assumed to solve compatibility.

## Capacity check

Read-only scheduler inspection at2026-11-07 23:56:44UTC; oneGPU/eightCPUs/32GiB/twohours, non-submitting sbatch --test-only on both partitions. Ordinary preemptible projected Nov8 16:33:29 in scheduler-local time. Quick preemptible returned an earlier-than-observation estimate and explicitly listed preemption victims3602571,3512979,3544876,3617868; this is not proof of idle capacity and is unsuitable as an assumption of harmless spare compute. No jobs submitted/cancelled, drivers changed, training or monitors mutated.

## Recommended development sequence (proposal)

1. Make hardware compatibility the first gate. For Isaac Sim/Lab, identify an available supported RTX host and pin a matching Sim/Lab/driver combination. Do not upgrade shared cluster drivers.
2. On Chimera H100s, first evaluate current MuJoCo Warp support against the actual WowRobo model. Historical SO101 GPU prototypes are different task/model versions and do not establish current parity. Port all required custom SDF and supplemental contacts explicitly.
3. If evaluating Isaac, use an isolated backend within the canonical repository. Start with one real cover/connector insertion including both grippers and actual concave geometry. Then validate passive upper/final hinges and any retained thread mechanics. Isaac Lab Factory/FORGE provides relevant examples, not drop-in task equivalence.
4. Compare native and candidate reset validity, grip retention, interference/jamming, forces, passive motion, observations/rewards and success criteria. Engine trajectories need not match bitwise; behavioral acceptance must be explicit.
5. Batch64/256/1024 worlds; measure complete valid transitions/s and PPO iteration walltime including resets/observations/rewards/transfers, memory and failure rates. Scale to more GPUs only after a correct singleGPU result beats the CPU baseline. Policy tensors may be reusable, but normalizers/reset state/engine dynamics and checkpoint provenance require a versioned qualification.

## Sources

- Isaac Sim requirements: https://docs.isaacsim.omniverse.nvidia.com/latest/installation/requirements.html (checked2026-11-07; excludes H100/A100)
- Isaac Lab assembly examples: https://isaac-sim.github.io/IsaacLab/v2.3.0/source/overview/environments.html
- Current MuJoCo Warp capabilities/SDF hooks: https://mujoco.readthedocs.io/en/latest/mjwarp/
- Local bottleneck evidence: ../update-speed-audit-20261107/README.md

User asked whether to begin Isaac development; no production backend switch or Isaac job launch is implied by this feasibility assessment.
