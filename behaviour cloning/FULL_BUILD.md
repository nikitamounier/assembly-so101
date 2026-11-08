# Full-build behaviour cloning
Goal: Train one WowRobo actor on the continuous build from loose parts through final release.

Check out `codex/behaviour-cloning`. Use Python 3.12 with NumPy, SciPy, and PyTorch. From the repository root:

```sh
cd "behaviour cloning"
python wowrobo_full_bc.py generate --model kinematic-full-build-model.npz --episodes 60 --seed 43 --out full-build-data
python wowrobo_full_bc.py fit --data full-build-data --steps 3500 --seed 43 --out full-build-fit
```

The full-build checkpoint will be `full-build-fit/actor.pt`, with validation MSE for each stage in `full-build-fit/fit.json`. Use fresh output directories on reruns. The existing `actor.pt` in this folder is the earlier side-cover checkpoint.

Each episode covers the side cover, shoulder connector, pin, upper arm, and forearm in order. It includes both support handoffs and the final release. The generator perturbs the loose position and yaw of each part and retargets the approaching arm. It blends to the original receiver-relative insertion, so the seated geometry stays registered. It removes the idle opening prefix and retimes commands to 40 ms ticks. The baked data includes the ordered point tables for all five tasks. Generation and fitting run on CPU without a simulator, CAD downloads, or the original video.

The fit holds out whole builds and trains a single actor across all five stages. The data selects the active part's 448-point cloud from the reference chapter and carries the five-frame proprioception history across stage changes. For a continuous rollout, the caller must supply that same active-task observation and advance it at the corresponding handoffs without resetting the assembly. The current native trainer's per-task episodes have their own horizon and termination. This package trains the weights and normalization, and does not add a native full-build sequencer.

The targets are kinematic references with scripted part poses. Validation MSE measures imitation on held-out kinematic builds. Native contact and end-to-end assembly success require separate rollout evaluation.
