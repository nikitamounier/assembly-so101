# WowRobo side-cover behaviour cloning
Goal: Seed a fresh side-cover PPO experiment with an actor pretrained on randomized kinematic trajectories.

To train one actor on all five stages of the continuous assembly, use the [full-build instructions](FULL_BUILD.md) and the included full-build reference.

`actor.pt` contains the pretrained `actor` and observation `norm` state dictionaries. Load those into the corresponding modules of the WowRobo ActorCritic. Initialize the critic, PPO optimizer, and sampler for the new experiment. The published WowRobo trainer rejects the legacy `--init-policy` flag, so apply the two state dictionaries at initialization in the receiving trainer.

The checkpoint uses the side-cover observation contract: 448 ordered XYZ points in the gripper frame, 250 proprioception values across five frames, and 12 normalized action outputs at 40 ms. The reference action scales are 0.035 rad per arm action unit and 0.05 rad per jaw action unit. The published side-cover-only observation signature is `822b7a5214eec356bdf05917af326da1a5fa163cdd4e1b92e2ceb85edcec2fcc`. Match the point order, proprioception layout, and action semantics when transferring the weights.

The fit used 16 randomized side-cover trajectories and held out four full trajectories. See [fit.json](fit.json) for the split and MSE. The labels came from collision-free kinematics with ideal joint tracking and an authored cover pose. Pickup and seating under native contact physics still need evaluation. Use fresh native rollouts for PPO.

Checkpoint SHA-256 is `1230371330f988b3bd8c07f2fc04e52bb8dc71f86fe6d2c84cefa1201d162a58`.

## Reproduce the fit

Check out `codex/behaviour-cloning`. From the repository root, use Python 3.12 with NumPy, SciPy, and PyTorch installed. The baked reference is included, so data generation and fitting run on the CPU without MuJoCo.

```sh
cd "behaviour cloning"
python wowrobo_kinematic_bc.py generate --model kinematic-cover-model.npz --episodes 20 --seed 43 --out repro-data
python wowrobo_kinematic_bc.py fit --data repro-data --steps 700 --seed 43 --out repro-fit
```

The output is `repro-fit/actor.pt` and `repro-fit/fit.json`. Use fresh output directories when repeating the commands. The original fit used AdamW with a batch of 256, learning rate 0.0003, weight decay 0.0001, and gradient norm clipping at 1.0. Expect 4,431 training samples and 1,109 held-out samples on the reference setup.

The shared checkpoint scored 0/6 on the MacBook's local closed-loop probe with approximate mesh contacts. The probe used the published seating criteria. The exact native PPO runtime result remains unmeasured.
