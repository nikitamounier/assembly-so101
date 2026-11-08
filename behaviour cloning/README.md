# WowRobo side-cover behaviour cloning
Goal: Seed a fresh side-cover PPO experiment with an actor pretrained on randomized kinematic trajectories.

`actor.pt` contains the pretrained `actor` and observation `norm` state dictionaries. Load those into the corresponding modules of the WowRobo ActorCritic. Initialize the critic, PPO optimizer, and sampler for the new experiment. The published WowRobo trainer rejects the legacy `--init-policy` flag, so apply the two state dictionaries at initialization in the receiving trainer.

The checkpoint uses the side-cover observation contract: 448 ordered XYZ points in the gripper frame, 250 proprioception values across five frames, and 12 normalized action outputs at 40 ms. The reference action scales are 0.035 rad per arm action unit and 0.05 rad per jaw action unit. The published side-cover-only observation signature is `822b7a5214eec356bdf05917af326da1a5fa163cdd4e1b92e2ceb85edcec2fcc`. Match the point order, proprioception layout, and action semantics when transferring the weights.

The fit used 16 randomized side-cover trajectories and held out four full trajectories. See [fit.json](fit.json) for the split and MSE. The labels came from collision-free kinematics with ideal joint tracking and an authored cover pose. Pickup and seating under native contact physics still need evaluation. Use fresh native rollouts for PPO.

Checkpoint SHA-256 is `1230371330f988b3bd8c07f2fc04e52bb8dc71f86fe6d2c84cefa1201d162a58`.
