# Direct policy weight editing: proposed assembly pilot

## 2026-11-07 [codex] Scope and evidence

User asks to plan applying Lingxiao Guo's robot weight-editing demonstration to our model and clarify simulation's role. This is planning; implementation, evaluation jobs and policy promotion are not yet performed.

Primary sources read in browser:
- https://x.com/Lingxiao234/status/2118048183540662745 — reported pi0.5/RoboTwin, G1 RL and FastWAM results.
- https://x.com/Lingxiao234/status/2118048223210402109 — saved-input feature/attention probes, candidate edits on those inputs, then rollout-based adjustment and validation.
- https://x.com/Lingxiao234/status/2118048266378191115 — humanoid example rewires low-influence neurons to introduce a control prior.
- https://x.com/Lingxiao234/status/2118048278277493090 — authors hypothesize edits help when a skill exists but is blocked.

These are author-reported demonstrations, not a reproduced assembly result. No exact replication recipe/released robot-editing implementation was established from the thread.

## Model-specific adaptation

Inspected assembly-so101/assembly_policy/rl/nets.py, train.py and wowrobo/env.py. ActorCritic uses separate point-encoding SimBa actor/critic, width256/two residual blocks/1024 expansion, shared running proprioceptive normalization and state-independent Gaussian log_std. Current backend passes12 action dimensions explicitly (the class's historical default8 is not the current action layout). Current actions are relative targets for five joints and one gripper per arm at40ms control intervals. There are no attention heads in this network.

Use hidden-feature ablation and final actor-head connections/biases first. Do not assume an individual hidden unit has a clean semantic meaning; correlate within matched task/start/phase cohorts, then test causal interventions. Keep critic, running normalization, action noise, controller, geometry, contacts, rewards, safety and observation construction fixed for the initial actor-edit comparison. Any direct gain/controller change is a separately labeled control experiment.

## Proposed experiment

1. Freeze a checkpoint and complete environment manifest: policy/normalizer, bank/split, model/native plugin, observation/action, rewards, controller and seeds. Fork an evaluation copy; preserve the active learner.
2. Establish baseline full episodes for one reproducible failure family with some existing success. Prefer held grasp near seating for cover, or another task with a clear mixture of successes/failures. Separate structural/contact/reset failures from policy mistakes before editing. If there is no demonstrable partial skill, continue diagnosis/training first.
3. Collect bounded research traces: exact policy points and five-frame proprioceptive history, action means/std and sampled actions, selected activations, full replayable simulator state including controls/warmstart/friction, success/termination, material/contact and grasp diagnostics. Dashboard body-pose clips support review but are insufficient for exact policy-input probes or simulator branching. Existing lightweight training recorder remains unchanged.
4. Offline on saved inputs, compare matched successes/failures; rank features by output sensitivity and behavior near failure. Try reversible activation ablations as diagnostics. Screen small actor weight edits (initially1–4 scalars, widening only with a stated hypothesis) for finite outputs, action change and preservation of successful cases. No gradients needed for this track. Record every attempted edit and rationale.
5. Run baseline and candidate in independent native MuJoCo worlds from identical complete states and random seeds. Include policy action randomness consistently; evaluate both under the same deployment mode. Use short branches to eliminate bad candidates, then complete15s episodes from actual resets. Saved-input action changes alone cannot establish a fix.
6. Proposed budget: up to3 hypotheses/12 candidates,8–16 paired development episodes per candidate, followed by32–64 fresh paired selection episodes for at most2 candidates. Freeze the winner before a final untouched test of at least100 paired target-task episodes and a coverage panel spanning all five tasks/approved start families. These counts are planning choices, not a power calculation; increase based on baseline variance and effect size. If the existing EVAL set has already guided development, reserve new physically admitted final-test seeds first.
7. Report paired target-task success difference with uncertainty, task/start-family breakdown, safety/penetration, retained grasp, timeout, stable seating and action saturation. Compare unchanged baseline, selected edit, and matched-size random edits; a small PPO continuation with measured compute is an optional later baseline. Require a useful held-out success gain with no material safety or cross-task regression. Predeclare margins before selection. Do not equate subtask success with full loose-part assembly.
8. Save an immutable candidate checkpoint plus a patch manifest containing base hash, exact tensor/index/old/new values, rationale, all selection attempts and evaluation receipts. Do not put edited weights into a live PPO process: inherited Adam moments need an explicit versioned continuation policy if later training is requested. Hardware transfer follows only after sim validation and separate perception/calibration qualification.

## Why simulation and where compute goes

Offline forward passes and activation probes need no simulator; a GPU is convenient. Candidate behavior and final outcomes require closed-loop native simulation with current custom contact physics; CPU physics is the present expensive component. Reuse a bounded isolated evaluator and parallelize independent paired episodes when authorized. Do not combine the editing experiment with an Isaac/Warp/timestep migration: changing both policy and physics prevents attribution.

Training supplied simulator geometry is not a camera deployment pipeline. Real transfer later requires camera-derived compatible observations, calibrated actions/latency and physical fit/friction/grip checks. Post results do not establish transfer or promise that four edits will unlock our unfinished assembly skill.

Next concrete deliverable, if implementation is requested: checkpoint-bound paired evaluator plus bounded trace capture for one chosen failure, then the first offline probe report. No cluster work or running-model edits were performed for this plan.
