# Policy, PPO and SGS

**Goal:** Make this section’s user intent, implementation and unfinished work easy to review.

Keep AssembleNIST’s PPO, SimBa actor/critic and per-entry SGS as the starting point. Train one shared policy across the five task identities; separate tasks do not imply five independent policies or a solved assembly sequence.

The inherited networks use width 256, two residual blocks, expansion 1024 and a 256→128 ELU point encoder. Training defaults remain five PPO epochs, four minibatches and initial action standard deviation 1. Learned world models, GRPO and larger-model experiments in the main plan are research ideas, not changes to this first baseline.

SGS chooses among existing reset entries using recent success history. The inherited sampler uses `epsilon + sqrt(p * (1-p))`, with a 50-outcome history by default. It does not generate new poses, enforce family quotas or prescribe a manipulation sequence. Balanced initial counts supply coverage; sampling weights then adapt.

**Current state:** The network/sampler implementations and PPO objective match the inherited reference checks. Checkpoint restore binds model, bank, observation layout, environment configuration and optimizer/normalizer/sampler state. Fresh weights are the baseline; do not resume a 0.4 mm objective as a 0.5 mm run silently.

**Evidence:** The v14 baseline passed 82 core tests and 31 preflight checks. Fresh guarded preflight passes 31 checks and a new RTX 5080 CUDA software/update/restore check passes 13/13 after the acceptance update. The complete 50-state staged pilot now independently passes 31 strict preflight and 13 CUDA update/restore checks. Its supervised bounded experiment is active; actual learning outcomes remain to assess.

**Still unknown:** Whether this policy learns the real placement tasks at the available throughput. Finite losses, gradients and changing weights are software evidence, not assembly success.

See [upstream project](https://alpha-nist.github.io/) and `assembly_policy/rl/{nets,sampler}.py`.
