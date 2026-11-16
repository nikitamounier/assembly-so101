# Rewards and evaluation

**Goal:** Make this section’s user intent, implementation and unfinished work easy to review.

Reward a correctly completed placement using the shared native task checks. Each task checks its actual receiving frame and the required predecessors. Success is sparse/binary, with control cost and the configured safety penalty; a dense shaping redesign is not selected.

**Current limits:** Relevant placement keypoints within 1 mm, release, and approximately 0.4 s stable retention. Pose stability uses 0.25 mm translation / 0.01 rad across eleven samples. Original-material and native penetration acceptance is now 0.5 mm. Screw completion also needs its thread/contact conditions. The supported launcher and SGS checks explicitly enable the native-depth guard; the low-level API retains its opt-in global guard, while the original-material success guard defaults to 0.5 mm.

Wrong placement, wrong predecessors, invalid overlap and already-completed starting states must be handled explicitly. A physics-valid drop does not necessarily assemble a part. Grasped partials must remain unfinished; do not fill the bank with trivial completed poses.

**Difference from AssembleNIST:** Its reference completion criterion is assembly error below threshold. This SO101 fork additionally checks release, stable dynamic predecessors and original-material overlap. Those additions are part of the current task definition and remain visible for your review.

**Evaluation:** Keep TRAIN and EVAL seeds/states independent. Report outcomes by task and family, and report subtask-loose separately from all-loose full-sequence evaluation. Compare initial/final policies on the same independent starts. Report success, safety stops, timeout and placement error, not loss alone.

**Evidence:** The pre-update v14 reward audit passed 34/34 cases. The new acceptance boundary has eight focused passing tests; all 50 pilot starts pass nonterminal/native qualification, with 60 production/fine-step receipts. Full production-bank qualification remains pending.

**For your review:** These are the intended success conditions. Hardware calibration and end-to-end learned assembly remain unproven.
