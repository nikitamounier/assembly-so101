# User-intent review by section

**Goal:** Let you review each part of the assembly project separately.

Review date: November 6, 2026 (Pacific). Each file separates your instructions, current implementation/evidence and remaining work. These documents do not imply new approval or pause the work you already requested.

| Section | Current status |
| --- | --- |
| [Assembly scope and hardware](assembly/USER_INTENT.md) | Scope modeled; hardware uncalibrated |
| [CAD, contacts and shared environment](physics/USER_INTENT.md) | 0.5 mm implemented; nominal baseline first |
| [SGS states and the nearby arm](sgs/USER_INTENT.md) | All 50 pilot states and 60 replay receipts pass |
| [Policy inputs and actions](observations/USER_INTENT.md) | Selected 448-point contract implemented |
| [Policy, PPO and SGS](policy/USER_INTENT.md) | Staged pilot preflight/CUDA checks pass; learning under assessment |
| [Rewards and evaluation](rewards/USER_INTENT.md) | Pilot native gates pass; learned outcomes pending |
| [Training run and compute readiness](training/USER_INTENT.md) | Bounded pilot active; full 8,000-state bank pending |
| [Meta Quest teleoperation and views](teleop/USER_INTENT.md) | 84-test baseline passes; live service stopped |
| [Simulation speed and engine comparison](performance/USER_INTENT.md) | Comparison follows qualified publication |
| [Repository, publication and evidence](repository/USER_INTENT.md) | Latest collaboration snapshot authorized; full-bank completion pending |

Start with **Physics**, **SGS states**, **Rewards** and **Training** for run readiness. Refer to a section name when requesting a change. The concise project agreement is [User Intent.md](../../User%20Intent.md); decision history remains in [MEMORY.md](../../MEMORY.md).

Cross-checked against the [main Google Doc](https://docs.google.com/document/d/1EHdy_YtqkQP9oIOBMTXQc2aok_c8v6XVW6KqGrLY-wo/edit?tab=t.0), read through the connected Drive tool. Later chat decisions take precedence over its older point counts, screw-scope notes and proposed arm-support priors. Friction variation is an implementation gap; world-model/GRPO items remain research ideas.
