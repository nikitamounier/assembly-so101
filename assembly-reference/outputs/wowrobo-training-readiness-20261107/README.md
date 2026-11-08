# Current assembly pilot preflight

No current-WowRobo training run was launched. The viewer's new geometric seating predicate is a shared reward contract, not yet an episode-return or termination implementation.

The exact currently displayed scene has zero gravity, no enabled collision geometries and no explicit contact pairs. It is suitable for saved-pose review. The existing training entry point still loads the older assembly and older observation tables. The current review bank has 29 rejected and 46 pending variations, with none physically admitted. A loaded-grip probe also fails; see outputs/cover-support-fix-20261107/README.md.

A meaningful bounded pilot first needs the current native contact model, actions, point builder, reward and termination wired together, plus a physically qualified small reset set and positive/negative reward checks. Then assess episode success, safety failures, returns, episode lengths, throughput and actual replay videos before scaling. Existing jobs remain untouched. Exact identities and counts: preflight.json.
