# Native registration review, 2026-11-07

No approved photo-v3 poses or vendor geometry were changed. These are separate
candidate/diagnostic poses, not admitted training resets. All native meshes and
all target pair checks remain in the audit; the strict review threshold remains
0.5 mm. A depth below that tolerance is still real material overlap, not positive
clearance or proof of rigid insertion.

| Registration check | Maximum relevant native sampled depth |
| --- | ---: |
| Original connector/horn | 0.855490 mm |
| Prior measured horn/slot normal correction | 0.436392 mm |
| Original upper/connector | 2.603910 mm |
| Prior actual case/socket frame correction | 0.273311 mm |
| Original final/upper | 2.56588 mm |
| Prior actual final case/pocket correction | 0.144422 mm |

`prior-feature-alignment-audit.json` checks the exact saved reference_qpos from
`outputs/linear-insertion-aligned-20261107/{connector,upper,final}/alignment.json`
against all native target surfaces. Pin is moved clear in this comparison to
avoid conflating the now repaired pin with the older probe registrations.
`composed-feature-alignment-audit.json` combines the corrected upper reference
with the final correction expressed relative to that upper receiver. The new
fit-pin-v2 registration is applied to the unchanged connector. This eliminates
the old multi-mm upper prerequisite and final-seat overlaps, but retains
0.273311/0.144422 mm local upper/final interference and 0.855490 mm connector/horn
interference. It has no fresh helper IK or force/settling qualification. The
candidate qpos is included for a separate review overlay only.

Connector evidence is stronger than a failed optimizer:
`../linear-insertion-diagnosis-20261107/connector/slot-plane-gap.json` records
37.250000 mm horn outside face span versus 36.400000 mm facing lip-plane gap.
The 0.850 mm local mismatch balances to about 0.425 mm per side. Source recesses
and chamfers mean these are local face dimensions, not a proof that all rocking
or compliant assembly paths are impossible. Straight rigid guided insertion
still has a measured normal obstruction; a tolerance pass cannot remove it.
A -0.425 mm normal correction also lowers the pin head stop; the current 14.5 mm
shaft would bottom there, so that correction has deliberately NOT been adopted.
The existing fit-pin-v2 dimensions/registration are unchanged.

The separate bounded translation grid found 0.485658 mm at [+0.3,+0.3,-0.48] mm;
this merely balances interference. Horn phase -90..90 degrees did not help
(0.854..0.856 mm). Cleaning native degenerate triangles and micrometre boundary
loops did not change the multi-mm upper/final failures. A fixed-case upper
translation/hinge search found 0.733496 mm but floated its old far end ~2.06 mm;
a stronger far-contact constraint worsened the native overlap to 4.13 mm. None
was adopted. These finite bounded searches do not prove global impossibility.

Reproduce the exact saved-feature checks from repository root:

```sh
PYTHONPATH=. ../assembly-so101/.venv/bin/python outputs/native-registration-review-20261107/audit_prior_feature_alignments.py
PYTHONPATH=. ../assembly-so101/.venv/bin/python outputs/native-registration-review-20261107/compose_feature_alignments.py
```

Next useful physical work is to establish whether source vendor horn/lip and
servo bevel/socket interference reflects the real printed clearance/compliance,
or select a properly supported rocking path. Do not resize vendor parts or
exclude contacts to qualify these poses.
