# Assembly results on Vercel

This isolated deployment contains the existing dashboard and reset browser.
The static reset export retains saved poses, policy-point tensors, reward regions,
qualification receipts and state identities. Exporting uses kinematics only;
it never steps physics or changes a training run.

`scripts/export_states.py` exports all 19,225 saved states.
`scripts/pack_states.py` creates lossless sixteen-state gzip blocks for HTTP Range
requests, so selecting a reset does not download every variation of its stage.
`scripts/upload_assets.mjs` uploads only this explicit immutable asset set to the
project's Blob store. `scripts/build.mjs` prepares the small deployment directory.

`scripts/publish.mjs` sends the existing loopback dashboard's observations to Blob.
The website reads the latest committed object without CDN caching. It uses no
public tunnel and has no scheduler or training mutation endpoint. Observation
timestamps and the existing stale-data state remain visible if the publisher stops.
Keep the publisher running for new observations; already published pages and
snapshots remain hosted if the laptop goes offline.

The optional persistent native MuJoCo inspector and tolerance-editing controls
remain local; the hosted reset browser provides the same saved Three.js views,
policy points and reward overlays. Its selection is per browser, rather than a
shared server-side selection.

Credentials stay in ignored `.env.local` and `.vercel/`, never in browser assets.
Do not deploy the parent workspace, source CAD folders, checkpoints, credentials,
SSH configuration or generated research logs.
