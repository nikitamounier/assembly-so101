# Current assembly point cloud

The reset explorer at http://localhost:8876/ overlays the current WowRobo assembly observation directly in the main 3D scene. Independent Points and 3D buttons show either layer, both or neither using the same camera; state changes retain the layer selections. The lower section retains metadata, Perspective and JSON download, with no second rendering canvas. The older assembly is not sampled by this path. The server and web sources reside in the existing viewer host, `../assembly-so101/`.

`assembly_policy/so101/wowrobo_obs.py::PointTables(model)` defines the shared current-model input contract. Pass the loaded current model, then call its ordered tensor encoder with the current body's float32 poses. The browser preserves the exact exported flat tensor and reverses its recorded frame pose and scale solely for a world-aligned overlay. There is no resampling, smoothing or coordinate normalization. Every point is rendered over the meshes, including occluded points, so all 448 remain inspectable. All 448 points remain present: 128 active part, 256 task-specific receiving assembly, 64 distal fingers. The encoder uses the left fixed-finger body frame and multiplies metre coordinates by ten. Color is display-only, not an extra policy feature.

Samples are deterministic and cached in individual body coordinates. Servo cases retain their own pivot body, so samples follow articulation. Both fingers on both arms are sampled from this model, not copied from an older asset table. Source visual meshes are used; collision copies and helper markers are excluded. Sampling/FPS/budget functions and tensor-encoding math reuse the established policy routines without loading an older assembly model or changing existing checkpoint contracts.

The current WowRobo observation builder is available to policy consumers, but no WowRobo trainer/checkpoint is connected to this viewer. The UI explicitly says “no policy connected”; these are the shared encoder's exact current-state values, not evidence that an inference call happened. Setup states have no active task and show an empty-state explanation. The export includes the frame position and row-major rotation used by the encoder, ordered XYZ and flat arrays, body labels, role ranges, float32 dtype, frame/scale, state ID, observation signature and the raw float32 tensor SHA256.

The cloud travels in the same snapshot response as the scene poses. Existing selection sequencing prevents late responses from overwriting a newer state. Reset state and dynamics are never stepped by this feature. The current review collection remains physically unqualified where its existing status says so.

Verification: 17 Python viewer/encoding tests and 14 frontend tests pass, including inverse-frame alignment without input mutation; current native model API checked for all five assembly tasks. Independent transform checks cover order, scale, frame, finite shape, repeatability, articulated-body motion and no saved-state mutation. Browser checked reference/variation selection, interactive point rendering and downloaded JSON. Evidence: `outputs/point-cloud-viewer-20261107/`.

```sh
cd '../assembly-so101'
.venv/bin/python -m pytest -q tests/test_viewer_point_cloud.py tests/test_reset_viewer.py tests/test_reset_viewer_stock.py tests/test_reset_viewer_variations.py
npm --prefix web test
npm --prefix web run build
```

## Seating reward and noise overlay

The user defines reward as complete insertion within shown allowed pose noise. `../assembly-so101/assembly_policy/so101/wowrobo_reward.py` implements a shared binary geometric seating predicate. `../assembly-so101/configs/wowrobo_reward_regions.json` binds the five exact current-model target geometries to verified full-depth native endpoint references, records source hashes and guards the model signature. Upper/final use the incoming mating case feature; other tasks use their incoming printed part/pin feature. Targets follow the current receiver and its rotation, including randomized states.

Default positional error is at most0.5mm per target-frame axis, rotation-angle error at most1degree. Reward=1 requires both for the active task and every preceding insertion. The gold box shows the incoming mating geometry at its terminal pose; it is not the acceptance volume. The small green box is the exact ±position-noise acceptance volume for the mating feature center. Orientation is checked separately and its tolerance is shown numerically. The Component camera focuses the target for inspection. A Reward region toggle is independent of mesh/point visibility.

Position and orientation controls with Apply tolerance update this viewer's live shared predicate through `/api/reward-tolerance` and refresh the same selected state. These inspection settings are not persisted to running training. The shared predicate can be consumed by the future current-assembly trainer; no WowRobo trainer/checkpoint is currently connected. This geometric condition does not include dynamic settling, material-interference or grasp-retention guards.

Validation:17Python tests and13frontend tests passed. Boundary tests distinguish0.499/0.501mm and0.9/1.1degrees, verify receiver-relative movement, adjustable tolerances and rejection of invalid values. The coherent final full51 endpoint gives reward1 with all five predecessor checks passed. The current selected upper near-state remains reward0 at1.99mm error. The viewer remains on its latest cover-support-v2 bank. Browser verified independent mesh/point/reward layers and tolerance editing.


## Task-specific geometry coverage

User requested refinement after the area-proportional predecessor sampler missed the pin hole. The current builder uses measured reference seating transforms only to select actual source triangles once. Dedicated feature budgets are48pin/connector and64cover/upper/final.64points retain broad installed context; the remainder cover surfaces within12mm of the incoming mating geometry at its seat.32active points cover mating geometry (pin8tip+24shaft/shoulder). Finger budget remains64.

Pin feature48 splits16rim/24bore/8blindstop and covers all4rim quadrants. All448points stay in stable order on their own actual bodies; no empty-space target markers or runtime contact-only resampling enter the tensor. Current-model topology and mesh fingerprints reject stale reference configurations. The downloaded sampling_budget records each task's allocation; observation_signature identifies the new contract.

A32/48/64/96feature sweep,100saved-state check and source-provenance audit are in `outputs/point-cloud-refinement-20261107/`. Nearest pin-mouth sample improves4.415→1.735mm; native surface residual<8nm. No learning-performance superiority is claimed.
