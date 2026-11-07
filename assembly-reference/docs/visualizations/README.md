# SO101 input visualization

**Goal:** Open the selected256 combined mating-point layout without a simulator or Codex installation.

Download [so101-mating-budget.html](so101-mating-budget.html) and open it in a browser. It contains the data, drawing code and standalone state bridge. GitHub's file viewer displays HTML source rather than running it. This commit does not enable GitHub Pages or create a public website.

The default is256 combined mating points. For screw, assembly128 + screwdriver128 =256; other tasks allocate256 to the complete predecessor assembly. The optional128 comparison is assembly64 + driver64 for screw. Active-part128 and distal-finger64 stay fixed. Use the task/view selectors, orbit/zoom and point inspector.

The figure preserves the reviewed prepared-stage snapshot from the Mac's v8 scene, including its then-current simplified tool. It is not a live viewer or a policy rollout. Newer tool/physics changes on main remain authoritative and are not overwritten by this snapshot. Sampled coordinates are preserved with display quantization; reference CAD is simplified. Original vendor CAD, native binaries, training runs, credentials and raw recordings are not included.

The current sampler and checkpoint contract are documented in [SO101 inputs](../SO101_MATING_INPUTS.md). Both presets must use newly trained weights matching their observation signatures.
