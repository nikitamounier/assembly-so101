# Simulation speed and engine comparison

**Goal:** Make this section’s user intent, implementation and unfinished work easy to review.

Follow the requested order: finish environment + all SGS states, push the qualified result, then investigate speed improvements and AGX/Isaac feasibility. Keep the working canonical baseline while comparing alternatives.

Compare equivalent geometry, starting states, insertion/drop/thread behavior, reward conditions and action/observation contracts. Measure simulation/control transitions per second, contact-rich throughput, memory, parallel scaling and learner throughput. High GPU utilization alone is not the goal.

**Current state:** MuJoCo native CPU physics plus a CUDA learner is implemented. Current v14 training throughput has not been independently benchmarked. Historical 33-transitions/s results predate the contact repairs and are not current performance claims. The contact/geometry qualification itself can take minutes per candidate.

AGX and Isaac are not qualified alternatives for this task. Check current APIs, hardware/storage requirements and licensing, then run the same correctness cases where feasible. A simulator that is faster because it omits the concave mating or thread contacts is not an equivalent result.

**For your review:** Preserve fidelity before speed. No backend switch, paid license or cloud spend has been silently approved or performed.

## Nikita’s native profiling follow-up — November 7

Profile the whole narrow-phase detection path, explicitly including custom contact supplements. The qualified v13 diagnostic replay and self-contained native flame graph are in [outputs/narrowphase-20261107](../../../outputs/narrowphase-20261107/README.md). Measured coarse stages plus separately traced native functions identify the base/motor supplement traversal as 75.76% of this replay’s narrow-phase time; all 241.7 million surface samples fail its target AABB check. Original-library state/contact replay parity passes. Fine function widths are instrumentation-affected estimates; the outer MuJoCo residual remains unexpanded. This does not benchmark local v14 or implement a physics optimization.

The user’s north star is **10–20× faster complete PPO iterations**, at fixed valid-transition count and physical acceptance. They explicitly authorize screwdriver removal and select `assembly-so101` for it; the screw and threaded mating remain. That local source change is implemented with new reset/reference and observation identities. The old v13 training environment is not mutated.

The next proposed optimization is a conservative hierarchy over the exact existing supplement samples, with original-order survivor evaluation. A new offline check on 320 no-driver v14/MuJoCo4 loose resets reduces the base/motor work from 76,287 point visits to about 13.43 box visits plus 0.166 leaf samples per query; a single root bound is insufficient in one direction. This is algorithmic feasibility, not a native timing result or positive-contact qualification. A follow-up on five geometric completion seeds finds all 62,599 motor samples inside the base box: plain box culling cannot accelerate that direction and adds traversal overhead. The contact-state design therefore also needs conservative target-grid distance bounds, with exact survivor evaluation. No such implementation has been qualified. Preserve material exclusion, support, friction, insertion, jamming and thread behavior.

Combining the separate v13 measurements gives illustrative whole-iteration scenarios: tool removal plus a 100× cheaper supplement ≈37 s / 8.1×; an additional 1.3× improvement in worker critical path ≈29 s / 10.3×; halving the remaining worker work too ≈16 s / 18.4×. Assumed phase representativeness, acceleration factors and balancing gains are unverified. A fresh no-driver v14/MuJoCo4 capture and contact-positive parity panel must precede performance claims. See `../../../../acceleration-20261107/assessment.txt` for the budget, code-level plan and limits. No supplement implementation, solver, timestep or live deployment change has been made.
