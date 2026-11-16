# Assembly project instructions

**Goal:** Keep every assembly chat aligned with the current user intent and one canonical environment.

- Read `User Intent.md` and `MEMORY.md` before project work. Read relevant architecture/status documents before changing their implementation.
- After a meaningful new user requirement or clarification, update `User Intent.md` before finishing the turn. Rewrite the affected section, remove superseded plans and obsolete actions, and retain consequential decision history in `MEMORY.md`. Do not append a transcript.
- Follow the latest explicit user instruction. If it conflicts with an existing decision, flag the conflict and update the documents; ask only when the intended outcome remains unclear.
- Distinguish user goals, agreed decisions, open questions and verified implementation. Link detailed evidence instead of copying old reports into the intent file.
- The canonical assembly repository is `Alessandro-Maniscalco/robotics-assembly-3.0`. On the Linux host use `/home/alessandro/robotics_ml/projects/assembly_3_0`; on Alessandro's Mac use `/Users/ale/Documents/ChatGPT/robotics-assembly`. Use branch `main` and an explicit working directory when the app opens an older worktree. Other hosts use their corresponding clone.
- All assembly chats should edit the canonical main checkout. Inspect existing changes, coordinate overlapping files, preserve other chats' edits and stage only owned changes. Do not reset a checkout to obtain a clean tree.
- Keep the physical environment builder/configuration inside this AssembleNIST fork. Training and the Quest adapter consume the same builder; project physics/layout changes must enter it. Keep controller mapping/render transport in the teleop integration.
- Record environment identity for experiments; do not mutate running training jobs when source changes. Do not claim a change is shared remotely until its push/release is verified.
- Maintain the existing inherited project rules. Keep credentials, private contact details, original vendor CAD and generated runs out of source commits.
