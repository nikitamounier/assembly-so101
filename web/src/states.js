import { resetIdentity } from "./catalog.js";

export function stateGroup(task) {
  const number = Number.parseInt(task, 10);
  if (number === 0) return "Setup";
  if (number < 7) return "Base side cover · Left arm";
  if (number < 12) return "Shoulder connector · Right arm";
  if (number < 18) return "Pin & support · Left arm";
  if (number < 26) return "Upper arm · Right arm";
  return "Final module · Left arm";
}

export function stateLabel(task) {
  const label = task
    .replace(/^\d+_/, "")
    .replace(/^(cover|connector|pin|upper|final)_/, "")
    .replaceAll("_", " ")
    .replace(/^./, (letter) => letter.toUpperCase());
  return (
    {
      Loose: "Loose parts",
      Half: "Partly inserted",
      Near: "Almost seated",
      Entry: "Insertion start",
      "Far side": "Far end resting",
      "Left support": "Left supporting",
      "Right support": "Right supporting",
    }[label] || label
  );
}

export function stateCatalog(entries, bank = "") {
  const states = new Map();
  for (const entry of entries) {
    if (bank && entry.bank !== bank) continue;
    let state = states.get(entry.task);
    if (!state) {
      state = {
        task: entry.task,
        label: stateLabel(entry.task),
        group: stateGroup(entry.task),
        reference: null,
        variations: [],
      };
      states.set(entry.task, state);
    }
    if (resetIdentity(entry).reference) state.reference ||= entry;
    else if (
      entry.review_kind === "variation" ||
      entry.variation?.kind === "variation"
    )
      state.variations.push(entry);
  }
  return [...states.values()]
    .filter((state) => state.reference || state.variations.length)
    .sort((a, b) => Number.parseInt(a.task, 10) - Number.parseInt(b.task, 10));
}

export function admissionKind(entry) {
  const variation = entry.variation || {};
  if (variation.training_eligible) return "qualified";
  if (variation.candidate_rejected || variation.rejected) return "rejected";
  return "pending";
}

export function stateVariations(state, split = "", status = "") {
  return (state?.variations || [])
    .filter(
      (entry) =>
        (!split || entry.split === split) &&
        (!status || admissionKind(entry) === status),
    )
    .sort(
      (a, b) =>
        a.seed - b.seed ||
        a.split.localeCompare(b.split) ||
        a.id.localeCompare(b.id),
    );
}
