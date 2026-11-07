import test from "node:test";
import assert from "node:assert/strict";
import {
  stateCatalog,
  stateVariations,
  stateGroup,
  stateLabel,
} from "../src/states.js";

const row = (task, id, kind, extra = {}) => ({
  task,
  id,
  bank: "a",
  review_kind: kind,
  seed: kind === "nominal" ? -1 : 2,
  split: "train",
  ...extra,
});
const entries = [
  row("08_connector_aligned", "v8", "variation"),
  row("00_loose", "r0", "nominal"),
  row("08_connector_aligned", "r8", "nominal"),
  row("08_connector_aligned", "e8", "variation", {
    split: "eval",
    variation: { candidate_rejected: true },
  }),
  row("08_connector_aligned", "q8", "variation", {
    seed: 1,
    variation: { training_eligible: true },
  }),
  row("31_final_near", "other", "nominal", { bank: "b" }),
];

test("states stay in assembly order with reference and variations together", () => {
  const states = stateCatalog(entries, "a");
  assert.deepEqual(
    states.map((state) => state.task),
    ["00_loose", "08_connector_aligned"],
  );
  assert.equal(states[1].reference.id, "r8");
  assert.equal(states[1].variations.length, 3);
  assert.equal(stateLabel(states[1].task), "Aligned");
  assert.equal(stateGroup(states[1].task), "Shoulder connector · Right arm");
});

test("variation filters cannot include another state or remove its reference", () => {
  const state = stateCatalog(entries, "a")[1];
  assert.deepEqual(
    stateVariations(state, "eval", "rejected").map((entry) => entry.id),
    ["e8"],
  );
  assert.deepEqual(
    stateVariations(state, "train", "qualified").map((entry) => entry.id),
    ["q8"],
  );
  assert.equal(stateVariations(state, "eval", "qualified").length, 0);
  assert.equal(state.reference.id, "r8");
  assert.equal(state.variations.length, 3);
});

test("variation order remains stable across filters and states without variants work", () => {
  const states = stateCatalog(entries, "a");
  assert.deepEqual(
    stateVariations(states[1]).map((entry) => entry.id),
    ["q8", "e8", "v8"],
  );
  assert.deepEqual(stateVariations(states[0]), []);
  assert.equal(states[0].reference.id, "r0");
});
