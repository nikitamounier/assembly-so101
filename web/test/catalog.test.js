import test from "node:test";
import assert from "node:assert/strict";
import { filterEntries, selectionAfterFilter } from "../src/catalog.js";

const entries = [
  {
    id: "a",
    bank: "one",
    task: "motor",
    split: "train",
    family: 0,
    family_name: "Loose",
    seed: 42,
  },
  {
    id: "b",
    bank: "one",
    task: "horn",
    split: "eval",
    family: 2,
    family_name: "Lifted",
    seed: 7,
  },
  {
    id: "c",
    bank: "two",
    task: "motor",
    split: "eval",
    family: 0,
    family_name: "Loose",
    seed: 12,
  },
];
const filters = { bank: "", task: "", split: "", family: "", search: "" };

test("combines bank/task/split/family and seed search without losing family zero", () => {
  assert.deepEqual(
    filterEntries(entries, {
      ...filters,
      task: "motor",
      split: "eval",
      family: "0",
      search: "12",
    }).map((e) => e.id),
    ["c"],
  );
  assert.deepEqual(
    filterEntries(entries, { ...filters, bank: "one", search: "LIFTED" }).map(
      (e) => e.id,
    ),
    ["b"],
  );
  assert.equal(filterEntries(entries, { ...filters, family: "0" }).length, 2);
});

test("keeps an existing selection and clears it when filters have no results", () => {
  assert.equal(selectionAfterFilter(entries, "b"), "b");
  assert.equal(selectionAfterFilter([], "b"), null);
  assert.equal(selectionAfterFilter(entries, "missing"), "a");
});

test("nominal seed ties follow original stage numbers while generated seeds retain order", () => {
  const rows = [
    { id: "a", task: "13_pin_aligned", seed: -1, review_kind: "nominal" },
    { id: "b", task: "04_cover_entry", seed: -1, review_kind: "nominal" },
    { id: "c", task: "00_loose", seed: -1, review_kind: "nominal" },
    { id: "d", task: "09_connector_entry", seed: -1, review_kind: "nominal" },
    { id: "e", task: "00_loose", seed: 4, review_kind: "variation" },
    { id: "f", task: "31_final_near", seed: 2, review_kind: "variation" },
  ];
  assert.deepEqual(
    filterEntries(rows, { family: "" }).map((e) => e.id),
    ["c", "b", "d", "a", "f", "e"],
  );
});

test("nominal references do not expose synthetic seed or TRAIN label", async () => {
  const { resetIdentity } = await import("../src/catalog.js");
  assert.deepEqual(
    resetIdentity({ seed: -1, split: "train", review_kind: "nominal" }),
    { reference: true, suffix: "Reference", splitLabel: "REF" },
  );
  assert.deepEqual(
    resetIdentity({ seed: 7, split: "eval", review_kind: "variation" }),
    { reference: false, suffix: "Seed 7", splitLabel: "EVAL" },
  );
});
