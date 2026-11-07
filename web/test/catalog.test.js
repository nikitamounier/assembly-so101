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
