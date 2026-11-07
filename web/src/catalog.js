export function filterEntries(entries, filters) {
  const query = (filters.search || "").toLowerCase().trim();
  return entries
    .filter(
      (entry) =>
        (!filters.bank || entry.bank === filters.bank) &&
        (!filters.task || entry.task === filters.task) &&
        (!filters.split || entry.split === filters.split) &&
        (!filters.kind || entry.review_kind === filters.kind) &&
        (filters.family === "" || String(entry.family) === filters.family) &&
        (!query ||
          `${entry.id} ${entry.seed} ${entry.task} ${entry.family_name}`
            .toLowerCase()
            .includes(query)),
    )
    .sort((a, b) => {
      const seedOrder = a.seed - b.seed;
      if (seedOrder) return seedOrder;
      if (a.review_kind === "nominal" && b.review_kind === "nominal") {
        const stageOrder =
          Number.parseInt(a.task, 10) - Number.parseInt(b.task, 10);
        if (stageOrder) return stageOrder;
      }
      return a.id.localeCompare(b.id);
    });
}

export function resetIdentity(entry) {
  const reference =
    entry.review_kind === "nominal" || entry.variation?.kind === "nominal";
  return {
    reference,
    suffix: reference ? "Reference" : `Seed ${entry.seed}`,
    splitLabel: reference ? "REF" : entry.split.toUpperCase(),
  };
}

export function selectionAfterFilter(entries, selected) {
  return entries.some((entry) => entry.id === selected)
    ? selected
    : (entries[0]?.id ?? null);
}
