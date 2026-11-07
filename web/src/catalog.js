export function filterEntries(entries, filters) {
  const query = (filters.search || "").toLowerCase().trim();
  return entries
    .filter(
      (entry) =>
        (!filters.bank || entry.bank === filters.bank) &&
        (!filters.task || entry.task === filters.task) &&
        (!filters.split || entry.split === filters.split) &&
        (filters.family === "" || String(entry.family) === filters.family) &&
        (!query ||
          `${entry.id} ${entry.seed} ${entry.task} ${entry.family_name}`
            .toLowerCase()
            .includes(query)),
    )
    .sort((a, b) => a.seed - b.seed || a.id.localeCompare(b.id));
}

export function selectionAfterFilter(entries, selected) {
  return entries.some((entry) => entry.id === selected)
    ? selected
    : (entries[0]?.id ?? null);
}
