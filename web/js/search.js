// Part search across GPUs, CPUs and unified-memory chips.

// Lowercase letters and digits only, so "RTX 4090", "rtx4090" and "RTX-4090" all match.
export const normalize = (s) => String(s).toLowerCase().replace(/[^a-z0-9]+/g, "");

// Newest first; within a launch year keep the reverse of preset order (presets list the
// smaller parts of a generation first, so the flagship comes first).
export function newestFirst(items) {
  return items.map((x, i) => [x, i])
    .sort(([a, i], [b, j]) => (b.year || 0) - (a.year || 0) || j - i)
    .map(([x]) => x);
}

function entries(presets) {
  return [
    ...presets.gpus.map((item) => ({ kind: "gpu", item })),
    ...presets.cpus.map((item) => ({ kind: "cpu", item })),
    ...presets.unified.map((item) => ({ kind: "soc", item })),
  ];
}

// Every word of the query must appear in the part's label, preset name or vendor.
// Label prefix matches rank first, then newer parts.
export function searchParts(query, presets, limit = 8) {
  const words = String(query).split(/\s+/).map(normalize).filter(Boolean);
  if (!words.length) return [];
  const hits = [];
  for (const e of entries(presets)) {
    const label = normalize(e.item.label || e.item.name);
    const hay = `${label} ${normalize(e.item.name)} ${normalize(e.item.vendor || "")}`;
    if (!words.every((w) => hay.includes(w))) continue;
    const score = (label.startsWith(words[0]) ? 2 : 0) + (label.includes(words.join("")) ? 1 : 0);
    hits.push({ ...e, score });
  }
  return hits
    .sort((a, b) => b.score - a.score || (b.item.year || 0) - (a.item.year || 0))
    .slice(0, limit);
}
