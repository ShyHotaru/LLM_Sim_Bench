// Real benchmark results shown in the "check" table next to the simulator's numbers.

const median = (xs) => {
  const s = [...xs].sort((a, b) => a - b);
  const m = s.length >> 1;
  return s.length % 2 ? s[m] : (s[m - 1] + s[m]) / 2;
};

// CPU-only runs grouped by model, quantization, memory, threads and test size, so
// repeated runs (and the same test on other llama.cpp builds) read as one row with
// the median result. Sources are numbered in order of first use.
export function groupCpuRuns(rows) {
  const sources = [];
  const groups = new Map();
  for (const r of rows) {
    const key = [r.model, r.quant, r.dimms, r.mem_mts, r.threads, r.n_prompt, r.n_gen].join("|");
    if (!groups.has(key)) groups.set(key, { ...r, runs: [], sources: [] });
    const g = groups.get(key);
    g.runs.push(r);
    if (!sources.includes(r.source)) sources.push(r.source);
    const n = sources.indexOf(r.source) + 1;
    if (!g.sources.includes(n)) g.sources.push(n);
  }
  const values = (g, k) => g.runs.map((r) => r[k]).filter((v) => v != null);
  return {
    sources,
    groups: [...groups.values()].map((g) => {
      const pp = values(g, "pp_tps");
      const tg = values(g, "tg_tps");
      return {
        model: g.model, sim_model: g.sim_model, quant: g.quant, threads: g.threads,
        dimms: g.dimms, mem_mts: g.mem_mts, ram_bandwidth_gbps: g.ram_bandwidth_gbps,
        n_prompt: g.n_prompt, n_gen: g.n_gen, n: g.runs.length, sources: g.sources,
        pp_tps: pp.length ? median(pp) : null, tg_tps: tg.length ? median(tg) : null,
      };
    }),
  };
}

// Decode speed llama-bench reports for `nGen` tokens generated from an empty context,
// as the calibration computes it.
export function benchGenTps(est, nGen) {
  let t = 0;
  for (let i = 0; i < nGen; i++) t += 1 / est.decodeTpsAt(i);
  return nGen / t;
}
