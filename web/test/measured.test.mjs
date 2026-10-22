import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { test } from "node:test";

import { buildSystem, estimate } from "../js/estimator.js";
import { benchGenTps, groupCpuRuns } from "../js/measured.js";

const presets = JSON.parse(readFileSync(new URL("../data/presets.json", import.meta.url), "utf8"));

const run = (over) => ({
  model: "m", sim_model: "m", quant: "Q8_0", threads: 16, dimms: 2, mem_mts: 6000,
  ram_bandwidth_gbps: 96, n_prompt: null, n_gen: 128, pp_tps: null, tg_tps: 10, source: "a", ...over,
});

test("CPU runs group by test and memory, with medians and numbered sources", () => {
  const { groups, sources } = groupCpuRuns([
    run({ tg_tps: 9 }), run({ tg_tps: 11, source: "b" }), run({ tg_tps: 10 }),
    run({ mem_mts: 4800, ram_bandwidth_gbps: 76.8, source: "b" }),
    run({ n_gen: null, n_prompt: 512, tg_tps: null, pp_tps: 50, source: "c" }),
    run({ n_gen: null, n_prompt: 512, tg_tps: null, pp_tps: 70, source: "c" }),
  ]);
  assert.deepEqual(sources, ["a", "b", "c"]);
  assert.equal(groups.length, 3);
  assert.deepEqual(groups.map((g) => [g.n, g.tg_tps, g.pp_tps, g.sources]),
    [[3, 10, null, [1, 2]], [1, 10, null, [2]], [2, null, 60, [3]]]);
});

test("exported CPU runs can all be simulated", () => {
  assert.ok(presets.cpu_measurements.length > 100);
  for (const m of presets.cpu_measurements) {
    assert.ok(presets.cpus.some((c) => c.name === m.preset), m.id);
    assert.ok(presets.models.some((x) => x.name === m.sim_model), m.id);
    assert.ok(m.pp_tps != null || m.tg_tps != null, m.id);
    if (m.tg_tps != null) assert.ok(m.ram_bandwidth_gbps > 0 && m.n_gen > 0, m.id);
    if (m.pp_tps != null) assert.ok(m.n_prompt > 0, m.id);
  }
});

test("benchGenTps averages the decode time over the generated tokens", () => {
  const system = buildSystem({ kind: "pc", cpu: "ryzen9-9950x", ram_gb: 1024, ram_bandwidth_gbps: 96, gpus: [] }, presets);
  const est = estimate(system, presets.models.find((x) => x.name === "llama-3.1-8b"), "Q8_0", 128, 16,
                       presets.efficiency, presets.quant_bpw);
  const tps = benchGenTps(est, 128);
  assert.ok(tps < est.decode_tps_empty && tps > est.decodeTpsAt(128));
});
