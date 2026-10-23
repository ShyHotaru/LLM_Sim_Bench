// Run with: node --test web/test/
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { test } from "node:test";

import { buildSystem, estimate, ramBandwidthGbps } from "../js/estimator.js";
import { decodeState, encodeState } from "../js/state.js";

const read = (p) => JSON.parse(readFileSync(new URL(p, import.meta.url), "utf8"));
const presets = read("../data/presets.json");
const golden = read("./golden.json");

function close(actual, expected, what) {
  const tol = 1e-9 * Math.max(1, Math.abs(expected));
  assert.ok(Math.abs(actual - expected) <= tol, `${what}: ${actual} != ${expected}`);
}

for (const [i, { input, expected }] of golden.entries()) {
  test(`golden case ${i}: ${input.model} ${input.quant} ctx ${input.context}`, () => {
    const model = presets.models.find((m) => m.name === input.model);
    const est = estimate(buildSystem(input.system, presets), model, input.quant,
                         input.context, input.kv_bits, presets.efficiency, presets.quant_bpw,
                         input.parallel, presets.quant_shared_bpw);
    close(est.weights_gb, expected.weights_gb, "weights_gb");
    close(est.kv_gb, expected.kv_gb, "kv_gb");
    assert.equal(est.fits, expected.fits);
    assert.equal(est.fit_context, expected.fit_context);
    assert.deepEqual(est.notes, expected.notes);
    assert.equal(est.placements.length, expected.placements.length);
    est.placements.forEach((p, j) => {
      const e = expected.placements[j];
      assert.equal(p.device, e.device);
      assert.equal(p.layers, e.layers);
      assert.equal(p.expert_layers, e.expert_layers);
      assert.equal(p.is_gpu, e.is_gpu);
      for (const k of ["weights_gb", "kv_gb", "bandwidth_gbps", "tflops", "overhead_s", "batch_s"]) {
        close(p[k], e[k], `placement ${j} ${k}`);
      }
    });
    close(est.prefill_tps, expected.prefill_tps, "prefill_tps");
    close(est.decode_tps_empty, expected.decode_tps_empty, "decode_tps_empty");
    close(est.decode_tps_full, expected.decode_tps_full, "decode_tps_full");
    for (const [fill, tps] of expected.decode_tps_at) close(est.decodeTpsAt(fill), tps, `at ${fill}`);
    for (const [n, depth, t] of expected.prefill_time) close(est.prefillTime(n, depth), t, `prefill ${n} after ${depth}`);
    for (const [fills, t] of expected.step_time) close(est.stepTime(fills), t, `step ${fills}`);
    for (const [n, fill, tps] of expected.parallel_tps) close(est.parallelTps(n, fill), tps, `${n} at ${fill}`);
  });
}

test("RAM bandwidth", () => {
  assert.equal(ramBandwidthGbps("ddr4-3200", 2), 51.2);
  assert.equal(ramBandwidthGbps("DDR5-6000", 2), 96);
  assert.throws(() => ramBandwidthGbps("fast-ram"));
});

test("URL state round-trips", () => {
  const state = { configs: [{ platform: "pc", gpus: ["rtx3090-24gb", "rtx3090-24gb"] }],
                  prompt: 2000 };
  assert.deepEqual(decodeState(encodeState(state)), state);
  assert.equal(decodeState("garbage%%"), null);
});
