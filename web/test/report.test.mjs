// Run with: node --test web/test/
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { test } from "node:test";

import { buildReport, fmt, fmtTime, markdown } from "../js/report.js";

const read = (p) => JSON.parse(readFileSync(new URL(p, import.meta.url), "utf8"));
const presets = read("../data/presets.json");
const cases = read("./report.json");

// Same values up to float rounding; strings, booleans and nulls exactly.
function same(actual, expected, path = "report") {
  if (typeof expected === "number") {
    assert.equal(typeof actual, "number", `${path}: ${actual} is not a number`);
    const tol = 1e-9 * Math.max(1, Math.abs(expected));
    assert.ok(Math.abs(actual - expected) <= tol, `${path}: ${actual} != ${expected}`);
  } else if (expected && typeof expected === "object") {
    assert.equal(typeof actual, "object", `${path}: not an object`);
    assert.deepEqual(Object.keys(actual ?? {}).sort(), Object.keys(expected).sort(), `${path}: keys`);
    for (const k of Object.keys(expected)) same(actual[k], expected[k], `${path}.${k}`);
  } else {
    assert.equal(actual, expected, path);
  }
}

for (const [i, c] of cases.entries()) {
  test(`report case ${i}: ${c.input.map((x) => x.model).join(" + ")}`, () => {
    const rep = buildReport(presets, c.input, c.surface, c.share_url);
    // The golden file was written with the Python package's version.
    rep.generated_by.version = c.report.generated_by.version;
    same(rep, c.report);
    assert.equal(markdown(rep), c.markdown);
  });
}

test("fmt matches Python's round-half-even formatting", () => {
  assert.equal(fmt(2.5, 0), "2");
  assert.equal(fmt(3.5, 0), "4");
  assert.equal(fmt(0.125, 2), "0.12");
  assert.equal(fmt(12345.678, 1), "12,345.7");
});

test("fmtTime matches report.fmt_time", () => {
  assert.equal(fmtTime(0.0904), "90 ms");
  assert.equal(fmtTime(0.9996), "1.0 s");
  assert.equal(fmtTime(59.96), "1 min 00 s");
  assert.equal(fmtTime(3725), "1 h 02 min");
});
