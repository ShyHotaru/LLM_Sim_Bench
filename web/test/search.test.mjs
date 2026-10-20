import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { test } from "node:test";

import { newestFirst, normalize, searchParts } from "../js/search.js";

const presets = JSON.parse(readFileSync(new URL("../data/presets.json", import.meta.url), "utf8"));
const names = (q) => searchParts(q, presets).map((h) => h.item.name);

test("normalize ignores case, spaces and punctuation", () => {
  assert.equal(normalize("RTX 4070 Ti-SUPER"), "rtx4070tisuper");
});

test("search finds parts of every kind", () => {
  assert.equal(names("4090")[0], "rtx4090-24gb");
  assert.ok(names("rtx4090").includes("rtx4090-24gb"));
  assert.ok(names("m4 max").every((n) => n.startsWith("apple-m4-max")));
  assert.equal(searchParts("dgx", presets)[0].kind, "soc");
  assert.ok(names("epyc 9654").includes("epyc-9654"));
  assert.ok(names("7900 xtx").includes("rx7900xtx-24gb"));
  assert.deepEqual(names(""), []);
  assert.deepEqual(names("voodoo2"), []);
});

test("newer parts come first", () => {
  const list = newestFirst([{ name: "old", year: 2020 }, { name: "a", year: 2025 },
                            { name: "b", year: 2025 }]);
  assert.deepEqual(list.map((x) => x.name), ["b", "a", "old"]);
});
