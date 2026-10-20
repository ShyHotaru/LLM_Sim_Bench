import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { test } from "node:test";

import { LANGS, format, pickLang, setMessages, t } from "../js/i18n.js";

const read = (p) => readFileSync(new URL(p, import.meta.url), "utf8");
const locales = Object.fromEntries(Object.keys(LANGS).map((c) => [c, JSON.parse(read(`../locales/${c}.json`))]));
const placeholders = (s) => [...s.matchAll(/\{(\w+)\}/g)].map((m) => m[1]).sort();

test("every locale has the same keys and placeholders as English", () => {
  const en = locales.en;
  for (const [code, msgs] of Object.entries(locales)) {
    assert.deepEqual(Object.keys(msgs).sort(), Object.keys(en).sort(), code);
    for (const k of Object.keys(en)) {
      assert.ok(msgs[k].trim(), `${code} ${k} is empty`);
      assert.deepEqual(placeholders(msgs[k]), placeholders(en[k]), `${code} ${k}`);
    }
  }
});

test("every key used by the app exists", () => {
  const used = new Set();
  for (const m of read("../js/app.js").matchAll(/\bt\("([\w.]+)"/g)) used.add(m[1]);
  for (const m of read("../index.html").matchAll(/data-i18n="([\w.]+)"/g)) used.add(m[1]);
  for (const m of read("../index.html").matchAll(/data-i18n-attr="[\w-]+:([\w.]+)"/g)) used.add(m[1]);
  assert.ok(used.size > 50);
  for (const k of used) assert.ok(k in locales.en, `missing key ${k}`);
});

test("language choice: URL, then saved, then browser, then English", () => {
  assert.equal(pickLang("?lang=ja", "ko", ["en-US"]), "ja");
  assert.equal(pickLang("", "ko", ["ja-JP"]), "ko");
  assert.equal(pickLang("", null, ["fr-FR", "ja-JP"]), "ja");
  assert.equal(pickLang("?lang=xx", null, ["de"]), "en");
});

test("t() formats and falls back to English", () => {
  setMessages("ko", { a: "A {x}", b: "B" }, { a: "가 {x}" });
  assert.equal(t("a", { x: 1 }), "가 1");
  assert.equal(t("b"), "B");
  assert.equal(t("missing"), "missing");
  assert.equal(format("{a}-{b}", { a: 1 }), "1-{b}");
});
