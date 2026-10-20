import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { test } from "node:test";

import { Schedule } from "../js/mock.js";
import { charsPerToken, chunkText } from "../js/tokenize.js";

const read = (p) => JSON.parse(readFileSync(new URL(p, import.meta.url), "utf8"));
const presets = read("../data/presets.json");
const manifest = read("../data/texts/manifest.json");
const credits = read("../data/texts/credits.json");
const close = (a, b, what) => assert.ok(Math.abs(a - b) < 1e-9, `${what}: ${a} != ${b}`);

test("chunkText keeps the text and matches the ratio", () => {
  const en = "Alice was beginning to get very tired of sitting by her sister on the bank.";
  const pieces = chunkText(en, 4);
  assert.equal(pieces.join(""), en);
  assert.equal(pieces.length, Math.round(en.length / 4));
  // Word-initial tokens carry the leading space, as BPE tokens do.
  const words = en.split(" ").length;
  assert.ok(pieces.filter((p) => p.startsWith(" ")).length >= words - 2, pieces.join("|"));

  const ko = "가을 하늘 공활한데 높고 구름 없이";
  const bytey = chunkText(ko, 0.7);
  assert.equal(bytey.join(""), ko);
  assert.equal(bytey.length, Math.round(Array.from(ko).length / 0.7));
  assert.ok(bytey.includes(""), "below one char per token some tokens show nothing");
  assert.deepEqual(chunkText("", 3), []);
  assert.equal(chunkText("😀😀😀", 1).join("|"), "😀|😀|😀");
});

test("charsPerToken falls back to the median for unknown tokenizers", () => {
  const ratios = { chars_per_token: { t: { a: 1, b: 2, c: 4 } } };
  assert.deepEqual(charsPerToken(ratios, "t", "b"), { value: 2, measured: true });
  assert.deepEqual(charsPerToken(ratios, "t", ""), { value: 2, measured: false });
});

test("schedule paces a single text by prefill and decode speed", () => {
  const s = new Schedule({ turns: [{ user: null, assistant: Array(11).fill("x") }], promptTokens: 100,
                           maxTokens: 50, prefillTps: 1000, decodeTps: () => 100, jitter: 0 });
  close(s.turns[0].ttft, 0.1, "ttft");
  close(s.totalTime, 0.2, "total");
  assert.equal(s.dueBy(0.05), 0);
  assert.equal(s.dueBy(0.1), 1);
  assert.equal(s.dueBy(0.155), 6);
  assert.equal(s.events.at(-1).fill, 111);
});

test("schedule caps each turn and prefills only new user tokens", () => {
  const s = new Schedule({
    turns: [{ user: ["hi", "!"], assistant: Array(5).fill("a") },
            { user: ["more"], assistant: Array(5).fill("b") }],
    promptTokens: 10, maxTokens: 3, prefillTps: 10, decodeTps: () => 10, pause: 1, jitter: 0,
  });
  assert.deepEqual(s.turns.map((x) => x.prefillTokens), [12, 1]);
  assert.deepEqual(s.turns.map((x) => x.tokens), [3, 3]);
  close(s.turns[0].ttft, 1.2, "turn 1 ttft");
  close(s.turns[1].start, 1.2 + 0.2 + 1, "turn 2 start");
  close(s.turns[1].ttft, 0.1, "turn 2 ttft");
  assert.deepEqual(s.events.map((e) => e.role[0]).join(""), "uaaauaaa");
  assert.equal(s.events.at(-1).fill, 10 + 2 + 3 + 1 + 3);
});

test("text presets follow the public-domain policy and have credits", () => {
  const ids = new Set();
  for (const x of manifest.texts) {
    assert.ok(!ids.has(x.id), `duplicate ${x.id}`);
    ids.add(x.id);
    assert.ok(["ko", "ja", "en"].includes(x.lang), x.id);
    const raw = readFileSync(new URL(`../data/texts/${x.file}`, import.meta.url), "utf8");
    assert.ok(raw.trim().length > 500, `${x.id} is too short`);
    if (x.kind === "chat") assert.ok(JSON.parse(raw).turns.length >= 2, x.id);
    if (x.origin === "authored") continue;
    assert.equal(x.origin, "public-domain", x.id);
    assert.ok(x.author_died <= 1925 && x.first_published <= 1930, `${x.id} fails the policy`);
    assert.ok(credits[x.id]?.retrieved, `${x.id} has no credits`);
    assert.ok(!/gutenberg/i.test(raw), `${x.id} mentions Project Gutenberg`);
  }
});

test("every text has a ratio for every model tokenizer", () => {
  const families = new Set(presets.models.map((m) => m.tokenizer));
  for (const x of manifest.texts) {
    for (const f of families) {
      assert.ok(presets.tokenizers.chars_per_token[x.id]?.[f] > 0, `${x.id} / ${f}`);
    }
  }
});
