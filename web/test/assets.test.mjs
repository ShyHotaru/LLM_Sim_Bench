import assert from "node:assert/strict";
import { existsSync, readFileSync } from "node:fs";
import { test } from "node:test";

const web = new URL("../", import.meta.url);
const read = (p) => readFileSync(new URL(p, web), "utf8");
const local = (html) => [...html.matchAll(/(?:href|src)="([^"#:]+)"/g)].map((m) => m[1]);
// Files the stylesheet loads (the footer logo mask), relative to web/.
const fromCss = () => [...read("css/style.css").matchAll(/url\(\.\.\/([^)]+)\)/g)].map((m) => m[1]);

test("every file the page and its stylesheet link exists", () => {
  assert.ok(fromCss().includes("brand/shy-hotaru.svg"));
  for (const path of [...local(read("index.html")), ...fromCss()]) assert.ok(existsSync(new URL(path, web)), path);
});

test("every font file exists and ships with its license", () => {
  const sheets = local(read("index.html")).filter((p) => p.startsWith("fonts/"));
  assert.equal(sheets.length, 3);
  for (const sheet of sheets) {
    const dir = new URL(sheet.replace(/[^/]+$/, ""), web);
    const files = [...read(sheet).matchAll(/url\(\.\/([^)]+)\)/g)].map((m) => m[1]);
    assert.ok(files.length > 0, sheet);
    for (const f of files) assert.ok(existsSync(new URL(f, dir)), `${sheet}: ${f}`);
    assert.match(readFileSync(new URL("OFL.txt", dir), "utf8"), /SIL Open Font License, Version 1\.1/);
  }
});

test("the Pages workflow copies everything the page links", () => {
  const workflow = readFileSync(new URL("../.github/workflows/pages.yml", web), "utf8");
  const copied = workflow.match(/cp -r (.+) _site\//)[1].split(/\s+/).map((p) => p.replace(/^web\//, ""));
  for (const path of [...local(read("index.html")), ...fromCss()]) {
    assert.ok(copied.includes(path.split("/")[0]), `${path} is not copied to the site`);
  }
});
