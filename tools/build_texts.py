"""Download public-domain book excerpts listed in web/data/texts/manifest.json.

For every manifest entry with a `source.fetch` URL this writes the excerpt to
web/data/texts/<file> and its provenance to web/data/texts/credits.json.
Authored texts are written by hand and are left alone.

- Aozora Bunko (青空文庫) allows free reuse of public-domain files and asks that
  the bibliographic notes (底本, 入力, 校正, file history and the 「青空文庫作成
  ファイル」 note) be kept; they go into credits.json.
- Project Gutenberg's site is for human visitors only, so its texts are fetched
  from the GITenberg mirror on GitHub. The Project Gutenberg header, license and
  any mention of the name are removed from the excerpt, as its license requires
  for modified copies; credits.json links to the ebook page as an acknowledgement.

Usage: python tools/build_texts.py [--only ID ...]
"""

from __future__ import annotations

import argparse
import datetime
import html
import json
import re
import sys
import time
import urllib.request
from pathlib import Path

TEXTS = Path(__file__).resolve().parents[1] / "web" / "data" / "texts"
USER_AGENT = "llm-sim-bench-texts/1.0 (one-off download of a few public-domain files)"


def fetch(url: str) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=60) as resp:
        return resp.read()


def _decode(raw: bytes) -> str:
    head = raw[:1000].decode("ascii", "ignore").lower()
    return raw.decode("shift_jis" if "shift_jis" in head else "utf-8")


def aozora(raw: bytes, limit: int) -> tuple[str, str]:
    """Return (excerpt, bibliographic notes) from an Aozora Bunko XHTML file."""
    s = _decode(raw)
    main = re.search(r'<div class="main_text">(.*?)<div class="bibliographical_information">',
                     s, re.S).group(1)
    bib = re.search(r'<div class="bibliographical_information">(.*?)</div>', s, re.S).group(1)
    main = re.sub(r"<rp>.*?</rp>|<rt>.*?</rt>", "", main, flags=re.S)  # ruby readings
    main = re.sub(r'<span class="notes">.*?</span>', "", main, flags=re.S)  # ［＃...］ notes
    main = re.sub(r"<img[^>]*>", "", main)  # gaiji images: cut the excerpt before them
    main = re.sub(r"<br\s*/?>", "\n", main)
    text = html.unescape(re.sub(r"<[^>]+>", "", main))
    text = "\n".join(line.strip() for line in text.splitlines())
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    text = text.split("")[0]
    notes = html.unescape(re.sub(r"<[^>]+>", "", bib))
    lines = [line.strip() for line in notes.splitlines() if line.strip()]
    return _cut(text, limit, "。」\n"), " / ".join(lines)


def gutenberg(raw: bytes, limit: int, start: str) -> str:
    text = raw.decode("utf-8").replace("\r\n", "\n")
    body = text[text.index(start):]
    end = re.search(r"\*\*\* ?END OF|End of (the )?Project Gutenberg", body, re.I)
    body = body[:end.start()] if end else body
    paragraphs = [" ".join(p.split()) for p in re.split(r"\n\s*\n", body) if p.strip()]
    out = _cut("\n\n".join(paragraphs), limit, "\n")
    if re.search(r"gutenberg", out, re.I):
        raise ValueError("excerpt still mentions Project Gutenberg")
    return out


def _cut(text: str, limit: int, enders: str) -> str:
    """Cut at the last sentence or paragraph end before `limit` characters."""
    if len(text) <= limit:
        return text
    head = text[:limit]
    pos = max(head.rfind(c) for c in enders)
    return head[:pos + 1].strip() if pos > 0 else head


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--only", nargs="*", help="manifest ids to rebuild")
    args = ap.parse_args(argv)

    manifest = json.loads((TEXTS / "manifest.json").read_text(encoding="utf-8"))
    credits_path = TEXTS / "credits.json"
    credits = json.loads(credits_path.read_text(encoding="utf-8")) if credits_path.exists() else {}
    today = datetime.date.today().isoformat()
    for entry in manifest["texts"]:
        src = entry.get("source", {})
        if "fetch" not in src or (args.only and entry["id"] not in args.only):
            continue
        print(f"{entry['id']}: {src['fetch']}", file=sys.stderr)
        raw = fetch(src["fetch"])
        if src["format"] == "aozora-xhtml":
            text, notes = aozora(raw, entry["excerpt_chars"])
        else:
            text, notes = gutenberg(raw, entry["excerpt_chars"], src["start"]), ""
        (TEXTS / entry["file"]).write_text(text + "\n", encoding="utf-8")
        credits[entry["id"]] = {"retrieved": today, "notes": notes, "chars": len(text)}
        time.sleep(2)  # be gentle with the source sites
    credits_path.write_text(json.dumps(credits, ensure_ascii=False, indent=1) + "\n",
                            encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
