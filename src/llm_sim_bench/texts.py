"""Output texts and tokenizer ratios for the mock, the Python side of web/js/tokenize.js.

The texts and data/tokenizer_ratios.json are copied into the package (`data/`)
by `llmsimbench export-web`, so an installed package can read them. Splitting
must match the JavaScript chunkText exactly; export-web writes golden vectors
to web/test/chunks.json and the web tests check them.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from functools import cache
from importlib import resources
from pathlib import Path

DEFAULT_TEXT = "en-answer"

# Characters matched by JavaScript's /\s/; str.isspace() differs (e.g. U+FEFF, U+001C).
_JS_SPACE = frozenset("\t\n\v\f\r \u00a0\u1680\u2000\u2001\u2002\u2003\u2004\u2005\u2006"
                      "\u2007\u2008\u2009\u200a\u2028\u2029\u202f\u205f\u3000\ufeff")

# Main blocks of the Unicode scripts guessLang counts with \p{Script=...}.
_HANGUL = ((0x1100, 0x11FF), (0x302E, 0x302F), (0x3131, 0x318E), (0x3200, 0x321E),
           (0x3260, 0x327E), (0xA960, 0xA97C), (0xAC00, 0xD7A3), (0xD7B0, 0xD7C6),
           (0xD7CB, 0xD7FB), (0xFFA0, 0xFFBE), (0xFFC2, 0xFFC7), (0xFFCA, 0xFFCF),
           (0xFFD2, 0xFFD7), (0xFFDA, 0xFFDC))
_KANA_HAN = ((0x2E80, 0x2E99), (0x2E9B, 0x2EF3), (0x2F00, 0x2FD5), (0x3005, 0x3005),
             (0x3007, 0x3007), (0x3021, 0x3029), (0x3038, 0x303B), (0x3041, 0x3096),
             (0x309D, 0x309F), (0x30A1, 0x30FA), (0x30FD, 0x30FF), (0x31F0, 0x31FF),
             (0x32D0, 0x32FE), (0x3300, 0x3357), (0x3400, 0x4DBF), (0x4E00, 0x9FFF),
             (0xF900, 0xFA6D), (0xFA70, 0xFAD9), (0xFF66, 0xFF6F), (0xFF71, 0xFF9D),
             (0x1B000, 0x1B122), (0x20000, 0x323AF))
_LATIN = ((0x41, 0x5A), (0x61, 0x7A), (0xAA, 0xAA), (0xBA, 0xBA), (0xC0, 0xD6), (0xD8, 0xF6),
          (0xF8, 0x2B8), (0x2E0, 0x2E4), (0x1D00, 0x1D25), (0x1E00, 0x1EFF), (0x2C60, 0x2C7F),
          (0xA722, 0xA787), (0xA78B, 0xA7FF), (0xFB00, 0xFB06), (0xFF21, 0xFF3A),
          (0xFF41, 0xFF5A))


def _data():
    return resources.files(__package__) / "data"


@cache
def manifest() -> dict:
    return json.loads((_data() / "texts" / "manifest.json").read_text(encoding="utf-8"))


@cache
def credits() -> dict:
    return json.loads((_data() / "texts" / "credits.json").read_text(encoding="utf-8"))


@cache
def ratios() -> dict:
    return json.loads((_data() / "tokenizer_ratios.json").read_text(encoding="utf-8"))


def entry(text_id: str) -> dict:
    for x in manifest()["texts"]:
        if x["id"] == text_id:
            return x
    ids = ", ".join(x["id"] for x in manifest()["texts"])
    raise ValueError(f"unknown text '{text_id}'. Available: {ids}")


def load(e: dict) -> str | list[dict]:
    """The text of an answer or book, or the turns ({user, assistant}) of a chat."""
    raw = (_data() / "texts" / e["file"]).read_text(encoding="utf-8")
    return json.loads(raw)["turns"] if e["kind"] == "chat" else raw


def js_round(x: float) -> int:
    """JavaScript Math.round: halves round up, unlike Python's round()."""
    f = math.floor(x)
    return f + 1 if x - f >= 0.5 else f


def chunk_text(text: str, chars_per_token: float) -> list[str]:
    """Pieces whose joined text equals `text`; their count matches the measured ratio.

    Below one character per token (byte-level fallback for CJK), some pieces are
    empty, so a character appears only after several tokens, as it does in reality.
    """
    n_chars = len(text)
    if not n_chars:
        return []
    n = max(1, js_round(n_chars / chars_per_token))
    pieces = []
    prev = 0
    for i in range(1, n + 1):
        pos = n_chars if i == n else js_round(i * n_chars / n)
        # Latin-script tokens usually start with a space: snap to one nearby.
        if chars_per_token >= 2 and i < n:
            pos = _snap_to_space(text, pos, prev, chars_per_token / 2)
        pos = max(pos, prev)
        pieces.append(text[prev:pos])
        prev = pos
    return pieces


def _snap_to_space(text: str, pos: int, lo: int, radius: float) -> int:
    d = 0
    while d <= radius:
        for p in (pos - d, pos + d):
            if lo < p < len(text) and text[p] in _JS_SPACE:
                return p
        d += 1
    return pos


def count_tokens(text: str, chars_per_token: float) -> int:
    """Token count of `text` at a ratio, the number of pieces chunk_text makes."""
    return max(1, js_round(len(text) / chars_per_token)) if text else 0


def chars_per_token(text_id: str, family: str, table: dict | None = None) -> tuple[float, bool]:
    """(characters per token, measured) for a text and tokenizer family.

    Unknown families fall back to the median of the measured ones for that text.
    """
    row = (table or ratios())["chars_per_token"].get(text_id, {})
    if family in row:
        return row[family], True
    values = sorted(row.values())
    if not values:
        return 4, False
    mid = len(values) // 2
    median = values[mid] if len(values) % 2 else (values[mid - 1] + values[mid]) / 2
    return median, False


def _in(ranges, cp: int) -> bool:
    return any(a <= cp <= b for a, b in ranges)


def guess_lang(text: str) -> str | None:
    """Language of free text from its letters, to borrow the ratio of a sample text.

    Hangul -> ko, kana or kanji -> ja, else en; a CJK character counts as two
    Latin letters. None when there are no letters.
    """
    scores = {"ko": 0, "ja": 0, "en": 0}
    for ch in text:
        cp = ord(ch)
        if cp < 0x80:
            scores["en"] += ch.isascii() and ch.isalpha()
        elif _in(_HANGUL, cp):
            scores["ko"] += 2
        elif _in(_KANA_HAN, cp):
            scores["ja"] += 2
        elif _in(_LATIN, cp):
            scores["en"] += 1
    best = max(scores, key=scores.get)  # first maximum wins, as in the JavaScript reduce
    return best if scores[best] else None


def prompt_ratio(text: str, family: str, fallback_text_id: str) -> float:
    """Characters per token for a prompt: the ratio of the sample answer in its language."""
    lang = guess_lang(text)
    ref = f"{lang}-answer" if lang and f"{lang}-answer" in ratios()["chars_per_token"] \
        else fallback_text_id
    return chars_per_token(ref, family)[0]


@dataclass
class Script:
    """A text split into tokens for one tokenizer family.

    turns: [(user pieces or None, assistant pieces)]; answers and books have one
    turn without a user message. `ratio_id` is the text whose measured ratio is used.
    """

    entry: dict
    family: str
    ratio_id: str
    chars_per_token: float
    measured: bool
    turns: list[tuple[list[str] | None, list[str]]]


def script(family: str, text_id: str = DEFAULT_TEXT, path: str | None = None,
           chars_per_token_override: float | None = None) -> Script:
    """Load a preset text (or the file at `path`) and split it for `family`.

    A file borrows the ratio of the sample answer in the language it looks like,
    as custom text does in the web app.
    """
    if path:
        content = Path(path).read_text(encoding="utf-8")
        if not content.strip():
            raise ValueError(f"{path} is empty")
        lang = guess_lang(content) or "en"
        e = {"id": "custom", "lang": lang, "kind": "answer", "origin": "custom", "title": path}
        ratio_id = f"{lang}-answer"
    else:
        e = entry(text_id)
        content = load(e)
        ratio_id = e["id"]
    cpt, measured = chars_per_token(ratio_id, family)
    if chars_per_token_override is not None:
        if chars_per_token_override <= 0:
            raise ValueError("characters per token must be positive")
        cpt, measured = chars_per_token_override, False
    if e["kind"] == "chat":
        turns = [(chunk_text(t["user"], cpt), chunk_text(t["assistant"], cpt)) for t in content]
    else:
        turns = [(None, chunk_text(content, cpt))]
    return Script(e, family, ratio_id, cpt, measured, turns)


def ratio_label(s: Script) -> str:
    if s.measured:
        return f"{s.family} tokenizer, {s.chars_per_token:.2f} chars/token"
    return f"{s.chars_per_token:.2f} chars/token (not measured for this model)"


def credit_line(e: dict) -> str:
    """One-line attribution for a text, shown beside the mock output."""
    if e["origin"] == "custom":
        return f"Text: {e['title']}"
    if e["origin"] != "public-domain":
        return f"Text: {e['title']} (written for LLM Sim Bench)"
    line = (f"Text: {e['title']} by {e['author']} ({e['first_published']}), public domain. "
            f"Source: {e['source']['name']} {e['source']['page']}")
    if credits().get(e["id"], {}).get("notes"):
        line += f" (edition notes: llmsimbench texts {e['id']})"
    return line


def describe(e: dict) -> str:
    """Full attribution, including the edition and transcription notes Aozora Bunko asks to keep."""
    lines = [f"{e['id']}  [{e['lang']}, {e['kind']}]  {e['title']}"]
    if e["origin"] == "public-domain":
        lines.append(f"  {e['author']} (died {e['author_died']}), first published "
                     f"{e['first_published']}; public domain")
        lines.append(f"  Source: {e['source']['name']} {e['source']['page']}")
        notes = credits().get(e["id"], {}).get("notes")
        if notes:
            lines += ["  " + part for part in notes.split(" / ")]
    else:
        lines.append("  Written for LLM Sim Bench")
    return "\n".join(lines)
