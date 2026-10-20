"""Measure characters per token for every output text and tokenizer family.

The web mock splits text into token-sized pieces using these ratios, so the
speed in characters per second matches what each model family would show.
Only the resulting counts are stored (data/tokenizer_ratios.json); tokenizer
files are cached under .cache/ and never committed.

This is a development tool and needs `pip install tokenizers`; the package
itself stays dependency-free.

Usage: python tools/measure_tokenizers.py
"""

from __future__ import annotations

import datetime
import json
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TEXTS = ROOT / "web" / "data" / "texts"
OUT = ROOT / "data" / "tokenizer_ratios.json"
CACHE = ROOT / ".cache" / "tokenizers"

# family -> (Hugging Face repo holding tokenizer.json, license of that tokenizer)
FAMILIES = {
    "llama2": ("NousResearch/Llama-2-7b-hf", "Llama 2 Community License"),
    "mistral": ("mistralai/Mistral-7B-v0.1", "Apache-2.0"),
    "llama3": ("NousResearch/Meta-Llama-3-8B", "Meta Llama 3 Community License"),
    "qwen2.5": ("Qwen/Qwen2.5-7B-Instruct", "Apache-2.0"),
    "qwen3": ("Qwen/Qwen3-8B", "Apache-2.0"),
    "gemma2": ("unsloth/gemma-2-9b-it", "Gemma Terms of Use"),
    "gpt-oss": ("openai/gpt-oss-20b", "Apache-2.0"),
}


def tokenizer_file(repo: str) -> Path:
    path = CACHE / f"{repo.replace('/', '__')}.json"
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        url = f"https://huggingface.co/{repo}/resolve/main/tokenizer.json"
        print(f"downloading {url}", file=sys.stderr)
        with urllib.request.urlopen(url, timeout=300) as resp:
            path.write_bytes(resp.read())
    return path


def text_of(entry: dict) -> str:
    raw = (TEXTS / entry["file"]).read_text(encoding="utf-8")
    if entry["kind"] == "chat":
        turns = json.loads(raw)["turns"]
        return "\n".join(part for turn in turns for part in (turn["user"], turn["assistant"]))
    return raw


def main() -> int:
    from tokenizers import Tokenizer

    manifest = json.loads((TEXTS / "manifest.json").read_text(encoding="utf-8"))
    texts = {e["id"]: text_of(e) for e in manifest["texts"]}
    families, ratios = {}, {tid: {} for tid in texts}
    for family, (repo, license_name) in FAMILIES.items():
        tok = Tokenizer.from_file(str(tokenizer_file(repo)))
        families[family] = {"repo": repo, "license": license_name,
                            "vocab_size": tok.get_vocab_size()}
        for tid, text in texts.items():
            n = len(tok.encode(text, add_special_tokens=False).ids)
            ratios[tid][family] = round(len(text) / n, 3)
    OUT.write_text(json.dumps({
        "measured": datetime.date.today().isoformat(),
        "unit": "characters (Unicode code points) per token, no special tokens",
        "families": families,
        "chars_per_token": ratios,
    }, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(f"wrote {OUT}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
