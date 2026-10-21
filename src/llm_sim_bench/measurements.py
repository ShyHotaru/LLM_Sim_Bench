"""Real-world benchmark records stored as JSON Lines in data/measurements/.

Each line is one llama-bench style result: hardware, software, model, run
settings and the measured prefill (pp) and decode (tg) speeds, plus where
the numbers came from. Fields a source does not state are null, never guessed.
See data/README.md for the schema.
"""

from __future__ import annotations

import json
from pathlib import Path

BACKENDS = ("CUDA", "Metal", "ROCm", "Vulkan", "SYCL", "CPU")

# section -> required keys
SCHEMA: dict[str, tuple[str, ...]] = {
    "source": ("url", "title", "retrieved"),
    "hardware": ("gpu", "gpu_count", "vram_gb", "memory_type", "bus_width_bits", "gpu_cores",
                 "bandwidth_gbps", "cpu", "ram_gb", "gpu_preset"),
    "software": ("runtime", "backend", "commit", "flash_attn"),
    "model": ("name", "quant", "file"),
    "run": ("n_gpu_layers", "n_prompt", "n_gen", "depth"),
    "results": ("pp_tps", "pp_std", "tg_tps", "tg_std"),
}


def default_dir() -> Path:
    """data/measurements in a source checkout."""
    return Path(__file__).resolve().parents[2] / "data" / "measurements"


def load(path: str | Path | None = None) -> list[dict]:
    """Load every record from a .jsonl file or a directory of them."""
    path = Path(path) if path else default_dir()
    files = sorted(path.glob("*.jsonl")) if path.is_dir() else [path]
    records = []
    for f in files:
        with open(f, encoding="utf-8") as fh:
            for n, line in enumerate(fh, 1):
                if line.strip():
                    try:
                        records.append(json.loads(line))
                    except json.JSONDecodeError as e:
                        raise ValueError(f"{f.name}:{n}: {e}") from None
    return records


def validate(rec: dict) -> list[str]:
    """Return a list of problems with one record; empty means valid."""
    errors = []
    if not isinstance(rec.get("id"), str) or not rec["id"]:
        errors.append("missing id")
    if not isinstance(rec.get("flags"), list):
        errors.append("flags must be a list")
    for section, keys in SCHEMA.items():
        body = rec.get(section)
        if not isinstance(body, dict):
            errors.append(f"missing section {section!r}")
            continue
        errors += [f"{section}.{k} missing" for k in keys if k not in body]
    if errors:
        return errors

    if not rec["source"]["url"].startswith("https://"):
        errors.append("source.url must be https")
    if rec["software"]["backend"] not in BACKENDS:
        errors.append(f"unknown backend {rec['software']['backend']!r}")
    if not isinstance(rec["model"]["quant"], str) or not rec["model"]["quant"]:
        errors.append("model.quant must be a string")
    res = rec["results"]
    if res["pp_tps"] is None and res["tg_tps"] is None:
        errors.append("no results")
    for k in ("pp_tps", "tg_tps"):
        if res[k] is not None and not (isinstance(res[k], (int, float)) and res[k] > 0):
            errors.append(f"results.{k} must be a positive number or null")
    return errors


def check_all(records: list[dict]) -> list[str]:
    """Validate records and require unique ids."""
    problems, seen = [], set()
    for rec in records:
        rid = rec.get("id", "?")
        problems += [f"{rid}: {e}" for e in validate(rec)]
        if rid in seen:
            problems.append(f"{rid}: duplicate id")
        seen.add(rid)
    return problems
