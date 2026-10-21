"""Export presets, measurements and golden test vectors for the web app.

Python is the reference implementation. The web app reads presets.json, and
its JavaScript estimator is tested against golden.json so the two stay in sync.
Regenerate with `llmsimbench export-web` after changing presets, data or the model.
"""

from __future__ import annotations

import json
from dataclasses import asdict, replace
from pathlib import Path

from . import calibrate, measurements
from .estimator import Efficiency, estimate
from .profiles import (CPU_PRESETS, GPU_PRESETS, MODEL_PRESETS, QUANT_BPW, RAM, UNIFIED_PRESETS,
                       System, lookup, unified_system)

WEB_DIR = Path(__file__).resolve().parents[2] / "web"
TOKENIZER_RATIOS = Path(__file__).resolve().parents[2] / "data" / "tokenizer_ratios.json"


def build_system(spec: dict) -> System:
    """Build a System from the JSON spec shared with the web app.

    {"kind": "pc", "cpu": name, "ram_gb": n, "ram_bandwidth_gbps": n, "gpus": [names],
     "backend": optional override of every GPU's backend, e.g. "vulkan"}
    {"kind": "unified", "chip": name, "memory_gb": n}
    """
    if spec["kind"] in ("unified", "apple"):
        chip = lookup(UNIFIED_PRESETS, spec["chip"], "unified-memory chip")
        if spec.get("backend"):
            chip = replace(chip, backend=spec["backend"])
        return unified_system(chip, spec["memory_gb"])
    gpus = tuple(lookup(GPU_PRESETS, g, "GPU") for g in spec["gpus"])
    if spec.get("backend"):
        gpus = tuple(replace(g, backend=spec["backend"]) for g in gpus)
    return System(lookup(CPU_PRESETS, spec["cpu"], "CPU"),
                  RAM(spec["ram_gb"], spec["ram_bandwidth_gbps"]), gpus)


GOLDEN_CASES = [
    ({"kind": "pc", "cpu": "ryzen7-5800x", "ram_gb": 64, "ram_bandwidth_gbps": 51.2,
      "gpus": ["rtx3080-10gb"]}, "qwen2.5-14b", "Q4_K_M", 8192, 16),
    ({"kind": "pc", "cpu": "i7-13700k", "ram_gb": 32, "ram_bandwidth_gbps": 89.6,
      "gpus": ["rtx4090-24gb"]}, "llama-3.1-8b", "Q8_0", 32768, 16),
    ({"kind": "pc", "cpu": "ryzen9-7950x", "ram_gb": 64, "ram_bandwidth_gbps": 96,
      "gpus": ["rtx3090-24gb", "rtx3090-24gb"]}, "llama-3.1-70b", "Q4_K_M", 8192, 8),
    ({"kind": "pc", "cpu": "i5-12400", "ram_gb": 16, "ram_bandwidth_gbps": 51.2,
      "gpus": []}, "llama-3.1-70b", "Q4_K_M", 4096, 16),
    ({"kind": "pc", "cpu": "ryzen9-7950x", "ram_gb": 128, "ram_bandwidth_gbps": 96,
      "gpus": ["rtx5090-32gb"]}, "gpt-oss-120b", "MXFP4", 131072, 16),
    ({"kind": "pc", "cpu": "ryzen7-5800x", "ram_gb": 32, "ram_bandwidth_gbps": 51.2,
      "gpus": ["rtx4060ti-16gb"]}, "qwen3-30b-a3b", "Q4_K_M", 16384, 4),
    ({"kind": "unified", "chip": "apple-m4-max-40c", "memory_gb": 48},
     "llama-3.1-70b", "Q4_K_M", 8192, 16),
    ({"kind": "unified", "chip": "apple-m2-ultra-76c", "memory_gb": 192},
     "qwen3-32b", "Q6_K", 32768, 16),
    ({"kind": "unified", "chip": "apple-m1-8c", "memory_gb": 8},
     "llama-3.2-3b", "Q4_0", 4096, 16),
]


def golden() -> list[dict]:
    cases = []
    for spec, model, quant, context, kv_bits in GOLDEN_CASES:
        est = estimate(build_system(spec), MODEL_PRESETS[model], quant, context, kv_bits)
        cases.append({
            "input": {"system": spec, "model": model, "quant": quant, "context": context,
                      "kv_bits": kv_bits},
            "expected": {
                "weights_gb": est.weights_gb, "kv_gb": est.kv_gb, "fits": est.fits,
                "placements": [asdict(p) for p in est.placements],
                "prefill_tps": est.prefill_tps,
                "decode_tps_empty": est.decode_tps_empty,
                "decode_tps_full": est.decode_tps_full,
                "decode_tps_at": [[f, est.decode_tps_at(f)] for f in (0, 1000, context // 2)],
                "notes": est.notes,
            },
        })
    return cases


def presets(records: list[dict]) -> dict:
    stats = calibrate.summary(calibrate.evaluate(records, Efficiency()))
    return {
        "efficiency": asdict(Efficiency()),
        "quant_bpw": QUANT_BPW,
        "cpus": [asdict(c) for c in CPU_PRESETS.values()],
        "gpus": [asdict(g) for g in GPU_PRESETS.values()],
        "unified": [asdict(a) for a in UNIFIED_PRESETS.values()],
        "models": [asdict(m) for m in MODEL_PRESETS.values()],
        "tokenizers": json.loads(TOKENIZER_RATIOS.read_text(encoding="utf-8")),
        "calibration": {k: {"n": v["n"], "median": round(v["median"], 4),
                            "max": round(v["max"], 4)} for k, v in stats.items()},
        "measurements": [{
            "id": r["id"], "gpu": r["hardware"]["gpu"], "preset": r["hardware"]["gpu_preset"],
            "backend": r["software"]["backend"], "flash_attn": r["software"]["flash_attn"],
            "model": r["model"]["name"], "quant": r["model"]["quant"],
            "n_prompt": r["run"]["n_prompt"], "n_gen": r["run"]["n_gen"],
            "pp_tps": r["results"]["pp_tps"], "tg_tps": r["results"]["tg_tps"],
            "source": r["source"]["url"], "commit": r["software"]["commit"],
            "old_build": "old_build" in r["flags"],
        } for r in records if r["hardware"]["gpu_preset"] and "suspect" not in r["flags"]
            and r["model"]["name"] in MODEL_PRESETS and r["model"]["quant"] in QUANT_BPW
            and r["run"]["n_prompt"] and r["run"]["n_gen"]],
    }


def _dump(obj) -> str:
    return json.dumps(obj, indent=1, ensure_ascii=False) + "\n"


def outputs(records: list[dict]) -> dict[Path, str]:
    return {
        WEB_DIR / "data" / "presets.json": _dump(presets(records)),
        WEB_DIR / "test" / "golden.json": _dump(golden()),
    }


def write(records: list[dict] | None = None) -> list[Path]:
    files = outputs(records if records is not None else measurements.load())
    for path, text in files.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    return list(files)


def stale(records: list[dict] | None = None) -> list[Path]:
    """Exported files that differ from what the current code would write."""
    files = outputs(records if records is not None else measurements.load())
    return [p for p, text in files.items()
            if not p.exists() or p.read_text(encoding="utf-8") != text]
