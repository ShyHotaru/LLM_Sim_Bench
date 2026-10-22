"""Export presets, measurements and golden test vectors for the web app.

Python is the reference implementation. The web app reads presets.json, and
its JavaScript estimator is tested against golden.json (and its text splitting
against chunks.json) so the two stay in sync. The output texts and tokenizer
ratios are also copied into the package so an installed CLI can read them.
Regenerate with `llmsimbench export-web` after changing presets, data, texts or the model.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, replace
from pathlib import Path

from . import __version__, calibrate, measurements, report, texts
from .estimator import Efficiency, estimate
from .profiles import (CPU_PRESETS, GPU_PRESETS, MODEL_PRESETS, QUANT_BPW, QUANT_SHARED_BPW, RAM,
                       UNIFIED_PRESETS, System, lookup, unified_system)

WEB_DIR = Path(__file__).resolve().parents[2] / "web"
TOKENIZER_RATIOS = Path(__file__).resolve().parents[2] / "data" / "tokenizer_ratios.json"
TEXTS_DIR = WEB_DIR / "data" / "texts"
PACKAGE_DATA = Path(__file__).resolve().parent / "data"


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
    # Both SYCL builds on a measured Intel GPU.
    ({"kind": "pc", "cpu": "ryzen7-5800x", "ram_gb": 32, "ram_bandwidth_gbps": 51.2,
      "gpus": ["arc-b580-12gb"]}, "llama-3.1-8b", "Q4_K_M", 8192, 16),
    ({"kind": "pc", "cpu": "ryzen7-5800x", "ram_gb": 32, "ram_bandwidth_gbps": 51.2,
      "gpus": ["arc-b580-12gb"], "backend": "sycl-f16"}, "llama-3.1-8b", "Q4_K_M", 8192, 16),
    # Request slots (the sixth value): KV cache for each slot, batched decode steps.
    ({"kind": "unified", "chip": "nvidia-dgx-spark", "memory_gb": 128},
     "qwen2.5-7b", "Q8_0", 8192, 16, 8),
    ({"kind": "pc", "cpu": "ryzen7-5800x", "ram_gb": 32, "ram_bandwidth_gbps": 51.2,
      "gpus": ["rtx4060ti-16gb"]}, "qwen3-30b-a3b", "Q4_K_M", 8192, 8, 4),
    ({"kind": "pc", "cpu": "ryzen9-7950x", "ram_gb": 64, "ram_bandwidth_gbps": 96,
      "gpus": []}, "gpt-oss-20b", "MXFP4", 4096, 16, 16),
]
# (prompt tokens, cached tokens) of the prompt times the golden file records: inside and
# past a sliding window of 128 (+ 512) tokens, and a long context.
PREFILL_CASES = [(1, 0), (512, 0), (100, 600), (2048, 4096), (1000, 30000)]
# KV fills of the requests in the decode steps the golden file records.
STEP_FILLS = [[0], [1000], [0, 0], [100, 2000, 4000], [500] * 8, [0] * 16]


def golden() -> list[dict]:
    cases = []
    for spec, model, quant, context, kv_bits, *rest in GOLDEN_CASES:
        parallel = rest[0] if rest else 1
        est = estimate(build_system(spec), MODEL_PRESETS[model], quant, context, kv_bits,
                       parallel=parallel)
        cases.append({
            "input": {"system": spec, "model": model, "quant": quant, "context": context,
                      "kv_bits": kv_bits, "parallel": parallel},
            "expected": {
                "weights_gb": est.weights_gb, "kv_gb": est.kv_gb, "fits": est.fits,
                "placements": [asdict(p) for p in est.placements],
                "prefill_tps": est.prefill_tps,
                "decode_tps_empty": est.decode_tps_empty,
                "decode_tps_full": est.decode_tps_full,
                "decode_tps_at": [[f, est.decode_tps_at(f)] for f in (0, 1000, context // 2)],
                "prefill_time": [[n, d, est.prefill_time(n, d)] for n, d in PREFILL_CASES],
                "step_time": [[fills, est.step_time(fills)] for fills in STEP_FILLS],
                "parallel_tps": [[n, f, est.parallel_tps(n, f)] for n in (1, 3, 32)
                                 for f in (0, context)],
                "notes": est.notes,
            },
        })
    return cases


# Reports for the web app to reproduce: (label, system spec, model, quant, context,
# kv_bits, timing). Cases with the same share_url form one multi-configuration report.
REPORT_CASES = [
    [(None, {"kind": "pc", "cpu": "i7-13700k", "ram_gb": 32, "ram_bandwidth_gbps": 89.6,
             "gpus": ["rtx4090-24gb"]}, "llama-3.1-8b", "Q4_K_M", 8192, 16,
      {"prompt_tokens": 1000, "max_tokens": 300, "depth": 0})],
    [(None, {"kind": "pc", "cpu": "ryzen9-7950x", "ram_gb": 64, "ram_bandwidth_gbps": 96,
             "gpus": []}, "llama-3.1-8b", "Q4_0", 4096, 16, None)],
    [("A", {"kind": "unified", "chip": "apple-m4-max-40c", "memory_gb": 48},
      "llama-3.1-70b", "Q4_K_M", 8192, 16, {"prompt_tokens": 2000, "max_tokens": 4000,
                                            "depth": 4096}),
     ("B", {"kind": "pc", "cpu": "ryzen7-5800x", "ram_gb": 32, "ram_bandwidth_gbps": 51.2,
            "gpus": ["rtx4060ti-16gb"], "backend": "vulkan"}, "qwen3-30b-a3b", "Q4_K_M",
      16384, 4, {"prompt_tokens": 1000, "max_tokens": 300, "depth": 0})],
    [(None, {"kind": "pc", "cpu": "i5-12400", "ram_gb": 16, "ram_bandwidth_gbps": 51.2,
             "gpus": ["rtx3060-12gb"]}, "llama-3.1-70b", "Q4_K_M", 200000, 16, None)],
    # Request slots (the eighth value): concurrency table, timing and check, warnings.
    [(None, {"kind": "unified", "chip": "nvidia-dgx-spark", "memory_gb": 128}, "qwen2.5-7b",
      "Q8_0", 8192, 16, {"prompt_tokens": 1000, "max_tokens": 300, "depth": 0}, 8)],
    [("A", {"kind": "pc", "cpu": "ryzen7-5800x", "ram_gb": 64, "ram_bandwidth_gbps": 51.2,
            "gpus": ["rtx3090-24gb"]}, "gpt-oss-20b", "MXFP4", 16384, 16,
      {"prompt_tokens": 2000, "max_tokens": 200, "depth": 1000}, 6),
     ("B", {"kind": "unified", "chip": "apple-m4-max-40c", "memory_gb": 64}, "qwen3-32b",
      "Q4_K_M", 8192, 8, {"prompt_tokens": 2000, "max_tokens": 200, "depth": 1000}, 4)],
]
SHARE_URL = "https://example.invalid/#state"


def report_golden(data: dict) -> list[dict]:
    """Python reports (JSON and Markdown) that web/js/report.js must reproduce."""
    cases = []
    for configs in REPORT_CASES:
        results = []
        for label, spec, model, quant, context, kv_bits, timing, *rest in configs:
            system = build_system(spec)
            est = estimate(system, MODEL_PRESETS[model], quant, context, kv_bits,
                           parallel=rest[0] if rest else 1)
            results.append(report.result(system, est, kv_bits, data, timing, label))
        surface = "web" if len(configs) > 1 else "cli"
        rep = report.report(results, surface, SHARE_URL if surface == "web" else None)
        cases.append({
            "input": [{"label": label, "system": spec, "model": model, "quant": quant,
                       "context": context, "kv_bits": kv_bits, "timing": timing,
                       "parallel": rest[0] if rest else 1}
                      for label, spec, model, quant, context, kv_bits, timing, *rest in configs],
            "surface": surface, "share_url": rep["share_url"],
            "report": rep, "markdown": report.markdown(rep),
        })
    return cases


# Edge cases for chunk_text, stored with every piece: ties at .5 (Python's round()
# differs from Math.round), the whitespace /\s/ matches, ratios below one and at two.
CHUNK_CASES = [
    ("", 3), ("\U0001F600\U0001F600\U0001F600", 1), ("abcde", 2), ("abcdefg", 2), ("ab", 4),
    ("x" * 10, 4), ("x" * 18, 4), ("Hello, world!", 2), ("Hello, world!", 1.99),
    ("Alice was beginning to get very tired of sitting by her sister on the bank.", 4),
    ("Alice was beginning to get very tired of sitting by her sister on the bank.", 3.5),
    ("a\u00a0b\u3000c\ufeffd\u2003e\u001cf\u0085g\u2028h\ti\nj\rk", 2),
    ("word\ufeffword\u001cword word", 2.5),
    ("\uac00\uc744 \ud558\ub298 \uacf5\ud65c\ud55c\ub370", 0.7),
    ("\u543e\u8f29\u306f\u732b\u3067\u3042\u308b\u3002", 1.5),
]

# Free text for guess_lang, which approximates the JavaScript \p{Script=...} classes.
LANG_CASES = [
    "", "12345 !?", "RTX 4090\uc758 VRAM\uc740 24GB\uc785\ub2c8\ub2e4",
    "\u543e\u8f29\u306f\u732b\u3067\u3042\u308b", "\u30ab\u30bf\u30ab\u30ca\u30fc",
    "\uff21\uff22\uff23", "Caf\u00e9 \u00e0 la cr\u00e8me", "\u1100\u1161 abc",
    "ab \u4e00", "abcd \u4e00",
]


def _digest(pieces: list[str]) -> str:
    return hashlib.sha256("\u0000".join(pieces).encode("utf-8")).hexdigest()


def chunk_golden() -> dict:
    """Python chunk_text, chars_per_token and guess_lang results the JavaScript must match."""
    table = json.loads(TOKENIZER_RATIOS.read_text(encoding="utf-8"))
    manifest = json.loads((TEXTS_DIR / "manifest.json").read_text(encoding="utf-8"))
    cases = [{"text": t, "chars_per_token": c, "pieces": texts.chunk_text(t, c)}
             for t, c in CHUNK_CASES]
    for e in manifest["texts"]:
        raw = (TEXTS_DIR / e["file"]).read_text(encoding="utf-8")
        parts = ([(i, role, turn[role]) for i, turn in enumerate(json.loads(raw)["turns"])
                  for role in ("user", "assistant")] if e["kind"] == "chat"
                 else [(None, None, raw)])
        for family in [*table["chars_per_token"][e["id"]], "unknown"]:
            cpt, measured = texts.chars_per_token(e["id"], family, table)
            for turn, role, text in parts:
                pieces = texts.chunk_text(text, cpt)
                cases.append({"text_id": e["id"], "turn": turn, "role": role, "family": family,
                              "chars_per_token": cpt, "measured": measured, "count": len(pieces),
                              "head": pieces[:8], "sha256": _digest(pieces)})
    return {
        "note": "Generated by `llmsimbench export-web` from the Python implementation in "
                "src/llm_sim_bench/texts.py; web/test/tokenize.test.mjs checks tokenize.js.",
        "digest": "sha256 of the pieces joined with U+0000, UTF-8",
        "chunks": cases,
        "guess_lang": [{"text": t, "lang": texts.guess_lang(t)} for t in LANG_CASES] + [
            {"text_id": e["id"], "lang": texts.guess_lang(
                (TEXTS_DIR / e["file"]).read_text(encoding="utf-8"))} for e in manifest["texts"]],
    }


def package_data() -> dict[Path, bytes]:
    """Copies of the output texts and tokenizer ratios read by the installed package."""
    files = {PACKAGE_DATA / "tokenizer_ratios.json": TOKENIZER_RATIOS.read_bytes()}
    for src in sorted(TEXTS_DIR.iterdir()):
        if src.is_file():
            files[PACKAGE_DATA / "texts" / src.name] = src.read_bytes()
    return files


def presets(records: list[dict]) -> dict:
    return {
        "version": __version__,
        "efficiency": asdict(Efficiency()),
        "quant_bpw": QUANT_BPW,
        "quant_shared_bpw": QUANT_SHARED_BPW,
        "cpus": [asdict(c) for c in CPU_PRESETS.values()],
        "gpus": [asdict(g) for g in GPU_PRESETS.values()],
        "unified": [asdict(a) for a in UNIFIED_PRESETS.values()],
        "models": [asdict(m) for m in MODEL_PRESETS.values()],
        "tokenizers": json.loads(TOKENIZER_RATIOS.read_text(encoding="utf-8")),
        **calibration_data(records),
    }


def calibration_data(records: list[dict]) -> dict:
    """Calibration error stats and the benchmark rows of the check tables.

    The web app reads them from presets.json; the package keeps a copy for reports.
    """
    stats = calibrate.summary(calibrate.evaluate(records, Efficiency()))
    batch = calibrate.batch_points(records, Efficiency())
    stats.update(calibrate.batch_summary(batch))
    stats.update(calibrate.depth_summary(calibrate.depth_points(records, Efficiency())))
    return {
        "calibration": {k: {"n": v["n"], "median": round(v["median"], 4),
                            "max": round(v["max"], 4)} for k, v in stats.items()},
        "batch_devices": calibrate.batch_devices(batch),
        "bands": calibrate.error_bands(records, Efficiency()),
        "measurements": [{
            "id": r["id"], "gpu": r["hardware"]["gpu"], "preset": r["hardware"]["gpu_preset"],
            "backend": r["software"]["backend"], "sim_backend": calibrate.backend_of(r),
            "flash_attn": r["software"]["flash_attn"],
            "model": r["model"]["name"], "quant": r["model"]["quant"],
            "n_prompt": r["run"]["n_prompt"], "n_gen": r["run"]["n_gen"],
            "pp_tps": r["results"]["pp_tps"], "tg_tps": r["results"]["tg_tps"],
            "source": r["source"]["url"], "commit": r["software"]["commit"],
            "old_build": "old_build" in r["flags"],
        } for r in records if r["hardware"]["gpu_preset"] and "suspect" not in r["flags"]
            and "batched" not in r["flags"]
            and r["model"]["name"] in MODEL_PRESETS and r["model"]["quant"] in QUANT_BPW
            and r["run"]["n_prompt"] and r["run"]["n_gen"] and not r["run"]["depth"]
            and calibrate.backend_of(r)],
        "cpu_measurements": cpu_measurements(records),
        "batched_measurements": batched_measurements(records),
    }


def batched_measurements(records: list[dict]) -> list[dict]:
    """llama-batched-bench runs the concurrency check compares with the simulator: the
    shortest prompt of each model on each device, every sequence count."""
    recs = [r for r in records if calibrate.usable_batched(r)]
    shortest = {}
    for r in recs:
        k = (r["hardware"]["gpu_preset"], r["software"]["backend"], r["model"]["name"])
        shortest[k] = min(shortest.get(k, r["run"]["n_prompt"]), r["run"]["n_prompt"])
    return [{
        "id": r["id"], "preset": r["hardware"]["gpu_preset"], "backend": r["software"]["backend"],
        "sim_backend": calibrate.backend_of(r), "model": r["model"]["name"],
        "sim_model": calibrate._model(r).name,
        "quant": r["model"]["quant"], "n_prompt": r["run"]["n_prompt"], "n_gen": r["run"]["n_gen"],
        "n_parallel": r["run"]["n_parallel"], "pp_tps": r["results"]["pp_tps"],
        "tg_tps": r["results"]["tg_tps"], "source": r["source"]["url"],
        "commit": r["software"]["commit"],
    } for r in recs if r["run"]["n_prompt"] == shortest[(r["hardware"]["gpu_preset"],
                                                         r["software"]["backend"],
                                                         r["model"]["name"])]]


def cpu_measurements(records: list[dict]) -> list[dict]:
    """CPU-only runs the app can compare with the simulator, filtered like calibration.

    A result is dropped when calibration ignores it (OpenBLAS prompt processing,
    thread-limited decode) or when the run's memory speed is unknown (decode only).
    """
    rows = []
    for r in records:
        if r["software"]["backend"] != "CPU" or not calibrate.usable(r):
            continue
        hw, run, res, flags = r["hardware"], r["run"], r["results"], set(r["flags"])
        bandwidth = calibrate._ram_bandwidth(hw)
        pp = res["pp_tps"] if run["n_prompt"] and not calibrate._NO_PP & flags else None
        tg = (res["tg_tps"] if run["n_gen"] and bandwidth and not calibrate._NO_TG & flags
              else None)
        if pp is None and tg is None:
            continue
        rows.append({
            "id": r["id"], "cpu": hw["cpu"], "preset": hw["cpu_preset"],
            "model": r["model"]["name"], "sim_model": calibrate._model(r).name,
            "quant": r["model"]["quant"], "threads": hw.get("threads"),
            "dimms": hw.get("dimms"), "mem_mts": hw.get("mem_mts"),
            "ram_bandwidth_gbps": bandwidth,
            "n_prompt": run["n_prompt"] if pp is not None else None,
            "n_gen": run["n_gen"] if tg is not None else None,
            "pp_tps": pp, "tg_tps": tg, "source": r["source"]["url"],
            "commit": r["software"]["commit"],
        })
    return rows


def _dump(obj) -> str:
    return json.dumps(obj, indent=1, ensure_ascii=False) + "\n"


def outputs(records: list[dict]) -> dict[Path, bytes]:
    data = calibration_data(records)
    return {
        WEB_DIR / "data" / "presets.json": _dump(presets(records)).encode("utf-8"),
        WEB_DIR / "test" / "golden.json": _dump(golden()).encode("utf-8"),
        WEB_DIR / "test" / "chunks.json": _dump(chunk_golden()).encode("utf-8"),
        WEB_DIR / "test" / "report.json": _dump(report_golden(data)).encode("utf-8"),
        report.CALIBRATION_DATA: _dump(data).encode("utf-8"),
        **package_data(),
    }


def _extra_package_files(files: dict[Path, bytes]) -> list[Path]:
    """Package text copies whose source was removed from web/data/texts."""
    folder = PACKAGE_DATA / "texts"
    return [p for p in folder.iterdir() if p not in files] if folder.is_dir() else []


def write(records: list[dict] | None = None) -> list[Path]:
    files = outputs(records if records is not None else measurements.load())
    for path, data in files.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
    for path in _extra_package_files(files):
        path.unlink()
    return list(files)


def stale(records: list[dict] | None = None) -> list[Path]:
    """Exported files that differ from what the current code would write."""
    files = outputs(records if records is not None else measurements.load())
    # A Windows checkout may have CRLF line endings; they do not make a file stale.
    def same(path: Path, data: bytes) -> bool:
        return path.read_bytes().replace(b"\r\n", b"\n") == data.replace(b"\r\n", b"\n")

    return [p for p, data in files.items()
            if not p.exists() or not same(p, data)] + _extra_package_files(files)
