"""Estimate reports as JSON and Markdown.

A report holds one result per configuration: the virtual system, the model, memory
placement, speeds, notes, warnings about uncalibrated parts, and the check table that
compares the simulator with real benchmarks on the same hardware. `web/js/report.js`
builds the same report in the browser; web/test/report.json (written by
`llmsimbench export-web`) keeps the two in sync.

Reports are in English. Every note and warning has a code and parameters so tools
can read them without parsing text.
"""

from __future__ import annotations

import json
import statistics
from dataclasses import replace
from pathlib import Path

from . import __version__
from .estimator import Efficiency, Estimate, add_in_order, estimate, parallel_levels
from .profiles import CPU_PRESETS, GPU_PRESETS, MODEL_PRESETS, RAM, UNIFIED_PRESETS, System

SCHEMA = "llmsimbench.report/1"
CALIBRATION_DATA = Path(__file__).resolve().parent / "data" / "calibration.json"
BACKEND_NAMES = {"cuda": "CUDA", "metal": "Metal", "rocm": "ROCm", "vulkan": "Vulkan",
                 "sycl": "SYCL", "cpu": "CPU"}
# Contexts from which an uncalibrated slowdown with depth is worth a warning.
LONG_CONTEXT = 8192

# English texts for note and warning codes; the web app shows localized versions.
TEXTS = {
    "context_exceeds_max": "Context {context} exceeds the model's maximum of {max}.",
    "ram_short": "Needs {need_gb} GB in system RAM but only {free_gb} GB is free; "
                 "it would page to disk.",
    "layers_on_cpu": "{cpu_layers} of {total_layers} layers are offloaded to the CPU.",
    "prefill_device_uncalibrated": "No prompt-processing measurements for {device} with this "
                                   "backend: its prompt speed uses the backend average and can "
                                   "be far off.",
    "cpu_device_uncalibrated": "No decode measurements for the {cpu}: its memory efficiency uses "
                               "the CPU average. Single-CCD Ryzen parts measure about 25 % slower "
                               "than that.",
    "backend_uncalibrated": "No measured {backend} results yet, so its speeds are uncalibrated "
                            "guesses.",
    "decode_uncalibrated": "{backend} generation speed is not calibrated yet.",
    "prefill_uncalibrated": "{backend} prompt processing speed is not calibrated yet.",
    "moe_uncalibrated": "Mixture-of-experts overhead is not calibrated for {backend}, so "
                        "generation and prompt processing may be optimistic.",
    "moe_prefill_uncalibrated": "Mixture-of-experts prompt processing is not calibrated for "
                                "{backend}, so it may be optimistic.",
    "depth_uncalibrated": "The slowdown as the context fills is not calibrated for {backend} "
                          "yet, so long contexts may be slower than shown.",
    "parallel_uncalibrated": "No concurrent-request measurements for {backend} yet: the "
                             "speed-up from batching uses the CUDA factor.",
    "parallel_device_uncalibrated": "Concurrent-request speeds are fitted to other hardware "
                                    "only, not to {device}.",
    "moe_parallel": "Concurrent requests of a mixture-of-experts model are not calibrated: the "
                    "estimate assumes every request picks experts at random, while measured "
                    "speed-ups are higher.",
}


def load_data(path: Path = CALIBRATION_DATA) -> dict:
    """Calibration error stats and benchmark rows written by `llmsimbench export-web`."""
    return json.loads(path.read_text(encoding="utf-8"))


def _item(code: str, **params) -> dict:
    shown = {k: fmt(v, 0) if isinstance(v, int) else v for k, v in params.items()}
    return {"code": code, "params": params, "text": TEXTS[code].format(**shown)}


def _gb1(x: float) -> str:
    return fmt(x, 1)


def _label(name: str) -> str:
    for presets in (GPU_PRESETS, UNIFIED_PRESETS, CPU_PRESETS):
        if name in presets and presets[name].label:
            return presets[name].label
    return name


def device_label(system: System, p) -> str:
    if p.is_gpu:
        return _label(p.device)
    return "CPU" if system.unified else f"CPU {_label(system.cpu.name)}"


def system_info(system: System) -> dict:
    gpus = [{"name": g.name, "label": _label(g.name), "backend": g.backend,
             "vram_gb": None if system.unified else g.vram_gb} for g in system.gpus]
    return {
        "kind": "unified" if system.unified else "pc",
        "cpu": None if system.unified else {"name": system.cpu.name,
                                            "label": _label(system.cpu.name)},
        "ram_gb": system.ram.size_gb,
        "ram_bandwidth_gbps": system.ram.bandwidth_gbps,
        "gpus": gpus,
    }


def ram_use(system: System, est: Estimate, eff: Efficiency = Efficiency()) -> tuple[float, float]:
    """(GB of layers in system RAM, GB of system RAM free for them)."""
    cpu = [p for p in est.placements if not p.is_gpu]
    free = system.ram.size_gb - eff.ram_reserve_gb
    if system.unified:
        free -= add_in_order(p.weights_gb + p.kv_gb for p in est.placements if p.is_gpu)
    return (cpu[0].weights_gb + cpu[0].kv_gb if cpu else 0.0), free


def notes(system: System, est: Estimate, eff: Efficiency = Efficiency()) -> list[dict]:
    out = []
    m = est.model
    cpu = [p for p in est.placements if not p.is_gpu]
    if est.context > m.max_context:
        out.append(_item("context_exceeds_max", context=est.context, max=m.max_context))
    if cpu and not est.fits:
        need, free = ram_use(system, est, eff)
        out.append(_item("ram_short", need_gb=_gb1(need), free_gb=_gb1(free)))
    if cpu and system.gpus:
        out.append(_item("layers_on_cpu", cpu_layers=cpu[0].layers, total_layers=m.n_layers))
    return out


def warnings(system: System, est: Estimate, data: dict,
             eff: Efficiency = Efficiency()) -> list[dict]:
    """Where the numbers rest on uncalibrated assumptions rather than measurements."""
    cal = data["calibration"]
    gpu_backend = {g.name: g.backend for g in system.gpus}
    backends = list(dict.fromkeys(gpu_backend[p.device] if p.is_gpu else "cpu"
                                  for p in est.placements))
    out = []
    devices = []
    for g in system.gpus:
        if f"{g.backend}/pp" in cal and f"{g.name}|{g.backend}" not in eff.device_compute:
            if _label(g.name) not in devices:
                devices.append(_label(g.name))
    out += [_item("prefill_device_uncalibrated", device=d) for d in devices]
    if ("cpu" in backends and not system.unified
            and f"{system.cpu.name}|cpu" not in eff.device_bandwidth):
        out.append(_item("cpu_device_uncalibrated", cpu=_label(system.cpu.name)))
    for b in backends:
        tg, pp = f"{b}/tg" in cal, f"{b}/pp" in cal
        if not tg and not pp:
            out.append(_item("backend_uncalibrated", backend=BACKEND_NAMES[b]))
        elif not tg:
            out.append(_item("decode_uncalibrated", backend=BACKEND_NAMES[b]))
        elif not pp:
            out.append(_item("prefill_uncalibrated", backend=BACKEND_NAMES[b]))
    for b in backends:
        if est.model.active_params_b and f"{b}/moe_tg" not in cal:
            out.append(_item("moe_uncalibrated", backend=BACKEND_NAMES[b]))
        elif est.model.active_params_b and f"{b}/moe_pp" not in cal:
            out.append(_item("moe_prefill_uncalibrated", backend=BACKEND_NAMES[b]))
        if est.context >= LONG_CONTEXT and f"{b}/depth_tg" not in cal:
            out.append(_item("depth_uncalibrated", backend=BACKEND_NAMES[b]))
    if est.parallel > 1:
        batched = set(data["batch_devices"])
        for b in backends:
            if f"{b}/batch" not in cal:
                out.append(_item("parallel_uncalibrated", backend=BACKEND_NAMES[b]))
        devices = []
        for g in system.gpus:
            if (f"{g.backend}/batch" in cal and f"{g.name}|{g.backend}" not in batched
                    and _label(g.name) not in devices):
                devices.append(_label(g.name))
        out += [_item("parallel_device_uncalibrated", device=d) for d in devices]
        if est.model.active_params_b:
            out.append(_item("moe_parallel"))
    return out


def gen_time(est: Estimate, filled: int, max_tokens: int) -> float:
    """Seconds for tokens 2..max_tokens of a reply after `filled` context tokens."""
    return add_in_order(1 / est.decode_tps_at(filled + i) for i in range(1, max_tokens))


def _with_backend(system: System, backend: str) -> System:
    return replace(system, gpus=tuple(replace(g, backend=backend) for g in system.gpus))


def gpu_checks(system: System, kv_bits: int, data: dict) -> dict | None:
    """Benchmarks of the selected GPU (single-GPU PC) or chip, each simulated with the
    backend, model and test size of the run."""
    if len(system.gpus) != 1:
        return None
    rows = [m for m in data["measurements"] if m["preset"] == system.gpus[0].name]
    if not rows:
        return None
    out = []
    for m in rows:
        est = estimate(_with_backend(system, m["backend"].lower()), MODEL_PRESETS[m["model"]],
                       m["quant"], m["n_prompt"] + m["n_gen"], kv_bits)
        out.append({
            "backend": m["backend"], "model": m["model"], "quant": m["quant"],
            "flash_attn": m["flash_attn"], "old_build": m["old_build"],
            "n_prompt": m["n_prompt"], "n_gen": m["n_gen"],
            "tg_measured": m["tg_tps"], "tg_simulated": m["n_gen"] / gen_time(est, 0, m["n_gen"] + 1),
            "pp_measured": m["pp_tps"],
            "pp_simulated": m["n_prompt"] / est.prefill_time(m["n_prompt"]),
            "sources": [1],
        })
    return {"kind": "gpu", "rows": out, "sources": [rows[0]["source"]]}


def group_cpu_runs(rows: list[dict]) -> tuple[list[dict], list[str]]:
    """CPU runs grouped like web/js/measured.js groupCpuRuns: one row per model, quant,
    memory, threads and test size with the median result."""
    sources: list[str] = []
    groups: dict[tuple, dict] = {}
    for r in rows:
        key = tuple(r[k] for k in ("model", "quant", "dimms", "mem_mts", "threads",
                                   "n_prompt", "n_gen"))
        g = groups.setdefault(key, {**r, "runs": [], "sources": []})
        g["runs"].append(r)
        if r["source"] not in sources:
            sources.append(r["source"])
        n = sources.index(r["source"]) + 1
        if n not in g["sources"]:
            g["sources"].append(n)

    def med(g, k):
        vals = [r[k] for r in g["runs"] if r[k] is not None]
        return statistics.median(vals) if vals else None

    return [{
        "model": g["model"], "sim_model": g["sim_model"], "quant": g["quant"],
        "threads": g["threads"], "dimms": g["dimms"], "mem_mts": g["mem_mts"],
        "ram_bandwidth_gbps": g["ram_bandwidth_gbps"], "n_prompt": g["n_prompt"],
        "n_gen": g["n_gen"], "n": len(g["runs"]), "sources": g["sources"],
        "pp_tps": med(g, "pp_tps"), "tg_tps": med(g, "tg_tps"),
    } for g in groups.values()], sources


def bench_gen_tps(est: Estimate, n_gen: int) -> float:
    """Decode speed llama-bench reports for n_gen tokens from an empty context."""
    return n_gen / add_in_order(1 / est.decode_tps_at(i) for i in range(n_gen))


def cpu_checks(system: System, data: dict) -> dict | None:
    """CPU-only runs on the selected CPU, each simulated with the memory that run used."""
    if system.gpus or system.unified:
        return None
    groups, sources = group_cpu_runs([m for m in data["cpu_measurements"]
                                      if m["preset"] == system.cpu.name])
    if not groups:
        return None
    out = []
    for g in groups:
        sim = System(system.cpu, RAM(1024, g["ram_bandwidth_gbps"] or 1))
        est = estimate(sim, MODEL_PRESETS[g["sim_model"]], g["quant"],
                       max(g["n_prompt"] or 0, g["n_gen"] or 0, 1), 16)
        tg, pp = g["tg_tps"], g["pp_tps"]
        out.append({
            "backend": "CPU", "model": g["model"], "quant": g["quant"],
            "dimms": g["dimms"], "mem_mts": g["mem_mts"], "threads": g["threads"],
            "n_prompt": g["n_prompt"], "n_gen": g["n_gen"], "runs": g["n"],
            "tg_measured": tg,
            "tg_simulated": bench_gen_tps(est, g["n_gen"]) if tg is not None else None,
            "pp_measured": pp,
            "pp_simulated": g["n_prompt"] / est.prefill_time(g["n_prompt"]) if pp is not None
            else None,
            "sources": g["sources"],
        })
    return {"kind": "cpu", "rows": out, "sources": sources}


def timing_info(est: Estimate, timing: dict) -> dict:
    """Time to the first token and to the whole reply for {"prompt_tokens", "max_tokens",
    "depth"}, as the mock plays it. With several slots, also the times when every slot
    gets such a request at once: the prompts share prompt processing, then the replies
    share each decode step."""
    prompt, depth, max_tokens = timing["prompt_tokens"], timing["depth"], timing["max_tokens"]
    ttft = est.prefill_time(prompt, depth)
    out = {**timing, "ttft_s": ttft, "reply_s": ttft + gen_time(est, depth + prompt, max_tokens)}
    if est.parallel > 1:
        n = est.parallel
        ttft_all = n * ttft
        gen = add_in_order(est.step_time([depth + prompt + i] * n) for i in range(1, max_tokens))
        out.update(ttft_all_s=ttft_all, reply_all_s=ttft_all + gen)
    return out


def concurrency(est: Estimate) -> list[dict] | None:
    """Speeds with 1, 2, 4, ... requests active, up to the number of slots."""
    if est.parallel == 1:
        return None
    rows = []
    for n in parallel_levels(est.parallel):
        empty, full = est.parallel_tps(n, 0), est.parallel_tps(n, est.context)
        rows.append({"requests": n, "per_request_tps_empty": empty, "per_request_tps_full": full,
                     "total_tps_empty": n * empty, "total_tps_full": n * full})
    return rows


def parallel_checks(system: System, est: Estimate, kv_bits: int, data: dict) -> dict | None:
    """llama-batched-bench runs on the selected GPU or chip (only those of the selected
    model if there are any), each simulated with the backend, model, prompt length and
    number of sequences of the run."""
    if est.parallel == 1 or len(system.gpus) != 1:
        return None
    rows = [m for m in data["batched_measurements"] if m["preset"] == system.gpus[0].name]
    rows = [m for m in rows if m["sim_model"] == est.model.name] or rows
    if not rows:
        return None
    sources: list[str] = []
    out = []
    for m in rows:
        if m["source"] not in sources:
            sources.append(m["source"])
        n, pp, tg = m["n_parallel"], m["n_prompt"], m["n_gen"]
        sim = estimate(_with_backend(system, m["backend"].lower()), MODEL_PRESETS[m["sim_model"]],
                       m["quant"], pp + tg, kv_bits, parallel=n)
        t = add_in_order(sim.step_time([pp + i] * n) for i in range(tg))
        out.append({
            "backend": m["backend"], "model": m["model"], "quant": m["quant"], "n_prompt": pp,
            "n_gen": tg, "requests": n, "tg_measured": m["tg_tps"], "tg_simulated": n * tg / t,
            "pp_measured": m["pp_tps"], "pp_simulated": pp / sim.prefill_time(pp),
            "sources": [sources.index(m["source"]) + 1],
        })
    return {"kind": "parallel", "rows": out, "sources": sources}


def result(system: System, est: Estimate, kv_bits: int, data: dict, timing: dict | None = None,
           label: str | None = None) -> dict:
    """One configuration's part of a report; `timing` adds timing_info()."""
    m = est.model
    need, free = ram_use(system, est)
    return {
        "label": label,
        "config": {
            "system": system_info(system),
            "model": {"name": m.name, "params_b": m.params_b, "active_params_b": m.active_params_b,
                      "layers": m.n_layers, "max_context": m.max_context},
            "quant": est.quant, "context": est.context, "kv_bits": kv_bits,
            "parallel": est.parallel,
        },
        "memory": {
            "weights_gb": est.weights_gb, "kv_gb": est.kv_gb,
            "total_gb": est.weights_gb + est.kv_gb, "fits": est.fits,
            "ram_needed_gb": need, "ram_free_gb": free,
            "placements": [{"device": p.device, "label": device_label(system, p),
                            "is_gpu": p.is_gpu, "layers": p.layers,
                            "weights_gb": p.weights_gb, "kv_gb": p.kv_gb}
                           for p in est.placements],
        },
        "speed": {
            "prefill_tps": est.prefill_tps,
            "decode_tps_empty": est.decode_tps_empty,
            "decode_tps_full": est.decode_tps_full,
            "concurrency": concurrency(est),
        },
        "timing": timing_info(est, timing) if timing else None,
        "notes": notes(system, est),
        "warnings": warnings(system, est, data),
        "check": gpu_checks(system, kv_bits, data) or cpu_checks(system, data),
        "parallel_check": parallel_checks(system, est, kv_bits, data),
    }


def report(results: list[dict], surface: str = "cli", share_url: str | None = None,
           version: str = __version__) -> dict:
    return {
        "schema": SCHEMA,
        "generated_by": {"tool": "llmsimbench", "version": version, "surface": surface},
        "share_url": share_url,
        "results": results,
    }


# ---------- Markdown ----------

def fmt(x: float, digits: int) -> str:
    return f"{x:,.{digits}f}"


def fmt_tps(x: float | None) -> str:
    if x is None:
        return "–"
    if x >= 100:
        return fmt(x, 0)
    if x >= 10:
        return fmt(x, 1)
    return fmt(x, 2)


def fmt_time(s: float) -> str:
    # Compare the rounded text, so 59.96 s reads "1 min 00 s" rather than "60.0 s".
    if fmt(s * 1000, 0) != "1,000" and s < 1:
        return f"{fmt(s * 1000, 0)} ms"
    if fmt(s, 1) != "60.0" and s < 60:
        return f"{fmt(s, 1)} s"
    m, sec = divmod(int(fmt(s, 0).replace(",", "")), 60)
    return f"{m} min {sec:02d} s" if m < 60 else f"{m // 60} h {m % 60:02d} min"


def _cell(text) -> str:
    return str(text).replace("|", "\\|")


def _table(head: list[str], rows: list[list], align: str = "") -> list[str]:
    align = align or "l" + "r" * (len(head) - 1)
    lines = ["| " + " | ".join(head) + " |",
             "|" + "|".join(" --- " if a == "l" else " ---: " for a in align) + "|"]
    lines += ["| " + " | ".join(_cell(c) for c in row) + " |" for row in rows]
    return lines


def _system_line(s: dict) -> str:
    gpus = ", ".join(f"{g['label']} ({BACKEND_NAMES[g['backend']]})" for g in s["gpus"])
    if s["kind"] == "unified":
        return (f"{gpus}, {fmt(s['ram_gb'], 0)} GB unified memory "
                f"({fmt(s['ram_bandwidth_gbps'], 0)} GB/s)")
    return (f"CPU {s['cpu']['label']} · RAM {fmt(s['ram_gb'], 0)} GB "
            f"({fmt(s['ram_bandwidth_gbps'], 1)} GB/s) · GPU {gpus or 'none'}")


def _model_line(c: dict) -> str:
    m = c["model"]
    size = f"{m['params_b']:g}B" + (f", {m['active_params_b']:g}B active"
                                     if m["active_params_b"] else "")
    slots = ""
    if c["parallel"] > 1:
        slots = (f" per request × {c['parallel']} slots (llama-server -np {c['parallel']} "
                 f"-c {c['context'] * c['parallel']})")
    return (f"{m['name']} ({size}), {c['quant']}, context {fmt(c['context'], 0)}{slots}, "
            f"KV cache {c['kv_bits']}-bit")


def _check_lines(check: dict) -> list[str]:
    rows = []
    for r in check["rows"]:
        if check["kind"] == "gpu":
            run = f"{r['backend']} · {r['quant']}" + (" · FA" if r["flash_attn"] else "")
            if r["old_build"]:
                run += " (2023 build)"
        else:
            memory = (f"{r['mem_mts']} MT/s × {r['dimms']}" if r["dimms"] and r["mem_mts"]
                      else "memory not stated")
            test = " ".join(x for x in (f"pp{r['n_prompt']}" if r["n_prompt"] else "",
                                        f"tg{r['n_gen']}" if r["n_gen"] else "") if x)
            meta = [memory, f"{r['threads']} threads" if r["threads"] else "", test,
                    f"×{r['runs']}"]
            run = f"{r['model']} · {r['quant']} · " + " · ".join(x for x in meta if x)
        refs = "".join(f"[{n}]" for n in r["sources"])
        rows.append([f"{run} {refs}", fmt_tps(r["tg_measured"]), fmt_tps(r["tg_simulated"]),
                     fmt_tps(r["pp_measured"]), fmt_tps(r["pp_simulated"])])
    lines = ["### Check against real benchmarks", "",
             "llama.cpp results on this hardware, measured vs. this simulator (tok/s)."
             + (" CPU rows are simulated with the memory each run used." if check["kind"] == "cpu"
                else ""), ""]
    lines += _table(["Run", "Gen real", "Gen sim", "Prompt real", "Prompt sim"], rows)
    if any(r.get("old_build") for r in check["rows"]):
        lines += ["", "Rows marked as 2023 builds ran on an older, slower llama.cpp; current "
                  "builds are faster."]
    lines += [""] + [f"[{i}] {url}  " for i, url in enumerate(check["sources"], 1)]
    lines[-1] = lines[-1].rstrip()
    return lines


def _result_lines(r: dict, heading: str) -> list[str]:
    c, mem, sp = r["config"], r["memory"], r["speed"]
    lines = [f"## {heading}", ""]
    lines += _table(["", ""], [["System", _system_line(c["system"])], ["Model", _model_line(c)]],
                    "ll")
    lines += ["", "### Memory", ""]
    rows = [[p["label"], p["layers"], fmt(p["weights_gb"], 2), fmt(p["kv_gb"], 2)]
            for p in mem["placements"]]
    rows.append(["Total", c["model"]["layers"], fmt(mem["weights_gb"], 2), fmt(mem["kv_gb"], 2)])
    lines += _table(["Device", "Layers", "Weights GB", "KV cache GB"], rows)
    lines += ["", f"Fits in memory: {'yes' if mem['fits'] else '**no**'}", "", "### Speed", ""]
    rows = [["Generation, empty context", f"{fmt_tps(sp['decode_tps_empty'])} tok/s"],
            ["Generation, full context", f"{fmt_tps(sp['decode_tps_full'])} tok/s"],
            ["Prompt processing", f"{fmt_tps(sp['prefill_tps'])} tok/s"]]
    if r["timing"]:
        t = r["timing"]
        depth = f", {fmt(t['depth'], 0)} tokens already cached" if t["depth"] else ""
        rows += [[f"First token ({fmt(t['prompt_tokens'], 0)} prompt tokens{depth})",
                  fmt_time(t["ttft_s"])],
                 [f"Whole reply ({fmt(t['max_tokens'], 0)} tokens)", fmt_time(t["reply_s"])]]
        if "ttft_all_s" in t:
            n = c["parallel"]
            rows += [[f"First token, {n} such requests at once", fmt_time(t["ttft_all_s"])],
                     [f"Whole reply, {n} such requests at once", fmt_time(t["reply_all_s"])]]
    lines += _table(["Metric", "Estimate"], rows)
    if sp["concurrency"]:
        lines += ["", "### Concurrent requests", "",
                  "Generation speed with this many requests active (tok/s); the total is the "
                  "throughput of all of them together.", ""]
        lines += _table(["Requests", "Each, empty context", "Each, full context",
                         "Total, empty context", "Total, full context"],
                        [[r["requests"], fmt_tps(r["per_request_tps_empty"]),
                          fmt_tps(r["per_request_tps_full"]), fmt_tps(r["total_tps_empty"]),
                          fmt_tps(r["total_tps_full"])] for r in sp["concurrency"]])
    if r["notes"]:
        lines += ["", "### Notes", ""] + [f"- {n['text']}" for n in r["notes"]]
    if r["warnings"]:
        lines += ["", "### Estimate only", ""] + [f"- {w['text']}" for w in r["warnings"]]
    if r["check"]:
        lines += [""] + _check_lines(r["check"])
    if r["parallel_check"]:
        lines += [""] + _parallel_check_lines(r["parallel_check"])
    return lines


def _parallel_check_lines(check: dict) -> list[str]:
    rows = [[f"{r['backend']} · {r['model']} · {r['quant']} · pp{r['n_prompt']} tg{r['n_gen']} "
             + "".join(f"[{n}]" for n in r["sources"]),
             r["requests"], fmt_tps(r["tg_measured"]), fmt_tps(r["tg_simulated"])]
            for r in check["rows"]]
    lines = ["### Check against concurrent benchmarks", "",
             "llama-batched-bench on this hardware: total generation speed of all sequences, "
             "measured vs. this simulator (tok/s).", ""]
    lines += _table(["Run", "Requests", "Gen real", "Gen sim"], rows)
    lines += [""] + [f"[{i}] {url}  " for i, url in enumerate(check["sources"], 1)]
    lines[-1] = lines[-1].rstrip()
    return lines


def markdown(rep: dict) -> str:
    g = rep["generated_by"]
    lines = ["# LLM Sim Bench report", "",
             f"Generated by llmsimbench {g['version']} ({g['surface']}). These are estimates "
             "from a first-order model, not measurements."]
    if rep["share_url"]:
        lines += ["", f"Setup link: <{rep['share_url']}>"]
    for r in rep["results"]:
        lines += [""] + _result_lines(r, f"Configuration {r['label']}" if r["label"] else "Result")
    return "\n".join(lines) + "\n"
