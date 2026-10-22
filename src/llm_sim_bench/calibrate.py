"""Fit efficiency factors to measurement records and report estimate errors.

Decode is fitted per backend as  1/tg = x / bandwidth_eff + layers * overhead,
where x is the time per token at 100 % of peak bandwidth. The fit minimizes
relative error so fast and slow devices count equally. Prefill efficiency is
the median of measured / peak-compute speed, per backend and, for prompt
processing, also per measured device (stored in device_calibration.json).

Batched records (llama-batched-bench, several sequences at once) fit the
per-backend `*_batch_compute` factor instead: how the aggregate decode speed
grows from one sequence to B, so that errors in the single-request speed do not
enter the fit. Mixture-of-experts batched records are only evaluated.

llama-bench runs from llama.cpp's benches/ folder (`run.tool` set, several context
depths per model) fit the long-context factors from how each run slows down
relative to its own depth-0 result: `*_kv_read` from decode (every model) and
`*_attention` from prompt processing (dense models). Their depth-0 results of
mixture-of-experts models fit `*_moe_layer_overhead_us` (decode) and
`*_moe_compute` (prompt processing). They stay out of the fits above, whose
records use other test sizes.
"""

from __future__ import annotations

import json
import math
import statistics
from dataclasses import dataclass, replace

from .estimator import (DEVICE_CALIBRATION, BACKENDS, Efficiency, add_in_order, backend_prefix,
                        compute_field, estimate, without_flash_attention)
from .profiles import (CPU, CPU_PRESETS, GPU_PRESETS, MODEL_PRESETS, QUANT_BPW, RAM,
                       UNIFIED_PRESETS, System, ram_bandwidth_gbps, unified_system)

# Device used for records whose model fits fully in VRAM; it never runs layers.
_HOST = (CPU("host", 8, 1.0), RAM(128, 51.2))
_BACKEND_KEY = {"CUDA": "cuda", "Metal": "metal", "ROCm": "rocm", "Vulkan": "vulkan",
                "SYCL": "sycl", "CPU": "cpu"}
# Flags that exclude a record from calibration, or only one of its two results.
_EXCLUDE = {"suspect", "old_build", "multi-socket", "mobile-class-cpu", "cpu-model-inferred",
            "batched", "virtual-gpu"}
_NO_PP = {"openblas-build"}  # OpenBLAS runs prompt batches: not llama.cpp's default CPU path
_NO_TG = {"tg_thread_limited"}
# Fine-tunes share their base model's architecture and therefore its speed.
MODEL_ALIASES = {"llama-3.1-tulu-3-8b": "llama-3.1-8b",
                 "deepseek-r1-distill-llama-8b": "llama-3.1-8b", "llama-3-8b": "llama-3.1-8b",
                 "deepseek-r1-distill-llama-70b": "llama-3.1-70b",
                 "deepseek-r1-0528-qwen3-8b": "qwen3-8b",
                 "qwen2.5-coder-7b": "qwen2.5-7b", "qwen3-coder-30b-a3b-instruct": "qwen3-30b-a3b"}
# llama.cpp's benches/ does not say whether its M2 Ultra has 60 or 76 GPU cores. Both have
# 800 GB/s, and the depth and MoE fits use it only through memory bandwidth and speed
# ratios, which do not depend on the core count, so those fits use the 76-core preset.
_STAND_IN = {"M2 Ultra": "apple-m2-ultra-76c"}
# Peak hardware with no overhead: the fit's x values.
_UNIT = Efficiency(device_compute={}, device_bandwidth={},
                   **{f"{backend_prefix(b)}_{k}": v for b in BACKENDS
                      for k, v in (("bandwidth", 1), ("layer_overhead_us", 0))},
                   **{compute_field(b): 1 for b in BACKENDS},
                   **{f"{compute_field(b)}_no_fa": 1 for b in BACKENDS})


def backend_of(rec: dict) -> str | None:
    """Simulator backend of a record; SYCL records name their build (software.sycl_f16)."""
    backend = _BACKEND_KEY.get(rec["software"]["backend"])
    if backend == "sycl":
        f16 = rec["software"].get("sycl_f16")
        return None if f16 is None else "sycl-f16" if f16 else "sycl"
    return backend


@dataclass
class Point:
    id: str
    backend: str
    layers: int
    pp_meas: float | None
    tg_meas: float | None
    pp_pred: float | None = None
    tg_pred: float | None = None
    device: str = ""  # "<preset>|<backend>", e.g. "rtx4090-24gb|cuda" or "ryzen9-7950x|cpu"
    tg_x: float | None = None  # decode seconds per token at 100 % bandwidth (no overhead)
    no_fa: bool = False  # flash attention off (null in the record counts as llama.cpp's default)

    @property
    def pp_err(self) -> float | None:
        return _rel(self.pp_pred, self.pp_meas)

    @property
    def tg_err(self) -> float | None:
        return _rel(self.tg_pred, self.tg_meas)


def _rel(pred, meas):
    return None if pred is None or meas is None else pred / meas - 1


def _model(rec: dict):
    name = rec["model"]["name"]
    return MODEL_PRESETS.get(MODEL_ALIASES.get(name, name))


def _ram_bandwidth(hw: dict) -> float | None:
    """Bandwidth from the stated memory speed and channels, else None.

    Sources often give the DIMM count instead of channels; DIMMs are assumed to fill
    separate channels first (one per channel), up to the CPU's channel count.
    """
    channels, mts = hw.get("ram_channels"), hw.get("mem_mts")
    cpu = CPU_PRESETS.get(hw.get("cpu_preset"))
    if not channels and hw.get("dimms") and cpu and cpu.mem_channels:
        channels = min(hw["dimms"], cpu.mem_channels)
    if mts and channels:
        return mts * 8 * channels / 1000
    try:
        kind = hw.get("memory_type")
        return ram_bandwidth_gbps(kind, channels) if kind and channels else None
    except ValueError:
        return None


def system_for(rec: dict, stand_in: bool = False) -> System | None:
    """The record's system; with `stand_in`, also for devices in _STAND_IN."""
    hw = rec["hardware"]
    backend = backend_of(rec)
    if backend == "cpu":
        cpu = CPU_PRESETS.get(hw.get("cpu_preset"))
        # Capacity does not matter for speed; bandwidth may be unknown (prefill only).
        return System(cpu, RAM(1024, _ram_bandwidth(hw) or 1.0)) if cpu else None
    preset = hw["gpu_preset"] or (_STAND_IN.get(hw["gpu"]) if stand_in else None)
    if preset in GPU_PRESETS:  # the record's backend, e.g. Vulkan on an NVIDIA card
        # Several GPUs only with a layer split, which the estimator models.
        n = hw["gpu_count"] if rec["run"].get("split_mode") == "layer" else 1
        return System(*_HOST, (replace(GPU_PRESETS[preset], backend=backend),) * n)
    if preset in UNIFIED_PRESETS:
        chip = replace(UNIFIED_PRESETS[preset], backend=backend)
        return unified_system(chip, max(chip.memory_options_gb))
    return None


def usable(rec: dict) -> bool:
    """Known presets and backend, not flagged; GPU runs single-GPU and fully offloaded,
    CPU runs single-socket with nothing offloaded."""
    backend = backend_of(rec)
    model = _model(rec)
    if (not backend or not model or model.active_params_b  # MoE is fitted separately
            or rec["model"]["quant"] not in QUANT_BPW or _EXCLUDE & set(rec["flags"])
            or "tool" in rec["run"] or system_for(rec) is None):
        return False
    ngl = rec["run"]["n_gpu_layers"]
    if backend == "cpu":
        return ngl == 0 and rec["hardware"].get("sockets", 1) == 1
    # -1 is llama-bench's default ("auto": as many layers as fit) and null means not stated;
    # predict() then only uses the record when the estimate places every layer on GPUs.
    return (rec["hardware"]["gpu_count"] == 1 and "split_mode" not in rec["run"]
            and (ngl in (-1, None) or ngl >= model.n_layers))


def usable_multi_gpu(rec: dict) -> bool:
    """Runs split by layers over several GPUs of one preset: compared with the estimate,
    not fitted."""
    return (rec["run"].get("split_mode") == "layer" and rec["hardware"]["gpu_count"] > 1
            and bool(_model(rec)) and rec["model"]["quant"] in QUANT_BPW
            and not _EXCLUDE & set(rec["flags"]) and system_for(rec) is not None)


def multi_gpu_summary(records: list[dict], eff: Efficiency) -> dict[str, dict[str, float]]:
    """Per backend: n, median and max absolute error of multi-GPU decode and prompt speeds."""
    points = [predict(r, eff) for r in records if usable_multi_gpu(r)]
    out = {}
    for backend in sorted({p.backend for p in points}):
        for metric in ("tg", "pp"):
            errs = [abs(e) for p in points if p.backend == backend
                    for e in [getattr(p, f"{metric}_err")] if e is not None]
            if errs:
                out[f"{backend}/multi_{metric}"] = {"n": len(errs),
                                                    "median": statistics.median(errs),
                                                    "max": max(errs)}
    return out


def predict(rec: dict, eff: Efficiency) -> Point:
    model = _model(rec)
    backend = backend_of(rec)
    run, res = rec["run"], rec["results"]
    n_prompt, n_gen = run["n_prompt"] or 0, run["n_gen"] or 0
    no_fa = rec["software"]["flash_attn"] is False
    est = estimate(system_for(rec), model, rec["model"]["quant"],
                   context=run["depth"] + max(n_prompt, n_gen, 1),
                   eff=without_flash_attention(eff) if no_fa else eff)
    depth = run["depth"]
    on_device = backend == "cpu" or all(p.is_gpu for p in est.placements)
    tg = None
    if on_device and n_gen and (backend != "cpu" or _ram_bandwidth(rec["hardware"])):
        t = sum(1 / est.decode_tps_at(run["depth"] + i) for i in range(n_gen))
        tg = n_gen / t
    hw = rec["hardware"]
    device = f"{hw['cpu_preset'] if backend == 'cpu' else hw['gpu_preset']}|{backend}"
    flags = set(rec["flags"])
    return Point(rec["id"], backend, model.n_layers,
                 None if _NO_PP & flags else res["pp_tps"],
                 None if _NO_TG & flags else res["tg_tps"],
                 n_prompt / est.prefill_time(n_prompt, depth) if on_device and n_prompt
                 else None, tg, device,
                 1 / tg if tg else None, no_fa)


def fit(records: list[dict]) -> Efficiency:
    """Return an Efficiency with every measured backend's factors fitted to the records."""
    points = [predict(r, _UNIT) for r in records if usable(r)]
    fitted = {}
    # Decode per field prefix (both SYCL builds decode alike), prompt processing per backend.
    for prefix in sorted({backend_prefix(p.backend) for p in points}):
        dec = [p for p in points if backend_prefix(p.backend) == prefix and p.tg_meas
               and p.tg_pred]
        if dec:
            inv_eff, overhead = _fit_decode([(1 / p.tg_pred, p.layers, 1 / p.tg_meas)
                                             for p in dec])
            fitted[f"{prefix}_bandwidth"] = round(1 / inv_eff, 3)
            fitted[f"{prefix}_layer_overhead_us"] = round(overhead * 1e6, 1)
    # Prompt compute with flash attention on; runs without it count through their backend's
    # ratio, fitted to runs measured both ways.
    no_fa = _fa_ratios(records)
    fitted.update({f"{compute_field(b)}_no_fa": r for b, r in no_fa.items()})
    ratio = lambda p: p.pp_meas / p.pp_pred / (no_fa.get(p.backend, 1.0) if p.no_fa else 1)  # noqa: E731
    for backend in sorted({p.backend for p in points}):
        pre = [ratio(p) for p in points if p.backend == backend and p.pp_meas and p.pp_pred]
        if pre:
            fitted[compute_field(backend)] = round(statistics.median(pre), 3)
    by_device: dict[str, list[float]] = {}
    for p in points:
        if p.device and p.pp_meas and p.pp_pred:
            by_device.setdefault(p.device, []).append(ratio(p))
    devices = {d: round(statistics.median(v), 3) for d, v in sorted(by_device.items())}
    # CPUs: bandwidth efficiency that reproduces each measured decode speed, given the
    # fitted per-layer overhead.
    overhead = fitted.get("cpu_layer_overhead_us", Efficiency().cpu_layer_overhead_us) / 1e6
    by_cpu: dict[str, list[float]] = {}
    for p in points:
        if p.backend == "cpu" and p.tg_meas and p.tg_x:
            busy = 1 / p.tg_meas - p.layers * overhead
            if busy > 0:
                by_cpu.setdefault(p.device, []).append(p.tg_x / busy)
    bandwidth = {d: round(statistics.median(v), 3) for d, v in sorted(by_cpu.items())}
    base = replace(Efficiency(), device_compute=devices, device_bandwidth=bandwidth, **fitted)
    base = replace(base, **fit_depth(records, base))
    batch = {}
    for backend in sorted({p.backend for p in batch_points(records, base)
                           if not p.moe and p.pred}):
        batch[f"{backend_prefix(backend)}_batch_compute"] = _fit_batch(records, backend, base)
    return replace(base, **batch)


def _p80(values: list[float]) -> float:
    """80th percentile (nearest rank)."""
    v = sorted(values)
    return v[max(0, math.ceil(0.8 * len(v)) - 1)]


def error_bands(records: list[dict], eff: Efficiency, min_n: int = 5) -> dict[str, dict]:
    """Per backend: the error 8 of 10 results stay within, judged on results not used to
    fit them. "pp_measured": prompt speed of a device with its own factor, each result left
    out of that factor; "pp_other": a device without one, its results left out of the
    backend factor. Decode has no per-device GPU factors; "tg_other" drops CPUs' own memory
    factors. A band needs min_n results."""
    no_fa = {b: getattr(eff, f"{compute_field(b)}_no_fa") for b in BACKENDS}
    points = [predict(r, _UNIT) for r in records if usable(r)]
    x = [(p.backend, p.device, p.pp_meas / p.pp_pred / (no_fa[p.backend] if p.no_fa else 1))
         for p in points if p.pp_meas and p.pp_pred]
    errs: dict[str, dict[str, list[float]]] = {}
    for i, (b, d, v) in enumerate(x):
        same = [w for j, (_, dd, w) in enumerate(x) if dd == d and j != i]
        rest = [w for bb, dd, w in x if bb == b and dd != d]
        e = errs.setdefault(b, {})
        if same:
            e.setdefault("pp_measured", []).append(abs(v / statistics.median(same) - 1))
        if rest:
            e.setdefault("pp_other", []).append(abs(v / statistics.median(rest) - 1))
    for key, e2 in (("tg_measured", eff), ("tg_other", replace(eff, device_bandwidth={}))):
        for p in evaluate(records, e2):
            if p.tg_err is not None:
                errs.setdefault(p.backend, {}).setdefault(key, []).append(abs(p.tg_err))
    return {b: {k: round(_p80(v), 3) if len(v) >= min_n else None
                for k, v in sorted(e.items())} for b, e in sorted(errs.items())}


def _fa_ratios(records: list[dict]) -> dict[str, float]:
    """Per backend: median prompt speed with flash attention off over on, from runs that
    differ only in it (same source, device, build, model, quantization and host)."""
    key = lambda r: (r["source"]["url"], r["hardware"]["gpu"], r["hardware"].get("cpu"),  # noqa: E731
                     r["software"]["commit"], backend_of(r), r["model"]["name"],
                     r["model"]["quant"], r["run"]["n_prompt"])
    runs: dict[tuple, dict[bool, float]] = {}
    for r in records:
        fa, pp = r["software"]["flash_attn"], r["results"]["pp_tps"]
        if usable(r) and fa is not None and pp and not _NO_PP & set(r["flags"]):
            runs.setdefault(key(r), {})[fa] = pp
    by_backend: dict[str, list[float]] = {}
    for k, v in runs.items():
        if True in v and False in v:
            by_backend.setdefault(k[4], []).append(v[False] / v[True])
    return {b: round(statistics.median(v), 2) for b, v in sorted(by_backend.items())}


@dataclass
class DepthPoint:
    """A benches/ llama-bench result: the speed at a depth (or of a MoE model at depth 0)."""

    id: str
    backend: str
    metric: str  # "tg", "pp" (slowdown with depth), "moe_tg", "moe_pp" (depth 0)
    moe: bool
    depth: int
    meas: float
    pred: float | None = None

    @property
    def err(self) -> float | None:
        return _rel(self.pred, self.meas)


def depth_series(records: list[dict]) -> list[tuple[dict, list[dict]]]:
    """(depth-0 record, deeper records) of each llama-bench run at several depths whose
    model and device are known and that fits on the GPUs (several only split by layers)."""
    recs = [r for r in records if r["run"].get("tool") == "llama-bench"
            and not _EXCLUDE & set(r["flags"]) and _model(r)
            and r["model"]["quant"] in QUANT_BPW
            and (r["hardware"]["gpu_count"] == 1 or r["run"].get("split_mode") == "layer")
            and system_for(r, stand_in=True) is not None]
    key = lambda r: (r["source"]["url"], r["model"]["name"], r["model"]["quant"])  # noqa: E731
    out = []
    for r in recs:
        if r["run"]["depth"] == 0:
            deeper = sorted((d for d in recs if key(d) == key(r) and d["run"]["depth"]),
                            key=lambda d: d["run"]["depth"])
            est = estimate(system_for(r, stand_in=True), _model(r), r["model"]["quant"],
                           context=deeper[-1]["run"]["depth"] + 4096 if deeper else 4096)
            if all(p.is_gpu for p in est.placements):
                out.append((r, deeper))
    return out


def _bench_tg(rec: dict, eff: Efficiency) -> float:
    run = rec["run"]
    est = estimate(system_for(rec, stand_in=True), _model(rec), rec["model"]["quant"],
                   context=run["depth"] + run["n_gen"], eff=eff)
    return run["n_gen"] / add_in_order(1 / est.decode_tps_at(run["depth"] + i)
                                       for i in range(run["n_gen"]))


def _bench_pp_time(rec: dict, eff: Efficiency) -> float:
    run = rec["run"]
    est = estimate(system_for(rec, stand_in=True), _model(rec), rec["model"]["quant"],
                   context=run["depth"] + run["n_prompt"], eff=eff)
    return est.prefill_time(run["n_prompt"], run["depth"])


def depth_points(records: list[dict], eff: Efficiency) -> list[DepthPoint]:
    """Speeds at depth predicted from the measured depth-0 speed and the modeled slowdown,
    and MoE depth-0 speeds predicted outright (prompt processing only with a preset)."""
    points = []
    for zero, deeper in depth_series(records):
        backend = backend_of(zero)
        moe = bool(_model(zero).active_params_b)
        res0 = zero["results"]
        if moe and res0["tg_tps"]:
            points.append(DepthPoint(zero["id"], backend, "moe_tg", True, 0, res0["tg_tps"],
                                     _bench_tg(zero, eff)))
        if moe and res0["pp_tps"] and zero["hardware"]["gpu_preset"]:
            points.append(DepthPoint(zero["id"], backend, "moe_pp", True, 0, res0["pp_tps"],
                                     zero["run"]["n_prompt"] / _bench_pp_time(zero, eff)))
        tg0, pp0 = _bench_tg(zero, eff), _bench_pp_time(zero, eff)
        for r in deeper:
            res, depth = r["results"], r["run"]["depth"]
            if res["tg_tps"] and res0["tg_tps"]:
                slower = 1 / _bench_tg(r, eff) - 1 / tg0
                points.append(DepthPoint(r["id"], backend, "tg", moe, depth, res["tg_tps"],
                                         1 / (1 / res0["tg_tps"] + slower)))
            if res["pp_tps"] and res0["pp_tps"]:
                ratio = _bench_pp_time(r, eff) / pp0
                points.append(DepthPoint(r["id"], backend, "pp", moe, depth, res["pp_tps"],
                                         res0["pp_tps"] / ratio))
    return points


def _grid_fit(records: list[dict], eff: Efficiency, field: str, values, backend: str,
              metric: str, dense_only: bool = False) -> float:
    """The value of `field` minimizing the squared log error of the backend's points."""
    records = [r for r in records if r["run"].get("tool") == "llama-bench"
               and backend_of(r) == backend]
    best = None
    for v in values:
        pts = [p for p in depth_points(records, replace(eff, **{field: v}))
               if p.backend == backend and p.metric == metric and p.pred
               and not (dense_only and p.moe)]
        loss = sum(math.log(p.pred / p.meas) ** 2 for p in pts)
        if best is None or loss < best[0]:
            best = (loss, v)
    return best[1]


def fit_depth(records: list[dict], eff: Efficiency) -> dict[str, float]:
    """Long-context factors, then MoE factors (given those), for every backend with
    benches/ measurements."""
    fitted: dict[str, float] = {}
    for step in ("depth", "moe"):
        points = depth_points(records, eff)
        for backend in sorted({p.backend for p in points}):
            prefix = backend_prefix(backend)
            has = {(p.metric, p.moe) for p in points if p.backend == backend}
            if step == "depth" and ({("tg", False), ("tg", True)} & has):
                fitted[f"{prefix}_kv_read"] = _grid_fit(
                    records, eff, f"{prefix}_kv_read", [k / 100 for k in range(50, 501)],
                    backend, "tg")
            if step == "depth" and ("pp", False) in has:
                fitted[f"{prefix}_attention"] = _grid_fit(
                    records, eff, f"{prefix}_attention", [k / 100 for k in range(5, 301)],
                    backend, "pp", dense_only=True)
            if step == "moe" and ("moe_tg", True) in has:
                fitted[f"{prefix}_moe_layer_overhead_us"] = float(_grid_fit(
                    records, eff, f"{prefix}_moe_layer_overhead_us", range(0, 501), backend,
                    "moe_tg"))
            if step == "moe" and ("moe_pp", True) in has:
                fitted[f"{prefix}_moe_compute"] = _grid_fit(
                    records, eff, f"{prefix}_moe_compute", [k / 100 for k in range(5, 201)],
                    backend, "moe_pp")
        eff = replace(eff, **fitted)
    return fitted


def depth_summary(points: list[DepthPoint]) -> dict[str, dict[str, float]]:
    """Per backend and metric: n, median and max absolute error. Prompt slowdown with
    depth counts dense models only, as in the fit."""
    out = {}
    names = {"tg": "depth_tg", "pp": "depth_pp", "moe_tg": "moe_tg", "moe_pp": "moe_pp"}
    for backend in sorted({p.backend for p in points}):
        for metric, name in names.items():
            errs = [abs(p.err) for p in points if p.backend == backend and p.metric == metric
                    and p.err is not None and not (metric == "pp" and p.moe)]
            if errs:
                out[f"{backend}/{name}"] = {"n": len(errs), "median": statistics.median(errs),
                                            "max": max(errs)}
    return out


@dataclass
class BatchPoint:
    """A batched result with B > 1, as the speed-up of aggregate decode over B = 1."""

    id: str
    backend: str
    device: str
    n: int
    moe: bool
    meas: float  # measured S_TG(B) / S_TG(1)
    pred: float | None = None

    @property
    def err(self) -> float | None:
        return _rel(self.pred, self.meas)


def usable_batched(rec: dict) -> bool:
    """llama-batched-bench runs on a known single GPU or chip, fully offloaded, each
    sequence with its own prompt."""
    backend = backend_of(rec)
    model = _model(rec)
    run = rec["run"]
    if ("batched" not in rec["flags"] or _EXCLUDE - {"batched"} & set(rec["flags"])
            or backend in (None, "cpu") or not model or rec["model"]["quant"] not in QUANT_BPW
            or run.get("tool") != "llama-batched-bench" or run.get("pp_shared")
            or not run["n_gen"] or not rec["results"]["tg_tps"] or system_for(rec) is None):
        return False
    ngl = run["n_gpu_layers"]
    return rec["hardware"]["gpu_count"] == 1 and (ngl == -1 or ngl >= model.n_layers)


def batched_tg(rec: dict, eff: Efficiency, n: int) -> float | None:
    """Simulated S_TG of the record's run with n sequences, or None if not all on the GPU."""
    run = rec["run"]
    est = estimate(system_for(rec), _model(rec), rec["model"]["quant"],
                   context=run["n_prompt"] + run["n_gen"], eff=eff, parallel=n)
    if not all(p.is_gpu for p in est.placements):
        return None
    t = sum(est.step_time([run["n_prompt"] + i] * n) for i in range(run["n_gen"]))
    return n * run["n_gen"] / t


def batch_points(records: list[dict], eff: Efficiency) -> list[BatchPoint]:
    """Every usable batched result with B > 1 whose run also has a B = 1 result."""
    recs = [r for r in records if usable_batched(r)]
    key = lambda r: (r["source"]["url"], r["model"]["name"], r["model"]["quant"],  # noqa: E731
                     r["run"]["n_prompt"], r["run"]["n_gen"])
    single = {key(r): r for r in recs if r["run"]["n_parallel"] == 1}
    points = []
    for r in recs:
        n, one = r["run"]["n_parallel"], single.get(key(r))
        if n == 1 or one is None:
            continue
        backend = backend_of(r)
        p = BatchPoint(r["id"], backend, f"{r['hardware']['gpu_preset']}|{backend}", n,
                       bool(_model(r).active_params_b),
                       r["results"]["tg_tps"] / one["results"]["tg_tps"])
        tg_n, tg_1 = batched_tg(r, eff, n), batched_tg(r, eff, 1)
        if tg_n and tg_1:
            p.pred = tg_n / tg_1
        points.append(p)
    return points


def _fit_batch(records: list[dict], backend: str, eff: Efficiency) -> float:
    """batch_compute minimizing the squared log error of the speed-ups (dense models)."""
    field = f"{backend_prefix(backend)}_batch_compute"
    records = [r for r in records if usable_batched(r) and not _model(r).active_params_b
               and backend_of(r) == backend]
    best = None
    for c in (k / 100 for k in range(5, 301)):
        pts = [p for p in batch_points(records, replace(eff, **{field: c})) if p.pred]
        loss = sum(math.log(p.pred / p.meas) ** 2 for p in pts)
        if best is None or loss < best[0]:
            best = (loss, c)
    return best[1]


def batch_summary(points: list[BatchPoint]) -> dict[str, dict[str, float]]:
    """Per backend: n, median and max absolute error of the dense-model speed-ups."""
    out = {}
    for backend in sorted({p.backend for p in points}):
        errs = [abs(p.err) for p in points if p.backend == backend and not p.moe
                and p.err is not None]
        if errs:
            out[f"{backend}/batch"] = {"n": len(errs), "median": statistics.median(errs),
                                       "max": max(errs)}
    return out


def batch_devices(points: list[BatchPoint]) -> list[str]:
    """"<preset>|<backend>" of the devices whose batched results calibrate the model."""
    return sorted({p.device for p in points if not p.moe and p.err is not None})


def write_device_calibration(eff: Efficiency) -> None:
    data = {"compute": eff.device_compute, "bandwidth": eff.device_bandwidth}
    DEVICE_CALIBRATION.write_text(json.dumps(data, indent=1) + "\n", encoding="utf-8")


def _fit_decode(rows: list[tuple[float, int, float]]) -> tuple[float, float]:
    """Least squares for y = a*x + b*L, weighted by 1/y^2 (relative error)."""
    sxx = sxl = sll = sxy = sly = 0.0
    for x, layers, y in rows:
        w = 1 / (y * y)
        sxx += w * x * x
        sxl += w * x * layers
        sll += w * layers * layers
        sxy += w * x * y
        sly += w * layers * y
    det = sxx * sll - sxl * sxl
    if abs(det) < 1e-30:  # every record has the same layer count: fall back to ratio
        a = sxy / sxx
        return a, 0.0
    a = (sxy * sll - sly * sxl) / det
    b = (sly * sxx - sxy * sxl) / det
    if b < 0:  # negative overhead is unphysical
        return sxy / sxx, 0.0
    return a, b


def evaluate(records: list[dict], eff: Efficiency = Efficiency()) -> list[Point]:
    return [predict(r, eff) for r in records if usable(r)]


def summary(points: list[Point]) -> dict[str, dict[str, float]]:
    """Per backend and metric: n, median and max absolute relative error."""
    out = {}
    for backend in sorted({p.backend for p in points}):
        for metric in ("pp", "tg"):
            errs = [abs(e) for p in points if p.backend == backend
                    for e in [getattr(p, f"{metric}_err")] if e is not None]
            if errs:
                out[f"{backend}/{metric}"] = {"n": len(errs), "median": statistics.median(errs),
                                              "max": max(errs)}
    return out


def format_report(records: list[dict], eff: Efficiency) -> str:
    points = evaluate(records, eff)
    lines = [f"{len(points)} of {len(records)} records usable for calibration",
             f"{'backend/metric':<16}{'n':>5}{'median err':>12}{'max err':>10}"]
    for key, s in summary(points).items():
        lines.append(f"{key:<16}{s['n']:>5}{s['median']:>11.1%}{s['max']:>10.1%}")
    worst = sorted((p for p in points if p.tg_err is not None),
                   key=lambda p: -abs(p.tg_err))[:5]
    lines.append("Largest decode errors:")
    lines += [f"  {p.id:<44} pred {p.tg_pred:7.1f}  meas {p.tg_meas:7.1f}  {p.tg_err:+.0%}"
              for p in worst]
    lines.append("Several GPUs split by layers (not fitted):")
    for key, s in multi_gpu_summary(records, eff).items():
        lines.append(f"  {key:<20}{s['n']:>5}{s['median']:>11.1%}{s['max']:>10.1%}")
    lines.append("Long contexts and mixture-of-experts models (llama-bench in benches/):")
    for key, s in depth_summary(depth_points(records, eff)).items():
        lines.append(f"  {key:<20}{s['n']:>5}{s['median']:>11.1%}{s['max']:>10.1%}")
    batch = batch_points(records, eff)
    lines.append("Concurrent decode, speed-up of B sequences over one (llama-batched-bench):")
    for moe in (False, True):
        for backend in sorted({p.backend for p in batch}):
            errs = [abs(p.err) for p in batch if p.backend == backend and p.moe == moe
                    and p.err is not None]
            if errs:
                lines.append(f"  {backend + (' MoE (not fitted)' if moe else ''):<22}"
                             f"{len(errs):>5}{statistics.median(errs):>11.1%}{max(errs):>10.1%}")
    return "\n".join(lines)
