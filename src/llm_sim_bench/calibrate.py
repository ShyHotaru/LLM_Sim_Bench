"""Fit efficiency factors to measurement records and report estimate errors.

Decode is fitted per backend as  1/tg = x / bandwidth_eff + layers * overhead,
where x is the time per token at 100 % of peak bandwidth. The fit minimizes
relative error so fast and slow devices count equally. Prefill efficiency is
the median of measured / peak-compute speed, per backend and, for prompt
processing, also per measured device (stored in device_calibration.json).
"""

from __future__ import annotations

import json
import statistics
from dataclasses import dataclass, replace

from .estimator import DEVICE_CALIBRATION, BACKENDS, Efficiency, backend_prefix, estimate
from .profiles import (CPU, CPU_PRESETS, GPU_PRESETS, MODEL_PRESETS, QUANT_BPW, RAM,
                       UNIFIED_PRESETS, System, ram_bandwidth_gbps, unified_system)

# Device used for records whose model fits fully in VRAM; it never runs layers.
_HOST = (CPU("host", 8, 1.0), RAM(128, 51.2))
_BACKEND_KEY = {"CUDA": "cuda", "Metal": "metal", "ROCm": "rocm", "Vulkan": "vulkan",
                "SYCL": "sycl", "CPU": "cpu"}
# Flags that exclude a record from calibration, or only one of its two results.
_EXCLUDE = {"suspect", "old_build", "multi-socket", "mobile-class-cpu", "cpu-model-inferred",
            "batched"}
_NO_PP = {"openblas-build"}  # OpenBLAS runs prompt batches: not llama.cpp's default CPU path
_NO_TG = {"tg_thread_limited"}
# Fine-tunes share their base model's architecture and therefore its speed.
MODEL_ALIASES = {"llama-3.1-tulu-3-8b": "llama-3.1-8b",
                 "deepseek-r1-distill-llama-8b": "llama-3.1-8b", "llama-3-8b": "llama-3.1-8b"}
# Peak hardware with no overhead: the fit's x values.
_UNIT = Efficiency(device_compute={}, device_bandwidth={},
                   **{f"{backend_prefix(b)}_{k}": v for b in BACKENDS
                      for k, v in (("bandwidth", 1), ("compute", 1), ("layer_overhead_us", 0))})


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


def system_for(rec: dict) -> System | None:
    hw = rec["hardware"]
    backend = _BACKEND_KEY.get(rec["software"]["backend"])
    if backend == "cpu":
        cpu = CPU_PRESETS.get(hw.get("cpu_preset"))
        # Capacity does not matter for speed; bandwidth may be unknown (prefill only).
        return System(cpu, RAM(1024, _ram_bandwidth(hw) or 1.0)) if cpu else None
    preset = hw["gpu_preset"]
    if preset in GPU_PRESETS:  # the record's backend, e.g. Vulkan on an NVIDIA card
        return System(*_HOST, (replace(GPU_PRESETS[preset], backend=backend),))
    if preset in UNIFIED_PRESETS:
        chip = replace(UNIFIED_PRESETS[preset], backend=backend)
        return unified_system(chip, max(chip.memory_options_gb))
    return None


def usable(rec: dict) -> bool:
    """Known presets and backend, not flagged; GPU runs single-GPU and fully offloaded,
    CPU runs single-socket with nothing offloaded."""
    backend = _BACKEND_KEY.get(rec["software"]["backend"])
    model = _model(rec)
    if (not backend or not model or model.active_params_b  # MoE routing is not modeled
            or rec["model"]["quant"] not in QUANT_BPW or _EXCLUDE & set(rec["flags"])
            or system_for(rec) is None):
        return False
    ngl = rec["run"]["n_gpu_layers"]
    if backend == "cpu":
        return ngl == 0 and rec["hardware"].get("sockets", 1) == 1
    # -1 is llama-bench's default ("auto": as many layers as fit); predict() then only
    # uses the record when the estimate places every layer on the GPU.
    return rec["hardware"]["gpu_count"] == 1 and (ngl == -1 or ngl >= model.n_layers)


def predict(rec: dict, eff: Efficiency) -> Point:
    model = _model(rec)
    backend = _BACKEND_KEY[rec["software"]["backend"]]
    run, res = rec["run"], rec["results"]
    n_prompt, n_gen = run["n_prompt"] or 0, run["n_gen"] or 0
    est = estimate(system_for(rec), model, rec["model"]["quant"],
                   context=run["depth"] + max(n_prompt, n_gen, 1), eff=eff)
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
                 est.prefill_tps if on_device and n_prompt else None, tg, device,
                 1 / tg if tg else None)


def fit(records: list[dict]) -> Efficiency:
    """Return an Efficiency with every measured backend's factors fitted to the records."""
    points = [predict(r, _UNIT) for r in records if usable(r)]
    fitted = {}
    for backend in sorted({p.backend for p in points}):
        pts = [p for p in points if p.backend == backend]
        prefix = backend_prefix(backend)
        dec = [p for p in pts if p.tg_meas and p.tg_pred]
        if dec:
            inv_eff, overhead = _fit_decode([(1 / p.tg_pred, p.layers, 1 / p.tg_meas)
                                             for p in dec])
            fitted[f"{prefix}_bandwidth"] = round(1 / inv_eff, 3)
            fitted[f"{prefix}_layer_overhead_us"] = round(overhead * 1e6, 1)
        pre = [p.pp_meas / p.pp_pred for p in pts if p.pp_meas and p.pp_pred]
        if pre:
            fitted[f"{prefix}_compute"] = round(statistics.median(pre), 3)
    by_device: dict[str, list[float]] = {}
    for p in points:
        if p.device and p.pp_meas and p.pp_pred:
            by_device.setdefault(p.device, []).append(p.pp_meas / p.pp_pred)
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
    return replace(Efficiency(), device_compute=devices, device_bandwidth=bandwidth, **fitted)


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
    return "\n".join(lines)
