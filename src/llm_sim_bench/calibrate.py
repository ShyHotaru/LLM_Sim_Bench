"""Fit efficiency factors to measurement records and report estimate errors.

Decode is fitted per backend as  1/tg = x / bandwidth_eff + layers * overhead,
where x is the time per token at 100 % of peak bandwidth. The fit minimizes
relative error so fast and slow devices count equally. Prefill efficiency is
the median of measured / peak-compute speed.
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass, replace

from .estimator import Efficiency, estimate
from .profiles import (APPLE_PRESETS, CPU, GPU_PRESETS, MODEL_PRESETS, RAM, System,
                       apple_system)

# Device used for records whose model fits fully in VRAM; it never runs layers.
_HOST = (CPU("host", 8, 1.0), RAM(128, 51.2))
_BACKEND_KEY = {"CUDA": "cuda", "Metal": "metal"}
# Peak hardware with no overhead: the fit's x values.
_UNIT = Efficiency(gpu_bandwidth=1, gpu_compute=1, gpu_layer_overhead_us=0,
                   metal_bandwidth=1, metal_compute=1, metal_layer_overhead_us=0)


@dataclass
class Point:
    id: str
    backend: str
    layers: int
    pp_meas: float | None
    tg_meas: float | None
    pp_pred: float | None = None
    tg_pred: float | None = None

    @property
    def pp_err(self) -> float | None:
        return _rel(self.pp_pred, self.pp_meas)

    @property
    def tg_err(self) -> float | None:
        return _rel(self.tg_pred, self.tg_meas)


def _rel(pred, meas):
    return None if pred is None or meas is None else pred / meas - 1


def system_for(rec: dict) -> System | None:
    preset = rec["hardware"]["gpu_preset"]
    if preset in GPU_PRESETS:
        return System(*_HOST, (GPU_PRESETS[preset],))
    if preset in APPLE_PRESETS:
        chip = APPLE_PRESETS[preset]
        return apple_system(chip, max(chip.memory_options_gb))
    return None


def usable(rec: dict) -> bool:
    """Single-GPU, fully offloaded, known presets, not flagged."""
    return (rec["software"]["backend"] in _BACKEND_KEY
            and rec["hardware"]["gpu_count"] == 1
            and system_for(rec) is not None
            and rec["model"]["name"] in MODEL_PRESETS
            and rec["run"]["n_gpu_layers"] >= MODEL_PRESETS[rec["model"]["name"]].n_layers
            and not rec["flags"])


def predict(rec: dict, eff: Efficiency) -> Point:
    model = MODEL_PRESETS[rec["model"]["name"]]
    run, res = rec["run"], rec["results"]
    est = estimate(system_for(rec), model, rec["model"]["quant"],
                   context=run["depth"] + max(run["n_prompt"], run["n_gen"]), eff=eff)
    gpu_only = all(p.is_gpu for p in est.placements)
    tg = None
    if gpu_only and run["n_gen"]:
        t = sum(1 / est.decode_tps_at(run["depth"] + i) for i in range(run["n_gen"]))
        tg = run["n_gen"] / t
    return Point(rec["id"], _BACKEND_KEY[rec["software"]["backend"]], model.n_layers,
                 res["pp_tps"], res["tg_tps"],
                 est.prefill_tps if gpu_only and run["n_prompt"] else None, tg)


def fit(records: list[dict]) -> Efficiency:
    """Return an Efficiency with CUDA and Metal factors fitted to the records."""
    points = [predict(r, _UNIT) for r in records if usable(r)]
    fitted = {}
    for backend in sorted({p.backend for p in points}):
        pts = [p for p in points if p.backend == backend]
        dec = [p for p in pts if p.tg_meas and p.tg_pred]
        inv_eff, overhead = _fit_decode([(1 / p.tg_pred, p.layers, 1 / p.tg_meas) for p in dec])
        pre = [p.pp_meas / p.pp_pred for p in pts if p.pp_meas and p.pp_pred]
        prefix = "gpu" if backend == "cuda" else backend
        fitted[f"{prefix}_bandwidth"] = round(1 / inv_eff, 3)
        fitted[f"{prefix}_layer_overhead_us"] = round(overhead * 1e6, 1)
        fitted[f"{prefix}_compute"] = round(statistics.median(pre), 3)
    return replace(Efficiency(), **fitted)


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
