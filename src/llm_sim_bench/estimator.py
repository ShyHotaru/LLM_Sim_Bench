"""First-order throughput model for single-stream local inference.

Decode is treated as memory-bandwidth bound: every generated token reads
all active weights plus the KV cache filled so far. Prefill is treated as
compute bound: roughly 2 FLOPs per active parameter per prompt token.
Each layer also costs a fixed per-token overhead (kernel launches and
synchronization), which dominates on fast GPUs with small models.
Layers are offloaded to GPUs first (like llama.cpp's n_gpu_layers) and the
rest stays in system RAM. Devices run layers sequentially, so per-token
times add up. On unified-memory systems (Apple Silicon) the GPU budget is a
fraction of system RAM instead of dedicated VRAM.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from .profiles import QUANT_BPW, Model, System


def add_in_order(values) -> float:
    """Add floats left to right, like the JavaScript port.

    The built-in sum() uses compensated summation since Python 3.12, so its last
    digit can differ between Python versions and from JavaScript; the generated
    golden files must not depend on the Python version.
    """
    total = 0.0
    for v in values:
        total += v
    return total


GB = 1e9
DEVICE_CALIBRATION = Path(__file__).with_name("device_calibration.json")


def _device(kind: str) -> dict[str, float]:
    """Per-device efficiency from measurements, "<preset>|<backend>" -> fraction."""
    return json.loads(DEVICE_CALIBRATION.read_text(encoding="utf-8"))[kind]


@dataclass(frozen=True)
class Efficiency:
    """Fraction of theoretical peak that real runtimes typically reach.

    Values are fitted to measured llama.cpp results with `llmsimbench calibrate`
    (Metal on 2026 builds only). Each backend has a bandwidth and compute
    efficiency and a per-layer overhead; "cuda" uses the gpu_* fields.
    """

    gpu_bandwidth: float = 0.84
    gpu_compute: float = 0.49
    gpu_layer_overhead_us: float = 36.0
    metal_bandwidth: float = 0.99
    metal_compute: float = 0.66
    metal_layer_overhead_us: float = 72.0
    rocm_bandwidth: float = 0.97
    rocm_compute: float = 0.31
    rocm_layer_overhead_us: float = 114.0
    vulkan_bandwidth: float = 0.80
    vulkan_compute: float = 0.40
    vulkan_layer_overhead_us: float = 22.0
    sycl_bandwidth: float = 0.85
    sycl_compute: float = 0.10
    sycl_layer_overhead_us: float = 58.0
    cpu_bandwidth: float = 0.79
    cpu_compute: float = 0.35
    cpu_layer_overhead_us: float = 119.0
    vram_reserve_gb: float = 1.0  # CUDA context + compute buffers, per GPU
    ram_reserve_gb: float = 4.0  # OS and other processes
    unified_gpu_fraction: float = 0.75  # share of unified memory the GPU may wire
    # Measured devices reach very different shares of peak compute in prompt processing
    # (data-center GPUs far less than consumer ones), so they override the backend value.
    device_compute: dict[str, float] = field(default_factory=lambda: _device("compute"))
    # CPUs differ in how much DRAM bandwidth reaches the cores (single-CCD Ryzen parts are
    # limited by their fabric link), so measured CPUs override cpu_bandwidth for decode.
    device_bandwidth: dict[str, float] = field(default_factory=lambda: _device("bandwidth"))

    def for_backend(self, backend: str) -> tuple[float, float, float]:
        """(bandwidth, compute, per-layer overhead in seconds) for a device backend."""
        p = backend_prefix(backend)
        return (getattr(self, f"{p}_bandwidth"), getattr(self, f"{p}_compute"),
                getattr(self, f"{p}_layer_overhead_us") / 1e6)


BACKENDS = ("cuda", "metal", "rocm", "vulkan", "sycl", "cpu")


def backend_prefix(backend: str) -> str:
    """Efficiency field prefix for a backend ("cuda" predates the others as "gpu")."""
    if backend not in BACKENDS:
        raise ValueError(f"Unknown backend {backend!r}. Known: {', '.join(BACKENDS)}")
    return "gpu" if backend == "cuda" else backend


@dataclass
class Placement:
    device: str
    layers: int
    weights_gb: float
    kv_gb: float
    bandwidth_gbps: float
    tflops: float
    is_gpu: bool
    overhead_s: float = 0.0  # fixed per-token time for this device's layers


@dataclass
class Estimate:
    model: Model
    quant: str
    context: int
    weights_gb: float
    kv_gb: float
    placements: list[Placement]
    fits: bool
    prefill_tps: float
    decode_tps_empty: float
    decode_tps_full: float
    notes: list[str] = field(default_factory=list)
    _kv_per_token: list[tuple[float, float]] = field(default_factory=list, repr=False)
    _base_time: float = 0.0

    def decode_tps_at(self, filled_tokens: int) -> float:
        """Decode speed when `filled_tokens` are already in the KV cache."""
        t = self._base_time + add_in_order(
            b * filled_tokens / bw for b, bw in self._kv_per_token)
        return 1.0 / t if t > 0 else float("inf")


def kv_bytes_per_token(model: Model, kv_bits: int = 16) -> float:
    return 2 * model.n_layers * model.n_kv_heads * model.head_dim * kv_bits / 8


def estimate(
    system: System,
    model: Model,
    quant: str = "Q4_K_M",
    context: int = 4096,
    kv_bits: int = 16,
    eff: Efficiency = Efficiency(),
) -> Estimate:
    if quant not in QUANT_BPW:
        raise ValueError(f"Unknown quant {quant!r}. Known: {', '.join(QUANT_BPW)}")
    notes: list[str] = []
    if context > model.max_context:
        notes.append(f"context {context} exceeds model max {model.max_context}")

    weights = model.params_b * 1e9 * QUANT_BPW[quant] / 8
    kv_tok = kv_bytes_per_token(model, kv_bits)
    kv_total = kv_tok * context
    layer_w = weights / model.n_layers
    layer_kv = kv_total / model.n_layers

    placements: list[Placement] = []
    remaining = model.n_layers
    gpu_used_gb = 0.0
    for gpu in system.gpus:
        frac = system.gpu_fraction or eff.unified_gpu_fraction
        vram = system.ram.size_gb * frac if system.unified else gpu.vram_gb
        usable = max(0.0, vram - eff.vram_reserve_gb) * GB
        n = min(remaining, int(usable // (layer_w + layer_kv)))
        if n > 0:
            bw_eff, c_eff, layer_s = eff.for_backend(gpu.backend)
            c_eff = eff.device_compute.get(f"{gpu.name}|{gpu.backend}", c_eff)
            placements.append(
                Placement(gpu.name, n, n * layer_w / GB, n * layer_kv / GB,
                          gpu.bandwidth_gbps * bw_eff, gpu.fp16_tflops * c_eff, True,
                          n * layer_s)
            )
            remaining -= n
            gpu_used_gb += n * (layer_w + layer_kv) / GB

    fits = True
    if remaining:
        cpu_gb = remaining * (layer_w + layer_kv) / GB
        bw_eff, c_eff, layer_s = eff.for_backend("cpu")
        bw_eff = eff.device_bandwidth.get(f"{system.cpu.name}|cpu", bw_eff)
        c_eff = eff.device_compute.get(f"{system.cpu.name}|cpu", c_eff)
        placements.append(
            Placement(f"cpu:{system.cpu.name}", remaining, remaining * layer_w / GB,
                      remaining * layer_kv / GB,
                      system.ram.bandwidth_gbps * bw_eff,
                      system.cpu.fp32_tflops * c_eff, False, remaining * layer_s)
        )
        ram_free = system.ram.size_gb - eff.ram_reserve_gb
        if system.unified:
            ram_free -= gpu_used_gb
        if cpu_gb > ram_free:
            fits = False
            notes.append(f"needs {cpu_gb:.1f} GB in RAM but only {ram_free:.1f} GB is free; "
                         "it would page to disk")
        if system.gpus:
            notes.append(f"{remaining}/{model.n_layers} layers offloaded to CPU")

    active_ratio = model.active_b / model.params_b
    base_time = 0.0
    kv_terms: list[tuple[float, float]] = []
    prefill_time = 0.0
    for p in placements:
        frac = p.layers / model.n_layers
        bw = p.bandwidth_gbps * GB
        base_time += p.weights_gb * GB * active_ratio / bw + p.overhead_s
        kv_terms.append((kv_tok * frac, bw))
        prefill_time += 2 * model.active_b * 1e9 * frac / (p.tflops * 1e12)

    est = Estimate(
        model=model, quant=quant, context=context,
        weights_gb=weights / GB, kv_gb=kv_total / GB,
        placements=placements, fits=fits,
        prefill_tps=1.0 / prefill_time,
        decode_tps_empty=0.0, decode_tps_full=0.0, notes=notes,
        _kv_per_token=kv_terms, _base_time=base_time,
    )
    est.decode_tps_empty = est.decode_tps_at(0)
    est.decode_tps_full = est.decode_tps_at(context)
    return est


def format_estimate(est: Estimate) -> str:
    m = est.model
    lines = [
        f"Model     {m.name} ({m.params_b:g}B"
        + (f", {m.active_b:g}B active" if m.active_params_b else "")
        + f") {est.quant}, context {est.context}",
        f"Memory    weights {est.weights_gb:.2f} GB + KV cache {est.kv_gb:.2f} GB"
        f" = {est.weights_gb + est.kv_gb:.2f} GB",
        "Placement",
    ]
    for p in est.placements:
        lines.append(f"  {p.device:<24} {p.layers:>3} layers  "
                     f"{p.weights_gb + p.kv_gb:6.2f} GB")
    lines += [
        f"Fits      {'yes' if est.fits else 'NO'}",
        f"Prefill   ~{est.prefill_tps:,.0f} tok/s",
        f"Decode    ~{est.decode_tps_empty:,.1f} tok/s (empty context)"
        f"  ->  ~{est.decode_tps_full:,.1f} tok/s (full context)",
    ]
    lines += [f"Note      {n}" for n in est.notes]
    return "\n".join(lines)
