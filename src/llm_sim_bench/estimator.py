"""First-order throughput model for single-stream local inference.

Decode is treated as memory-bandwidth bound: every generated token reads
all active weights plus the KV cache filled so far. Prefill is treated as
compute bound: roughly 2 FLOPs per active parameter per prompt token.
Layers are offloaded to GPUs first (like llama.cpp's n_gpu_layers) and the
rest stays in system RAM. Devices run layers sequentially, so per-token
times add up.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .profiles import QUANT_BPW, Model, System

GB = 1e9


@dataclass(frozen=True)
class Efficiency:
    """Fraction of theoretical peak that real runtimes typically reach."""

    gpu_bandwidth: float = 0.75
    cpu_bandwidth: float = 0.60
    gpu_compute: float = 0.50
    cpu_compute: float = 0.40
    vram_reserve_gb: float = 1.0  # CUDA context + compute buffers, per GPU
    ram_reserve_gb: float = 4.0  # OS and other processes


@dataclass
class Placement:
    device: str
    layers: int
    weights_gb: float
    kv_gb: float
    bandwidth_gbps: float
    tflops: float
    is_gpu: bool


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
        t = self._base_time + sum(b * filled_tokens / bw for b, bw in self._kv_per_token)
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
    for gpu in system.gpus:
        usable = max(0.0, gpu.vram_gb - eff.vram_reserve_gb) * GB
        n = min(remaining, int(usable // (layer_w + layer_kv)))
        if n > 0:
            placements.append(
                Placement(gpu.name, n, n * layer_w / GB, n * layer_kv / GB,
                          gpu.bandwidth_gbps * eff.gpu_bandwidth,
                          gpu.fp16_tflops * eff.gpu_compute, True)
            )
            remaining -= n

    fits = True
    if remaining:
        cpu_gb = remaining * (layer_w + layer_kv) / GB
        placements.append(
            Placement(f"cpu:{system.cpu.name}", remaining, remaining * layer_w / GB,
                      remaining * layer_kv / GB,
                      system.ram.bandwidth_gbps * eff.cpu_bandwidth,
                      system.cpu.fp32_tflops * eff.cpu_compute, False)
        )
        ram_free = system.ram.size_gb - eff.ram_reserve_gb
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
        base_time += p.weights_gb * GB * active_ratio / bw
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
