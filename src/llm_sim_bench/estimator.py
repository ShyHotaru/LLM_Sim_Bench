"""First-order throughput model for local inference.

Decode is treated as memory-bandwidth bound: every generated token reads
all active weights plus the KV cache filled so far (attention reads the cache
more slowly than the weights, `*_kv_read`). Prefill is treated as compute
bound: roughly 2 FLOPs per active parameter per prompt token, plus attention
over the tokens already in the cache, which grows with the context
(`*_attention`). Sliding-window attention layers only see their window. Each
layer also costs a fixed per-token overhead (kernel launches and
synchronization), which dominates on fast GPUs with small models; layers of a
mixture-of-experts model cost more (`*_moe_layer_overhead_us`).
Layers are offloaded to GPUs first (like llama.cpp's n_gpu_layers) and the
rest stays in system RAM. Devices run layers sequentially, so per-token
times add up. On unified-memory systems (Apple Silicon) the GPU budget is a
fraction of system RAM instead of dedicated VRAM.

Concurrent requests (llama-server --parallel slots with continuous batching)
share one decode step: the weights are read once for the whole batch (for
mixture-of-experts models, the union of the experts the batch is routed to),
each request reads its own KV cache, and every request beyond the first adds
compute at a fraction of the prompt-processing rate (`*_batch_compute`).
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

from .profiles import QUANT_BPW, QUANT_SHARED_BPW, Model, System


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
# llama.cpp keeps n_swa + n_ubatch tokens in a sliding-window layer's cache (default -ub 512;
# it also pads to 256), unless --swa-full: llama-kv-cache-iswa.cpp.
SWA_UBATCH = 512
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
    metal_bandwidth: float = 0.97
    metal_compute: float = 0.66
    metal_layer_overhead_us: float = 67.0
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
    # Long contexts, fitted to llama-bench runs at several depths (llama.cpp benches/):
    # decode reads the KV cache *_kv_read times slower than weights, and prompt attention
    # runs at *_attention times the prompt-processing FLOP rate. Backends without depth
    # measurements keep 1.0.
    gpu_kv_read: float = 1.21
    gpu_attention: float = 1.61
    metal_kv_read: float = 2.30
    metal_attention: float = 0.53
    rocm_kv_read: float = 1.0
    rocm_attention: float = 1.0
    vulkan_kv_read: float = 1.0
    vulkan_attention: float = 1.0
    sycl_kv_read: float = 1.0
    sycl_attention: float = 1.0
    cpu_kv_read: float = 1.0
    cpu_attention: float = 1.0
    # Mixture-of-experts models, fitted to the same runs: extra per-token cost of a layer
    # (routing, gathering the experts; 0 without measurements) and the share of the
    # prompt-processing FLOP rate their weights reach (1.0 without measurements).
    gpu_moe_layer_overhead_us: float = 0.0
    metal_moe_layer_overhead_us: float = 117.0
    rocm_moe_layer_overhead_us: float = 0.0
    vulkan_moe_layer_overhead_us: float = 0.0
    sycl_moe_layer_overhead_us: float = 0.0
    cpu_moe_layer_overhead_us: float = 0.0
    gpu_moe_compute: float = 0.56
    metal_moe_compute: float = 1.0
    rocm_moe_compute: float = 1.0
    vulkan_moe_compute: float = 1.0
    sycl_moe_compute: float = 1.0
    cpu_moe_compute: float = 1.0
    # Share of the prompt-processing compute rate that each request beyond the first
    # reaches in a decode batch. Fitted for CUDA to llama-batched-bench results (DGX
    # Spark); the other backends use the CUDA value until they have measurements.
    gpu_batch_compute: float = 0.49
    metal_batch_compute: float = 0.49
    rocm_batch_compute: float = 0.49
    vulkan_batch_compute: float = 0.49
    sycl_batch_compute: float = 0.49
    cpu_batch_compute: float = 0.49
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

    def batch_compute(self, backend: str) -> float:
        return getattr(self, f"{backend_prefix(backend)}_batch_compute")

    def for_depth(self, backend: str) -> tuple[float, float]:
        """(KV read cost, attention rate) for long contexts."""
        p = backend_prefix(backend)
        return getattr(self, f"{p}_kv_read"), getattr(self, f"{p}_attention")

    def for_moe(self, backend: str) -> tuple[float, float]:
        """(extra per-layer overhead in seconds, prompt compute share) of MoE models."""
        p = backend_prefix(backend)
        return (getattr(self, f"{p}_moe_layer_overhead_us") / 1e6,
                getattr(self, f"{p}_moe_compute"))


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
    batch_s: float = 0.0  # compute time each extra request adds to a decode step


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
    parallel: int = 1  # request slots; each gets `context` tokens of KV cache
    # Per device: (KV bytes per token of full-attention layers, of sliding-window layers,
    # effective bandwidth for reading the cache in bytes/s).
    _kv_per_token: list[tuple[float, float, float]] = field(default_factory=list, repr=False)
    # Per device: (attention FLOPs per prompt token and cached token, full layers;
    # sliding-window layers; attention FLOP/s).
    _attention: list[tuple[float, float, float]] = field(default_factory=list, repr=False)
    _base_time: float = 0.0
    _overhead: float = 0.0  # per-step fixed time of all devices
    _weight_read: float = 0.0  # time to read every weight once (all experts)
    _batch_time: float = 0.0  # compute time each extra request adds to a step

    def _kv_time(self, filled: int) -> float:
        """Seconds to read the KV cache of one request holding `filled` tokens."""
        window = min(filled, swa_cells(self.model))
        return add_in_order((full * filled + swa * window) / bw
                            for full, swa, bw in self._kv_per_token)

    def decode_tps_at(self, filled_tokens: int) -> float:
        """Decode speed of a single request when `filled_tokens` are in its KV cache."""
        t = self._base_time + self._kv_time(filled_tokens)
        return 1.0 / t if t > 0 else float("inf")

    def prefill_time(self, tokens: int, depth: int = 0) -> float:
        """Seconds to process a prompt of `tokens` tokens after `depth` cached tokens.

        Token i attends to depth + i + 1 tokens, a sliding-window layer to at most the
        tokens its cache keeps, so prompts get slower as the context fills. `prefill_tps`
        leaves out attention: it is the speed of a short prompt in an empty context.
        """
        full = context_sum(depth + 1, depth + tokens)
        swa = context_sum(depth + 1, depth + tokens, swa_cells(self.model))
        return tokens / self.prefill_tps + add_in_order(
            (f * full + w * swa) / rate for f, w, rate in self._attention)

    def step_time(self, fills: Sequence[int]) -> float:
        """Seconds for one decode step that gives each active request one token.

        `fills` holds the KV fill of every active request. With one request this
        equals 1 / decode_tps_at(fill) up to rounding.
        """
        n = len(fills)
        t = (self._overhead + self._weight_read * weight_read_ratio(self.model, n, self.quant)
             + (n - 1) * self._batch_time)
        for f in fills:
            t += self._kv_time(f)
        return t

    def parallel_tps(self, n: int, filled_tokens: int) -> float:
        """Decode speed of each of `n` concurrent requests with `filled_tokens` in the cache.

        The aggregate throughput is n times this.
        """
        return 1.0 / self.step_time([filled_tokens] * n)


def context_sum(first: int, last: int, cap: int = 0) -> float:
    """Sum of min(c, cap) for c = first..last (no cap if cap is 0); 0 if last < first."""
    if last < first:
        return 0.0
    if not cap or last <= cap:
        return (first + last) * (last - first + 1) / 2
    if first >= cap:
        return cap * (last - first + 1)
    return (first + cap) * (cap - first + 1) / 2 + cap * (last - cap)


def weight_bytes(model: Model, quant: str) -> tuple[float, float]:
    """(bytes of the shared weights, bytes of all routed experts); dense models have no
    experts. Some quantizations apply to the experts only (QUANT_SHARED_BPW)."""
    bpw = QUANT_BPW[quant]
    if not model.n_experts:
        return model.params_b * 1e9 * bpw / 8, 0.0
    share = model.experts_per_token / model.n_experts
    expert = (model.params_b - model.active_b) / (1 - share)  # B params in all experts
    shared = model.params_b - expert
    return shared * 1e9 * QUANT_SHARED_BPW.get(quant, bpw) / 8, expert * 1e9 * bpw / 8


def weight_read_ratio(model: Model, n: int, quant: str = "Q4_K_M") -> float:
    """Share of the weight bytes read in a decode step for `n` requests.

    Dense models read everything once. A mixture-of-experts model reads its shared
    weights and the experts any request is routed to; with uniform independent
    routing to k of E experts, a step touches 1 - (1 - k/E)^n of them.
    """
    if not model.n_experts:
        return model.active_b / model.params_b
    shared, expert = weight_bytes(model, quant)
    share = model.experts_per_token / model.n_experts
    touched = share if n <= 1 else 1 - (1 - share) ** n
    return (shared + expert * touched) / (shared + expert)


def kv_bytes_per_token(model: Model, kv_bits: int = 16) -> float:
    """KV cache bytes per token if every layer kept the whole context."""
    return 2 * model.n_layers * model.n_kv_heads * model.head_dim * kv_bits / 8


def swa_cells(model: Model) -> int:
    """Tokens a sliding-window layer keeps in its cache, reads and attends to; 0 if none."""
    return model.sliding_window + SWA_UBATCH if model.sliding_window else 0


def attention_flops(model: Model) -> float:
    """FLOPs per layer for one token attending to one cached token (QK^T and AV)."""
    return 4 * (model.n_heads or model.n_kv_heads) * model.head_dim


def estimate(
    system: System,
    model: Model,
    quant: str = "Q4_K_M",
    context: int = 4096,
    kv_bits: int = 16,
    eff: Efficiency = Efficiency(),
    parallel: int = 1,
) -> Estimate:
    """Estimate memory and speed; `parallel` request slots each get `context` tokens of
    KV cache (llama-server -np N -c N*context)."""
    if quant not in QUANT_BPW:
        raise ValueError(f"Unknown quant {quant!r}. Known: {', '.join(QUANT_BPW)}")
    if parallel < 1:
        raise ValueError("parallel must be at least 1")
    notes: list[str] = []
    if context > model.max_context:
        notes.append(f"context {context} exceeds model max {model.max_context}")

    weights = add_in_order(weight_bytes(model, quant))
    kv_tok = kv_bytes_per_token(model, kv_bits)
    # A sliding-window layer's cache keeps only about its window.
    swa_share = model.swa_layers / model.n_layers
    kv_total = kv_tok * (context * (1 - swa_share)
                         + min(context, swa_cells(model)) * swa_share) * parallel
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
            if model.n_experts:
                layer_s += eff.for_moe(gpu.backend)[0]
            c_eff = eff.device_compute.get(f"{gpu.name}|{gpu.backend}", c_eff)
            tflops = gpu.fp16_tflops * c_eff
            placements.append(
                Placement(gpu.name, n, n * layer_w / GB, n * layer_kv / GB,
                          gpu.bandwidth_gbps * bw_eff, tflops, True, n * layer_s,
                          _flop_time(model, n, tflops) / eff.batch_compute(gpu.backend))
            )
            remaining -= n
            gpu_used_gb += n * (layer_w + layer_kv) / GB

    fits = True
    if remaining:
        cpu_gb = remaining * (layer_w + layer_kv) / GB
        bw_eff, c_eff, layer_s = eff.for_backend("cpu")
        if model.n_experts:
            layer_s += eff.for_moe("cpu")[0]
        bw_eff = eff.device_bandwidth.get(f"{system.cpu.name}|cpu", bw_eff)
        c_eff = eff.device_compute.get(f"{system.cpu.name}|cpu", c_eff)
        tflops = system.cpu.fp32_tflops * c_eff
        placements.append(
            Placement(f"cpu:{system.cpu.name}", remaining, remaining * layer_w / GB,
                      remaining * layer_kv / GB,
                      system.ram.bandwidth_gbps * bw_eff, tflops, False, remaining * layer_s,
                      _flop_time(model, remaining, tflops) / eff.batch_compute("cpu"))
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

    active_ratio = weight_read_ratio(model, 1, quant)
    gpu_backend = {g.name: g.backend for g in system.gpus}
    base_time = 0.0
    kv_terms: list[tuple[float, float, float]] = []
    attention: list[tuple[float, float, float]] = []
    prefill_time = 0.0
    overhead = weight_read = batch_time = 0.0
    for p in placements:
        frac = p.layers / model.n_layers
        bw = p.bandwidth_gbps * GB
        backend = gpu_backend[p.device] if p.is_gpu else "cpu"
        kv_read, attn = eff.for_depth(backend)
        moe_compute = eff.for_moe(backend)[1] if model.n_experts else 1.0
        base_time += p.weights_gb * GB * active_ratio / bw + p.overhead_s
        kv_terms.append((kv_tok * frac * (1 - swa_share), kv_tok * frac * swa_share,
                         bw / kv_read))
        flops = attention_flops(model) * p.layers
        attention.append((flops * (1 - swa_share), flops * swa_share, p.tflops * 1e12 * attn))
        prefill_time += _flop_time(model, p.layers, p.tflops * moe_compute)
        overhead += p.overhead_s
        weight_read += p.weights_gb * GB / bw
        batch_time += p.batch_s

    est = Estimate(
        model=model, quant=quant, context=context,
        weights_gb=weights / GB, kv_gb=kv_total / GB,
        placements=placements, fits=fits,
        prefill_tps=1.0 / prefill_time,
        decode_tps_empty=0.0, decode_tps_full=0.0, notes=notes, parallel=parallel,
        _kv_per_token=kv_terms, _attention=attention, _base_time=base_time,
        _overhead=overhead,
        _weight_read=weight_read, _batch_time=batch_time,
    )
    est.decode_tps_empty = est.decode_tps_at(0)
    est.decode_tps_full = est.decode_tps_at(context)
    return est


def _flop_time(model: Model, layers: int, tflops: float) -> float:
    """Seconds of compute per token for `layers` layers: 2 FLOPs per active parameter."""
    return 2 * model.active_b * 1e9 * (layers / model.n_layers) / (tflops * 1e12)


def parallel_levels(n: int) -> list[int]:
    """Concurrency levels shown for up to n requests: powers of two and n itself."""
    levels, k = [], 1
    while k < n:
        levels.append(k)
        k *= 2
    return levels + [n]


def format_estimate(est: Estimate) -> str:
    m = est.model
    lines = [
        f"Model     {m.name} ({m.params_b:g}B"
        + (f", {m.active_b:g}B active" if m.active_params_b else "")
        + f") {est.quant}, context {est.context}"
        + (f" x {est.parallel} slots (llama-server -np {est.parallel} "
           f"-c {est.context * est.parallel})" if est.parallel > 1 else ""),
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
    if est.parallel > 1:
        lines += ["Parallel  requests  each (empty -> full context)   total (empty -> full)"]
        for n in parallel_levels(est.parallel):
            each = est.parallel_tps(n, 0), est.parallel_tps(n, est.context)
            lines.append(f"  {n:>16}  {each[0]:8,.1f} -> {each[1]:8,.1f} tok/s   "
                         f"{n * each[0]:8,.1f} -> {n * each[1]:8,.1f} tok/s")
    lines += [f"Note      {n}" for n in est.notes]
    return "\n".join(lines)
