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
Layers are placed like llama-server's default --fit: whole layers go to the
GPUs and the rest stays in system RAM, except that a mixture-of-experts model
keeps its routed experts in RAM first, so that attention, shared weights and
KV cache of every layer can stay on the GPU. Devices run layers sequentially,
so per-token times add up. Prompt batches of 32 tokens or more multiply the
weights held in RAM on the first GPU, which copies them over PCIe for each
batch (llama.cpp's op offload). On unified-memory systems (Apple Silicon) the
GPU budget is a fraction of system RAM instead of dedicated VRAM.

Concurrent requests (llama-server --parallel slots with continuous batching)
share one decode step: the weights are read once for the whole batch (for
mixture-of-experts models, the union of the experts the batch is routed to),
each request reads its own KV cache, and every request beyond the first adds
compute at a fraction of the prompt-processing rate (`*_batch_compute`).
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path

from .profiles import QUANT_BPW, QUANT_SHARED_BPW, Model, System, pcie_gbps


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
# llama.cpp processes prompts in batches of n_ubatch tokens (default -ub 512).
N_UBATCH = 512
# llama.cpp keeps n_swa + n_ubatch tokens in a sliding-window layer's cache (it also pads to
# 256), unless --swa-full: llama-kv-cache-iswa.cpp.
SWA_UBATCH = N_UBATCH
# Ops on weights in system RAM with at least this many tokens run on the GPU instead, which
# copies the weights over first (GGML_OP_OFFLOAD_MIN_BATCH, ggml-cuda.cu).
OP_OFFLOAD_MIN_BATCH = 32
# llama-server without -c shrinks the context to fit the GPUs, but not below this
# (common.h fit_params_min_ctx), in steps of 256 tokens (common/fit.cpp).
FIT_MIN_CONTEXT = 4096
FIT_CONTEXT_STEP = 256
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

    gpu_bandwidth: float = 0.83
    gpu_compute: float = 0.52
    gpu_layer_overhead_us: float = 36.0
    metal_bandwidth: float = 0.96
    metal_compute: float = 0.65
    metal_layer_overhead_us: float = 77.0
    rocm_bandwidth: float = 0.95
    rocm_compute: float = 0.33
    rocm_layer_overhead_us: float = 114.0
    vulkan_bandwidth: float = 0.79
    vulkan_compute: float = 0.41
    vulkan_layer_overhead_us: float = 22.0
    sycl_bandwidth: float = 0.84
    sycl_compute: float = 0.08
    sycl_layer_overhead_us: float = 58.0
    # SYCL builds with GGML_SYCL_F16=ON ("sycl-f16") share every SYCL factor but this one:
    # their prompt processing runs in FP16 and is about twice as fast as the default build.
    sycl_f16_compute: float = 0.10
    # The *_compute factors describe prompt processing with flash attention on (llama.cpp's
    # default, "auto"); without it, prompts reach *_compute_no_fa times that rate. Fitted to
    # runs measured both ways; 1.0 without such pairs. Decode barely changes.
    gpu_compute_no_fa: float = 0.89
    metal_compute_no_fa: float = 1.0
    rocm_compute_no_fa: float = 0.95
    vulkan_compute_no_fa: float = 0.92
    sycl_compute_no_fa: float = 0.86
    sycl_f16_compute_no_fa: float = 2.05
    cpu_compute_no_fa: float = 1.0
    cpu_bandwidth: float = 0.75
    cpu_compute: float = 0.31
    cpu_layer_overhead_us: float = 118.0
    # Long contexts, fitted to llama-bench runs at several depths (llama.cpp benches/):
    # decode reads the KV cache *_kv_read times slower than weights, and prompt attention
    # runs at *_attention times the prompt-processing FLOP rate. Backends without depth
    # measurements keep 1.0.
    gpu_kv_read: float = 0.94
    gpu_attention: float = 0.96
    metal_kv_read: float = 2.27
    metal_attention: float = 0.62
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
    metal_moe_layer_overhead_us: float = 102.0
    rocm_moe_layer_overhead_us: float = 0.0
    vulkan_moe_layer_overhead_us: float = 0.0
    sycl_moe_layer_overhead_us: float = 0.0
    cpu_moe_layer_overhead_us: float = 0.0
    gpu_moe_compute: float = 0.49
    metal_moe_compute: float = 1.0
    rocm_moe_compute: float = 1.0
    vulkan_moe_compute: float = 1.0
    sycl_moe_compute: float = 1.0
    cpu_moe_compute: float = 1.0
    # Share of the prompt-processing compute rate that each request beyond the first
    # reaches in a decode batch. Fitted for CUDA to llama-batched-bench results (DGX
    # Spark); the other backends use the CUDA value until they have measurements.
    gpu_batch_compute: float = 0.40
    metal_batch_compute: float = 0.40
    rocm_batch_compute: float = 0.40
    vulkan_batch_compute: float = 0.40
    sycl_batch_compute: float = 0.40
    cpu_batch_compute: float = 0.40
    vram_reserve_gb: float = 1.0  # CUDA context + compute buffers, per GPU
    # Share of the theoretical PCIe rate reached copying weights from system RAM to the GPU.
    # Not calibrated yet: no measurements with weights in RAM.
    pcie_bandwidth: float = 1.0
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
        return (getattr(self, f"{p}_bandwidth"), getattr(self, compute_field(backend)),
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


# "sycl" is llama.cpp's default SYCL build (FP32); "sycl-f16" is built with GGML_SYCL_F16=ON.
BACKENDS = ("cuda", "metal", "rocm", "vulkan", "sycl", "sycl-f16", "cpu")


def backend_prefix(backend: str) -> str:
    """Efficiency field prefix for a backend ("cuda" predates the others as "gpu"; both SYCL
    builds share the "sycl" fields)."""
    if backend not in BACKENDS:
        raise ValueError(f"Unknown backend {backend!r}. Known: {', '.join(BACKENDS)}")
    return {"cuda": "gpu", "sycl-f16": "sycl"}.get(backend, backend)


def compute_field(backend: str) -> str:
    """Efficiency field of a backend's prompt-processing compute share."""
    return "sycl_f16_compute" if backend == "sycl-f16" else f"{backend_prefix(backend)}_compute"


def without_flash_attention(eff: Efficiency) -> Efficiency:
    """Efficiency for runs with flash attention off: every prompt compute share, per backend
    and per device, scaled by its backend's *_compute_no_fa."""
    scale = {b: getattr(eff, f"{compute_field(b)}_no_fa") for b in BACKENDS}
    fields = {compute_field(b): getattr(eff, compute_field(b)) * scale[b] for b in BACKENDS}
    devices = {k: v * scale[k.split("|")[1]] for k, v in eff.device_compute.items()}
    return replace(eff, device_compute=devices, **fields)


@dataclass
class Placement:
    device: str
    layers: int  # layers whose attention, shared weights and KV cache are here
    weights_gb: float
    kv_gb: float
    bandwidth_gbps: float
    tflops: float
    is_gpu: bool
    overhead_s: float = 0.0  # fixed per-token time for this device's layers
    batch_s: float = 0.0  # compute time each extra request adds to a decode step
    # Layers whose routed experts are here; equals `layers` unless --fit kept the experts
    # of a mixture-of-experts model in system RAM.
    expert_layers: int = 0


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
    # Context per slot llama-server would pick without -c to keep the whole model on the
    # GPUs, if that is smaller than `context` and not below FIT_MIN_CONTEXT; else 0.
    fit_context: int = 0
    # Per device: (KV bytes per token of full-attention layers, of sliding-window layers,
    # effective bandwidth for reading the cache in bytes/s).
    _kv_per_token: list[tuple[float, float, float]] = field(default_factory=list, repr=False)
    # Per device: (attention FLOPs per prompt token and cached token, full layers;
    # sliding-window layers; attention FLOP/s).
    _attention: list[tuple[float, float, float]] = field(default_factory=list, repr=False)
    # Per device: (bytes of shared weights a decode step reads, bytes of routed experts held,
    # bandwidth in bytes/s).
    _weights: list[tuple[float, float, float]] = field(default_factory=list, repr=False)
    _base_time: float = 0.0
    _overhead: float = 0.0  # per-step fixed time of all devices
    _batch_time: float = 0.0  # compute time each extra request adds to a step
    # Prompt processing: seconds per token for the weights in GPU memory and, for the weights
    # in system RAM, on the GPU (op offload) and on the CPU (batches too small to offload);
    # seconds to copy the weights in RAM to the GPU once per batch.
    _prompt_s: float = 0.0
    _host_gpu_s: float = 0.0
    _host_cpu_s: float = 0.0
    _transfer_s: float = 0.0

    def _kv_time(self, filled: int) -> float:
        """Seconds to read the KV cache of one request holding `filled` tokens."""
        window = min(filled, swa_cells(self.model))
        return add_in_order((full * filled + swa * window) / bw
                            for full, swa, bw in self._kv_per_token)

    def decode_tps_at(self, filled_tokens: int) -> float:
        """Decode speed of a single request when `filled_tokens` are in its KV cache."""
        t = self._base_time + self._kv_time(filled_tokens)
        return 1.0 / t if t > 0 else float("inf")

    def _ubatch_time(self, tokens: int) -> float:
        """Seconds for one prompt batch of `tokens` tokens, without attention."""
        if tokens >= OP_OFFLOAD_MIN_BATCH:
            return tokens * (self._prompt_s + self._host_gpu_s) + self._transfer_s
        return tokens * (self._prompt_s + self._host_cpu_s)

    def prefill_time(self, tokens: int, depth: int = 0) -> float:
        """Seconds to process a prompt of `tokens` tokens after `depth` cached tokens.

        The prompt runs in batches of N_UBATCH tokens. Token i attends to depth + i + 1
        tokens, a sliding-window layer to at most the tokens its cache keeps, so prompts
        get slower as the context fills. `prefill_tps` leaves out attention: it is the
        speed of one full batch in an empty context.
        """
        batches, rest = divmod(tokens, N_UBATCH)
        full = context_sum(depth + 1, depth + tokens)
        swa = context_sum(depth + 1, depth + tokens, swa_cells(self.model))
        return (batches * self._ubatch_time(N_UBATCH)
                + (self._ubatch_time(rest) if rest else 0.0)
                + add_in_order((f * full + w * swa) / rate for f, w, rate in self._attention))

    def step_time(self, fills: Sequence[int]) -> float:
        """Seconds for one decode step that gives each active request one token.

        `fills` holds the KV fill of every active request. With one request this
        equals 1 / decode_tps_at(fill) up to rounding.
        """
        n = len(fills)
        touched = experts_touched(self.model, n)
        t = (self._overhead
             + add_in_order((shared + expert * touched) / bw for shared, expert, bw in self._weights)
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


def expert_params(model: Model) -> float:
    """Billions of parameters in all routed experts; 0 for dense models."""
    if not model.n_experts:
        return 0.0
    if model.expert_params_b:
        return model.expert_params_b
    share = model.experts_per_token / model.n_experts
    return (model.params_b - model.active_b) / (1 - share)


def _shared_bpw(model: Model, quant: str) -> float:
    return QUANT_SHARED_BPW.get(quant, QUANT_BPW[quant]) if model.n_experts else QUANT_BPW[quant]


def weight_bytes(model: Model, quant: str) -> tuple[float, float]:
    """(bytes of the shared weights, bytes of all routed experts); dense models have no
    experts. Some quantizations apply to the experts only (QUANT_SHARED_BPW)."""
    bpw = QUANT_BPW[quant]
    if not model.n_experts:
        return model.params_b * 1e9 * bpw / 8, 0.0
    expert = expert_params(model)
    shared = model.params_b - expert
    return shared * 1e9 * _shared_bpw(model, quant) / 8, expert * 1e9 * bpw / 8


def lookup_params(model: Model) -> float:
    """Billions of token-embedding parameters a token only looks up, so never reads in full
    (none when the output layer reuses the matrix)."""
    return 0.0 if model.tied_embeddings else model.embedding_params_b


def experts_touched(model: Model, n: int) -> float:
    """Share of a mixture-of-experts model's routed experts a decode step for `n` requests
    reads: with uniform independent routing to k of E experts, 1 - (1 - k/E)^n."""
    share = model.experts_per_token / model.n_experts if model.n_experts else 0.0
    return share if n <= 1 else 1 - (1 - share) ** n


def shared_read_bytes(model: Model, quant: str) -> float:
    """Bytes of the shared weights a decode step reads: all but the token embedding, which
    is a lookup (for a dense model given only an active size, its active share)."""
    shared = weight_bytes(model, quant)[0]
    if not model.n_experts and model.active_params_b:
        return shared * model.active_b / model.params_b
    return shared - lookup_params(model) * 1e9 * _shared_bpw(model, quant) / 8


def weight_read_ratio(model: Model, n: int, quant: str = "Q4_K_M") -> float:
    """Share of the weight bytes read in a decode step for `n` requests: the shared weights
    but the token embedding (shared_read_bytes) and the experts any request is routed to
    (experts_touched)."""
    if not model.n_experts and model.active_params_b:  # only an active size is known
        return model.active_b / model.params_b
    shared, expert = weight_bytes(model, quant)
    return ((shared_read_bytes(model, quant) + expert * experts_touched(model, n))
            / (shared + expert))


def matmul_split(model: Model, prompt: bool) -> tuple[float, float]:
    """Billions of weights each token multiplies, as (shared weights, routed experts): the
    shared weights but the token embedding (a lookup) and, for prompt tokens, the output
    layer (llama.cpp computes logits only for the last token of a prompt), and the experts
    the token is routed to."""
    if not model.n_experts and model.active_params_b:  # only an active size is known
        return model.active_b, 0.0
    expert = expert_params(model)
    share = model.experts_per_token / model.n_experts if model.n_experts else 0.0
    output = model.embedding_params_b if prompt else 0.0
    return model.params_b - expert - lookup_params(model) - output, expert * share


def matmul_params(model: Model, prompt: bool) -> float:
    """Billions of weights each token multiplies (matmul_split)."""
    return add_in_order(matmul_split(model, prompt))


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

    shared, expert = weight_bytes(model, quant)
    weights = add_in_order((shared, expert))
    kv_tok = kv_bytes_per_token(model, kv_bits)
    # A sliding-window layer's cache keeps only about its window.
    swa_share = model.swa_layers / model.n_layers

    def kv_bytes(ctx: int) -> float:
        return kv_tok * (ctx * (1 - swa_share)
                         + min(ctx, swa_cells(model)) * swa_share) * parallel

    kv_total = kv_bytes(context)
    n_layers = model.n_layers
    layer_d = shared / n_layers
    layer_e = expert / n_layers
    layer_kv = kv_total / n_layers
    read_d = shared_read_bytes(model, quant) / n_layers
    share = model.experts_per_token / model.n_experts if model.n_experts else 0.0

    budgets = []
    for gpu in system.gpus:
        frac = system.gpu_fraction or eff.unified_gpu_fraction
        vram = system.ram.size_gb * frac if system.unified else gpu.vram_gb
        budgets.append(max(0.0, vram - eff.vram_reserve_gb) * GB)
    dense, full = fit_layers(budgets, n_layers, layer_d, layer_e, layer_kv)

    def placement(device: str, n: int, f: int, backend: str, bandwidth_gbps: float,
                  tflops: float, is_gpu: bool) -> Placement:
        layer_s = eff.for_backend(backend)[2]
        moe_s = eff.for_moe(backend)[0] if model.n_experts else 0.0
        # A device holding only the experts of a layer still runs a step of its own for it.
        return Placement(device, n, (n * layer_d + f * layer_e) / GB, n * layer_kv / GB,
                         bandwidth_gbps, tflops, is_gpu, max(n, f) * layer_s + f * moe_s,
                         _flop_time(model, n, f, tflops, prompt=False)
                         / eff.batch_compute(backend), f)

    def gpu_tflops(gpu) -> float:
        c_eff = eff.for_backend(gpu.backend)[1]
        return gpu.fp16_tflops * eff.device_compute.get(f"{gpu.name}|{gpu.backend}", c_eff)

    placements: list[Placement] = []
    gpu_used_gb = 0.0
    for gpu, n, f in zip(system.gpus, dense, full):
        if n or f:
            placements.append(placement(
                gpu.name, n, f, gpu.backend,
                gpu.bandwidth_gbps * eff.for_backend(gpu.backend)[0], gpu_tflops(gpu), True))
            gpu_used_gb += placements[-1].weights_gb + placements[-1].kv_gb

    fits = True
    cpu_layers = n_layers - sum(dense)
    cpu_experts = n_layers - sum(full)
    cpu = None
    if cpu_experts:
        bw_eff, c_eff, _ = eff.for_backend("cpu")
        bw_eff = eff.device_bandwidth.get(f"{system.cpu.name}|cpu", bw_eff)
        c_eff = eff.device_compute.get(f"{system.cpu.name}|cpu", c_eff)
        cpu = placement(f"cpu:{system.cpu.name}", cpu_layers, cpu_experts, "cpu",
                        system.ram.bandwidth_gbps * bw_eff, system.cpu.fp32_tflops * c_eff,
                        False)
        placements.append(cpu)
        cpu_gb = cpu.weights_gb + cpu.kv_gb
        ram_free = system.ram.size_gb - eff.ram_reserve_gb
        if system.unified:
            ram_free -= gpu_used_gb
        if cpu_gb > ram_free:
            fits = False
            notes.append(f"needs {cpu_gb:.1f} GB in RAM but only {ram_free:.1f} GB is free; "
                         "it would page to disk")
        if system.gpus and cpu_layers:
            notes.append(f"{cpu_layers}/{n_layers} layers offloaded to CPU")
        if system.gpus and cpu_experts > cpu_layers:
            notes.append(f"routed experts of {cpu_experts - cpu_layers}/{n_layers} layers "
                         "kept in RAM")

    gpu_backend = {g.name: g.backend for g in system.gpus}
    base_time = 0.0
    weight_terms: list[tuple[float, float, float]] = []
    kv_terms: list[tuple[float, float, float]] = []
    attention: list[tuple[float, float, float]] = []
    prompt_s = 0.0
    overhead = batch_time = 0.0
    for p in placements:
        frac = p.layers / n_layers
        bw = p.bandwidth_gbps * GB
        backend = gpu_backend[p.device] if p.is_gpu else "cpu"
        kv_read, attn = eff.for_depth(backend)
        weight_terms.append((p.layers * read_d, p.expert_layers * layer_e, bw))
        base_time += (p.layers * read_d + p.expert_layers * layer_e * share) / bw + p.overhead_s
        kv_terms.append((kv_tok * frac * (1 - swa_share), kv_tok * frac * swa_share,
                         bw / kv_read))
        flops = attention_flops(model) * p.layers
        attention.append((flops * (1 - swa_share), flops * swa_share, p.tflops * 1e12 * attn))
        if p.is_gpu:
            prompt_s += _flop_time(model, p.layers, p.expert_layers,
                                   p.tflops * _moe_compute(eff, model, backend))
        overhead += p.overhead_s
        batch_time += p.batch_s

    # Weights in system RAM: small prompt batches multiply them on the CPU; larger ones on
    # the first GPU, which copies them over its host link for every batch (llama.cpp's op
    # offload). Unified memory copies within RAM, reading and writing every byte.
    host_gpu_s = host_cpu_s = transfer_s = 0.0
    if cpu:
        host_cpu_s = _flop_time(model, cpu.layers, cpu.expert_layers,
                                cpu.tflops * _moe_compute(eff, model, "cpu"))
        host_gpu_s = host_cpu_s
        if system.gpus:
            gpu = system.gpus[0]
            link = (system.ram.bandwidth_gbps / 2 if system.unified
                    else pcie_gbps(gpu.pcie_gen, gpu.pcie_lanes) * eff.pcie_bandwidth)
            if link:
                host_gpu_s = _flop_time(model, cpu.layers, cpu.expert_layers,
                                        gpu_tflops(gpu) * _moe_compute(eff, model, gpu.backend))
                transfer_s = cpu.weights_gb / link

    # llama-server without -c starts from the model's context and shrinks it until the
    # model fits the GPUs (common/fit.cpp), keeping at least FIT_MIN_CONTEXT.
    fit_context = 0
    if cpu and system.gpus:
        top = min(context - 1, model.max_context) // FIT_CONTEXT_STEP * FIT_CONTEXT_STEP
        for ctx in range(top, FIT_MIN_CONTEXT - 1, -FIT_CONTEXT_STEP):
            if weights + kv_bytes(ctx) <= add_in_order(budgets):
                fit_context = ctx
                break

    est = Estimate(
        model=model, quant=quant, context=context,
        weights_gb=weights / GB, kv_gb=kv_total / GB,
        placements=placements, fits=fits,
        prefill_tps=0.0,
        decode_tps_empty=0.0, decode_tps_full=0.0, notes=notes, parallel=parallel,
        fit_context=fit_context,
        _kv_per_token=kv_terms, _attention=attention, _weights=weight_terms,
        _base_time=base_time, _overhead=overhead, _batch_time=batch_time,
        _prompt_s=prompt_s,
        _host_gpu_s=host_gpu_s, _host_cpu_s=host_cpu_s, _transfer_s=transfer_s,
    )
    est.prefill_tps = N_UBATCH / est._ubatch_time(N_UBATCH)
    est.decode_tps_empty = est.decode_tps_at(0)
    est.decode_tps_full = est.decode_tps_at(context)
    return est


def fit_layers(budgets: Sequence[float], n_layers: int, layer_bytes: float,
               expert_bytes: float, kv_bytes: float) -> tuple[list[int], list[int]]:
    """Layers per GPU as llama-server's --fit places them (common/fit.cpp), as (layers whose
    attention, shared weights and KV cache are on the GPU, layers whose routed experts are).

    `budgets` are the bytes each GPU may fill; the bytes are per layer, `layer_bytes` without
    the routed experts. Whole layers fill one GPU after another. If a mixture-of-experts
    model does not fit, its experts go to system RAM first; once every other part of every
    layer is on the GPUs, the room left takes back experts layer by layer.
    """
    def fill(per_layer: float, room: Sequence[float]) -> list[int]:
        out, left = [], n_layers
        for b in room:
            n = min(left, int(b // per_layer))
            out.append(n)
            left -= n
        return out

    full = fill(layer_bytes + expert_bytes + kv_bytes, budgets)
    if not expert_bytes or sum(full) == n_layers:
        return full, full
    dense = fill(layer_bytes + kv_bytes, budgets)
    if sum(dense) < n_layers:
        return dense, [0] * len(budgets)
    return dense, fill(expert_bytes, [b - n * (layer_bytes + kv_bytes)
                                      for b, n in zip(budgets, dense)])


def _moe_compute(eff: Efficiency, model: Model, backend: str) -> float:
    return eff.for_moe(backend)[1] if model.n_experts else 1.0


def _flop_time(model: Model, layers: int, expert_layers: int, tflops: float,
               prompt: bool = True) -> float:
    """Seconds of compute per token for the shared weights of `layers` layers and the routed
    experts of `expert_layers` layers: 2 FLOPs per multiplied weight."""
    shared, expert = matmul_split(model, prompt)
    return (2 * (shared * layers + expert * expert_layers) * 1e9 / model.n_layers
            / (tflops * 1e12))


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
        experts = (f", experts of {p.expert_layers}" if p.expert_layers != p.layers else "")
        lines.append(f"  {p.device:<24} {p.layers:>3} layers  "
                     f"{p.weights_gb + p.kv_gb:6.2f} GB{experts}")
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
