"""Hardware and model profiles.

All preset numbers are approximate public specs. They are meant for
order-of-magnitude planning, not for exact benchmarking.
"""

from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class GPU:
    name: str
    vram_gb: float
    bandwidth_gbps: float  # memory bandwidth, GB/s
    fp16_tflops: float  # dense FP16 tensor throughput
    backend: str = "cuda"  # default runtime backend: "cuda", "metal", "rocm", "vulkan", "sycl"
    vendor: str = ""
    segment: str = ""  # "consumer", "workstation" or "datacenter"
    label: str = ""  # display name, e.g. "Radeon RX 7900 XTX"


@dataclass(frozen=True)
class CPU:
    name: str
    cores: int
    fp32_tflops: float  # peak SIMD FP32 throughput
    vendor: str = ""
    segment: str = ""  # "desktop", "workstation" or "server"
    mem_kind: str = ""  # fastest officially supported memory, e.g. "ddr5-5600"
    mem_channels: int = 0  # 64-bit memory channels per socket


@dataclass(frozen=True)
class RAM:
    size_gb: float
    bandwidth_gbps: float
    label: str = ""


@dataclass(frozen=True)
class Model:
    name: str
    params_b: float  # total parameters, billions
    n_layers: int
    n_kv_heads: int
    head_dim: int
    max_context: int
    active_params_b: float | None = None  # MoE: parameters used per token
    tokenizer: str = ""  # tokenizer family in data/tokenizer_ratios.json, "" if unknown

    @property
    def active_b(self) -> float:
        return self.active_params_b or self.params_b


@dataclass(frozen=True)
class System:
    cpu: CPU
    ram: RAM
    gpus: tuple[GPU, ...] = ()
    unified: bool = False  # GPU shares system RAM; GPU vram_gb is ignored
    gpu_fraction: float | None = None  # unified: share of RAM the GPU may use (default: Efficiency)


@dataclass(frozen=True)
class UnifiedChip:
    """A system on a chip whose CPU and GPU share one memory pool (Apple Silicon, DGX Spark)."""

    name: str
    gpu_cores: int
    bandwidth_gbps: float
    gpu_tflops: float  # dense FP16 throughput of the GPU
    cpu_cores: int
    cpu_tflops: float
    memory_options_gb: tuple[int, ...]
    vendor: str = "Apple"
    backend: str = "metal"
    gpu_fraction: float | None = None  # share of memory the GPU may use; None: Efficiency default
    label: str = ""  # display name, e.g. "M4 Max" or "DGX Spark"


# Effective bits per weight for common llama.cpp / GGUF quantizations.
QUANT_BPW: dict[str, float] = {
    "Q3_K_M": 3.91,
    "Q4_0": 4.55,
    "Q4_K_M": 4.89,
    "MXFP4": 4.25,
    "Q5_K_M": 5.69,
    "Q6_K": 6.56,
    "Q8_0": 8.50,
    "F16": 16.0,
    "BF16": 16.0,
}

GPU_PRESETS: dict[str, GPU] = {
    g.name: g
    for g in [
        GPU("rtx2080ti-11gb", 11, 616, 108),
        GPU("rtx3060-12gb", 12, 360, 51),
        GPU("rtx3080-10gb", 10, 760, 119),
        GPU("rtx3090-24gb", 24, 936, 142),
        GPU("rtx3090ti-24gb", 24, 1008, 160),
        GPU("rtx4060ti-8gb", 8, 288, 88),
        GPU("rtx4060ti-16gb", 16, 288, 88),
        GPU("rtx4070tisuper-16gb", 16, 672, 176),
        GPU("rtx4080-16gb", 16, 717, 195),
        GPU("rtx4080super-16gb", 16, 736, 209),
        GPU("rtx4090-24gb", 24, 1008, 330),
        GPU("rtx5060ti-16gb", 16, 448, 95),
        GPU("rtx5070-12gb", 12, 672, 124),
        GPU("rtx5070ti-16gb", 16, 896, 176),
        GPU("rtx5080-16gb", 16, 960, 225),
        GPU("rtx5090-32gb", 32, 1792, 419),
    ]
}

def _cpu(name: str, vendor: str, segment: str, mem_kind: str, mem_channels: int,
         *groups: tuple[int, float, float, int]) -> CPU:
    """CPU preset from (cores, base GHz, max boost GHz, FP32 FLOPs/cycle) per core type.

    The vector clock under an all-core load is taken as the midpoint of base and max
    boost, a rough figure: CPU efficiency factors are not calibrated yet.
    """
    tflops = sum(n * (base + boost) / 2 * flops for n, base, boost, flops in groups) / 1000
    return CPU(name, sum(g[0] for g in groups), round(tflops, 2), vendor, segment, mem_kind,
               mem_channels)


# FP32 FLOPs per cycle per core: Zen 3/Zen 4 and Intel client P-cores 32 (2x256-bit FMA;
# Zen 4 runs AVX-512 on 256-bit units), Zen 5/Zen 5c in Ryzen 9000, Threadripper 9000 and
# EPYC 9005 64 (full 512-bit datapath), Sapphire/Granite Rapids Xeons 64 (2x512-bit FMA),
# Gracemont 16 and Skymont 32 (128-bit FMA pipes; less certain). Clocks, core counts and
# memory support come from AMD product pages, Intel datasheets and Wikipedia
# (see data/README.md).
CPU_PRESETS: dict[str, CPU] = {
    c.name: c
    for c in [
        _cpu("ryzen5-5600x", "AMD", "desktop", "ddr4-3200", 2, (6, 3.7, 4.6, 32)),
        _cpu("ryzen7-5800x", "AMD", "desktop", "ddr4-3200", 2, (8, 3.8, 4.7, 32)),
        _cpu("ryzen7-5800x3d", "AMD", "desktop", "ddr4-3200", 2, (8, 3.4, 4.5, 32)),
        _cpu("ryzen9-5950x", "AMD", "desktop", "ddr4-3200", 2, (16, 3.4, 4.9, 32)),
        _cpu("ryzen5-7600x", "AMD", "desktop", "ddr5-5200", 2, (6, 4.7, 5.3, 32)),
        _cpu("ryzen7-7800x3d", "AMD", "desktop", "ddr5-5200", 2, (8, 4.2, 5.0, 32)),
        _cpu("ryzen9-7950x", "AMD", "desktop", "ddr5-5200", 2, (16, 4.5, 5.7, 32)),
        _cpu("ryzen5-9600x", "AMD", "desktop", "ddr5-5600", 2, (6, 3.9, 5.4, 64)),
        _cpu("ryzen7-9700x", "AMD", "desktop", "ddr5-5600", 2, (8, 3.8, 5.5, 64)),
        _cpu("ryzen7-9800x3d", "AMD", "desktop", "ddr5-5600", 2, (8, 4.7, 5.2, 64)),
        _cpu("ryzen9-9950x", "AMD", "desktop", "ddr5-5600", 2, (16, 4.3, 5.7, 64)),
        _cpu("ryzen9-9950x3d", "AMD", "desktop", "ddr5-5600", 2, (16, 4.3, 5.7, 64)),
        _cpu("threadripper-7970x", "AMD", "workstation", "ddr5-5200", 4, (32, 4.0, 5.3, 32)),
        _cpu("threadripper-7980x", "AMD", "workstation", "ddr5-5200", 4, (64, 3.2, 5.1, 32)),
        _cpu("threadripper-pro-7995wx", "AMD", "workstation", "ddr5-5200", 8,
             (96, 2.5, 5.1, 32)),
        _cpu("threadripper-9980x", "AMD", "workstation", "ddr5-6400", 4, (64, 3.2, 5.4, 64)),
        _cpu("threadripper-pro-9995wx", "AMD", "workstation", "ddr5-6400", 8,
             (96, 2.5, 5.4, 64)),
        _cpu("epyc-9354", "AMD", "server", "ddr5-4800", 12, (32, 3.25, 3.8, 32)),
        _cpu("epyc-9554", "AMD", "server", "ddr5-4800", 12, (64, 3.1, 3.75, 32)),
        _cpu("epyc-9654", "AMD", "server", "ddr5-4800", 12, (96, 2.4, 3.7, 32)),
        _cpu("epyc-9575f", "AMD", "server", "ddr5-6400", 12, (64, 3.3, 5.0, 64)),
        _cpu("epyc-9755", "AMD", "server", "ddr5-6400", 12, (128, 2.7, 4.1, 64)),
        _cpu("epyc-9965", "AMD", "server", "ddr5-6400", 12, (192, 2.25, 3.7, 64)),
        _cpu("i5-12400", "Intel", "desktop", "ddr5-4800", 2, (6, 2.5, 4.4, 32)),
        _cpu("i5-13600k", "Intel", "desktop", "ddr5-5600", 2, (6, 3.5, 5.1, 32),
             (8, 2.6, 3.9, 16)),
        _cpu("i7-13700k", "Intel", "desktop", "ddr5-5600", 2, (8, 3.4, 5.4, 32),
             (8, 2.5, 4.2, 16)),
        _cpu("i9-13900k", "Intel", "desktop", "ddr5-5600", 2, (8, 3.0, 5.8, 32),
             (16, 2.2, 4.3, 16)),
        _cpu("i9-14900k", "Intel", "desktop", "ddr5-5600", 2, (8, 3.2, 6.0, 32),
             (16, 2.4, 4.4, 16)),
        _cpu("core-ultra7-265k", "Intel", "desktop", "ddr5-6400", 2, (8, 3.9, 5.5, 32),
             (12, 3.3, 4.6, 32)),
        _cpu("core-ultra9-285k", "Intel", "desktop", "ddr5-6400", 2, (8, 3.7, 5.7, 32),
             (16, 3.2, 4.6, 32)),
        _cpu("xeon-w7-3465x", "Intel", "workstation", "ddr5-4800", 8, (28, 2.5, 4.8, 64)),
        _cpu("xeon-w9-3495x", "Intel", "workstation", "ddr5-4800", 8, (56, 1.9, 4.8, 64)),
        _cpu("xeon-platinum-8480plus", "Intel", "server", "ddr5-4800", 8, (56, 2.0, 3.8, 64)),
        _cpu("xeon-6980p", "Intel", "server", "ddr5-6400", 12, (128, 2.0, 3.9, 64)),
    ]
}

# Per-GPU-core FP16 TFLOPS by generation: 128 ALUs x 2 FLOPs x GPU clock. M5 adds a
# Neural Accelerator to every GPU core; Apple quotes over 4x M4's peak GPU AI compute.
_APPLE_CORE_GFLOPS = {"M1": 0.327, "M2": 0.358, "M3": 0.353, "M4": 0.404, "M5": 4 * 0.404}


def _apple(name: str, gpu_cores: int, bw: float, cpu_cores: int, cpu_tflops: float,
           memory: tuple[int, ...]) -> UnifiedChip:
    tflops = round(gpu_cores * _APPLE_CORE_GFLOPS[name.split()[0]], 1)
    slug = f"apple-{name.lower().replace(' ', '-')}-{gpu_cores}c"
    return UnifiedChip(slug, gpu_cores, bw, tflops, cpu_cores, cpu_tflops, memory, label=name)


APPLE_PRESETS: dict[str, UnifiedChip] = {
    a.name: a
    for a in [
        _apple("M1", 7, 68, 8, 0.6, (8, 16)),
        _apple("M1", 8, 68, 8, 0.6, (8, 16)),
        _apple("M1 Pro", 14, 200, 8, 0.7, (16, 32)),
        _apple("M1 Pro", 16, 200, 10, 0.9, (16, 32)),
        _apple("M1 Max", 24, 400, 10, 0.9, (32, 64)),
        _apple("M1 Max", 32, 400, 10, 0.9, (32, 64)),
        _apple("M1 Ultra", 48, 800, 20, 1.8, (64, 128)),
        _apple("M1 Ultra", 64, 800, 20, 1.8, (64, 128)),
        _apple("M2", 8, 100, 8, 0.7, (8, 16, 24)),
        _apple("M2", 10, 100, 8, 0.7, (8, 16, 24)),
        _apple("M2 Pro", 16, 200, 10, 0.9, (16, 32)),
        _apple("M2 Pro", 19, 200, 12, 1.1, (16, 32)),
        _apple("M2 Max", 30, 400, 12, 1.1, (32, 64, 96)),
        _apple("M2 Max", 38, 400, 12, 1.1, (32, 64, 96)),
        _apple("M2 Ultra", 60, 800, 24, 2.2, (64, 128, 192)),
        _apple("M2 Ultra", 76, 800, 24, 2.2, (64, 128, 192)),
        _apple("M3", 10, 100, 8, 0.8, (8, 16, 24)),
        _apple("M3 Pro", 14, 150, 11, 1.0, (18, 36)),
        _apple("M3 Pro", 18, 150, 12, 1.1, (18, 36)),
        _apple("M3 Max", 30, 300, 14, 1.4, (36, 96)),
        _apple("M3 Max", 40, 400, 16, 1.6, (48, 64, 128)),
        _apple("M3 Ultra", 60, 800, 28, 2.8, (96, 256)),
        _apple("M3 Ultra", 80, 800, 32, 3.2, (96, 256, 512)),
        _apple("M4", 10, 120, 10, 1.0, (16, 24, 32)),
        _apple("M4 Pro", 16, 273, 12, 1.3, (24, 48)),
        _apple("M4 Pro", 20, 273, 14, 1.5, (24, 48, 64)),
        _apple("M4 Max", 32, 410, 14, 1.5, (36,)),
        _apple("M4 Max", 40, 546, 16, 1.7, (48, 64, 128)),
        # M5 family: Apple newsroom and Mac/MacBook tech specs (2025-10, 2026-03, 2026-08).
        _apple("M5", 8, 153.6, 10, 1.1, (16, 24, 32)),
        _apple("M5", 10, 153.6, 10, 1.1, (16, 24, 32)),
        _apple("M5 Pro", 16, 307, 15, 1.6, (24, 48)),
        _apple("M5 Pro", 20, 307, 18, 1.9, (24, 48, 64)),
        _apple("M5 Max", 32, 460, 18, 1.9, (36,)),
        _apple("M5 Max", 40, 614, 18, 1.9, (48, 64, 128)),
        _apple("M5 Ultra", 64, 1228.8, 30, 3.2, (96, 256)),
        _apple("M5 Ultra", 80, 1228.8, 36, 3.8, (96, 256, 512)),
    ]
}


UNIFIED_PRESETS: dict[str, UnifiedChip] = {**APPLE_PRESETS}


def unified_system(chip: UnifiedChip, memory_gb: float) -> System:
    """A Mac or other SoC system where CPU and GPU share one pool of unified memory."""
    return System(
        cpu=CPU(f"{chip.name}-cpu", chip.cpu_cores, chip.cpu_tflops),
        ram=RAM(memory_gb, chip.bandwidth_gbps, "unified"),
        gpus=(GPU(chip.name, memory_gb, chip.bandwidth_gbps, chip.gpu_tflops, chip.backend,
                  chip.vendor),),
        unified=True,
        gpu_fraction=chip.gpu_fraction,
    )


MODEL_PRESETS: dict[str, Model] = {
    m.name: m
    for m in [
        Model("llama-2-7b", 6.74, 32, 32, 128, 4096, tokenizer="llama2"),
        Model("llama-3.2-3b", 3.21, 28, 8, 128, 131072, tokenizer="llama3"),
        Model("llama-3.1-8b", 8.03, 32, 8, 128, 131072, tokenizer="llama3"),
        Model("llama-3.1-70b", 70.6, 80, 8, 128, 131072, tokenizer="llama3"),
        Model("mistral-7b", 7.24, 32, 8, 128, 32768, tokenizer="mistral"),
        Model("gemma-2-9b", 9.24, 42, 8, 256, 8192, tokenizer="gemma2"),
        Model("qwen2.5-7b", 7.62, 28, 4, 128, 131072, tokenizer="qwen2.5"),
        Model("qwen2.5-14b", 14.8, 48, 8, 128, 131072, tokenizer="qwen2.5"),
        Model("qwen2.5-32b", 32.8, 64, 8, 128, 131072, tokenizer="qwen2.5"),
        Model("qwen3-8b", 8.19, 36, 8, 128, 32768, tokenizer="qwen3"),
        Model("qwen3-14b", 14.8, 40, 8, 128, 32768, tokenizer="qwen3"),
        Model("qwen3-32b", 32.8, 64, 8, 128, 32768, tokenizer="qwen3"),
        Model("qwen3-30b-a3b", 30.5, 48, 4, 128, 131072, active_params_b=3.3, tokenizer="qwen3"),
        Model("gpt-oss-20b", 21.0, 24, 8, 64, 131072, active_params_b=3.6, tokenizer="gpt-oss"),
        Model("gpt-oss-120b", 117.0, 36, 8, 64, 131072, active_params_b=5.1, tokenizer="gpt-oss"),
    ]
}


def ram_bandwidth_gbps(kind: str, channels: int = 2) -> float:
    """Theoretical bandwidth for a DDR kind such as 'ddr5-5600'.

    Each 64-bit channel moves 8 bytes per transfer.
    """
    m = re.fullmatch(r"(?:l?pddr|ddr)\d-(\d+)", kind.strip().lower())
    if not m:
        raise ValueError(f"Unrecognized RAM kind {kind!r}; expected e.g. 'ddr5-5600'")
    return int(m.group(1)) * 8 * channels / 1000


def lookup(presets: dict, name: str, what: str):
    try:
        return presets[name]
    except KeyError:
        known = ", ".join(sorted(presets))
        raise ValueError(f"Unknown {what} preset {name!r}. Known: {known}") from None
