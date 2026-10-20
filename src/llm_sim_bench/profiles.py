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

_GPU_BACKEND = {"NVIDIA": "cuda", "AMD": "rocm", "Intel": "sycl"}


def _gpu(name: str, label: str, vendor: str, segment: str, vram_gb: float, bandwidth_gbps: float,
         fp16_tflops: float) -> GPU:
    return GPU(name, vram_gb, bandwidth_gbps, fp16_tflops, _GPU_BACKEND[vendor], vendor, segment,
               label)


# Dense FP16 tensor/matrix TFLOPS without sparsity: GeForce parts with FP16 accumulate;
# AMD "FP16 matrix" (RDNA2: packed FP16 vector, no matrix units); Intel XMX = INT8 TOPS / 2.
# Some figures are derived from vendor "AI TOPS" (see data/README.md).
GPU_PRESETS: dict[str, GPU] = {
    g.name: g
    for g in [
        _gpu("rtx2060super-8gb", "RTX 2060 SUPER", "NVIDIA", "consumer", 8, 448, 57.4),
        _gpu("rtx2070super-8gb", "RTX 2070 SUPER", "NVIDIA", "consumer", 8, 448, 72.5),
        _gpu("rtx2080ti-11gb", "RTX 2080 Ti", "NVIDIA", "consumer", 11, 616, 107.6),
        _gpu("rtx3060-12gb", "RTX 3060 12GB", "NVIDIA", "consumer", 12, 360, 51.2),
        _gpu("rtx3060ti-8gb", "RTX 3060 Ti", "NVIDIA", "consumer", 8, 448, 64.8),
        _gpu("rtx3070-8gb", "RTX 3070", "NVIDIA", "consumer", 8, 448, 81.3),
        _gpu("rtx3080-10gb", "RTX 3080 10GB", "NVIDIA", "consumer", 10, 760, 119),
        _gpu("rtx3080ti-12gb", "RTX 3080 Ti", "NVIDIA", "consumer", 12, 912, 136.4),
        _gpu("rtx3090-24gb", "RTX 3090", "NVIDIA", "consumer", 24, 936, 142),
        _gpu("rtx3090ti-24gb", "RTX 3090 Ti", "NVIDIA", "consumer", 24, 1008, 160),
        _gpu("rtx4060-8gb", "RTX 4060", "NVIDIA", "consumer", 8, 272, 60.5),
        _gpu("rtx4060ti-8gb", "RTX 4060 Ti 8GB", "NVIDIA", "consumer", 8, 288, 88.25),
        _gpu("rtx4060ti-16gb", "RTX 4060 Ti 16GB", "NVIDIA", "consumer", 16, 288, 88.25),
        _gpu("rtx4070-12gb", "RTX 4070", "NVIDIA", "consumer", 12, 504, 116.6),
        _gpu("rtx4070super-12gb", "RTX 4070 SUPER", "NVIDIA", "consumer", 12, 504, 142),
        _gpu("rtx4070tisuper-16gb", "RTX 4070 Ti SUPER", "NVIDIA", "consumer", 16, 672, 176.5),
        _gpu("rtx4080-16gb", "RTX 4080", "NVIDIA", "consumer", 16, 716.8, 194.9),
        _gpu("rtx4080super-16gb", "RTX 4080 SUPER", "NVIDIA", "consumer", 16, 736, 209),
        _gpu("rtx4090-24gb", "RTX 4090", "NVIDIA", "consumer", 24, 1008, 330.3),
        _gpu("rtx5060-8gb", "RTX 5060", "NVIDIA", "consumer", 8, 448, 76.75),
        _gpu("rtx5060ti-16gb", "RTX 5060 Ti 16GB", "NVIDIA", "consumer", 16, 448, 94.9),
        _gpu("rtx5070-12gb", "RTX 5070", "NVIDIA", "consumer", 12, 672, 123.5),
        _gpu("rtx5070ti-16gb", "RTX 5070 Ti", "NVIDIA", "consumer", 16, 896, 175.8),
        _gpu("rtx5080-16gb", "RTX 5080", "NVIDIA", "consumer", 16, 960, 225.1),
        _gpu("rtx5090-32gb", "RTX 5090", "NVIDIA", "consumer", 32, 1792, 419),
        _gpu("rtx-a4000-16gb", "RTX A4000", "NVIDIA", "workstation", 16, 448, 76.7),
        _gpu("rtx-a5000-24gb", "RTX A5000", "NVIDIA", "workstation", 24, 768, 111.1),
        _gpu("rtx-a6000-48gb", "RTX A6000", "NVIDIA", "workstation", 48, 768, 154.8),
        _gpu("rtx4000ada-20gb", "RTX 4000 Ada", "NVIDIA", "workstation", 20, 360, 81.9),
        _gpu("rtx6000ada-48gb", "RTX 6000 Ada", "NVIDIA", "workstation", 48, 960, 364.25),
        _gpu("rtxpro4000-24gb", "RTX PRO 4000 Blackwell", "NVIDIA", "workstation", 24, 672, 147.25),
        _gpu("rtxpro4500-32gb", "RTX PRO 4500 Blackwell", "NVIDIA", "workstation", 32, 896, 202.1),
        _gpu("rtxpro6000-96gb", "RTX PRO 6000 Blackwell (Workstation)", "NVIDIA", "workstation",
             96, 1792, 500),
        _gpu("t4-16gb", "Tesla T4", "NVIDIA", "datacenter", 16, 320, 65),
        _gpu("v100-sxm2-32gb", "Tesla V100 SXM2 32GB", "NVIDIA", "datacenter", 32, 900, 125),
        _gpu("a10-24gb", "A10", "NVIDIA", "datacenter", 24, 600, 125),
        _gpu("a100-pcie-40gb", "A100 40GB PCIe", "NVIDIA", "datacenter", 40, 1555, 312),
        _gpu("a100-pcie-80gb", "A100 80GB PCIe", "NVIDIA", "datacenter", 80, 1935, 312),
        _gpu("a100-sxm-80gb", "A100 80GB SXM", "NVIDIA", "datacenter", 80, 2039, 312),
        _gpu("l4-24gb", "L4", "NVIDIA", "datacenter", 24, 300, 121),
        _gpu("l40s-48gb", "L40S", "NVIDIA", "datacenter", 48, 864, 362.05),
        _gpu("h100-pcie-80gb", "H100 PCIe", "NVIDIA", "datacenter", 80, 2000, 756.4),
        _gpu("h100-sxm-80gb", "H100 SXM", "NVIDIA", "datacenter", 80, 3350, 989.5),
        _gpu("h200-141gb", "H200", "NVIDIA", "datacenter", 141, 4800, 989.5),
        _gpu("b200-180gb", "B200", "NVIDIA", "datacenter", 180, 8000, 2250),
        _gpu("rx6800xt-16gb", "Radeon RX 6800 XT", "AMD", "consumer", 16, 512, 41.47),
        _gpu("rx6900xt-16gb", "Radeon RX 6900 XT", "AMD", "consumer", 16, 512, 46.08),
        _gpu("rx7600xt-16gb", "Radeon RX 7600 XT", "AMD", "consumer", 16, 288, 45.1),
        _gpu("rx7800xt-16gb", "Radeon RX 7800 XT", "AMD", "consumer", 16, 624, 74.6),
        _gpu("rx7900gre-16gb", "Radeon RX 7900 GRE", "AMD", "consumer", 16, 576, 92),
        _gpu("rx7900xt-20gb", "Radeon RX 7900 XT", "AMD", "consumer", 20, 800, 103),
        _gpu("rx7900xtx-24gb", "Radeon RX 7900 XTX", "AMD", "consumer", 24, 960, 123),
        _gpu("rx9060xt-16gb", "Radeon RX 9060 XT 16GB", "AMD", "consumer", 16, 320, 103),
        _gpu("rx9070-16gb", "Radeon RX 9070", "AMD", "consumer", 16, 640, 145),
        _gpu("rx9070xt-16gb", "Radeon RX 9070 XT", "AMD", "consumer", 16, 640, 195),
        _gpu("radeon-pro-w7800-32gb", "Radeon PRO W7800", "AMD", "workstation", 32, 576, 90.4),
        _gpu("radeon-pro-w7900-48gb", "Radeon PRO W7900", "AMD", "workstation", 48, 864, 123),
        _gpu("radeon-ai-pro-r9700-32gb", "Radeon AI PRO R9700", "AMD", "workstation", 32, 640,
             191),
        _gpu("mi100-32gb", "Instinct MI100", "AMD", "datacenter", 32, 1228.8, 184.6),
        _gpu("mi210-64gb", "Instinct MI210", "AMD", "datacenter", 64, 1638.4, 181),
        # MI250X is two GCDs that software sees as two GPUs; add two for the whole card.
        _gpu("mi250x-gcd-64gb", "Instinct MI250X (one GCD)", "AMD", "datacenter", 64, 1638.4,
             191.5),
        _gpu("mi300x-192gb", "Instinct MI300X", "AMD", "datacenter", 192, 5300, 1307.4),
        _gpu("mi325x-256gb", "Instinct MI325X", "AMD", "datacenter", 256, 6000, 1307.4),
        _gpu("mi355x-288gb", "Instinct MI355X", "AMD", "datacenter", 288, 8000, 2516.6),
        _gpu("arc-a770-16gb", "Arc A770 16GB", "Intel", "consumer", 16, 560, 131),
        _gpu("arc-b570-10gb", "Arc B570", "Intel", "consumer", 10, 380, 101.5),
        _gpu("arc-b580-12gb", "Arc B580", "Intel", "consumer", 12, 456, 116.5),
        _gpu("arc-pro-b60-24gb", "Arc Pro B60", "Intel", "workstation", 24, 456, 98.5),
        _gpu("arc-pro-b70-32gb", "Arc Pro B70", "Intel", "workstation", 32, 608, 183.5),
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


# Other unified-memory systems. Their GPU FP16 figures are not published and are derived:
# DGX Spark from NVIDIA's "1 PFLOP FP4 with sparsity" with the GeForce Blackwell ratio (/8),
# Ryzen AI Max from RDNA 3.5 compute units x 512 FP16 FLOPs/clock x boost clock. CPU figures
# are rough.
OTHER_UNIFIED: dict[str, UnifiedChip] = {
    c.name: c
    for c in [
        UnifiedChip("nvidia-dgx-spark", 48, 273, 125, 20, 1.5, (128,), "NVIDIA", "cuda",
                    label="DGX Spark (GB10)"),
        UnifiedChip("nvidia-jetson-agx-orin-64gb", 16, 204.8, 43, 12, 0.42, (64,), "NVIDIA", "cuda",
                    label="Jetson AGX Orin 64GB"),
        UnifiedChip("amd-ryzen-ai-max-plus-395", 40, 256, 59.4, 16, 2.07, (128,), "AMD", "rocm",
                    label="Ryzen AI Max+ 395 (Radeon 8060S)"),
        UnifiedChip("amd-ryzen-ai-max-390", 32, 256, 45.9, 12, 1.55, (128,), "AMD", "rocm",
                    label="Ryzen AI Max 390 (Radeon 8050S)"),
        UnifiedChip("amd-ryzen-ai-max-385", 32, 256, 45.9, 8, 1.04, (128,), "AMD", "rocm",
                    label="Ryzen AI Max 385 (Radeon 8050S)"),
    ]
}

UNIFIED_PRESETS: dict[str, UnifiedChip] = {**APPLE_PRESETS, **OTHER_UNIFIED}


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
