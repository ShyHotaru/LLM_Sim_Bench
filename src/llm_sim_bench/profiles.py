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
    backend: str = "cuda"  # selects efficiency factors: "cuda" or "metal"


@dataclass(frozen=True)
class CPU:
    name: str
    cores: int
    fp32_tflops: float  # peak SIMD FP32 throughput


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

    @property
    def active_b(self) -> float:
        return self.active_params_b or self.params_b


@dataclass(frozen=True)
class System:
    cpu: CPU
    ram: RAM
    gpus: tuple[GPU, ...] = ()
    unified: bool = False  # GPU shares system RAM (Apple Silicon); GPU vram_gb is ignored


@dataclass(frozen=True)
class AppleChip:
    name: str
    gpu_cores: int
    bandwidth_gbps: float
    gpu_tflops: float  # FP16 shader throughput (same rate as FP32 on Apple GPUs)
    cpu_cores: int
    cpu_tflops: float
    memory_options_gb: tuple[int, ...]


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

CPU_PRESETS: dict[str, CPU] = {
    c.name: c
    for c in [
        CPU("i5-12400", 6, 0.77),
        CPU("ryzen7-5800x", 8, 1.02),
        CPU("i7-13700k", 16, 1.80),
        CPU("ryzen9-7950x", 16, 2.30),
    ]
}

# Per-GPU-core FP32/FP16 GFLOPS by generation: 128 ALUs x 2 FLOPs x GPU clock.
_APPLE_CORE_GFLOPS = {"M1": 0.327, "M2": 0.358, "M3": 0.353, "M4": 0.404}


def _apple(name: str, gpu_cores: int, bw: float, cpu_cores: int, cpu_tflops: float,
           memory: tuple[int, ...]) -> AppleChip:
    tflops = round(gpu_cores * _APPLE_CORE_GFLOPS[name.split()[0]], 1)
    slug = f"apple-{name.lower().replace(' ', '-')}-{gpu_cores}c"
    return AppleChip(slug, gpu_cores, bw, tflops, cpu_cores, cpu_tflops, memory)


APPLE_PRESETS: dict[str, AppleChip] = {
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
    ]
}


def apple_system(chip: AppleChip, memory_gb: float) -> System:
    """A Mac where CPU and GPU share one pool of unified memory."""
    return System(
        cpu=CPU(f"{chip.name}-cpu", chip.cpu_cores, chip.cpu_tflops),
        ram=RAM(memory_gb, chip.bandwidth_gbps, "unified"),
        gpus=(GPU(chip.name, memory_gb, chip.bandwidth_gbps, chip.gpu_tflops, "metal"),),
        unified=True,
    )


MODEL_PRESETS: dict[str, Model] = {
    m.name: m
    for m in [
        Model("llama-2-7b", 6.74, 32, 32, 128, 4096),
        Model("llama-3.2-3b", 3.21, 28, 8, 128, 131072),
        Model("llama-3.1-8b", 8.03, 32, 8, 128, 131072),
        Model("llama-3.1-70b", 70.6, 80, 8, 128, 131072),
        Model("mistral-7b", 7.24, 32, 8, 128, 32768),
        Model("gemma-2-9b", 9.24, 42, 8, 256, 8192),
        Model("qwen2.5-7b", 7.62, 28, 4, 128, 131072),
        Model("qwen2.5-14b", 14.8, 48, 8, 128, 131072),
        Model("qwen2.5-32b", 32.8, 64, 8, 128, 131072),
        Model("qwen3-8b", 8.19, 36, 8, 128, 32768),
        Model("qwen3-14b", 14.8, 40, 8, 128, 32768),
        Model("qwen3-32b", 32.8, 64, 8, 128, 32768),
        Model("qwen3-30b-a3b", 30.5, 48, 4, 128, 131072, active_params_b=3.3),
        Model("gpt-oss-20b", 21.0, 24, 8, 64, 131072, active_params_b=3.6),
        Model("gpt-oss-120b", 117.0, 36, 8, 64, 131072, active_params_b=5.1),
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
