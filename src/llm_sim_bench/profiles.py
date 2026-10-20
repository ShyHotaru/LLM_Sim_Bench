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
        GPU("rtx3060-12gb", 12, 360, 51),
        GPU("rtx3080-10gb", 10, 760, 119),
        GPU("rtx3090-24gb", 24, 936, 142),
        GPU("rtx4060ti-16gb", 16, 288, 88),
        GPU("rtx4090-24gb", 24, 1008, 330),
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

MODEL_PRESETS: dict[str, Model] = {
    m.name: m
    for m in [
        Model("llama-3.1-8b", 8.03, 32, 8, 128, 131072),
        Model("llama-3.1-70b", 70.6, 80, 8, 128, 131072),
        Model("mistral-7b", 7.24, 32, 8, 128, 32768),
        Model("gemma-2-9b", 9.24, 42, 8, 256, 8192),
        Model("qwen2.5-7b", 7.62, 28, 4, 128, 131072),
        Model("qwen2.5-14b", 14.8, 48, 8, 128, 131072),
        Model("qwen2.5-32b", 32.8, 64, 8, 128, 131072),
        Model("qwen3-30b-a3b", 30.5, 48, 4, 128, 131072, active_params_b=3.3),
        Model("gpt-oss-20b", 21.0, 24, 8, 64, 131072, active_params_b=3.6),
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
