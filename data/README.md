# Measurement data

Real-world llama.cpp benchmark results used to calibrate and test the estimator
(`llmsim calibrate`, `tests/test_measurements.py`).

## Files

| File | Source | Records |
| --- | --- | --- |
| `measurements/llamacpp-scoreboards.jsonl` | llama.cpp community scoreboards: [CUDA (#15013)](https://github.com/ggml-org/llama.cpp/discussions/15013), [Apple Silicon (#4167)](https://github.com/ggml-org/llama.cpp/discussions/4167) | 191 |

All records are Llama 2 7B, `llama-bench -p 512 -n 128` with every layer on the GPU.
Values were transcribed from the scoreboard tables on the `retrieved` date. Only the
numbers and hardware descriptions were copied; contributor names were not.

## Record schema

One JSON object per line. A field the source does not state is `null`; never fill it with a
guess.

| Field | Meaning |
| --- | --- |
| `id` | Unique, stable identifier |
| `source.url`, `source.title`, `source.retrieved` | Where the numbers come from and when they were copied (YYYY-MM-DD) |
| `hardware.gpu`, `hardware.gpu_count` | GPU name as written in the source, number of GPUs used |
| `hardware.vram_gb`, `hardware.memory_type`, `hardware.bus_width_bits` | Memory description from the source |
| `hardware.gpu_cores`, `hardware.bandwidth_gbps` | Apple GPU core count and memory bandwidth from the source |
| `hardware.cpu`, `hardware.ram_gb` | Host CPU and RAM, when stated |
| `hardware.gpu_preset` | Matching preset in `profiles.py`, or `null`; only records with a preset are used for calibration |
| `software.runtime`, `software.backend`, `software.commit`, `software.flash_attn` | Runtime, backend (`CUDA`, `Metal`, `ROCm`, `Vulkan`, `SYCL`, `CPU`), build commit, flash attention on/off |
| `model.name`, `model.quant`, `model.file` | Model preset name, quantization, GGUF file name |
| `run.n_gpu_layers`, `run.n_prompt`, `run.n_gen`, `run.depth` | llama-bench `-ngl`, `-p`, `-n`, `-d` |
| `results.pp_tps`, `results.pp_std`, `results.tg_tps`, `results.tg_std` | Prefill and decode tokens/s with standard deviation |
| `flags` | Quality flags. `suspect`: the source value looks wrong; excluded from calibration |

## Notes and gaps

- Apple Silicon rows were measured on llama.cpp commit `8e672ef` (November 2023); newer builds
  are faster. CUDA rows use a mix of 2025 builds.
- One CUDA row is flagged `suspect` (Quadro T1000 with flash attention: the source lists the
  same value for prefill and decode).
- No CPU-only, ROCm or Vulkan data yet, so CPU efficiency factors are not calibrated.
- No depth (`-d`) data yet, so the slowdown as the context fills is not calibrated.
