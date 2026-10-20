# Measurement data

Real-world llama.cpp benchmark results used to calibrate and test the estimator
(`llmsim calibrate`, `tests/test_measurements.py`).

## Files

| File | Source | Records |
| --- | --- | --- |
| `measurements/llamacpp-scoreboards.jsonl` | llama.cpp community scoreboards: [CUDA (#15013)](https://github.com/ggml-org/llama.cpp/discussions/15013), [Apple Silicon (#4167)](https://github.com/ggml-org/llama.cpp/discussions/4167) | 191 |
| `measurements/llamacpp-apple-2026.jsonl` | Apple Silicon scoreboard rows measured on 2026 builds (`c1d0e7a`, `60081bb`): M5 Pro, M5 Max, M5 Ultra, M6 | 15 |

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
| `flags` | Quality flags, each excluding the record from calibration. `suspect`: the source value looks wrong. `old_build`: measured on a llama.cpp build much older than current ones (kept for reference) |

## Notes and gaps

- The M1–M4 rows were measured on llama.cpp commit `8e672ef` (November 2023) and are flagged
  `old_build`: newer builds have far less per-token overhead (the fitted Metal per-layer cost
  drops from about 184 µs to 72 µs), so Metal is calibrated on the 2026 M5 rows only. CUDA
  rows use a mix of 2025 builds.
- The M5 row with a 10-core GPU was not recorded: the source lists four values without making
  clear which quantization columns they belong to.
- M6 rows have no matching preset yet (`gpu_preset` is null).
- M5 GPU compute assumes Apple's "over 4x M4 peak GPU AI compute" for the Neural Accelerators;
  it matches measured prompt processing within about 10 %, except the four-die M5 Ultra, which
  is about 40 % slower than that scaling suggests.
- One CUDA row is flagged `suspect` (Quadro T1000 with flash attention: the source lists the
  same value for prefill and decode).
- No CPU-only, ROCm or Vulkan data yet, so CPU efficiency factors are not calibrated.
- No depth (`-d`) data yet, so the slowdown as the context fills is not calibrated.

# Output texts and tokenizer ratios

The web mock streams real text instead of filler. Texts live in `web/data/texts/`
(`manifest.json` lists them):

- **Authored**: an LLM-style answer and a four-turn conversation in Korean, Japanese and
  English, written for this project.
- **Public domain**: excerpts of works whose author died in or before 1925 and that were first
  published before 1931, so they are free in every country (including life+100 terms and the
  US publication rule). Japanese texts come from 青空文庫 (Aozora Bunko), which allows free
  reuse and asks that the edition and transcription notes be kept; they are stored in
  `credits.json` and shown in the app. English texts come from Project Gutenberg via the
  GITenberg mirror (its site forbids scripted downloads); the Project Gutenberg license and
  name are removed from the excerpts as its license requires, and the app links to the ebook
  page as an acknowledgement. 夢十夜 was left out because its opening contains characters that
  Aozora renders as images.

`tools/build_texts.py` rebuilds the excerpts and `credits.json` from the manifest.

`tokenizer_ratios.json` holds the measured characters per token for every text and tokenizer
family, produced by `tools/measure_tokenizers.py` (needs `pip install tokenizers`). Only the
counts are stored; tokenizer files are cached in `.cache/` and never committed. Llama 2,
Llama 3 and Gemma 2 tokenizers were read from ungated community mirrors of the gated official
repositories.
