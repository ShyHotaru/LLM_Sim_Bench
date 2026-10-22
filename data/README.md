# Measurement data

Real-world llama.cpp benchmark results used to calibrate and test the estimator
(`llmsimbench calibrate`, `tests/test_measurements.py`).

## Files

| File | Source | Records |
| --- | --- | --- |
| `measurements/llamacpp-scoreboards.jsonl` | llama.cpp community scoreboards: [CUDA (#15013)](https://github.com/ggml-org/llama.cpp/discussions/15013), [Apple Silicon (#4167)](https://github.com/ggml-org/llama.cpp/discussions/4167) | 191 |
| `measurements/llamacpp-apple-2026.jsonl` | Apple Silicon scoreboard rows measured on 2026 builds (`c1d0e7a`, `60081bb`, `0c1e570`): M5 Pro, M5 Max, M5 Ultra, M6, and an M1 Ultra from a comment | 18 |
| `measurements/llamacpp-rocm.jsonl` | [llama.cpp ROCm/HIP scoreboard (#15021)](https://github.com/ggml-org/llama.cpp/discussions/15021); MI300X runs from its comments and [#14640](https://github.com/ggml-org/llama.cpp/discussions/14640) | 50 |
| `measurements/llamacpp-vulkan.jsonl` | [llama.cpp Vulkan scoreboard (#10879)](https://github.com/ggml-org/llama.cpp/discussions/10879) | 237 |
| `measurements/llamacpp-sycl.jsonl` | [llama.cpp Intel SYCL scoreboard (#23313)](https://github.com/ggml-org/llama.cpp/discussions/23313); its opening post asks that the data not be used for commercial purposes (see [Licensing](#licensing)) | 104 |
| `measurements/llamacpp-batched.jsonl` | `llama-batched-bench` tables in llama.cpp's [`benches/`](https://github.com/ggml-org/llama.cpp/tree/c7fc5094179bf8a9c73c18593db714f8da41a402/benches) folder (DGX Spark, Nemotron on DGX Spark, Mac M2 Ultra), MIT license; written by `tools/import_llamacpp_benches.py` from a llama.cpp checkout | 252 |
| `measurements/llamacpp-depth.jsonl` | `llama-bench` tables at depths 0–32768 in the same `benches/` files (pp2048 and tg32 at each depth), written by the same tool | 70 |
| `measurements/llamacpp-multi-gpu.jsonl` | `-sm layer` column of the tables in [llama.cpp PR #19378](https://github.com/ggml-org/llama.cpp/pull/19378): Llama 3 8B on two and four RTX 4090s, pp512 and tg128 at depths up to 131072 | 10 |
| `measurements/cpu-runs.jsonl` | CPU-only runs: OpenBenchmarking.org `pts/llama-cpp` results (read through Wayback Machine copies), [ik_llama.cpp #164](https://github.com/ikawrakow/ik_llama.cpp/discussions/164) (mainline column), [llama.cpp #11733](https://github.com/ggml-org/llama.cpp/discussions/11733), [geerlingguy/ai-benchmarks](https://github.com/geerlingguy/ai-benchmarks) | 1594 |

`llmsimbench export-web` copies the calibration error stats and the rows of the web app's check
tables into `web/data/presets.json` and `src/llm_sim_bench/data/calibration.json` (for
reports from the installed command-line tool); do not edit the copies.

GPU scoreboard records are Llama 2 7B Q4_0 (SYCL also lists other models), `llama-bench -p 512
-n 128`. Values were transcribed from the scoreboard tables on the `retrieved` date. Only the
numbers and hardware descriptions were copied; contributor names were not.

## Licensing

This project's compilation of the records is licensed under CC BY 4.0
([`LICENSE-DATA`](../LICENSE-DATA)); the results themselves were reported by the sources above.
The SYCL scoreboard's opening post asks that its data not be used for commercial purposes, so
`llamacpp-sycl.jsonl` is excluded from that license. See
[`THIRD_PARTY_NOTICES.md`](../THIRD_PARTY_NOTICES.md) for each source's terms.

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
| `hardware.gpu_preset` | Matching preset in `profiles.py`, or `null`; only records with a preset are used for calibration. A name is linked only when it identifies a single configuration (e.g. "RTX 3080" without a memory size is not linked, since the 10 GB and 12 GB cards differ in bandwidth) |
| `hardware.cpu_preset`, `hardware.dimms`, `hardware.mem_mts`, `hardware.ram_channels`, `hardware.sockets`, `hardware.threads` | CPU runs only: matching CPU preset, DIMM count, memory speed (MT/s), memory channels, sockets and threads used, each when stated |
| `notes` | Optional free-form details from the source (test profile, run id, standard error, ...) |
| `software.runtime`, `software.backend`, `software.commit`, `software.flash_attn` | Runtime, backend (`CUDA`, `Metal`, `ROCm`, `Vulkan`, `SYCL`, `CPU`), build commit, flash attention on/off |
| `software.sycl_f16` | SYCL records only: built with `GGML_SYCL_F16=ON` (`true`) or the default FP32 build (`false`), from the scoreboard's `Fp32/FP16` column; FP16 builds are calibrated as their own backend |
| `model.name`, `model.quant`, `model.file` | Model preset name, quantization, GGUF file name |
| `run.n_gpu_layers`, `run.n_prompt`, `run.n_gen`, `run.depth` | llama-bench `-ngl` (`-1`: the tool's default, as many layers as fit), `-p`, `-n`, `-d` |
| `run.split_mode` | Several GPUs only: llama-bench `-sm` (`layer`, `row`, `tensor`). Only `layer` runs are simulated (checked, not fitted) |
| `run.tool`, `run.n_parallel`, `run.pp_shared`, `run.n_batch`, `run.n_ubatch` | Records from llama.cpp's `benches/` and PR #19378 only: the benchmark tool (`llama-batched-bench` or `llama-bench`), and for batched records (required with the `batched` flag, except the batch sizes) the number of sequences run at once (`B`), whether they share one prompt (`-pps`), `-b` and `-ub`. Batched `results.pp_tps` and `results.tg_tps` are throughput over all sequences (`S_PP`, `S_TG`). Records with `run.tool` stay out of the main fit and calibrate long contexts, mixture-of-experts models and concurrency instead |
| `results.pp_tps`, `results.pp_std`, `results.tg_tps`, `results.tg_std` | Prefill and decode tokens/s with standard deviation |
| `flags` | Quality flags. Excluded from calibration: `suspect` (the source value looks wrong), `old_build` (a llama.cpp build much older than current ones), `multi-socket`, `mobile-class-cpu`, `cpu-model-inferred`, `batched` (many parallel sequences), `virtual-gpu` (a virtual function on a shared cloud host, such as the MI300X VF results, which reach far less of the GPU's bandwidth than the model expects). Partly excluded: `openblas-build` (prompt processing ran through OpenBLAS, not the default CPU path; decode is kept), `tg_thread_limited` (decode run with too few threads) |

## Notes and gaps

- The M1–M4 rows were measured on llama.cpp commit `8e672ef` (November 2023) and are flagged
  `old_build`: newer builds have far less per-token overhead (the fitted Metal per-layer cost
  drops from about 184 µs to 72 µs), so Metal is calibrated on the 2026 M5 rows only. CUDA
  rows use a mix of 2025 builds.
- The M5 row with a 10-core GPU was not recorded: the source lists four values without making
  clear which quantization columns they belong to.
- The MI300X rows are flagged `virtual-gpu`. The scoreboard's MI300X row without flash
  attention combines the prompt speed of one run (default build `2bf3fbf`) with the decode
  speed of another (rocWMMA build `ee3a9fc`); the records keep each run separate.
- The M6 rows follow the original comment (12-core GPU, 32 GB, build `0c1e570`); the
  scoreboard table lists 8 GPU cores and commit `c1d0e7a` for them.
- M5 GPU compute assumes Apple's "over 4x M4 peak GPU AI compute" for the Neural Accelerators;
  it matches measured prompt processing within about 10 %, except the four-die M5 Ultra, which
  is about 20 % slower than that scaling suggests. The M6 assumes Apple's "nearly 30 percent"
  more than the 10-core M5 and matches within about 6 %.
- One CUDA row is flagged `suspect` (Quadro T1000 with flash attention: the source lists the
  same value for prefill and decode).
- CPU runs give DIMM counts rather than memory channels. Calibration assumes DIMMs fill
  separate channels first, up to the CPU's channel count.
- Prompt-processing efficiency differs strongly between devices (data-center GPUs reach far
  less of their peak at this batch size than consumer cards; FP16 SYCL builds process prompts
  about 2.7 times faster than the default build), so every measured device gets its own factor
  in `src/llm_sim_bench/device_calibration.json` (`llmsimbench calibrate --write`). The same file
  holds per-CPU decode bandwidth factors: single-CCD Ryzen parts reach about 0.6 of their
  DRAM bandwidth, dual-CCD parts about 0.85.
- Mixture-of-experts models stay out of the main fit; their overhead is fitted to the
  `benches/` depth-0 results (`llmsimbench calibrate`).
- Depth data (`llamacpp-depth.jsonl`) covers DGX Spark and the M2 Ultra only. Each run is
  compared with its own depth-0 result, so the slowdown fit does not depend on the device's
  calibration. The M2 Ultra rows have no preset (60 or 76 GPU cores); calibration uses them only
  where the core count does not matter (decode, and how much prompt processing slows down).
  `llama-bench` printed no `ngl` column for the M2 Ultra and Nemotron runs, so their
  `run.n_gpu_layers` is null.
- Batched records: the M2 Ultra file does not say whether its GPU has 60 or 76 cores, so those
  rows have no preset and are not used. Gemma 3, GLM-4.7-Flash and Nemotron have no model
  preset. The Nemotron Super quantization is recorded as `Q4_K`, as the source prints it.
  Only dense models fit `*_batch_compute` (Qwen2.5-Coder 7B, which shares Qwen2.5 7B's
  architecture); the mixture-of-experts rows are only compared in `llmsimbench calibrate`.

# Output texts and tokenizer ratios

The mock (web app, `llmsimbench run` and `serve`) streams real text instead of filler. Texts live in `web/data/texts/`
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
`llmsimbench export-web` copies the texts and `tokenizer_ratios.json` into
`src/llm_sim_bench/data/` for the command-line tool and the mock server (do not edit the
copies), and writes `web/test/chunks.json` so the web tests check that `web/js/tokenize.js`
splits text exactly like `src/llm_sim_bench/texts.py`.

`tokenizer_ratios.json` holds the measured characters per token for every text and tokenizer
family, produced by `tools/measure_tokenizers.py` (needs `pip install tokenizers`). Only the
counts are stored; tokenizer files are cached in `.cache/` and never committed. The stored
Llama 2, Llama 3 and Gemma 2 ratios were measured with tokenizers read from ungated community
mirrors of the gated official repositories (the file records which); the tool now reads the
official repositories and needs an `HF_TOKEN` for an account that accepted their licenses.

# Hardware preset sources

Presets in `src/llm_sim_bench/profiles.py` are approximate public specs.

- **AMD CPUs** (Ryzen 5000/7000/9000, Threadripper 7000/9000, EPYC 9004/9005): core counts,
  base and boost clocks, memory channels and speeds from amd.com product pages, cross-checked
  with Wikipedia.
- **Intel CPUs**: intel.com and ARK were not reachable, so values come from Wikipedia (Alder
  Lake, Raptor Lake, Arrow Lake, Sapphire Rapids, Granite Rapids), with base clocks and memory
  speeds checked against Intel's 13th/14th Gen Core datasheet (743844) and execution units
  against the Intel Optimization Reference Manual (671488).
- **CPU FP32 throughput** = cores × midpoint of base and boost clock × FLOPs per cycle per core.
  The Gracemont (16) and Skymont (32) FLOPs per cycle are inferred from published port layouts
  rather than stated by Intel. CPU efficiency factors are not calibrated, so CPU prefill
  estimates are rough.
- **Default memory per CPU** is the fastest officially supported speed with one DIMM per
  channel; more DIMMs per channel usually run slower, and Arrow Lake reaches DDR5-6400 only
  with CUDIMMs.
- **Apple M5 family**: Apple Newsroom announcements (October 2025, March 2026) and the Mac
  Studio and MacBook Pro tech specs pages.
- **GPUs**: NVIDIA whitepapers and datasheets, AMD product pages and datasheets, Intel product
  pages and datasheets (through Wayback Machine copies), cross-checked with Wikipedia. FP16 is
  the dense tensor/matrix figure without sparsity. Where vendors publish only "AI TOPS" or
  sparse figures, it is derived with ratios confirmed on cards of the same generation
  (GeForce 40: AI TOPS / 4; Blackwell: FP4 sparse / 8; RTX Ada: FP8 sparse / 4; Intel: INT8
  TOPS / 2). RDNA2 cards have no matrix units, so their packed FP16 vector peak is used.
- **Unified-memory PCs**: DGX Spark FP16 is derived from NVIDIA's "1 PFLOP FP4 with sparsity"
  (/8); Ryzen AI Max FP16 from AMD's compute-unit count and GPU clock (RDNA 3.5, 512 FP16
  FLOPs per CU per clock). CPU figures for these systems are rough. Jetson AGX Thor was left
  out because no usable FP16 figure is published.
