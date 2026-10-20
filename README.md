# LLM Sim Bench

Simulate a local LLM rig before you build it. Describe a virtual CPU, RAM, GPU(s) and model,
get an estimate of memory use and tokens/sec, then run a mock inference engine that streams
dummy tokens at that exact speed — in the terminal or behind an OpenAI-compatible API — so you
can *feel* the latency in your real tools before spending money on hardware.

## Why

Spec sheets do not tell you whether 15 tok/s is pleasant or painful for your workflow, or
whether a model spills out of VRAM into system RAM. LLM Sim Bench answers three questions:

1. **Does it fit?** Weights + KV cache vs. VRAM and RAM, with llama.cpp-style layer offloading.
2. **How fast is it?** Prefill (prompt processing) and decode (generation) tok/s, including the
   slowdown as the context fills up.
3. **How does that feel?** A speed-matched mock that your chat UI, agent or script can call.

## Quick start

Requires Python 3.11+. No third-party dependencies.

```bash
pip install -e .
llmsim list
llmsim estimate --gpu rtx3080-10gb --model qwen2.5-14b --quant Q4_K_M --ctx 8192
llmsim run --gpu rtx4090-24gb --model llama-3.1-8b --prompt-tokens 2000 --max-tokens 300
llmsim estimate --apple apple-m4-max-40c --ram-gb 64 --model qwen3-32b
llmsim serve --config examples/rtx3080_desktop.toml --port 8000
llmsim calibrate
```

Without installing: `PYTHONPATH=src python -m llm_sim_bench ...`

Example estimate:

```
System    CPU ryzen7-5800x | RAM 64 GB ddr4-3200 x2 (51.2 GB/s) | GPU rtx3080-10gb
Model     qwen2.5-14b (14.8B) Q4_K_M, context 8192
Memory    weights 9.05 GB + KV cache 1.61 GB = 10.66 GB
Placement
  rtx3080-10gb              40 layers    8.88 GB
  cpu:ryzen7-5800x           8 layers    1.78 GB
Fits      yes
Decode    ~16.0 tok/s (empty context)  ->  ~13.6 tok/s (full context)
Note      8/48 layers offloaded to CPU
```

## Web app

`web/` is a static site with the same estimator ported to JavaScript: pick a PC (GPUs, CPU,
RAM), a Mac, a unified-memory PC (DGX Spark, Ryzen AI Max) or a server (data-center GPUs and
server CPUs), a model and a quantization, and watch mock tokens stream at the simulated speed.
Lists show the newest parts first, and a search box finds any GPU, CPU or chip and switches to
the right tab. Add a second configuration to compare both side by side. The whole setup is
kept in the URL, so a link reproduces it. The interface is available in English, Korean and
Japanese (`web/locales/`); it follows the browser language, a `?lang=ko` style URL parameter, or
the picker in the header.

```bash
llmsim web            # then open http://127.0.0.1:8000
```

Without installing, run `PYTHONPATH=src python -m llm_sim_bench web` from the repository root.
The page must be served over HTTP; opening `index.html` as a file does not work.

`.github/workflows/pages.yml` publishes `web/` to GitHub Pages after the tests pass on `main`.
It deploys only once Pages is enabled in the repository settings with GitHub Actions as the
source.

The mock stream plays real text: an LLM-style answer or a multi-turn conversation in each
language, or excerpts of public-domain books. Text is split into tokens at the rate measured
with each model family's real tokenizer, so a Korean answer on a Llama 2 model streams far fewer
characters per second than on Llama 3 at the same tokens per second. You can also stream your
own text: it is kept only in your browser's local storage, never put into shared links, and
split at the rate of the sample text in the language it looks like. See
[`data/README.md`](data/README.md) for sources and licensing.

Python is the reference implementation. `llmsim export-web` writes `web/data/presets.json`
and the golden test vectors in `web/test/golden.json`; rerun it after changing presets,
measurement data or the estimator, then run `node --test web/test/*.test.mjs`.

## Configuring a virtual system

Every option can be passed as a flag or in a TOML file (`--config`); flags win.

| Flag | Meaning |
| --- | --- |
| `--cpu` | CPU preset |
| `--ram-gb`, `--ram-kind`, `--ram-channels` | RAM size, DDR kind (e.g. `ddr5-6000`) and channel count |
| `--gpu` | GPU preset; repeat for multi-GPU; `--no-gpu` for CPU-only |
| `--apple` | Apple Silicon chip preset; `--ram-gb` then sets the unified memory size |
| `--model`, `--quant`, `--ctx`, `--kv-bits` | Model preset, quantization, context length, KV cache precision |
| `--decode-tps`, `--prefill-tps` | Skip the estimate and force a target speed for the mock |

Custom hardware or models that are not presets can be defined field-by-field in TOML. See
[`examples/rtx3080_desktop.toml`](examples/rtx3080_desktop.toml).

## Mock OpenAI-compatible server

`llmsim serve` exposes `GET /v1/models` and `POST /v1/chat/completions` (streaming and
non-streaming). Point any OpenAI client or chat UI at `http://127.0.0.1:8000/v1` with any API
key. Time-to-first-token follows the simulated prefill speed for the prompt length, and each
token after that follows the simulated decode speed at the current context fill.

## How the estimate works

- **Weights** = parameters × bits-per-weight of the quantization.
- **KV cache** = 2 × layers × KV heads × head dim × bytes × context.
- **Placement**: whole layers go to GPUs first (minus a VRAM reserve), the rest to system RAM.
- **Decode** is memory-bandwidth bound: per token, each device reads its share of the active
  weights and the filled KV cache. Device times add up (layers run sequentially).
- **Overhead**: each layer adds a fixed per-token cost (kernel launches, synchronization).
- **Prefill** is compute bound: about 2 FLOPs per active parameter per prompt token.
- **Unified memory** (Apple Silicon): the GPU may use about 75 % of system RAM.
- Peak specs are scaled by efficiency factors (`Efficiency` in `estimator.py`).

All preset numbers are approximate public specs. Treat results as order-of-magnitude guidance.

### Calibration

Efficiency factors for every backend are fitted to public llama.cpp results
([`data/`](data/README.md)): the CUDA, ROCm, Vulkan, SYCL and Apple Silicon scoreboards and
CPU-only runs. Metal uses only results from 2026 builds, because 2023 builds had much more
per-token overhead. Measured devices also get their own prompt-processing factor, and
measured CPUs their own memory-bandwidth factor. `llmsim calibrate` refits everything and
prints the error against every record (`--write` stores the per-device factors); the test
suite fails if the defaults drift from the data. Current median error:

| Backend | Decode | Prefill |
| --- | --- | --- |
| CUDA (55 results) | 6.4 % | 6.1 % |
| ROCm (33) | 4.3 % | 2.6 % |
| Vulkan (62) | 7.6 % | 5.3 % |
| SYCL (21) | 1.6 % | 29 % |
| Metal (12, M5 family) | 1.4 % | 0.6 % |
| CPU (174 decode, 29 prefill) | 3.1 % | 6.4 % |

GPU calibration data is Llama 2 7B at short context, so larger models, long contexts and CPU
offload are extrapolations. The app marks results that rest on uncalibrated parts (a backend
or device without measurements, or a mixture-of-experts model).

### Known limitations

- Prefill with partial CPU offload is pessimistic: real runtimes push large prompt batches
  through the GPU even for CPU-resident layers.
- MoE decode is optimistic: routing overhead and shared weights are not modeled yet.
- Single request only; no batching or concurrency effects.
- Devices without their own measurements use backend averages for prompt processing, which can
  be far off (data-center GPUs reach much less of their peak than consumer cards).

## Roadmap

Before a public release:

- Enable GitHub Pages (the workflow is ready) and choose a custom domain.

Calibration data:

- Long-context results (`llama-bench -d`): decode slowdown and prompt processing at depth are
  modeled but not calibrated.
- Larger models and multi-GPU layer splits (all GPU data is Llama 2 7B on one GPU).
- Mixture-of-experts routing overhead (currently excluded from calibration and flagged).
- CPU prompt processing on more CPUs (only native-build data for a few CPUs).
- Apple M1–M4 on current llama.cpp builds (only 2023 builds are recorded), an M6 preset, and
  the M5 Ultra prompt-processing gap.
- Intel SYCL FP16 vs. FP32 builds as a separate option.
- Prompt processing for unmeasured data-center GPUs (H200, B200, MI325X, MI355X).
- A human pass over transcribed scoreboard tables (they were read with an automated
  summarizer and spot-checked).

Features:

- Start playback at a given context fill to feel the slowdown at long contexts.
- Show CPU-only measurements in the benchmark check table.
- Use the text presets and tokenizer ratios in the CLI and the OpenAI-compatible mock server.
- Concurrent-request simulation and report export (JSON/Markdown).

## Development

```bash
PYTHONPATH=src python -m unittest discover -s tests -v
PYTHONPATH=src python -m llm_sim_bench export-web   # regenerate web data
node --test web/test/*.test.mjs                     # JavaScript estimator vs. Python
```

## License

| What | License |
| --- | --- |
| Code (everything outside `data/`, `web/data/` and `web/fonts/`) | [MIT](LICENSE) |
| Data in `data/` and `web/data/`, and the sample texts written for this project | [CC BY 4.0](LICENSE-DATA) |
| SYCL scoreboard records (`data/measurements/llamacpp-sycl.jsonl`) | Not licensed by this project; the source asks for no commercial use |
| Book excerpts in `web/data/texts/` | Public domain |
| Fonts in `web/fonts/` (Noto Sans KR, Noto Sans JP, JetBrains Mono) | SIL Open Font License 1.1 |

Benchmark results come from public llama.cpp scoreboards, OpenBenchmarking.org and other
sources; see [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md) for each source's terms.
