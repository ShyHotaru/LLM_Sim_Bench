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
RAM) or a Mac (chip, unified memory), a model and a quantization, compare two configurations
side by side, and watch both stream mock tokens at their simulated speed. The whole setup is
kept in the URL, so a link reproduces it.

```bash
python -m http.server -d web 8000   # then open http://127.0.0.1:8000
```

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

CUDA and Metal efficiency factors are fitted to 109 public llama.cpp scoreboard results
([`data/`](data/README.md)). `llmsim calibrate` refits them and prints the error against every
record; the test suite fails if the defaults drift from the data. Current median error:

| Backend | Decode | Prefill |
| --- | --- | --- |
| CUDA (30 results) | 4.0 % | 7.0 % |
| Metal (79 results) | 3.2 % | 4.5 % |

All calibration data is Llama 2 7B at short context, so larger models, long contexts and CPU
offload are extrapolations.

### Known limitations

- Prefill with partial CPU offload is pessimistic: real runtimes push large prompt batches
  through the GPU even for CPU-resident layers.
- MoE decode is optimistic: routing overhead and shared weights are not modeled yet.
- Single request only; no batching or concurrency effects.
- CPU, ROCm and Vulkan factors are not calibrated yet. Dual-die Apple Ultra chips decode up to
  ~20 % slower than estimated, and Turing (RTX 20) prefill is overestimated.

## Roadmap

- More calibration data: CPU-only, AMD, long-context (`llama-bench -d`) and larger models.
- AMD GPU presets.
- Concurrent-request simulation and a small benchmark harness that replays a prompt set.
- Report export (JSON/Markdown) for comparing candidate builds side by side.

## Development

```bash
PYTHONPATH=src python -m unittest discover -s tests -v
PYTHONPATH=src python -m llm_sim_bench export-web   # regenerate web data
node --test web/test/*.test.mjs                     # JavaScript estimator vs. Python
```
