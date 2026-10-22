# LLM Sim Bench

Simulate a local LLM rig before you build it. Describe a virtual CPU, RAM, GPU(s) and model,
get an estimate of memory use and tokens/sec, then run a mock inference engine that streams
text at that exact speed — in the terminal or behind an OpenAI-compatible API — so you
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
llmsimbench list
llmsimbench estimate --gpu rtx3080-10gb --model qwen2.5-14b --quant Q4_K_M --ctx 8192
llmsimbench run --gpu rtx4090-24gb --model llama-3.1-8b --prompt-tokens 2000 --max-tokens 300
llmsimbench run --gpu rtx3060-12gb --model qwen2.5-14b --text ko-chat
llmsimbench estimate --apple apple-m4-max-40c --ram-gb 64 --model qwen3-32b
llmsimbench estimate --gpu rtx4090-24gb --model qwen3-32b --format markdown -o report.md
llmsimbench estimate --soc nvidia-dgx-spark --model qwen2.5-7b --quant Q8_0 --parallel 8
llmsimbench serve --config examples/rtx3080_desktop.toml --port 8000
llmsimbench calibrate
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
the right tab. Add a second configuration to compare both side by side. "Concurrent requests" sets the
number of llama-server slots: the result then shows the speed per request and in total for up to
that many requests at once, the memory for every slot's KV cache, and, where measured, a check
against `llama-batched-bench` results; playback shows one of that many requests that arrive
together. Playback can start with
the context already partly filled (cached, so it adds no prompt-processing time) to feel the
slowdown at long contexts. Under each result, a check table compares the simulator with real
llama.cpp results for the selected GPU, chip or, in a CPU-only setup, CPU. The whole setup is
kept in the URL, so a link reproduces it, and "Export report" downloads every configuration's
results as JSON or Markdown (see [Reports](#reports)). The interface is available in English, Korean and
Japanese (`web/locales/`); it follows the browser language, a `?lang=ko` style URL parameter, or
the picker in the header.

```bash
llmsimbench web       # then open http://127.0.0.1:8000
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

Python is the reference implementation. `llmsimbench export-web` writes `web/data/presets.json`
and the golden test vectors in `web/test/golden.json` (estimator), `web/test/chunks.json`
(text splitting) and `web/test/report.json` (reports), and copies the texts, tokenizer ratios
and calibration data into `src/llm_sim_bench/data/` for the installed package; rerun it after changing presets, measurement data, texts or the
estimator, then run `node --test web/test/*.test.mjs`.

## Configuring a virtual system

Every option can be passed as a flag or in a TOML file (`--config`); flags win.

| Flag | Meaning |
| --- | --- |
| `--cpu` | CPU preset |
| `--ram-gb`, `--ram-kind`, `--ram-channels` | RAM size, DDR kind (e.g. `ddr5-6000`) and channel count |
| `--gpu` | GPU preset; repeat for multi-GPU; `--no-gpu` for CPU-only |
| `--apple` | Apple Silicon chip preset; `--ram-gb` then sets the unified memory size |
| `--model`, `--quant`, `--ctx`, `--kv-bits` | Model preset, quantization, context length, KV cache precision |
| `--parallel` | Concurrent request slots, each with `--ctx` tokens of KV cache (llama-server `-np N -c N*ctx`) |
| `--decode-tps`, `--prefill-tps` | Skip the estimate and force a target speed for the mock |

Custom hardware or models that are not presets can be defined field-by-field in TOML. See
[`examples/rtx3080_desktop.toml`](examples/rtx3080_desktop.toml).

## Mock OpenAI-compatible server

`llmsimbench serve` exposes `GET /v1/models` and `POST /v1/chat/completions` (streaming and
non-streaming). Point any OpenAI client or chat UI at `http://127.0.0.1:8000/v1` with any API
key. Time-to-first-token follows the simulated prefill speed for the prompt length, and each
token after that follows the simulated decode speed at the current context fill.

With `--parallel N` the server behaves like llama-server with N slots: up to N requests stream at
once and share each decode step, so each one slows down as more run; further requests wait for
a free slot; and while a new request's prompt is processed, the other streams pause. The log
line of each request shows its wait (`queued=`). A forced `--decode-tps` holds for every request
however many run at once.

## Reports

`llmsimbench estimate --format json` or `--format markdown` prints a report instead of the
text summary (`-o FILE` writes it to a file). The web app's "Export report" links download the
same report for every configuration on the page, with a link back to the setup. A report holds:

- the virtual system and model, memory placement per device, and whether it fits;
- generation and prompt-processing speeds, and with `--prompt-tokens`, `--max-tokens` or
  `--depth` (in the web app, the playback settings) the time to the first token and to the
  whole reply;
- notes and "estimate only" warnings, each with a `code` and `params` as well as English text;
- the check against real llama.cpp benchmarks on the same GPU, chip or CPU;
- with `--parallel N`: speeds per request and in total for 1, 2, 4, ... N active requests,
  the times when all N slots get a request at once, and the check against `llama-batched-bench`
  results on the same hardware.

Reports are in English. The JSON has `"schema": "llmsimbench.report/1"`; numbers are not
rounded. `src/llm_sim_bench/report.py` builds it and `web/js/report.js` is its port, kept in
sync by `web/test/report.json`. `estimate` exits with 2 when the model does not fit, in every
format.

## Output texts

`run` and `serve` stream real text, the same presets as the web app (`llmsimbench texts`
lists them with their sources): sample answers and conversations written for this project in
Korean, Japanese and English, and public-domain book excerpts. `--text ID` picks one (default
`en-answer`), `--text-file PATH` streams your own UTF-8 file.

Text is split into token-sized pieces with the characters per token measured for the model's
tokenizer family (`data/tokenizer_ratios.json`), so characters appear at the rate that model
would show them; `--chars-per-token` overrides the ratio. Prompt lengths are counted with the
ratio of the sample answer in the prompt's language. An answer stops at the end of the text
(`finish_reason: "stop"`) or at `--max-tokens` / the request's `max_tokens` (`"length"`).

With a conversation preset, `run` plays every turn and prefills only each new user message,
since earlier turns stay in the KV cache. `serve` answers the k-th user message of a request
with the k-th scripted answer (wrapping around), so a chat UI follows the script; it prefills
the whole prompt on every request.

## How the estimate works

- **Weights** = parameters × bits-per-weight of the quantization.
- **KV cache** = 2 × layers × KV heads × head dim × bytes × context. Sliding-window layers
  (Gemma 2, gpt-oss) keep only their window plus one batch, as llama.cpp does.
- **Placement**: whole layers go to GPUs first (minus a VRAM reserve), the rest to system RAM.
- **Decode** is memory-bandwidth bound: per token, each device reads its share of the active
  weights (not the token embedding, which is a lookup) and the filled KV cache (attention reads
  the cache more slowly than the weights).
  Device times add up (layers run sequentially).
- **Overhead**: each layer adds a fixed per-token cost (kernel launches, synchronization).
- **Prefill** is compute bound: about 2 FLOPs per multiplied weight per prompt token (without
  the output layer, which llama.cpp runs only for the last prompt token), plus attention over
  every token already in the context, so prompts get slower as it fills.
- **Mixture-of-experts** models read their shared weights and the routed experts; their layers
  cost extra time per token, and their prompt processing reaches less of the GPU's compute.
  gpt-oss MXFP4 files keep everything but the experts in Q8_0.
- **Concurrent requests** (llama-server `--parallel` slots, continuous batching) share each
  decode step: the weights are read once for the whole batch, every request reads its own KV
  cache, and each request beyond the first adds compute at a fitted fraction of the
  prompt-processing rate. Mixture-of-experts models read the experts any request in the batch
  is routed to, assuming random routing. Prompts that arrive together share prompt processing.
- **Unified memory** (Apple Silicon): the GPU may use about 75 % of system RAM.
- Peak specs are scaled by efficiency factors (`Efficiency` in `estimator.py`).

All preset numbers are approximate public specs. Treat results as order-of-magnitude guidance.

### Calibration

Efficiency factors for every backend are fitted to public llama.cpp results
([`data/`](data/README.md)): the CUDA, ROCm, Vulkan, SYCL and Apple Silicon scoreboards and
CPU-only runs. Metal uses only results from 2026 builds, because 2023 builds had much more
per-token overhead. Measured devices also get their own prompt-processing factor, and
measured CPUs their own memory-bandwidth factor. `llmsimbench calibrate` refits everything and
prints the error against every record (`--write` stores the per-device factors); the test
suite fails if the defaults drift from the data. Current median error:

| Backend | Decode | Prefill |
| --- | --- | --- |
| CUDA (55 results) | 6.0 % | 6.0 % |
| ROCm (31) | 5.1 % | 3.3 % |
| Vulkan (62) | 7.6 % | 5.3 % |
| SYCL, default FP32 build (9) | 2.0 % | 4.2 % |
| SYCL, FP16 build (12) | 2.1 % | 25 % |
| Metal (18: M5 family, M6, M1 Ultra) | 3.4 % | 0.9 % |
| CPU (174 decode, 29 prefill) | 1.0 % | 5.2 % |

The MI300X results come from a virtual function on a cloud host and reach far less of the
GPU's bandwidth than the model expects (a 70B model decodes about a third slower than
estimated), so they are shown in the app's check table but not fitted.

Intel GPUs have two SYCL choices: llama.cpp's default build and one with `GGML_SYCL_F16=ON`,
whose prompt processing is about 2.7 times faster (more without flash attention, less with
it; the model does not distinguish the two, hence the larger error).

Long contexts and mixture-of-experts models are fitted to `llama-bench` runs at depths of
up to 32768 tokens in llama.cpp's `benches/` folder, DGX Spark (CUDA) and an M2 Ultra
(Metal), with Qwen2.5-Coder 7B, Qwen3-Coder 30B-A3B, gpt-oss 20B and 120B, and for CUDA also
Llama 3 8B on two and four RTX 4090s at up to 131072 tokens (llama.cpp PR #19378). Given the
speed at an empty context, the speed at depth is predicted with a median error of 2.7 % (11 %
at most, 27 results) on CUDA and 1.0 % (2.8 %, 16) on Metal for decode, and 3.6 % (24 %, 15)
and 0.4 % (0.5 %, 4) for prompt processing of dense models. The two CUDA devices slow down
differently, so the CUDA factors are a compromise. Mixture-of-experts decode is within 5 %
(CUDA) and 3 % (Metal); their prompt processing is calibrated on CUDA only, within 31 %.
Other backends have no such measurements and keep the plain model (no extra cost for reading
the KV cache, attention at the full compute rate, no MoE overhead); the app says so.

Several GPUs split by layers are checked, not fitted: on two and four RTX 4090s the estimate
is 10 % slow for decode (median; 13 % at most); short prompts are within 14 % for Q4_0 but
16–22 % fast for F16, whose prompts reach less of the GPU's compute (the model does not
distinguish quantizations there). Long prompts on several GPUs are faster than estimated because llama.cpp pipelines
them across the GPUs, which the model does not.

Concurrent decode is fitted to how the total speed grows from one sequence to 2–32 in
`llama-batched-bench` results from llama.cpp's own `benches/` folder: 6.1 % median error (12 %
at most) over 15 CUDA results, all of them Qwen2.5-Coder 7B on a DGX Spark. Every other
backend uses the CUDA factor, so concurrency estimates are rough everywhere else.

Most GPU calibration data is Llama 2 7B at short context, so larger dense models and CPU
offload are extrapolations. The app marks results that rest on uncalibrated
parts (a backend or device without measurements, long contexts or mixture-of-experts models on
a backend without benches/ results).

### Known limitations

- Prefill with partial CPU offload is pessimistic: real runtimes push large prompt batches
  through the GPU even for CPU-resident layers.
- Long-context and MoE factors come from one device per backend (DGX Spark, M2 Ultra) and
  only CUDA and Metal; MoE prompt processing differs by model (within 27 % on CUDA).
- Concurrent requests are calibrated on one dense model on one device (DGX Spark). For
  mixture-of-experts models the measured speed-up is much larger than estimated (at 32
  sequences, the estimate is 11–54 % low), so their concurrent totals are pessimistic.
- Concurrent prompt processing is modeled as one shared queue; llama-server's chunked
  scheduling of prompts between decode steps is not modeled.
- Devices without their own measurements use backend averages for prompt processing, which can
  be far off (data-center GPUs reach much less of their peak than consumer cards).

## Roadmap

Site:

- A custom domain for the GitHub Pages site.

Calibration data:

- Long-context results (`llama-bench -d`) for ROCm, Vulkan, SYCL and CPU, and for more
  devices (only DGX Spark and M2 Ultra so far).
- More larger dense models and multi-GPU results (so far a 70B model only on an MI300X
  virtual function, and Llama 3 8B on two and four RTX 4090s).
- Mixture-of-experts results beyond DGX Spark and M2 Ultra, and MoE prompt processing on
  Metal (the M2 Ultra's GPU core count is unknown).
- CPU prompt processing on more CPUs (only native-build data for a few CPUs).
- Apple M1–M4 on current llama.cpp builds (only the M1 Ultra 64-core so far), and the M5
  Ultra prompt-processing gap (about 20 % below the other M5 chips per GPU core).
- Concurrent-request (`llama-batched-bench`) results beyond DGX Spark, and the GPU core count of
  the M2 Ultra in llama.cpp's `benches/` (its 108 batched results are recorded but unused;
  its `llama-bench` results calibrate only what does not depend on the core count).
- Prompt processing for unmeasured data-center GPUs (H200, B200, MI325X, MI355X).
- A check of the OpenBenchmarking.org CPU results against their pages (every record taken
  from GitHub pages has been compared with the page text by script; the OpenBenchmarking
  pages could not be fetched).

Features:

- Web playback of several concurrent streams side by side (playback now shows one of them).

## Development

```bash
PYTHONPATH=src python -m unittest discover -s tests -v
PYTHONPATH=src python -m llm_sim_bench export-web   # regenerate web data
node --test web/test/*.test.mjs                     # JavaScript estimator vs. Python
```

## License

| What | License |
| --- | --- |
| Code (everything outside `data/`, `web/data/`, `src/llm_sim_bench/data/`, `web/fonts/` and `web/brand/`) | [MIT](LICENSE) |
| Data in `data/`, `web/data/` and its packaged copy in `src/llm_sim_bench/data/`, and the sample texts written for this project | [CC BY 4.0](LICENSE-DATA) |
| SYCL scoreboard records (`data/measurements/llamacpp-sycl.jsonl`, their copies in `web/data/presets.json`, `src/llm_sim_bench/data/calibration.json` and in reports for Intel GPUs) | Not licensed by this project; the source asks for no commercial use |
| Book excerpts in `web/data/texts/` (and their copies in `src/llm_sim_bench/data/texts/`) | Public domain |
| Fonts in `web/fonts/` (Noto Sans KR, Noto Sans JP, JetBrains Mono) | SIL Open Font License 1.1 |
| The author's logo in `web/brand/` | Not licensed; all rights reserved |

Benchmark results come from public llama.cpp scoreboards, OpenBenchmarking.org and other
sources; see [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md) for each source's terms.
