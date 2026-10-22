# Third-party sources

LLM Sim Bench bundles data and text from the sources below. Each keeps its own terms; the
project's licenses ([`LICENSE`](LICENSE) for code, [`LICENSE-DATA`](LICENSE-DATA) for data)
apply only to this project's own work. Record-level sources are listed in
[`data/README.md`](data/README.md). The author's logo in `web/brand/` is not licensed at all
(see [`web/brand/README.md`](web/brand/README.md)).

## Benchmark results

Only numeric results and hardware descriptions were copied, each record with its source URL
and retrieval date. No comment text or contributor names were copied.

| Source | Records | Terms |
| --- | --- | --- |
| llama.cpp community scoreboards on GitHub Discussions: [CUDA #15013](https://github.com/ggml-org/llama.cpp/discussions/15013), [Apple Silicon #4167](https://github.com/ggml-org/llama.cpp/discussions/4167), [ROCm #15021](https://github.com/ggml-org/llama.cpp/discussions/15021), [Vulkan #10879](https://github.com/ggml-org/llama.cpp/discussions/10879); [llama.cpp #11733](https://github.com/ggml-org/llama.cpp/discussions/11733); [ik_llama.cpp #164](https://github.com/ikawrakow/ik_llama.cpp/discussions/164) | `llamacpp-scoreboards.jsonl`, `llamacpp-apple-2026.jsonl`, `llamacpp-rocm.jsonl`, `llamacpp-vulkan.jsonl`, part of `cpu-runs.jsonl` | Results posted by individual contributors. The discussions state no license; GitHub's [Terms of Service](https://docs.github.com/en/site-policy/github-terms/github-terms-of-service) apply to the posts. |
| [llama.cpp Intel SYCL scoreboard #23313](https://github.com/ggml-org/llama.cpp/discussions/23313) | `llamacpp-sycl.jsonl` | The opening post says the data is shared for reference only, is not double-checked, and "can not be used as any commercial purpose". These records are **not** covered by `LICENSE-DATA`; do not use them commercially. The same applies to their copies in `web/data/presets.json` and `src/llm_sim_bench/data/calibration.json`, and to the check table of a report for an Intel GPU. |
| [OpenBenchmarking.org](https://openbenchmarking.org/) `pts/llama-cpp` results, read through Wayback Machine copies | most of `cpu-runs.jsonl` | Results uploaded by Phoronix Test Suite users. The site states no license for result data (see its [legal page](https://openbenchmarking.org/legal)). |
| llama.cpp [`benches/`](https://github.com/ggml-org/llama.cpp/tree/c7fc5094179bf8a9c73c18593db714f8da41a402/benches) (`dgx-spark`, `nemotron`, `mac-m2-ultra`), read at commit `c7fc509` | `llamacpp-batched.jsonl`, `llamacpp-depth.jsonl` | Part of the llama.cpp repository, MIT license, "Copyright (c) 2023-2026 The ggml authors" (full text below). |
| [llama.cpp discussion #14640](https://github.com/ggml-org/llama.cpp/discussions/14640), [llama.cpp PR #19378](https://github.com/ggml-org/llama.cpp/pull/19378) | MI300X rows in `llamacpp-rocm.jsonl`, `llamacpp-multi-gpu.jsonl` | Posted in a discussion and a pull request description; GitHub's Terms of Service apply. |
| [geerlingguy/ai-benchmarks issue #53](https://github.com/geerlingguy/ai-benchmarks/issues/53) | 2 records in `cpu-runs.jsonl` | Posted in an issue; GitHub's Terms of Service apply. |

### llama.cpp license

```
MIT License

Copyright (c) 2023-2026 The ggml authors

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## Book excerpts (public domain)

The works below were first published before 1931 by authors who died in or before 1922, so
they are in the public domain in the United States, Japan and Korea. The excerpts are in
`web/data/texts/`, copied into `src/llm_sim_bench/data/texts/` for the command-line tool.

**青空文庫 (Aozora Bunko)**: 夏目漱石『吾輩は猫である』『坊っちゃん』, 森鴎外『高瀬舟』.
Aozora Bunko allows free reuse of these files and asks that their edition, transcription and
proofreading notes be kept
([収録ファイルの取り扱い規準](https://www.aozora.gr.jp/guide/kijyunn.html)). The notes are
stored in `web/data/texts/credits.json`, shown in the app next to each excerpt and printed by
`llmsimbench texts`. Thanks to
the Aozora Bunko volunteers who typed in and proofread these texts.

**Project Gutenberg**: Lewis Carroll, *Alice's Adventures in Wonderland* (eBook #11); Jane
Austen, *Pride and Prejudice* (eBook #1342), downloaded from the
[GITenberg](https://github.com/GITenberg) mirror. As the
[Project Gutenberg license](https://www.gutenberg.org/policy/license.html) requires for
modified copies, the Project Gutenberg license and all references to Project Gutenberg were
removed from the excerpts. With thanks to [Project Gutenberg](https://www.gutenberg.org/),
whose ebook pages the app and the command-line tool link to.

## Tokenizer ratios

`data/tokenizer_ratios.json` stores characters-per-token counts measured on the texts above.
No tokenizer files are included. The tokenizers used and their licenses:

| Family | Tokenizer | License |
| --- | --- | --- |
| Llama 2 | Llama 2 7B | Llama 2 Community License |
| Llama 3 | Meta Llama 3 8B | Meta Llama 3 Community License |
| Gemma 2 | Gemma 2 9B IT | Gemma Terms of Use |
| Mistral | Mistral 7B v0.1 | Apache-2.0 |
| Qwen 2.5, Qwen 3 | Qwen2.5 7B Instruct, Qwen3 8B | Apache-2.0 |
| gpt-oss | gpt-oss-20b | Apache-2.0 |

The Llama 2, Llama 3 and Gemma 2 tokenizers were read from ungated community copies of the
gated official repositories (recorded in the file). `tools/measure_tokenizers.py` now reads the
official repositories, which require accepting each license on Hugging Face.

## Fonts

The web app ships three fonts in `web/fonts/`, unmodified Google Fonts builds as packaged by
[Fontsource](https://fontsource.org/). Each directory holds the font's `OFL.txt`.

| Font | Copyright | License |
| --- | --- | --- |
| Noto Sans KR, Noto Sans JP | Adobe (with Reserved Font Name "Source") | SIL Open Font License 1.1 |
| JetBrains Mono | The JetBrains Mono Project Authors | SIL Open Font License 1.1 |

## Hardware specifications

Preset specs in `src/llm_sim_bench/profiles.py` are approximate public figures from vendor
pages, datasheets and Wikipedia (see [`data/README.md`](data/README.md#hardware-preset-sources)).
Product names are trademarks of their owners.
