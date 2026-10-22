"""Turn the llama-batched-bench tables in llama.cpp's benches/ folder into records.

ggml-org/llama.cpp keeps benchmark write-ups in benches/<name>/<name>.md (MIT
license, like the rest of the repository). Each model section has a
llama-batched-bench table (PP, TG, B, N_KV, T_PP, S_PP, T_TG, S_TG, T, S), a
llama-bench table whose model column names the quantization, and a build line.

    python tools/import_llamacpp_benches.py ~/src/llama.cpp > data/measurements/llamacpp-batched.jsonl

Only values printed in the files are copied; anything else is null. S_PP and S_TG
are throughput over all B sequences, stored as results.pp_tps and results.tg_tps.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
from pathlib import Path

# benches file -> hardware fields. M2 Ultra comes with a 60- or 76-core GPU and the
# file does not say which, so it has no preset and is not used for calibration.
FILES = {
    "benches/dgx-spark/dgx-spark.md": {
        "title": "llama.cpp benches: DGX Spark", "gpu": "NVIDIA GB10",
        "gpu_preset": "nvidia-dgx-spark", "backend": "CUDA",
    },
    "benches/nemotron/nemotron-dgx-spark.md": {
        "title": "llama.cpp benches: Nemotron on DGX Spark", "gpu": "NVIDIA GB10",
        "gpu_preset": "nvidia-dgx-spark", "backend": "CUDA",
    },
    "benches/mac-m2-ultra/mac-m2-ultra.md": {
        "title": "llama.cpp benches: Mac M2 Ultra", "gpu": "M2 Ultra",
        "gpu_preset": None, "backend": "Metal",
    },
}

ROW = re.compile(r"^\|\s*(\d+)\s*\|\s*(\d+)\s*\|\s*(\d+)\s*\|\s*(\d+)\s*\|" + r"\s*([\d.]+)\s*\|" * 6,
                 re.M)
MAIN = re.compile(r"^main: (.*)$", re.M)
BUILD = re.compile(r"^build: ([0-9a-f]+) \((\d+)\)", re.M)
# llama-bench model column, e.g. "qwen2 7B Q8_0" or "gpt-oss 20B MXFP4 MoE".
BENCH_MODEL = re.compile(r"^\| (\S.*?)\s+\|\s+[\d.]+ GiB \|", re.M)


def model_name(repo: str) -> str:
    """ggml-org/Qwen2.5-Coder-7B-Q8_0-GGUF -> qwen2.5-coder-7b"""
    name = repo.split("/", 1)[1].removesuffix("-GGUF")
    return re.sub(r"-(Q\d\w*|F16|BF16|MXFP4)$", "", name).lower()


def records(root: Path, path: str, info: dict, commit: str, retrieved: str) -> list[dict]:
    text = (root / path).read_text(encoding="utf-8")
    out = []
    for section in re.split(r"^## ", text, flags=re.M)[1:]:
        heading = section.splitlines()[0].strip()
        if not heading.startswith("ggml-org/") or "llama-batched-bench" not in section:
            continue
        main = dict(kv.split(" = ") for kv in MAIN.search(section).group(1).split(", "))
        build = BUILD.search(section)
        quant = BENCH_MODEL.search(section).group(1).split()[2]
        name = model_name(heading)
        batched = section[section.index("llama-batched-bench"):section.index("llama-bench`")]
        for pp, tg, b, _n_kv, _t_pp, s_pp, _t_tg, s_tg, _t, _s in ROW.findall(batched):
            out.append({
                "id": f"llamacpp-benches-{Path(path).stem}-{name}-pp{pp}-tg{tg}-b{b}",
                "source": {"url": f"https://github.com/ggml-org/llama.cpp/blob/{commit}/{path}",
                           "title": info["title"], "retrieved": retrieved},
                "hardware": {"gpu": info["gpu"], "gpu_count": 1, "vram_gb": None,
                             "memory_type": None, "bus_width_bits": None, "gpu_cores": None,
                             "bandwidth_gbps": None, "cpu": None, "ram_gb": None,
                             "gpu_preset": info["gpu_preset"]},
                "software": {"runtime": "llama.cpp", "backend": info["backend"],
                             "commit": build.group(1), "flash_attn": main["flash_attn"] == "1"},
                "model": {"name": name, "quant": quant, "file": None},
                "run": {"n_gpu_layers": int(main["n_gpu_layers"]), "n_prompt": int(pp),
                        "n_gen": int(tg), "depth": 0, "tool": "llama-batched-bench",
                        "n_parallel": int(b), "pp_shared": main["is_pp_shared"] == "1",
                        "n_batch": int(main["n_batch"]), "n_ubatch": int(main["n_ubatch"])},
                "results": {"pp_tps": float(s_pp), "pp_std": None, "tg_tps": float(s_tg),
                            "tg_std": None},
                "notes": f"llama-batched-bench S_PP and S_TG (all {b} sequences together); "
                         f"build {build.group(2)}",
                "flags": ["batched"],
            })
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("llama_cpp", type=Path, help="llama.cpp checkout")
    p.add_argument("--retrieved", required=True, help="date the files were read (YYYY-MM-DD)")
    args = p.parse_args()
    commit = subprocess.run(["git", "-C", str(args.llama_cpp), "rev-parse", "HEAD"],
                            capture_output=True, text=True, check=True).stdout.strip()
    for path, info in FILES.items():
        for rec in records(args.llama_cpp, path, info, commit, args.retrieved):
            print(json.dumps(rec, ensure_ascii=False))


if __name__ == "__main__":
    main()
