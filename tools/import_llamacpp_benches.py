"""Turn the benchmark tables in llama.cpp's benches/ folder into records.

ggml-org/llama.cpp keeps benchmark write-ups in benches/<name>/<name>.md (MIT
license, like the rest of the repository). Each model section has a
llama-batched-bench table (PP, TG, B, N_KV, T_PP, S_PP, T_TG, S_TG, T, S), a
llama-bench table at several context depths (tests such as "pp2048 @ d4096")
whose model column names the quantization, and a build line.

    python tools/import_llamacpp_benches.py ~/src/llama.cpp --retrieved YYYY-MM-DD \
        --batched data/measurements/llamacpp-batched.jsonl \
        --depth data/measurements/llamacpp-depth.jsonl

Only values printed in the files are copied; anything else is null. S_PP and S_TG
are throughput over all B sequences, stored as results.pp_tps and results.tg_tps.
Each llama-bench depth becomes one record with that depth's pp and tg results.
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
# llama-bench test column, e.g. "pp2048", "tg32 @ d4096"; value "2250.28 ± 6.41".
BENCH_TEST = re.compile(r"^(pp|tg)(\d+)(?: @ d(\d+))?$")
BENCH_VALUE = re.compile(r"^([\d.]+) ± ([\d.]+)$")


def model_name(repo: str) -> str:
    """ggml-org/Qwen2.5-Coder-7B-Q8_0-GGUF -> qwen2.5-coder-7b"""
    name = repo.split("/", 1)[1].removesuffix("-GGUF")
    return re.sub(r"-(Q\d\w*|F16|BF16|MXFP4)$", "", name).lower()


def sections(root: Path, path: str):
    """(model heading, section text) for every model section of a benches file."""
    text = (root / path).read_text(encoding="utf-8")
    for section in re.split(r"^## ", text, flags=re.M)[1:]:
        heading = section.splitlines()[0].strip()
        if heading.startswith("ggml-org/") and "llama-bench`" in section:
            yield heading, section


def _base(path: str, info: dict, commit: str, retrieved: str, build, flash_attn: bool,
          name: str, quant: str) -> dict:
    return {
        "source": {"url": f"https://github.com/ggml-org/llama.cpp/blob/{commit}/{path}",
                   "title": info["title"], "retrieved": retrieved},
        "hardware": {"gpu": info["gpu"], "gpu_count": 1, "vram_gb": None,
                     "memory_type": None, "bus_width_bits": None, "gpu_cores": None,
                     "bandwidth_gbps": None, "cpu": None, "ram_gb": None,
                     "gpu_preset": info["gpu_preset"]},
        "software": {"runtime": "llama.cpp", "backend": info["backend"],
                     "commit": build.group(1), "flash_attn": flash_attn},
        "model": {"name": name, "quant": quant, "file": None},
    }


def records(root: Path, path: str, info: dict, commit: str, retrieved: str) -> list[dict]:
    """llama-batched-bench rows."""
    out = []
    for heading, section in sections(root, path):
        if "llama-batched-bench" not in section:
            continue
        main = dict(kv.split(" = ") for kv in MAIN.search(section).group(1).split(", "))
        build = BUILD.search(section)
        quant = BENCH_MODEL.search(section).group(1).split()[2]
        name = model_name(heading)
        batched = section[section.index("llama-batched-bench"):section.index("llama-bench`")]
        for pp, tg, b, _n_kv, _t_pp, s_pp, _t_tg, s_tg, _t, _s in ROW.findall(batched):
            out.append({
                "id": f"llamacpp-benches-{Path(path).stem}-{name}-pp{pp}-tg{tg}-b{b}",
                **_base(path, info, commit, retrieved, build, main["flash_attn"] == "1", name,
                        quant),
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


def depth_records(root: Path, path: str, info: dict, commit: str,
                  retrieved: str) -> list[dict]:
    """llama-bench rows, one record per depth with its pp and tg results."""
    out = []
    for heading, section in sections(root, path):
        table = [line for line in section[section.index("llama-bench`"):].splitlines()
                 if line.startswith("|")]
        cols = [c.strip() for c in table[0].strip("|").split("|")]
        rows = [dict(zip(cols, (c.strip() for c in line.strip("|").split("|"))))
                for line in table[2:]]
        build = BUILD.search(section)
        name = model_name(heading)
        by_depth: dict[int, dict] = {}
        for row in rows:
            kind, n, depth = BENCH_TEST.match(row["test"]).groups()
            value, std = BENCH_VALUE.match(row["t/s"]).groups()
            by_depth.setdefault(int(depth or 0), {"row": row})[kind] = (int(n), float(value),
                                                                         float(std))
        for depth, got in sorted(by_depth.items()):
            row, pp, tg = got["row"], got.get("pp"), got.get("tg")
            ngl, ubatch = row.get("ngl"), row.get("n_ubatch")
            out.append({
                "id": f"llamacpp-benches-{Path(path).stem}-{name}-llama-bench-d{depth}",
                **_base(path, info, commit, retrieved, build, row["fa"] == "1", name,
                        row["model"].split()[2]),
                "run": {"n_gpu_layers": int(ngl) if ngl else None,
                        "n_prompt": pp[0] if pp else None, "n_gen": tg[0] if tg else None,
                        "depth": depth, "tool": "llama-bench",
                        "n_ubatch": int(ubatch) if ubatch else None},
                "results": {"pp_tps": pp[1] if pp else None, "pp_std": pp[2] if pp else None,
                            "tg_tps": tg[1] if tg else None, "tg_std": tg[2] if tg else None},
                "notes": f"llama-bench, build {build.group(2)}"
                         + ("" if ngl else "; the table has no ngl column"),
                "flags": [],
            })
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("llama_cpp", type=Path, help="llama.cpp checkout")
    p.add_argument("--retrieved", required=True, help="date the files were read (YYYY-MM-DD)")
    p.add_argument("--batched", type=Path, required=True, help="output for batched records")
    p.add_argument("--depth", type=Path, required=True, help="output for llama-bench records")
    args = p.parse_args()
    commit = subprocess.run(["git", "-C", str(args.llama_cpp), "rev-parse", "HEAD"],
                            capture_output=True, text=True, check=True).stdout.strip()
    for out, make in ((args.batched, records), (args.depth, depth_records)):
        with open(out, "w", encoding="utf-8") as fh:
            for path, info in FILES.items():
                for rec in make(args.llama_cpp, path, info, commit, args.retrieved):
                    fh.write(json.dumps(rec, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
