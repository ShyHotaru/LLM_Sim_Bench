"""Command-line interface: `llmsim list | estimate | run | serve | calibrate | export-web | web`."""

from __future__ import annotations

import argparse
import sys
import tomllib
from dataclasses import replace

from .estimator import Estimate, estimate, format_estimate
from .mock import MockEngine, StreamStats, count_tokens
from .profiles import (APPLE_PRESETS, CPU, CPU_PRESETS, GPU, GPU_PRESETS, MODEL_PRESETS,
                       QUANT_BPW, RAM, Model, System, apple_system, lookup, ram_bandwidth_gbps)

DEFAULTS = {
    "cpu": "ryzen7-5800x",
    "ram": {"size_gb": 32, "kind": "ddr4-3200", "channels": 2},
    "gpu": [],
    "model": {"preset": "llama-3.1-8b", "quant": "Q4_K_M", "context": 4096},
}


def _profile(presets, cls, spec, what):
    """Build a profile from a preset name or a table, optionally extending a preset."""
    if isinstance(spec, str):
        return lookup(presets, spec, what)
    spec = dict(spec)
    base = spec.pop("preset", None)
    return replace(lookup(presets, base, what), **spec) if base else cls(**spec)


def _ram(spec: dict) -> RAM:
    kind = spec.get("kind", "ddr4-3200")
    bw = spec.get("bandwidth_gbps") or ram_bandwidth_gbps(kind, spec.get("channels", 2))
    label = f"{kind} x{spec.get('channels', 2)}" if "bandwidth_gbps" not in spec else "custom"
    return RAM(float(spec.get("size_gb", 32)), float(bw), label)


def build(args) -> tuple[System, Model, dict]:
    cfg = dict(DEFAULTS)
    if args.config:
        with open(args.config, "rb") as f:
            cfg.update(tomllib.load(f))

    ram_spec = dict(cfg["ram"])
    for key, val in (("size_gb", args.ram_gb), ("kind", args.ram_kind),
                     ("channels", args.ram_channels)):
        if val is not None:
            ram_spec[key] = val
            if key != "size_gb":
                ram_spec.pop("bandwidth_gbps", None)

    apple = args.apple or cfg.get("apple")
    if apple:
        chip = lookup(APPLE_PRESETS, apple, "Apple chip")
        system = apple_system(chip, float(args.ram_gb or cfg.get("ram", {}).get(
            "size_gb", max(chip.memory_options_gb))))
    else:
        gpu_specs = [] if args.no_gpu else (args.gpu if args.gpu else cfg["gpu"])
        system = System(
            cpu=_profile(CPU_PRESETS, CPU, args.cpu or cfg["cpu"], "CPU"),
            ram=_ram(ram_spec),
            gpus=tuple(_profile(GPU_PRESETS, GPU, g, "GPU") for g in gpu_specs),
        )

    model_spec = dict(cfg["model"])
    run = {
        "quant": args.quant or model_spec.pop("quant", "Q4_K_M"),
        "context": args.ctx or model_spec.pop("context", 4096),
        "kv_bits": args.kv_bits or model_spec.pop("kv_bits", 16),
        "decode_tps": args.decode_tps or cfg.get("sim", {}).get("decode_tps"),
        "prefill_tps": args.prefill_tps or cfg.get("sim", {}).get("prefill_tps"),
    }
    for k in ("quant", "context", "kv_bits"):
        model_spec.pop(k, None)
    model = _profile(MODEL_PRESETS, Model, args.model or model_spec, "model")
    return system, model, run


def describe_system(system: System) -> str:
    if system.unified:
        return (f"System    {system.gpus[0].name} | {system.ram.size_gb:g} GB unified memory "
                f"({system.ram.bandwidth_gbps:.0f} GB/s)")
    gpus = ", ".join(g.name for g in system.gpus) or "none"
    return (f"System    CPU {system.cpu.name} | RAM {system.ram.size_gb:g} GB "
            f"{system.ram.label} ({system.ram.bandwidth_gbps:.1f} GB/s) | GPU {gpus}")


def make_engine(est: Estimate, run: dict, seed=None) -> MockEngine:
    decode = (lambda _fill, r=run["decode_tps"]: r) if run["decode_tps"] else est.decode_tps_at
    return MockEngine(run["prefill_tps"] or est.prefill_tps, decode, seed=seed)


def _estimate(args):
    system, model, run = build(args)
    est = estimate(system, model, run["quant"], run["context"], run["kv_bits"])
    return system, est, run


def cmd_list(_args):
    print("CPUs:  ", ", ".join(CPU_PRESETS))
    print("GPUs:  ", ", ".join(GPU_PRESETS))
    print("Apple: ", ", ".join(APPLE_PRESETS))
    print("Models:", ", ".join(MODEL_PRESETS))
    print("Quants:", ", ".join(QUANT_BPW))
    print("RAM:    any 'ddrN-SPEED' kind, e.g. ddr4-3200, ddr5-6000")


def cmd_estimate(args):
    system, est, _ = _estimate(args)
    print(describe_system(system))
    print(format_estimate(est))
    return 0 if est.fits else 2


def cmd_run(args):
    system, est, run = _estimate(args)
    print(describe_system(system))
    print(format_estimate(est))
    if not est.fits:
        print("Warning: configuration does not fit in memory; simulating anyway.")
    prompt_tokens = args.prompt_tokens or (count_tokens(args.prompt) if args.prompt else 128)
    print(f"\n--- mock output ({prompt_tokens} prompt tokens) ---")
    stats = StreamStats()
    for tok in make_engine(est, run, args.seed).stream(prompt_tokens, args.max_tokens, stats):
        sys.stdout.write(tok)
        sys.stdout.flush()
    print("\n---")
    print(f"TTFT {stats.ttft_s:.2f}s | {stats.completion_tokens} tokens in "
          f"{stats.total_s:.2f}s | decode {stats.decode_tps:.1f} tok/s")
    return 0


def cmd_serve(args):
    from .server import serve

    system, est, run = _estimate(args)
    print(describe_system(system))
    print(format_estimate(est))
    serve(make_engine(est, run), est.model.name, args.host, args.port, args.max_tokens)
    return 0


def cmd_calibrate(args):
    from . import calibrate, measurements
    from .estimator import Efficiency

    records = measurements.load(args.data)
    problems = measurements.check_all(records)
    if problems:
        print("\n".join(problems), file=sys.stderr)
        return 1
    print("Current defaults")
    print(calibrate.format_report(records, Efficiency()))
    fitted = calibrate.fit(records)
    print("\nFitted factors (copy into Efficiency defaults to adopt)")
    for k in ("gpu_bandwidth", "gpu_compute", "gpu_layer_overhead_us", "metal_bandwidth",
              "metal_compute", "metal_layer_overhead_us"):
        print(f"  {k:<24} {getattr(fitted, k)}")
    print(calibrate.format_report(records, fitted))
    return 0


def cmd_export_web(args):
    from . import export

    if args.check:
        stale = export.stale()
        for path in stale:
            print(f"out of date: {path}", file=sys.stderr)
        return 1 if stale else 0
    for path in export.write():
        print(f"wrote {path}")
    return 0


def make_web_server(host: str, port: int):
    """HTTP server for the static web app in web/, independent of the working directory."""
    import functools
    import http.server

    from .export import WEB_DIR

    if not (WEB_DIR / "index.html").is_file():
        raise OSError(f"web app not found at {WEB_DIR}; run llmsim from a source checkout")

    class Handler(http.server.SimpleHTTPRequestHandler):
        # Windows can map .js to text/plain via the registry, which browsers
        # reject for module scripts.
        extensions_map = {**http.server.SimpleHTTPRequestHandler.extensions_map,
                          ".js": "text/javascript", ".mjs": "text/javascript",
                          ".json": "application/json", ".css": "text/css"}

    handler = functools.partial(Handler, directory=str(WEB_DIR))
    return http.server.ThreadingHTTPServer((host, port), handler)


def cmd_web(args):
    with make_web_server(args.host, args.port) as httpd:
        print(f"Serving the web app at http://{args.host}:{httpd.server_address[1]}/ "
              "(Ctrl+C to stop)")
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            pass
    return 0


def parser() -> argparse.ArgumentParser:
    common = argparse.ArgumentParser(add_help=False)
    g = common.add_argument_group("virtual system")
    g.add_argument("--config", help="TOML file describing the system and model")
    g.add_argument("--cpu", help="CPU preset")
    g.add_argument("--ram-gb", type=float, help="system RAM size in GB")
    g.add_argument("--ram-kind", help="RAM kind, e.g. ddr5-5600")
    g.add_argument("--ram-channels", type=int, help="number of memory channels")
    g.add_argument("--gpu", action="append", help="GPU preset (repeat for multi-GPU)")
    g.add_argument("--no-gpu", action="store_true", help="CPU-only system")
    g.add_argument("--apple", help="Apple Silicon chip preset; --ram-gb sets unified memory")
    m = common.add_argument_group("model")
    m.add_argument("--model", help="model preset")
    m.add_argument("--quant", help="quantization, e.g. Q4_K_M")
    m.add_argument("--ctx", type=int, help="context length to allocate")
    m.add_argument("--kv-bits", type=int, choices=(4, 8, 16), help="KV cache precision")
    s = common.add_argument_group("speed overrides")
    s.add_argument("--decode-tps", type=float, help="force a fixed decode tok/s")
    s.add_argument("--prefill-tps", type=float, help="force a fixed prefill tok/s")

    p = argparse.ArgumentParser(prog="llmsim", description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("list", help="show available presets").set_defaults(func=cmd_list)
    sub.add_parser("estimate", parents=[common], help="estimate memory and tok/s"
                   ).set_defaults(func=cmd_estimate)

    r = sub.add_parser("run", parents=[common], help="stream mock tokens in the terminal")
    r.add_argument("--prompt", help="prompt text (only its length matters)")
    r.add_argument("--prompt-tokens", type=int, help="prompt length in tokens")
    r.add_argument("--max-tokens", type=int, default=200)
    r.add_argument("--seed", type=int)
    r.set_defaults(func=cmd_run)

    c = sub.add_parser("calibrate", help="fit efficiency factors to measurement records")
    c.add_argument("--data", help="measurement .jsonl file or directory "
                   "(default: data/measurements in the source tree)")
    c.set_defaults(func=cmd_calibrate)

    ex = sub.add_parser("export-web", help="write presets and golden tests for the web app")
    ex.add_argument("--check", action="store_true", help="only verify the files are current")
    ex.set_defaults(func=cmd_export_web)

    w = sub.add_parser("web", help="serve the web app locally")
    w.add_argument("--host", default="127.0.0.1")
    w.add_argument("--port", type=int, default=8000)
    w.set_defaults(func=cmd_web)

    sv = sub.add_parser("serve", parents=[common], help="run a mock OpenAI-compatible API")
    sv.add_argument("--host", default="127.0.0.1")
    sv.add_argument("--port", type=int, default=8000)
    sv.add_argument("--max-tokens", type=int, default=256, help="default when request omits it")
    sv.set_defaults(func=cmd_serve)
    return p


def main(argv=None) -> int:
    args = parser().parse_args(argv)
    try:
        return args.func(args) or 0
    except (ValueError, OSError, tomllib.TOMLDecodeError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
