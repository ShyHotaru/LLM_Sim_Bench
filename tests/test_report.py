import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path

from llm_sim_bench import report
from llm_sim_bench.cli import main
from llm_sim_bench.estimator import estimate
from llm_sim_bench.profiles import CPU_PRESETS, GPU_PRESETS, MODEL_PRESETS, RAM, System

DATA = report.load_data()


def _run(argv):
    out = io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(io.StringIO()):
        code = main(argv)
    return code, out.getvalue()


class ReportTests(unittest.TestCase):
    def test_gpu_report_has_check_rows(self):
        system = System(CPU_PRESETS["ryzen7-5800x"], RAM(64, 51.2), (GPU_PRESETS["rtx4090-24gb"],))
        est = estimate(system, MODEL_PRESETS["llama-3.1-8b"], "Q4_K_M", 8192)
        res = report.result(system, est, 16, DATA,
                            {"prompt_tokens": 1000, "max_tokens": 300, "depth": 0})
        self.assertEqual(res["check"]["kind"], "gpu")
        self.assertTrue(res["check"]["rows"])
        self.assertAlmostEqual(res["timing"]["ttft_s"], est.prefill_time(1000))
        self.assertGreater(res["timing"]["reply_s"], res["timing"]["ttft_s"])
        self.assertEqual(res["notes"], [])

    def test_moe_and_offload_are_reported(self):
        system = System(CPU_PRESETS["ryzen7-5800x"], RAM(32, 51.2),
                        (GPU_PRESETS["rtx4060ti-16gb"],))
        est = estimate(system, MODEL_PRESETS["qwen3-30b-a3b"], "Q4_K_M", 16384)
        res = report.result(system, est, 16, DATA)
        codes = [w["code"] for w in res["warnings"]]
        # MoE and long-context factors are measured for CUDA, not for the offloaded layers.
        moe = [w for w in res["warnings"] if w["code"] == "moe_uncalibrated"]
        self.assertEqual([w["params"]["backend"] for w in moe], ["CPU"])
        self.assertIn("depth_uncalibrated", codes)
        self.assertIn("layers_on_cpu", [n["code"] for n in res["notes"]])
        self.assertIsNone(res["timing"])

    def test_error_bands_widen_for_unmeasured_devices(self):
        measured = System(CPU_PRESETS["ryzen7-5800x"], RAM(64, 51.2), (GPU_PRESETS["rtx4090-24gb"],))
        other = System(CPU_PRESETS["ryzen7-5800x"], RAM(64, 51.2),
                       (GPU_PRESETS["rtx3070-8gb"],))
        m = report.speed_bands(measured, estimate(measured, MODEL_PRESETS["llama-3.1-8b"]), DATA)
        o = report.speed_bands(other, estimate(other, MODEL_PRESETS["llama-3.1-8b"]), DATA)
        self.assertEqual(m["prefill"], DATA["bands"]["cuda"]["pp_measured"])
        self.assertEqual(o["prefill"], DATA["bands"]["cuda"]["pp_other"])
        self.assertLess(m["prefill"], o["prefill"])
        # Offloaded layers widen the band to the CPU's.
        small = System(CPU_PRESETS["ryzen7-5800x"], RAM(64, 51.2), (GPU_PRESETS["rtx3080-10gb"],))
        off = report.speed_bands(small, estimate(small, MODEL_PRESETS["llama-3.1-70b"]), DATA)
        self.assertGreaterEqual(off["prefill"], DATA["bands"]["cpu"]["pp_other"])

    def test_markdown_lists_every_configuration(self):
        system = System(CPU_PRESETS["ryzen9-7950x"], RAM(64, 96))
        est = estimate(system, MODEL_PRESETS["llama-3.1-8b"], "Q4_0", 4096)
        res = report.result(system, est, 16, DATA)
        md = report.markdown(report.report([{**res, "label": "A"}, {**res, "label": "B"}],
                                           "web", "https://example.invalid/#x"))
        self.assertIn("## Configuration A", md)
        self.assertIn("## Configuration B", md)
        self.assertIn("Setup link: <https://example.invalid/#x>", md)
        self.assertIn("CPU rows are simulated", md)

    def test_formatting_rounds_like_javascript_report(self):
        self.assertEqual(report.fmt(2.5, 0), "2")
        self.assertEqual(report.fmt(70.35, 1), "70.3")
        self.assertEqual(report.fmt(999.96, 1), "1,000.0")
        self.assertEqual(report.fmt_tps(1234.5), "1,234")
        self.assertEqual(report.fmt_tps(None), "–")
        self.assertEqual(report.fmt_time(0.0904), "90 ms")
        self.assertEqual(report.fmt_time(59.96), "1 min 00 s")
        self.assertEqual(report.fmt_time(0.9996), "1.0 s")
        self.assertEqual(report.fmt_time(3725), "1 h 02 min")


class EstimateCliTests(unittest.TestCase):
    ARGS = ["estimate", "--gpu", "rtx3090-24gb", "--model", "qwen2.5-7b"]

    def test_json_report(self):
        code, out = _run(self.ARGS + ["--format", "json", "--prompt-tokens", "500"])
        self.assertEqual(code, 0)
        rep = json.loads(out)
        self.assertEqual(rep["schema"], report.SCHEMA)
        self.assertEqual(rep["generated_by"]["surface"], "cli")
        self.assertEqual(rep["results"][0]["timing"]["prompt_tokens"], 500)
        self.assertEqual(rep["results"][0]["timing"]["max_tokens"], 200)

    def test_markdown_report_to_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "r.md"
            code, out = _run(self.ARGS + ["--format", "markdown", "-o", str(path)])
            self.assertEqual(code, 0)
            self.assertEqual(out, "")
            self.assertTrue(path.read_text(encoding="utf-8").startswith("# LLM Sim Bench report"))

    def test_text_timing_and_no_fit_exit_code(self):
        code, out = _run(["estimate", "--gpu", "rtx3060-12gb", "--ram-gb", "16",
                          "--model", "llama-3.1-70b", "--max-tokens", "50"])
        self.assertEqual(code, 2)
        self.assertIn("First tok", out)
        self.assertIn("(50 tokens)", out)


if __name__ == "__main__":
    unittest.main()
