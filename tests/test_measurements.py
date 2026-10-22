import unittest

from llm_sim_bench import calibrate, measurements
from llm_sim_bench.cli import main
from llm_sim_bench.estimator import Efficiency, estimate
from llm_sim_bench.profiles import UNIFIED_PRESETS, MODEL_PRESETS, unified_system

RECORDS = measurements.load()


class MeasurementDataTests(unittest.TestCase):
    def test_records_are_valid(self):
        self.assertGreater(len(RECORDS), 100)
        self.assertEqual(measurements.check_all(RECORDS), [])

    def test_validate_reports_problems(self):
        rec = dict(RECORDS[0], results={"pp_tps": -1, "pp_std": None, "tg_tps": None,
                                        "tg_std": None})
        self.assertIn("results.pp_tps must be a positive number or null",
                      measurements.validate(rec))
        self.assertIn("missing section 'run'",
                      measurements.validate({k: v for k, v in RECORDS[0].items() if k != "run"}))

    def test_flagged_records_are_not_used(self):
        flagged = [r for r in RECORDS if calibrate._EXCLUDE & set(r["flags"])]
        self.assertTrue(flagged)
        self.assertFalse(any(calibrate.usable(r) for r in flagged))
        # OpenBLAS builds keep their decode result but not prompt processing.
        blas = next(r for r in RECORDS if "openblas-build" in r["flags"] and calibrate.usable(r)
                    and r["results"]["pp_tps"])
        self.assertIsNone(calibrate.predict(blas, Efficiency()).pp_meas)


class CalibrationTests(unittest.TestCase):
    """Guard the estimator against drifting away from real measurements."""

    # Median and worst-case relative error allowed per backend and metric.
    LIMITS = {"tg": {"median": 0.08, "max": 0.60}, "pp": {"median": 0.10, "max": 1.5}}
    PP_MEDIAN = {"sycl": 0.35}  # Intel results mix FP16 and FP32 builds

    def test_defaults_match_measurements(self):
        stats = calibrate.summary(calibrate.evaluate(RECORDS, Efficiency()))
        for backend in ("cuda", "metal", "rocm", "vulkan", "sycl", "cpu"):
            for metric, limit in self.LIMITS.items():
                s = stats[f"{backend}/{metric}"]
                median = self.PP_MEDIAN.get(backend, limit["median"]) if metric == "pp" \
                    else limit["median"]
                self.assertGreaterEqual(s["n"], 10, backend)
                self.assertLess(s["median"], median, f"{backend}/{metric}")
                self.assertLess(s["max"], limit["max"], f"{backend}/{metric}")

    def test_defaults_are_close_to_a_fresh_fit(self):
        fitted, default = calibrate.fit(RECORDS), Efficiency()
        for k, v in vars(default).items():
            if k.endswith(("_bandwidth", "_compute")) and not k.startswith("device"):
                self.assertAlmostEqual(getattr(fitted, k), v, delta=0.02, msg=k)
        for k in ("device_compute", "device_bandwidth"):
            self.assertEqual(getattr(fitted, k), getattr(default, k),
                             "run `llmsimbench calibrate --write` to store the device calibration")

    def test_cli_calibrate(self):
        self.assertEqual(main(["calibrate"]), 0)


class ExportTests(unittest.TestCase):
    def test_web_files_are_current(self):
        from llm_sim_bench import export

        self.assertEqual(export.stale(RECORDS), [],
                         "run `llmsimbench export-web` to regenerate the web data")


    def test_cpu_measurements_follow_calibration_filters(self):
        from llm_sim_bench import calibrate, export

        rows = export.cpu_measurements(RECORDS)
        by_id = {r["id"]: r for r in RECORDS}
        self.assertGreater(len(rows), 100)
        for row in rows:
            rec = by_id[row["id"]]
            self.assertTrue(calibrate.usable(rec), row["id"])
            point = calibrate.predict(rec, calibrate.Efficiency())
            # Exactly the results calibration compares with a prediction are shown.
            self.assertEqual(row["pp_tps"], point.pp_meas if point.pp_pred else None, row["id"])
            self.assertEqual(row["tg_tps"], point.tg_meas if point.tg_pred else None, row["id"])


class WebServerTests(unittest.TestCase):
    def test_serves_app_from_any_working_directory(self):
        import os
        import tempfile
        import threading
        import urllib.request

        from llm_sim_bench.cli import make_web_server

        cwd = os.getcwd()
        with tempfile.TemporaryDirectory() as tmp:
            os.chdir(tmp)
            try:
                httpd = make_web_server("127.0.0.1", 0)
            finally:
                os.chdir(cwd)
            threading.Thread(target=httpd.serve_forever, daemon=True).start()
            try:
                base = f"http://127.0.0.1:{httpd.server_address[1]}/"
                with urllib.request.urlopen(base) as r:
                    self.assertIn(b"LLM Sim Bench", r.read())
                with urllib.request.urlopen(base + "js/app.js") as r:
                    self.assertEqual(r.headers.get_content_type(), "text/javascript")
                with urllib.request.urlopen(base + "data/presets.json") as r:
                    self.assertEqual(r.status, 200)
            finally:
                httpd.shutdown()
                httpd.server_close()


class PresetTests(unittest.TestCase):
    def test_cpu_presets_have_memory_defaults(self):
        from llm_sim_bench.profiles import CPU_PRESETS, ram_bandwidth_gbps

        self.assertGreaterEqual(len(CPU_PRESETS), 30)
        for cpu in CPU_PRESETS.values():
            self.assertIn(cpu.vendor, ("AMD", "Intel"), cpu.name)
            self.assertIn(cpu.segment, ("desktop", "workstation", "server"), cpu.name)
            self.assertGreater(ram_bandwidth_gbps(cpu.mem_kind, cpu.mem_channels), 40, cpu.name)
            self.assertGreater(cpu.fp32_tflops, 0.5, cpu.name)


class UnifiedMemoryTests(unittest.TestCase):
    def test_gpu_budget_is_a_share_of_unified_memory(self):
        chip = UNIFIED_PRESETS["apple-m4-max-40c"]
        llama70 = MODEL_PRESETS["llama-3.1-70b"]
        # ~43 GB of weights: 48 GB x 0.75 is too small for the GPU, 64 GB is enough.
        small = estimate(unified_system(chip, 48), llama70)
        big = estimate(unified_system(chip, 64), llama70)
        self.assertEqual([p.is_gpu for p in small.placements], [True, False])
        self.assertEqual([p.is_gpu for p in big.placements], [True])
        self.assertTrue(big.fits)

    def test_cli_apple(self):
        self.assertEqual(main(["estimate", "--apple", "apple-m2-ultra-76c", "--ram-gb", "192",
                               "--model", "llama-3.1-70b"]), 0)


if __name__ == "__main__":
    unittest.main()
