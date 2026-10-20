import unittest

from llm_sim_bench import calibrate, measurements
from llm_sim_bench.cli import main
from llm_sim_bench.estimator import Efficiency, estimate
from llm_sim_bench.profiles import APPLE_PRESETS, MODEL_PRESETS, apple_system

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
        flagged = [r for r in RECORDS if r["flags"]]
        self.assertTrue(flagged)
        self.assertFalse(any(calibrate.usable(r) for r in flagged))


class CalibrationTests(unittest.TestCase):
    """Guard the estimator against drifting away from real measurements."""

    def test_defaults_match_measurements(self):
        stats = calibrate.summary(calibrate.evaluate(RECORDS, Efficiency()))
        for backend in ("cuda", "metal"):
            self.assertGreaterEqual(stats[f"{backend}/tg"]["n"], 20)
            self.assertLess(stats[f"{backend}/tg"]["median"], 0.06, backend)
            self.assertLess(stats[f"{backend}/tg"]["max"], 0.30, backend)
            self.assertLess(stats[f"{backend}/pp"]["median"], 0.10, backend)

    def test_defaults_are_close_to_a_fresh_fit(self):
        fitted, default = calibrate.fit(RECORDS), Efficiency()
        for k in ("gpu_bandwidth", "gpu_compute", "metal_bandwidth", "metal_compute"):
            self.assertAlmostEqual(getattr(fitted, k), getattr(default, k), delta=0.02, msg=k)

    def test_cli_calibrate(self):
        self.assertEqual(main(["calibrate"]), 0)


class UnifiedMemoryTests(unittest.TestCase):
    def test_gpu_budget_is_a_share_of_unified_memory(self):
        chip = APPLE_PRESETS["apple-m4-max-40c"]
        llama70 = MODEL_PRESETS["llama-3.1-70b"]
        # ~43 GB of weights: 48 GB x 0.75 is too small for the GPU, 64 GB is enough.
        small = estimate(apple_system(chip, 48), llama70)
        big = estimate(apple_system(chip, 64), llama70)
        self.assertEqual([p.is_gpu for p in small.placements], [True, False])
        self.assertEqual([p.is_gpu for p in big.placements], [True])
        self.assertTrue(big.fits)

    def test_cli_apple(self):
        self.assertEqual(main(["estimate", "--apple", "apple-m2-ultra-76c", "--ram-gb", "192",
                               "--model", "llama-3.1-70b"]), 0)


if __name__ == "__main__":
    unittest.main()
