import time
import unittest

from llm_sim_bench.cli import main
from llm_sim_bench.estimator import estimate, kv_bytes_per_token
from llm_sim_bench.mock import MockEngine, StreamStats
from llm_sim_bench.profiles import (CPU_PRESETS, GPU_PRESETS, MODEL_PRESETS, RAM, System,
                                    ram_bandwidth_gbps)

CPU = CPU_PRESETS["ryzen7-5800x"]
RAM64 = RAM(64, ram_bandwidth_gbps("ddr4-3200", 2))
LLAMA8 = MODEL_PRESETS["llama-3.1-8b"]


class EstimatorTests(unittest.TestCase):
    def test_ram_bandwidth(self):
        self.assertAlmostEqual(ram_bandwidth_gbps("ddr4-3200", 2), 51.2)
        self.assertAlmostEqual(ram_bandwidth_gbps("DDR5-6000", 2), 96.0)
        with self.assertRaises(ValueError):
            ram_bandwidth_gbps("fast-ram")

    def test_weights_and_kv_size(self):
        est = estimate(System(CPU, RAM64), LLAMA8, "Q4_K_M", 8192)
        self.assertAlmostEqual(est.weights_gb, 8.03 * 4.89 / 8, places=2)
        # 2 (K,V) * 32 layers * 8 heads * 128 dim * 2 bytes = 128 KiB per token
        self.assertEqual(kv_bytes_per_token(LLAMA8), 131072)
        self.assertAlmostEqual(est.kv_gb, 131072 * 8192 / 1e9, places=3)

    def test_small_model_fits_fully_on_gpu(self):
        est = estimate(System(CPU, RAM64, (GPU_PRESETS["rtx3080-10gb"],)), LLAMA8)
        self.assertEqual([p.is_gpu for p in est.placements], [True])
        self.assertTrue(est.fits)
        # Plausible band for an 8B Q4 model on a 3080.
        self.assertGreater(est.decode_tps_empty, 80)
        self.assertLess(est.decode_tps_empty, 160)

    def test_large_model_offloads_and_slows(self):
        sys_ = System(CPU, RAM64, (GPU_PRESETS["rtx3080-10gb"],))
        big = estimate(sys_, MODEL_PRESETS["qwen2.5-32b"])
        self.assertEqual([p.is_gpu for p in big.placements], [True, False])
        self.assertLess(big.decode_tps_empty, 10)

    def test_does_not_fit_in_ram(self):
        est = estimate(System(CPU, RAM(16, 51.2)), MODEL_PRESETS["llama-3.1-70b"])
        self.assertFalse(est.fits)

    def test_decode_slows_with_context(self):
        est = estimate(System(CPU, RAM64, (GPU_PRESETS["rtx4090-24gb"],)), LLAMA8,
                       context=32768)
        self.assertLess(est.decode_tps_full, est.decode_tps_empty)

    def test_moe_decodes_faster_than_dense_of_same_size(self):
        sys_ = System(CPU, RAM64)
        moe = estimate(sys_, MODEL_PRESETS["qwen3-30b-a3b"])
        dense = estimate(sys_, MODEL_PRESETS["qwen2.5-32b"])
        self.assertGreater(moe.decode_tps_empty, 5 * dense.decode_tps_empty)


class ParallelTests(unittest.TestCase):
    SYSTEM = System(CPU, RAM64, (GPU_PRESETS["rtx3090-24gb"],))

    def test_one_request_matches_single_stream(self):
        est = estimate(self.SYSTEM, LLAMA8, "Q4_K_M", 8192)
        for fill in (0, 1000, 8192):
            self.assertAlmostEqual(est.step_time([fill]), 1 / est.decode_tps_at(fill), places=12)
            self.assertAlmostEqual(est.parallel_tps(1, fill), est.decode_tps_at(fill), places=9)

    def test_slots_get_their_own_kv_cache(self):
        one = estimate(self.SYSTEM, LLAMA8, "Q4_K_M", 8192)
        four = estimate(self.SYSTEM, LLAMA8, "Q4_K_M", 8192, parallel=4)
        self.assertAlmostEqual(four.kv_gb, 4 * one.kv_gb)
        self.assertEqual(four.context, 8192)
        self.assertAlmostEqual(four.decode_tps_full, one.decode_tps_full)
        with self.assertRaises(ValueError):
            estimate(self.SYSTEM, LLAMA8, parallel=0)

    def test_batching_trades_speed_per_request_for_throughput(self):
        est = estimate(self.SYSTEM, LLAMA8, "Q4_K_M", 4096, parallel=16)
        prev_each, prev_total = float("inf"), 0.0
        for n in (1, 2, 4, 8, 16):
            each = est.parallel_tps(n, 1000)
            self.assertLess(each, prev_each)
            self.assertGreater(n * each, prev_total)
            prev_each, prev_total = each, n * each

    def test_moe_reads_more_experts_with_more_requests(self):
        from llm_sim_bench.estimator import weight_read_ratio

        moe = MODEL_PRESETS["qwen3-30b-a3b"]
        ratios = [weight_read_ratio(moe, n) for n in (1, 2, 8, 64, 10_000)]
        self.assertAlmostEqual(ratios[0], moe.active_b / moe.params_b)
        self.assertEqual(ratios, sorted(ratios))
        self.assertAlmostEqual(ratios[-1], 1.0, places=6)
        self.assertEqual(weight_read_ratio(LLAMA8, 32), 1.0)

    def test_parallel_levels(self):
        from llm_sim_bench.estimator import parallel_levels

        self.assertEqual(parallel_levels(1), [1])
        self.assertEqual(parallel_levels(8), [1, 2, 4, 8])
        self.assertEqual(parallel_levels(6), [1, 2, 4, 6])

    def test_cli_parallel_table(self):
        import contextlib
        import io

        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            code = main(["estimate", "--gpu", "rtx3090-24gb", "--model", "llama-3.1-8b",
                         "--parallel", "4"])
        self.assertEqual(code, 0)
        self.assertIn("x 4 slots (llama-server -np 4 -c 16384)", out.getvalue())
        self.assertIn("Parallel", out.getvalue())


class MockTests(unittest.TestCase):
    def test_pacing_matches_target(self):
        engine = MockEngine(prefill_tps=1000, decode_tps=lambda _: 200, jitter=0, seed=1)
        stats = StreamStats()
        start = time.perf_counter()
        tokens = list(engine.stream(prompt_tokens=50, pieces=["x"] * 41, stats=stats))
        elapsed = time.perf_counter() - start
        self.assertEqual(len(tokens), 41)
        # 0.05 s prefill + 40 * 5 ms decode = 0.25 s
        self.assertAlmostEqual(elapsed, 0.25, delta=0.04)
        self.assertAlmostEqual(stats.decode_tps, 200, delta=20)


class CliTests(unittest.TestCase):
    def test_estimate_with_example_config(self):
        self.assertEqual(main(["estimate", "--config", "examples/rtx3080_desktop.toml"]), 0)

    def test_unknown_preset_is_an_error(self):
        self.assertEqual(main(["estimate", "--gpu", "voodoo2"]), 1)


if __name__ == "__main__":
    unittest.main()
