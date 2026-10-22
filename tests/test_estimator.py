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
        from llm_sim_bench.estimator import matmul_params, weight_read_ratio

        moe = MODEL_PRESETS["qwen3-30b-a3b"]
        ratios = [weight_read_ratio(moe, n) for n in (1, 2, 8, 64, 10_000)]
        # One request reads the shared weights but the token embedding, and its own experts.
        self.assertAlmostEqual(ratios[0], matmul_params(moe, prompt=False) / moe.params_b)
        self.assertEqual(ratios, sorted(ratios))
        # Enough requests touch every expert: all weights but the token embedding.
        self.assertAlmostEqual(ratios[-1], 1 - 0.311 / moe.params_b, places=6)
        # Dense models read everything but the token embedding, a lookup.
        self.assertAlmostEqual(weight_read_ratio(LLAMA8, 32), 1 - 0.525 / 8.03)
        tied = MODEL_PRESETS["llama-3.2-3b"]
        self.assertEqual(weight_read_ratio(tied, 1), 1.0)

    def test_mxfp4_keeps_shared_weights_in_q8_0(self):
        # ggml-org/gpt-oss-20b-GGUF: 12.10 GB of tensors, experts MXFP4 and the rest Q8_0.
        est = estimate(System(CPU, RAM64), MODEL_PRESETS["gpt-oss-20b"], "MXFP4", 4096)
        self.assertAlmostEqual(est.weights_gb, 12.1, delta=0.5)
        dense = estimate(System(CPU, RAM64), LLAMA8, "MXFP4", 4096)
        self.assertAlmostEqual(dense.weights_gb, 8.03 * 4.25 / 8, places=2)

    def test_sliding_window_layers_keep_only_their_window(self):
        oss = MODEL_PRESETS["gpt-oss-20b"]
        short = estimate(System(CPU, RAM64), oss, "MXFP4", 512)
        long = estimate(System(CPU, RAM64), oss, "MXFP4", 65536)
        per_token = kv_bytes_per_token(oss) / 1e9
        self.assertAlmostEqual(short.kv_gb, per_token * 512)
        # Half the layers keep 128 + 512 tokens, the other half the whole context.
        self.assertAlmostEqual(long.kv_gb, per_token * (65536 + 640) / 2)

    def test_prompts_slow_down_as_the_context_fills(self):
        est = estimate(System(CPU, RAM64, (GPU_PRESETS["rtx4090-24gb"],)), LLAMA8, "Q4_K_M",
                       65536)
        empty, deep = est.prefill_time(1000), est.prefill_time(1000, 32000)
        self.assertGreater(empty, 1000 / est.prefill_tps)
        self.assertGreater(deep, 1.5 * empty)
        # Attention grows linearly with the cached tokens.
        self.assertAlmostEqual(est.prefill_time(1000, 64000) - deep, deep - empty,
                               delta=1e-3 * deep)
        # Sliding-window layers stop growing past their window; full layers do not.
        oss = estimate(System(CPU, RAM64, (GPU_PRESETS["rtx4090-24gb"],)),
                       MODEL_PRESETS["gpt-oss-20b"], "MXFP4", 65536)
        grow = [oss.prefill_time(100, d + 1000) - oss.prefill_time(100, d)
                for d in (2000, 30000)]
        self.assertAlmostEqual(grow[0], grow[1], delta=1e-6 * grow[0])
        self.assertGreater(grow[0], 0)

    def test_context_sum(self):
        from llm_sim_bench.estimator import context_sum

        for first, last, cap in [(1, 10, 0), (1, 10, 4), (5, 9, 4), (3, 3, 4), (1, 0, 0),
                                 (2, 20, 7)]:
            want = sum(min(c, cap) if cap else c for c in range(first, last + 1))
            self.assertEqual(context_sum(first, last, cap), want, (first, last, cap))

    def test_prompts_skip_the_output_layer(self):
        from llm_sim_bench.estimator import matmul_params

        # Untied: the token embedding is a lookup; prompts also skip the output layer.
        self.assertAlmostEqual(matmul_params(LLAMA8, prompt=False), 8.03 - 0.525)
        self.assertAlmostEqual(matmul_params(LLAMA8, prompt=True), 8.03 - 2 * 0.525)
        # Tied: one matrix, used as the output layer in decode only.
        tied = MODEL_PRESETS["llama-3.2-3b"]
        self.assertAlmostEqual(matmul_params(tied, prompt=False), 3.21)
        self.assertAlmostEqual(matmul_params(tied, prompt=True), 3.21 - 0.394)
        # MoE: shared weights and the routed share of the experts.
        oss = MODEL_PRESETS["gpt-oss-20b"]
        self.assertAlmostEqual(matmul_params(oss, prompt=False),
                               21.0 - 19.11 + 19.11 * 4 / 32 - 0.579)

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

    @staticmethod
    def _run_together(engine, jobs, delays=None):
        """Stream (prompt_tokens, n_pieces) jobs in threads; return their stats."""
        import threading

        stats = [StreamStats() for _ in jobs]

        def run(i):
            time.sleep((delays or [0] * len(jobs))[i])
            list(engine.stream(jobs[i][0], ["x"] * jobs[i][1], stats[i]))

        threads = [threading.Thread(target=run, args=(i,)) for i in range(len(jobs))]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        return stats

    def test_concurrent_streams_share_decode_steps(self):
        # 10 ms per step for one stream, 15 ms when two share it.
        engine = MockEngine(1e6, lambda _: 100, jitter=0, seed=1, slots=2,
                            step_time=lambda fills: 0.005 + 0.005 * len(fills))
        stats = self._run_together(engine, [(1, 31), (1, 31)])
        for s in stats:
            self.assertAlmostEqual(s.decode_tps, 1 / 0.015, delta=8)
            self.assertLess(s.queued_s, 0.01)

    def test_requests_beyond_the_slots_wait(self):
        engine = MockEngine(1e6, lambda _: 200, jitter=0, seed=1, slots=1)
        stats = self._run_together(engine, [(1, 21), (1, 21)], delays=[0, 0.02])
        self.assertLess(stats[0].queued_s, 0.01)
        # The second request waits for the first to finish (about 0.1 s).
        self.assertAlmostEqual(stats[1].queued_s, 0.08, delta=0.03)
        self.assertGreaterEqual(stats[1].ttft_s, stats[1].queued_s)

    def test_prompt_time_depends_on_cached_tokens(self):
        calls = []

        def prefill(tokens, cached):
            calls.append((tokens, cached))
            return 0.0

        engine = MockEngine(1000, lambda _: 1e5, jitter=0, prefill_time=prefill)
        list(engine.stream(10, ["x"], cached_tokens=500))
        self.assertEqual(calls, [(10, 500)])

    def test_a_new_prompt_pauses_other_streams(self):
        # The first stream decodes 31 tokens at 5 ms (0.15 s); a 0.1 s prompt arrives
        # meanwhile and holds the engine.
        engine = MockEngine(1000, lambda _: 200, jitter=0, seed=1, slots=2,
                            step_time=lambda fills: 0.005)
        stats = self._run_together(engine, [(1, 31), (100, 2)], delays=[0, 0.03])
        self.assertAlmostEqual(stats[0].total_s, 0.25, delta=0.04)
        self.assertAlmostEqual(stats[1].ttft_s, 0.1, delta=0.03)


class CliTests(unittest.TestCase):
    def test_estimate_with_example_config(self):
        self.assertEqual(main(["estimate", "--config", "examples/rtx3080_desktop.toml"]), 0)

    def test_unknown_preset_is_an_error(self):
        self.assertEqual(main(["estimate", "--gpu", "voodoo2"]), 1)


if __name__ == "__main__":
    unittest.main()
