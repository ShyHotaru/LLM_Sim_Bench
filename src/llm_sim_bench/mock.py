"""Mock inference engine that streams text pieces at a simulated speed."""

from __future__ import annotations

import random
import time
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass


@dataclass
class StreamStats:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    ttft_s: float = 0.0
    total_s: float = 0.0

    @property
    def decode_tps(self) -> float:
        gen = self.total_s - self.ttft_s
        return (self.completion_tokens - 1) / gen if gen > 0 and self.completion_tokens > 1 else 0.0


class MockEngine:
    """Paces token output by a prefill rate and a context-dependent decode rate.

    `decode_tps` is a callable of the current KV fill so decode slows as the
    context grows, matching the estimator. Pieces come from texts.chunk_text, one
    per token.
    """

    def __init__(
        self,
        prefill_tps: float,
        decode_tps: Callable[[int], float],
        jitter: float = 0.05,
        seed: int | None = None,
    ):
        self.prefill_tps = prefill_tps
        self.decode_tps = decode_tps
        self.jitter = jitter
        self.rng = random.Random(seed)

    def stream(self, prompt_tokens: int, pieces: Sequence[str],
               stats: StreamStats | None = None, cached_tokens: int = 0) -> Iterator[str]:
        """Yield `pieces` one token at a time after prefilling `prompt_tokens`.

        `cached_tokens` are already in the KV cache (earlier turns of a chat): they
        are not prefilled again but slow decode like the rest of the context.
        """
        stats = stats if stats is not None else StreamStats()
        stats.prompt_tokens = prompt_tokens
        start = time.perf_counter()
        # Prefill produces the first token.
        deadline = start + prompt_tokens / self.prefill_tps
        for i, piece in enumerate(pieces):
            if i:
                rate = self.decode_tps(cached_tokens + prompt_tokens + i)
                deadline += self._jittered(1.0 / rate)
            _sleep_until(deadline)
            if i == 0:
                stats.ttft_s = time.perf_counter() - start
            stats.completion_tokens = i + 1
            stats.total_s = time.perf_counter() - start
            yield piece

    def _jittered(self, dt: float) -> float:
        return max(0.0, dt * (1 + self.rng.uniform(-self.jitter, self.jitter)))


def _sleep_until(deadline: float) -> None:
    # Absolute deadlines avoid cumulative drift; spin the last ~2 ms because
    # time.sleep granularity is coarse on Windows.
    while True:
        left = deadline - time.perf_counter()
        if left <= 0:
            return
        if left > 0.002:
            time.sleep(left - 0.002)
