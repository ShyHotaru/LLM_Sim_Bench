"""Mock inference engine that streams text pieces at a simulated speed."""

from __future__ import annotations

import contextlib
import random
import threading
import time
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass


@dataclass
class StreamStats:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    queued_s: float = 0.0  # waiting for a free slot
    ttft_s: float = 0.0  # from the request, including the wait
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

    Streams running at the same time (threads of the mock server) share the engine
    like llama-server's `slots` with continuous batching: at most `slots` run, the
    rest wait; prompts are processed one after another, and the other streams pause
    meanwhile; each decode step takes `step_time(fills)` for the KV fills of all
    active streams. Without `step_time`, every stream decodes at `decode_tps`.
    """

    def __init__(
        self,
        prefill_tps: float,
        decode_tps: Callable[[int], float],
        jitter: float = 0.05,
        seed: int | None = None,
        step_time: Callable[[Sequence[int]], float] | None = None,
        slots: int = 1,
    ):
        self.prefill_tps = prefill_tps
        self.decode_tps = decode_tps
        self.jitter = jitter
        self.rng = random.Random(seed)
        self.step_time = step_time
        self.slots = slots
        self._free = threading.BoundedSemaphore(slots)
        self._lock = threading.Lock()
        self._fills: dict[object, int] = {}  # decoding stream -> KV fill
        self._prefill_free = 0.0  # when the prompt queue is empty
        self._prefills: list[tuple[object, float, float]] = []  # (stream, start, end)

    def stream(self, prompt_tokens: int, pieces: Sequence[str],
               stats: StreamStats | None = None, cached_tokens: int = 0) -> Iterator[str]:
        """Yield `pieces` one token at a time after prefilling `prompt_tokens`.

        `cached_tokens` are already in the KV cache (earlier turns of a chat): they
        are not prefilled again but slow decode like the rest of the context.
        """
        stats = stats if stats is not None else StreamStats()
        stats.prompt_tokens = prompt_tokens
        start = time.perf_counter()
        with self._slot(stats, start) as key:
            # Prefill produces the first token.
            with self._lock:
                now = time.perf_counter()
                begin = max(now, self._prefill_free)
                deadline = begin + prompt_tokens / self.prefill_tps
                self._prefill_free = deadline
                self._prefills = [p for p in self._prefills if p[2] > now] + [
                    (key, begin, deadline)]
                self._fills[key] = cached_tokens + prompt_tokens
            for i, piece in enumerate(pieces):
                if i:
                    deadline = self._next_deadline(key, deadline,
                                                    cached_tokens + prompt_tokens + i)
                _sleep_until(deadline)
                if i == 0:
                    stats.ttft_s = time.perf_counter() - start
                stats.completion_tokens = i + 1
                stats.total_s = time.perf_counter() - start
                yield piece

    @contextlib.contextmanager
    def _slot(self, stats: StreamStats, start: float):
        self._free.acquire()
        stats.queued_s = time.perf_counter() - start
        key = object()
        try:
            yield key
        finally:
            with self._lock:
                self._fills.pop(key, None)
            self._free.release()

    def _next_deadline(self, key: object, prev: float, fill: int) -> float:
        """Deadline of a stream's next token: one decode step for every active stream,
        plus the time other streams' prompts held the engine in between."""
        with self._lock:
            self._fills[key] = fill
            if self.step_time:
                dt = self.step_time(list(self._fills.values()))
            else:
                dt = 1.0 / self.decode_tps(fill)
            # Spend the step outside the other streams' prompt processing.
            t, need = prev, self._jittered(dt)
            for other, begin, end in sorted(self._prefills, key=lambda p: p[1]):
                if other is key or end <= t:
                    continue
                if begin >= t + need:
                    break
                need -= max(0.0, begin - t)
                t = end
            return t + need

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
