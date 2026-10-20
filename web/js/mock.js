// Speed-matched dummy token stream, the browser counterpart of mock.py.
// The emission time of every token is computed up front, so the UI just asks
// how many tokens are due at the current elapsed time on each animation frame.

const WORDS = (
  "the model simulates local inference on virtual hardware so you can feel " +
  "the latency before buying anything memory bandwidth decides decode speed " +
  "while compute decides prefill speed and context length slowly drags it down"
).split(" ");

function mulberry32(seed) {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

export class MockStream {
  constructor({ promptTokens, maxTokens, prefillTps, decodeTps, jitter = 0.05, seed = 1 }) {
    const rand = mulberry32(seed);
    this.times = new Float64Array(maxTokens);
    this.words = [];
    // Prefill produces the first token; each later token follows the decode
    // speed at the current context fill.
    let t = promptTokens / prefillTps;
    for (let i = 0; i < maxTokens; i++) {
      if (i) {
        const dt = 1 / decodeTps(promptTokens + i);
        t += Math.max(0, dt * (1 + jitter * (2 * rand() - 1)));
      }
      this.times[i] = t;
      this.words.push(WORDS[Math.floor(rand() * WORDS.length)]);
    }
  }

  get ttft() {
    return this.times.length ? this.times[0] : 0;
  }

  get totalTime() {
    return this.times.length ? this.times[this.times.length - 1] : 0;
  }

  // Number of tokens emitted by `elapsed` seconds (binary search).
  tokensDueBy(elapsed) {
    let lo = 0;
    let hi = this.times.length;
    while (lo < hi) {
      const mid = (lo + hi) >> 1;
      if (this.times[mid] <= elapsed) lo = mid + 1;
      else hi = mid;
    }
    return lo;
  }

  text(n) {
    return this.words.slice(0, n).join(" ");
  }
}
