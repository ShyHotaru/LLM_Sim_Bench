// Speed-matched mock playback, the browser counterpart of mock.py.
// The time of every event is computed up front, so the UI only asks how many
// events are due at the current elapsed time on each animation frame.

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

// turns: [{ user: pieces | null, assistant: pieces }], where pieces are token
// texts from chunkText(). Before the first turn the context already holds
// `cachedTokens` tokens, which cost no time (a conversation or prompt cache that is
// already in the KV cache), followed by `promptTokens` tokens that the first turn
// has to process. Each turn prefills only its new user tokens (the
// earlier conversation stays in the KV cache, as llama.cpp's prompt cache does),
// then generates up to `maxTokens` assistant tokens at the decode speed for the
// current context fill. `pause` seconds separate turns.
//
// Events: { t, role: "user" | "assistant", turn, text, fill }. A user event
// carries the whole message; an assistant event is one token.
export class Schedule {
  constructor({ turns, promptTokens, maxTokens, prefillTps, decodeTps, cachedTokens = 0,
                pause = 1, jitter = 0.05, seed = 1 }) {
    const rand = mulberry32(seed);
    this.events = [];
    this.turns = [];
    let t = 0;
    let fill = cachedTokens + promptTokens;
    turns.forEach((turn, k) => {
      if (k) t += pause;
      const start = t;
      let newTokens = k ? 0 : promptTokens;
      if (turn.user) {
        this.events.push({ t, role: "user", turn: k, text: turn.user.join(""), fill });
        newTokens += turn.user.length;
        fill += turn.user.length;
      }
      t += newTokens / prefillTps;
      const first = t;
      const pieces = turn.assistant.slice(0, maxTokens);
      pieces.forEach((text, i) => {
        if (i) t += Math.max(0, (1 / decodeTps(fill)) * (1 + jitter * (2 * rand() - 1)));
        fill += 1;
        this.events.push({ t, role: "assistant", turn: k, text, fill });
      });
      this.turns.push({ start, prefillTokens: newTokens, ttft: first - start, firstAt: first,
                        end: t, tokens: pieces.length });
    });
  }

  get totalTime() {
    return this.events.length ? this.events[this.events.length - 1].t : 0;
  }

  // Number of events that have happened by `elapsed` seconds (binary search).
  dueBy(elapsed) {
    let lo = 0;
    let hi = this.events.length;
    while (lo < hi) {
      const mid = (lo + hi) >> 1;
      if (this.events[mid].t <= elapsed) lo = mid + 1;
      else hi = mid;
    }
    return lo;
  }
}
