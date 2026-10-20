// Split text into token-sized pieces so a mock stream shows text at the rate a
// real tokenizer would. `charsPerToken` comes from data/tokenizer_ratios.json,
// measured with the actual tokenizers on the same texts.

// Pieces whose joined text equals `text`; their count matches the measured ratio.
// Below one character per token (byte-level fallback for CJK), some pieces are
// empty, so a character appears only after several tokens, as it does in reality.
export function chunkText(text, charsPerToken) {
  const chars = Array.from(text);
  if (!chars.length) return [];
  const n = Math.max(1, Math.round(chars.length / charsPerToken));
  const pieces = [];
  let prev = 0;
  for (let i = 1; i <= n; i++) {
    let pos = i === n ? chars.length : Math.round((i * chars.length) / n);
    // Latin-script tokens usually start with a space: snap to one nearby.
    if (charsPerToken >= 2 && i < n) pos = snapToSpace(chars, pos, prev, charsPerToken / 2);
    pos = Math.max(pos, prev);
    pieces.push(chars.slice(prev, pos).join(""));
    prev = pos;
  }
  return pieces;
}

function snapToSpace(chars, pos, min, radius) {
  for (let d = 0; d <= radius; d++) {
    for (const p of [pos - d, pos + d]) {
      if (p > min && p < chars.length && /\s/.test(chars[p])) return p;
    }
  }
  return pos;
}

// Characters per token for a text and tokenizer family. Unknown families fall
// back to the median of the measured ones for that text.
export function charsPerToken(ratios, textId, family) {
  const row = ratios.chars_per_token[textId] || {};
  if (family in row) return { value: row[family], measured: true };
  const values = Object.values(row).sort((a, b) => a - b);
  if (!values.length) return { value: 4, measured: false };
  const mid = values.length >> 1;
  const median = values.length % 2 ? values[mid] : (values[mid - 1] + values[mid]) / 2;
  return { value: median, measured: false };
}
