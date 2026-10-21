// Share the whole UI state in the URL hash as base64url-encoded JSON.

export function encodeState(state) {
  const bytes = new TextEncoder().encode(JSON.stringify(state));
  let bin = "";
  for (const b of bytes) bin += String.fromCharCode(b);
  return btoa(bin).replace(/\+/g, "-").replace(/\//g, "_").replace(/=+$/, "");
}

export function decodeState(text) {
  try {
    const bin = atob(text.replace(/-/g, "+").replace(/_/g, "/"));
    const bytes = Uint8Array.from(bin, (c) => c.charCodeAt(0));
    const state = JSON.parse(new TextDecoder().decode(bytes));
    return state && typeof state === "object" ? state : null;
  } catch {
    return null;
  }
}
