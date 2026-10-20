// UI translations. Strings live in web/locales/<lang>.json; English is the
// source and the fallback for any missing key.

export const LANGS = { en: "English", ko: "한국어", ja: "日本語" };
export const DEFAULT_LANG = "en";
const STORAGE_KEY = "llmsim.lang";

let base = {};
let messages = {};
export let lang = DEFAULT_LANG;

// Choose a language: ?lang= in the URL, then a saved choice, then the browser's.
export function pickLang(search, saved, preferred = []) {
  const fromUrl = new URLSearchParams(search).get("lang");
  for (const cand of [fromUrl, saved, ...preferred]) {
    const code = String(cand || "").toLowerCase().split("-")[0];
    if (code in LANGS) return code;
  }
  return DEFAULT_LANG;
}

export function format(template, params = {}) {
  return template.replace(/\{(\w+)\}/g, (m, k) => (k in params ? String(params[k]) : m));
}

export function setMessages(code, english, translated) {
  lang = code;
  base = english;
  messages = translated;
}

export function t(key, params) {
  const template = messages[key] ?? base[key] ?? key;
  return format(template, params);
}

// Locale tag for number formatting.
export function numberLocale() {
  return { en: "en-US", ko: "ko-KR", ja: "ja-JP" }[lang];
}

export async function loadLocale(code) {
  const get = async (c) => (await fetch(`locales/${c}.json`)).json();
  const english = await get(DEFAULT_LANG);
  setMessages(code, english, code === DEFAULT_LANG ? english : await get(code));
}

export function savedLang() {
  try {
    return localStorage.getItem(STORAGE_KEY);
  } catch {
    return null;
  }
}

export function saveLang(code) {
  try {
    localStorage.setItem(STORAGE_KEY, code);
  } catch { /* storage may be blocked; the choice then lasts for this page only */ }
}

// Fill static markup: data-i18n sets text, data-i18n-attr="attr:key" sets an attribute.
export function translateDom(root = document) {
  for (const el of root.querySelectorAll("[data-i18n]")) el.textContent = t(el.dataset.i18n);
  for (const el of root.querySelectorAll("[data-i18n-attr]")) {
    const [attr, key] = el.dataset.i18nAttr.split(":");
    el.setAttribute(attr, t(key));
  }
}
