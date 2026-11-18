// The Unicode character properties the pipeline reads, as Python's str methods and
// unicodedata give them. JS regex classes answer most; `exceptions` (uexc.json, made by
// run/unicode_exceptions.mjs) holds the characters of the recognizers' vocabularies where they
// disagree. Bits: 1 isalpha, 2 isdigit, 4 isupper, 8 islower, 16 isspace, 32 bidi R/AL,
// 64 bidi L, 128 bidi EN/AN, 256 category Mn/Me, 512 category Mc, 1024 category Lo,
// 2048 east-asian wide (W/F).

const RE = {
  alpha: /^[\p{L}]$/u,
  digit: /^[\p{Nd}]$/u,
  upper: /^[\p{Lu}\p{Lt}]$/u,
  lower: /^[\p{Ll}]$/u,
  space: /^\s$/u,
  rtl: /^[\p{Script=Hebrew}\p{Script=Arabic}\p{Script=Syriac}\p{Script=Thaana}\p{Script=Nko}\p{Script=Samaritan}\p{Script=Mandaic}]$/u,
  en: /^[0-9٠-٩۰-۹²³¹⁰-⁹₀-₉]$/u,
  mark: /^[\p{Mn}\p{Me}]$/u,
  mc: /^\p{Mc}$/u,
  lo: /^\p{Lo}$/u,
  wide: /^[ᄀ-ᅟ⺀-〾ぁ-㏿㐀-䶿一-鿿ꀀ-꓏가-힣豈-﫿︰-﹏＀-｠￠-￦\u{20000}-\u{2FFFD}\u{30000}-\u{3FFFD}]$/u,
};

export function approxFlags(c) {
  let f = 0;
  if (RE.alpha.test(c)) f |= 1;
  if (RE.digit.test(c)) f |= 2;
  if (RE.upper.test(c)) f |= 4;
  if (RE.lower.test(c)) f |= 8;
  if (RE.space.test(c)) f |= 16;
  const mark = RE.mark.test(c);
  if (RE.en.test(c)) f |= 128;
  else if (RE.rtl.test(c) && !mark) f |= 32;
  else if ((f & 1) || RE.mc.test(c)) f |= 64;
  if (mark) f |= 256;
  if (RE.mc.test(c)) f |= 512;
  if (RE.lo.test(c)) f |= 1024;
  if (RE.wide.test(c)) f |= 2048;
  return f;
}

let EXC = {};
export function setExceptions(e) { EXC = e || {}; }
const memo = new Map();
export function flags(c) {
  let f = memo.get(c);
  if (f === undefined) {
    f = c in EXC ? EXC[c] : approxFlags(c);
    memo.set(c, f);
  }
  return f;
}

export const isAlpha = (c) => (flags(c) & 1) !== 0;
export const isDigit = (c) => (flags(c) & 2) !== 0;
export const isUpper = (c) => (flags(c) & 4) !== 0;
export const isLower = (c) => (flags(c) & 8) !== 0;
export const isSpace = (c) => (flags(c) & 16) !== 0;
export const isRtl = (c) => (flags(c) & 32) !== 0;
export const isBidiL = (c) => (flags(c) & 64) !== 0;
export const isLtr = (c) => (flags(c) & (64 | 128)) !== 0;
export const isMark = (c) => (flags(c) & 256) !== 0;
export const isMc = (c) => (flags(c) & 512) !== 0;
export const isLo = (c) => (flags(c) & 1024) !== 0;
export const isWide = (c) => (flags(c) & 2048) !== 0;
