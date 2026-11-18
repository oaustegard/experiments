// Text helpers ported from indra-ocr (vocab.py decode, rtl.py flip, layout_post.join_lines).

import { isAlpha, isBidiL, isLtr, isLower, isMark, isRtl } from "./uprops.js";

export class Vocab {
  constructor(chars) {
    this.chars = chars;
    this.index = new Map(chars.map((c, i) => [c, i + 1]));
    this.hasCommands = chars.some((c) => [...c].length > 1);
    this.jamo = this.index.has("ᄀ");
  }
  decode(ids) {
    const toks = [];
    for (const i of ids) if (i > 0 && i <= this.chars.length) toks.push(this.chars[i - 1]);
    if (this.jamo) return toks.join("").normalize("NFC");
    if (!this.hasCommands) return toks.join("");
    const out = [];
    toks.forEach((t, k) => {
      out.push(t);
      const nxt = k + 1 < toks.length ? toks[k + 1] : "";
      if (t.length > 2 && t[0] === "\\" && /^[A-Za-z]+$/.test(t.slice(1)) && nxt && isAlpha([...nxt][0])) out.push(" ");
    });
    return out.join("");
  }
}

const chars = (s) => Array.from(s);

export const hasRtl = (t) => chars(t).some(isRtl);

function rtlBase(t) {
  let r = 0, l = 0;
  for (const c of chars(t)) { if (isRtl(c)) r++; if (isBidiL(c)) l++; }
  return r > l;
}

const MIRROR = Object.fromEntries([..."()[]{}<>«»‹›"].map((c, i) => [c, ")(][}{><»«›‹"[i]]));

function clusters(t) {
  const out = [];
  for (const c of chars(t)) {
    if (out.length && isMark(c)) out[out.length - 1] += c;
    else out.push(c);
  }
  return out;
}

const kind = (cl) => { const c = chars(cl)[0]; return isRtl(c) ? "R" : isLtr(c) ? "L" : "N"; };

function runs(kinds, strong, stop) {
  const spans = [];
  let start = null, last = null;
  [...kinds, "#"].forEach((k, i) => {
    if (k === strong) { if (start === null) start = i; last = i; }
    else if (stop.has(k) || k === "#") { if (start !== null) spans.push([start, last + 1]); start = last = null; }
  });
  return spans;
}

const mirror = (cl) => { const c = chars(cl); return (MIRROR[c[0]] ?? c[0]) + c.slice(1).join(""); };

// rtl.flip: logical order <-> visual order
export function flip(text, mirrorOn = true, baseRtl = null) {
  if (!hasRtl(text)) return text;
  const cls = clusters(text);
  const kinds = cls.map(kind);
  if (baseRtl === null ? rtlBase(text) : baseRtl) {
    const units = [];
    const spans = runs(kinds, "L", new Set(["R"]));
    let i = 0;
    for (const [a, b] of [...spans, [cls.length, cls.length]]) {
      for (let j = i; j < a; j++) units.push(mirrorOn ? mirror(cls[j]) : cls[j]);
      if (a < b) units.push(cls.slice(a, b).join(""));
      i = b;
    }
    return units.reverse().join("");
  }
  const out = cls.slice();
  for (const [a, b] of runs(kinds, "R", new Set(["L"]))) {
    const seg = cls.slice(a, b).reverse();
    for (let k = 0; k < seg.length; k++) out[a + k] = mirrorOn ? mirror(seg[k]) : seg[k];
  }
  return out.join("");
}

function cjk(c) {
  const o = c.codePointAt(0);
  return (0x3000 <= o && o <= 0x30ff) || (0x3400 <= o && o <= 0x9fff) || (0xf900 <= o && o <= 0xfaff) ||
    (0xff00 <= o && o <= 0xffef) || (0x20000 <= o && o <= 0x2fa1f);
}

// Python str.strip() / split() whitespace
export const pyStrip = (s) => s.replace(/^\s+|\s+$/gu, "");
export const pySplit = (s) => s.split(/\s+/u).filter((x) => x.length);

// layout_post.join_lines
export function joinLines(texts) {
  let out = "";
  for (let t of texts) {
    t = pyStrip(t);
    if (!t) continue;
    const oc = chars(out), tc = chars(t);
    if (oc.length > 1 && out.endsWith("-") && isAlpha(oc[oc.length - 2]) && isAlpha(tc[0]) && isLower(tc[0])) {
      out = out.slice(0, -1) + t;
    } else if (out && cjk(oc[oc.length - 1]) && cjk(tc[0])) {
      out += t;
    } else {
      out = out ? `${out} ${t}` : t;
    }
  }
  return out;
}

export function htmlEscape(s) {
  return s.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
}
