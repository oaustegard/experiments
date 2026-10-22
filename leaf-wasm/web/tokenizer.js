// BERT uncased WordPiece, written to match HF tokenizers' BertNormalizer + BertPreTokenizer +
// WordPiece + [CLS]/[SEP] template for mdbr-leaf-ir (lowercase, strip accents, CJK split).
// Checked id-for-id against HF on every eval text (test/parity.mjs).

// HF special tokens are matched on the raw text, before normalization.
const SPECIAL_RE = /\[(?:PAD|UNK|CLS|SEP|MASK)\]/g;
// Removed by clean_text: NUL, U+FFFD, and Unicode category C* except \t \n \r.
const DROP_RE = /[\0\ufffd]|(?![\t\n\r])\p{C}/gu;
const SPACE_RE = /\s/gu; // after DROP_RE, JS \s and Rust White_Space agree
const MARK_RE = /\p{Mn}/gu;
// ASCII punctuation (Rust is_ascii_punctuation, which includes $+<=>^`|~) or Unicode P*.
const PUNCT_RE = /^(?:[!-\/:-@\[-`{-~]|\p{P})$/u;

function isCJK(cp) {
  return (cp >= 0x4e00 && cp <= 0x9fff) || (cp >= 0x3400 && cp <= 0x4dbf) ||
    (cp >= 0x20000 && cp <= 0x2a6df) || (cp >= 0x2a700 && cp <= 0x2b73f) ||
    (cp >= 0x2b740 && cp <= 0x2b81f) || (cp >= 0x2b920 && cp <= 0x2ceaf) ||
    (cp >= 0xf900 && cp <= 0xfaff) || (cp >= 0x2f800 && cp <= 0x2fa1f);
}

function normalize(text) {
  let s = text.replace(DROP_RE, "").replace(SPACE_RE, " ");
  let out = "";
  for (const c of s) out += isCJK(c.codePointAt(0)) ? ` ${c} ` : c;
  s = out.normalize("NFD").replace(MARK_RE, "");
  out = "";
  for (const c of s) out += c.toLowerCase(); // per character, like Rust: no final-sigma context
  return out;
}

function preTokenize(s) {
  const words = [];
  let cur = "";
  for (const c of s) {
    if (c === " ") {
      if (cur) words.push(cur), (cur = "");
    } else if (PUNCT_RE.test(c)) {
      if (cur) words.push(cur), (cur = "");
      words.push(c);
    } else cur += c;
  }
  if (cur) words.push(cur);
  return words;
}

export class WordPiece {
  constructor(vocabText, maxLen = 128) {
    this.vocab = new Map();
    vocabText.split("\n").forEach((t, i) => { if (t) this.vocab.set(t, i); });
    this.maxLen = maxLen;
    [this.pad, this.unk, this.cls, this.sep] = ["[PAD]", "[UNK]", "[CLS]", "[SEP]"].map((t) => this.vocab.get(t));
  }

  word(w, out) {
    const chars = [...w];
    if (chars.length > 100) return out.push(this.unk);
    const pieces = [];
    for (let start = 0; start < chars.length;) {
      let end = chars.length, id;
      for (; end > start; end--) {
        id = this.vocab.get((start ? "##" : "") + chars.slice(start, end).join(""));
        if (id !== undefined) break;
      }
      if (end === start) return out.push(this.unk);
      pieces.push(id);
      start = end;
    }
    out.push(...pieces);
  }

  encode(text) {
    const ids = [];
    let last = 0;
    const segment = (s) => { for (const w of preTokenize(normalize(s))) this.word(w, ids); };
    for (const m of text.matchAll(SPECIAL_RE)) {
      segment(text.slice(last, m.index));
      ids.push(this.vocab.get(m[0]));
      last = m.index + m[0].length;
    }
    segment(text.slice(last));
    return [this.cls, ...ids.slice(0, this.maxLen - 2), this.sep];
  }
}

