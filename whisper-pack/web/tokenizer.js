// Dependency-free Whisper token-id decoder (byte-level BPE) for a Hugging Face
// tokenizer.json. Works in a browser, a Web Worker and Node. Decode only.

// GPT-2 byte <-> unicode table: printable bytes map to themselves, the rest to 256+n.
function unicodeToByte() {
  const keep = [];
  for (let b = 33; b <= 126; b++) keep.push(b);
  for (let b = 161; b <= 172; b++) keep.push(b);
  for (let b = 174; b <= 255; b++) keep.push(b);
  const map = new Map();
  const seen = new Set(keep);
  for (const b of keep) map.set(String.fromCharCode(b), b);
  let n = 0;
  for (let b = 0; b < 256; b++) {
    if (!seen.has(b)) map.set(String.fromCharCode(256 + n++), b);
  }
  return map;
}

export class Tokenizer {
  static async load(url) {
    const res = await fetch(url);
    if (!res.ok) throw new Error(`tokenizer fetch failed: ${res.status} ${url}`);
    return Tokenizer.fromJSON(await res.json());
  }

  static fromJSON(obj) {
    return new Tokenizer(obj);
  }

  constructor(obj) {
    const vocab = obj.model.vocab;
    const u2b = unicodeToByte();
    this.tokens = new Map();   // id -> Uint8Array of bytes (regular tokens)
    for (const [str, id] of Object.entries(vocab)) {
      const bytes = new Uint8Array([...str].map((ch) => {
        const b = u2b.get(ch);
        if (b === undefined) throw new Error(`unmapped char in vocab: ${JSON.stringify(str)}`);
        return b;
      }));
      this.tokens.set(id, bytes);
    }
    this.added = new Map();    // id -> literal text
    for (const t of obj.added_tokens || []) this.added.set(t.id, t.content);
    const ts = (obj.added_tokens || []).find((t) => t.content === '<|0.00|>');
    this.timestampBegin = ts ? ts.id : null;
    this.timestampEnd = null;
    if (ts) {
      let last = ts.id;
      while (this.added.get(last + 1) && /^<\|\d+\.\d\d\|>$/.test(this.added.get(last + 1))) last++;
      this.timestampEnd = last;
    }
    this._dec = new TextDecoder('utf-8', { fatal: false });
    this._enc = new TextEncoder();
  }

  isSpecial(id) {
    return this.added.has(id);
  }

  timestampSeconds(id) {
    if (this.timestampBegin === null || id < this.timestampBegin || id > this.timestampEnd) return null;
    return Math.round((id - this.timestampBegin) * 2) / 100;
  }

  // Timestamp tokens never appear in the output, matching HF's default.
  decode(ids, { skipSpecial = true } = {}) {
    // Collect bytes of consecutive regular tokens; added tokens are emitted as
    // literal text between byte runs (decoded per run, like HF).
    let out = '';
    let run = [];
    let runLen = 0;
    const flush = () => {
      if (!runLen) return;
      const buf = new Uint8Array(runLen);
      let o = 0;
      for (const b of run) { buf.set(b, o); o += b.length; }
      out += this._dec.decode(buf);
      run = []; runLen = 0;
    };
    for (const id of ids) {
      const lit = this.added.get(id);
      // HF's WhisperTokenizer.decode drops timestamp tokens even with
      // skip_special_tokens=False (decode_with_timestamps defaults to False).
      if (this.timestampSeconds(id) !== null) continue;
      if (lit !== undefined) {
        if (!skipSpecial) { flush(); out += lit; }
        continue;
      }
      const b = this.tokens.get(id);
      if (b === undefined) continue; // unknown id: HF yields nothing for it
      run.push(b); runLen += b.length;
    }
    flush();
    return out;
  }
}
