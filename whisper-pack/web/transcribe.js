// Transcription on top of the engine: language detection, Whisper's timestamp rules, a repeated
// n-gram ban, segment parsing and the seek loop for audio longer than one 30 s window.
//
//   const tr = new Transcriber(whisper, tokenizer);
//   const { text, chunks, language } = await tr.transcribe(pcm16k, { language: "auto", onPartial });
//
// chunks are [{ timestamp: [start, end], text }] in seconds, the shape transformers.js returns.

import { logMel } from "./mel.js";

const SR = 16000, WINDOW = 30, STEP = 0.02;

export class Transcriber {
  constructor(whisper, tokenizer) {
    this.w = whisper;
    this.tok = tokenizer;
    const g = this.g = whisper.gen;
    this.tsBegin = g.no_timestamps + 1;          // <|0.00|>
    this.maxInitial = g.max_initial_timestamp_index ?? 50;
    this.langIds = g.multilingual ? Object.entries(g.lang_to_id).map(([k, v]) => [k.slice(2, -2), v]) : [];
  }

  // Language of the window last given to encode(): the most likely language token after start-of-transcript.
  async detectLanguage() {
    if (!this.g.multilingual) return "en";
    const logits = await this.w.step(this.g.sot, 0);
    let best = this.langIds[0];
    for (const l of this.langIds) if (logits[l[1]] > logits[best[1]]) best = l;
    return best[0];
  }

  // Whisper's timestamp rules, applied in place. `seq` is the tokens generated so far in this window.
  _timestampRules(logits, seq) {
    const g = this.g, tb = this.tsBegin, V = logits.length;
    logits[g.no_timestamps] = -Infinity;
    const n = seq.length;
    const lastTs = n >= 1 && seq[n - 1] >= tb;
    const penTs = n < 2 || seq[n - 2] >= tb;
    if (lastTs) {
      if (penTs) for (let v = tb; v < V; v++) logits[v] = -Infinity;   // two in a row: text must follow
      else for (let v = 0; v < g.eot; v++) logits[v] = -Infinity;       // text then one: a second or the end
    }
    let lastStamp = -1;
    for (let i = n - 1; i >= 0; i--) if (seq[i] >= tb) { lastStamp = seq[i]; break; }
    if (lastStamp >= 0) {   // timestamps never go backwards
      const floor = lastTs && !penTs ? lastStamp : lastStamp + 1;
      for (let v = tb; v < floor; v++) logits[v] = -Infinity;
    }
    if (n === 0) {
      for (let v = 0; v < tb; v++) logits[v] = -Infinity;               // a window opens with a timestamp
      for (let v = tb + this.maxInitial + 1; v < V; v++) logits[v] = -Infinity;
    }
    // if the timestamps together outweigh every text token, take a timestamp
    let mx = -Infinity;
    for (let v = 0; v < V; v++) if (logits[v] > mx) mx = logits[v];
    let sumTs = 0, maxText = -Infinity;
    for (let v = tb; v < V; v++) sumTs += Math.exp(logits[v] - mx);
    for (let v = 0; v < tb; v++) if (logits[v] > maxText) maxText = logits[v];
    if (Math.log(sumTs) + mx > maxText) for (let v = 0; v < tb; v++) logits[v] = -Infinity;
  }

  // Ban any token that would repeat an n-gram already generated in this window.
  _noRepeat(logits, seq, n) {
    if (!n || seq.length < n - 1) return;
    const tail = seq.slice(seq.length - (n - 1));
    for (let i = 0; i + n - 1 < seq.length; i++) {
      let same = true;
      for (let j = 0; j < n - 1; j++) if (seq[i + j] !== tail[j]) { same = false; break; }
      if (same) logits[seq[i + n - 1]] = -Infinity;
    }
  }

  // Greedy decode of the encoded window. Returns the generated ids (end-of-text excluded).
  async decodeWindow(opts = {}) {
    const g = this.g, w = this.w;
    const timestamps = opts.timestamps !== false;
    const prompt = [g.sot];
    if (g.multilingual) prompt.push(g.lang_to_id[`<|${opts.language || "en"}|>`], g.task_to_id[opts.task || "transcribe"]);
    if (!timestamps) prompt.push(g.no_timestamps);
    if (prompt.includes(undefined)) throw new Error("unknown language or task: " + opts.language + " / " + opts.task);
    const maxNew = Math.min(opts.maxNew ?? 224, w.dims.n_text_ctx - prompt.length - 1);
    let pos = 0, logits = null;
    for (let i = 0; i < prompt.length; i++) logits = await w.step(prompt[i], pos++, i === prompt.length - 1);
    const seq = [];
    while (seq.length < maxNew) {
      for (const t of g.suppress_tokens || []) logits[t] = -Infinity;
      if (seq.length === 0) for (const t of g.begin_suppress_tokens || []) logits[t] = -Infinity;
      if (timestamps) this._timestampRules(logits, seq);
      this._noRepeat(logits, seq, opts.noRepeatNgram ?? 0);
      let best = 0;
      for (let v = 1; v < logits.length; v++) if (logits[v] > logits[best]) best = v;
      if (best === g.eot) break;
      seq.push(best);
      if (opts.onToken) opts.onToken(best, seq);
      logits = await w.step(best, pos++);
    }
    return seq;
  }

  // Split a window's ids into segments. Returns { chunks, advance } where advance is how many
  // seconds of this window the segments account for (the seek step for long audio).
  parseWindow(ids, offset, windowSeconds) {
    const tb = this.tsBegin, chunks = [];
    let start = null, text = [], lastPairEnd = null, last = null;
    const close = (end) => {
      const t = this.tok.decode(text);
      if (t.trim()) chunks.push({ timestamp: [offset + (start ?? 0), end == null ? null : offset + end], text: t });
      text = [];
    };
    for (let i = 0; i < ids.length; i++) {
      const id = ids[i];
      if (id >= tb) {
        const s = (id - tb) * STEP;
        if (text.length) { close(s); start = null; if (i + 1 < ids.length && ids[i + 1] >= tb) lastPairEnd = s; }
        else start = s;
        last = s;
      } else text.push(id);
    }
    const endsOnOne = ids.length >= 1 && ids[ids.length - 1] >= tb && !(ids.length >= 2 && ids[ids.length - 2] >= tb);
    if (text.length) close(null);
    // a closing pair means speech was cut at that time: resume there; otherwise the window is done
    const advance = !endsOnOne && lastPairEnd != null && lastPairEnd > 0 ? lastPairEnd : windowSeconds;
    return { chunks, advance, last };
  }

  // pcm: Float32Array, 16 kHz mono, any length. opts.timings, if given an object, receives the
  // milliseconds spent per phase summed over windows: mel, encode, detect, decode, parse, total.
  async transcribe(pcm, opts = {}) {
    const tm = opts.timings || null, now = () => performance.now(), t0 = now();
    const add = (k, t) => { if (tm) tm[k] = (tm[k] || 0) + (now() - t); };
    const total = pcm.length / SR;
    const chunks = [];
    let language = opts.language && opts.language !== "auto" ? opts.language : null;
    let seek = 0, windows = 0;
    const single = total <= WINDOW;
    while (seek < total - 0.1 || windows === 0) {
      const from = Math.round(seek * SR);
      let t = now();
      const mel = logMel(pcm.subarray(from, from + WINDOW * SR), this.w.dims.n_mels);
      add("mel", t); t = now();
      await this.w.encode(mel);
      add("encode", t); t = now();
      if (!language) { language = await this.detectLanguage(); add("detect", t); }
      t = now();
      let live = [];
      const ids = await this.decodeWindow({
        language, task: opts.task, timestamps: opts.timestamps, noRepeatNgram: opts.noRepeatNgram ?? 3,
        onToken: (id, seq) => { if (opts.onPartial && id < this.tsBegin) { live.push(id); opts.onPartial(this.tok.decode(live), seek); } },
      });
      add("decode", t); t = now();
      const left = Math.min(WINDOW, total - seek);
      const p = this.parseWindow(ids, seek, left);
      add("parse", t);
      chunks.push(...p.chunks);
      windows++;
      seek += single ? WINDOW : Math.max(p.advance, 1);
      if (opts.onProgress) opts.onProgress(Math.min(1, seek / total), windows);
    }
    if (tm) tm.total = now() - t0;
    return { text: chunks.map((c) => c.text).join("").trim(), chunks, language, windows };
  }
}
