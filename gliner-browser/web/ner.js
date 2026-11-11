// Zero-shot NER in the browser: gliner-bi-edge-v2.0 (Ettin-32m text encoder, MiniLM label encoder,
// markerV0 span head), packed by run/gpack.py. The text encoder runs on WebGPU (engine.js); the label
// encoder, span head and decoding run in plain JavaScript (jshead.js). run/gref.py (GRefBi) is the spec.
//
//   const ner = await NER.load("models/gliner-bi-edge-v2.0-t4-w8L63-h8/");
//   await ner.setLabels(["person", "company", "city"]);      // encoded once, cached per label string
//   const ents = await ner.extract("Ada Lovelace met Babbage in London.", { threshold: 0.5 });
//   // [{ start, end, text, label, score }], start/end in code points (GRef's offsets); u16Start/u16End
//   // index the JavaScript string (text === str.slice(u16Start, u16End))
//   const many = await ner.extract(posts);                     // string[] -> one list per text
//   ner.lastTimings                                            // ms per stage, token and word counts
//
// Many short texts go through the text encoder together (packed batches of up to maxTokens tokens /
// maxSeqs texts); the span head and decoding then run per text, overlapped with the next batch's GPU work.

import { TextEngine, MAX_SEQ } from "./engine.js";
import { Pack, LabelEncoder, SpanHead, decodeSpans, splitWords } from "./jshead.js";

export const TOKENIZERS = "https://cdn.jsdelivr.net/npm/@huggingface/tokenizers@0.2.0/dist/tokenizers.mjs";
const now = () => performance.now();

export class NER {
  // modelUrl: directory with manifest.json, weights.bin, tokenizer.json, labels_tokenizer.json.
  // opts: device (a GPUDevice), tokenizers (the imported tokenizer module), maxTokens, maxSeqs, maxPos.
  static async load(modelUrl, opts = {}) {
    const base = new URL(modelUrl.endsWith("/") ? modelUrl : modelUrl + "/", globalThis.location?.href);
    const t0 = now();
    const get = (f) => fetch(new URL(f, base)).then((r) => { if (!r.ok) throw new Error(`${f}: HTTP ${r.status}`); return r; });
    const manifest = await get("manifest.json").then((r) => r.json());
    const g = manifest.gliner;
    const [tj, ltj, data] = await Promise.all([get(g.tokenizer).then((r) => r.json()),
      get(g.label_tokenizer).then((r) => r.json()), get("weights.bin").then((r) => r.arrayBuffer())]);
    const ner = await NER.from({ manifest, tj, ltj, data }, opts);
    ner.info.fetchMs = now() - t0 - ner.info.initMs;
    return ner;
  }

  // Same, from parts already in memory (a GPU Bridge job gets them as inputs). data: ArrayBuffer or Uint8Array.
  static async from({ manifest, tj, ltj, data }, opts = {}) {
    const g = manifest.gliner;
    if (g?.encoder_kind !== "bi" || g.span_mode !== "markerV0" || g.subtoken_pooling !== "first" || manifest.lstm?.length || manifest.projection)
      throw new Error("NER runs the packed bi-encoder markerV0 model (first-subtoken pooling, no BiLSTM, no projection)");
    const t0 = now();
    let device = opts.device;
    if (!device) {
      const adapter = await globalThis.navigator?.gpu?.requestAdapter({ powerPreference: "high-performance" });
      if (!adapter) throw new Error("WebGPU is not available in this browser: the NER text encoder needs it (Chrome/Edge 113+, Safari 26, or Firefox 141+ on Windows)");
      device = await adapter.requestDevice({ requiredLimits: {
        maxStorageBufferBindingSize: adapter.limits.maxStorageBufferBindingSize, maxBufferSize: adapter.limits.maxBufferSize } });
    }
    const { Tokenizer } = opts.tokenizers ?? await import(TOKENIZERS);
    const bytes = data instanceof Uint8Array ? data : new Uint8Array(data);
    const engine = await TextEngine.create(device, manifest, bytes,
      { maxTokens: opts.maxTokens ?? 8192, maxSeqs: opts.maxSeqs ?? 512, maxPos: opts.maxPos ?? 8192 });
    const pack = new Pack(bytes);
    const ner = new NER();
    Object.assign(ner, {
      manifest, g, engine, device,
      tok: new Tokenizer(tj, {}), ltok: new Tokenizer(ltj, {}),
      lenc: new LabelEncoder(pack, manifest.label_encoder),
      head: new SpanHead(pack, manifest.ghead, g.max_width),
      wcache: new Map(), lcache: new Map(), labels: null,
      info: { initMs: now() - t0, weightBytes: bytes.byteLength, variant: manifest.variant, headFormat: manifest.head_format },
      lastTimings: null, labelTimings: null,
    });
    return ner;
  }

  // ---- input construction (GRef.split / word_ids / build with no prompt) ----
  wordIds(w) {
    let ids = this.wcache.get(w);
    if (!ids) {
      if (this.wcache.size > 200000) this.wcache.clear();
      ids = this.tok.encode(w, { add_special_tokens: false }).ids;
      this.wcache.set(w, ids);
    }
    return ids;
  }

  // text -> { words, ids, first (Int32Array: each word's subtoken position, -1 if none), n }, or null
  // for a blank text. truncate: drop the words past the encoder's 1,024-token window instead of throwing.
  prepare(text, truncate = false) {
    const words = splitWords(text, this.g.max_len);
    if (!words.length) return null;   // GRef: not text.strip() -> []
    const { cls, sep } = this.g.special;
    const room = Math.min(this.g.model_max_length, MAX_SEQ) - 2;
    const ids = [cls], first = new Int32Array(words.length).fill(-1);
    let n = words.length;
    for (let i = 0; i < words.length; i++) {
      const sub = this.wordIds(words[i].w);
      if (ids.length - 1 + sub.length > room) {
        if (this.g.model_max_length - 2 > room && !truncate)
          throw new Error(`text of ${words.length} words needs more than the browser encoder's ${room} tokens (word ${i}); split it or pass { truncate: true }`);
        if (truncate) { n = i; break; }
      }
      if (sub.length && ids.length - 1 < room) first[i] = ids.length;
      for (const id of sub) if (ids.length - 1 < room) ids.push(id);
    }
    ids.push(sep);
    return { words: words.slice(0, n), ids, first: first.subarray(0, n), n };
  }

  // GRef.build's words_mask for a prepared text (1-based word index at each word's first subtoken).
  static wordsMask(p) {
    const wm = new Array(p.ids.length).fill(0);
    p.first.forEach((t, i) => { if (t >= 0) wm[t] = i + 1; });
    return wm;
  }

  // ---- labels ----
  static normLabels(labels) { return [...new Set((typeof labels === "string" ? [labels] : labels).map(String))]; }

  // The label tokenizer's [CLS] ... [SEP] ids, truncated to label_max_length as the library's call does.
  labelIds(label) {
    let ids = this.ltok.encode(String(label)).ids;
    const L = this.g.label_max_length;
    if (L && ids.length > L) ids = ids.slice(0, L - 1).concat([ids[ids.length - 1]]);
    return ids;
  }

  // Encode the labels not yet cached (all in one pass through the label encoder), then fix the class order.
  async setLabels(labels) {
    const t0 = now();
    const norm = NER.normLabels(labels);
    const fresh = norm.filter((l) => !this.lcache.has(l));
    const tm = { labels: norm.length, encoded: fresh.length, labelTokenizeMs: 0, labelEncoderMs: 0, labelProjectionMs: 0, labelTokens: 0 };
    if (fresh.length) {
      const idsList = fresh.map((l) => this.labelIds(l));
      const t1 = now();
      tm.labelTokenizeMs = t1 - t0;
      tm.labelTokens = idsList.reduce((s, x) => s + x.length, 0);
      const { pooled } = this.lenc.encode(idsList);
      const t2 = now();
      tm.labelEncoderMs = t2 - t1;
      const { q, r } = this.head.labels(pooled, fresh.length);
      const H = this.head.H, D = this.head.D;
      fresh.forEach((l, i) => this.lcache.set(l, { ids: idsList[i], pooled: pooled.slice(i * H, i * H + H), q: q.slice(i * D, i * D + D), r: r[i] }));
      tm.labelProjectionMs = now() - t2;
    }
    const C = norm.length, D = this.head.D;
    const Q = new Float32Array(C * D), R = new Float64Array(C);
    norm.forEach((l, c) => { const e = this.lcache.get(l); Q.set(e.q, c * D); R[c] = e.r; });
    this.labels = norm; this.Q = Q; this.R = R;
    tm.totalMs = now() - t0;
    this.labelTimings = tm;
    return tm;
  }

  // ---- extraction ----
  // texts: string or string[]. opts: threshold (0.5), flat (true), multiLabel (false), labels (calls
  // setLabels first), truncate (false), pipeline (true: span head of batch i while the GPU encodes i + 1),
  // batchTokens (cap a GPU batch below the engine's maxTokens), keepLogits (store per-text logits
  // [n, K, C] in this.lastLogits).
  async extract(texts, opts = {}) {
    const { threshold = 0.5, flat = true, multiLabel = false, truncate = false, pipeline = true, keepLogits = false } = opts;
    let labelT = null;
    if (opts.labels) labelT = await this.setLabels(opts.labels);
    if (!this.labels) throw new Error("call setLabels(labels) before extract()");
    const single = typeof texts === "string", list = single ? [texts] : Array.from(texts);
    const tm = { texts: list.length, labels: this.labels.length, tokens: 0, words: 0, batches: 0,
      tokenizeMs: 0, textEncoderGpuMs: 0, readbackMs: 0, labelEncoderMs: labelT ? labelT.labelTokenizeMs + labelT.labelEncoderMs + labelT.labelProjectionMs : 0,
      spanHeadMs: 0, spanWordsMs: 0, spanScoreMs: 0, decodeMs: 0, totalMs: 0 };
    const T0 = now();
    const preps = list.map((t) => (typeof t === "string" ? this.prepare(t, truncate) : null));
    tm.tokenizeMs = now() - T0;
    const results = list.map(() => []);
    const logits = keepLogits ? list.map(() => null) : null;
    // batches of texts, within the engine's token / sequence / row limits
    const e = this.engine, batches = [], maxTok = Math.min(e.maxTokens, opts.batchTokens ?? Infinity);
    let cur = [], toks = 0, rows = 0;
    preps.forEach((p, i) => {
      if (!p) return;
      tm.tokens += p.ids.length; tm.words += p.n;
      if (cur.length && (toks + p.ids.length > maxTok || cur.length >= e.maxSeqs || rows + p.n > e.maxPos)) { batches.push(cur); cur = []; toks = 0; rows = 0; }
      cur.push(i); toks += p.ids.length; rows += p.n;
    });
    if (cur.length) batches.push(cur);
    tm.batches = batches.length;
    const C = this.labels.length, K = this.g.max_width, H = this.head.H;
    const T = this.g.logit_temperature ?? 1, b = this.g.logit_bias ?? 0;
    const submit = (bi) => {
      const idx = [];
      let off = 0;
      for (const i of batches[bi]) {
        const p = preps[i];
        for (let w = 0; w < p.n; w++) if (p.first[w] >= 0) idx.push(off + p.first[w]);
        off += p.ids.length;
      }
      return e.submit(batches[bi].map((i) => preps[i].ids), new Uint32Array(idx));
    };
    let job = batches.length ? submit(0) : null;
    let Wbuf = new Float32Array(0), Z = new Float32Array(0);
    for (let bi = 0; bi < batches.length; bi++) {
      const r = await job.read();
      tm.textEncoderGpuMs += r.gpuMs; tm.readbackMs += r.readMs;
      job = null;
      if (pipeline && bi + 1 < batches.length) job = submit(bi + 1);
      let ro = 0;
      for (const i of batches[bi]) {
        const p = preps[i], n = p.n;
        // word vectors: the gathered row at each word's first subtoken, zeros for a word with none (GRef.words_from_hidden)
        let W;
        if (p.first.every((t) => t >= 0)) { W = r.rows.subarray(ro * H, (ro + n) * H); ro += n; }
        else {
          if (Wbuf.length < n * H) Wbuf = new Float32Array(n * H);
          W = Wbuf.subarray(0, n * H); W.fill(0);
          for (let w = 0; w < n; w++) if (p.first[w] >= 0) { W.set(r.rows.subarray(ro * H, (ro + 1) * H), w * H); ro++; }
        }
        if (!C) continue;
        const t1 = now();
        const { A, B } = this.head.words(W, n);
        const t2 = now();
        if (Z.length < n * K * C) Z = new Float32Array(n * K * C);
        const z = this.head.scores(A, B, n, this.Q, this.R, C, Z);
        const t3 = now();
        const spans = decodeSpans(z, n, K, C, threshold, { flat, multiLabel, T, b });
        results[i] = spans.map((s) => {
          const a = p.words[s.start], o = p.words[s.end];
          return { start: a.cs, end: o.ce, text: list[i].slice(a.us, o.ue), label: this.labels[s.cls], score: s.score, u16Start: a.us, u16End: o.ue };
        });
        if (logits) logits[i] = z.slice(0, n * K * C);
        const t4 = now();
        tm.spanWordsMs += t2 - t1; tm.spanScoreMs += t3 - t2; tm.decodeMs += t4 - t3;
      }
      if (!pipeline && bi + 1 < batches.length) job = submit(bi + 1);
    }
    tm.spanHeadMs = tm.spanWordsMs + tm.spanScoreMs;
    tm.totalMs = now() - T0 + tm.labelEncoderMs;
    this.lastTimings = tm;
    if (logits) this.lastLogits = logits;
    return single ? results[0] : results;
  }

  destroy() { this.engine.destroy(); }
}
