// Kokoro-82M in the browser on onnxruntime-web: phonemes in, audio and word times out.
//
// Text -> phonemes is done ahead of time in Python (misaki; bake.py), so a line arrives as KPipeline would hand it
// to the model: chunks of at most 510 phonemes, each with its tokens. Word times use KPipeline's own rule
// (kokoro/pipeline.py join_timestamps) on the model's "duration" output, as kinescript-kokoro's onnx engine does.

export const SAMPLE_RATE = 24000;
const WORD = /[\p{L}\p{N}_]/u; // Python's \w, which is Unicode-aware

/** Load onnxruntime-web's session for the model. `ort` is the onnxruntime-web module. `lean` turns off the CPU
 * memory arena and memory-pattern planning, which trade memory for speed on repeated runs of one shape. */
export async function loadModel(ort, modelBytes, { ep = "wasm", lean = false } = {}) {
  const session = await ort.InferenceSession.create(modelBytes, {
    executionProviders: [ep],
    graphOptimizationLevel: "all",
    ...(lean ? { enableCpuMemArena: false, enableMemPattern: false } : {}),
  });
  return { ort, session };
}

/** float16 bytes -> Float32Array (no Float16Array needed). */
export function f16ToF32(buffer) {
  const h = new Uint16Array(buffer), out = new Float32Array(h.length);
  for (let i = 0; i < h.length; i++) {
    const x = h[i], s = x & 0x8000 ? -1 : 1, e = (x >> 10) & 0x1f, m = x & 0x3ff;
    out[i] = e === 0 ? s * m * 2 ** -24 : e === 31 ? (m ? NaN : s * Infinity) : s * (1 + m / 1024) * 2 ** (e - 15);
  }
  return out;
}

/** "a,a,b" -> the mean of those packs ([510 * 256] each), as KPipeline.load_voice averages a blend. */
export function blend(packs, speaker) {
  const names = speaker.split(","), out = new Float32Array(510 * 256);
  for (const n of names) {
    const p = packs[n];
    if (!p) throw new Error(`no voice pack ${n}`);
    for (let i = 0; i < out.length; i++) out[i] += p[i];
  }
  for (let i = 0; i < out.length; i++) out[i] /= names.length;
  return out;
}

/** KPipeline.join_timestamps: set start/end (seconds) on tokens from frames per phoneme (40 frames a second). */
export function joinTimestamps(tokens, dur) {
  const D = 80; // half-frames per second
  if (!tokens.length || dur.length < 3) return;
  let left = 2 * Math.max(0, dur[0] - 3), right = left, i = 1;
  for (const t of tokens) {
    if (i >= dur.length - 1) break;
    const n = [...t.phonemes].length;
    if (!n) {
      if (t.ws) { i += 1; left = right + dur[i]; right = left + dur[i]; i += 1; }
      continue;
    }
    const j = i + n;
    if (j >= dur.length) break;
    t.start = left / D;
    let tok = 0;
    for (let k = i; k < j; k++) tok += dur[k];
    const space = t.ws ? dur[j] : 0;
    left = right + 2 * tok + space;
    t.end = left / D;
    right = left + space;
    i = j + (t.ws ? 1 : 0);
  }
}

/** kinescript-kokoro's words_from_tokens: tokens -> [{word, start, end}] split where the text has spaces. */
export function wordsFromTokens(tokens, offset) {
  const out = [];
  let cur = null;
  for (const t of tokens) {
    cur ??= { word: "", start: null, end: null };
    cur.word += t.text;
    if (WORD.test(t.text) && t.start != null && t.end != null) {
      if (cur.start == null) cur.start = t.start + offset;
      cur.end = t.end + offset;
    }
    if (t.ws) { out.push(cur); cur = null; }
  }
  if (cur) out.push(cur);
  const timed = [];
  let last = offset;
  for (const w of out) {
    if (!WORD.test(w.word)) continue;
    if (w.start == null) w.start = w.end = last;
    timed.push({ word: w.word.trim(), start: +w.start.toFixed(4), end: +w.end.toFixed(4) });
    last = w.end;
  }
  return timed;
}

/** Speak one baked line: returns {audio: Float32Array, words, ms} (ms = time spent in session.run). */
export async function speak({ ort, session }, vocab, pack, line) {
  const parts = [], words = [];
  let at = 0, ms = 0;
  for (const chunk of line.chunks) {
    const ps = [...chunk.ps];
    const ids = ps.map((p) => vocab[p]).filter((x) => x !== undefined); // KModel drops unknown phonemes
    const input = new BigInt64Array(ids.length + 2);
    ids.forEach((x, k) => (input[k + 1] = BigInt(x)));
    const row = ps.length - 1; // style row by phoneme count, as KPipeline indexes pack[len(ps) - 1]
    const t0 = performance.now();
    const out = await session.run({
      input_ids: new ort.Tensor("int64", input, [1, input.length]),
      style: new ort.Tensor("float32", pack.slice(row * 256, row * 256 + 256), [1, 256]),
      speed: new ort.Tensor("float32", new Float32Array([line.rate]), [1]),
    });
    ms += performance.now() - t0;
    const audio = out.waveform.data, dur = Array.from(out.duration.data, Number);
    const tokens = chunk.tokens.map((t) => ({ ...t }));
    joinTimestamps(tokens, dur);
    words.push(...wordsFromTokens(tokens, at / SAMPLE_RATE));
    parts.push(audio);
    at += audio.length;
  }
  const audio = new Float32Array(at);
  let o = 0;
  for (const p of parts) { audio.set(p, o); o += p.length; }
  return { audio, words, ms };
}
