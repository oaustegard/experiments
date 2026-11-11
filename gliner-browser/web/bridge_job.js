// GPU Bridge job body (async function (ctx)): the browser NER runtime (ner.js) on a real GPU, parity
// against the NumPy reference's fixtures, then per-stage timings.
// Inputs (u8 = a file's raw bytes): weights (weights.bin), manifest, tokenizer, labels_tokenizer,
// fixtures (the pack's JSON files), bench (out/web_bench_texts.json from run/bench_texts.py: a seeded
// sample of mit-movie and CrossNER test sentences with 10 / 20 / 50 label sets),
// cfg = { sha, reps = 7, n = 500, heads = ["gpu", "js", "gpu-words"] }. Code comes from jsDelivr at the pushed commit `sha`.
// Parity runs once per span head in { gpu, js } (gpu: per-word layers and span scores on the GPU; js: jshead.js),
// plus the two heads' logits against each other. Timed per head in cfg.heads: (a) one 256-token text, 10
// labels; (b) one 30-token text with 10 and with 50 labels; (c) n short texts in one extract() call with 20
// labels (texts per second, pipelined and not). With head gpu, textEncoderGpuMs covers the encoder and the
// head (one submit); its difference from head js is the head's GPU time. Label vectors are cached per label
// string, so the label encoder is timed separately, cold, per label set (once, not per head).
// Submit from the hub (paths relative to the experiments checkout; M = gliner-browser/web/models/gliner-bi-edge-v2.0-t4-w8L63-h8):
//   python3 scripts/gpu_bridge.py submit gliner-browser/web/bridge_job.js --id gliner-web-1 --note "gliner browser runtime" \
//     --input weights=$M/weights.bin:u8 --input manifest=$M/manifest.json:u8 --input tokenizer=$M/tokenizer.json:u8 \
//     --input labels_tokenizer=$M/labels_tokenizer.json:u8 --input fixtures=$M/fixtures.json:u8 \
//     --input bench=gliner-browser/out/web_bench_texts.json:u8 --input cfg=@'{"sha":"<40-hex>","reps":7,"n":500}' --target Mac --timeout 900000
const { cfg } = ctx.inputs;
const dec = new TextDecoder();
const json = (k) => JSON.parse(dec.decode(ctx.inputs[k]));
const base = `https://cdn.jsdelivr.net/gh/oaustegard/experiments@${cfg.sha}/gliner-browser/web/`;
ctx.progress(0.02, "loading code");
const { NER, TOKENIZERS } = await ctx.importModule(base + "ner.js");
const tokenizers = await ctx.importModule(TOKENIZERS);
const manifest = json("manifest"), tj = json("tokenizer"), ltj = json("labels_tokenizer"), fx = json("fixtures"), bt = json("bench");
const device = await ctx.gpu();
let t = performance.now();
const ner = await NER.from({ manifest, tj, ltj, data: ctx.inputs.weights }, { device, tokenizers });
const setupMs = performance.now() - t;

// ---- parity (same checks as web/check.html, compact) ----
ctx.progress(0.08, "parity");
const same = (a, b) => JSON.stringify(a) === JSON.stringify(b);
const K = ner.g.max_width;
const par = {};
let labVec = 0;
await ner.setLabels(Object.keys(fx.label_encoder.labels));
for (const [lab, v] of Object.entries(fx.label_encoder.labels)) {
  const got = ner.lcache.get(lab).pooled;
  for (let j = 0; j < got.length; j++) labVec = Math.max(labVec, Math.abs(got[j] - v.pooled[j]));
}
par.labelVecMaxAbsDiff = labVec;
const parHead = async (head) => {
  ner.headMode = head;
  const p = { head, cases: fx.cases.length, ids: 0, labelIds: 0, logitMaxAbsDiff: 0, entityChecks: 0, entityMatches: 0, entityWorstScoreDiff: 0, headsUsed: [], mismatches: [] };
  const zs = [];
  for (const c of fx.cases) {
    p.ids += same(ner.prepare(c.text).ids, c.input_ids);
    p.labelIds += same(c.labels.map((l) => ner.labelIds(l)), c.label_input_ids);
    await ner.setLabels(c.labels_in);
    const got = { "0.5": await ner.extract(c.text, { threshold: 0.5, keepLogits: true }) };
    if (!p.headsUsed.includes(ner.lastTimings.head)) p.headsUsed.push(ner.lastTimings.head);
    const z = ner.lastLogits[0], C = c.labels.length;
    zs.push(z);
    c.logits.forEach((row, s) => row.forEach((kk, k) => { if (kk) kk.forEach((v, cc) => { p.logitMaxAbsDiff = Math.max(p.logitMaxAbsDiff, Math.abs(v - z[(s * K + k) * C + cc])); }); }));
    got["0.3"] = await ner.extract(c.text, { threshold: 0.3 });
    for (const th of ["0.5", "0.3"]) {
      const g = got[th], e = c.entities[th];
      p.entityChecks++;
      let ok = g.length === e.length, worst = 0;
      if (ok) g.forEach((x, i) => { const y = e[i]; if (x.start !== y.start || x.end !== y.end || x.label !== y.label || x.text !== y.text) ok = false; worst = Math.max(worst, Math.abs(x.score - y.score)); });
      if (ok && worst <= 1e-3) { p.entityMatches++; p.entityWorstScoreDiff = Math.max(p.entityWorstScoreDiff, worst); }
      else p.mismatches.push({ text: c.text, threshold: th, got: g, want: e });
    }
  }
  p.pass = p.ids === p.cases && p.labelIds === p.cases && p.entityMatches === p.entityChecks && p.logitMaxAbsDiff < 2e-3 && same(p.headsUsed, [head]);
  return { p, zs };
};
par.heads = {};
const zByHead = {};
for (const h of ["gpu", "js"]) { const { p, zs } = await parHead(h); par.heads[h] = p; zByHead[h] = zs; }
par.headsMaxAbsDiff = 0;
zByHead.gpu.forEach((z, i) => { const w = zByHead.js[i]; for (let j = 0; j < z.length; j++) if (z[j] !== w[j]) par.headsMaxAbsDiff = Math.max(par.headsMaxAbsDiff, Math.abs(z[j] - w[j])); });
par.pass = Object.values(par.heads).every((p) => p.pass) && par.headsMaxAbsDiff < 2e-3;
ner.headMode = "gpu";

// ---- timing ----
const reps = cfg.reps ?? 7;
const med = (xs) => { const s = [...xs].sort((a, b) => a - b); return s[Math.floor(s.length / 2)]; };
const STAGES = ["tokenizeMs", "textEncoderGpuMs", "readbackMs", "spanWordsMs", "spanScoreMs", "spanHeadMs", "decodeMs", "totalMs"];
const heads = cfg.heads ?? ["gpu", "js", "gpu-words"];
const medStages = (runs) => Object.fromEntries(STAGES.map((k) => [k, med(runs.map((r) => r[k]))]));
// a text of about `target` tokens: words of the bench sample until one more would pass it
const words = bt.texts.join(" ").split(" ");
const make = (target) => {
  let k = 1;
  while (k < words.length && ner.prepare(words.slice(0, k + 1).join(" ")).ids.length <= target) k++;
  return words.slice(0, k).join(" ");
};
async function labelsCold(labels) {
  ner.lcache.clear();
  const tm = await ner.setLabels(labels);
  return { labels: labels.length, labelTokens: tm.labelTokens, labelEncoderMs: tm.labelEncoderMs, labelProjectionMs: tm.labelProjectionMs,
    labelTokenizeMs: tm.labelTokenizeMs, msPerLabel: tm.totalMs / labels.length };
}
async function single(text, labels, head) {
  await ner.setLabels(labels);
  await ner.extract(text, { head });   // warm: compiles nothing new, but JITs the head at these shapes
  const runs = [];
  for (let r = 0; r < reps; r++) { const t0 = performance.now(); await ner.extract(text, { head }); runs.push({ ...ner.lastTimings, wallMs: performance.now() - t0 }); }
  return { head: runs[0].head, tokens: runs[0].tokens, words: runs[0].words, labels: labels.length, median: medStages(runs), wallMsMedian: med(runs.map((r) => r.wallMs)) };
}
const out = { labelEncoderCold: {} };
ctx.progress(0.15, "label encoder, cold");
for (const nl of [10, 20, 50]) out.labelEncoderCold[`labels${nl}`] = await labelsCold(bt[`labels${nl}`]);
const t256 = make(256), t30 = make(30);
const n = Math.min(cfg.n ?? 500, bt.texts.length), texts = bt.texts.slice(0, n);
heads.forEach((h) => { out[h] = {}; });
for (const [hi, head] of heads.entries()) {
  const o = out[head], f = (x) => 0.2 + 0.8 * (hi + x) / heads.length;
  ctx.progress(f(0), `${head}: 256-token text, 10 labels`);
  o.text256_labels10 = await single(t256, bt.labels10, head);
  ctx.progress(f(0.1), `${head}: 30-token text, 10 and 50 labels`);
  o.text30_labels10 = await single(t30, bt.labels10, head);
  o.text30_labels50 = await single(t30, bt.labels50, head);
  for (const pipeline of [true, false]) {
    ctx.progress(f(pipeline ? 0.3 : 0.65), `${head}: ${n} texts, 20 labels, pipeline ${pipeline}`);
    await ner.setLabels(bt.labels20);
    await ner.extract(texts, { pipeline, head });
    const runs = [];
    for (let r = 0; r < Math.min(reps, 5); r++) {
      const t0 = performance.now();
      await ner.extract(texts, { pipeline, head });
      runs.push({ ...ner.lastTimings, wallMs: performance.now() - t0 });
    }
    const m = medStages(runs), wall = med(runs.map((r) => r.wallMs));
    o[`batch${n}_labels20_${pipeline ? "pipelined" : "serial"}`] = {
      head: runs[0].head, texts: n, tokens: runs[0].tokens, words: runs[0].words, batches: runs[0].batches, labels: 20,
      median: m, wallMsMedian: wall, textsPerSecond: n / (wall / 1000),
      perText: Object.fromEntries(["tokenizeMs", "textEncoderGpuMs", "readbackMs", "spanWordsMs", "spanScoreMs", "decodeMs"].map((k) => [k, m[k] / n])),
    };
  }
}
ctx.progress(1, "done");
return {
  describe: ctx.describe ? await ctx.describe() : null,
  model: { variant: manifest.variant, headFormat: manifest.head_format, weightBytes: ctx.inputs.weights.byteLength },
  setupMs, parity: par, timing: out, reps, heads,
  note: "textEncoderGpuMs is submit-to-onSubmittedWorkDone (with head gpu / gpu-words it includes the span head's GPU work); in the pipelined batch run it overlaps the previous batch's JavaScript, so take stage shares from the serial run",
};
