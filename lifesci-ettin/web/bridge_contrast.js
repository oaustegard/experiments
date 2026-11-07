// GPU Bridge job: one stretch of a contrastive arm (web/distill.js with loss "nce", _lib/wgsl/train/nce.js).
// cfg = { sha, arm, V, pad, from, to, total, lr, warm, scale, save, urls: { params, ids, lens, pairs } }
// pairs: u32 [steps * P * 2], (query row, positive row) per pair, from run/pairs.py; the same file for
// every arm, since both arms' shards hold the same rows in the same order. A batch is the step's P
// queries followed by their P positives, padded to the longest. ctx.cache keeps the arm's trainer
// between jobs, as bridge_train.js does.
const { cfg } = ctx.inputs;
const gh = `https://cdn.jsdelivr.net/gh/oaustegard/experiments@${cfg.sha}/lifesci-ettin/web/`;
const { DistillTrainer } = await ctx.importModule(gh + "distill.js");
const C = ctx.cache, P = 64, BS = 2 * P;
const t0 = performance.now();
if (C.get("pairsKey") !== cfg.urls.pairs) {
  const b = await ctx.fetchBytes(cfg.urls.pairs);
  C.set("pairs", new Uint32Array(b.buffer.slice(b.byteOffset, b.byteOffset + b.length)));
  C.set("pairsKey", cfg.urls.pairs);
}
const pairs = C.get("pairs");
const key = `carm:${cfg.arm}`;
let A = C.get(key);
if (cfg.from === 0) {
  for (const k of [...C.keys()]) if (k.startsWith("arm:") || k.startsWith("carm:")) C.delete(k);
  const lens = new Uint16Array((await ctx.fetchBytes(cfg.urls.lens)).slice().buffer);
  const ib = await ctx.fetchBytes(cfg.urls.ids);
  const ids = new Uint16Array(ib.buffer.slice(ib.byteOffset, ib.byteOffset + ib.length));
  const offs = new Uint32Array(lens.length + 1);
  for (let i = 0; i < lens.length; i++) offs[i + 1] = offs[i] + lens[i];
  const pb = await ctx.fetchBytes(cfg.urls.params);
  const params = new Float32Array(pb.buffer.slice(pb.byteOffset, pb.byteOffset + pb.length));
  const tr = await DistillTrainer.create(await ctx.gpu(), { V: cfg.V, maxRows: BS * 128, maxSeqs: BS });
  if (params.length !== tr.nParams) throw new Error(`params ${params.length} != ${tr.nParams}`);
  tr.setParams(params);
  A = { tr, ids, lens, offs };
  C.set(key, A);
} else if (!A) throw new Error(`arm ${cfg.arm} has no trainer in this tab's cache; restart it from step 0`);
const { tr, ids, lens, offs } = A;
const loadMs = performance.now() - t0;
const sched = (s) => cfg.lr * Math.min(1, (s + 1) / cfg.warm) * Math.max(0, (cfg.total - s) / Math.max(1, cfg.total - cfg.warm));
const losses = [];
let pend = 0, tokens = 0;
const t1 = performance.now();
for (let s = cfg.from; s < cfg.to; s++) {
  const pr = pairs.subarray(s * P * 2, (s + 1) * P * 2);
  const rows = new Array(BS);
  for (let i = 0; i < P; i++) { rows[i] = pr[2 * i]; rows[P + i] = pr[2 * i + 1]; }
  let S = 0;
  for (const r of rows) S = Math.max(S, lens[r]);
  const bIds = new Int32Array(BS * S).fill(cfg.pad), bLens = new Int32Array(BS);
  rows.forEach((r, b) => {
    bLens[b] = lens[r];
    tokens += lens[r];
    for (let t = 0; t < lens[r]; t++) bIds[b * S + t] = ids[offs[r] + t];
  });
  tr.loadBatch({ ids: bIds, B: BS, S, lens: bLens });
  tr.forwardBackward({ slot: pend, loss: "nce", scale: cfg.scale });
  tr.adamStep({ lr: sched(s), step: s + 1 });
  pend++;
  if (pend === 64 || s === cfg.to - 1) {
    const l = await tr.read(tr.LOSS, pend);
    losses.push(...l);
    pend = 0;
    ctx.progress((s + 1 - cfg.from) / (cfg.to - cfg.from), `${cfg.arm} step ${s + 1}/${cfg.total} loss ${(l.reduce((a, b) => a + b, 0) / l.length).toFixed(4)}`);
  } else await tr.flush(false);
}
const trainMs = performance.now() - t1;
const out = { arm: cfg.arm, from: cfg.from, to: cfg.to, loadMs, trainMs, msPerStep: trainMs / (cfg.to - cfg.from), tokens, losses: Float32Array.from(losses) };
if (cfg.save) {
  const p = await tr.read(tr.PRM);
  if (!p.every(Number.isFinite)) throw new Error("non-finite params");
  const Q = 3 << 20;  // pieces of 12 MB: the GitHub blob API refuses one of 51 MB
  out.nPieces = Math.ceil(p.length / Q);
  for (let i = 0; i < out.nPieces; i++) out[`p${i}`] = p.slice(i * Q, (i + 1) * Q);
}
return out;
