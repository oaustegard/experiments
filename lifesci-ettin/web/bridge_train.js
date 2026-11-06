// GPU Bridge job: one stretch of a distillation arm (web/distill.js), resumable within the tab.
// cfg = { sha, arm, V, pad, from, to, total, lr, warm, save, urls: { params, ids, lens, order, teacher: [..] } }
// ctx.cache keeps the teacher vectors (f16, ~1 GB), the batch order, and the arm's trainer between jobs,
// so an arm's epochs are consecutive jobs in the same tab; a job that starts mid-arm without its trainer
// in the cache fails at once.
const { cfg } = ctx.inputs;
const gh = `https://cdn.jsdelivr.net/gh/oaustegard/experiments@${cfg.sha}/lifesci-ettin/web/`;
const { DistillTrainer } = await ctx.importModule(gh + "distill.js");
const C = ctx.cache, OUT = 768, BS = 128;
const t0 = performance.now();
const fetchAll = async (urls, label) => {
  const parts = [];
  for (let i = 0; i < urls.length; i++) { ctx.progress(null, `${label} ${i + 1}/${urls.length}`); parts.push(await ctx.fetchBytes(urls[i])); }
  const n = parts.reduce((a, p) => a + p.length, 0), all = new Uint8Array(n);
  let o = 0;
  for (const p of parts) { all.set(p, o); o += p.length; }
  return all;
};
if (C.get("teacherKey") !== cfg.urls.teacher.join()) {
  C.delete("teacher");
  const b = await fetchAll(cfg.urls.teacher, "teacher");
  C.set("teacher", new Uint16Array(b.buffer, b.byteOffset, b.length / 2));
  C.set("teacherKey", cfg.urls.teacher.join());
}
if (C.get("orderKey") !== cfg.urls.order) {
  const b = await ctx.fetchBytes(cfg.urls.order);
  C.set("order", new Uint32Array(b.buffer.slice(b.byteOffset, b.byteOffset + b.length)));
  C.set("orderKey", cfg.urls.order);
}
if (!C.get("f16")) {
  const t = new Float32Array(65536);
  for (let h = 0; h < 65536; h++) {
    const s = h & 0x8000 ? -1 : 1, e = (h >> 10) & 31, m = h & 1023;
    t[h] = e === 0 ? s * m * 2 ** -24 : e === 31 ? (m ? NaN : s * Infinity) : s * (1 + m / 1024) * 2 ** (e - 15);
  }
  C.set("f16", t);
}
const teacher = C.get("teacher"), order = C.get("order"), F16 = C.get("f16");
const key = `arm:${cfg.arm}`;
let A = C.get(key);
if (cfg.from === 0) {
  for (const k of [...C.keys()]) if (k.startsWith("arm:")) { C.get(k).tr?.device && C.delete(k); }
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
  A = { tr, ids, lens, offs, losses: [] };
  C.set(key, A);
} else if (!A) throw new Error(`arm ${cfg.arm} has no trainer in this tab's cache; restart it from step 0`);
const { tr, ids, lens, offs } = A;
const loadMs = performance.now() - t0;
const sched = (s) => cfg.lr * Math.min(1, (s + 1) / cfg.warm) * Math.max(0, (cfg.total - s) / Math.max(1, cfg.total - cfg.warm));
const tgt = new Float32Array(BS * OUT);
const losses = [];
let pend = 0;
const t1 = performance.now();
for (let s = cfg.from; s < cfg.to; s++) {
  const rows = order.subarray(s * BS, (s + 1) * BS);
  let S = 0;
  for (const r of rows) S = Math.max(S, lens[r]);
  const bIds = new Int32Array(BS * S).fill(cfg.pad), bLens = new Int32Array(BS);
  rows.forEach((r, b) => {
    bLens[b] = lens[r];
    for (let t = 0; t < lens[r]; t++) bIds[b * S + t] = ids[offs[r] + t];
    const o = r * OUT;
    for (let j = 0; j < OUT; j++) tgt[b * OUT + j] = F16[teacher[o + j]];
  });
  tr.loadBatch({ ids: bIds, B: BS, S, lens: bLens, tgt });
  tr.forwardBackward({ slot: pend });
  tr.adamStep({ lr: sched(s), step: s + 1 });
  pend++;
  if (pend === 64 || s === cfg.to - 1) {
    const l = await tr.read(tr.LOSS, pend);
    losses.push(...l);
    pend = 0;
    const done = (s + 1 - cfg.from) / (cfg.to - cfg.from);
    ctx.progress(done, `${cfg.arm} step ${s + 1}/${cfg.total} loss ${(l.reduce((a, b) => a + b, 0) / l.length).toFixed(4)}`);
  } else await tr.flush(false);
}
const trainMs = performance.now() - t1;
A.losses.push(...losses);
const out = { arm: cfg.arm, from: cfg.from, to: cfg.to, loadMs, trainMs, msPerStep: trainMs / (cfg.to - cfg.from), losses: Float32Array.from(losses) };
if (cfg.save) {
  const p = await tr.read(tr.PRM);
  // in pieces of 3M floats (12 MB): the GitHub blob API refuses one of 51 MB
  const P = 3 << 20;
  out.nPieces = Math.ceil(p.length / P);
  for (let i = 0; i < out.nPieces; i++) out[`p${i}`] = p.slice(i * P, (i + 1) * P);
  if (!p.every(Number.isFinite)) throw new Error("non-finite params");
}
return out;
