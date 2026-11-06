// GPU Bridge job: mean-pooled 256-d features (before the Dense) for pre-tokenized texts, forward only.
// cfg = { sha, V, pad, params (URL, f32 flat) | arm (use that arm's trainer in this tab's cache), texts: { ids, lens } URLs }
// Output: pooled f32 [N, 256] in pieces of 3M floats, in input order. The Dense is applied on the CPU.
const { cfg } = ctx.inputs;
const gh = `https://cdn.jsdelivr.net/gh/oaustegard/experiments@${cfg.sha}/lifesci-ettin/web/`;
const { DistillTrainer } = await ctx.importModule(gh + "distill.js");
const ROWS = 128 * 128, H = 256;
let tr;
if (cfg.arm) {
  tr = ctx.cache.get(`arm:${cfg.arm}`)?.tr;
  if (!tr) throw new Error(`no cached trainer for ${cfg.arm}`);
} else {
  tr = await DistillTrainer.create(await ctx.gpu(), { V: cfg.V, maxRows: ROWS, maxSeqs: 128 });
  const pb = await ctx.fetchBytes(cfg.params);
  tr.setParams(new Float32Array(pb.buffer.slice(pb.byteOffset, pb.byteOffset + pb.length)));
}
const lens = new Uint16Array((await ctx.fetchBytes(cfg.texts.lens)).slice().buffer);
const ib = await ctx.fetchBytes(cfg.texts.ids);
const ids = new Uint16Array(ib.buffer.slice(ib.byteOffset, ib.byteOffset + ib.length));
const N = lens.length, offs = new Uint32Array(N + 1);
for (let i = 0; i < N; i++) offs[i + 1] = offs[i] + lens[i];
const order = Array.from({ length: N }, (_, i) => i).sort((a, b) => lens[a] - lens[b]);
const out = new Float32Array(N * H);
const t0 = performance.now();
for (let i = 0; i < N;) {
  const S = lens[order[Math.min(N - 1, i + 127)]];
  const B = Math.max(1, Math.min(128, Math.floor(ROWS / S), N - i));
  const rows = order.slice(i, i + B), Sb = lens[rows[rows.length - 1]];
  const bIds = new Int32Array(B * Sb).fill(cfg.pad), bLens = new Int32Array(B);
  rows.forEach((r, b) => { bLens[b] = lens[r]; for (let t = 0; t < lens[r]; t++) bIds[b * Sb + t] = ids[offs[r] + t]; });
  tr.loadBatch({ ids: bIds, B, S: Sb, lens: bLens });
  tr.forward();
  const pl = await tr.read(tr.PL, B * H);
  rows.forEach((r, b) => out.set(pl.subarray(b * H, (b + 1) * H), r * H));
  i += B;
  ctx.progress(i / N, `encode ${i}/${N}`);
}
const res = { N, ms: performance.now() - t0 };
const P = 3 << 20;
res.nPieces = Math.ceil(out.length / P);
for (let k = 0; k < res.nPieces; k++) res[`p${k}`] = out.slice(k * P, (k + 1) * P);
return res;
