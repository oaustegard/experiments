// GPU Bridge job body (async function (ctx)): where the Ettin-17m decider's GPU time goes, kernel by kernel,
// and what merging the LoRA spike into the int8 base would cost in accuracy. See ../PLAN.md.
//
// Subjects, both unchanged code from the repo at cfg.sha:
//   S  adapter-library SpikeEngine: int8 base (±31 codes), LoRA side path as two f32 matmuls per module (rank 8 padded to 32)
//   T  ettin-wgsl-train Trainer: f32 dequantized base, rank-8 LoRA, forward / forward+backward+AdamW
// Measurements:
//   e2e      the graph as recorded, `reps` copies in one submission; wall ms and GPU timestamp ms per copy
//   drop     the same with one kernel group left out (the "minus lora" arm is the merged model's graph)
//   iso      each dispatch alone, repeated k times in its own pass; per-dispatch µs
//   null     an empty kernel dispatched 4000 times, with and without a read-write binding (per-dispatch floor)
//   merge    the invoice spike merged into the base and requantized per column: fresh ±31, fresh ±127, and
//            "aligned" (scale = base scale / 4, so base codes stay exact and only the delta rounds)
// Inputs: cfg { sha, base?: URL prefix of the repo (default jsDelivr at sha), quick?: bool }.
const { cfg } = ctx.inputs;
const base = cfg.base ?? `https://cdn.jsdelivr.net/gh/oaustegard/experiments@${cfg.sha}/`;
const quick = !!cfg.quick, smoke = !!cfg.smoke;   // smoke: SwiftShader correctness run, every loop cut short
const code = base + "decider-lora-creator/web/", models = base + "adapter-library/web/models/";
ctx.progress(0.01, "loading code");
const C = await ctx.importModule(code + "creator.js");
const Bk = await ctx.importModule(code + "backends.js");
const { SpikeEngine } = await ctx.importModule(base + "adapter-library/web/engine.js");
const { Trainer } = await ctx.importModule(base + "ettin-wgsl-train/web/train.js");
const dev = await ctx.gpu();
const out = { sha: cfg.sha, quick, describe: ctx.describe ? await ctx.describe() : null };
const hasTS = dev.features.has("timestamp-query");
out.timestampQuery = hasTS;
const FB = (u) => ctx.fetchBytes(u);
const J = async (u) => JSON.parse(new TextDecoder().decode(await FB(u)));

ctx.progress(0.03, "loading model");
const [manifest, wb, tok, demo, sm, sbytes] = await Promise.all([J(models + "base31/manifest.json"), FB(models + "base31/weights.bin"),
  C.loadTokenizer(models + "base31/tokenizer.json"), J(code + "demo/invoice_processing.json"),
  J(models + "workflows/spikes/invoice_processing.json"), FB(models + "workflows/spikes/invoice_processing.bin")]);
const testCases = demo.cases.filter((c) => c.split === "test"), trainCases = demo.cases.filter((c) => c.split !== "test");
let test = C.examples(demo.decision, testCases, tok.build);
let train = C.examples(demo.decision, trainCases, tok.build);
const shortest = (xs, n) => xs.slice().sort((a, b) => a.ids.length - b.ids.length).slice(0, n);
if (smoke) { test = shortest(test, 4); train = shortest(train, 2); }
if (quick) test = test.slice(0, 12);
const one = test.filter((e) => e.caseId === test[0].caseId).slice(0, smoke ? 2 : 99);            // one decision: every question of one case
const b16 = test.slice(0, smoke ? 3 : 16);
const toks = (xs) => xs.reduce((a, x) => a + x.ids.length, 0);
out.workloads = { one: { seqs: one.length, tokens: toks(one) }, b16: { seqs: b16.length, tokens: toks(b16) },
  test: { seqs: test.length, tokens: toks(test), meanLen: toks(test) / test.length } };
ctx.log(`workloads ${JSON.stringify(out.workloads)}`);

// ---------- timing helpers ----------
const med = (a) => { const s = a.slice().sort((x, y) => x - y); return s[s.length >> 1]; };
const qs = hasTS ? dev.createQuerySet({ type: "timestamp", count: 2 }) : null;
const qres = hasTS ? dev.createBuffer({ size: 16, usage: GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC }) : null;
const qread = hasTS ? dev.createBuffer({ size: 16, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST }) : null;
// encodeInto(pass) records into one compute pass; returns { wall, gpu } ms per copy, medians over trials (first dropped)
async function timePass(encodeInto, reps, trials = 5) {
  if (smoke) { reps = 1; trials = 1; }
  const W = [], G = [];
  for (let tr = 0; tr <= trials; tr++) {
    const enc = dev.createCommandEncoder();
    const pass = enc.beginComputePass(hasTS ? { timestampWrites: { querySet: qs, beginningOfPassWriteIndex: 0, endOfPassWriteIndex: 1 } } : undefined);
    for (let r = 0; r < reps; r++) encodeInto(pass);
    pass.end();
    if (hasTS) { enc.resolveQuerySet(qs, 0, 2, qres, 0); enc.copyBufferToBuffer(qres, 0, qread, 0, 16); }
    const t0 = performance.now();
    dev.queue.submit([enc.finish()]);
    await dev.queue.onSubmittedWorkDone();
    const t1 = performance.now();
    let g = null;
    if (hasTS) { await qread.mapAsync(GPUMapMode.READ); const v = new BigUint64Array(qread.getMappedRange().slice(0)); qread.unmap(); g = Number(v[1] - v[0]) / 1e6; }
    if (tr > 0) { W.push((t1 - t0) / reps); if (g !== null) G.push(g / reps); }
  }
  return { wall: med(W), gpu: G.length ? med(G) : null };
}
// isolated cost of one dispatch: k back-to-back copies in one pass, k grown until the pass takes >= 3 ms
async function isoUs(encodeOne) {
  let k = 8, r;
  for (;;) {
    r = await timePass((p) => { for (let i = 0; i < k; i++) encodeOne(p); }, 1, 3);
    const ms = r.gpu ?? r.wall;
    if (ms >= 3 || k >= 1024 || smoke) break;
    k = Math.min(1024, Math.ceil(k * Math.max(2, 4 / Math.max(ms, 0.05))));
  }
  return { us: 1000 * (r.gpu ?? r.wall) / k, k };
}

// ---------- per-dispatch floor ----------
ctx.progress(0.06, "null dispatches");
{
  const mk = (code) => dev.createComputePipeline({ layout: "auto", compute: { module: dev.createShaderModule({ code }), entryPoint: "main" } });
  const p0 = mk(`@compute @workgroup_size(64) fn main() {}`);
  const buf = dev.createBuffer({ size: 256, usage: GPUBufferUsage.STORAGE });
  const p1 = mk(`@group(0) @binding(0) var<storage, read_write> a: array<f32>;
@compute @workgroup_size(64) fn main(@builtin(local_invocation_index) l: u32) { if (l == 0u) { a[0] = a[0] + 1.0; } }`);
  const bg1 = dev.createBindGroup({ layout: p1.getBindGroupLayout(0), entries: [{ binding: 0, resource: { buffer: buf } }] });
  const N = smoke ? 50 : 4000;
  const a = await timePass((p) => { p.setPipeline(p0); for (let i = 0; i < N; i++) p.dispatchWorkgroups(1); }, 1);
  const b = await timePass((p) => { p.setPipeline(p1); p.setBindGroup(0, bg1); for (let i = 0; i < N; i++) p.dispatchWorkgroups(1); }, 1);
  out.null = { noBinding_us: 1000 * (a.gpu ?? a.wall) / N, rwBinding_us: 1000 * (b.gpu ?? b.wall) / N,
    noBinding_wall_us: 1000 * a.wall / N, rwBinding_wall_us: 1000 * b.wall / N };
  ctx.log(`null dispatch ${JSON.stringify(out.null)}`);
}

// ---------- S: SpikeEngine ----------
const groupS = (k) => k.startsWith("matmul_f32") ? "lora" : k.startsWith("matmul_i8") ? "base_mm" : k.startsWith("norm") ? "norm"
  : k.startsWith("attn") ? "attn" : k;
const maxLen = Math.max(...test.map((x) => x.ids.length), ...train.map((x) => x.ids.length));
const mkSpike = async (bytes) => {
  const e = await SpikeEngine.create(dev, manifest, bytes, { maxTokens: 16 * maxLen, maxSeqs: 16, maxPos: 2048 });
  e.setSpike(sm, sbytes);
  return e;
};
const scoreAll = async (e, exs) => { const z = []; for (let i = 0; i < exs.length; i += 16) z.push(...await e.score(exs.slice(i, i + 16))); return z; };

if (!cfg.skipS) {   // S and merge arms (skipS: trainer-only smoke run)
ctx.progress(0.1, "SpikeEngine");
const eS = await mkSpike(wb);
out.S = { steps: eS.steps.length, byGroup: {} };
for (const s of eS.steps) out.S.byGroup[groupS(s.kind)] = (out.S.byGroup[groupS(s.kind)] ?? 0) + 1;
const recS = (e, steps, n) => (p) => { for (const s of steps) { p.setPipeline(e.pipes[s.kind]); p.setBindGroup(0, s.bg); p.dispatchWorkgroups(...s.grid(n)); } };
for (const [wname, seqs] of [["one", one], ["b16", b16]]) {
  ctx.log(`S ${wname}: start`);
  await eS.score(seqs);                                     // leaves tok / seqs / gather inputs set for this batch
  const n = [toks(seqs), seqs.length];
  const W = { score_wall_ms: null, e2e: null, drop: {}, iso: [] };
  const sw = [];
  for (let i = 0; i < (smoke ? 1 : 8); i++) { await eS.score(seqs); const t0 = performance.now(); await eS.score(seqs); sw.push(performance.now() - t0); }
  W.score_wall_ms = med(sw);                                // what the page sees: submit, GPU, readback, JS head
  const reps = wname === "one" ? 20 : 5;
  W.e2e = await timePass(recS(eS, eS.steps, n), reps);
  ctx.log(`S ${wname}: e2e ${JSON.stringify(W.e2e)}`);
  for (const g of Object.keys(out.S.byGroup)) {
    if (g === "gather") continue;
    W.drop[g] = await timePass(recS(eS, eS.steps.filter((s) => groupS(s.kind) !== g), n), reps);
  }
  W.drop.all_but_base_mm = await timePass(recS(eS, eS.steps.filter((s) => groupS(s.kind) === "base_mm"), n), reps);
  if (!quick || wname === "one") {
    for (const [i, s] of eS.steps.entries()) {
      if (smoke && i > 3) break;
      const r = await isoUs((p) => { p.setPipeline(eS.pipes[s.kind]); p.setBindGroup(0, s.bg); p.dispatchWorkgroups(...s.grid(n)); });
      W.iso.push({ i, kind: s.kind, group: groupS(s.kind), grid: s.grid(n), us: +r.us.toFixed(2), k: r.k });
    }
    W.isoSum_ms = W.iso.reduce((a, x) => a + x.us, 0) / 1000;
    W.isoByGroup_ms = {};
    for (const x of W.iso) W.isoByGroup_ms[x.group] = (W.isoByGroup_ms[x.group] ?? 0) + x.us / 1000;
  }
  out.S[wname] = W;
  ctx.log(`S ${wname}: e2e ${JSON.stringify(W.e2e)} score ${W.score_wall_ms.toFixed(3)} ms; drop lora ${JSON.stringify(W.drop.lora)}; isoSum ${W.isoSum_ms?.toFixed(3)}`);
  ctx.progress(wname === "one" ? 0.25 : 0.4, `SpikeEngine ${wname} done`);
}

// ---------- merge arms ----------
ctx.progress(0.42, "merge arms");
const u = C.unpackSpike(sm, sbytes);
function mergedBytes(mode) {    // mode: "fresh31" | "fresh127" | "aligned"
  const outB = wb.slice(), q8 = new Int8Array(outB.buffer), dv = new DataView(outB.buffer);
  const src = new DataView(wb.buffer, wb.byteOffset, wb.byteLength), qin = new Int8Array(wb.buffer, wb.byteOffset, wb.byteLength);
  const st = { maxDeltaSteps: 0, clipped: 0, alignDiv: {} };
  manifest.layers.forEach((L, li) => {
    for (const k of C.MODS) {
      const w = L[k], K = w.K, N = w.N, m = u.layers[li][k], r = m.r;
      const s = new Float32Array(N);
      for (let n = 0; n < N; n++) s[n] = src.getFloat32(w.s.offset + 4 * n, true);
      const Wm = new Float32Array(K * N);
      for (let kk = 0; kk < K; kk++) for (let n = 0; n < N; n++) Wm[kk * N + n] = qin[w.q.offset + kk * N + n] * s[n];
      for (let n = 0; n < N; n++) for (let j = 0; j < r; j++) {
        const bv = m.B[n * r + j];
        if (bv === 0) continue;
        const A = m.A.subarray(j * K, (j + 1) * K);
        for (let kk = 0; kk < K; kk++) Wm[kk * N + n] += bv * A[kk];
      }
      for (let n = 0; n < N; n++) {
        let mx = 0, md = 0;
        for (let kk = 0; kk < K; kk++) { mx = Math.max(mx, Math.abs(Wm[kk * N + n])); md = Math.max(md, Math.abs(Wm[kk * N + n] - qin[w.q.offset + kk * N + n] * s[n])); }
        st.maxDeltaSteps = Math.max(st.maxDeltaSteps, s[n] > 0 ? md / s[n] : 0);
        let sc;
        if (mode === "fresh31") sc = mx / 31;
        else if (mode === "fresh127") sc = mx / 127;
        else { let d = 4; while (d > 1 && mx / (s[n] / d) > 127) d--; sc = s[n] / d; st.alignDiv[d] = (st.alignDiv[d] ?? 0) + 1; }
        if (!(sc > 0)) sc = 1;
        const Q = mode === "fresh31" ? 31 : 127;
        dv.setFloat32(w.s.offset + 4 * n, sc, true);
        for (let kk = 0; kk < K; kk++) {
          let q = Math.round(Wm[kk * N + n] / sc);
          if (q > Q || q < -Q) { st.clipped++; q = Math.max(-Q, Math.min(Q, q)); }
          q8[w.q.offset + kk * N + n] = q;
        }
      }
    }
  });
  return { bytes: outB, st };
}
const temps = sm.temps ?? {};
const acc = (z) => C.metrics(test, C.probsOf(test, z, temps));
ctx.log("merge: scoring unmerged");
const zRef = await scoreAll(eS, test);
out.merge = { unmerged: acc(zRef) };
const cmp = (z) => {
  let agree = 0, mad = 0, mx = 0, cnt = 0;
  z.forEach((v, i) => {
    let a = 0, b = 0;
    for (let j = 0; j < v.length; j++) { if (v[j] > v[a]) a = j; if (zRef[i][j] > zRef[i][b]) b = j; const d = Math.abs(v[j] - zRef[i][j]); mad += d; mx = Math.max(mx, d); cnt++; }
    agree += a === b;
  });
  return { argmaxAgree: agree / z.length, meanAbsDlogit: mad / cnt, maxAbsDlogit: mx };
};
for (const mode of ["aligned", "fresh127", "fresh31"]) {
  const t0 = performance.now();
  const { bytes, st } = mergedBytes(mode);
  const mergeMs = performance.now() - t0;
  const e = await mkSpike(bytes);
  dev.queue.writeBuffer(e.loraW, 0, new Float32Array(e.loraWords));   // LoRA path contributes zero: the delta is in the base
  const z = await scoreAll(e, test);
  out.merge[mode] = { ...acc(z), ...cmp(z), mergeMs, ...st };
  e.destroy();
  ctx.log(`merge ${mode}: acc ${out.merge[mode].accuracy.toFixed(4)} vs ${out.merge.unmerged.accuracy.toFixed(4)}, agree ${out.merge[mode].argmaxAgree.toFixed(4)}, maxΔ ${out.merge[mode].maxAbsDlogit.toFixed(4)}`);
}
// control: the unmerged engine with its LoRA path zeroed (how much the spike's delta matters at all)
{
  const e = await mkSpike(wb);
  dev.queue.writeBuffer(e.loraW, 0, new Float32Array(e.loraWords));
  const z = await scoreAll(e, test);
  out.merge.noDelta = { ...acc(z), ...cmp(z) };
  e.destroy();
}
eS.destroy();
}
ctx.progress(0.55, "Trainer");

// ---------- T: Trainer ----------
const roleT = (o, t) => {
  if (o.name === "mm") {
    if (o.bufs[1] === t.WB) return "base_mm";
    if (o.bufs[2] === t.PART) return "lora_grad";
    if (o.bufs[1] === t.PRM && (o.vals.bOff ?? 0) < t.headOff && o.bufs[2] !== t.GRD) return "lora_mm";
    return "head";
  }
  if (o.name === "sumz") return "lora_grad";
  if (["sqsum", "norm", "adamw"].includes(o.name)) return "optimizer";
  if (o.name.startsWith("attn")) return "attn";
  if (o.name.startsWith("head") || ["bias_gelu", "loss", "colsum", "gather", "scatter"].includes(o.name)) return "head";
  return o.name;   // ln, ln_bwd, ln_pgrad, rope, rope_bwd, geglu, geglu_bwd, gelu_bwd, copy, embed
};
const SLOT = 256;
function prepOps(t, ops) {
  if (t.ubo.size !== SLOT * 2048) throw new Error("trainer uniform layout changed");
  let slots = 0;
  for (const o of ops) {
    if (o.clear) continue;
    o.slot = slots++;
    o.k.u.forEach(([nm, ty], i) => { const v = o.vals[nm] ?? 0; if (ty === "f") t.uview.setFloat32(o.slot * SLOT + 4 * i, v, true); else t.uview.setUint32(o.slot * SLOT + 4 * i, v, true); });
    const key = o.name + "|" + o.bufs.map((b) => b._id).join(",");
    let bg = t.bgCache.get(key);
    if (!bg) {
      bg = dev.createBindGroup({ layout: o.k.bgl, entries: [{ binding: 0, resource: { buffer: t.ubo, offset: 0, size: SLOT } }, ...o.bufs.map((b, i) => ({ binding: i + 1, resource: { buffer: b } }))] });
      t.bgCache.set(key, bg);
    }
    o.bg = bg;
  }
  dev.queue.writeBuffer(t.ubo, 0, t.udata, 0, slots * SLOT);
}
const disp = (p, o) => { p.setPipeline(o.k.pipe); p.setBindGroup(0, o.bg, [o.slot * SLOT]); p.dispatchWorkgroups(...o.groups); };
async function wallOps(ops, reps = 3, trials = 5) {
  if (smoke) { reps = 1; trials = 1; }   // the graph with its clears, as flush() records it
  const W = [];
  for (let tr = 0; tr <= trials; tr++) {
    const enc = dev.createCommandEncoder();
    for (let r = 0; r < reps; r++) {
      let pass = null;
      for (const o of ops) {
        if (o.clear) { if (pass) { pass.end(); pass = null; } enc.clearBuffer(o.clear); continue; }
        if (!pass) pass = enc.beginComputePass();
        disp(pass, o);
      }
      if (pass) pass.end();
    }
    const t0 = performance.now();
    dev.queue.submit([enc.finish()]);
    await dev.queue.onSubmittedWorkDone();
    if (tr > 0) W.push((performance.now() - t0) / reps);
  }
  return med(W);
}
const BS = smoke ? 2 : 16;
const tb = await Bk.WgslTrainerBackend.create(dev, manifest, wb, { Trainer, pad: tok.pad, maxSeqs: BS, maxRows: BS * maxLen });
await tb.loadSpike(sm, sbytes);
const t = tb.t;
ctx.log("T: trainer ready");
const tbatch = train.slice(0, BS);
t.loadBatch(tb.collate(tbatch, tbatch.map((x) => x.target)));
out.T = { batch: { seqs: BS, S: t.b.S, rows: t.b.R, realTokens: toks(tbatch) } };
// real step once so buffers hold real values
t.forwardBackward({ T: 1, slot: 0 }); t.adamStep({ lr: 5e-4, step: 1 }); await t.flush();
for (const phase of ["forward", "step"]) {
  if (phase === "forward") t.forward();
  else { t.forwardBackward({ T: 1, slot: 0 }); t.adamStep({ lr: 5e-4, step: 2 }); }
  const ops = t.ops; t.ops = [];
  prepOps(t, ops);
  const lossAt = ops.findIndex((o) => o.name === "loss");
  ops.forEach((o, i) => { if (!o.clear) { o.role = roleT(o, t); o.dir = lossAt < 0 || i < lossAt ? "fwd" : "bwd"; } });
  const P = { dispatches: ops.filter((o) => !o.clear).length, clears: ops.filter((o) => o.clear).length, byRole: {} };
  for (const o of ops) if (!o.clear) P.byRole[o.role] = (P.byRole[o.role] ?? 0) + 1;
  P.e2e_wall_ms = await wallOps(ops);
  P.drop = {};
  for (const role of Object.keys(P.byRole).slice(0, smoke ? 1 : 99)) P.drop[role] = await wallOps(ops.filter((o) => o.clear || o.role !== role));
  P.iso = [];
  const real = ops.filter((o) => !o.clear);
  for (const [i, o] of real.entries()) {
    if (smoke && i > 3) break;
    const r = await isoUs((p) => disp(p, o));
    P.iso.push({ i, name: o.name, role: o.role, dir: o.dir, groups: o.groups, us: +r.us.toFixed(2), k: r.k });
    if (i % 40 === 0) ctx.progress(0.55 + (phase === "forward" ? 0.1 : 0.25) * (i / real.length) + (phase === "step" ? 0.12 : 0), `Trainer ${phase} iso ${i}/${real.length}`);
  }
  P.isoSum_ms = P.iso.reduce((a, x) => a + x.us, 0) / 1000;
  P.isoByRole_ms = {};
  for (const x of P.iso) { const key = `${x.dir}:${x.role}`; P.isoByRole_ms[key] = (P.isoByRole_ms[key] ?? 0) + x.us / 1000; }
  out.T[phase] = P;
  ctx.log(`T ${phase}: ${P.dispatches} dispatches, e2e ${P.e2e_wall_ms.toFixed(2)} ms, isoSum ${P.isoSum_ms.toFixed(2)} ms`);
}
ctx.progress(1, "done");
return out;
