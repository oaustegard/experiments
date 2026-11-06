// Pure-JS checks for the creator (no GPU): head gradient vs central differences, fp16 bit patterns,
// pack/unpack against a real adapter-library spike (byte-identical repack), the exact library merge,
// and rank truncation (singular values written to out/truncate.json for run/check_truncate.py).
//   node run/check_js.mjs
import fs from "node:fs";
import path from "node:path";
globalThis.GPUBufferUsage = {}; globalThis.GPUMapMode = {};   // engine.js reads these at import time
const here = path.dirname(new URL(import.meta.url).pathname), web = path.join(here, "..", "web");
const C = await import(path.join(web, "creator.js"));
const Bk = await import(path.join(web, "backends.js"));
const results = {};
let fail = 0;
const check = (name, ok, info) => { results[name] = { ok, ...info }; if (!ok) fail++; console.log(ok ? "PASS" : "FAIL", name, JSON.stringify(info)); };

// 1. head gradient check, float64 params
{
  const H = 24, r = C.rng(3), rnd = (n, s = 1) => Float64Array.from({ length: n }, () => (r() * 2 - 1) * s);
  const p = { w0: rnd(H * H, 0.3), b0: rnd(H, 0.1), lnw: Float64Array.from({ length: H }, () => 1 + 0.2 * (r() - 0.5)), lnb: rnd(H, 0.1), w1: rnd(H, 0.5), b1: rnd(1, 0.1) };
  const rows = [rnd(3 * H), rnd(4 * H), rnd(2 * H)];
  const targets = [Float32Array.of(0.7, 0.2, 0.1), Float32Array.of(0, 0, 1, 0), Float32Array.of(0.5, 0.5)];
  const g = Object.fromEntries(Object.entries(p).map(([k, v]) => [k, new Float64Array(v.length)]));
  Bk.headLossGrad(p, rows, targets, H, 1e-5, g);
  let worst = 0, n = 0;
  for (const k of Object.keys(p)) {
    for (let t = 0; t < Math.min(12, p[k].length); t++) {
      const i = Math.floor(r() * p[k].length), h = 1e-5, x = p[k][i];
      p[k][i] = x + h; const lp = Bk.headLossGrad(p, rows, targets, H, 1e-5);
      p[k][i] = x - h; const lm = Bk.headLossGrad(p, rows, targets, H, 1e-5);
      p[k][i] = x;
      const fd = (lp - lm) / (2 * h), rel = Math.abs(fd - g[k][i]) / Math.max(1e-4, Math.abs(fd) + Math.abs(g[k][i]));   // floor: lnb and b1 shift every logit equally, so their true gradient is 0
      worst = Math.max(worst, rel); n++;
    }
  }
  check("head_gradcheck", worst < 1e-4, { params_checked: n, worst_rel_err: worst });
}

// 2. fp16 bit patterns (IEEE round-to-nearest-even)
{
  const cases = [[1, 0x3c00], [-2, 0xc000], [65504, 0x7bff], [1e-8, 0], [6e-8, 1], [0.1, 0x2e66], [1 / 3, 0x3555], [70000, 0x7c00], [2 ** -14, 0x0400], [-0.000123, 0x8808]];
  const bad = cases.filter(([x, h]) => C.toF16(x) !== h).map(([x, h]) => [x, h.toString(16), C.toF16(x).toString(16)]);
  check("fp16_bits", bad.length === 0, { bad });
}

// 3. unpack + repack a real spike: identical bytes and offsets
const spikeDir = path.join(web, "..", "..", "adapter-library", "web", "models");
const load = (set, id) => ({ sm: JSON.parse(fs.readFileSync(path.join(spikeDir, set, "spikes", id + ".json"))), bytes: new Uint8Array(fs.readFileSync(path.join(spikeDir, set, "spikes", id + ".bin"))) });
{
  const s = load("everyday", "emails"), u = C.unpackSpike(s.sm, s.bytes);
  const { sm, bytes } = C.packSpike(u, {});
  let same = bytes.length === s.bytes.length;
  for (let i = 0; same && i < bytes.length; i++) if (bytes[i] !== s.bytes[i]) same = false;
  const offs = JSON.stringify(sm.layers) === JSON.stringify(s.sm.layers) && JSON.stringify({ ...sm.head, b1: 0 }) === JSON.stringify({ ...s.sm.head, b1: 0 });
  check("repack_identical", same && offs, { bytes: bytes.length, sameBytes: same, sameOffsets: offs });
}

// 4. the stacked-rank merge equals the mean of the deltas
const ws = ["agent_trace_observability", "customer_service", "security_incidents"].map((id) => C.unpackSpike(load("workflows", id).sm, load("workflows", id).bytes));
const merged = C.mergeSpikes(ws);
{
  const li = 2, k = "Wi", m = merged.layers[li][k], { K, N } = m;
  let worst = 0;
  for (let t = 0; t < 200; t++) {
    const n = (t * 37) % N, c = (t * 53) % K;
    let a = 0; for (let j = 0; j < m.r; j++) a += m.B[n * m.r + j] * m.A[j * K + c];
    let b = 0; for (const w of ws) { const q = w.layers[li][k]; let d = 0; for (let j = 0; j < q.r; j++) d += q.B[n * q.r + j] * q.A[j * K + c]; b += d / ws.length; }
    worst = Math.max(worst, Math.abs(a - b));
  }
  check("merge_exact", worst < 1e-6 && m.r === 24, { rank: m.r, worst_abs_err: worst });
}

// 5. rank-8 truncation of the merge: singular values and residual for run/check_truncate.py
{
  const t0 = Date.now(), tr = Bk.truncateRank({ layers: [merged.layers[2]], head: merged.head }, 8);
  const ms = Date.now() - t0, m = merged.layers[2].Wi, q = tr.layers[0].Wi, { K, N } = m;
  const dense = (x) => { const D = new Float64Array(N * K); for (let n = 0; n < N; n++) for (let j = 0; j < x.r; j++) { const b = x.B[n * x.r + j]; for (let c = 0; c < K; c++) D[n * K + c] += b * x.A[j * K + c]; } return D; };
  const Dm = dense(m), Dq = dense(q);
  let n2 = 0, e2 = 0; for (let i = 0; i < Dm.length; i++) { n2 += Dm[i] ** 2; e2 += (Dm[i] - Dq[i]) ** 2; }
  const sig = Array.from({ length: 8 }, (_, j) => { let a = 0, b = 0; for (let c = 0; c < K; c++) a += q.A[j * K + c] ** 2; for (let n = 0; n < N; n++) b += q.B[n * 8 + j] ** 2; return Math.sqrt(a * b); });
  fs.mkdirSync(path.join(here, "..", "out"), { recursive: true });
  fs.writeFileSync(path.join(here, "..", "out", "truncate.json"), JSON.stringify({ layer: 2, module: "Wi", sig, norm2: n2, err2: e2, ms }));
  check("truncate_ran", sig.every((x, i) => i === 0 || x <= sig[i - 1] * (1 + 1e-6)), { sig: sig.map((x) => +x.toFixed(5)), err2: e2, norm2: n2, ms });
}

// 6. pause and resume: a run stopped at any step and resumed from its (structured-cloned) checkpoint on a
// fresh backend ends exactly where the uninterrupted run ends: same step losses, best epoch and weights.
{
  const { FakeBackend: Fake } = await import(path.join(web, "fake_backend.js"));
  const r = C.rng(5), mk = (n, tag) => Array.from({ length: n }, (_, i) => {
    const P = 2 + (i % 3), ids = Array.from({ length: 12 }, () => Math.floor(r() * 500)), t = new Float32Array(P); t[Math.floor(r() * P)] = 1;
    return { key: `${tag}${i}`, q: `q${i % 2}`, type: "choice", ids, pos: Array.from({ length: P }, (_, j) => j), target: t };
  });
  const data = { train: mk(37, "t"), val: mk(9, "v"), test: mk(11, "s") }, opts = { epochs: 4, lr: 0.05, bs: 8, seed: 3 };
  const runLog = async (stopAt, every = 0) => {
    const losses = [], cps = [];
    let b = new Fake(), seen = 0;
    let res = await C.train(b, data, opts, (e) => { if (e.kind === "step") { losses.push(e.loss); seen = e.step; } }, () => stopAt !== null && seen === stopAt,
      { onCheckpoint: every ? (cp) => cps.push(structuredClone(cp)) : null, checkpointSteps: every });
    let paused = !!res.paused;
    if (res.paused || (every && cps.length)) {
      const cp = res.paused ? structuredClone(res.checkpoint) : cps[2];
      if (!res.paused) losses.length = cp.step;   // a crash after the third checkpoint: drop the steps it lost
      b = new Fake();
      res = await C.train(b, data, opts, (e) => { if (e.kind === "step") losses.push(e.loss); }, () => false, { resume: cp });
    }
    return { res, losses, paused, w: res.spike.sm.w };
  };
  const A = await runLog(null);
  const cases = { mid_epoch: await runLog(7), epoch_end: await runLog(5), first_step: await runLog(1), from_periodic_checkpoint: await runLog(null, 3) };
  const same = (x) => x.losses.length === A.losses.length && x.losses.every((l, i) => l === A.losses[i]) && x.w.every((v, i) => v === A.w[i]) &&
    x.res.bestEpoch === A.res.bestEpoch && x.res.test.accuracy === A.res.test.accuracy;
  const detail = Object.fromEntries(Object.entries(cases).map(([k, x]) => [k, { paused: x.paused, identical: same(x) }]));
  check("pause_resume_exact", Object.values(cases).every(same) && cases.mid_epoch.paused && cases.epoch_end.paused, { steps: A.losses.length, bestEpoch: A.res.bestEpoch, cases: detail });
}

// 6b. train more: a finished run extended by M epochs continues its weights, optimizer, shuffles and epoch
// numbers with a fresh schedule over the added steps; a pause inside the extension resumes exactly; only a
// finished run can be extended.
{
  const { FakeBackend: Fake } = await import(path.join(web, "fake_backend.js"));
  const r = C.rng(9), mk = (n, tag) => Array.from({ length: n }, (_, i) => {
    const P = 2 + (i % 3), ids = Array.from({ length: 12 }, () => Math.floor(r() * 500)), t = new Float32Array(P); t[Math.floor(r() * P)] = 1;
    return { key: `${tag}${i}`, q: `q${i % 2}`, type: "choice", ids, pos: Array.from({ length: P }, (_, j) => j), target: t };
  });
  const data = { train: mk(37, "t"), val: mk(9, "v"), test: mk(11, "s") }, opts = { epochs: 2, lr: 0.05, bs: 8, seed: 3 }, more = { ...opts, epochs: 3 };
  const lrs = [];
  class Spy extends Fake { async step(sq, tg, o) { lrs.push(o.lr); return super.step(sq, tg, o); } }
  const b = new Spy();
  const first = await C.train(b, data, opts);
  const n1 = lrs.length, ext = await C.train(b, data, more, () => {}, () => false, { extend: first.final });
  const spe = Math.ceil(37 / 8), lr2 = lrs.slice(n1);
  // the same extension, paused at its 4th step and resumed on a fresh backend
  const b2 = new Fake(); const f2 = await C.train(b2, data, opts);
  let seen = 0;
  const p = await C.train(b2, data, more, (e) => { if (e.kind === "step") seen = e.step; }, () => seen === f2.final.step + 4, { extend: f2.final });
  const b3 = new Fake(), ext2 = await C.train(b3, data, more, () => {}, () => false, { resume: structuredClone(p.checkpoint) });
  let refused = false;
  try { await C.train(new Fake(), data, more, () => {}, () => false, { extend: p.checkpoint }); } catch { refused = true; }
  const ok = ext.epochsRun === 5 && b.t === 5 * spe && lr2.length === 3 * spe && lr2[0] > 5 * lrs[n1 - 1] && lr2[lr2.length - 1] < lr2[0] / 5 && p.paused && ext2.spike.sm.w.every((v, i) => v === ext.spike.sm.w[i]) && ext2.bestEpoch === ext.bestEpoch && refused &&
    ext.bestEpoch >= first.bestEpoch;
  check("train_more", ok, { epochsRun: ext.epochsRun, adamSteps: b.t, bestEpoch: [first.bestEpoch, ext.bestEpoch], lrFirstPhaseEnd: lrs[n1 - 1], lrExtension: [lr2[0], Math.max(...lr2), lr2[lr2.length - 1]],
    pausedInExtension: p.paused, resumedIdentical: ext2.spike.sm.w.every((v, i) => v === ext.spike.sm.w[i]), extendUnfinishedRefused: refused });
}

// 7. the agent prompt's example output passes the page's own validator, with and without a draft decision
{
  const { agentPrompt } = await import(path.join(web, "agent_prompt.js"));
  const p = agentPrompt(), a = p.indexOf("nothing else:\n") + "nothing else:\n".length, b = p.indexOf("\nLeave \"split\"");
  let ex = null, err = null;
  try { ex = JSON.parse(p.slice(a, b)); } catch (e) { err = String(e); }
  const errs = ex ? C.validate(ex.decision, C.normalizeCases(ex.cases)) : ["unparsed"];
  const withDraft = agentPrompt({ id: "x", questions: { q: { type: "noul", instructions: "It is." } } }).includes('"instructions": "It is."');
  const bools = C.normalizeCases([{ state: "s", gold: { u: true, n: 2 } }])[0];
  check("agent_prompt_example", !err && errs.length === 0 && withDraft && bools.gold.u === "true" && bools.gold.n === "2" && bools.id === "c1",
    { parseError: err, validateErrors: errs, cases: ex?.cases.length, draftIncluded: withDraft, chars: p.length });
}

// 8. synthetic cases only train; without them the split is unchanged
{
  const mkc = (n, src) => Array.from({ length: n }, (_, i) => ({ id: `${src}${String(i).padStart(3, "0")}`, state: "s", gold: {}, ...(src === "y" ? { source: "synthetic" } : {}) }));
  const real = mkc(40, "r"), syn = mkc(15, "y");
  syn[0].split = "test";
  const a = C.splitCases(real, { seed: 1 }), b = C.splitCases([...real, ...syn], { seed: 1 });
  const ids = (x) => x.map((c) => c.id).join();
  const ok = ids(a.val) === ids(b.val) && ids(a.test) === ids(b.test) && b.train.length === a.train.length + 15 &&
    ![...b.val, ...b.test].some((c) => c.source === "synthetic");
  const w = C.datasetWarnings({ questions: {} }, [...mkc(3, "r"), ...syn]);
  check("synthetic_train_only", ok && w.length === 2, { train: [a.train.length, b.train.length], warnings: w });
}

fs.writeFileSync(path.join(here, "..", "results", "check_js.json"), JSON.stringify(results, null, 1));
process.exit(fail ? 1 : 0);
