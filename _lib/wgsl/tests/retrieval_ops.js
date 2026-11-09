// Parity for the retrieval ops in ../retrieval.js: kNN (queries against docs, and the self-excluding
// graph form), multi-pass top-k, prefix scan and radix sort, against exact JS references. kNN and
// top-k compare indices exactly and scores at tol relative; scan and sort compare every value exactly.
//
//   const { runRetrievalOps } = await import(".../_lib/wgsl/tests/retrieval_ops.js");
//   const res = await runRetrievalOps(device, { only: "sort" });   // [{ name, op, pass, rel, ... }]
//
// check_ops.mjs runs these after ops.js and train_ops.js; gpu-bridge/jobs/wgsl-ops runs them on a real GPU.

import { compile, knnWGSL, TOPK, planTopk, scanPipelines, planScan, sortPipelines, planSort, record } from "../retrieval.js";
import { rng } from "./ops.js";

const U = () => GPUBufferUsage;
function upload(dev, a) {
  const b = dev.createBuffer({ size: Math.max(16, a.byteLength), usage: U().STORAGE | U().COPY_SRC | U().COPY_DST });
  dev.queue.writeBuffer(b, 0, a);
  return b;
}
async function read(dev, buf, n, Type = Float32Array) {
  const rb = dev.createBuffer({ size: Math.max(16, n * 4), usage: U().MAP_READ | U().COPY_DST });
  const enc = dev.createCommandEncoder();
  enc.copyBufferToBuffer(buf, 0, rb, 0, n * 4);
  dev.queue.submit([enc.finish()]);
  await rb.mapAsync(GPUMapMode.READ);
  const x = new Type(rb.getMappedRange(0, n * 4).slice(0));
  rb.destroy();
  return x;
}
const unif = (dev, words) => { const u = dev.createBuffer({ size: 16, usage: U().UNIFORM | U().COPY_DST }); dev.queue.writeBuffer(u, 0, Uint32Array.from(words)); return u; };

// Exact top-k of one row: higher score first, lower index on ties (the kernels' order).
function topkRef(row, k, skip = -1) {
  const ix = [...row.keys()].filter((i) => i !== skip);
  ix.sort((a, b) => row[b] - row[a] || a - b);
  return ix.slice(0, k);
}
// float32 dot products the way the kernel accumulates them is not reproducible from JS, so scores are
// compared at a relative tolerance and indices exactly; inputs are random, so near-ties are rare, and a
// swapped pair whose scores agree to 1e-6 counts as a match.
function compareTopk(gotIdx, gotScore, rows, k, refScore, tol) {
  let bad = 0, maxErr = 0, maxRef = 0;
  rows.forEach((ref, q) => ref.forEach((i, j) => {
    const g = gotIdx[q * k + j], rs = refScore(q, i), gs = gotScore[q * k + j];
    maxErr = Math.max(maxErr, Math.abs(gs - rs)); maxRef = Math.max(maxRef, Math.abs(rs));
    if (g !== i && Math.abs(refScore(q, g) - rs) > 1e-6 * Math.max(1, Math.abs(rs))) bad++;
  }));
  const rel = maxErr / Math.max(maxRef, 1e-30);
  return { pass: bad === 0 && rel <= tol, rel, mismatches: bad };
}

const knnCase = (selfGraph) => ({
  name: selfGraph ? "knn graph excludeSelf D64" : "knn D256 f32", op: "knn",
  async run(dev, r, tol) {
    const D = selfGraph ? 64 : 256, K = 10, QB = 4, n = selfGraph ? 512 : 3000, Q = selfGraph ? n : 12;
    const docs = Float32Array.from({ length: n * D }, () => r() - 0.5), qs = selfGraph ? docs : Float32Array.from({ length: Q * D }, () => r() - 0.5);
    const [pipe] = await compile(dev, knnWGSL({ D, K, QB, excludeSelf: selfGraph }));
    const bD = upload(dev, docs), bQ = selfGraph ? bD : upload(dev, qs), bI = upload(dev, new Uint32Array(Q * K)), bS = upload(dev, new Float32Array(Q * K));
    const BATCH = selfGraph ? 128 : Q, dispatches = [];
    for (let q0 = 0; q0 < Q; q0 += BATCH)
      dispatches.push([pipe, dev.createBindGroup({ layout: pipe.getBindGroupLayout(0), entries: [bD, bQ, bI, bS, unif(dev, [n, q0, 0, 0])].map((buffer, binding) => ({ binding, resource: { buffer } })) }), BATCH / QB]);
    const enc = dev.createCommandEncoder(); record(enc, dispatches); dev.queue.submit([enc.finish()]);
    const idx = await read(dev, bI, Q * K, Uint32Array), score = await read(dev, bS, Q * K);
    const dot = (q, i) => { let a = 0; for (let d = 0; d < D; d++) a += qs[q * D + d] * docs[i * D + d]; return a; };
    const rows = Array.from({ length: Q }, (_, q) => topkRef(Float64Array.from({ length: n }, (_, i) => dot(q, i)), K, selfGraph ? q : -1));
    return compareTopk(idx, score, rows, K, dot, tol);
  },
});

const topkCase = {
  name: "topk 3 rows x 70000, k 50 (3 passes)", op: "topk",
  async run(dev, r, tol) {
    const nq = 3, n = 70000, stride = 70016, k = 50;
    const s = Float32Array.from({ length: nq * stride }, () => r() * 4 - 2);
    for (let q = 0; q < nq; q++) s[q * stride + 777] = s[q * stride + 4242];   // one exact tie per row
    const [pipe] = await compile(dev, TOPK);
    const plan = planTopk(dev, pipe, upload(dev, s), { nq, n, inStride: stride, k });
    const enc = dev.createCommandEncoder(); record(enc, plan.dispatches); dev.queue.submit([enc.finish()]);
    const idx = await read(dev, plan.idx, nq * k, Uint32Array), score = await read(dev, plan.scores, nq * k);
    const rows = Array.from({ length: nq }, (_, q) => topkRef(s.subarray(q * stride, q * stride + n), k));
    const res = compareTopk(idx, score, rows, k, (q, i) => s[q * stride + i], tol);
    return { ...res, passes: plan.dispatches.length };
  },
};

const scanCase = {
  name: "exclusive scan n 300001 (3 levels)", op: "scan",
  async run(dev, r) {
    const n = 300001, data = Uint32Array.from({ length: n }, () => Math.floor(r() * 16));
    const b = upload(dev, data), plan = planScan(dev, await scanPipelines(dev), b, n);
    const enc = dev.createCommandEncoder(); record(enc, plan.dispatches); dev.queue.submit([enc.finish()]);
    const out = await read(dev, b, n, Uint32Array);
    let acc = 0, bad = 0;
    for (let i = 0; i < n; i++) { if (out[i] !== acc) bad++; acc = (acc + data[i]) >>> 0; }
    return { pass: bad === 0, rel: bad / n, mismatches: bad, levels: plan.levels.length };
  },
};

const sortCase = {
  name: "radix sort n 100003", op: "sort",
  async run(dev, r) {
    const n = 100003, keys = Uint32Array.from({ length: n }, () => Math.floor(r() * 4294967296));
    const A = upload(dev, keys), B = upload(dev, new Uint32Array(n));
    const plan = planSort(dev, await sortPipelines(dev), A, B, n);
    const enc = dev.createCommandEncoder(); record(enc, plan.dispatches); dev.queue.submit([enc.finish()]);
    const out = await read(dev, A, n, Uint32Array), ref = keys.slice().sort();
    let bad = 0;
    for (let i = 0; i < n; i++) if (out[i] !== ref[i]) bad++;
    return { pass: bad === 0, rel: bad / n, mismatches: bad };
  },
};

export const retrievalCases = [knnCase(false), knnCase(true), topkCase, scanCase, sortCase];

export async function runRetrievalOps(device, { only = "", seed = 1, tol = 1e-4, log = () => {} } = {}) {
  const res = [];
  for (const [i, c] of retrievalCases.entries()) {
    const name = `retrieval ${c.name}`;
    if (only && !name.includes(only) && c.op !== only) continue;
    let row = { name, op: c.op };
    try { row = { ...row, ...(await c.run(device, rng(seed + 31337 + i * 7919), tol)) }; }
    catch (err) { row = { ...row, pass: false, error: String(err.message || err) }; }
    log(`${row.pass ? "ok  " : "FAIL"} ${row.name} rel ${row.rel?.toExponential(2)}${row.mismatches ? ` mismatches ${row.mismatches}` : ""}${row.error ? " " + row.error : ""}`);
    res.push(row);
  }
  return res;
}
