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

fs.writeFileSync(path.join(here, "..", "results", "check_js.json"), JSON.stringify(results, null, 1));
process.exit(fail ? 1 : 0);
