// Per-kernel parity for the training kernels in ../train/ (kernel library step 3).
//
// Each case runs a forward kernel and its backward kernel(s) through SpecRunner on small random inputs.
// The forward output is compared with a float64 reference in this file, and each gradient with central
// finite differences of that reference: d/dx sum(R * f(x)) for a random R fed in as the upstream
// gradient. A case passes when every check has max |gpu - ref| <= tol * max |ref|. The finite
// differences run in float64 on the f32 inputs, so their own error (about 1e-10) is far below tol.
//
//   const { trainCases, runTrainOps } = await import(".../_lib/wgsl/tests/train_ops.js");
//   const res = await runTrainOps(device, { only: "attn" });   // [{ name, op, pass, rel, checks }]
//
// Run headless with check_ops.mjs (it runs these after ops.js); on a real GPU through gpu-bridge/jobs/wgsl-ops.

import { SpecRunner } from "../train/spec.js";
import { fullKernels } from "../train/kernels.js";
import { recordNce } from "../train/nce.js";
import { rng, randn, gelu } from "./ops.js";

const C = { H: 64, NH: 2, D: 32, I: 48, OUT: 96 };
const { H, NH, D, I, OUT } = C;

const f64 = (a) => Float64Array.from(a);
const f32 = (a) => Float32Array.from(a);
const u32 = (a) => Uint32Array.from(a);
const cat = (...as) => f32(as.flatMap((a) => Array.from(a)));

// Gradient of sum(R * f(xs)) with respect to xs[name], by central differences in float64.
function fd(f, xs, name, R, h = 1e-5) {
  const x = xs[name], g = new Float64Array(x.length);
  const L = () => { const y = f(xs); let s = 0; for (let i = 0; i < y.length; i++) s += R[i] * y[i]; return s; };
  for (let k = 0; k < x.length; k++) {
    const v = x[k];
    x[k] = v + h; const a = L();
    x[k] = v - h; const b = L();
    x[k] = v;
    g[k] = (a - b) / (2 * h);
  }
  return g;
}

function lnRef(X, R, w, b, eps) {
  const Y = new Float64Array(R * H);
  for (let r = 0; r < R; r++) {
    let mu = 0, va = 0;
    for (let j = 0; j < H; j++) mu += X[r * H + j];
    mu /= H;
    for (let j = 0; j < H; j++) { const d = X[r * H + j] - mu; va += d * d; }
    const rs = 1 / Math.sqrt(va / H + eps);
    for (let j = 0; j < H; j++) Y[r * H + j] = (X[r * H + j] - mu) * rs * w[j] + (b ? b[j] : 0);
  }
  return Y;
}

function colSum(P, Z, n) {
  const s = new Float64Array(n);
  for (let z = 0; z < Z; z++) for (let c = 0; c < n; c++) s[c] += P[z * n + c];
  return s;
}

// ---- cases: run(rn, r) records and reads back, returns [[label, gpu, ref], ...] ----------------------
const lnCase = {
  name: "ln fwd, bwd, weight grads", op: "ln_bwd",
  async run(rn, r) {
    const R = 5, eps = 1e-5, chunk = 2, Z = Math.ceil(R / chunk);
    const X = randn(r, R * H), w = randn(r, H, 0.2).map((v) => 1 + v), b = randn(r, H, 0.1), DY = randn(r, R * H), base = randn(r, R * H);
    const bX = rn.buf("X", R * H, X), bW = rn.buf("W", 2 * H, cat(w, b)), bY = rn.buf("Y", R * H), bST = rn.buf("ST", 2 * R);
    const bDY = rn.buf("DY", R * H, DY), bDX = rn.buf("DX", R * H), bDX2 = rn.buf("DX2", R * H, base), bG = rn.buf("G", 2 * H), bP = rn.buf("PART", Z * H);
    rn.op("ln", [bX, bW, bY, bST], { R, eps, wOff: 0, bOff: H, hasB: 1 }, R);
    rn.op("ln_bwd", [bX, bST, bW, bDY, bDX], { R, wOff: 0, acc: 0 }, R);
    rn.op("ln_bwd", [bX, bST, bW, bDY, bDX2], { R, wOff: 0, acc: 1 }, R);
    rn.op("ln_pgrad", [bX, bST, bDY, bG], { R, gwOff: 0, gbOff: H }, 1);
    rn.op("ln_wpart", [bX, bST, bDY, bP], { R, chunk }, Z);
    const xs = { X: f64(X), w: f64(w), b: f64(b) }, f = (s) => lnRef(s.X, R, s.w, s.b, eps);
    const dX = fd(f, xs, "X", DY), dW = fd(f, xs, "w", DY), G = await rn.read(bG);
    return [["Y", await rn.read(bY), f(xs)], ["dX", await rn.read(bDX), dX], ["dX acc", await rn.read(bDX2), dX.map((v, i) => v + base[i])],
      ["dW", G.slice(0, H), dW], ["dB", G.slice(H), fd(f, xs, "b", DY)], ["dW ln_wpart", colSum(await rn.read(bP), Z, H), dW]];
  },
};

const ropeCase = {
  name: "rope fwd, bwd", op: "rope_bwd",
  async run(rn, r) {
    const S = 4, R = 2 * S, half = D / 2, sinOff = S * half;
    const QKV = randn(r, R * 3 * H), th = Array.from({ length: S * half }, () => 6 * r());
    const CS = cat(th.map(Math.cos), th.map(Math.sin)), DQR = randn(r, R * 2 * H);
    const bQKV = rn.buf("QKV", R * 3 * H, QKV), bCS = rn.buf("CS", 2 * S * half, CS), bQR = rn.buf("QR", R * 2 * H);
    const bDQR = rn.buf("DQR", R * 2 * H, DQR), bDQKV = rn.buf("DQKV", R * 3 * H);
    rn.op("rope", [bQKV, bCS, bQR], { R, S, sinOff }, R);
    rn.op("rope_bwd", [bDQR, bCS, bDQKV], { R, S, sinOff }, R);
    const f = (s) => {
      const Y = new Float64Array(R * 2 * H);
      for (let row = 0; row < R; row++) for (let sel = 0; sel < 2; sel++) for (let h = 0; h < NH; h++) for (let d = 0; d < half; d++) {
        const pos = row % S, c = CS[pos * half + d], sn = CS[sinOff + pos * half + d];
        const bi = row * 3 * H + sel * H + h * D, bo = row * 2 * H + sel * H + h * D, x1 = s.QKV[bi + d], x2 = s.QKV[bi + d + half];
        Y[bo + d] = x1 * c - x2 * sn;
        Y[bo + d + half] = x2 * c + x1 * sn;
      }
      return Y;
    };
    const xs = { QKV: f64(QKV) };
    return [["QR", await rn.read(bQR), f(xs)], ["dQKV", await rn.read(bDQKV), fd(f, xs, "QKV", DQR)]];
  },
};

const attnCase = (isLocal, win) => ({
  name: `attn fwd, bwd ${isLocal ? `window ${win}` : "global"}`, op: "attn_bkv",
  async run(rn, r) {
    const S = 6, LENS = [6, 4], B = LENS.length, R = B * S, scale = 1 / Math.sqrt(D);
    const QR = randn(r, R * 2 * H), QKV = randn(r, R * 3 * H), DO = randn(r, R * H);
    const bQR = rn.buf("QR", R * 2 * H, QR), bQKV = rn.buf("QKV", R * 3 * H, QKV), bL = rn.buf("LENS", B, u32(LENS));
    const bO = rn.buf("O", R * H), bLSE = rn.buf("LSE", R * NH), bDO = rn.buf("DO", R * H, DO), bDD = rn.buf("DD", R * NH);
    const bDQR = rn.buf("DQR", R * 2 * H), bDQKV = rn.buf("DQKV", R * 3 * H), u = { R, S, isLocal, win };
    rn.op("attn", [bQR, bQKV, bL, bO, bLSE], u, R * NH);
    rn.op("attn_bq", [bQR, bQKV, bL, bLSE, bDO, bO, bDQR, bDD], u, R * NH);
    rn.op("attn_bkv", [bQR, bQKV, bL, bLSE, bDO, bDD, bDQR, bDQKV], u, R * NH);
    const f = (s) => {
      const O = new Float64Array(R * H);
      for (let bq = 0; bq < B; bq++) for (let i = 0; i < LENS[bq]; i++) for (let h = 0; h < NH; h++) {
        const len = LENS[bq], lo = isLocal ? Math.max(0, i - win) : 0, hi = isLocal ? Math.min(len, i + win + 1) : len, row = bq * S + i;
        const sc = [];
        for (let j = lo; j < hi; j++) {
          let d = 0;
          for (let k = 0; k < D; k++) d += s.QR[row * 2 * H + h * D + k] * s.QR[(bq * S + j) * 2 * H + H + h * D + k];
          sc.push(d * scale);
        }
        const mx = Math.max(...sc), e = sc.map((v) => Math.exp(v - mx)), l = e.reduce((a, b) => a + b, 0);
        for (let k = 0; k < D; k++) {
          let o = 0;
          for (let j = lo; j < hi; j++) o += e[j - lo] * s.QKV[(bq * S + j) * 3 * H + 2 * H + h * D + k];
          O[row * H + h * D + k] = o / l;
        }
      }
      return O;
    };
    const xs = { QR: f64(QR), QKV: f64(QKV) };
    return [["O", await rn.read(bO), f(xs)], ["dQR", await rn.read(bDQR), fd(f, xs, "QR", DO)], ["dQKV", await rn.read(bDQKV), fd(f, xs, "QKV", DO)]];
  },
});

const gegluCase = {
  name: "geglu fwd, bwd", op: "geglu_bwd",
  async run(rn, r) {
    const R = 3, n = R * I, U = randn(r, R * 2 * I), DG = randn(r, n);
    const bU = rn.buf("U", R * 2 * I, U), bG = rn.buf("G", n), bDG = rn.buf("DG", n, DG), bDU = rn.buf("DU", R * 2 * I);
    rn.op("geglu", [bU, bG], { n }, Math.ceil(n / 256));
    rn.op("geglu_bwd", [bU, bDG, bDU], { n }, Math.ceil(n / 256));
    const f = (s) => Float64Array.from({ length: n }, (_, e) => { const row = Math.floor(e / I), c = e % I; return gelu(s.U[row * 2 * I + c]) * s.U[row * 2 * I + I + c]; });
    const xs = { U: f64(U) };
    return [["G", await rn.read(bG), f(xs)], ["dU", await rn.read(bDU), fd(f, xs, "U", DG)]];
  },
};

const biasGeluCase = {
  name: "bias_gelu fwd, gelu_bwd, colsum", op: "gelu_bwd",
  async run(rn, r) {
    const R = 3, n = R * H, bOff = 7, z = randn(r, n), bias = randn(r, H, 0.5), DA = randn(r, n);
    const bPRM = rn.buf("PRM", bOff + H, cat(randn(r, bOff), bias)), bZ = rn.buf("Z", n, z), bA = rn.buf("A", n);
    const bDA = rn.buf("DA", n, DA), bDZ = rn.buf("DZ", n), bG = rn.buf("G", H);
    rn.op("bias_gelu", [bPRM, bZ, bA], { R, bOff }, R);
    rn.op("gelu_bwd", [bZ, bDA, bDZ], { n }, Math.ceil(n / 256));
    rn.op("colsum", [bDZ, bG], { R, gOff: 0 }, 1);
    const f = (s) => Float64Array.from({ length: n }, (_, e) => gelu(s.z[e] + s.b[e % H]));
    const xs = { z: f64(z), b: f64(bias) };
    return [["A", await rn.read(bA), f(xs)], ["dZ", await rn.read(bDZ), fd(f, xs, "z", DA)], ["dB colsum", await rn.read(bG), fd(f, xs, "b", DA)]];
  },
};

const headOutCase = {
  name: "head_out fwd, bwd", op: "head_out_bwd",
  async run(rn, r) {
    const R = 4, X = randn(r, R * H), w = randn(r, H, 0.2), b = f32([0.3]), DL = randn(r, R);
    const bX = rn.buf("X", R * H, X), bPRM = rn.buf("PRM", H + 1, cat(w, b)), bY = rn.buf("Y", R), bDL = rn.buf("DL", R, DL);
    const bDX = rn.buf("DX", R * H), bG = rn.buf("G", H + 1);
    rn.op("head_out", [bX, bPRM, bY], { R, wOff: 0, bOff: H }, R);
    rn.op("head_out_bwd", [bDL, bX, bPRM, bDX, bG], { R, wOff: 0, gwOff: 0, gbOff: H }, 1);
    const f = (s) => Float64Array.from({ length: R }, (_, row) => { let a = s.b[0]; for (let j = 0; j < H; j++) a += s.X[row * H + j] * s.w[j]; return a; });
    const xs = { X: f64(X), w: f64(w), b: f64(b) }, G = await rn.read(bG);
    return [["Y", await rn.read(bY), f(xs)], ["dX", await rn.read(bDX), fd(f, xs, "X", DL)], ["dW", G.slice(0, H), fd(f, xs, "w", DL)], ["dB", G.slice(H), fd(f, xs, "b", DL)]];
  },
};

const lossCase = {
  name: "segment softmax cross-entropy", op: "loss",
  async run(rn, r) {
    const SEG = [3, 4, 2], T = 0.7, n = 9, Z = randn(r, n, 2), GOLD = f32([1, 0, 0, 0.5, 0.5, 0, 0, 0, 1]);
    const bZ = rn.buf("Z", n, Z), bSEG = rn.buf("SEG", 3, u32(SEG)), bGOLD = rn.buf("GOLD", n, GOLD), bDZ = rn.buf("DZ", n), bL = rn.buf("LOSS", 2);
    rn.op("loss", [bZ, bSEG, bGOLD, bDZ, bL], { nSeg: SEG.length, T, slot: 1 }, 1);
    const f = (s) => {
      let o = 0, tot = 0;
      for (const k of SEG) {
        const v = Array.from(s.Z.slice(o, o + k), (x) => x / T), mx = Math.max(...v);
        const lse = mx + Math.log(v.reduce((a, x) => a + Math.exp(x - mx), 0));
        for (let j = 0; j < k; j++) tot -= GOLD[o + j] * (v[j] - lse);
        o += k;
      }
      return [tot / SEG.length];
    };
    const xs = { Z: f64(Z) };
    return [["loss", (await rn.read(bL)).slice(1), f(xs)], ["dZ", await rn.read(bDZ), fd(f, xs, "Z", [1])]];
  },
};

const gatherCase = {
  name: "gather fwd, scatter bwd", op: "scatter",
  async run(rn, r) {
    const N = 6, POS = [4, 1, 5], R = POS.length, X = randn(r, N * H), DY = randn(r, R * H);
    const bPOS = rn.buf("POS", R, u32(POS)), bX = rn.buf("X", N * H, X), bY = rn.buf("Y", R * H), bDY = rn.buf("DY", R * H, DY), bDX = rn.buf("DX", N * H);
    rn.op("gather", [bPOS, bX, bY], { R }, R);
    rn.op("scatter", [bPOS, bDY, bDX], { R }, R);
    const f = (s) => Float64Array.from({ length: R * H }, (_, e) => s.X[POS[Math.floor(e / H)] * H + (e % H)]);
    const xs = { X: f64(X) };
    return [["Y", await rn.read(bY), f(xs)], ["dX", await rn.read(bDX), fd(f, xs, "X", DY)]];
  },
};

const poolCase = {
  name: "mean pool fwd, unpool bwd", op: "unpool",
  async run(rn, r) {
    const S = 5, LENS = [5, 3], B = LENS.length, R = B * S, X = randn(r, R * H), DPL = randn(r, B * H);
    const bX = rn.buf("X", R * H, X), bL = rn.buf("LENS", B, u32(LENS)), bPL = rn.buf("PL", B * H), bDPL = rn.buf("DPL", B * H, DPL), bDX = rn.buf("DX", R * H);
    rn.op("pool", [bX, bL, bPL], { B, S }, B);
    rn.op("unpool", [bDPL, bL, bDX], { R, S }, R);
    const f = (s) => Float64Array.from({ length: B * H }, (_, e) => {
      const b = Math.floor(e / H), c = e % H;
      let a = 0;
      for (let t = 0; t < LENS[b]; t++) a += s.X[(b * S + t) * H + c];
      return a / LENS[b];
    });
    const xs = { X: f64(X) };
    return [["PL", await rn.read(bPL), f(xs)], ["dX", await rn.read(bDX), fd(f, xs, "X", DPL)]];
  },
};

const cosCase = {
  name: "cosine loss", op: "cosloss",
  async run(rn, r) {
    const B = 3, Y = randn(r, B * OUT), TGT = randn(r, B * OUT);
    const bY = rn.buf("Y", B * OUT, Y), bT = rn.buf("TGT", B * OUT, TGT), bDY = rn.buf("DY", B * OUT), bL = rn.buf("LOSS", 1);
    rn.op("cosloss", [bY, bT, bDY, bL], { B, slot: 0 }, 1);
    const f = (s) => {
      let tot = 0;
      for (let b = 0; b < B; b++) {
        let yt = 0, yy = 0, tt = 0;
        for (let j = 0; j < OUT; j++) { const y = s.Y[b * OUT + j], g = TGT[b * OUT + j]; yt += y * g; yy += y * y; tt += g * g; }
        tot += 1 - yt / Math.sqrt(yy * tt);
      }
      return [tot / B];
    };
    const xs = { Y: f64(Y) };
    return [["loss", await rn.read(bL), f(xs)], ["dY", await rn.read(bDY), fd(f, xs, "Y", [1])]];
  },
};

const embedCase = {
  name: "embed_f fwd, token-table and LN-weight grads", op: "tgrad",
  async run(rn, r) {
    const V = 7, IDS = [3, 1, 3, 6, 1, 3], R = IDS.length, eps = 1e-5;
    const UTOK = [1, 3, 6], USTART = [0, 2, 5, 6], UROWS = [1, 4, 0, 2, 5, 3];
    const w = randn(r, H, 0.2).map((v) => 1 + v), table = randn(r, V * H), DY = randn(r, R * H);
    const bIDS = rn.buf("IDS", R, u32(IDS)), bPRM = rn.buf("PRM", H + V * H, cat(w, table)), bXE = rn.buf("XE", R * H), bX = rn.buf("X", R * H), bST = rn.buf("ST", 2 * R);
    const bDY = rn.buf("DY", R * H, DY), bDXE = rn.buf("DXE", R * H), bG = rn.buf("G", H + V * H), bP = rn.buf("PART", H);
    const bUT = rn.buf("UTOK", 3, u32(UTOK)), bUS = rn.buf("USTART", 4, u32(USTART)), bUR = rn.buf("UROWS", R, u32(UROWS));
    rn.op("embed_f", [bIDS, bPRM, bXE, bX, bST], { R, eps, tOff: H, wOff: 0 }, R);
    rn.op("ln_bwd", [bXE, bST, bPRM, bDY, bDXE], { R, wOff: 0, acc: 0 }, R);
    rn.op("tgrad", [bUT, bUS, bUR, bDXE, bG], { nU: UTOK.length, tOff: H }, UTOK.length);
    rn.op("ln_wpart", [bXE, bST, bDY, bP], { R, chunk: R }, 1);
    const f = (s) => lnRef(Float64Array.from({ length: R * H }, (_, e) => s.table[IDS[Math.floor(e / H)] * H + (e % H)]), R, s.w, null, eps);
    const xs = { w: f64(w), table: f64(table) };
    return [["X", await rn.read(bX), f(xs)], ["dTable", (await rn.read(bG)).slice(H), fd(f, xs, "table", DY)], ["dW", await rn.read(bP), fd(f, xs, "w", DY)]];
  },
};

const nceCase = {
  name: "InfoNCE loss through recordNce", op: "nce",
  async run(rn, r) {
    const P = 3, B = 2 * P, scale = 20, Y = randn(r, B * OUT);
    Object.assign(rn, { C, Y: rn.buf("Y", B * OUT, Y), YN: rn.buf("YN", B * OUT), NR: rn.buf("NR", B), SIM: rn.buf("SIM", P * P), DS: rn.buf("DS", P * P),
      LROW: rn.buf("LROW", P), DYN: rn.buf("DYN", B * OUT), DY: rn.buf("DY", B * OUT), LOSS: rn.buf("LOSS", 1) });
    recordNce(rn, { P, scale, slot: 0 });
    const f = (s) => {
      const yn = [];
      for (let b = 0; b < B; b++) { const v = s.Y.slice(b * OUT, (b + 1) * OUT), n = Math.hypot(...v); yn.push(v.map((x) => x / n)); }
      let tot = 0;
      for (let i = 0; i < P; i++) {
        const sim = Array.from({ length: P }, (_, j) => scale * yn[i].reduce((a, x, k) => a + x * yn[P + j][k], 0)), mx = Math.max(...sim);
        tot += mx + Math.log(sim.reduce((a, x) => a + Math.exp(x - mx), 0)) - sim[i];
      }
      return [tot / P];
    };
    const xs = { Y: f64(Y) };
    return [["loss", await rn.read(rn.LOSS), f(xs)], ["dY", await rn.read(rn.DY), fd(f, xs, "Y", [1])]];
  },
};

const mmCase = {
  name: "mm split-K, sumz, strided operands", op: "mm",
  async run(rn, r) {
    const M = 37, N = 70, K = 90, kChunk = 32, Z = Math.ceil(K / kChunk);
    const A = randn(r, K * M), Bm = randn(r, K * N), base = randn(r, M * N);   // A stored [K, M], read transposed
    const bA = rn.buf("A", K * M, A), bB = rn.buf("B", K * N, Bm), bP = rn.buf("PART", Z * M * N), bC = rn.buf("C", M * N, base), bCT = rn.buf("CT", M * N);
    const s = { M, N, K, aOff: 0, aRS: 1, aCS: M, bOff: 0, bRS: N, bCS: 1, cOff: 0, acc: 0 }, g = [Math.ceil(N / 64), Math.ceil(M / 64)];
    rn.op("mm", [bA, bB, bP], { ...s, kChunk, cRS: N, cCS: 1, cZS: M * N, alpha: 0.5 }, [...g, Z]);
    rn.op("sumz", [bP, bC], { M, N, Z, cOff: 0, cRS: N, cCS: 1, acc: 1, alpha: 1 }, Math.ceil((M * N) / 256));
    rn.op("mm", [bA, bB, bCT], { ...s, kChunk: K, cRS: 1, cCS: M, cZS: 0, alpha: 1 }, [...g, 1]);
    const AB = new Float64Array(M * N);
    for (let m = 0; m < M; m++) for (let n = 0; n < N; n++) { let a = 0; for (let k = 0; k < K; k++) a += A[k * M + m] * Bm[k * N + n]; AB[m * N + n] = a; }
    const CT = new Float64Array(M * N);
    for (let m = 0; m < M; m++) for (let n = 0; n < N; n++) CT[n * M + m] = AB[m * N + n];
    return [["C += 0.5 AB", await rn.read(bC), AB.map((v, i) => base[i] + 0.5 * v)], ["(AB)^T", await rn.read(bCT), CT]];
  },
};

const adamwCase = {
  name: "grad norm, clip, AdamW step", op: "adamw",
  async run(rn, r) {
    const n = 300, o = { n, headOff: 200, lr: 0.01, lrHead: 0.05, wd: 0.01, b1: 0.9, b2: 0.999, eps: 1e-8, bc1: 1 - 0.9 ** 3, bc2s: Math.sqrt(1 - 0.999 ** 3), clip: 1 };
    const P0 = randn(r, n), G = randn(r, n), M0 = randn(r, n, 0.1), V0 = randn(r, n, 0.1).map((v) => v * v);
    const bP = rn.buf("P", n, P0), bG = rn.buf("G", n, G), bM = rn.buf("M", n, M0), bV = rn.buf("V", n, V0), bPART = rn.buf("PART", 2), bN = rn.buf("NRM", 1);
    rn.op("sqsum", [bG, bPART], { n }, 2);
    rn.op("norm", [bPART, bN], { n: 2 }, 1);
    rn.op("adamw", [bP, bG, bM, bV, bN], o, Math.ceil(n / 256));
    const nrm = Math.sqrt(G.reduce((a, g) => a + g * g, 0)), cf = Math.min(1, o.clip / (nrm + 1e-6));
    const rM = new Float64Array(n), rV = new Float64Array(n), rD = new Float64Array(n);
    for (let e = 0; e < n; e++) {
      const g = G[e] * cf, lr = e >= o.headOff ? o.lrHead : o.lr, w = P0[e] * (1 - lr * o.wd);
      rM[e] = o.b1 * M0[e] + (1 - o.b1) * g;
      rV[e] = o.b2 * V0[e] + (1 - o.b2) * g * g;
      rD[e] = w - (lr / o.bc1) * rM[e] / (Math.sqrt(rV[e]) / o.bc2s + o.eps) - P0[e];
    }
    const P1 = await rn.read(bP);
    return [["norm", await rn.read(bN), [nrm]], ["M", await rn.read(bM), rM], ["V", await rn.read(bV), rV], ["P step", P1.map((v, i) => v - P0[i]), rD]];
  },
};

const embedQ4Case = {
  name: "embed 4-bit table + LayerNorm, copy", op: "embed",
  async run(rn, r) {
    const V = 5, IDS = [2, 0, 4, 2], R = IDS.length, eps = 1e-5, wOff = 3;
    const codes = Array.from({ length: V * H }, () => Math.floor(16 * r())), scales = randn(r, (V * H) / 32, 0.1).map(Math.abs);
    const packed = new Uint32Array((V * H) / 8);
    codes.forEach((q, e) => { packed[e >> 3] |= q << ((e % 8) * 4); });
    const W = cat(randn(r, wOff), randn(r, H, 0.2).map((v) => 1 + v)), base = randn(r, R * H);
    const bIDS = rn.buf("IDS", R, u32(IDS)), bC = rn.buf("CODES", packed.length, packed), bS = rn.buf("SCALES", scales.length, scales);
    const bW = rn.buf("W", W.length, W), bX = rn.buf("X", R * H), bY = rn.buf("Y", R * H), bY2 = rn.buf("Y2", R * H, base);
    rn.op("embed", [bIDS, bC, bS, bW, bX], { R, eps, wOff }, R);
    rn.op("copy", [bX, bY], { n: R * H, acc: 0 }, Math.ceil((R * H) / 256));
    rn.op("copy", [bX, bY2], { n: R * H, acc: 1 }, Math.ceil((R * H) / 256));
    const rows = Float64Array.from({ length: R * H }, (_, e) => {
      const tok = IDS[Math.floor(e / H)], j = e % H, k = tok * H + j;
      return (codes[k] - 8) * scales[tok * (H / 32) + Math.floor(j / 32)];
    });
    const X = lnRef(rows, R, W.slice(wOff), null, eps);
    return [["X", await rn.read(bX), X], ["copy", await rn.read(bY), X], ["copy acc", await rn.read(bY2), X.map((v, i) => v + base[i])]];
  },
};

export const trainCases = [embedQ4Case, lnCase, ropeCase, attnCase(0, 0), attnCase(1, 2), gegluCase, biasGeluCase, headOutCase, lossCase,
  gatherCase, poolCase, cosCase, embedCase, nceCase, mmCase, adamwCase];

// opts: only (substring of case name or op), seed, tol (default 1e-4). Rows: { name, op, pass, rel, checks: [{ label, rel, n }] }.
export async function runTrainOps(device, { only = "", seed = 1, tol = 1e-4, log = () => {} } = {}) {
  const sel = trainCases.filter((c) => !only || `train ${c.name}`.includes(only) || c.op === only);
  if (!sel.length) return [];
  const rn = new SpecRunner(device);
  await rn.compile(fullKernels(C));
  const res = [];
  for (const [i, c] of sel.entries()) {
    const r = rng(seed + 104729 + i * 7919);
    let row = { name: `train ${c.name}`, op: c.op };
    try {
      const checks = (await c.run(rn, r)).map(([label, got, ref]) => {
        if (got.length !== ref.length) throw new Error(`${label}: ${got.length} values, reference ${ref.length}`);
        let maxErr = 0, maxRef = 0;
        for (let k = 0; k < ref.length; k++) { maxErr = Math.max(maxErr, Math.abs(got[k] - ref[k])); maxRef = Math.max(maxRef, Math.abs(ref[k])); }
        return { label, rel: maxErr / Math.max(maxRef, 1e-30), n: ref.length };
      });
      const rel = Math.max(...checks.map((x) => x.rel));
      row = { ...row, pass: checks.every((x) => Number.isFinite(x.rel) && x.rel <= tol), rel, checks };
    } catch (err) {
      row = { ...row, pass: false, error: String(err.message || err) };
    }
    const worst = row.checks ? row.checks.reduce((a, b) => (b.rel > a.rel ? b : a)) : null;
    log(`${row.pass ? "ok  " : "FAIL"} ${row.name} rel ${row.rel?.toExponential(2)}${worst ? ` (worst ${worst.label})` : ""}${row.error ? " " + row.error : ""}`);
    res.push(row);
  }
  return res;
}
