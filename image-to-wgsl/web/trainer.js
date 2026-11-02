// In-browser encoder for image-to-wgsl: fits ordered soft-ellipse splats to an image with
// TensorFlow.js autodiff, then writes the same WGSL shader as codec.py's emit_wgsl (packed u32).
// No DOM: the gallery page and a GPU-bridge worker both import it and pass in `tf`.
//
//   const r = await encode(tf, {rgba, width, height}, {n: 256, onProgress});
//   r.wgsl, r.points[{n, psnr}], r.P (Int32Array n*10), r.bg (Int32Array 12)

// Bit layout: must match codec.py Layout (x, y, sx, sy, th, r, g, b, alpha, shape).
export const LAYOUT = { bits: [10, 9, 6, 6, 6, 6, 6, 6, 4, 2], bbg: 6, sLo: -9, sHi: 0 };
const B = LAYOUT.bits, MAX = B.map((b) => 2 ** b - 1), MBG = 2 ** LAYOUT.bbg - 1;
const SPAN = (LAYOUT.sHi - LAYOUT.sLo) / MAX[2], LN2 = Math.LN2;
// Adam step per field, as a fraction of the field's range (codec.py STEP_FRAC).
const STEP_FRAC = [0.004, 0.004, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.02, 0.03];

function rng(seed) {             // mulberry32
  let a = seed >>> 0;
  return () => { a = (a + 0x6D2B79F5) >>> 0; let t = a; t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61); return ((t ^ (t >>> 14)) >>> 0) / 4294967296; };
}

export function checkpoints(n, lo = 16) {
  const ks = [];
  for (let k = lo; k < n; k *= 2) ks.push(k);
  ks.push(n);
  return ks;
}

function makeOps(tf, A, maxChunk = 256) {
  const M = tf.tensor1d(MAX);
  const steRound = tf.customGrad((x) => ({ value: tf.round(x), gradFunc: (dy) => dy }));

  function physical(P, quant) {
    let Q = tf.minimum(tf.maximum(P, 0), M);
    if (quant) Q = steRound(Q);
    const [qx, qy, qsx, qsy, qth, qr, qg, qb, qal, qm] = tf.unstack(Q, 1);
    return {
      x: qx.mul(A / MAX[0]), y: qy.mul(1 / MAX[1]),
      sx: tf.exp(qsx.mul(SPAN).add(LAYOUT.sLo).mul(LN2)), sy: tf.exp(qsy.mul(SPAN).add(LAYOUT.sLo).mul(LN2)),
      th: qth.mul(Math.PI / 2 ** B[4]), col: tf.stack([qr, qg, qb], 1).div(MAX[5]),
      al: qal.add(1).div(2 ** B[8]), m: tf.exp(qm.mul(LN2)),
    };
  }

  function bgColors(bgp, quant) {
    let b = tf.clipByValue(bgp, 0, MBG);
    if (quant) b = steRound(b);
    return b.div(MBG).reshape([4, 3]);
  }

  function background(bg, X, Y) {
    const h = X.div(A).expandDims(1), v = Y.expandDims(1);
    const [c0, c1, c2, c3] = tf.unstack(bg, 0);
    const top = c0.mul(tf.sub(1, h)).add(c1.mul(h)), bot = c2.mul(tf.sub(1, h)).add(c3.mul(h));
    return top.mul(tf.sub(1, v)).add(bot.mul(v));
  }

  // Splat opacity w[g, p]: mirrors the WGSL decoder in emitWGSL.
  function weights(s, i0, i1, X, Y) {
    const sl = (t) => t.slice([i0], [i1 - i0]).expandDims(1);
    const dx = X.expandDims(0).sub(sl(s.x)), dy = Y.expandDims(0).sub(sl(s.y));
    const th = sl(s.th), c = tf.cos(th), sn = tf.sin(th);
    const a = dx.mul(c).add(dy.mul(sn)).div(sl(s.sx));
    const b = dy.mul(c).sub(dx.mul(sn)).div(sl(s.sy));
    const e = tf.pow(tf.clipByValue(a.mul(a).add(b.mul(b)), 0, 40).add(1e-12), sl(s.m));
    return sl(s.al).mul(tf.exp(e.mul(-0.5)));
  }

  // Paint splats [i0, i1) over c in order; closed form in log space so exponents stay <= 0.
  // Sums over the splat axis are matrix products with triangular ones matrices, not tf.cumsum, whose
  // gradient tfjs 4.22 gets wrong on WebGPU from row length 32 and transposes for axis 0. The WebGPU
  // backend also gets [P,G]x[G,3] matMul wrong from G = 64, so train on the WebGL backend
  // (GPU bridge, iPhone, 2026-11-02: results/bridge_i2w-gradcheck-2.json).
  const tri = new Map();
  function strictUpper(G) {          // S[l, j] = 1 when l > j, so (logom @ S)[p, j] = sum over l > j
    if (!tri.has(G)) {
      const a = new Float32Array(G * G);
      for (let l = 0; l < G; l++) for (let j = 0; j < l; j++) a[l * G + j] = 1;
      tri.set(G, tf.keep(tf.tensor2d(a, [G, G])));
    }
    return tri.get(G);
  }
  function composite(c, w, col) {
    const wt = w.transpose();                                      // [P, G]
    const lom = tf.log(tf.sub(1, tf.minimum(wt, 0.9999)));         // log(1 - w)
    const after = tf.exp(tf.matMul(lom, strictUpper(wt.shape[1]))); // prod_{l>j} (1 - w_l)
    const total = tf.exp(tf.sum(lom, 1, true));                    // prod over the chunk, [P, 1]
    return c.mul(total).add(tf.matMul(wt.mul(after), col));
  }

  function render(P, bgp, X, Y, quant, ks, chunk = maxChunk) {
    const s = physical(P, quant);
    let c = background(bgColors(bgp, quant), X, Y);
    const n = P.shape[0], edges = [...new Set([...ks.filter((k) => k > 0 && k < n), n])].sort((p, q) => p - q);
    const snaps = new Map();
    let start = 0;
    for (const e of edges) {
      for (let i = start; i < e; i += chunk) {
        const j = Math.min(e, i + chunk);
        c = composite(c, weights(s, i, j, X, Y), s.col.slice([i], [j - i]));
      }
      snaps.set(e, c);
      start = e;
    }
    return { c, snaps };
  }

  return { M, physical, render, dispose: () => { for (const t of tri.values()) t.dispose(); tri.clear(); } };
}

function grid(W, H, A) {
  const X = new Float32Array(W * H), Y = new Float32Array(W * H);
  for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) { X[y * W + x] = (x + 0.5) / W * A; Y[y * W + x] = (y + 0.5) / H; }
  return { X, Y };
}

// Render the image (quantized) on the CPU side of tf at W x H; returns Float32Array RGB in [0,1].
export async function renderImage(tf, ops, Pq, bgq, W, H, A, n = null, pixChunk = 16384) {
  const { X, Y } = grid(W, H, A);
  const out = new Float32Array(W * H * 3);
  const Pn = n == null ? Pq : Pq.slice([0, 0], [n, 10]);
  for (let i = 0; i < W * H; i += pixChunk) {
    const j = Math.min(W * H, i + pixChunk);
    const part = tf.tidy(() =>
      ops.render(Pn, bgq, tf.tensor1d(X.subarray(i, j)), tf.tensor1d(Y.subarray(i, j)), true, []).c.clipByValue(0, 1));
    out.set(await part.data(), i * 3);
    part.dispose();
  }
  if (Pn !== Pq) Pn.dispose();
  return out;
}

export function psnr8(a, b) {        // a, b: float RGB in [0,1], same length
  let s = 0;
  for (let i = 0; i < a.length; i++) { const d = Math.round(a[i] * 255) - Math.round(b[i] * 255); s += d * d; }
  return 10 * Math.log10(255 * 255 / Math.max(s / a.length, 1e-12));
}

export async function encode(tf, img, opts = {}) {
  const { n = 256, stepsStage = 100, stepsFinal = 300, stepsQat = 150, pixBudget = 2 ** 21, seed = 0,
          onProgress = null, onPreview = null, previewEvery = 25, previewWidth = 160,
          chunk = tf.getBackend() === "webgpu" ? 32 : 256 } = opts;   // WebGPU is correct only up to 32 (see composite)
  const W = img.width, H = img.height, A = W / H, HW = W * H;
  const rand = rng(seed);
  const ops = makeOps(tf, A, chunk);
  const tgt = new Float32Array(HW * 3);
  for (let i = 0; i < HW; i++) for (let c = 0; c < 3; c++) tgt[i * 3 + c] = img.rgba[i * 4 + c] / 255;
  const target = tf.tensor2d(tgt, [HW, 3]);
  const g = grid(W, H, A), Xall = tf.tensor1d(g.X), Yall = tf.tensor1d(g.Y);
  // small target for placing new splats
  const sw = Math.max(16, Math.round(Math.min(W, 128))), sh = Math.max(8, Math.round(sw / A));
  const tsmall = tf.tidy(() => tf.image.resizeBilinear(target.reshape([H, W, 3]), [sh, sw], true).reshape([sh * sw, 3]));
  const tsmallArr = await tsmall.data();
  // background: quadrant means
  const bg0 = new Float32Array(12);
  for (let q = 0; q < 4; q++) {
    const x0 = q % 2 ? W >> 1 : 0, x1 = q % 2 ? W : W >> 1, y0 = q > 1 ? H >> 1 : 0, y1 = q > 1 ? H : H >> 1;
    let r = 0, gg = 0, bb = 0, cnt = 0;
    for (let y = y0; y < y1; y++) for (let x = x0; x < x1; x++) { const i = (y * W + x) * 3; r += tgt[i]; gg += tgt[i + 1]; bb += tgt[i + 2]; cnt++; }
    bg0.set([r / cnt * MBG, gg / cnt * MBG, bb / cnt * MBG], q * 3);
  }
  let P = tf.zeros([0, 10]), bgp = tf.tensor1d(bg0);
  const ks = checkpoints(n);
  const sizes = ks.slice();
  const scale = tf.tensor1d(MAX.map((m, i) => m * STEP_FRAC[i])).expandDims(0), bscale = MBG * 0.01;
  const totalSteps = sizes.reduce((a, s, i) => a + (i === sizes.length - 1 ? stepsFinal : stepsStage), 0) + stepsQat;
  let done = 0, lastLoss = null;
  const t0 = performance.now();
  const smallGrid = grid(sw, sh, A);

  async function optimise(steps, lr0, lr1, quant, phase) {
    const base = tf.variable(P.div(scale)), bbase = tf.variable(bgp.div(bscale));
    const opt = tf.train.adam(lr0, 0.9, 0.99);
    const nNow = P.shape[0], kNow = checkpoints(nNow).filter((k) => k <= nNow);
    // sampled pixels x splats bounds the autodiff tape; iOS kills a tab past a few hundred MB
    const nPix = opts.nPix ?? Math.min(8192, Math.max(1024, Math.floor(pixBudget / nNow)));
    for (let it = 0; it < steps; it++) {
      const f = it / Math.max(1, steps - 1);
      opt.learningRate = lr1 + 0.5 * (lr0 - lr1) * (1 + Math.cos(Math.PI * f));
      const cost = opt.minimize(() => {
        const sel = tf.randomUniform([nPix], 0, HW, "int32", seed * 100003 + done);
        const Xs = tf.gather(Xall, sel), Ys = tf.gather(Yall, sel), T = tf.gather(target, sel);
        const { snaps } = ops.render(base.mul(scale), bbase.mul(bscale), Xs, Ys, quant, kNow);
        const ls = kNow.map((k) => tf.log(tf.mean(tf.squaredDifference(snaps.get(k), T))));
        return tf.mean(tf.stack(ls));
      }, true, [base, bbase]);
      tf.tidy(() => { base.assign(tf.minimum(tf.maximum(base.mul(scale), 0), ops.M).div(scale)); });
      done++;
      if (it % 10 === 0 || it === steps - 1) {
        lastLoss = (await cost.data())[0];
        onProgress?.({ phase, n: nNow, step: done, totalSteps, psnrSample: -10 * lastLoss / Math.LN10, seconds: (performance.now() - t0) / 1000 });
      }
      cost.dispose();
      if (onPreview && (it % previewEvery === 0 || it === steps - 1)) {
        const Pq = tf.tidy(() => tf.round(tf.minimum(tf.maximum(base.mul(scale), 0), ops.M)));
        const bq = tf.tidy(() => tf.round(tf.clipByValue(bbase.mul(bscale), 0, MBG)));
        const pw = previewWidth, ph = Math.max(1, Math.round(pw / A));
        onPreview({ rgb: await renderImage(tf, ops, Pq, bq, pw, ph, A), width: pw, height: ph });
        Pq.dispose(); bq.dispose();
      }
      if (it % 5 === 4) await new Promise((r) => setTimeout(r, 0));
    }
    P.dispose(); bgp.dispose();
    P = tf.tidy(() => base.mul(scale)); bgp = tf.tidy(() => bbase.mul(bscale));
    base.dispose(); bbase.dispose(); opt.dispose();
  }

  for (let si = 0; si < sizes.length; si++) {
    const size = sizes[si], nNew = size - P.shape[0];
    // error map at small size -> new splats where the error is largest
    const Pc = tf.tidy(() => tf.minimum(tf.maximum(P, 0), ops.M));
    const cur = await renderImage(tf, ops, Pc, bgp, sw, sh, A);
    Pc.dispose();
    const keys = [];
    for (let i = 0; i < sw * sh; i++) {
      let e = 1e-8;
      for (let c = 0; c < 3; c++) { const d = cur[i * 3 + c] - tsmallArr[i * 3 + c]; e += d * d; }
      keys.push([Math.log(rand() + 1e-300) / e, i]);      // Efraimidis-Spirakis: weight e
    }
    keys.sort((p, q) => q[0] - p[0]);
    const sigma = 0.6 * Math.sqrt(A / size), rows = [];
    for (let k = 0; k < nNew; k++) {
      const i = keys[k][1], ix = i % sw, iy = Math.floor(i / sw);
      const ls = Math.log2(sigma) + (rand() * 0.6 - 0.3);
      const qs = (ls - LAYOUT.sLo) / (LAYOUT.sHi - LAYOUT.sLo) * MAX[2];
      rows.push([(ix + 0.5) / sw * MAX[0], (iy + 0.5) / sh * MAX[1], qs, qs + rand() * 4 - 2, rand() * 2 ** B[4],
        tsmallArr[i * 3] * MAX[5], tsmallArr[i * 3 + 1] * MAX[5], tsmallArr[i * 3 + 2] * MAX[5], 0.75 * 2 ** B[8] - 1, 0]
        .map((v, f) => Math.min(Math.max(v, 0), MAX[f])));
    }
    const Pnew = tf.concat([P, tf.tensor2d(rows, [nNew, 10])], 0);
    P.dispose(); P = Pnew;
    const final = si === sizes.length - 1;
    await optimise(final ? stepsFinal : stepsStage, 1.0, final ? 0.03 : 0.1, false, `${size} splats`);
  }
  await optimise(stepsQat, 0.3, 0.02, true, "rounding to bits");

  const Pq = tf.tidy(() => tf.round(tf.minimum(tf.maximum(P, 0), ops.M)));
  const bq = tf.tidy(() => tf.round(tf.clipByValue(bgp, 0, MBG)));
  const points = [];
  for (const k of ks) points.push({ n: k, psnr: psnr8(await renderImage(tf, ops, Pq, bq, W, H, A, k), tgt) });
  const result = { P: Int32Array.from(await Pq.data()), bg: Int32Array.from(await bq.data()), A, W, H, n, points,
                   seconds: (performance.now() - t0) / 1000, backend: tf.getBackend() };
  [P, bgp, Pq, bq, target, Xall, Yall, tsmall, scale, ops.M].forEach((t) => t.dispose());
  ops.dispose();
  return result;
}

// ---------------------------------------------------------------- WGSL emission (mirrors codec.py)

function fnum(v) {
  let s = String(parseFloat(Math.fround(v).toPrecision(9)));
  if (!/[.e]/.test(s)) s += ".";
  return s;
}

export function packBits(fields) {      // [[value, bits]] -> u32 words + one zero pad word
  const total = fields.reduce((a, f) => a + f[1], 0);
  const words = new Uint32Array(Math.ceil(total / 32) + 1);
  let off = 0;
  for (const [v, b] of fields) {
    const w = off >>> 5, s = off & 31;
    words[w] = (words[w] | (v << s)) >>> 0;
    if (s + b > 32) words[w + 1] = (words[w + 1] | (v >>> (32 - s))) >>> 0;
    off += b;
  }
  return words;
}

// One self-contained shader: uniform vec4f(width, height, time, 0); time <= 0 paints everything,
// time t > 0 paints the first N * min(1, (t / 6)^2) splats.
export function emitWGSL(r, n = null, paintSeconds = 6) {
  const N = n ?? r.n, A = r.A;
  const fields = [];
  for (const v of r.bg) fields.push([v, LAYOUT.bbg]);
  for (let i = 0; i < N; i++) for (let f = 0; f < 10; f++) fields.push([r.P[i * 10 + f], B[f]]);
  const words = packBits(fields);
  if (words.length > 2047) throw new Error(`${words.length} words: over WebKit's 2047-element const array limit`);
  const off = [0]; for (const b of B) off.push(off[off.length - 1] + b);
  const fld = (j) => `r(o+${off[j]}u,${B[j]}u)`;
  const bgv = (j) => `vec3f(r(${3 * j * LAYOUT.bbg}u,6u),r(${(3 * j + 1) * LAYOUT.bbg}u,6u),r(${(3 * j + 2) * LAYOUT.bbg}u,6u))`;
  const sig = (j) => `exp2(${fnum(LAYOUT.sLo)}+${fld(j)}*${fnum(SPAN)})`;
  return `@group(0)@binding(0)var<uniform>u:vec4f;const A=${fnum(A)};` +
    `const D=array<u32,${words.length}>(${Array.from(words).join(",")});` +
    `fn r(o:u32,n:u32)->f32{let w=o>>5u;let s=o&31u;var v=D[w]>>s;if(s+n>32u){v|=D[w+1u]<<(32u-s);}return f32(v&((1u<<n)-1u));}` +
    `@vertex fn vs(@builtin(vertex_index)i:u32)->@builtin(position)vec4f{return vec4f(vec2f(f32(i%2u),f32(i/2u))*4.-1.,0.,1.);}` +
    `@fragment fn fs(@builtin(position)p:vec4f)->@location(0)vec4f{let k=min(u.x/A,u.y);let q=(p.xy-(u.xy-vec2f(A,1.)*k)*.5)/k;` +
    `if(q.x<0.||q.y<0.||q.x>A||q.y>1.){return vec4f(1.);}let h=q.x/A;` +
    `var c=mix(mix(${bgv(0)},${bgv(1)},h),mix(${bgv(2)},${bgv(3)},h),q.y)/${fnum(MBG)};` +
    `var n=${N}u;if(u.z>0.){n=min(n,u32(f32(n)*min(1.,u.z*u.z/${fnum(paintSeconds ** 2)})));}` +
    `for(var i=0u;i<n;i++){let o=${12 * LAYOUT.bbg}u+i*${B.reduce((a, b) => a + b, 0)}u;` +
    `let d=q-vec2f(${fld(0)}*${fnum(A / MAX[0])},${fld(1)}*${fnum(1 / MAX[1])});let t=${fld(4)}*${fnum(Math.PI / 2 ** B[4])};` +
    `let cs=cos(t);let sn=sin(t);let a=(d.x*cs+d.y*sn)/${sig(2)};let b=(d.y*cs-d.x*sn)/${sig(3)};` +
    `var e=min(a*a+b*b,40.);for(var j=0u;j<u32(${fld(9)});j++){e*=e;}` +
    `let w=(${fld(8)}+1.)*${fnum(1 / 2 ** B[8])}*exp(-.5*e);c=mix(c,vec3f(${fld(5)},${fld(6)},${fld(7)})*${fnum(1 / MAX[5])},w);}` +
    `return vec4f(c,1.);}`;
}

export async function gzipSize(text) {
  const s = new Blob([text]).stream().pipeThrough(new CompressionStream("gzip"));
  return (await new Response(s).arrayBuffer()).byteLength;
}
